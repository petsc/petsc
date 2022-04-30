
#include <../src/mat/impls/adj/mpi/mpiadj.h>       /*I "petscmat.h" I*/

#if defined(PETSC_HAVE_UNISTD_H)
#include <unistd.h>
#endif

#if defined(PETSC_HAVE_CHACO_INT_ASSIGNMENT)
#include <chaco.h>
#else
/* Older versions of Chaco do not have an include file */
PETSC_EXTERN int interface(int nvtxs, int *start, int *adjacency, int *vwgts,
                     float *ewgts, float *x, float *y, float *z, char *outassignname,
                     char *outfilename, short *assignment, int architecture, int ndims_tot,
                     int mesh_dims[3], double *goal, int global_method, int local_method,
                     int rqi_flag, int vmax, int ndims, double eigtol, long seed);
#endif

extern int FREE_GRAPH;

/*
int       nvtxs;                number of vertices in full graph
int      *start;                start of edge list for each vertex
int      *adjacency;            edge list data
int      *vwgts;                weights for all vertices
float    *ewgts;                weights for all edges
float    *x, *y, *z;            coordinates for inertial method
char     *outassignname;        name of assignment output file
char     *outfilename;          output file name
short    *assignment;           set number of each vtx (length n)
int       architecture;         0 => hypercube, d => d-dimensional mesh
int       ndims_tot;            total number of cube dimensions to divide
int       mesh_dims[3];         dimensions of mesh of processors
double   *goal;                 desired set sizes for each set
int       global_method;        global partitioning algorithm
int       local_method;         local partitioning algorithm
int       rqi_flag;             should I use RQI/Symmlq eigensolver?
int       vmax;                 how many vertices to coarsen down to?
int       ndims;                number of eigenvectors (2^d sets)
double    eigtol;               tolerance on eigenvectors
long      seed;                 for random graph mutations
*/

typedef struct {
  PetscBool         verbose;
  PetscInt          eignum;
  PetscReal         eigtol;
  MPChacoGlobalType global_method;          /* global method */
  MPChacoLocalType  local_method;           /* local method */
  MPChacoEigenType  eigen_method;           /* eigensolver */
  PetscInt          nbvtxcoarsed;           /* number of vertices for the coarse graph */
} MatPartitioning_Chaco;

#define SIZE_LOG 10000          /* size of buffer for mesg_log */

static PetscErrorCode MatPartitioningApply_Chaco(MatPartitioning part,IS *partitioning)
{
  int                   cerr;
  PetscInt              *parttab,*locals,i,nb_locals,M,N;
  PetscMPIInt           size,rank;
  Mat                   mat = part->adj,matAdj,matSeq,*A;
  Mat_MPIAdj            *adj;
  MatPartitioning_Chaco *chaco = (MatPartitioning_Chaco*)part->data;
  PetscBool             flg;
  IS                    isrow, iscol;
  int                   nvtxs,*start,*adjacency,*vwgts,architecture,ndims_tot;
  int                   mesh_dims[3],global_method,local_method,rqi_flag,vmax,ndims;
#if defined(PETSC_HAVE_CHACO_INT_ASSIGNMENT)
  int                   *assignment;
#else
  short                 *assignment;
#endif
  double                eigtol;
  long                  seed;
  char                  *mesg_log;
#if defined(PETSC_HAVE_UNISTD_H)
  int                   fd_stdout,fd_pipe[2],count,err;
#endif

  PetscFunctionBegin;
  PetscCheck(!part->use_edge_weights,PetscObjectComm((PetscObject)part),PETSC_ERR_SUP,"Chaco does not support edge weights");
  FREE_GRAPH = 0; /* otherwise Chaco will attempt to free memory for adjacency graph */
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)mat),&size));
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)mat),&rank));
  PetscCall(PetscObjectTypeCompare((PetscObject)mat,MATMPIADJ,&flg));
  if (size>1) {
    if (flg) {
      PetscCall(MatMPIAdjToSeq(mat,&matSeq));
    } else {
      PetscCall(PetscInfo(part,"Converting distributed matrix to sequential: this could be a performance loss\n"));
      PetscCall(MatGetSize(mat,&M,&N));
      PetscCall(ISCreateStride(PETSC_COMM_SELF,M,0,1,&isrow));
      PetscCall(ISCreateStride(PETSC_COMM_SELF,N,0,1,&iscol));
      PetscCall(MatCreateSubMatrices(mat,1,&isrow,&iscol,MAT_INITIAL_MATRIX,&A));
      PetscCall(ISDestroy(&isrow));
      PetscCall(ISDestroy(&iscol));
      matSeq = *A;
      PetscCall(PetscFree(A));
    }
  } else {
    PetscCall(PetscObjectReference((PetscObject)mat));
    matSeq = mat;
  }

  if (!flg) { /* convert regular matrix to MPIADJ */
    PetscCall(MatConvert(matSeq,MATMPIADJ,MAT_INITIAL_MATRIX,&matAdj));
  } else {
    PetscCall(PetscObjectReference((PetscObject)matSeq));
    matAdj = matSeq;
  }

  adj = (Mat_MPIAdj*)matAdj->data;  /* finaly adj contains adjacency graph */

  /* arguments for Chaco library */
  nvtxs         = mat->rmap->N;           /* number of vertices in full graph */
  start         = adj->i;                 /* start of edge list for each vertex */
  vwgts         = part->vertex_weights;   /* weights for all vertices */
  architecture  = 1;                      /* 0 => hypercube, d => d-dimensional mesh */
  ndims_tot     = 0;                      /* total number of cube dimensions to divide */
  mesh_dims[0]  = part->n;                /* dimensions of mesh of processors */
  global_method = chaco->global_method;   /* global partitioning algorithm */
  local_method  = chaco->local_method;    /* local partitioning algorithm */
  rqi_flag      = chaco->eigen_method;    /* should I use RQI/Symmlq eigensolver? */
  vmax          = chaco->nbvtxcoarsed;    /* how many vertices to coarsen down to? */
  ndims         = chaco->eignum;          /* number of eigenvectors (2^d sets) */
  eigtol        = chaco->eigtol;          /* tolerance on eigenvectors */
  seed          = 123636512;              /* for random graph mutations */

  PetscCall(PetscMalloc1(mat->rmap->N,&assignment));
  PetscCall(PetscMalloc1(start[nvtxs],&adjacency));
  for (i=0; i<start[nvtxs]; i++) adjacency[i] = (adj->j)[i] + 1; /* 1-based indexing */

  /* redirect output to buffer */
#if defined(PETSC_HAVE_UNISTD_H)
  fd_stdout = dup(1);
  PetscCheck(!pipe(fd_pipe),PETSC_COMM_SELF,PETSC_ERR_SYS,"Could not open pipe");
  close(1);
  dup2(fd_pipe[1],1);
  PetscCall(PetscMalloc1(SIZE_LOG,&mesg_log));
#endif

  /* library call */
  cerr = interface(nvtxs,start,adjacency,vwgts,NULL,NULL,NULL,NULL,
                  NULL,NULL,assignment,architecture,ndims_tot,mesh_dims,
                  NULL,global_method,local_method,rqi_flag,vmax,ndims,eigtol,seed);

#if defined(PETSC_HAVE_UNISTD_H)
  err = fflush(stdout);
  PetscCheck(!err,PETSC_COMM_SELF,PETSC_ERR_SYS,"fflush() failed on stdout");
  count = read(fd_pipe[0],mesg_log,(SIZE_LOG-1)*sizeof(char));
  if (count<0) count = 0;
  mesg_log[count] = 0;
  close(1);
  dup2(fd_stdout,1);
  close(fd_stdout);
  close(fd_pipe[0]);
  close(fd_pipe[1]);
  if (chaco->verbose) {
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)mat),"%s",mesg_log));
  }
  PetscCall(PetscFree(mesg_log));
#endif
  PetscCheck(!cerr,PETSC_COMM_SELF,PETSC_ERR_LIB,"Chaco failed");

  PetscCall(PetscMalloc1(mat->rmap->N,&parttab));
  for (i=0; i<nvtxs; i++) parttab[i] = assignment[i];

  /* creation of the index set */
  nb_locals = mat->rmap->n;
  locals    = parttab + mat->rmap->rstart;
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)part),nb_locals,locals,PETSC_COPY_VALUES,partitioning));

  /* clean up */
  PetscCall(PetscFree(parttab));
  PetscCall(PetscFree(adjacency));
  PetscCall(PetscFree(assignment));
  PetscCall(MatDestroy(&matSeq));
  PetscCall(MatDestroy(&matAdj));
  PetscFunctionReturn(0);
}

PetscErrorCode MatPartitioningView_Chaco(MatPartitioning part, PetscViewer viewer)
{
  MatPartitioning_Chaco *chaco = (MatPartitioning_Chaco*)part->data;
  PetscBool             isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii));
  if (isascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Global method: %s\n",MPChacoGlobalTypes[chaco->global_method]));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Local method: %s\n",MPChacoLocalTypes[chaco->local_method]));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Number of vertices for the coarse graph: %" PetscInt_FMT "\n",chaco->nbvtxcoarsed));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Eigensolver: %s\n",MPChacoEigenTypes[chaco->eigen_method]));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Tolerance for eigensolver: %g\n",chaco->eigtol));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Number of eigenvectors: %" PetscInt_FMT "\n",chaco->eignum));
  }
  PetscFunctionReturn(0);
}

/*@
   MatPartitioningChacoSetGlobal - Set global method for Chaco partitioner.

   Collective on MatPartitioning

   Input Parameters:
+  part - the partitioning context
-  method - one of MP_CHACO_MULTILEVEL, MP_CHACO_SPECTRAL, MP_CHACO_LINEAR,
            MP_CHACO_RANDOM or MP_CHACO_SCATTERED

   Options Database:
.  -mat_partitioning_chaco_global <method> - the global method

   Level: advanced

   Notes:
   The default is the multi-level method. See Chaco documentation for
   additional details.

.seealso: MatPartitioningChacoSetLocal(),MatPartitioningChacoGetGlobal()
@*/
PetscErrorCode MatPartitioningChacoSetGlobal(MatPartitioning part,MPChacoGlobalType method)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(part,MAT_PARTITIONING_CLASSID,1);
  PetscValidLogicalCollectiveEnum(part,method,2);
  PetscTryMethod(part,"MatPartitioningChacoSetGlobal_C",(MatPartitioning,MPChacoGlobalType),(part,method));
  PetscFunctionReturn(0);
}

PetscErrorCode MatPartitioningChacoSetGlobal_Chaco(MatPartitioning part,MPChacoGlobalType method)
{
  MatPartitioning_Chaco *chaco = (MatPartitioning_Chaco*)part->data;

  PetscFunctionBegin;
  switch (method) {
  case MP_CHACO_MULTILEVEL:
  case MP_CHACO_SPECTRAL:
  case MP_CHACO_LINEAR:
  case MP_CHACO_RANDOM:
  case MP_CHACO_SCATTERED:
    chaco->global_method = method; break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Chaco: Unknown or unsupported option");
  }
  PetscFunctionReturn(0);
}

/*@
   MatPartitioningChacoGetGlobal - Get global method for Chaco partitioner.

   Not Collective

   Input Parameter:
.  part - the partitioning context

   Output Parameter:
.  method - the method

   Level: advanced

.seealso: MatPartitioningChacoSetGlobal()
@*/
PetscErrorCode MatPartitioningChacoGetGlobal(MatPartitioning part,MPChacoGlobalType *method)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(part,MAT_PARTITIONING_CLASSID,1);
  PetscValidPointer(method,2);
  PetscTryMethod(part,"MatPartitioningChacoGetGlobal_C",(MatPartitioning,MPChacoGlobalType*),(part,method));
  PetscFunctionReturn(0);
}

PetscErrorCode MatPartitioningChacoGetGlobal_Chaco(MatPartitioning part,MPChacoGlobalType *method)
{
  MatPartitioning_Chaco *chaco = (MatPartitioning_Chaco*)part->data;

  PetscFunctionBegin;
  *method = chaco->global_method;
  PetscFunctionReturn(0);
}

/*@
   MatPartitioningChacoSetLocal - Set local method for Chaco partitioner.

   Collective on MatPartitioning

   Input Parameters:
+  part - the partitioning context
-  method - one of MP_CHACO_KERNIGHAN or MP_CHACO_NONE

   Options Database:
.  -mat_partitioning_chaco_local <method> - the local method

   Level: advanced

   Notes:
   The default is to apply the Kernighan-Lin heuristic. See Chaco documentation
   for additional details.

.seealso: MatPartitioningChacoSetGlobal(),MatPartitioningChacoGetLocal()
@*/
PetscErrorCode MatPartitioningChacoSetLocal(MatPartitioning part,MPChacoLocalType method)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(part,MAT_PARTITIONING_CLASSID,1);
  PetscValidLogicalCollectiveEnum(part,method,2);
  PetscTryMethod(part,"MatPartitioningChacoSetLocal_C",(MatPartitioning,MPChacoLocalType),(part,method));
  PetscFunctionReturn(0);
}

PetscErrorCode MatPartitioningChacoSetLocal_Chaco(MatPartitioning part,MPChacoLocalType method)
{
  MatPartitioning_Chaco *chaco = (MatPartitioning_Chaco*)part->data;

  PetscFunctionBegin;
  switch (method) {
  case MP_CHACO_KERNIGHAN:
  case MP_CHACO_NONE:
    chaco->local_method = method; break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Chaco: Unknown or unsupported option");
  }
  PetscFunctionReturn(0);
}

/*@
   MatPartitioningChacoGetLocal - Get local method for Chaco partitioner.

   Not Collective

   Input Parameter:
.  part - the partitioning context

   Output Parameter:
.  method - the method

   Level: advanced

.seealso: MatPartitioningChacoSetLocal()
@*/
PetscErrorCode MatPartitioningChacoGetLocal(MatPartitioning part,MPChacoLocalType *method)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(part,MAT_PARTITIONING_CLASSID,1);
  PetscValidPointer(method,2);
  PetscUseMethod(part,"MatPartitioningChacoGetLocal_C",(MatPartitioning,MPChacoLocalType*),(part,method));
  PetscFunctionReturn(0);
}

PetscErrorCode MatPartitioningChacoGetLocal_Chaco(MatPartitioning part,MPChacoLocalType *method)
{
  MatPartitioning_Chaco *chaco = (MatPartitioning_Chaco*)part->data;

  PetscFunctionBegin;
  *method = chaco->local_method;
  PetscFunctionReturn(0);
}

/*@
   MatPartitioningChacoSetCoarseLevel - Set the coarse level parameter for the
   Chaco partitioner.

   Collective on MatPartitioning

   Input Parameters:
+  part - the partitioning context
-  level - the coarse level in range [0.0,1.0]

   Options Database:
.  -mat_partitioning_chaco_coarse <l> - Coarse level

   Level: advanced
@*/
PetscErrorCode MatPartitioningChacoSetCoarseLevel(MatPartitioning part,PetscReal level)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(part,MAT_PARTITIONING_CLASSID,1);
  PetscValidLogicalCollectiveReal(part,level,2);
  PetscTryMethod(part,"MatPartitioningChacoSetCoarseLevel_C",(MatPartitioning,PetscReal),(part,level));
  PetscFunctionReturn(0);
}

PetscErrorCode MatPartitioningChacoSetCoarseLevel_Chaco(MatPartitioning part,PetscReal level)
{
  MatPartitioning_Chaco *chaco = (MatPartitioning_Chaco*)part->data;

  PetscFunctionBegin;
  PetscCheck(level >= 0.0 && level < 1.0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Chaco: level of coarsening out of range [0.0-1.0]");
  chaco->nbvtxcoarsed = (PetscInt)(part->adj->cmap->N * level);
  if (chaco->nbvtxcoarsed < 20) chaco->nbvtxcoarsed = 20;
  PetscFunctionReturn(0);
}

/*@
   MatPartitioningChacoSetEigenSolver - Set eigensolver method for Chaco partitioner.

   Collective on MatPartitioning

   Input Parameters:
+  part - the partitioning context
-  method - one of MP_CHACO_LANCZOS or MP_CHACO_RQI

   Options Database:
.  -mat_partitioning_chaco_eigen_solver <method> - the eigensolver

   Level: advanced

   Notes:
   The default is to use a Lanczos method. See Chaco documentation for details.

.seealso: MatPartitioningChacoSetEigenTol(),MatPartitioningChacoSetEigenNumber(),
          MatPartitioningChacoGetEigenSolver()
@*/
PetscErrorCode MatPartitioningChacoSetEigenSolver(MatPartitioning part,MPChacoEigenType method)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(part,MAT_PARTITIONING_CLASSID,1);
  PetscValidLogicalCollectiveEnum(part,method,2);
  PetscTryMethod(part,"MatPartitioningChacoSetEigenSolver_C",(MatPartitioning,MPChacoEigenType),(part,method));
  PetscFunctionReturn(0);
}

PetscErrorCode MatPartitioningChacoSetEigenSolver_Chaco(MatPartitioning part,MPChacoEigenType method)
{
  MatPartitioning_Chaco *chaco = (MatPartitioning_Chaco*)part->data;

  PetscFunctionBegin;
  switch (method) {
  case MP_CHACO_LANCZOS:
  case MP_CHACO_RQI:
    chaco->eigen_method = method; break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Chaco: Unknown or unsupported option");
  }
  PetscFunctionReturn(0);
}

/*@
   MatPartitioningChacoGetEigenSolver - Get local method for Chaco partitioner.

   Not Collective

   Input Parameter:
.  part - the partitioning context

   Output Parameter:
.  method - the method

   Level: advanced

.seealso: MatPartitioningChacoSetEigenSolver()
@*/
PetscErrorCode MatPartitioningChacoGetEigenSolver(MatPartitioning part,MPChacoEigenType *method)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(part,MAT_PARTITIONING_CLASSID,1);
  PetscValidPointer(method,2);
  PetscUseMethod(part,"MatPartitioningChacoGetEigenSolver_C",(MatPartitioning,MPChacoEigenType*),(part,method));
  PetscFunctionReturn(0);
}

PetscErrorCode MatPartitioningChacoGetEigenSolver_Chaco(MatPartitioning part,MPChacoEigenType *method)
{
  MatPartitioning_Chaco *chaco = (MatPartitioning_Chaco*)part->data;

  PetscFunctionBegin;
  *method = chaco->eigen_method;
  PetscFunctionReturn(0);
}

/*@
   MatPartitioningChacoSetEigenTol - Sets the tolerance for the eigensolver.

   Collective on MatPartitioning

   Input Parameters:
+  part - the partitioning context
-  tol  - the tolerance

   Options Database:
.  -mat_partitioning_chaco_eigen_tol <tol> - Tolerance for eigensolver

   Note:
   Must be positive. The default value is 0.001.

   Level: advanced

.seealso: MatPartitioningChacoSetEigenSolver(), MatPartitioningChacoGetEigenTol()
@*/
PetscErrorCode MatPartitioningChacoSetEigenTol(MatPartitioning part,PetscReal tol)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(part,MAT_PARTITIONING_CLASSID,1);
  PetscValidLogicalCollectiveReal(part,tol,2);
  PetscTryMethod(part,"MatPartitioningChacoSetEigenTol_C",(MatPartitioning,PetscReal),(part,tol));
  PetscFunctionReturn(0);
}

PetscErrorCode MatPartitioningChacoSetEigenTol_Chaco(MatPartitioning part,PetscReal tol)
{
  MatPartitioning_Chaco *chaco = (MatPartitioning_Chaco*)part->data;

  PetscFunctionBegin;
  if (tol==PETSC_DEFAULT) chaco->eigtol = 0.001;
  else {
    PetscCheck(tol>0.0,PetscObjectComm((PetscObject)part),PETSC_ERR_ARG_OUTOFRANGE,"Tolerance must be positive");
    chaco->eigtol = tol;
  }
  PetscFunctionReturn(0);
}

/*@
   MatPartitioningChacoGetEigenTol - Gets the eigensolver tolerance.

   Not Collective

   Input Parameter:
.  part - the partitioning context

   Output Parameter:
.  tol  - the tolerance

   Level: advanced

.seealso: MatPartitioningChacoSetEigenTol()
@*/
PetscErrorCode MatPartitioningChacoGetEigenTol(MatPartitioning part,PetscReal *tol)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(part,MAT_PARTITIONING_CLASSID,1);
  PetscValidRealPointer(tol,2);
  PetscUseMethod(part,"MatPartitioningChacoGetEigenTol_C",(MatPartitioning,PetscReal*),(part,tol));
  PetscFunctionReturn(0);
}

PetscErrorCode MatPartitioningChacoGetEigenTol_Chaco(MatPartitioning part,PetscReal *tol)
{
  MatPartitioning_Chaco *chaco = (MatPartitioning_Chaco*)part->data;

  PetscFunctionBegin;
  *tol = chaco->eigtol;
  PetscFunctionReturn(0);
}

/*@
   MatPartitioningChacoSetEigenNumber - Sets the number of eigenvectors to compute
   during partitioning.

   Collective on MatPartitioning

   Input Parameters:
+  part - the partitioning context
-  num  - the number of eigenvectors

   Options Database:
.  -mat_partitioning_chaco_eigen_number <n> - Number of eigenvectors

   Note:
   Accepted values are 1, 2 or 3, indicating partitioning by bisection,
   quadrisection, or octosection.

   Level: advanced

.seealso: MatPartitioningChacoSetEigenSolver(), MatPartitioningChacoGetEigenTol()
@*/
PetscErrorCode MatPartitioningChacoSetEigenNumber(MatPartitioning part,PetscInt num)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(part,MAT_PARTITIONING_CLASSID,1);
  PetscValidLogicalCollectiveInt(part,num,2);
  PetscTryMethod(part,"MatPartitioningChacoSetEigenNumber_C",(MatPartitioning,PetscInt),(part,num));
  PetscFunctionReturn(0);
}

PetscErrorCode MatPartitioningChacoSetEigenNumber_Chaco(MatPartitioning part,PetscInt num)
{
  MatPartitioning_Chaco *chaco = (MatPartitioning_Chaco*)part->data;

  PetscFunctionBegin;
  if (num==PETSC_DEFAULT) chaco->eignum = 1;
  else {
    PetscCheck(num >= 1 && num <= 3,PetscObjectComm((PetscObject)part),PETSC_ERR_ARG_OUTOFRANGE,"Can only specify 1, 2 or 3 eigenvectors");
    chaco->eignum = num;
  }
  PetscFunctionReturn(0);
}

/*@
   MatPartitioningChacoGetEigenNumber - Gets the number of eigenvectors used by Chaco.

   Not Collective

   Input Parameter:
.  part - the partitioning context

   Output Parameter:
.  num  - number of eigenvectors

   Level: advanced

.seealso: MatPartitioningChacoSetEigenNumber()
@*/
PetscErrorCode MatPartitioningChacoGetEigenNumber(MatPartitioning part,PetscInt *num)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(part,MAT_PARTITIONING_CLASSID,1);
  PetscValidIntPointer(num,2);
  PetscUseMethod(part,"MatPartitioningChacoGetEigenNumber_C",(MatPartitioning,PetscInt*),(part,num));
  PetscFunctionReturn(0);
}

PetscErrorCode MatPartitioningChacoGetEigenNumber_Chaco(MatPartitioning part,PetscInt *num)
{
  MatPartitioning_Chaco *chaco = (MatPartitioning_Chaco*)part->data;

  PetscFunctionBegin;
  *num = chaco->eignum;
  PetscFunctionReturn(0);
}

PetscErrorCode MatPartitioningSetFromOptions_Chaco(PetscOptionItems *PetscOptionsObject,MatPartitioning part)
{
  PetscInt              i;
  PetscReal             r;
  PetscBool             flag;
  MatPartitioning_Chaco *chaco = (MatPartitioning_Chaco*)part->data;
  MPChacoGlobalType     global;
  MPChacoLocalType      local;
  MPChacoEigenType      eigen;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject,"Chaco partitioning options");
  PetscCall(PetscOptionsEnum("-mat_partitioning_chaco_global","Global method","MatPartitioningChacoSetGlobal",MPChacoGlobalTypes,(PetscEnum)chaco->global_method,(PetscEnum*)&global,&flag));
  if (flag) PetscCall(MatPartitioningChacoSetGlobal(part,global));
  PetscCall(PetscOptionsEnum("-mat_partitioning_chaco_local","Local method","MatPartitioningChacoSetLocal",MPChacoLocalTypes,(PetscEnum)chaco->local_method,(PetscEnum*)&local,&flag));
  if (flag) PetscCall(MatPartitioningChacoSetLocal(part,local));
  PetscCall(PetscOptionsReal("-mat_partitioning_chaco_coarse","Coarse level","MatPartitioningChacoSetCoarseLevel",0.0,&r,&flag));
  if (flag) PetscCall(MatPartitioningChacoSetCoarseLevel(part,r));
  PetscCall(PetscOptionsEnum("-mat_partitioning_chaco_eigen_solver","Eigensolver method","MatPartitioningChacoSetEigenSolver",MPChacoEigenTypes,(PetscEnum)chaco->eigen_method,(PetscEnum*)&eigen,&flag));
  if (flag) PetscCall(MatPartitioningChacoSetEigenSolver(part,eigen));
  PetscCall(PetscOptionsReal("-mat_partitioning_chaco_eigen_tol","Eigensolver tolerance","MatPartitioningChacoSetEigenTol",chaco->eigtol,&r,&flag));
  if (flag) PetscCall(MatPartitioningChacoSetEigenTol(part,r));
  PetscCall(PetscOptionsInt("-mat_partitioning_chaco_eigen_number","Number of eigenvectors: 1, 2, or 3 (bi-, quadri-, or octosection)","MatPartitioningChacoSetEigenNumber",chaco->eignum,&i,&flag));
  if (flag) PetscCall(MatPartitioningChacoSetEigenNumber(part,i));
  PetscCall(PetscOptionsBool("-mat_partitioning_chaco_verbose","Show library output","",chaco->verbose,&chaco->verbose,NULL));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(0);
}

PetscErrorCode MatPartitioningDestroy_Chaco(MatPartitioning part)
{
  MatPartitioning_Chaco *chaco = (MatPartitioning_Chaco*) part->data;

  PetscFunctionBegin;
  PetscCall(PetscFree(chaco));
  /* clear composed functions */
  PetscCall(PetscObjectComposeFunction((PetscObject)part,"MatPartitioningChacoSetGlobal_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)part,"MatPartitioningChacoGetGlobal_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)part,"MatPartitioningChacoSetLocal_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)part,"MatPartitioningChacoGetLocal_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)part,"MatPartitioningChacoSetCoarseLevel_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)part,"MatPartitioningChacoSetEigenSolver_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)part,"MatPartitioningChacoGetEigenSolver_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)part,"MatPartitioningChacoSetEigenTol_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)part,"MatPartitioningChacoGetEigenTol_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)part,"MatPartitioningChacoSetEigenNumber_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)part,"MatPartitioningChacoGetEigenNumber_C",NULL));
  PetscFunctionReturn(0);
}

/*MC
   MATPARTITIONINGCHACO - Creates a partitioning context via the external package Chaco.

   Level: beginner

   Notes:
    See http://www.cs.sandia.gov/CRF/chac.html
    Does not using MatPartitioningSetUseEdgeWeights()

.seealso: MatPartitioningSetType(), MatPartitioningType
M*/

PETSC_EXTERN PetscErrorCode MatPartitioningCreate_Chaco(MatPartitioning part)
{
  MatPartitioning_Chaco *chaco;

  PetscFunctionBegin;
  PetscCall(PetscNewLog(part,&chaco));
  part->data = (void*)chaco;

  chaco->global_method = MP_CHACO_MULTILEVEL;
  chaco->local_method  = MP_CHACO_KERNIGHAN;
  chaco->eigen_method  = MP_CHACO_LANCZOS;
  chaco->nbvtxcoarsed  = 200;
  chaco->eignum        = 1;
  chaco->eigtol        = 0.001;
  chaco->verbose       = PETSC_FALSE;

  part->ops->apply          = MatPartitioningApply_Chaco;
  part->ops->view           = MatPartitioningView_Chaco;
  part->ops->destroy        = MatPartitioningDestroy_Chaco;
  part->ops->setfromoptions = MatPartitioningSetFromOptions_Chaco;

  PetscCall(PetscObjectComposeFunction((PetscObject)part,"MatPartitioningChacoSetGlobal_C",MatPartitioningChacoSetGlobal_Chaco));
  PetscCall(PetscObjectComposeFunction((PetscObject)part,"MatPartitioningChacoGetGlobal_C",MatPartitioningChacoGetGlobal_Chaco));
  PetscCall(PetscObjectComposeFunction((PetscObject)part,"MatPartitioningChacoSetLocal_C",MatPartitioningChacoSetLocal_Chaco));
  PetscCall(PetscObjectComposeFunction((PetscObject)part,"MatPartitioningChacoGetLocal_C",MatPartitioningChacoGetLocal_Chaco));
  PetscCall(PetscObjectComposeFunction((PetscObject)part,"MatPartitioningChacoSetCoarseLevel_C",MatPartitioningChacoSetCoarseLevel_Chaco));
  PetscCall(PetscObjectComposeFunction((PetscObject)part,"MatPartitioningChacoSetEigenSolver_C",MatPartitioningChacoSetEigenSolver_Chaco));
  PetscCall(PetscObjectComposeFunction((PetscObject)part,"MatPartitioningChacoGetEigenSolver_C",MatPartitioningChacoGetEigenSolver_Chaco));
  PetscCall(PetscObjectComposeFunction((PetscObject)part,"MatPartitioningChacoSetEigenTol_C",MatPartitioningChacoSetEigenTol_Chaco));
  PetscCall(PetscObjectComposeFunction((PetscObject)part,"MatPartitioningChacoGetEigenTol_C",MatPartitioningChacoGetEigenTol_Chaco));
  PetscCall(PetscObjectComposeFunction((PetscObject)part,"MatPartitioningChacoSetEigenNumber_C",MatPartitioningChacoSetEigenNumber_Chaco));
  PetscCall(PetscObjectComposeFunction((PetscObject)part,"MatPartitioningChacoGetEigenNumber_C",MatPartitioningChacoGetEigenNumber_Chaco));
  PetscFunctionReturn(0);
}
