#define PETSCMAT_DLL

#include "../src/mat/impls/adj/mpi/mpiadj.h"       /*I "petscmat.h" I*/

#ifdef PETSC_HAVE_UNISTD_H
#include <unistd.h>
#endif

#ifdef PETSC_HAVE_STDLIB_H
#include <stdlib.h>
#endif

EXTERN_C_BEGIN
/* Chaco does not have an include file */
extern int interface(int nvtxs, int *start, int *adjacency, int *vwgts,
    float *ewgts, float *x, float *y, float *z, char *outassignname,
    char *outfilename, short *assignment, int architecture, int ndims_tot,
    int mesh_dims[3], double *goal, int global_method, int local_method,
    int rqi_flag, int vmax, int ndims, double eigtol, long seed);

extern int FREE_GRAPH;        

/*
int       nvtxs;		number of vertices in full graph 
int      *start;		start of edge list for each vertex 
int      *adjacency;	        edge list data 
int      *vwgts;	        weights for all vertices 
float    *ewgts;	        weights for all edges 
float    *x, *y, *z;	        coordinates for inertial method 
char     *outassignname;        name of assignment output file 
char     *outfilename;          output file name 
short    *assignment;	        set number of each vtx (length n) 
int       architecture;         0 => hypercube, d => d-dimensional mesh 
int       ndims_tot;	        total number of cube dimensions to divide 
int       mesh_dims[3];         dimensions of mesh of processors 
double   *goal;	                desired set sizes for each set 
int       global_method;        global partitioning algorithm 
int       local_method;         local partitioning algorithm 
int       rqi_flag;	        should I use RQI/Symmlq eigensolver? 
int       vmax;	                how many vertices to coarsen down to? 
int       ndims;	        number of eigenvectors (2^d sets) 
double    eigtol;	        tolerance on eigenvectors 
long      seed;	                for random graph mutations 
*/

EXTERN_C_END 

typedef struct {
    int architecture;
    int ndims_tot;
    int mesh_dims[3];
    int rqi_flag;
    int numbereigen;
    double eigtol;
    int global_method;          /* global method */
    int local_method;           /* local method */
    int nbvtxcoarsed;           /* number of vertices for the coarse graph */
    char *mesg_log;
} MatPartitioning_Chaco;

#define SIZE_LOG 10000          /* size of buffer for msg_log */

#undef __FUNCT__
#define __FUNCT__ "MatPartitioningApply_Chaco"
static PetscErrorCode MatPartitioningApply_Chaco(MatPartitioning part, IS *partitioning)
{
    PetscErrorCode ierr;
    int  *parttab, *locals, i, size, rank;
    Mat mat = part->adj, matMPI, matSeq;
    int nb_locals;              
    Mat_MPIAdj *adj;
    MatPartitioning_Chaco *chaco = (MatPartitioning_Chaco *) part->data;
    PetscTruth flg;
#ifdef PETSC_HAVE_UNISTD_H
    int fd_stdout, fd_pipe[2], count,err;
#endif

    PetscFunctionBegin;

    FREE_GRAPH = 0; /* otherwise Chaco will attempt to free memory for adjacency graph */
    
    ierr = MPI_Comm_size(((PetscObject)mat)->comm, &size);CHKERRQ(ierr);

    ierr = PetscTypeCompare((PetscObject) mat, MATMPIADJ, &flg);CHKERRQ(ierr);

    /* check if the matrix is sequential, use MatGetSubMatrices if necessary */
    if (size > 1) {
        int M, N;
        IS isrow, iscol;
        Mat *A;

        if (flg) {
            SETERRQ(0, "Distributed matrix format MPIAdj is not supported for sequential partitioners");
        }
        PetscPrintf(((PetscObject)part)->comm, "Converting distributed matrix to sequential: this could be a performance loss\n");CHKERRQ(ierr);

        ierr = MatGetSize(mat, &M, &N);CHKERRQ(ierr);
        ierr = ISCreateStride(PETSC_COMM_SELF, M, 0, 1, &isrow);CHKERRQ(ierr);
        ierr = ISCreateStride(PETSC_COMM_SELF, N, 0, 1, &iscol);CHKERRQ(ierr);
        ierr = MatGetSubMatrices(mat, 1, &isrow, &iscol, MAT_INITIAL_MATRIX, &A);CHKERRQ(ierr);
        ierr = ISDestroy(isrow);CHKERRQ(ierr);
        ierr = ISDestroy(iscol);CHKERRQ(ierr);
        matSeq = *A;
        ierr   = PetscFree(A);CHKERRQ(ierr);
    } else
        matSeq = mat;

    /* check for the input format that is supported only for a MPIADJ type 
       and set it to matMPI */
    if (!flg) {
        ierr = MatConvert(matSeq, MATMPIADJ, MAT_INITIAL_MATRIX, &matMPI);CHKERRQ(ierr);
    } else {
        matMPI = matSeq;
    }
    adj = (Mat_MPIAdj *) matMPI->data;  /* finaly adj contains adjacency graph */

    {
        /* arguments for Chaco library */
        int nvtxs = mat->rmap->N;                /* number of vertices in full graph */
        int *start = adj->i;                    /* start of edge list for each vertex */
        int *adjacency;                         /* = adj -> j; edge list data  */
        int *vwgts = NULL;                      /* weights for all vertices */
        float *ewgts = NULL;                    /* weights for all edges */
        float *x = NULL, *y = NULL, *z = NULL;  /* coordinates for inertial method */
        char *outassignname = NULL;             /*  name of assignment output file */
        char *outfilename = NULL;               /* output file name */
        short *assignment;                      /* set number of each vtx (length n) */
        int architecture = chaco->architecture; /* 0 => hypercube, d => d-dimensional mesh */
        int ndims_tot = chaco->ndims_tot;       /* total number of cube dimensions to divide */
        int *mesh_dims = chaco->mesh_dims;      /* dimensions of mesh of processors */
        double *goal = NULL;                    /* desired set sizes for each set */
        int global_method = chaco->global_method; /* global partitioning algorithm */
        int local_method = chaco->local_method; /* local partitioning algorithm */
        int rqi_flag = chaco->rqi_flag;         /* should I use RQI/Symmlq eigensolver? */
        int vmax = chaco->nbvtxcoarsed;         /* how many vertices to coarsen down to? */
        int ndims = chaco->numbereigen;         /* number of eigenvectors (2^d sets) */
        double eigtol = chaco->eigtol;          /* tolerance on eigenvectors */
        long seed = 123636512;                  /* for random graph mutations */

        /* return value of Chaco */
        ierr = PetscMalloc((mat->rmap->N) * sizeof(short), &assignment);CHKERRQ(ierr);          
        
        /* index change for libraries that have fortran implementation */
        ierr = PetscMalloc(sizeof(int) * start[nvtxs], &adjacency);CHKERRQ(ierr);
        for (i = 0; i < start[nvtxs]; i++)
            adjacency[i] = (adj->j)[i] + 1;

        /* redirect output to buffer: chaco -> mesg_log */
#ifdef PETSC_HAVE_UNISTD_H
        fd_stdout = dup(1);
        pipe(fd_pipe);
        close(1);
        dup2(fd_pipe[1], 1);
        ierr = PetscMalloc(SIZE_LOG * sizeof(char), &(chaco->mesg_log));CHKERRQ(ierr);
#endif

        /* library call */
        ierr = interface(nvtxs, start, adjacency, vwgts, ewgts, x, y, z,
            outassignname, outfilename, assignment, architecture, ndims_tot,
            mesh_dims, goal, global_method, local_method, rqi_flag, vmax, ndims,
            eigtol, seed);

#ifdef PETSC_HAVE_UNISTD_H
        err = fflush(stdout);
        if (err) SETERRQ(PETSC_ERR_SYS,"fflush() failed on stdout");    
        count =  read(fd_pipe[0], chaco->mesg_log, (SIZE_LOG - 1) * sizeof(char));
        if (count < 0)
            count = 0;
        chaco->mesg_log[count] = 0;
        close(1);
        dup2(fd_stdout, 1);
        close(fd_stdout);
        close(fd_pipe[0]);
        close(fd_pipe[1]);
#endif

        if (ierr) { SETERRQ(PETSC_ERR_LIB, chaco->mesg_log); }

        ierr = PetscFree(adjacency);CHKERRQ(ierr);

        ierr = PetscMalloc((mat->rmap->N) * sizeof(int), &parttab);CHKERRQ(ierr);          
        for (i = 0; i < nvtxs; i++) {
            parttab[i] = assignment[i];
        }
        ierr = PetscFree(assignment);CHKERRQ(ierr);
    }

    /* Creation of the index set */
    ierr = MPI_Comm_rank(((PetscObject)part)->comm, &rank);CHKERRQ(ierr);
    ierr = MPI_Comm_size(((PetscObject)part)->comm, &size);CHKERRQ(ierr);
    nb_locals = mat->rmap->N / size;
    locals = parttab + rank * nb_locals;
    if (rank < mat->rmap->N % size) {
        nb_locals++;
        locals += rank;
    } else
        locals += mat->rmap->N % size;
    ierr = ISCreateGeneral(((PetscObject)part)->comm, nb_locals, locals, partitioning);CHKERRQ(ierr);

    /* destroy temporary objects */
    ierr = PetscFree(parttab);CHKERRQ(ierr);
    if (matSeq != mat) {
        ierr = MatDestroy(matSeq);CHKERRQ(ierr); 
    }
    if (matMPI != mat) {
        ierr = MatDestroy(matMPI);CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatPartitioningView_Chaco"
PetscErrorCode MatPartitioningView_Chaco(MatPartitioning part, PetscViewer viewer)
{

    MatPartitioning_Chaco *chaco = (MatPartitioning_Chaco *) part->data;
    PetscErrorCode        ierr;
    PetscMPIInt           rank;
    PetscTruth            iascii;

    PetscFunctionBegin;
    ierr = MPI_Comm_rank(((PetscObject)part)->comm, &rank);CHKERRQ(ierr);
    ierr = PetscTypeCompare((PetscObject) viewer, PETSC_VIEWER_ASCII, &iascii);CHKERRQ(ierr);
    if (iascii) {
      if (!rank && chaco->mesg_log) {
        ierr = PetscViewerASCIIPrintf(viewer, "%s\n", chaco->mesg_log);CHKERRQ(ierr);
      }
    } else {
      SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported for this Chaco partitioner",((PetscObject) viewer)->type_name);
    }
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPartitioningChacoSetGlobal"
/*@
     MatPartitioningChacoSetGlobal - Set method for global partitioning.

  Input Parameter:
.  part - the partitioning context
.  method - MP_CHACO_MULTILEVEL_KL, MP_CHACO_SPECTRAL, MP_CHACO_LINEAR, 
    MP_CHACO_RANDOM or MP_CHACO_SCATTERED

   Level: advanced

@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatPartitioningChacoSetGlobal(MatPartitioning part, MPChacoGlobalType method)
{
    MatPartitioning_Chaco *chaco = (MatPartitioning_Chaco *) part->data;

    PetscFunctionBegin;

    switch (method) {
    case MP_CHACO_MULTILEVEL_KL:
        chaco->global_method = 1;
        break;
    case MP_CHACO_SPECTRAL:
        chaco->global_method = 2;
        break;
    case MP_CHACO_LINEAR:
        chaco->global_method = 4;
        break;
    case MP_CHACO_RANDOM:
        chaco->global_method = 5;
        break;
    case MP_CHACO_SCATTERED:
        chaco->global_method = 6;
        break;
    default:
        SETERRQ(PETSC_ERR_SUP, "Chaco: Unknown or unsupported option");
    }
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPartitioningChacoSetLocal"
/*@
     MatPartitioningChacoSetLocal - Set method for local partitioning.

  Input Parameter:
.  part - the partitioning context
.  method - MP_CHACO_KERNIGHAN_LIN or MP_CHACO_NONE

   Level: advanced

@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatPartitioningChacoSetLocal(MatPartitioning part, MPChacoLocalType method)
{
    MatPartitioning_Chaco *chaco = (MatPartitioning_Chaco *) part->data;

    PetscFunctionBegin;

    switch (method) {
    case MP_CHACO_KERNIGHAN_LIN:
        chaco->local_method = 1;
        break;
    case MP_CHACO_NONE:
        chaco->local_method = 2;
        break;
    default:
        SETERRQ(PETSC_ERR_ARG_CORRUPT, "Chaco: Unknown or unsupported option");
    }

    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPartitioningChacoSetCoarseLevel"
/*@
    MatPartitioningChacoSetCoarseLevel - Set the coarse level 
    
  Input Parameter:
.  part - the partitioning context
.  level - the coarse level in range [0.0,1.0]

   Level: advanced

@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatPartitioningChacoSetCoarseLevel(MatPartitioning part, PetscReal level)
{
    MatPartitioning_Chaco *chaco = (MatPartitioning_Chaco *) part->data;

    PetscFunctionBegin;

    if (level < 0 || level > 1.0) {
        SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,
            "Chaco: level of coarsening out of range [0.01-1.0]");
    } else
        chaco->nbvtxcoarsed = (int)(part->adj->cmap->N * level);

    if (chaco->nbvtxcoarsed < 20)
        chaco->nbvtxcoarsed = 20;

    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPartitioningChacoSetEigenSolver"
/*@
     MatPartitioningChacoSetEigenSolver - Set method for eigensolver.

  Input Parameter:
.  method - MP_CHACO_LANCZOS or MP_CHACO_RQI_SYMMLQ

   Level: advanced

@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatPartitioningChacoSetEigenSolver(MatPartitioning part,
    MPChacoEigenType method)
{
    MatPartitioning_Chaco *chaco = (MatPartitioning_Chaco *) part->data;

    PetscFunctionBegin;

    switch (method) {
    case MP_CHACO_LANCZOS:
        chaco->rqi_flag = 0;
        break;
    case MP_CHACO_RQI_SYMMLQ:
        chaco->rqi_flag = 1;
        break;
    default:
        SETERRQ(PETSC_ERR_ARG_CORRUPT, "Chaco: Unknown or unsupported option");
    }

    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPartitioningChacoSetEigenTol"
/*@
     MatPartitioningChacoSetEigenTol - Set tolerance for eigensolver.

  Input Parameter:
.  tol - Tolerance requested.

   Level: advanced

@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatPartitioningChacoSetEigenTol(MatPartitioning part, PetscReal tol)
{
    MatPartitioning_Chaco *chaco = (MatPartitioning_Chaco *) part->data;

    PetscFunctionBegin;

    if (tol <= 0.0) {
        SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,
            "Chaco: Eigensolver tolerance out of range");
    } else
        chaco->eigtol = tol;

    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPartitioningChacoSetEigenNumber"
/*@
     MatPartitioningChacoSetEigenNumber - Set number of eigenvectors for partitioning.

  Input Parameter:
.  num - This argument should have a value of 1, 2 or 3 indicating  
    partitioning by bisection, quadrisection, or octosection.

   Level: advanced

@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatPartitioningChacoSetEigenNumber(MatPartitioning part, int num)
{
    MatPartitioning_Chaco *chaco = (MatPartitioning_Chaco *) part->data;

    PetscFunctionBegin;

    if (num > 3 || num < 1) {
        SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,
            "Chaco: number of eigenvectors out of range");
    } else
        chaco->numbereigen = num;

    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPartitioningSetFromOptions_Chaco"
PetscErrorCode MatPartitioningSetFromOptions_Chaco(MatPartitioning part)
{
    PetscErrorCode ierr;
    int  i;
    PetscReal r;
    PetscTruth flag;
    const char *global[] =
        { "multilevel-kl", "spectral", "linear", "random", "scattered" };
    const char *local[] = { "kernighan-lin", "none" };
    const char *eigen[] = { "lanczos", "rqi_symmlq" };

    PetscFunctionBegin;
    ierr = PetscOptionsHead("Set Chaco partitioning options");CHKERRQ(ierr);

    ierr = PetscOptionsEList("-mat_partitioning_chaco_global",
        "Global method to use", "MatPartitioningChacoSetGlobal", global, 5,
        global[0], &i, &flag);CHKERRQ(ierr);
    if (flag)
        ierr = MatPartitioningChacoSetGlobal(part, (MPChacoGlobalType)i);CHKERRQ(ierr);

    ierr = PetscOptionsEList("-mat_partitioning_chaco_local",
        "Local method to use", "MatPartitioningChacoSetLocal", local, 2,
        local[0], &i, &flag);CHKERRQ(ierr);
    if (flag)
        ierr = MatPartitioningChacoSetLocal(part, (MPChacoLocalType)i);CHKERRQ(ierr);

    ierr = PetscOptionsReal("-mat_partitioning_chaco_coarse_level",
        "Coarse level", "MatPartitioningChacoSetCoarseLevel", 0, &r,
        &flag);CHKERRQ(ierr);
    if (flag)
        ierr = MatPartitioningChacoSetCoarseLevel(part, r);CHKERRQ(ierr);

    ierr = PetscOptionsEList("-mat_partitioning_chaco_eigen_solver",
        "Eigensolver to use in spectral method", "MatPartitioningChacoSetEigenSolver",
        eigen, 2, eigen[0], &i, &flag);CHKERRQ(ierr);
    if (flag)
        ierr = MatPartitioningChacoSetEigenSolver(part, (MPChacoEigenType)i);CHKERRQ(ierr);

    ierr = PetscOptionsReal("-mat_partitioning_chaco_eigen_tol",
        "Tolerance for eigensolver", "MatPartitioningChacoSetEigenTol", 0.001, 
	&r, &flag);CHKERRQ(ierr);
    if (flag)
        ierr = MatPartitioningChacoSetEigenTol(part, r);CHKERRQ(ierr);

    ierr = PetscOptionsInt("-mat_partitioning_chaco_eigen_number",
        "Number of eigenvectors: 1, 2, or 3 (bi-, quadri-, or octosection)",
        "MatPartitioningChacoSetEigenNumber", 1, &i, &flag);CHKERRQ(ierr);
    if (flag)
        ierr = MatPartitioningChacoSetEigenNumber(part, i);CHKERRQ(ierr);

    ierr = PetscOptionsTail();CHKERRQ(ierr);
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatPartitioningDestroy_Chaco"
PetscErrorCode MatPartitioningDestroy_Chaco(MatPartitioning part)
{
    MatPartitioning_Chaco *chaco = (MatPartitioning_Chaco *) part->data;
    PetscErrorCode        ierr;

    PetscFunctionBegin;
    ierr = PetscFree(chaco->mesg_log);CHKERRQ(ierr);
    ierr = PetscFree(chaco);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

/*MC
   MAT_PARTITIONING_CHACO - Creates a partitioning context via the external package Chaco.

   Collective on MPI_Comm

   Input Parameter:
.  part - the partitioning context

   Options Database Keys:
+  -mat_partitioning_chaco_global <multilevel-kl> (one of) multilevel-kl spectral linear random scattered
.  -mat_partitioning_chaco_local <kernighan-lin> (one of) kernighan-lin none
.  -mat_partitioning_chaco_coarse_level <0>: Coarse level (MatPartitioningChacoSetCoarseLevel)
.  -mat_partitioning_chaco_eigen_solver <lanczos> (one of) lanczos rqi_symmlq
.  -mat_partitioning_chaco_eigen_tol <0.001>: Tolerance for eigensolver (MatPartitioningChacoSetEigenTol)
-  -mat_partitioning_chaco_eigen_number <1>: Number of eigenvectors: 1, 2, or 3 (bi-, quadri-, or octosection) (MatPartitioningChacoSetEigenNumber)

   Level: beginner

   Notes: See http://www.cs.sandia.gov/CRF/chac.html

.keywords: Partitioning, create, context

.seealso: MatPartitioningSetType(), MatPartitioningType

M*/
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatPartitioningCreate_Chaco"
PetscErrorCode PETSCMAT_DLLEXPORT MatPartitioningCreate_Chaco(MatPartitioning part)
{
    PetscErrorCode ierr;
    MatPartitioning_Chaco *chaco;

    PetscFunctionBegin;
    ierr = PetscNewLog(part,MatPartitioning_Chaco, &chaco);CHKERRQ(ierr);
    part->data = (void*) chaco;

    chaco->architecture = 1;
    chaco->ndims_tot = 0;
    chaco->mesh_dims[0] = part->n;
    chaco->global_method = 1;
    chaco->local_method = 1;
    chaco->rqi_flag = 0;
    chaco->nbvtxcoarsed = 200;
    chaco->numbereigen = 1;
    chaco->eigtol = 0.001;
    chaco->mesg_log = NULL;

    part->ops->apply = MatPartitioningApply_Chaco;
    part->ops->view = MatPartitioningView_Chaco;
    part->ops->destroy = MatPartitioningDestroy_Chaco;
    part->ops->setfromoptions = MatPartitioningSetFromOptions_Chaco;

    PetscFunctionReturn(0);
}

EXTERN_C_END
