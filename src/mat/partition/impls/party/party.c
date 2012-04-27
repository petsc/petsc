
#include <../src/mat/impls/adj/mpi/mpiadj.h>       /*I "petscmat.h" I*/

#ifdef PETSC_HAVE_UNISTD_H
#include <unistd.h>
#endif

#ifdef PETSC_HAVE_STDLIB_H
#include <stdlib.h>
#endif

/* 
   Currently using Party-1.99
*/
EXTERN_C_BEGIN
#include <party_lib.h>
EXTERN_C_END 

typedef struct {
  PetscBool redm;
  PetscBool redo;
  PetscBool recursive;
  PetscBool verbose;
  char      global[15];         /* global method */
  char      local[15];          /* local method */
  PetscInt  nbvtxcoarsed;       /* number of vertices for the coarse graph */
} MatPartitioning_Party;

#define SIZE_LOG 10000          /* size of buffer for mesg_log */

#undef __FUNCT__
#define __FUNCT__ "MatPartitioningApply_Party"
static PetscErrorCode MatPartitioningApply_Party(MatPartitioning part,IS *partitioning)
{
  PetscErrorCode        ierr;
  PetscInt              i,*parttab,*locals,nb_locals,M,N;
  PetscMPIInt           size,rank;
  Mat                   mat = part->adj,matAdj,matSeq,*A;
  Mat_MPIAdj            *adj;
  MatPartitioning_Party *party = (MatPartitioning_Party*)part->data;
  PetscBool             flg;
  IS                    isrow, iscol;
  int                   n,*edge_p,*edge,*vertex_w,p,*part_party,cutsize,redl,rec;
  const char            *redm,*redo;
  char                  *mesg_log;
#ifdef PETSC_HAVE_UNISTD_H
  int fd_stdout,fd_pipe[2],count,err;
#endif

  PetscFunctionBegin;
  ierr = MPI_Comm_size(((PetscObject)mat)->comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(((PetscObject)mat)->comm,&rank);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)mat,MATMPIADJ,&flg);CHKERRQ(ierr);
  if (size>1) {
    if (flg) SETERRQ(((PetscObject)mat)->comm,PETSC_ERR_SUP,"Distributed matrix format MPIAdj is not supported for sequential partitioners");
    ierr = PetscInfo(part,"Converting distributed matrix to sequential: this could be a performance loss\n");CHKERRQ(ierr);
    ierr = MatGetSize(mat,&M,&N);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,M,0,1,&isrow);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,N,0,1,&iscol);CHKERRQ(ierr);
    ierr = MatGetSubMatrices(mat,1,&isrow,&iscol,MAT_INITIAL_MATRIX,&A);CHKERRQ(ierr);
    ierr = ISDestroy(&isrow);CHKERRQ(ierr);
    ierr = ISDestroy(&iscol);CHKERRQ(ierr);
    matSeq = *A;
    ierr = PetscFree(A);CHKERRQ(ierr);
  } else {
    ierr = PetscObjectReference((PetscObject)mat);CHKERRQ(ierr);
    matSeq = mat;
  }

  if (!flg) { /* convert regular matrix to MPIADJ */
    ierr = MatConvert(matSeq,MATMPIADJ,MAT_INITIAL_MATRIX,&matAdj);CHKERRQ(ierr);
  } else {
    ierr = PetscObjectReference((PetscObject)matSeq);CHKERRQ(ierr);
    matAdj = matSeq;
  }

  adj = (Mat_MPIAdj*)matAdj->data;  /* finaly adj contains adjacency graph */

  /* arguments for Party library */
  n        = mat->rmap->N;            /* number of vertices in full graph */
  edge_p   = adj->i;                  /* start of edge list for each vertex */
  edge     = adj->j;                  /* edge list data */
  vertex_w = part->vertex_weights;    /* weights for all vertices */
  p        = part->n;                 /* number of parts to create */
  redl     = party->nbvtxcoarsed;     /* how many vertices to coarsen down to? */
  rec      = party->recursive? 1: 0;  /* recursive bisection */
  redm     = party->redm? "lam": "";  /* matching method */
  redo     = party->redo? "w3": "";   /* matching optimization method */

  ierr = PetscMalloc((mat->rmap->N)*sizeof(int),&part_party);CHKERRQ(ierr);

  /* redirect output to buffer */
#ifdef PETSC_HAVE_UNISTD_H
  fd_stdout = dup(1);
  if (pipe(fd_pipe)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SYS,"Could not open pipe");
  close(1);
  dup2(fd_pipe[1],1);
  ierr = PetscMalloc(SIZE_LOG*sizeof(char),&mesg_log);CHKERRQ(ierr);
#endif

  /* library call */
  party_lib_times_start();
  ierr = party_lib(n,vertex_w,PETSC_NULL,PETSC_NULL,PETSC_NULL,edge_p,edge,
                   PETSC_NULL,p,part_party,&cutsize,redl,(char*)redm,(char*)redo,
                   party->global,party->local,rec,1);

  party_lib_times_output(1);
  part_info(n,vertex_w,edge_p,edge,PETSC_NULL,p,part_party,1);

#ifdef PETSC_HAVE_UNISTD_H
  err = fflush(stdout);
  if (err) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SYS,"fflush() failed on stdout");    
  count = read(fd_pipe[0],mesg_log,(SIZE_LOG-1)*sizeof(char));
  if (count<0) count = 0;
  mesg_log[count] = 0;
  close(1);
  dup2(fd_stdout,1);
  close(fd_stdout);
  close(fd_pipe[0]);
  close(fd_pipe[1]);
  if (party->verbose) { ierr = PetscPrintf(((PetscObject)mat)->comm,mesg_log); }
  ierr = PetscFree(mesg_log);CHKERRQ(ierr);
#endif
  if (ierr) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Party failed");

  ierr = PetscMalloc((mat->rmap->N)*sizeof(PetscInt),&parttab);CHKERRQ(ierr);          
  for (i=0;i<mat->rmap->N;i++) parttab[i] = part_party[i];

  /* creation of the index set */
  nb_locals = mat->rmap->N / size;
  locals = parttab + rank*nb_locals;
  if (rank < mat->rmap->N % size) {
    nb_locals++;
    locals += rank;
  } else locals += mat->rmap->N % size;

  ierr = ISCreateGeneral(((PetscObject)part)->comm,nb_locals,locals,PETSC_COPY_VALUES,partitioning);CHKERRQ(ierr);

  /* clean up */
  ierr = PetscFree(parttab);CHKERRQ(ierr);
  ierr = PetscFree(part_party);CHKERRQ(ierr);
  ierr = MatDestroy(&matSeq);CHKERRQ(ierr);
  ierr = MatDestroy(&matAdj);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPartitioningView_Party"
PetscErrorCode MatPartitioningView_Party(MatPartitioning part,PetscViewer viewer)
{
  MatPartitioning_Party *party = (MatPartitioning_Party*)part->data;
  PetscErrorCode        ierr;
  PetscBool             isascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  Global method: %s\n",party->global);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  Local method: %s\n",party->local);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  Number of vertices for the coarse graph: %d\n",party->nbvtxcoarsed);CHKERRQ(ierr);
    if (party->redm) {
      ierr = PetscViewerASCIIPrintf(viewer,"  Using matching method for graph reduction\n");CHKERRQ(ierr);
    }
    if (party->redo) {
      ierr = PetscViewerASCIIPrintf(viewer,"  Using matching optimization\n");CHKERRQ(ierr);
    }
    if (party->recursive) {
      ierr = PetscViewerASCIIPrintf(viewer,"  Using recursive bipartitioning\n");CHKERRQ(ierr);
    }
  } else {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Viewer type %s not supported for Party partitioner",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPartitioningPartySetGlobal"
/*@C
   MatPartitioningPartySetGlobal - Set global method for Party partitioner.

   Collective on MatPartitioning

   Input Parameters:
+  part - the partitioning context
-  method - a string representing the method

   Options Database:
.  -mat_partitioning_party_global <method> - the global method

   Level: advanced

   Notes:
   The method may be one of MP_PARTY_OPT, MP_PARTY_LIN, MP_PARTY_SCA, 
   MP_PARTY_RAN, MP_PARTY_GBF, MP_PARTY_GCF, MP_PARTY_BUB or MP_PARTY_DEF, or
   alternatively a string describing the method. Two or more methods can be 
   combined like "gbf,gcf". Check the Party Library Users Manual for details.

.seealso: MatPartitioningPartySetLocal()
@*/
PetscErrorCode MatPartitioningPartySetGlobal(MatPartitioning part,const char *global)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part,MAT_PARTITIONING_CLASSID,1);
  ierr = PetscTryMethod(part,"MatPartitioningPartySetGlobal_C",(MatPartitioning,const char*),(part,global));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatPartitioningPartySetGlobal_Party"
PetscErrorCode MatPartitioningPartySetGlobal_Party(MatPartitioning part,const char* global)
{
  MatPartitioning_Party *party = (MatPartitioning_Party*)part->data;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  ierr = PetscStrncpy(party->global,global,15);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "MatPartitioningPartySetLocal"
/*@C
   MatPartitioningPartySetLocal - Set local method for Party partitioner.

   Collective on MatPartitioning

   Input Parameters:
+  part - the partitioning context
-  method - a string representing the method

   Options Database:
.  -mat_partitioning_party_local <method> - the local method

   Level: advanced

   Notes:
   The method may be one of MP_PARTY_HELPFUL_SETS, MP_PARTY_KERNIGHAN_LIN, or
   MP_PARTY_NONE. Check the Party Library Users Manual for details.

.seealso: MatPartitioningPartySetGlobal()
@*/
PetscErrorCode MatPartitioningPartySetLocal(MatPartitioning part,const char *local)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part,MAT_PARTITIONING_CLASSID,1);
  ierr = PetscTryMethod(part,"MatPartitioningPartySetLocal_C",(MatPartitioning,const char*),(part,local));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatPartitioningPartySetLocal_Party"
PetscErrorCode MatPartitioningPartySetLocal_Party(MatPartitioning part,const char* local)

{
  MatPartitioning_Party *party = (MatPartitioning_Party*)part->data;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  ierr = PetscStrncpy(party->local,local,15);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "MatPartitioningPartySetCoarseLevel"
/*@
   MatPartitioningPartySetCoarseLevel - Set the coarse level parameter for the
   Party partitioner.
 
   Collective on MatPartitioning

   Input Parameters:
+  part - the partitioning context
-  level - the coarse level in range [0.0,1.0]

   Options Database:
.  -mat_partitioning_party_coarse <l> - Coarse level

   Level: advanced
@*/
PetscErrorCode MatPartitioningPartySetCoarseLevel(MatPartitioning part,PetscReal level)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part,MAT_PARTITIONING_CLASSID,1);
  PetscValidLogicalCollectiveReal(part,level,2);
  ierr = PetscTryMethod(part,"MatPartitioningPartySetCoarseLevel_C",(MatPartitioning,PetscReal),(part,level));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatPartitioningPartySetCoarseLevel_Party"
PetscErrorCode MatPartitioningPartySetCoarseLevel_Party(MatPartitioning part,PetscReal level)
{
  MatPartitioning_Party *party = (MatPartitioning_Party*)part->data;

  PetscFunctionBegin;
  if (level<0.0 || level>1.0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Party: level of coarsening out of range [0.0-1.0]");
  party->nbvtxcoarsed = (PetscInt)(part->adj->cmap->N * level);
  if (party->nbvtxcoarsed < 20) party->nbvtxcoarsed = 20;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "MatPartitioningPartySetMatchOptimization"
/*@
   MatPartitioningPartySetMatchOptimization - Activate matching optimization for
   graph reduction.
 
   Collective on MatPartitioning

   Input Parameters:
+  part - the partitioning context
-  opt - boolean flag

   Options Database:
.  -mat_partitioning_party_match_optimization - Matching optimization on/off

   Level: advanced
@*/
PetscErrorCode MatPartitioningPartySetMatchOptimization(MatPartitioning part,PetscBool opt)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part,MAT_PARTITIONING_CLASSID,1);
  PetscValidLogicalCollectiveBool(part,opt,2);
  ierr = PetscTryMethod(part,"MatPartitioningPartySetMatchOptimization_C",(MatPartitioning,PetscBool),(part,opt));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatPartitioningPartySetMatchOptimization_Party"
PetscErrorCode MatPartitioningPartySetMatchOptimization_Party(MatPartitioning part,PetscBool opt)
{
  MatPartitioning_Party *party = (MatPartitioning_Party*)part->data;

  PetscFunctionBegin;
  party->redo = opt;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "MatPartitioningPartySetBipart"
/*@
   MatPartitioningPartySetBipart - Activate or deactivate recursive bisection.
 
   Collective on MatPartitioning

   Input Parameters:
+  part - the partitioning context
-  bp - boolean flag

   Options Database:
-  -mat_partitioning_party_bipart - Bipartitioning option on/off

   Level: advanced
@*/
PetscErrorCode MatPartitioningPartySetBipart(MatPartitioning part,PetscBool bp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part,MAT_PARTITIONING_CLASSID,1);
  PetscValidLogicalCollectiveBool(part,bp,2);
  ierr = PetscTryMethod(part,"MatPartitioningPartySetBipart_C",(MatPartitioning,PetscBool),(part,bp));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatPartitioningPartySetBipart_Party"
PetscErrorCode MatPartitioningPartySetBipart_Party(MatPartitioning part,PetscBool bp)
{
  MatPartitioning_Party *party = (MatPartitioning_Party*)part->data;

  PetscFunctionBegin;
  party->recursive = bp;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "MatPartitioningSetFromOptions_Party"
PetscErrorCode MatPartitioningSetFromOptions_Party(MatPartitioning part)
{
  PetscErrorCode        ierr;
  PetscBool             flag;
  char                  value[256];
  PetscReal             r;
  MatPartitioning_Party *party = (MatPartitioning_Party*)part->data;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("Set Party partitioning options");CHKERRQ(ierr);
    ierr = PetscOptionsString("-mat_partitioning_party_global","Global method","MatPartitioningPartySetGlobal",party->global,value,256,&flag);CHKERRQ(ierr);
    if (flag) { ierr = MatPartitioningPartySetGlobal(part,value);CHKERRQ(ierr); }
    ierr = PetscOptionsString("-mat_partitioning_party_local","Local method","MatPartitioningPartySetLocal",party->local,value,256,&flag);CHKERRQ(ierr);
    if (flag) { ierr = MatPartitioningPartySetLocal(part,value);CHKERRQ(ierr); }
    ierr = PetscOptionsReal("-mat_partitioning_party_coarse","Coarse level","MatPartitioningPartySetCoarseLevel",0.0,&r,&flag);CHKERRQ(ierr);
    if (flag) { ierr = MatPartitioningPartySetCoarseLevel(part,r);CHKERRQ(ierr); }
    ierr = PetscOptionsBool("-mat_partitioning_party_match_optimization","Matching optimization on/off","MatPartitioningPartySetMatchOptimization",party->redo,&party->redo,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-mat_partitioning_party_bipart","Bipartitioning on/off","MatPartitioningPartySetBipart",party->recursive,&party->recursive,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-mat_partitioning_party_verbose","Show library output","",party->verbose,&party->verbose,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPartitioningDestroy_Party"
PetscErrorCode MatPartitioningDestroy_Party(MatPartitioning part)
{
  MatPartitioning_Party *party = (MatPartitioning_Party*)part->data;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  ierr = PetscFree(party);CHKERRQ(ierr);
  /* clear composed functions */
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)part,"MatPartitioningPartySetGlobal_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)part,"MatPartitioningPartySetLocal_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)part,"MatPartitioningPartySetCoarseLevel_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)part,"MatPartitioningPartySetMatchOptimization_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)part,"MatPartitioningPartySetBipart_C","",PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
   MATPARTITIONINGPARTY - Creates a partitioning context via the external package Party.

   Level: beginner

   Notes: See http://wwwcs.upb.de/fachbereich/AG/monien/RESEARCH/PART/party.html

.keywords: Partitioning, create, context

.seealso: MatPartitioningSetType(), MatPartitioningType

M*/

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatPartitioningCreate_Party"
PetscErrorCode  MatPartitioningCreate_Party(MatPartitioning part)
{
  PetscErrorCode        ierr;
  MatPartitioning_Party *party;

  PetscFunctionBegin;
  ierr = PetscNewLog(part,MatPartitioning_Party,&party);CHKERRQ(ierr);
  part->data = (void*)party;

  ierr = PetscStrcpy(party->global,"gcf,gbf");CHKERRQ(ierr);
  ierr = PetscStrcpy(party->local,"kl");CHKERRQ(ierr);
  party->redm         = PETSC_TRUE;
  party->redo         = PETSC_TRUE;
  party->recursive    = PETSC_TRUE;
  party->verbose      = PETSC_FALSE;
  party->nbvtxcoarsed = 200;

  part->ops->apply          = MatPartitioningApply_Party;
  part->ops->view           = MatPartitioningView_Party;
  part->ops->destroy        = MatPartitioningDestroy_Party;
  part->ops->setfromoptions = MatPartitioningSetFromOptions_Party;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)part,"MatPartitioningPartySetGlobal_C","MatPartitioningPartySetGlobal_Party",MatPartitioningPartySetGlobal_Party);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)part,"MatPartitioningPartySetLocal_C","MatPartitioningPartySetLocal_Party",MatPartitioningPartySetLocal_Party);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)part,"MatPartitioningPartySetCoarseLevel_C","MatPartitioningPartySetCoarseLevel_Party",MatPartitioningPartySetCoarseLevel_Party);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)part,"MatPartitioningPartySetMatchOptimization_C","MatPartitioningPartySetMatchOptimization_Party",MatPartitioningPartySetMatchOptimization_Party);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)part,"MatPartitioningPartySetBipart_C","MatPartitioningPartySetBipart_Party",MatPartitioningPartySetBipart_Party);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
