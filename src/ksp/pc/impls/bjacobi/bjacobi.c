
/*
   Defines a block Jacobi preconditioner.
*/
#include <petsc-private/pcimpl.h>              /*I "petscpc.h" I*/
#include <../src/ksp/pc/impls/bjacobi/bjacobi.h>

static PetscErrorCode PCSetUp_BJacobi_Singleblock(PC,Mat,Mat);
static PetscErrorCode PCSetUp_BJacobi_Multiblock(PC,Mat,Mat);
static PetscErrorCode PCSetUp_BJacobi_Multiproc(PC);

#undef __FUNCT__
#define __FUNCT__ "PCSetUp_BJacobi"
static PetscErrorCode PCSetUp_BJacobi(PC pc)
{
  PC_BJacobi     *jac = (PC_BJacobi*)pc->data;
  Mat            mat = pc->mat,pmat = pc->pmat;
  PetscErrorCode ierr,(*f)(Mat,Mat*);
  PetscInt       N,M,start,i,sum,end;
  PetscInt       bs,i_start=-1,i_end=-1;
  PetscMPIInt    rank,size;
  const char     *pprefix,*mprefix;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(((PetscObject)pc)->comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(((PetscObject)pc)->comm,&size);CHKERRQ(ierr);
  ierr = MatGetLocalSize(pc->pmat,&M,&N);CHKERRQ(ierr);
  ierr = MatGetBlockSize(pc->pmat,&bs);CHKERRQ(ierr);

  if (jac->n > 0 && jac->n < size){
    ierr = PCSetUp_BJacobi_Multiproc(pc);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  /* --------------------------------------------------------------------------
      Determines the number of blocks assigned to each processor
  -----------------------------------------------------------------------------*/

  /*   local block count  given */
  if (jac->n_local > 0 && jac->n < 0) {
    ierr = MPI_Allreduce(&jac->n_local,&jac->n,1,MPIU_INT,MPI_SUM,((PetscObject)pc)->comm);CHKERRQ(ierr);
    if (jac->l_lens) { /* check that user set these correctly */
      sum = 0;
      for (i=0; i<jac->n_local; i++) {
        if (jac->l_lens[i]/bs*bs !=jac->l_lens[i]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Mat blocksize doesn't match block Jacobi layout");
        sum += jac->l_lens[i];
      }
      if (sum != M) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local lens set incorrectly");
    } else {
      ierr = PetscMalloc(jac->n_local*sizeof(PetscInt),&jac->l_lens);CHKERRQ(ierr);
      for (i=0; i<jac->n_local; i++) {
        jac->l_lens[i] = bs*((M/bs)/jac->n_local + (((M/bs) % jac->n_local) > i));
      }
    }
  } else if (jac->n > 0 && jac->n_local < 0) { /* global block count given */
    /* global blocks given: determine which ones are local */
    if (jac->g_lens) {
      /* check if the g_lens is has valid entries */
      for (i=0; i<jac->n; i++) {
        if (!jac->g_lens[i]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Zero block not allowed");
        if (jac->g_lens[i]/bs*bs != jac->g_lens[i]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Mat blocksize doesn't match block Jacobi layout");
      }
      if (size == 1) {
        jac->n_local = jac->n;
        ierr         = PetscMalloc(jac->n_local*sizeof(PetscInt),&jac->l_lens);CHKERRQ(ierr);
        ierr         = PetscMemcpy(jac->l_lens,jac->g_lens,jac->n_local*sizeof(PetscInt));CHKERRQ(ierr);
        /* check that user set these correctly */
        sum = 0;
        for (i=0; i<jac->n_local; i++) sum += jac->l_lens[i];
        if (sum != M) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Global lens set incorrectly");
      } else {
        ierr = MatGetOwnershipRange(pc->pmat,&start,&end);CHKERRQ(ierr);
        /* loop over blocks determing first one owned by me */
        sum = 0;
        for (i=0; i<jac->n+1; i++) {
          if (sum == start) { i_start = i; goto start_1;}
          if (i < jac->n) sum += jac->g_lens[i];
        }
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Block sizes used in PCBJacobiSetTotalBlocks()\nare not compatible with parallel matrix layout");
 start_1:
        for (i=i_start; i<jac->n+1; i++) {
          if (sum == end) { i_end = i; goto end_1; }
          if (i < jac->n) sum += jac->g_lens[i];
        }
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Block sizes used in PCBJacobiSetTotalBlocks()\nare not compatible with parallel matrix layout");
 end_1:
        jac->n_local = i_end - i_start;
        ierr         = PetscMalloc(jac->n_local*sizeof(PetscInt),&jac->l_lens);CHKERRQ(ierr);
        ierr         = PetscMemcpy(jac->l_lens,jac->g_lens+i_start,jac->n_local*sizeof(PetscInt));CHKERRQ(ierr);
      }
    } else { /* no global blocks given, determine then using default layout */
      jac->n_local = jac->n/size + ((jac->n % size) > rank);
      ierr         = PetscMalloc(jac->n_local*sizeof(PetscInt),&jac->l_lens);CHKERRQ(ierr);
      for (i=0; i<jac->n_local; i++) {
        jac->l_lens[i] = ((M/bs)/jac->n_local + (((M/bs) % jac->n_local) > i))*bs;
        if (!jac->l_lens[i]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Too many blocks given");
      }
    }
  } else if (jac->n < 0 && jac->n_local < 0) { /* no blocks given */
    jac->n         = size;
    jac->n_local   = 1;
    ierr           = PetscMalloc(sizeof(PetscInt),&jac->l_lens);CHKERRQ(ierr);
    jac->l_lens[0] = M;
  } else { /* jac->n > 0 && jac->n_local > 0 */
    if (!jac->l_lens) {
      ierr = PetscMalloc(jac->n_local*sizeof(PetscInt),&jac->l_lens);CHKERRQ(ierr);
      for (i=0; i<jac->n_local; i++) {
        jac->l_lens[i] = bs*((M/bs)/jac->n_local + (((M/bs) % jac->n_local) > i));
      }
    }
  }
  if (jac->n_local < 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Number of blocks is less than number of processors");

  /* -------------------------
      Determines mat and pmat
  ---------------------------*/
  ierr = PetscObjectQueryFunction((PetscObject)pc->mat,"MatGetDiagonalBlock_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (!f && size == 1) {
    mat  = pc->mat;
    pmat = pc->pmat;
  } else {
    if (jac->use_true_local) {
      /* use block from true matrix, not preconditioner matrix for local MatMult() */
      ierr = MatGetDiagonalBlock(pc->mat,&mat);CHKERRQ(ierr);
      /* make submatrix have same prefix as entire matrix */
      ierr = PetscObjectGetOptionsPrefix((PetscObject)pc->mat,&mprefix);CHKERRQ(ierr);
      ierr = PetscObjectSetOptionsPrefix((PetscObject)mat,mprefix);CHKERRQ(ierr);
    }
    if (pc->pmat != pc->mat || !jac->use_true_local) {
      ierr = MatGetDiagonalBlock(pc->pmat,&pmat);CHKERRQ(ierr);
      /* make submatrix have same prefix as entire matrix */
      ierr = PetscObjectGetOptionsPrefix((PetscObject)pc->pmat,&pprefix);CHKERRQ(ierr);
      ierr = PetscObjectSetOptionsPrefix((PetscObject)pmat,pprefix);CHKERRQ(ierr);
    } else {
      pmat = mat;
    }
  }

  /* ------
     Setup code depends on the number of blocks
  */
  if (jac->n_local == 1) {
    ierr = PCSetUp_BJacobi_Singleblock(pc,mat,pmat);CHKERRQ(ierr);
  } else {
    ierr = PCSetUp_BJacobi_Multiblock(pc,mat,pmat);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* Default destroy, if it has never been setup */
#undef __FUNCT__
#define __FUNCT__ "PCDestroy_BJacobi"
static PetscErrorCode PCDestroy_BJacobi(PC pc)
{
  PC_BJacobi     *jac = (PC_BJacobi*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(jac->g_lens);CHKERRQ(ierr);
  ierr = PetscFree(jac->l_lens);CHKERRQ(ierr);
  ierr = PetscFree(pc->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCSetFromOptions_BJacobi"

static PetscErrorCode PCSetFromOptions_BJacobi(PC pc)
{
  PC_BJacobi     *jac = (PC_BJacobi*)pc->data;
  PetscErrorCode ierr;
  PetscInt       blocks;
  PetscBool      flg;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("Block Jacobi options");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-pc_bjacobi_blocks","Total number of blocks","PCBJacobiSetTotalBlocks",jac->n,&blocks,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PCBJacobiSetTotalBlocks(pc,blocks,PETSC_NULL);CHKERRQ(ierr);
    }
    flg  = PETSC_FALSE;
    ierr = PetscOptionsBool("-pc_bjacobi_truelocal","Use the true matrix, not preconditioner matrix to define matrix vector product in sub-problems","PCBJacobiSetUseTrueLocal",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) {
      ierr = PCBJacobiSetUseTrueLocal(pc);CHKERRQ(ierr);
    }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCView_BJacobi"
static PetscErrorCode PCView_BJacobi(PC pc,PetscViewer viewer)
{
  PC_BJacobi           *jac = (PC_BJacobi*)pc->data;
  PC_BJacobi_Multiproc *mpjac = (PC_BJacobi_Multiproc*)jac->data;
  PetscErrorCode       ierr;
  PetscMPIInt          rank;
  PetscInt             i;
  PetscBool            iascii,isstring;
  PetscViewer          sviewer;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERSTRING,&isstring);CHKERRQ(ierr);
  if (iascii) {
    if (jac->use_true_local) {
      ierr = PetscViewerASCIIPrintf(viewer,"  block Jacobi: using true local matrix, number of blocks = %D\n",jac->n);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"  block Jacobi: number of blocks = %D\n",jac->n);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(((PetscObject)pc)->comm,&rank);CHKERRQ(ierr);
    if (jac->same_local_solves) {
      ierr = PetscViewerASCIIPrintf(viewer,"  Local solve is same for all blocks, in the following KSP and PC objects:\n");CHKERRQ(ierr);
      if (jac->ksp && !jac->psubcomm) {
        ierr = PetscViewerGetSingleton(viewer,&sviewer);CHKERRQ(ierr);
        if (!rank){
          ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
          ierr = KSPView(jac->ksp[0],sviewer);CHKERRQ(ierr);
          ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
        }
        ierr = PetscViewerRestoreSingleton(viewer,&sviewer);CHKERRQ(ierr);
      } else if (jac->psubcomm && !jac->psubcomm->color){
        ierr = PetscViewerASCIIGetStdout(mpjac->psubcomm->comm,&sviewer);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
        ierr = KSPView(*(jac->ksp),sviewer);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
      }
    } else {
      PetscInt n_global;
      ierr = MPI_Allreduce(&jac->n_local,&n_global,1,MPIU_INT,MPI_MAX,((PetscObject)pc)->comm);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedAllow(viewer,PETSC_TRUE);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  Local solve info for each block is in the following KSP and PC objects:\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] number of local blocks = %D, first local block number = %D\n",
                   rank,jac->n_local,jac->first_local);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      for (i=0; i<n_global; i++) {
        ierr = PetscViewerGetSingleton(viewer,&sviewer);CHKERRQ(ierr);
        if (i < jac->n_local) {
          ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] local block number %D\n",rank,i);CHKERRQ(ierr);
          ierr = KSPView(jac->ksp[i],sviewer);CHKERRQ(ierr);
          ierr = PetscViewerASCIISynchronizedPrintf(viewer,"- - - - - - - - - - - - - - - - - -\n");CHKERRQ(ierr);
        }
        ierr = PetscViewerRestoreSingleton(viewer,&sviewer);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedAllow(viewer,PETSC_FALSE);CHKERRQ(ierr);
    }
  } else if (isstring) {
    ierr = PetscViewerStringSPrintf(viewer," blks=%D",jac->n);CHKERRQ(ierr);
    ierr = PetscViewerGetSingleton(viewer,&sviewer);CHKERRQ(ierr);
    if (jac->ksp) {ierr = KSPView(jac->ksp[0],sviewer);CHKERRQ(ierr);}
    ierr = PetscViewerRestoreSingleton(viewer,&sviewer);CHKERRQ(ierr);
  } else {
    SETERRQ1(((PetscObject)pc)->comm,PETSC_ERR_SUP,"Viewer type %s not supported for block Jacobi",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------------*/

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PCBJacobiSetUseTrueLocal_BJacobi"
PetscErrorCode  PCBJacobiSetUseTrueLocal_BJacobi(PC pc)
{
  PC_BJacobi   *jac;

  PetscFunctionBegin;
  jac                 = (PC_BJacobi*)pc->data;
  jac->use_true_local = PETSC_TRUE;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PCBJacobiGetSubKSP_BJacobi"
PetscErrorCode  PCBJacobiGetSubKSP_BJacobi(PC pc,PetscInt *n_local,PetscInt *first_local,KSP **ksp)
{
  PC_BJacobi   *jac = (PC_BJacobi*)pc->data;;

  PetscFunctionBegin;
  if (!pc->setupcalled) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_ARG_WRONGSTATE,"Must call KSPSetUp() or PCSetUp() first");

  if (n_local)     *n_local     = jac->n_local;
  if (first_local) *first_local = jac->first_local;
  *ksp                          = jac->ksp;
  jac->same_local_solves        = PETSC_FALSE; /* Assume that local solves are now different;
                                                  not necessarily true though!  This flag is
                                                  used only for PCView_BJacobi() */
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PCBJacobiSetTotalBlocks_BJacobi"
PetscErrorCode  PCBJacobiSetTotalBlocks_BJacobi(PC pc,PetscInt blocks,PetscInt *lens)
{
  PC_BJacobi     *jac = (PC_BJacobi*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  if (pc->setupcalled > 0 && jac->n!=blocks) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_ORDER,"Cannot alter number of blocks after PCSetUp()/KSPSetUp() has been called");
  jac->n = blocks;
  if (!lens) {
    jac->g_lens = 0;
  } else {
    ierr = PetscMalloc(blocks*sizeof(PetscInt),&jac->g_lens);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory(pc,blocks*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscMemcpy(jac->g_lens,lens,blocks*sizeof(PetscInt));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PCBJacobiGetTotalBlocks_BJacobi"
PetscErrorCode  PCBJacobiGetTotalBlocks_BJacobi(PC pc, PetscInt *blocks, const PetscInt *lens[])
{
  PC_BJacobi *jac = (PC_BJacobi*) pc->data;

  PetscFunctionBegin;
  *blocks = jac->n;
  if (lens) *lens = jac->g_lens;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PCBJacobiSetLocalBlocks_BJacobi"
PetscErrorCode  PCBJacobiSetLocalBlocks_BJacobi(PC pc,PetscInt blocks,const PetscInt lens[])
{
  PC_BJacobi     *jac;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  jac = (PC_BJacobi*)pc->data;

  jac->n_local = blocks;
  if (!lens) {
    jac->l_lens = 0;
  } else {
    ierr = PetscMalloc(blocks*sizeof(PetscInt),&jac->l_lens);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory(pc,blocks*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscMemcpy(jac->l_lens,lens,blocks*sizeof(PetscInt));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PCBJacobiGetLocalBlocks_BJacobi"
PetscErrorCode  PCBJacobiGetLocalBlocks_BJacobi(PC pc, PetscInt *blocks, const PetscInt *lens[])
{
  PC_BJacobi *jac = (PC_BJacobi*) pc->data;

  PetscFunctionBegin;
  *blocks = jac->n_local;
  if (lens) *lens = jac->l_lens;
  PetscFunctionReturn(0);
}
EXTERN_C_END

/* -------------------------------------------------------------------------------------*/

#undef __FUNCT__
#define __FUNCT__ "PCBJacobiSetUseTrueLocal"
/*@
   PCBJacobiSetUseTrueLocal - Sets a flag to indicate that the block
   problem is associated with the linear system matrix instead of the
   default (where it is associated with the preconditioning matrix).
   That is, if the local system is solved iteratively then it iterates
   on the block from the matrix using the block from the preconditioner
   as the preconditioner for the local block.

   Logically Collective on PC

   Input Parameters:
.  pc - the preconditioner context

   Options Database Key:
.  -pc_bjacobi_truelocal - Activates PCBJacobiSetUseTrueLocal()

   Notes:
   For the common case in which the preconditioning and linear
   system matrices are identical, this routine is unnecessary.

   Level: intermediate

.keywords:  block, Jacobi, set, true, local, flag

.seealso: PCSetOperators(), PCBJacobiSetLocalBlocks()
@*/
PetscErrorCode  PCBJacobiSetUseTrueLocal(PC pc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCBJacobiSetUseTrueLocal_C",(PC),(pc));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBJacobiGetSubKSP"
/*@C
   PCBJacobiGetSubKSP - Gets the local KSP contexts for all blocks on
   this processor.

   Note Collective

   Input Parameter:
.  pc - the preconditioner context

   Output Parameters:
+  n_local - the number of blocks on this processor, or PETSC_NULL
.  first_local - the global number of the first block on this processor, or PETSC_NULL
-  ksp - the array of KSP contexts

   Notes:
   After PCBJacobiGetSubKSP() the array of KSP contexts is not to be freed.

   Currently for some matrix implementations only 1 block per processor
   is supported.

   You must call KSPSetUp() or PCSetUp() before calling PCBJacobiGetSubKSP().

   Fortran Usage: You must pass in a KSP array that is large enough to contain all the local KSPs.
      You can call PCBJacobiGetSubKSP(pc,nlocal,firstlocal,PETSC_NULL_OBJECT,ierr) to determine how large the
      KSP array must be.

   Level: advanced

.keywords:  block, Jacobi, get, sub, KSP, context

.seealso: PCBJacobiGetSubKSP()
@*/
PetscErrorCode  PCBJacobiGetSubKSP(PC pc,PetscInt *n_local,PetscInt *first_local,KSP *ksp[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscUseMethod(pc,"PCBJacobiGetSubKSP_C",(PC,PetscInt *,PetscInt *,KSP **),(pc,n_local,first_local,ksp));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBJacobiSetTotalBlocks"
/*@
   PCBJacobiSetTotalBlocks - Sets the global number of blocks for the block
   Jacobi preconditioner.

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
.  blocks - the number of blocks
-  lens - [optional] integer array containing the size of each block

   Options Database Key:
.  -pc_bjacobi_blocks <blocks> - Sets the number of global blocks

   Notes:
   Currently only a limited number of blocking configurations are supported.
   All processors sharing the PC must call this routine with the same data.

   Level: intermediate

.keywords:  set, number, Jacobi, global, total, blocks

.seealso: PCBJacobiSetUseTrueLocal(), PCBJacobiSetLocalBlocks()
@*/
PetscErrorCode  PCBJacobiSetTotalBlocks(PC pc,PetscInt blocks,const PetscInt lens[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  if (blocks <= 0) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Must have positive blocks");
  ierr = PetscTryMethod(pc,"PCBJacobiSetTotalBlocks_C",(PC,PetscInt,const PetscInt[]),(pc,blocks,lens));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBJacobiGetTotalBlocks"
/*@C
   PCBJacobiGetTotalBlocks - Gets the global number of blocks for the block
   Jacobi preconditioner.

   Not Collective

   Input Parameter:
.  pc - the preconditioner context

   Output parameters:
+  blocks - the number of blocks
-  lens - integer array containing the size of each block

   Level: intermediate

.keywords:  get, number, Jacobi, global, total, blocks

.seealso: PCBJacobiSetUseTrueLocal(), PCBJacobiGetLocalBlocks()
@*/
PetscErrorCode  PCBJacobiGetTotalBlocks(PC pc, PetscInt *blocks, const PetscInt *lens[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID,1);
  PetscValidIntPointer(blocks,2);
  ierr = PetscUseMethod(pc,"PCBJacobiGetTotalBlocks_C",(PC,PetscInt*, const PetscInt *[]),(pc,blocks,lens));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBJacobiSetLocalBlocks"
/*@
   PCBJacobiSetLocalBlocks - Sets the local number of blocks for the block
   Jacobi preconditioner.

   Not Collective

   Input Parameters:
+  pc - the preconditioner context
.  blocks - the number of blocks
-  lens - [optional] integer array containing size of each block

   Note:
   Currently only a limited number of blocking configurations are supported.

   Level: intermediate

.keywords: PC, set, number, Jacobi, local, blocks

.seealso: PCBJacobiSetUseTrueLocal(), PCBJacobiSetTotalBlocks()
@*/
PetscErrorCode  PCBJacobiSetLocalBlocks(PC pc,PetscInt blocks,const PetscInt lens[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  if (blocks < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Must have nonegative blocks");
  ierr = PetscTryMethod(pc,"PCBJacobiSetLocalBlocks_C",(PC,PetscInt,const PetscInt []),(pc,blocks,lens));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBJacobiGetLocalBlocks"
/*@C
   PCBJacobiGetLocalBlocks - Gets the local number of blocks for the block
   Jacobi preconditioner.

   Not Collective

   Input Parameters:
+  pc - the preconditioner context
.  blocks - the number of blocks
-  lens - [optional] integer array containing size of each block

   Note:
   Currently only a limited number of blocking configurations are supported.

   Level: intermediate

.keywords: PC, get, number, Jacobi, local, blocks

.seealso: PCBJacobiSetUseTrueLocal(), PCBJacobiGetTotalBlocks()
@*/
PetscErrorCode  PCBJacobiGetLocalBlocks(PC pc, PetscInt *blocks, const PetscInt *lens[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID,1);
  PetscValidIntPointer(blocks,2);
  ierr = PetscUseMethod(pc,"PCBJacobiGetLocalBlocks_C",(PC,PetscInt*, const PetscInt *[]),(pc,blocks,lens));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -----------------------------------------------------------------------------------*/

/*MC
   PCBJACOBI - Use block Jacobi preconditioning, each block is (approximately) solved with
           its own KSP object.

   Options Database Keys:
.  -pc_bjacobi_truelocal - Activates PCBJacobiSetUseTrueLocal()

   Notes: Each processor can have one or more blocks, but a block cannot be shared by more
     than one processor. Defaults to one block per processor.

     To set options on the solvers for each block append -sub_ to all the KSP, KSP, and PC
        options database keys. For example, -sub_pc_type ilu -sub_pc_factor_levels 1 -sub_ksp_type preonly

     To set the options on the solvers separate for each block call PCBJacobiGetSubKSP()
         and set the options directly on the resulting KSP object (you can access its PC
         KSPGetPC())

   Level: beginner

   Concepts: block Jacobi

   Developer Notes: This preconditioner does not currently work with CUDA/CUSP for a couple of reasons.
       (1) It creates seq vectors as work vectors that should be cusp
       (2) The use of VecPlaceArray() is not handled properly by CUSP (that is it will not know where
           the ownership of the vector is so may use wrong values) even if it did know the ownership
           it may induce extra copy ups and downs. Satish suggests a VecTransplantArray() to handle two
           vectors sharing the same pointer and handling the CUSP side as well instead of VecGetArray()/VecPlaceArray().


.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC,
           PCASM, PCBJacobiSetUseTrueLocal(), PCBJacobiGetSubKSP(), PCBJacobiSetTotalBlocks(),
           PCBJacobiSetLocalBlocks(), PCSetModifySubmatrices()
M*/

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PCCreate_BJacobi"
PetscErrorCode  PCCreate_BJacobi(PC pc)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  PC_BJacobi     *jac;

  PetscFunctionBegin;
  ierr = PetscNewLog(pc,PC_BJacobi,&jac);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(((PetscObject)pc)->comm,&rank);CHKERRQ(ierr);
  pc->ops->apply              = 0;
  pc->ops->applytranspose     = 0;
  pc->ops->setup              = PCSetUp_BJacobi;
  pc->ops->destroy            = PCDestroy_BJacobi;
  pc->ops->setfromoptions     = PCSetFromOptions_BJacobi;
  pc->ops->view               = PCView_BJacobi;
  pc->ops->applyrichardson    = 0;

  pc->data               = (void*)jac;
  jac->n                 = -1;
  jac->n_local           = -1;
  jac->first_local       = rank;
  jac->ksp               = 0;
  jac->use_true_local    = PETSC_FALSE;
  jac->same_local_solves = PETSC_TRUE;
  jac->g_lens            = 0;
  jac->l_lens            = 0;
  jac->psubcomm          = 0;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCBJacobiSetUseTrueLocal_C",
                    "PCBJacobiSetUseTrueLocal_BJacobi",
                    PCBJacobiSetUseTrueLocal_BJacobi);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCBJacobiGetSubKSP_C","PCBJacobiGetSubKSP_BJacobi",
                    PCBJacobiGetSubKSP_BJacobi);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCBJacobiSetTotalBlocks_C","PCBJacobiSetTotalBlocks_BJacobi",
                    PCBJacobiSetTotalBlocks_BJacobi);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCBJacobiGetTotalBlocks_C","PCBJacobiGetTotalBlocks_BJacobi",
                    PCBJacobiGetTotalBlocks_BJacobi);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCBJacobiSetLocalBlocks_C","PCBJacobiSetLocalBlocks_BJacobi",
                    PCBJacobiSetLocalBlocks_BJacobi);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCBJacobiGetLocalBlocks_C","PCBJacobiGetLocalBlocks_BJacobi",
                    PCBJacobiGetLocalBlocks_BJacobi);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END

/* --------------------------------------------------------------------------------------------*/
/*
        These are for a single block per processor; works for AIJ, BAIJ; Seq and MPI
*/
#undef __FUNCT__
#define __FUNCT__ "PCReset_BJacobi_Singleblock"
PetscErrorCode PCReset_BJacobi_Singleblock(PC pc)
{
  PC_BJacobi             *jac = (PC_BJacobi*)pc->data;
  PC_BJacobi_Singleblock *bjac = (PC_BJacobi_Singleblock*)jac->data;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  ierr = KSPReset(jac->ksp[0]);CHKERRQ(ierr);
  ierr = VecDestroy(&bjac->x);CHKERRQ(ierr);
  ierr = VecDestroy(&bjac->y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCDestroy_BJacobi_Singleblock"
PetscErrorCode PCDestroy_BJacobi_Singleblock(PC pc)
{
  PC_BJacobi             *jac = (PC_BJacobi*)pc->data;
  PC_BJacobi_Singleblock *bjac = (PC_BJacobi_Singleblock*)jac->data;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  ierr = PCReset_BJacobi_Singleblock(pc);CHKERRQ(ierr);
  ierr = KSPDestroy(&jac->ksp[0]);CHKERRQ(ierr);
  ierr = PetscFree(jac->ksp);CHKERRQ(ierr);
  ierr = PetscFree(jac->l_lens);CHKERRQ(ierr);
  ierr = PetscFree(jac->g_lens);CHKERRQ(ierr);
  ierr = PetscFree(bjac);CHKERRQ(ierr);
  ierr = PetscFree(pc->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCSetUpOnBlocks_BJacobi_Singleblock"
PetscErrorCode PCSetUpOnBlocks_BJacobi_Singleblock(PC pc)
{
  PetscErrorCode ierr;
  PC_BJacobi     *jac = (PC_BJacobi*)pc->data;

  PetscFunctionBegin;
  ierr = KSPSetUp(jac->ksp[0]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCApply_BJacobi_Singleblock"
PetscErrorCode PCApply_BJacobi_Singleblock(PC pc,Vec x,Vec y)
{
  PetscErrorCode         ierr;
  PC_BJacobi             *jac = (PC_BJacobi*)pc->data;
  PC_BJacobi_Singleblock *bjac = (PC_BJacobi_Singleblock*)jac->data;
  PetscScalar            *x_array,*y_array;
  PetscFunctionBegin;
  /*
      The VecPlaceArray() is to avoid having to copy the
    y vector into the bjac->x vector. The reason for
    the bjac->x vector is that we need a sequential vector
    for the sequential solve.
  */
  ierr = VecGetArray(x,&x_array);CHKERRQ(ierr);
  ierr = VecGetArray(y,&y_array);CHKERRQ(ierr);
  ierr = VecPlaceArray(bjac->x,x_array);CHKERRQ(ierr);
  ierr = VecPlaceArray(bjac->y,y_array);CHKERRQ(ierr);
  ierr = KSPSolve(jac->ksp[0],bjac->x,bjac->y);CHKERRQ(ierr);
  ierr = VecResetArray(bjac->x);CHKERRQ(ierr);
  ierr = VecResetArray(bjac->y);CHKERRQ(ierr);
  ierr = VecRestoreArray(x,&x_array);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&y_array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCApplySymmetricLeft_BJacobi_Singleblock"
PetscErrorCode PCApplySymmetricLeft_BJacobi_Singleblock(PC pc,Vec x,Vec y)
{
  PetscErrorCode         ierr;
  PC_BJacobi             *jac = (PC_BJacobi*)pc->data;
  PC_BJacobi_Singleblock *bjac = (PC_BJacobi_Singleblock*)jac->data;
  PetscScalar            *x_array,*y_array;
  PC                     subpc;

  PetscFunctionBegin;
  /*
      The VecPlaceArray() is to avoid having to copy the
    y vector into the bjac->x vector. The reason for
    the bjac->x vector is that we need a sequential vector
    for the sequential solve.
  */
  ierr = VecGetArray(x,&x_array);CHKERRQ(ierr);
  ierr = VecGetArray(y,&y_array);CHKERRQ(ierr);
  ierr = VecPlaceArray(bjac->x,x_array);CHKERRQ(ierr);
  ierr = VecPlaceArray(bjac->y,y_array);CHKERRQ(ierr);
  /* apply the symmetric left portion of the inner PC operator */
  /* note this by-passes the inner KSP and its options completely */
  ierr = KSPGetPC(jac->ksp[0],&subpc);CHKERRQ(ierr);
  ierr = PCApplySymmetricLeft(subpc,bjac->x,bjac->y);CHKERRQ(ierr);
  ierr = VecResetArray(bjac->x);CHKERRQ(ierr);
  ierr = VecResetArray(bjac->y);CHKERRQ(ierr);
  ierr = VecRestoreArray(x,&x_array);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&y_array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCApplySymmetricRight_BJacobi_Singleblock"
PetscErrorCode PCApplySymmetricRight_BJacobi_Singleblock(PC pc,Vec x,Vec y)
{
  PetscErrorCode         ierr;
  PC_BJacobi             *jac = (PC_BJacobi*)pc->data;
  PC_BJacobi_Singleblock *bjac = (PC_BJacobi_Singleblock*)jac->data;
  PetscScalar            *x_array,*y_array;
  PC                     subpc;

  PetscFunctionBegin;
  /*
      The VecPlaceArray() is to avoid having to copy the
    y vector into the bjac->x vector. The reason for
    the bjac->x vector is that we need a sequential vector
    for the sequential solve.
  */
  ierr = VecGetArray(x,&x_array);CHKERRQ(ierr);
  ierr = VecGetArray(y,&y_array);CHKERRQ(ierr);
  ierr = VecPlaceArray(bjac->x,x_array);CHKERRQ(ierr);
  ierr = VecPlaceArray(bjac->y,y_array);CHKERRQ(ierr);

  /* apply the symmetric right portion of the inner PC operator */
  /* note this by-passes the inner KSP and its options completely */

  ierr = KSPGetPC(jac->ksp[0],&subpc);CHKERRQ(ierr);
  ierr = PCApplySymmetricRight(subpc,bjac->x,bjac->y);CHKERRQ(ierr);

  ierr = VecRestoreArray(x,&x_array);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&y_array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCApplyTranspose_BJacobi_Singleblock"
PetscErrorCode PCApplyTranspose_BJacobi_Singleblock(PC pc,Vec x,Vec y)
{
  PetscErrorCode         ierr;
  PC_BJacobi             *jac = (PC_BJacobi*)pc->data;
  PC_BJacobi_Singleblock *bjac = (PC_BJacobi_Singleblock*)jac->data;
  PetscScalar            *x_array,*y_array;

  PetscFunctionBegin;
  /*
      The VecPlaceArray() is to avoid having to copy the
    y vector into the bjac->x vector. The reason for
    the bjac->x vector is that we need a sequential vector
    for the sequential solve.
  */
  ierr = VecGetArray(x,&x_array);CHKERRQ(ierr);
  ierr = VecGetArray(y,&y_array);CHKERRQ(ierr);
  ierr = VecPlaceArray(bjac->x,x_array);CHKERRQ(ierr);
  ierr = VecPlaceArray(bjac->y,y_array);CHKERRQ(ierr);
  ierr = KSPSolveTranspose(jac->ksp[0],bjac->x,bjac->y);CHKERRQ(ierr);
  ierr = VecResetArray(bjac->x);CHKERRQ(ierr);
  ierr = VecResetArray(bjac->y);CHKERRQ(ierr);
  ierr = VecRestoreArray(x,&x_array);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&y_array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCSetUp_BJacobi_Singleblock"
static PetscErrorCode PCSetUp_BJacobi_Singleblock(PC pc,Mat mat,Mat pmat)
{
  PC_BJacobi             *jac = (PC_BJacobi*)pc->data;
  PetscErrorCode         ierr;
  PetscInt               m;
  KSP                    ksp;
  PC_BJacobi_Singleblock *bjac;
  PetscBool              wasSetup = PETSC_TRUE;

  PetscFunctionBegin;

  if (!pc->setupcalled) {
    const char *prefix;

    if (!jac->ksp) {
      wasSetup = PETSC_FALSE;
      ierr = KSPCreate(PETSC_COMM_SELF,&ksp);CHKERRQ(ierr);
      ierr = PetscObjectIncrementTabLevel((PetscObject)ksp,(PetscObject)pc,1);CHKERRQ(ierr);
      ierr = PetscLogObjectParent(pc,ksp);CHKERRQ(ierr);
      ierr = KSPSetType(ksp,KSPPREONLY);CHKERRQ(ierr);
      ierr = PCGetOptionsPrefix(pc,&prefix);CHKERRQ(ierr);
      ierr = KSPSetOptionsPrefix(ksp,prefix);CHKERRQ(ierr);
      ierr = KSPAppendOptionsPrefix(ksp,"sub_");CHKERRQ(ierr);

      pc->ops->reset               = PCReset_BJacobi_Singleblock;
      pc->ops->destroy             = PCDestroy_BJacobi_Singleblock;
      pc->ops->apply               = PCApply_BJacobi_Singleblock;
      pc->ops->applysymmetricleft  = PCApplySymmetricLeft_BJacobi_Singleblock;
      pc->ops->applysymmetricright = PCApplySymmetricRight_BJacobi_Singleblock;
      pc->ops->applytranspose      = PCApplyTranspose_BJacobi_Singleblock;
      pc->ops->setuponblocks       = PCSetUpOnBlocks_BJacobi_Singleblock;

      ierr = PetscMalloc(sizeof(KSP),&jac->ksp);CHKERRQ(ierr);
      jac->ksp[0] = ksp;

      ierr = PetscNewLog(pc,PC_BJacobi_Singleblock,&bjac);CHKERRQ(ierr);
      jac->data    = (void*)bjac;
    } else {
      ksp  = jac->ksp[0];
      bjac = (PC_BJacobi_Singleblock *)jac->data;
    }

    /*
      The reason we need to generate these vectors is to serve
      as the right-hand side and solution vector for the solve on the
      block. We do not need to allocate space for the vectors since
      that is provided via VecPlaceArray() just before the call to
      KSPSolve() on the block.
    */
    ierr = MatGetSize(pmat,&m,&m);CHKERRQ(ierr);
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,m,PETSC_NULL,&bjac->x);CHKERRQ(ierr);
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,m,PETSC_NULL,&bjac->y);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(pc,bjac->x);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(pc,bjac->y);CHKERRQ(ierr);
  } else {
    ksp = jac->ksp[0];
    bjac = (PC_BJacobi_Singleblock *)jac->data;
  }
  if (jac->use_true_local) {
    ierr = KSPSetOperators(ksp,mat,pmat,pc->flag);CHKERRQ(ierr);
  }  else {
    ierr = KSPSetOperators(ksp,pmat,pmat,pc->flag);CHKERRQ(ierr);
  }
  if (!wasSetup && pc->setfromoptionscalled) {
    ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "PCReset_BJacobi_Multiblock"
PetscErrorCode PCReset_BJacobi_Multiblock(PC pc)
{
  PC_BJacobi            *jac = (PC_BJacobi*)pc->data;
  PC_BJacobi_Multiblock *bjac = (PC_BJacobi_Multiblock*)jac->data;
  PetscErrorCode        ierr;
  PetscInt              i;

  PetscFunctionBegin;
  if (bjac && bjac->pmat) {
    ierr = MatDestroyMatrices(jac->n_local,&bjac->pmat);CHKERRQ(ierr);
    if (jac->use_true_local) {
      ierr = MatDestroyMatrices(jac->n_local,&bjac->mat);CHKERRQ(ierr);
    }
  }

  for (i=0; i<jac->n_local; i++) {
    ierr = KSPReset(jac->ksp[i]);CHKERRQ(ierr);
    if (bjac && bjac->x) {
      ierr = VecDestroy(&bjac->x[i]);CHKERRQ(ierr);
      ierr = VecDestroy(&bjac->y[i]);CHKERRQ(ierr);
      ierr = ISDestroy(&bjac->is[i]);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(jac->l_lens);CHKERRQ(ierr);
  ierr = PetscFree(jac->g_lens);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCDestroy_BJacobi_Multiblock"
PetscErrorCode PCDestroy_BJacobi_Multiblock(PC pc)
{
  PC_BJacobi            *jac = (PC_BJacobi*)pc->data;
  PC_BJacobi_Multiblock *bjac = (PC_BJacobi_Multiblock*)jac->data;
  PetscErrorCode        ierr;
  PetscInt              i;

  PetscFunctionBegin;
  ierr = PCReset_BJacobi_Multiblock(pc);CHKERRQ(ierr);
  if (bjac) {
    ierr = PetscFree2(bjac->x,bjac->y);CHKERRQ(ierr);
    ierr = PetscFree(bjac->starts);CHKERRQ(ierr);
    ierr = PetscFree(bjac->is);CHKERRQ(ierr);
  }
  ierr = PetscFree(jac->data);CHKERRQ(ierr);
  for (i=0; i<jac->n_local; i++) {
    ierr = KSPDestroy(&jac->ksp[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(jac->ksp);CHKERRQ(ierr);
  ierr = PetscFree(pc->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCSetUpOnBlocks_BJacobi_Multiblock"
PetscErrorCode PCSetUpOnBlocks_BJacobi_Multiblock(PC pc)
{
  PC_BJacobi     *jac = (PC_BJacobi*)pc->data;
  PetscErrorCode ierr;
  PetscInt       i,n_local = jac->n_local;

  PetscFunctionBegin;
  for (i=0; i<n_local; i++) {
    ierr = KSPSetUp(jac->ksp[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
      Preconditioner for block Jacobi
*/
#undef __FUNCT__
#define __FUNCT__ "PCApply_BJacobi_Multiblock"
PetscErrorCode PCApply_BJacobi_Multiblock(PC pc,Vec x,Vec y)
{
  PC_BJacobi            *jac = (PC_BJacobi*)pc->data;
  PetscErrorCode        ierr;
  PetscInt              i,n_local = jac->n_local;
  PC_BJacobi_Multiblock *bjac = (PC_BJacobi_Multiblock*)jac->data;
  PetscScalar           *xin,*yin;

  PetscFunctionBegin;
  ierr = VecGetArray(x,&xin);CHKERRQ(ierr);
  ierr = VecGetArray(y,&yin);CHKERRQ(ierr);
  for (i=0; i<n_local; i++) {
    /*
       To avoid copying the subvector from x into a workspace we instead
       make the workspace vector array point to the subpart of the array of
       the global vector.
    */
    ierr = VecPlaceArray(bjac->x[i],xin+bjac->starts[i]);CHKERRQ(ierr);
    ierr = VecPlaceArray(bjac->y[i],yin+bjac->starts[i]);CHKERRQ(ierr);

    ierr = PetscLogEventBegin(PC_ApplyOnBlocks,jac->ksp[i],bjac->x[i],bjac->y[i],0);CHKERRQ(ierr);
    ierr = KSPSolve(jac->ksp[i],bjac->x[i],bjac->y[i]);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(PC_ApplyOnBlocks,jac->ksp[i],bjac->x[i],bjac->y[i],0);CHKERRQ(ierr);

    ierr = VecResetArray(bjac->x[i]);CHKERRQ(ierr);
    ierr = VecResetArray(bjac->y[i]);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(x,&xin);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&yin);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
      Preconditioner for block Jacobi
*/
#undef __FUNCT__
#define __FUNCT__ "PCApplyTranspose_BJacobi_Multiblock"
PetscErrorCode PCApplyTranspose_BJacobi_Multiblock(PC pc,Vec x,Vec y)
{
  PC_BJacobi            *jac = (PC_BJacobi*)pc->data;
  PetscErrorCode        ierr;
  PetscInt              i,n_local = jac->n_local;
  PC_BJacobi_Multiblock *bjac = (PC_BJacobi_Multiblock*)jac->data;
  PetscScalar           *xin,*yin;

  PetscFunctionBegin;
  ierr = VecGetArray(x,&xin);CHKERRQ(ierr);
  ierr = VecGetArray(y,&yin);CHKERRQ(ierr);
  for (i=0; i<n_local; i++) {
    /*
       To avoid copying the subvector from x into a workspace we instead
       make the workspace vector array point to the subpart of the array of
       the global vector.
    */
    ierr = VecPlaceArray(bjac->x[i],xin+bjac->starts[i]);CHKERRQ(ierr);
    ierr = VecPlaceArray(bjac->y[i],yin+bjac->starts[i]);CHKERRQ(ierr);

    ierr = PetscLogEventBegin(PC_ApplyTransposeOnBlocks,jac->ksp[i],bjac->x[i],bjac->y[i],0);CHKERRQ(ierr);
    ierr = KSPSolveTranspose(jac->ksp[i],bjac->x[i],bjac->y[i]);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(PC_ApplyTransposeOnBlocks,jac->ksp[i],bjac->x[i],bjac->y[i],0);CHKERRQ(ierr);

    ierr = VecResetArray(bjac->x[i]);CHKERRQ(ierr);
    ierr = VecResetArray(bjac->y[i]);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(x,&xin);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&yin);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCSetUp_BJacobi_Multiblock"
static PetscErrorCode PCSetUp_BJacobi_Multiblock(PC pc,Mat mat,Mat pmat)
{
  PC_BJacobi             *jac = (PC_BJacobi*)pc->data;
  PetscErrorCode         ierr;
  PetscInt               m,n_local,N,M,start,i;
  const char             *prefix,*pprefix,*mprefix;
  KSP                    ksp;
  Vec                    x,y;
  PC_BJacobi_Multiblock  *bjac = (PC_BJacobi_Multiblock*)jac->data;
  PC                     subpc;
  IS                     is;
  MatReuse               scall;

  PetscFunctionBegin;
  ierr = MatGetLocalSize(pc->pmat,&M,&N);CHKERRQ(ierr);

  n_local = jac->n_local;

  if (jac->use_true_local) {
    PetscBool  same;
    ierr = PetscObjectTypeCompare((PetscObject)mat,((PetscObject)pmat)->type_name,&same);CHKERRQ(ierr);
    if (!same) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_ARG_INCOMP,"Matrices not of same type");
  }

  if (!pc->setupcalled) {
    scall                  = MAT_INITIAL_MATRIX;

    if (!jac->ksp) {
      pc->ops->reset         = PCReset_BJacobi_Multiblock;
      pc->ops->destroy       = PCDestroy_BJacobi_Multiblock;
      pc->ops->apply         = PCApply_BJacobi_Multiblock;
      pc->ops->applytranspose= PCApplyTranspose_BJacobi_Multiblock;
      pc->ops->setuponblocks = PCSetUpOnBlocks_BJacobi_Multiblock;

      ierr = PetscNewLog(pc,PC_BJacobi_Multiblock,&bjac);CHKERRQ(ierr);
      ierr = PetscMalloc(n_local*sizeof(KSP),&jac->ksp);CHKERRQ(ierr);
      ierr = PetscLogObjectMemory(pc,sizeof(n_local*sizeof(KSP)));CHKERRQ(ierr);
      ierr = PetscMalloc2(n_local,Vec,&bjac->x,n_local,Vec,&bjac->y);CHKERRQ(ierr);
      ierr = PetscMalloc(n_local*sizeof(PetscScalar),&bjac->starts);CHKERRQ(ierr);
      ierr = PetscLogObjectMemory(pc,sizeof(n_local*sizeof(PetscScalar)));CHKERRQ(ierr);

      jac->data    = (void*)bjac;
      ierr = PetscMalloc(n_local*sizeof(IS),&bjac->is);CHKERRQ(ierr);
      ierr = PetscLogObjectMemory(pc,sizeof(n_local*sizeof(IS)));CHKERRQ(ierr);

      for (i=0; i<n_local; i++) {
        ierr = KSPCreate(PETSC_COMM_SELF,&ksp);CHKERRQ(ierr);
        ierr = PetscObjectIncrementTabLevel((PetscObject)ksp,(PetscObject)pc,1);CHKERRQ(ierr);
        ierr = PetscLogObjectParent(pc,ksp);CHKERRQ(ierr);
        ierr = KSPSetType(ksp,KSPPREONLY);CHKERRQ(ierr);
        ierr = KSPGetPC(ksp,&subpc);CHKERRQ(ierr);
        ierr = PCGetOptionsPrefix(pc,&prefix);CHKERRQ(ierr);
        ierr = KSPSetOptionsPrefix(ksp,prefix);CHKERRQ(ierr);
        ierr = KSPAppendOptionsPrefix(ksp,"sub_");CHKERRQ(ierr);
        jac->ksp[i]    = ksp;
      }
    } else {
      bjac = (PC_BJacobi_Multiblock*)jac->data;
    }

    start = 0;
    for (i=0; i<n_local; i++) {
      m = jac->l_lens[i];
      /*
      The reason we need to generate these vectors is to serve
      as the right-hand side and solution vector for the solve on the
      block. We do not need to allocate space for the vectors since
      that is provided via VecPlaceArray() just before the call to
      KSPSolve() on the block.

      */
      ierr = VecCreateSeq(PETSC_COMM_SELF,m,&x);CHKERRQ(ierr);
      ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,m,PETSC_NULL,&y);CHKERRQ(ierr);
      ierr = PetscLogObjectParent(pc,x);CHKERRQ(ierr);
      ierr = PetscLogObjectParent(pc,y);CHKERRQ(ierr);
      bjac->x[i]      = x;
      bjac->y[i]      = y;
      bjac->starts[i] = start;

      ierr = ISCreateStride(PETSC_COMM_SELF,m,start,1,&is);CHKERRQ(ierr);
      bjac->is[i] = is;
      ierr = PetscLogObjectParent(pc,is);CHKERRQ(ierr);

      start += m;
    }
  } else {
    bjac = (PC_BJacobi_Multiblock*)jac->data;
    /*
       Destroy the blocks from the previous iteration
    */
    if (pc->flag == DIFFERENT_NONZERO_PATTERN) {
      ierr = MatDestroyMatrices(n_local,&bjac->pmat);CHKERRQ(ierr);
      if (jac->use_true_local) {
        ierr = MatDestroyMatrices(n_local,&bjac->mat);CHKERRQ(ierr);
      }
      scall = MAT_INITIAL_MATRIX;
    } else {
      scall = MAT_REUSE_MATRIX;
    }
  }

  ierr = MatGetSubMatrices(pmat,n_local,bjac->is,bjac->is,scall,&bjac->pmat);CHKERRQ(ierr);
  if (jac->use_true_local) {
    ierr = PetscObjectGetOptionsPrefix((PetscObject)mat,&mprefix);CHKERRQ(ierr);
    ierr = MatGetSubMatrices(mat,n_local,bjac->is,bjac->is,scall,&bjac->mat);CHKERRQ(ierr);
  }
  /* Return control to the user so that the submatrices can be modified (e.g., to apply
     different boundary conditions for the submatrices than for the global problem) */
  ierr = PCModifySubMatrices(pc,n_local,bjac->is,bjac->is,bjac->pmat,pc->modifysubmatricesP);CHKERRQ(ierr);

  ierr = PetscObjectGetOptionsPrefix((PetscObject)pmat,&pprefix);CHKERRQ(ierr);
  for (i=0; i<n_local; i++) {
    ierr = PetscLogObjectParent(pc,bjac->pmat[i]);CHKERRQ(ierr);
    ierr = PetscObjectSetOptionsPrefix((PetscObject)bjac->pmat[i],pprefix);CHKERRQ(ierr);
    if (jac->use_true_local) {
      ierr = PetscLogObjectParent(pc,bjac->mat[i]);CHKERRQ(ierr);
      ierr = PetscObjectSetOptionsPrefix((PetscObject)bjac->mat[i],mprefix);CHKERRQ(ierr);
      ierr = KSPSetOperators(jac->ksp[i],bjac->mat[i],bjac->pmat[i],pc->flag);CHKERRQ(ierr);
    } else {
      ierr = KSPSetOperators(jac->ksp[i],bjac->pmat[i],bjac->pmat[i],pc->flag);CHKERRQ(ierr);
    }
    if (pc->setfromoptionscalled){
      ierr = KSPSetFromOptions(jac->ksp[i]);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------------------------*/
/*
      These are for a single block with multiple processes;
*/
#undef __FUNCT__
#define __FUNCT__ "PCReset_BJacobi_Multiproc"
static PetscErrorCode PCReset_BJacobi_Multiproc(PC pc)
{
  PC_BJacobi           *jac = (PC_BJacobi*)pc->data;
  PC_BJacobi_Multiproc *mpjac = (PC_BJacobi_Multiproc*)jac->data;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  ierr = VecDestroy(&mpjac->ysub);CHKERRQ(ierr);
  ierr = VecDestroy(&mpjac->xsub);CHKERRQ(ierr);
  ierr = MatDestroy(&mpjac->submats);CHKERRQ(ierr);
  if (jac->ksp){ierr = KSPReset(jac->ksp[0]);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCDestroy_BJacobi_Multiproc"
static PetscErrorCode PCDestroy_BJacobi_Multiproc(PC pc)
{
  PC_BJacobi           *jac = (PC_BJacobi*)pc->data;
  PC_BJacobi_Multiproc *mpjac = (PC_BJacobi_Multiproc*)jac->data;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  ierr = PCReset_BJacobi_Multiproc(pc);CHKERRQ(ierr);
  ierr = KSPDestroy(&jac->ksp[0]);CHKERRQ(ierr);
  ierr = PetscFree(jac->ksp);CHKERRQ(ierr);
  ierr = PetscSubcommDestroy(&mpjac->psubcomm);CHKERRQ(ierr);

  ierr = PetscFree(mpjac);CHKERRQ(ierr);
  ierr = PetscFree(pc->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCApply_BJacobi_Multiproc"
static PetscErrorCode PCApply_BJacobi_Multiproc(PC pc,Vec x,Vec y)
{
  PC_BJacobi           *jac = (PC_BJacobi*)pc->data;
  PC_BJacobi_Multiproc *mpjac = (PC_BJacobi_Multiproc*)jac->data;
  PetscErrorCode       ierr;
  PetscScalar          *xarray,*yarray;

  PetscFunctionBegin;
  /* place x's and y's local arrays into xsub and ysub */
  ierr = VecGetArray(x,&xarray);CHKERRQ(ierr);
  ierr = VecGetArray(y,&yarray);CHKERRQ(ierr);
  ierr = VecPlaceArray(mpjac->xsub,xarray);CHKERRQ(ierr);
  ierr = VecPlaceArray(mpjac->ysub,yarray);CHKERRQ(ierr);

  /* apply preconditioner on each matrix block */
  ierr = PetscLogEventBegin(PC_ApplyOnMproc,jac->ksp[0],mpjac->xsub,mpjac->ysub,0);CHKERRQ(ierr);
  ierr = KSPSolve(jac->ksp[0],mpjac->xsub,mpjac->ysub);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(PC_ApplyOnMproc,jac->ksp[0],mpjac->xsub,mpjac->ysub,0);CHKERRQ(ierr);

  ierr = VecResetArray(mpjac->xsub);CHKERRQ(ierr);
  ierr = VecResetArray(mpjac->ysub);CHKERRQ(ierr);
  ierr = VecRestoreArray(x,&xarray);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&yarray);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

extern PetscErrorCode MatGetMultiProcBlock_MPIAIJ(Mat,MPI_Comm,MatReuse,Mat*);
#undef __FUNCT__
#define __FUNCT__ "PCSetUp_BJacobi_Multiproc"
static PetscErrorCode PCSetUp_BJacobi_Multiproc(PC pc)
{
  PC_BJacobi           *jac = (PC_BJacobi*)pc->data;
  PC_BJacobi_Multiproc *mpjac = (PC_BJacobi_Multiproc*)jac->data;
  PetscErrorCode       ierr;
  PetscInt             m,n;
  MPI_Comm             comm = ((PetscObject)pc)->comm,subcomm=0;
  const char           *prefix;
  PetscBool            wasSetup = PETSC_TRUE;

  PetscFunctionBegin;
  if (jac->n_local > 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Only a single block in a subcommunicator is supported");
  jac->n_local = 1; /* currently only a single block is supported for a subcommunicator */
  if (!pc->setupcalled) {
    wasSetup = PETSC_FALSE;
    ierr = PetscNewLog(pc,PC_BJacobi_Multiproc,&mpjac);CHKERRQ(ierr);
    jac->data = (void*)mpjac;

    /* initialize datastructure mpjac */
    if (!jac->psubcomm) {
      /* Create default contiguous subcommunicatiors if user does not provide them */
      ierr = PetscSubcommCreate(comm,&jac->psubcomm);CHKERRQ(ierr);
      ierr = PetscSubcommSetNumber(jac->psubcomm,jac->n);CHKERRQ(ierr);
      ierr = PetscSubcommSetType(jac->psubcomm,PETSC_SUBCOMM_CONTIGUOUS);CHKERRQ(ierr);
      ierr = PetscLogObjectMemory(pc,sizeof(PetscSubcomm));CHKERRQ(ierr);
    }
    mpjac->psubcomm = jac->psubcomm;
    subcomm         = mpjac->psubcomm->comm;

    /* Get matrix blocks of pmat */
    ierr = MatGetMultiProcBlock_MPIAIJ(pc->pmat,subcomm,MAT_INITIAL_MATRIX,&mpjac->submats);CHKERRQ(ierr);

    /* create a new PC that processors in each subcomm have copy of */
    ierr = PetscMalloc(sizeof(KSP),&jac->ksp);CHKERRQ(ierr);
    ierr = KSPCreate(subcomm,&jac->ksp[0]);CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)jac->ksp[0],(PetscObject)pc,1);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(pc,jac->ksp[0]);CHKERRQ(ierr);
    ierr = KSPSetOperators(jac->ksp[0],mpjac->submats,mpjac->submats,pc->flag);CHKERRQ(ierr);
    ierr = KSPGetPC(jac->ksp[0],&mpjac->pc);CHKERRQ(ierr);

    ierr = PCGetOptionsPrefix(pc,&prefix);CHKERRQ(ierr);
    ierr = KSPSetOptionsPrefix(jac->ksp[0],prefix);CHKERRQ(ierr);
    ierr = KSPAppendOptionsPrefix(jac->ksp[0],"sub_");CHKERRQ(ierr);
    /*
      PetscMPIInt rank,subsize,subrank;
      ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
      ierr = MPI_Comm_size(subcomm,&subsize);CHKERRQ(ierr);
      ierr = MPI_Comm_rank(subcomm,&subrank);CHKERRQ(ierr);

      ierr = MatGetLocalSize(mpjac->submats,&m,PETSC_NULL);CHKERRQ(ierr);
      ierr = MatGetSize(mpjac->submats,&n,PETSC_NULL);CHKERRQ(ierr);
      ierr = PetscSynchronizedPrintf(comm,"[%d], sub-size %d,sub-rank %d\n",rank,subsize,subrank);
      ierr = PetscSynchronizedFlush(comm);CHKERRQ(ierr);
    */

    /* create dummy vectors xsub and ysub */
    ierr = MatGetLocalSize(mpjac->submats,&m,&n);CHKERRQ(ierr);
    ierr = VecCreateMPIWithArray(subcomm,1,n,PETSC_DECIDE,PETSC_NULL,&mpjac->xsub);CHKERRQ(ierr);
    ierr = VecCreateMPIWithArray(subcomm,1,m,PETSC_DECIDE,PETSC_NULL,&mpjac->ysub);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(pc,mpjac->xsub);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(pc,mpjac->ysub);CHKERRQ(ierr);

    pc->ops->reset   = PCReset_BJacobi_Multiproc;
    pc->ops->destroy = PCDestroy_BJacobi_Multiproc;
    pc->ops->apply   = PCApply_BJacobi_Multiproc;
  } else { /* pc->setupcalled */
    subcomm = mpjac->psubcomm->comm;
    if (pc->flag == DIFFERENT_NONZERO_PATTERN) {
      /* destroy old matrix blocks, then get new matrix blocks */
      if (mpjac->submats){ierr = MatDestroy(&mpjac->submats);CHKERRQ(ierr);}
      ierr = MatGetMultiProcBlock_MPIAIJ(pc->pmat,subcomm,MAT_INITIAL_MATRIX,&mpjac->submats);CHKERRQ(ierr);
    } else {
      ierr = MatGetMultiProcBlock_MPIAIJ(pc->pmat,subcomm,MAT_REUSE_MATRIX,&mpjac->submats);CHKERRQ(ierr);
    }
    ierr = KSPSetOperators(jac->ksp[0],mpjac->submats,mpjac->submats,pc->flag);CHKERRQ(ierr);
  }

  if (!wasSetup && pc->setfromoptionscalled){
    ierr = KSPSetFromOptions(jac->ksp[0]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
