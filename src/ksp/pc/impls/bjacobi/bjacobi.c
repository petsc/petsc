
/*
   Defines a block Jacobi preconditioner.
*/

#include <../src/ksp/pc/impls/bjacobi/bjacobi.h> /*I "petscpc.h" I*/

static PetscErrorCode PCSetUp_BJacobi_Singleblock(PC,Mat,Mat);
static PetscErrorCode PCSetUp_BJacobi_Multiblock(PC,Mat,Mat);
static PetscErrorCode PCSetUp_BJacobi_Multiproc(PC);

static PetscErrorCode PCSetUp_BJacobi(PC pc)
{
  PC_BJacobi     *jac = (PC_BJacobi*)pc->data;
  Mat            mat  = pc->mat,pmat = pc->pmat;
  PetscErrorCode ierr;
  PetscBool      hasop;
  PetscInt       N,M,start,i,sum,end;
  PetscInt       bs,i_start=-1,i_end=-1;
  PetscMPIInt    rank,size;
  const char     *pprefix,*mprefix;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)pc),&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)pc),&size);CHKERRQ(ierr);
  ierr = MatGetLocalSize(pc->pmat,&M,&N);CHKERRQ(ierr);
  ierr = MatGetBlockSize(pc->pmat,&bs);CHKERRQ(ierr);

  if (jac->n > 0 && jac->n < size) {
    ierr = PCSetUp_BJacobi_Multiproc(pc);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  /* --------------------------------------------------------------------------
      Determines the number of blocks assigned to each processor
  -----------------------------------------------------------------------------*/

  /*   local block count  given */
  if (jac->n_local > 0 && jac->n < 0) {
    ierr = MPIU_Allreduce(&jac->n_local,&jac->n,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)pc));CHKERRQ(ierr);
    if (jac->l_lens) { /* check that user set these correctly */
      sum = 0;
      for (i=0; i<jac->n_local; i++) {
        if (jac->l_lens[i]/bs*bs !=jac->l_lens[i]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Mat blocksize doesn't match block Jacobi layout");
        sum += jac->l_lens[i];
      }
      if (sum != M) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local lens set incorrectly");
    } else {
      ierr = PetscMalloc1(jac->n_local,&jac->l_lens);CHKERRQ(ierr);
      for (i=0; i<jac->n_local; i++) jac->l_lens[i] = bs*((M/bs)/jac->n_local + (((M/bs) % jac->n_local) > i));
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
        ierr         = PetscMalloc1(jac->n_local,&jac->l_lens);CHKERRQ(ierr);
        ierr         = PetscArraycpy(jac->l_lens,jac->g_lens,jac->n_local);CHKERRQ(ierr);
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
        ierr         = PetscMalloc1(jac->n_local,&jac->l_lens);CHKERRQ(ierr);
        ierr         = PetscArraycpy(jac->l_lens,jac->g_lens+i_start,jac->n_local);CHKERRQ(ierr);
      }
    } else { /* no global blocks given, determine then using default layout */
      jac->n_local = jac->n/size + ((jac->n % size) > rank);
      ierr         = PetscMalloc1(jac->n_local,&jac->l_lens);CHKERRQ(ierr);
      for (i=0; i<jac->n_local; i++) {
        jac->l_lens[i] = ((M/bs)/jac->n_local + (((M/bs) % jac->n_local) > i))*bs;
        if (!jac->l_lens[i]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Too many blocks given");
      }
    }
  } else if (jac->n < 0 && jac->n_local < 0) { /* no blocks given */
    jac->n         = size;
    jac->n_local   = 1;
    ierr           = PetscMalloc1(1,&jac->l_lens);CHKERRQ(ierr);
    jac->l_lens[0] = M;
  } else { /* jac->n > 0 && jac->n_local > 0 */
    if (!jac->l_lens) {
      ierr = PetscMalloc1(jac->n_local,&jac->l_lens);CHKERRQ(ierr);
      for (i=0; i<jac->n_local; i++) jac->l_lens[i] = bs*((M/bs)/jac->n_local + (((M/bs) % jac->n_local) > i));
    }
  }
  if (jac->n_local < 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Number of blocks is less than number of processors");

  /* -------------------------
      Determines mat and pmat
  ---------------------------*/
  ierr = MatHasOperation(pc->mat,MATOP_GET_DIAGONAL_BLOCK,&hasop);CHKERRQ(ierr);
  if (!hasop && size == 1) {
    mat  = pc->mat;
    pmat = pc->pmat;
  } else {
    if (pc->useAmat) {
      /* use block from Amat matrix, not Pmat for local MatMult() */
      ierr = MatGetDiagonalBlock(pc->mat,&mat);CHKERRQ(ierr);
      /* make submatrix have same prefix as entire matrix */
      ierr = PetscObjectGetOptionsPrefix((PetscObject)pc->mat,&mprefix);CHKERRQ(ierr);
      ierr = PetscObjectSetOptionsPrefix((PetscObject)mat,mprefix);CHKERRQ(ierr);
    }
    if (pc->pmat != pc->mat || !pc->useAmat) {
      ierr = MatGetDiagonalBlock(pc->pmat,&pmat);CHKERRQ(ierr);
      /* make submatrix have same prefix as entire matrix */
      ierr = PetscObjectGetOptionsPrefix((PetscObject)pc->pmat,&pprefix);CHKERRQ(ierr);
      ierr = PetscObjectSetOptionsPrefix((PetscObject)pmat,pprefix);CHKERRQ(ierr);
    } else pmat = mat;
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


static PetscErrorCode PCSetFromOptions_BJacobi(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PC_BJacobi     *jac = (PC_BJacobi*)pc->data;
  PetscErrorCode ierr;
  PetscInt       blocks,i;
  PetscBool      flg;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"Block Jacobi options");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_bjacobi_blocks","Total number of blocks","PCBJacobiSetTotalBlocks",jac->n,&blocks,&flg);CHKERRQ(ierr);
  if (flg) {ierr = PCBJacobiSetTotalBlocks(pc,blocks,NULL);CHKERRQ(ierr);}
  ierr = PetscOptionsInt("-pc_bjacobi_local_blocks","Local number of blocks","PCBJacobiSetLocalBlocks",jac->n_local,&blocks,&flg);CHKERRQ(ierr);
  if (flg) {ierr = PCBJacobiSetLocalBlocks(pc,blocks,NULL);CHKERRQ(ierr);}
  if (jac->ksp) {
    /* The sub-KSP has already been set up (e.g., PCSetUp_BJacobi_Singleblock), but KSPSetFromOptions was not called
     * unless we had already been called. */
    for (i=0; i<jac->n_local; i++) {
      ierr = KSPSetFromOptions(jac->ksp[i]);CHKERRQ(ierr);
    }
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#include <petscdraw.h>
static PetscErrorCode PCView_BJacobi(PC pc,PetscViewer viewer)
{
  PC_BJacobi           *jac   = (PC_BJacobi*)pc->data;
  PC_BJacobi_Multiproc *mpjac = (PC_BJacobi_Multiproc*)jac->data;
  PetscErrorCode       ierr;
  PetscMPIInt          rank;
  PetscInt             i;
  PetscBool            iascii,isstring,isdraw;
  PetscViewer          sviewer;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERSTRING,&isstring);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERDRAW,&isdraw);CHKERRQ(ierr);
  if (iascii) {
    if (pc->useAmat) {
      ierr = PetscViewerASCIIPrintf(viewer,"  using Amat local matrix, number of blocks = %D\n",jac->n);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"  number of blocks = %D\n",jac->n);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)pc),&rank);CHKERRQ(ierr);
    if (jac->same_local_solves) {
      ierr = PetscViewerASCIIPrintf(viewer,"  Local solver is the same for all blocks, as in the following KSP and PC objects on rank 0:\n");CHKERRQ(ierr);
      if (jac->ksp && !jac->psubcomm) {
        ierr = PetscViewerGetSubViewer(viewer,PETSC_COMM_SELF,&sviewer);CHKERRQ(ierr);
        if (!rank) {
          ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
          ierr = KSPView(jac->ksp[0],sviewer);CHKERRQ(ierr);
          ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
        }
        ierr = PetscViewerFlush(sviewer);CHKERRQ(ierr);
        ierr = PetscViewerRestoreSubViewer(viewer,PETSC_COMM_SELF,&sviewer);CHKERRQ(ierr);
      } else if (mpjac && jac->ksp && mpjac->psubcomm) {
        ierr = PetscViewerGetSubViewer(viewer,mpjac->psubcomm->child,&sviewer);CHKERRQ(ierr);
        if (!mpjac->psubcomm->color) {
          ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
          ierr = KSPView(*(jac->ksp),sviewer);CHKERRQ(ierr);
          ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
        }
        ierr = PetscViewerFlush(sviewer);CHKERRQ(ierr);
        ierr = PetscViewerRestoreSubViewer(viewer,mpjac->psubcomm->child,&sviewer);CHKERRQ(ierr);
      }
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      /*  extra call needed because of the two calls to PetscViewerASCIIPushSynchronized() in PetscViewerGetSubViewer() */
      ierr = PetscViewerASCIIPopSynchronized(viewer);CHKERRQ(ierr);
    } else {
      PetscInt n_global;
      ierr = MPIU_Allreduce(&jac->n_local,&n_global,1,MPIU_INT,MPI_MAX,PetscObjectComm((PetscObject)pc));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPushSynchronized(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  Local solve info for each block is in the following KSP and PC objects:\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] number of local blocks = %D, first local block number = %D\n",
                                                rank,jac->n_local,jac->first_local);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = PetscViewerGetSubViewer(viewer,PETSC_COMM_SELF,&sviewer);CHKERRQ(ierr);
      for (i=0; i<jac->n_local; i++) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] local block number %D\n",rank,i);CHKERRQ(ierr);
        ierr = KSPView(jac->ksp[i],sviewer);CHKERRQ(ierr);
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"- - - - - - - - - - - - - - - - - -\n");CHKERRQ(ierr);
      }
      ierr = PetscViewerRestoreSubViewer(viewer,PETSC_COMM_SELF,&sviewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopSynchronized(viewer);CHKERRQ(ierr);
    }
  } else if (isstring) {
    ierr = PetscViewerStringSPrintf(viewer," blks=%D",jac->n);CHKERRQ(ierr);
    ierr = PetscViewerGetSubViewer(viewer,PETSC_COMM_SELF,&sviewer);CHKERRQ(ierr);
    if (jac->ksp) {ierr = KSPView(jac->ksp[0],sviewer);CHKERRQ(ierr);}
    ierr = PetscViewerRestoreSubViewer(viewer,PETSC_COMM_SELF,&sviewer);CHKERRQ(ierr);
  } else if (isdraw) {
    PetscDraw draw;
    char      str[25];
    PetscReal x,y,bottom,h;

    ierr   = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
    ierr   = PetscDrawGetCurrentPoint(draw,&x,&y);CHKERRQ(ierr);
    ierr   = PetscSNPrintf(str,25,"Number blocks %D",jac->n);CHKERRQ(ierr);
    ierr   = PetscDrawStringBoxed(draw,x,y,PETSC_DRAW_RED,PETSC_DRAW_BLACK,str,NULL,&h);CHKERRQ(ierr);
    bottom = y - h;
    ierr   = PetscDrawPushCurrentPoint(draw,x,bottom);CHKERRQ(ierr);
    /* warning the communicator on viewer is different then on ksp in parallel */
    if (jac->ksp) {ierr = KSPView(jac->ksp[0],viewer);CHKERRQ(ierr);}
    ierr = PetscDrawPopCurrentPoint(draw);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------------*/

static PetscErrorCode  PCBJacobiGetSubKSP_BJacobi(PC pc,PetscInt *n_local,PetscInt *first_local,KSP **ksp)
{
  PC_BJacobi *jac = (PC_BJacobi*)pc->data;

  PetscFunctionBegin;
  if (!pc->setupcalled) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"Must call KSPSetUp() or PCSetUp() first");

  if (n_local) *n_local = jac->n_local;
  if (first_local) *first_local = jac->first_local;
  *ksp                   = jac->ksp;
  jac->same_local_solves = PETSC_FALSE;        /* Assume that local solves are now different;
                                                  not necessarily true though!  This flag is
                                                  used only for PCView_BJacobi() */
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCBJacobiSetTotalBlocks_BJacobi(PC pc,PetscInt blocks,PetscInt *lens)
{
  PC_BJacobi     *jac = (PC_BJacobi*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (pc->setupcalled > 0 && jac->n!=blocks) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_ORDER,"Cannot alter number of blocks after PCSetUp()/KSPSetUp() has been called");
  jac->n = blocks;
  if (!lens) jac->g_lens = NULL;
  else {
    ierr = PetscMalloc1(blocks,&jac->g_lens);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)pc,blocks*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscArraycpy(jac->g_lens,lens,blocks);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCBJacobiGetTotalBlocks_BJacobi(PC pc, PetscInt *blocks, const PetscInt *lens[])
{
  PC_BJacobi *jac = (PC_BJacobi*) pc->data;

  PetscFunctionBegin;
  *blocks = jac->n;
  if (lens) *lens = jac->g_lens;
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCBJacobiSetLocalBlocks_BJacobi(PC pc,PetscInt blocks,const PetscInt lens[])
{
  PC_BJacobi     *jac;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  jac = (PC_BJacobi*)pc->data;

  jac->n_local = blocks;
  if (!lens) jac->l_lens = NULL;
  else {
    ierr = PetscMalloc1(blocks,&jac->l_lens);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)pc,blocks*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscArraycpy(jac->l_lens,lens,blocks);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCBJacobiGetLocalBlocks_BJacobi(PC pc, PetscInt *blocks, const PetscInt *lens[])
{
  PC_BJacobi *jac = (PC_BJacobi*) pc->data;

  PetscFunctionBegin;
  *blocks = jac->n_local;
  if (lens) *lens = jac->l_lens;
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------------*/

/*@C
   PCBJacobiGetSubKSP - Gets the local KSP contexts for all blocks on
   this processor.

   Not Collective

   Input Parameter:
.  pc - the preconditioner context

   Output Parameters:
+  n_local - the number of blocks on this processor, or NULL
.  first_local - the global number of the first block on this processor, or NULL
-  ksp - the array of KSP contexts

   Notes:
   After PCBJacobiGetSubKSP() the array of KSP contexts is not to be freed.

   Currently for some matrix implementations only 1 block per processor
   is supported.

   You must call KSPSetUp() or PCSetUp() before calling PCBJacobiGetSubKSP().

   Fortran Usage: You must pass in a KSP array that is large enough to contain all the local KSPs.
      You can call PCBJacobiGetSubKSP(pc,nlocal,firstlocal,PETSC_NULL_KSP,ierr) to determine how large the
      KSP array must be.

   Level: advanced

.seealso: PCBJacobiGetSubKSP()
@*/
PetscErrorCode  PCBJacobiGetSubKSP(PC pc,PetscInt *n_local,PetscInt *first_local,KSP *ksp[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscUseMethod(pc,"PCBJacobiGetSubKSP_C",(PC,PetscInt*,PetscInt*,KSP **),(pc,n_local,first_local,ksp));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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

.seealso: PCSetUseAmat(), PCBJacobiSetLocalBlocks()
@*/
PetscErrorCode  PCBJacobiSetTotalBlocks(PC pc,PetscInt blocks,const PetscInt lens[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  if (blocks <= 0) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_OUTOFRANGE,"Must have positive blocks");
  ierr = PetscTryMethod(pc,"PCBJacobiSetTotalBlocks_C",(PC,PetscInt,const PetscInt[]),(pc,blocks,lens));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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

.seealso: PCSetUseAmat(), PCBJacobiGetLocalBlocks()
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

/*@
   PCBJacobiSetLocalBlocks - Sets the local number of blocks for the block
   Jacobi preconditioner.

   Not Collective

   Input Parameters:
+  pc - the preconditioner context
.  blocks - the number of blocks
-  lens - [optional] integer array containing size of each block

   Options Database Key:
.  -pc_bjacobi_local_blocks <blocks> - Sets the number of local blocks

   Note:
   Currently only a limited number of blocking configurations are supported.

   Level: intermediate

.seealso: PCSetUseAmat(), PCBJacobiSetTotalBlocks()
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

.seealso: PCSetUseAmat(), PCBJacobiGetTotalBlocks()
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
+  -pc_use_amat - use Amat to apply block of operator in inner Krylov method
-  -pc_bjacobi_blocks <n> - use n total blocks

   Notes:
    Each processor can have one or more blocks, or a single block can be shared by several processes. Defaults to one block per processor.

     To set options on the solvers for each block append -sub_ to all the KSP, KSP, and PC
        options database keys. For example, -sub_pc_type ilu -sub_pc_factor_levels 1 -sub_ksp_type preonly

     To set the options on the solvers separate for each block call PCBJacobiGetSubKSP()
         and set the options directly on the resulting KSP object (you can access its PC
         KSPGetPC())

     For GPU-based vectors (CUDA, ViennaCL) it is recommended to use exactly one block per MPI process for best
         performance.  Different block partitioning may lead to additional data transfers
         between host and GPU that lead to degraded performance.

     The options prefix for each block is sub_, for example -sub_pc_type lu.

     When multiple processes share a single block, each block encompasses exactly all the unknowns owned its set of processes.

   Level: beginner

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC,
           PCASM, PCSetUseAmat(), PCGetUseAmat(), PCBJacobiGetSubKSP(), PCBJacobiSetTotalBlocks(),
           PCBJacobiSetLocalBlocks(), PCSetModifySubMatrices()
M*/

PETSC_EXTERN PetscErrorCode PCCreate_BJacobi(PC pc)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  PC_BJacobi     *jac;

  PetscFunctionBegin;
  ierr = PetscNewLog(pc,&jac);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)pc),&rank);CHKERRQ(ierr);

  pc->ops->apply           = NULL;
  pc->ops->matapply        = NULL;
  pc->ops->applytranspose  = NULL;
  pc->ops->setup           = PCSetUp_BJacobi;
  pc->ops->destroy         = PCDestroy_BJacobi;
  pc->ops->setfromoptions  = PCSetFromOptions_BJacobi;
  pc->ops->view            = PCView_BJacobi;
  pc->ops->applyrichardson = NULL;

  pc->data               = (void*)jac;
  jac->n                 = -1;
  jac->n_local           = -1;
  jac->first_local       = rank;
  jac->ksp               = NULL;
  jac->same_local_solves = PETSC_TRUE;
  jac->g_lens            = NULL;
  jac->l_lens            = NULL;
  jac->psubcomm          = NULL;

  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBJacobiGetSubKSP_C",PCBJacobiGetSubKSP_BJacobi);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBJacobiSetTotalBlocks_C",PCBJacobiSetTotalBlocks_BJacobi);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBJacobiGetTotalBlocks_C",PCBJacobiGetTotalBlocks_BJacobi);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBJacobiSetLocalBlocks_C",PCBJacobiSetLocalBlocks_BJacobi);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBJacobiGetLocalBlocks_C",PCBJacobiGetLocalBlocks_BJacobi);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------------*/
/*
        These are for a single block per processor; works for AIJ, BAIJ; Seq and MPI
*/
static PetscErrorCode PCReset_BJacobi_Singleblock(PC pc)
{
  PC_BJacobi             *jac  = (PC_BJacobi*)pc->data;
  PC_BJacobi_Singleblock *bjac = (PC_BJacobi_Singleblock*)jac->data;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  ierr = KSPReset(jac->ksp[0]);CHKERRQ(ierr);
  ierr = VecDestroy(&bjac->x);CHKERRQ(ierr);
  ierr = VecDestroy(&bjac->y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDestroy_BJacobi_Singleblock(PC pc)
{
  PC_BJacobi             *jac  = (PC_BJacobi*)pc->data;
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

static PetscErrorCode PCSetUpOnBlocks_BJacobi_Singleblock(PC pc)
{
  PetscErrorCode     ierr;
  PC_BJacobi         *jac = (PC_BJacobi*)pc->data;
  KSP                subksp = jac->ksp[0];
  KSPConvergedReason reason;

  PetscFunctionBegin;
  ierr = KSPSetUp(subksp);CHKERRQ(ierr);
  ierr = KSPGetConvergedReason(subksp,&reason);CHKERRQ(ierr);
  if (reason == KSP_DIVERGED_PC_FAILED) {
    pc->failedreason = PC_SUBPC_ERROR;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApply_BJacobi_Singleblock(PC pc,Vec x,Vec y)
{
  PetscErrorCode         ierr;
  PC_BJacobi             *jac  = (PC_BJacobi*)pc->data;
  PC_BJacobi_Singleblock *bjac = (PC_BJacobi_Singleblock*)jac->data;

  PetscFunctionBegin;
  ierr = VecGetLocalVectorRead(x, bjac->x);CHKERRQ(ierr);
  ierr = VecGetLocalVector(y, bjac->y);CHKERRQ(ierr);
 /* Since the inner KSP matrix may point directly to the diagonal block of an MPI matrix the inner
     matrix may change even if the outer KSP/PC has not updated the preconditioner, this will trigger a rebuild
     of the inner preconditioner automatically unless we pass down the outer preconditioners reuse flag.*/
  ierr = KSPSetReusePreconditioner(jac->ksp[0],pc->reusepreconditioner);CHKERRQ(ierr);
  ierr = KSPSolve(jac->ksp[0],bjac->x,bjac->y);CHKERRQ(ierr);
  ierr = KSPCheckSolve(jac->ksp[0],pc,bjac->y);CHKERRQ(ierr);
  ierr = VecRestoreLocalVectorRead(x, bjac->x);CHKERRQ(ierr);
  ierr = VecRestoreLocalVector(y, bjac->y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCMatApply_BJacobi_Singleblock(PC pc,Mat X,Mat Y)
{
  PC_BJacobi     *jac  = (PC_BJacobi*)pc->data;
  Mat            sX,sY;
  PetscErrorCode ierr;

  PetscFunctionBegin;
 /* Since the inner KSP matrix may point directly to the diagonal block of an MPI matrix the inner
     matrix may change even if the outer KSP/PC has not updated the preconditioner, this will trigger a rebuild
     of the inner preconditioner automatically unless we pass down the outer preconditioners reuse flag.*/
  ierr = KSPSetReusePreconditioner(jac->ksp[0],pc->reusepreconditioner);CHKERRQ(ierr);
  ierr = MatDenseGetLocalMatrix(X,&sX);CHKERRQ(ierr);
  ierr = MatDenseGetLocalMatrix(Y,&sY);CHKERRQ(ierr);
  ierr = KSPMatSolve(jac->ksp[0],sX,sY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplySymmetricLeft_BJacobi_Singleblock(PC pc,Vec x,Vec y)
{
  PetscErrorCode         ierr;
  PC_BJacobi             *jac  = (PC_BJacobi*)pc->data;
  PC_BJacobi_Singleblock *bjac = (PC_BJacobi_Singleblock*)jac->data;
  PetscScalar            *y_array;
  const PetscScalar      *x_array;
  PC                     subpc;

  PetscFunctionBegin;
  /*
      The VecPlaceArray() is to avoid having to copy the
    y vector into the bjac->x vector. The reason for
    the bjac->x vector is that we need a sequential vector
    for the sequential solve.
  */
  ierr = VecGetArrayRead(x,&x_array);CHKERRQ(ierr);
  ierr = VecGetArray(y,&y_array);CHKERRQ(ierr);
  ierr = VecPlaceArray(bjac->x,x_array);CHKERRQ(ierr);
  ierr = VecPlaceArray(bjac->y,y_array);CHKERRQ(ierr);
  /* apply the symmetric left portion of the inner PC operator */
  /* note this by-passes the inner KSP and its options completely */
  ierr = KSPGetPC(jac->ksp[0],&subpc);CHKERRQ(ierr);
  ierr = PCApplySymmetricLeft(subpc,bjac->x,bjac->y);CHKERRQ(ierr);
  ierr = VecResetArray(bjac->x);CHKERRQ(ierr);
  ierr = VecResetArray(bjac->y);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(x,&x_array);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&y_array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplySymmetricRight_BJacobi_Singleblock(PC pc,Vec x,Vec y)
{
  PetscErrorCode         ierr;
  PC_BJacobi             *jac  = (PC_BJacobi*)pc->data;
  PC_BJacobi_Singleblock *bjac = (PC_BJacobi_Singleblock*)jac->data;
  PetscScalar            *y_array;
  const PetscScalar      *x_array;
  PC                     subpc;

  PetscFunctionBegin;
  /*
      The VecPlaceArray() is to avoid having to copy the
    y vector into the bjac->x vector. The reason for
    the bjac->x vector is that we need a sequential vector
    for the sequential solve.
  */
  ierr = VecGetArrayRead(x,&x_array);CHKERRQ(ierr);
  ierr = VecGetArray(y,&y_array);CHKERRQ(ierr);
  ierr = VecPlaceArray(bjac->x,x_array);CHKERRQ(ierr);
  ierr = VecPlaceArray(bjac->y,y_array);CHKERRQ(ierr);

  /* apply the symmetric right portion of the inner PC operator */
  /* note this by-passes the inner KSP and its options completely */

  ierr = KSPGetPC(jac->ksp[0],&subpc);CHKERRQ(ierr);
  ierr = PCApplySymmetricRight(subpc,bjac->x,bjac->y);CHKERRQ(ierr);

  ierr = VecRestoreArrayRead(x,&x_array);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&y_array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplyTranspose_BJacobi_Singleblock(PC pc,Vec x,Vec y)
{
  PetscErrorCode         ierr;
  PC_BJacobi             *jac  = (PC_BJacobi*)pc->data;
  PC_BJacobi_Singleblock *bjac = (PC_BJacobi_Singleblock*)jac->data;
  PetscScalar            *y_array;
  const PetscScalar      *x_array;

  PetscFunctionBegin;
  /*
      The VecPlaceArray() is to avoid having to copy the
    y vector into the bjac->x vector. The reason for
    the bjac->x vector is that we need a sequential vector
    for the sequential solve.
  */
  ierr = VecGetArrayRead(x,&x_array);CHKERRQ(ierr);
  ierr = VecGetArray(y,&y_array);CHKERRQ(ierr);
  ierr = VecPlaceArray(bjac->x,x_array);CHKERRQ(ierr);
  ierr = VecPlaceArray(bjac->y,y_array);CHKERRQ(ierr);
  ierr = KSPSolveTranspose(jac->ksp[0],bjac->x,bjac->y);CHKERRQ(ierr);
  ierr = KSPCheckSolve(jac->ksp[0],pc,bjac->y);CHKERRQ(ierr);
  ierr = VecResetArray(bjac->x);CHKERRQ(ierr);
  ierr = VecResetArray(bjac->y);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(x,&x_array);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&y_array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetUp_BJacobi_Singleblock(PC pc,Mat mat,Mat pmat)
{
  PC_BJacobi             *jac = (PC_BJacobi*)pc->data;
  PetscErrorCode         ierr;
  PetscInt               m;
  KSP                    ksp;
  PC_BJacobi_Singleblock *bjac;
  PetscBool              wasSetup = PETSC_TRUE;
  VecType                vectype;

  PetscFunctionBegin;
  if (!pc->setupcalled) {
    const char *prefix;

    if (!jac->ksp) {
      wasSetup = PETSC_FALSE;

      ierr = KSPCreate(PETSC_COMM_SELF,&ksp);CHKERRQ(ierr);
      ierr = KSPSetErrorIfNotConverged(ksp,pc->erroriffailure);CHKERRQ(ierr);
      ierr = PetscObjectIncrementTabLevel((PetscObject)ksp,(PetscObject)pc,1);CHKERRQ(ierr);
      ierr = PetscLogObjectParent((PetscObject)pc,(PetscObject)ksp);CHKERRQ(ierr);
      ierr = KSPSetType(ksp,KSPPREONLY);CHKERRQ(ierr);
      ierr = PCGetOptionsPrefix(pc,&prefix);CHKERRQ(ierr);
      ierr = KSPSetOptionsPrefix(ksp,prefix);CHKERRQ(ierr);
      ierr = KSPAppendOptionsPrefix(ksp,"sub_");CHKERRQ(ierr);

      pc->ops->reset               = PCReset_BJacobi_Singleblock;
      pc->ops->destroy             = PCDestroy_BJacobi_Singleblock;
      pc->ops->apply               = PCApply_BJacobi_Singleblock;
      pc->ops->matapply            = PCMatApply_BJacobi_Singleblock;
      pc->ops->applysymmetricleft  = PCApplySymmetricLeft_BJacobi_Singleblock;
      pc->ops->applysymmetricright = PCApplySymmetricRight_BJacobi_Singleblock;
      pc->ops->applytranspose      = PCApplyTranspose_BJacobi_Singleblock;
      pc->ops->setuponblocks       = PCSetUpOnBlocks_BJacobi_Singleblock;

      ierr        = PetscMalloc1(1,&jac->ksp);CHKERRQ(ierr);
      jac->ksp[0] = ksp;

      ierr      = PetscNewLog(pc,&bjac);CHKERRQ(ierr);
      jac->data = (void*)bjac;
    } else {
      ksp  = jac->ksp[0];
      bjac = (PC_BJacobi_Singleblock*)jac->data;
    }

    /*
      The reason we need to generate these vectors is to serve
      as the right-hand side and solution vector for the solve on the
      block. We do not need to allocate space for the vectors since
      that is provided via VecPlaceArray() just before the call to
      KSPSolve() on the block.
    */
    ierr = MatGetSize(pmat,&m,&m);CHKERRQ(ierr);
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,m,NULL,&bjac->x);CHKERRQ(ierr);
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,m,NULL,&bjac->y);CHKERRQ(ierr);
    ierr = MatGetVecType(pmat,&vectype);CHKERRQ(ierr);
    ierr = VecSetType(bjac->x,vectype);CHKERRQ(ierr);
    ierr = VecSetType(bjac->y,vectype);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)pc,(PetscObject)bjac->x);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)pc,(PetscObject)bjac->y);CHKERRQ(ierr);
  } else {
    ksp  = jac->ksp[0];
    bjac = (PC_BJacobi_Singleblock*)jac->data;
  }
  if (pc->useAmat) {
    ierr = KSPSetOperators(ksp,mat,pmat);CHKERRQ(ierr);
  } else {
    ierr = KSPSetOperators(ksp,pmat,pmat);CHKERRQ(ierr);
  }
  if (!wasSetup && pc->setfromoptionscalled) {
    /* If PCSetFromOptions_BJacobi is called later, KSPSetFromOptions will be called at that time. */
    ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------------------------*/
static PetscErrorCode PCReset_BJacobi_Multiblock(PC pc)
{
  PC_BJacobi            *jac  = (PC_BJacobi*)pc->data;
  PC_BJacobi_Multiblock *bjac = (PC_BJacobi_Multiblock*)jac->data;
  PetscErrorCode        ierr;
  PetscInt              i;

  PetscFunctionBegin;
  if (bjac && bjac->pmat) {
    ierr = MatDestroyMatrices(jac->n_local,&bjac->pmat);CHKERRQ(ierr);
    if (pc->useAmat) {
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

static PetscErrorCode PCDestroy_BJacobi_Multiblock(PC pc)
{
  PC_BJacobi            *jac  = (PC_BJacobi*)pc->data;
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

static PetscErrorCode PCSetUpOnBlocks_BJacobi_Multiblock(PC pc)
{
  PC_BJacobi         *jac = (PC_BJacobi*)pc->data;
  PetscErrorCode     ierr;
  PetscInt           i,n_local = jac->n_local;
  KSPConvergedReason reason;

  PetscFunctionBegin;
  for (i=0; i<n_local; i++) {
    ierr = KSPSetUp(jac->ksp[i]);CHKERRQ(ierr);
    ierr = KSPGetConvergedReason(jac->ksp[i],&reason);CHKERRQ(ierr);
    if (reason == KSP_DIVERGED_PC_FAILED) {
      pc->failedreason = PC_SUBPC_ERROR;
    }
  }
  PetscFunctionReturn(0);
}

/*
      Preconditioner for block Jacobi
*/
static PetscErrorCode PCApply_BJacobi_Multiblock(PC pc,Vec x,Vec y)
{
  PC_BJacobi            *jac = (PC_BJacobi*)pc->data;
  PetscErrorCode        ierr;
  PetscInt              i,n_local = jac->n_local;
  PC_BJacobi_Multiblock *bjac = (PC_BJacobi_Multiblock*)jac->data;
  PetscScalar           *yin;
  const PetscScalar     *xin;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(x,&xin);CHKERRQ(ierr);
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
    ierr = KSPCheckSolve(jac->ksp[i],pc,bjac->y[i]);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(PC_ApplyOnBlocks,jac->ksp[i],bjac->x[i],bjac->y[i],0);CHKERRQ(ierr);

    ierr = VecResetArray(bjac->x[i]);CHKERRQ(ierr);
    ierr = VecResetArray(bjac->y[i]);CHKERRQ(ierr);
  }
  ierr = VecRestoreArrayRead(x,&xin);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&yin);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
      Preconditioner for block Jacobi
*/
static PetscErrorCode PCApplyTranspose_BJacobi_Multiblock(PC pc,Vec x,Vec y)
{
  PC_BJacobi            *jac = (PC_BJacobi*)pc->data;
  PetscErrorCode        ierr;
  PetscInt              i,n_local = jac->n_local;
  PC_BJacobi_Multiblock *bjac = (PC_BJacobi_Multiblock*)jac->data;
  PetscScalar           *yin;
  const PetscScalar     *xin;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(x,&xin);CHKERRQ(ierr);
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
    ierr = KSPCheckSolve(jac->ksp[i],pc,bjac->y[i]);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(PC_ApplyTransposeOnBlocks,jac->ksp[i],bjac->x[i],bjac->y[i],0);CHKERRQ(ierr);

    ierr = VecResetArray(bjac->x[i]);CHKERRQ(ierr);
    ierr = VecResetArray(bjac->y[i]);CHKERRQ(ierr);
  }
  ierr = VecRestoreArrayRead(x,&xin);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&yin);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetUp_BJacobi_Multiblock(PC pc,Mat mat,Mat pmat)
{
  PC_BJacobi            *jac = (PC_BJacobi*)pc->data;
  PetscErrorCode        ierr;
  PetscInt              m,n_local,N,M,start,i;
  const char            *prefix,*pprefix,*mprefix;
  KSP                   ksp;
  Vec                   x,y;
  PC_BJacobi_Multiblock *bjac = (PC_BJacobi_Multiblock*)jac->data;
  PC                    subpc;
  IS                    is;
  MatReuse              scall;
  VecType               vectype;

  PetscFunctionBegin;
  ierr = MatGetLocalSize(pc->pmat,&M,&N);CHKERRQ(ierr);

  n_local = jac->n_local;

  if (pc->useAmat) {
    PetscBool same;
    ierr = PetscObjectTypeCompare((PetscObject)mat,((PetscObject)pmat)->type_name,&same);CHKERRQ(ierr);
    if (!same) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_INCOMP,"Matrices not of same type");
  }

  if (!pc->setupcalled) {
    scall = MAT_INITIAL_MATRIX;

    if (!jac->ksp) {
      pc->ops->reset         = PCReset_BJacobi_Multiblock;
      pc->ops->destroy       = PCDestroy_BJacobi_Multiblock;
      pc->ops->apply         = PCApply_BJacobi_Multiblock;
      pc->ops->matapply      = NULL;
      pc->ops->applytranspose= PCApplyTranspose_BJacobi_Multiblock;
      pc->ops->setuponblocks = PCSetUpOnBlocks_BJacobi_Multiblock;

      ierr = PetscNewLog(pc,&bjac);CHKERRQ(ierr);
      ierr = PetscMalloc1(n_local,&jac->ksp);CHKERRQ(ierr);
      ierr = PetscLogObjectMemory((PetscObject)pc,sizeof(n_local*sizeof(KSP)));CHKERRQ(ierr);
      ierr = PetscMalloc2(n_local,&bjac->x,n_local,&bjac->y);CHKERRQ(ierr);
      ierr = PetscMalloc1(n_local,&bjac->starts);CHKERRQ(ierr);
      ierr = PetscLogObjectMemory((PetscObject)pc,sizeof(n_local*sizeof(PetscScalar)));CHKERRQ(ierr);

      jac->data = (void*)bjac;
      ierr      = PetscMalloc1(n_local,&bjac->is);CHKERRQ(ierr);
      ierr      = PetscLogObjectMemory((PetscObject)pc,sizeof(n_local*sizeof(IS)));CHKERRQ(ierr);

      for (i=0; i<n_local; i++) {
        ierr = KSPCreate(PETSC_COMM_SELF,&ksp);CHKERRQ(ierr);
        ierr = KSPSetErrorIfNotConverged(ksp,pc->erroriffailure);CHKERRQ(ierr);
        ierr = PetscObjectIncrementTabLevel((PetscObject)ksp,(PetscObject)pc,1);CHKERRQ(ierr);
        ierr = PetscLogObjectParent((PetscObject)pc,(PetscObject)ksp);CHKERRQ(ierr);
        ierr = KSPSetType(ksp,KSPPREONLY);CHKERRQ(ierr);
        ierr = KSPGetPC(ksp,&subpc);CHKERRQ(ierr);
        ierr = PCGetOptionsPrefix(pc,&prefix);CHKERRQ(ierr);
        ierr = KSPSetOptionsPrefix(ksp,prefix);CHKERRQ(ierr);
        ierr = KSPAppendOptionsPrefix(ksp,"sub_");CHKERRQ(ierr);

        jac->ksp[i] = ksp;
      }
    } else {
      bjac = (PC_BJacobi_Multiblock*)jac->data;
    }

    start = 0;
    ierr = MatGetVecType(pmat,&vectype);CHKERRQ(ierr);
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
      ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,m,NULL,&y);CHKERRQ(ierr);
      ierr = VecSetType(x,vectype);CHKERRQ(ierr);
      ierr = VecSetType(y,vectype);CHKERRQ(ierr);
      ierr = PetscLogObjectParent((PetscObject)pc,(PetscObject)x);CHKERRQ(ierr);
      ierr = PetscLogObjectParent((PetscObject)pc,(PetscObject)y);CHKERRQ(ierr);

      bjac->x[i]      = x;
      bjac->y[i]      = y;
      bjac->starts[i] = start;

      ierr        = ISCreateStride(PETSC_COMM_SELF,m,start,1,&is);CHKERRQ(ierr);
      bjac->is[i] = is;
      ierr        = PetscLogObjectParent((PetscObject)pc,(PetscObject)is);CHKERRQ(ierr);

      start += m;
    }
  } else {
    bjac = (PC_BJacobi_Multiblock*)jac->data;
    /*
       Destroy the blocks from the previous iteration
    */
    if (pc->flag == DIFFERENT_NONZERO_PATTERN) {
      ierr = MatDestroyMatrices(n_local,&bjac->pmat);CHKERRQ(ierr);
      if (pc->useAmat) {
        ierr = MatDestroyMatrices(n_local,&bjac->mat);CHKERRQ(ierr);
      }
      scall = MAT_INITIAL_MATRIX;
    } else scall = MAT_REUSE_MATRIX;
  }

  ierr = MatCreateSubMatrices(pmat,n_local,bjac->is,bjac->is,scall,&bjac->pmat);CHKERRQ(ierr);
  if (pc->useAmat) {
    ierr = MatCreateSubMatrices(mat,n_local,bjac->is,bjac->is,scall,&bjac->mat);CHKERRQ(ierr);
  }
  /* Return control to the user so that the submatrices can be modified (e.g., to apply
     different boundary conditions for the submatrices than for the global problem) */
  ierr = PCModifySubMatrices(pc,n_local,bjac->is,bjac->is,bjac->pmat,pc->modifysubmatricesP);CHKERRQ(ierr);

  ierr = PetscObjectGetOptionsPrefix((PetscObject)pmat,&pprefix);CHKERRQ(ierr);
  for (i=0; i<n_local; i++) {
    ierr = PetscLogObjectParent((PetscObject)pc,(PetscObject)bjac->pmat[i]);CHKERRQ(ierr);
    ierr = PetscObjectSetOptionsPrefix((PetscObject)bjac->pmat[i],pprefix);CHKERRQ(ierr);
    if (pc->useAmat) {
      ierr = PetscLogObjectParent((PetscObject)pc,(PetscObject)bjac->mat[i]);CHKERRQ(ierr);
      ierr = PetscObjectGetOptionsPrefix((PetscObject)mat,&mprefix);CHKERRQ(ierr);
      ierr = PetscObjectSetOptionsPrefix((PetscObject)bjac->mat[i],mprefix);CHKERRQ(ierr);
      ierr = KSPSetOperators(jac->ksp[i],bjac->mat[i],bjac->pmat[i]);CHKERRQ(ierr);
    } else {
      ierr = KSPSetOperators(jac->ksp[i],bjac->pmat[i],bjac->pmat[i]);CHKERRQ(ierr);
    }
    if (pc->setfromoptionscalled) {
      ierr = KSPSetFromOptions(jac->ksp[i]);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------------------------*/
/*
      These are for a single block with multiple processes;
*/
static PetscErrorCode PCReset_BJacobi_Multiproc(PC pc)
{
  PC_BJacobi           *jac   = (PC_BJacobi*)pc->data;
  PC_BJacobi_Multiproc *mpjac = (PC_BJacobi_Multiproc*)jac->data;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  ierr = VecDestroy(&mpjac->ysub);CHKERRQ(ierr);
  ierr = VecDestroy(&mpjac->xsub);CHKERRQ(ierr);
  ierr = MatDestroy(&mpjac->submats);CHKERRQ(ierr);
  if (jac->ksp) {ierr = KSPReset(jac->ksp[0]);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDestroy_BJacobi_Multiproc(PC pc)
{
  PC_BJacobi           *jac   = (PC_BJacobi*)pc->data;
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

static PetscErrorCode PCApply_BJacobi_Multiproc(PC pc,Vec x,Vec y)
{
  PC_BJacobi           *jac   = (PC_BJacobi*)pc->data;
  PC_BJacobi_Multiproc *mpjac = (PC_BJacobi_Multiproc*)jac->data;
  PetscErrorCode       ierr;
  PetscScalar          *yarray;
  const PetscScalar    *xarray;
  KSPConvergedReason   reason;

  PetscFunctionBegin;
  /* place x's and y's local arrays into xsub and ysub */
  ierr = VecGetArrayRead(x,&xarray);CHKERRQ(ierr);
  ierr = VecGetArray(y,&yarray);CHKERRQ(ierr);
  ierr = VecPlaceArray(mpjac->xsub,xarray);CHKERRQ(ierr);
  ierr = VecPlaceArray(mpjac->ysub,yarray);CHKERRQ(ierr);

  /* apply preconditioner on each matrix block */
  ierr = PetscLogEventBegin(PC_ApplyOnBlocks,jac->ksp[0],mpjac->xsub,mpjac->ysub,0);CHKERRQ(ierr);
  ierr = KSPSolve(jac->ksp[0],mpjac->xsub,mpjac->ysub);CHKERRQ(ierr);
  ierr = KSPCheckSolve(jac->ksp[0],pc,mpjac->ysub);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(PC_ApplyOnBlocks,jac->ksp[0],mpjac->xsub,mpjac->ysub,0);CHKERRQ(ierr);
  ierr = KSPGetConvergedReason(jac->ksp[0],&reason);CHKERRQ(ierr);
  if (reason == KSP_DIVERGED_PC_FAILED) {
    pc->failedreason = PC_SUBPC_ERROR;
  }

  ierr = VecResetArray(mpjac->xsub);CHKERRQ(ierr);
  ierr = VecResetArray(mpjac->ysub);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(x,&xarray);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&yarray);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCMatApply_BJacobi_Multiproc(PC pc,Mat X,Mat Y)
{
  PC_BJacobi           *jac   = (PC_BJacobi*)pc->data;
  KSPConvergedReason   reason;
  Mat                  sX,sY;
  const PetscScalar    *x;
  PetscScalar          *y;
  PetscInt             m,N,lda,ldb;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  /* apply preconditioner on each matrix block */
  ierr = MatGetLocalSize(X,&m,NULL);CHKERRQ(ierr);
  ierr = MatGetSize(X,NULL,&N);CHKERRQ(ierr);
  ierr = MatDenseGetLDA(X,&lda);CHKERRQ(ierr);
  ierr = MatDenseGetLDA(Y,&ldb);CHKERRQ(ierr);
  ierr = MatDenseGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = MatDenseGetArrayWrite(Y,&y);CHKERRQ(ierr);
  ierr = MatCreateDense(PetscObjectComm((PetscObject)jac->ksp[0]),m,PETSC_DECIDE,PETSC_DECIDE,N,(PetscScalar*)x,&sX);CHKERRQ(ierr);
  ierr = MatCreateDense(PetscObjectComm((PetscObject)jac->ksp[0]),m,PETSC_DECIDE,PETSC_DECIDE,N,y,&sY);CHKERRQ(ierr);
  ierr = MatDenseSetLDA(sX,lda);CHKERRQ(ierr);
  ierr = MatDenseSetLDA(sY,ldb);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(PC_ApplyOnBlocks,jac->ksp[0],X,Y,0);CHKERRQ(ierr);
  ierr = KSPMatSolve(jac->ksp[0],sX,sY);CHKERRQ(ierr);
  ierr = KSPCheckSolve(jac->ksp[0],pc,NULL);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(PC_ApplyOnBlocks,jac->ksp[0],X,Y,0);CHKERRQ(ierr);
  ierr = MatDestroy(&sY);CHKERRQ(ierr);
  ierr = MatDestroy(&sX);CHKERRQ(ierr);
  ierr = MatDenseRestoreArrayWrite(Y,&y);CHKERRQ(ierr);
  ierr = MatDenseRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = KSPGetConvergedReason(jac->ksp[0],&reason);CHKERRQ(ierr);
  if (reason == KSP_DIVERGED_PC_FAILED) {
    pc->failedreason = PC_SUBPC_ERROR;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetUp_BJacobi_Multiproc(PC pc)
{
  PC_BJacobi           *jac   = (PC_BJacobi*)pc->data;
  PC_BJacobi_Multiproc *mpjac = (PC_BJacobi_Multiproc*)jac->data;
  PetscErrorCode       ierr;
  PetscInt             m,n;
  MPI_Comm             comm,subcomm=0;
  const char           *prefix;
  PetscBool            wasSetup = PETSC_TRUE;
  VecType              vectype;


  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)pc,&comm);CHKERRQ(ierr);
  if (jac->n_local > 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Only a single block in a subcommunicator is supported");
  jac->n_local = 1; /* currently only a single block is supported for a subcommunicator */
  if (!pc->setupcalled) {
    wasSetup  = PETSC_FALSE;
    ierr      = PetscNewLog(pc,&mpjac);CHKERRQ(ierr);
    jac->data = (void*)mpjac;

    /* initialize datastructure mpjac */
    if (!jac->psubcomm) {
      /* Create default contiguous subcommunicatiors if user does not provide them */
      ierr = PetscSubcommCreate(comm,&jac->psubcomm);CHKERRQ(ierr);
      ierr = PetscSubcommSetNumber(jac->psubcomm,jac->n);CHKERRQ(ierr);
      ierr = PetscSubcommSetType(jac->psubcomm,PETSC_SUBCOMM_CONTIGUOUS);CHKERRQ(ierr);
      ierr = PetscLogObjectMemory((PetscObject)pc,sizeof(PetscSubcomm));CHKERRQ(ierr);
    }
    mpjac->psubcomm = jac->psubcomm;
    subcomm         = PetscSubcommChild(mpjac->psubcomm);

    /* Get matrix blocks of pmat */
    ierr = MatGetMultiProcBlock(pc->pmat,subcomm,MAT_INITIAL_MATRIX,&mpjac->submats);CHKERRQ(ierr);

    /* create a new PC that processors in each subcomm have copy of */
    ierr = PetscMalloc1(1,&jac->ksp);CHKERRQ(ierr);
    ierr = KSPCreate(subcomm,&jac->ksp[0]);CHKERRQ(ierr);
    ierr = KSPSetErrorIfNotConverged(jac->ksp[0],pc->erroriffailure);CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)jac->ksp[0],(PetscObject)pc,1);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)pc,(PetscObject)jac->ksp[0]);CHKERRQ(ierr);
    ierr = KSPSetOperators(jac->ksp[0],mpjac->submats,mpjac->submats);CHKERRQ(ierr);
    ierr = KSPGetPC(jac->ksp[0],&mpjac->pc);CHKERRQ(ierr);

    ierr = PCGetOptionsPrefix(pc,&prefix);CHKERRQ(ierr);
    ierr = KSPSetOptionsPrefix(jac->ksp[0],prefix);CHKERRQ(ierr);
    ierr = KSPAppendOptionsPrefix(jac->ksp[0],"sub_");CHKERRQ(ierr);

    /* create dummy vectors xsub and ysub */
    ierr = MatGetLocalSize(mpjac->submats,&m,&n);CHKERRQ(ierr);
    ierr = VecCreateMPIWithArray(subcomm,1,n,PETSC_DECIDE,NULL,&mpjac->xsub);CHKERRQ(ierr);
    ierr = VecCreateMPIWithArray(subcomm,1,m,PETSC_DECIDE,NULL,&mpjac->ysub);CHKERRQ(ierr);
    ierr = MatGetVecType(mpjac->submats,&vectype);CHKERRQ(ierr);
    ierr = VecSetType(mpjac->xsub,vectype);CHKERRQ(ierr);
    ierr = VecSetType(mpjac->ysub,vectype);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)pc,(PetscObject)mpjac->xsub);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)pc,(PetscObject)mpjac->ysub);CHKERRQ(ierr);

    pc->ops->reset   = PCReset_BJacobi_Multiproc;
    pc->ops->destroy = PCDestroy_BJacobi_Multiproc;
    pc->ops->apply   = PCApply_BJacobi_Multiproc;
    pc->ops->matapply= PCMatApply_BJacobi_Multiproc;
  } else { /* pc->setupcalled */
    subcomm = PetscSubcommChild(mpjac->psubcomm);
    if (pc->flag == DIFFERENT_NONZERO_PATTERN) {
      /* destroy old matrix blocks, then get new matrix blocks */
      if (mpjac->submats) {ierr = MatDestroy(&mpjac->submats);CHKERRQ(ierr);}
      ierr = MatGetMultiProcBlock(pc->pmat,subcomm,MAT_INITIAL_MATRIX,&mpjac->submats);CHKERRQ(ierr);
    } else {
      ierr = MatGetMultiProcBlock(pc->pmat,subcomm,MAT_REUSE_MATRIX,&mpjac->submats);CHKERRQ(ierr);
    }
    ierr = KSPSetOperators(jac->ksp[0],mpjac->submats,mpjac->submats);CHKERRQ(ierr);
  }

  if (!wasSetup && pc->setfromoptionscalled) {
    ierr = KSPSetFromOptions(jac->ksp[0]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
