
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
  PetscBool      hasop;
  PetscInt       N,M,start,i,sum,end;
  PetscInt       bs,i_start=-1,i_end=-1;
  PetscMPIInt    rank,size;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)pc),&rank));
  CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)pc),&size));
  CHKERRQ(MatGetLocalSize(pc->pmat,&M,&N));
  CHKERRQ(MatGetBlockSize(pc->pmat,&bs));

  if (jac->n > 0 && jac->n < size) {
    CHKERRQ(PCSetUp_BJacobi_Multiproc(pc));
    PetscFunctionReturn(0);
  }

  /* --------------------------------------------------------------------------
      Determines the number of blocks assigned to each processor
  -----------------------------------------------------------------------------*/

  /*   local block count  given */
  if (jac->n_local > 0 && jac->n < 0) {
    CHKERRMPI(MPIU_Allreduce(&jac->n_local,&jac->n,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)pc)));
    if (jac->l_lens) { /* check that user set these correctly */
      sum = 0;
      for (i=0; i<jac->n_local; i++) {
        PetscCheckFalse(jac->l_lens[i]/bs*bs !=jac->l_lens[i],PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Mat blocksize doesn't match block Jacobi layout");
        sum += jac->l_lens[i];
      }
      PetscCheckFalse(sum != M,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local lens set incorrectly");
    } else {
      CHKERRQ(PetscMalloc1(jac->n_local,&jac->l_lens));
      for (i=0; i<jac->n_local; i++) jac->l_lens[i] = bs*((M/bs)/jac->n_local + (((M/bs) % jac->n_local) > i));
    }
  } else if (jac->n > 0 && jac->n_local < 0) { /* global block count given */
    /* global blocks given: determine which ones are local */
    if (jac->g_lens) {
      /* check if the g_lens is has valid entries */
      for (i=0; i<jac->n; i++) {
        PetscCheckFalse(!jac->g_lens[i],PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Zero block not allowed");
        PetscCheckFalse(jac->g_lens[i]/bs*bs != jac->g_lens[i],PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Mat blocksize doesn't match block Jacobi layout");
      }
      if (size == 1) {
        jac->n_local = jac->n;
        CHKERRQ(PetscMalloc1(jac->n_local,&jac->l_lens));
        CHKERRQ(PetscArraycpy(jac->l_lens,jac->g_lens,jac->n_local));
        /* check that user set these correctly */
        sum = 0;
        for (i=0; i<jac->n_local; i++) sum += jac->l_lens[i];
        PetscCheckFalse(sum != M,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Global lens set incorrectly");
      } else {
        CHKERRQ(MatGetOwnershipRange(pc->pmat,&start,&end));
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
        CHKERRQ(PetscMalloc1(jac->n_local,&jac->l_lens));
        CHKERRQ(PetscArraycpy(jac->l_lens,jac->g_lens+i_start,jac->n_local));
      }
    } else { /* no global blocks given, determine then using default layout */
      jac->n_local = jac->n/size + ((jac->n % size) > rank);
      CHKERRQ(PetscMalloc1(jac->n_local,&jac->l_lens));
      for (i=0; i<jac->n_local; i++) {
        jac->l_lens[i] = ((M/bs)/jac->n_local + (((M/bs) % jac->n_local) > i))*bs;
        PetscCheckFalse(!jac->l_lens[i],PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Too many blocks given");
      }
    }
  } else if (jac->n < 0 && jac->n_local < 0) { /* no blocks given */
    jac->n         = size;
    jac->n_local   = 1;
    CHKERRQ(PetscMalloc1(1,&jac->l_lens));
    jac->l_lens[0] = M;
  } else { /* jac->n > 0 && jac->n_local > 0 */
    if (!jac->l_lens) {
      CHKERRQ(PetscMalloc1(jac->n_local,&jac->l_lens));
      for (i=0; i<jac->n_local; i++) jac->l_lens[i] = bs*((M/bs)/jac->n_local + (((M/bs) % jac->n_local) > i));
    }
  }
  PetscCheckFalse(jac->n_local < 1,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Number of blocks is less than number of processors");

  /* -------------------------
      Determines mat and pmat
  ---------------------------*/
  CHKERRQ(MatHasOperation(pc->mat,MATOP_GET_DIAGONAL_BLOCK,&hasop));
  if (!hasop && size == 1) {
    mat  = pc->mat;
    pmat = pc->pmat;
  } else {
    if (pc->useAmat) {
      /* use block from Amat matrix, not Pmat for local MatMult() */
      CHKERRQ(MatGetDiagonalBlock(pc->mat,&mat));
    }
    if (pc->pmat != pc->mat || !pc->useAmat) {
      CHKERRQ(MatGetDiagonalBlock(pc->pmat,&pmat));
    } else pmat = mat;
  }

  /* ------
     Setup code depends on the number of blocks
  */
  if (jac->n_local == 1) {
    CHKERRQ(PCSetUp_BJacobi_Singleblock(pc,mat,pmat));
  } else {
    CHKERRQ(PCSetUp_BJacobi_Multiblock(pc,mat,pmat));
  }
  PetscFunctionReturn(0);
}

/* Default destroy, if it has never been setup */
static PetscErrorCode PCDestroy_BJacobi(PC pc)
{
  PC_BJacobi     *jac = (PC_BJacobi*)pc->data;

  PetscFunctionBegin;
  CHKERRQ(PetscFree(jac->g_lens));
  CHKERRQ(PetscFree(jac->l_lens));
  CHKERRQ(PetscFree(pc->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetFromOptions_BJacobi(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PC_BJacobi     *jac = (PC_BJacobi*)pc->data;
  PetscInt       blocks,i;
  PetscBool      flg;

  PetscFunctionBegin;
  CHKERRQ(PetscOptionsHead(PetscOptionsObject,"Block Jacobi options"));
  CHKERRQ(PetscOptionsInt("-pc_bjacobi_blocks","Total number of blocks","PCBJacobiSetTotalBlocks",jac->n,&blocks,&flg));
  if (flg) CHKERRQ(PCBJacobiSetTotalBlocks(pc,blocks,NULL));
  CHKERRQ(PetscOptionsInt("-pc_bjacobi_local_blocks","Local number of blocks","PCBJacobiSetLocalBlocks",jac->n_local,&blocks,&flg));
  if (flg) CHKERRQ(PCBJacobiSetLocalBlocks(pc,blocks,NULL));
  if (jac->ksp) {
    /* The sub-KSP has already been set up (e.g., PCSetUp_BJacobi_Singleblock), but KSPSetFromOptions was not called
     * unless we had already been called. */
    for (i=0; i<jac->n_local; i++) {
      CHKERRQ(KSPSetFromOptions(jac->ksp[i]));
    }
  }
  CHKERRQ(PetscOptionsTail());
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
  PetscViewerFormat    format;
  const char           *prefix;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERSTRING,&isstring));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERDRAW,&isdraw));
  if (iascii) {
    if (pc->useAmat) {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"  using Amat local matrix, number of blocks = %D\n",jac->n));
    }
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  number of blocks = %D\n",jac->n));
    CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)pc),&rank));
    CHKERRQ(PetscViewerGetFormat(viewer,&format));
    if (format != PETSC_VIEWER_ASCII_INFO_DETAIL) {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Local solver information for first block is in the following KSP and PC objects on rank 0:\n"));
      CHKERRQ(PCGetOptionsPrefix(pc,&prefix));
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Use -%sksp_view ::ascii_info_detail to display information for all blocks\n",prefix?prefix:""));
      if (jac->ksp && !jac->psubcomm) {
        CHKERRQ(PetscViewerGetSubViewer(viewer,PETSC_COMM_SELF,&sviewer));
        if (rank == 0) {
          CHKERRQ(PetscViewerASCIIPushTab(viewer));
          CHKERRQ(KSPView(jac->ksp[0],sviewer));
          CHKERRQ(PetscViewerASCIIPopTab(viewer));
        }
        CHKERRQ(PetscViewerFlush(sviewer));
        CHKERRQ(PetscViewerRestoreSubViewer(viewer,PETSC_COMM_SELF,&sviewer));
        CHKERRQ(PetscViewerFlush(viewer));
        /*  extra call needed because of the two calls to PetscViewerASCIIPushSynchronized() in PetscViewerGetSubViewer() */
        CHKERRQ(PetscViewerASCIIPopSynchronized(viewer));
      } else if (mpjac && jac->ksp && mpjac->psubcomm) {
        CHKERRQ(PetscViewerGetSubViewer(viewer,mpjac->psubcomm->child,&sviewer));
        if (!mpjac->psubcomm->color) {
          CHKERRQ(PetscViewerASCIIPushTab(viewer));
          CHKERRQ(KSPView(*(jac->ksp),sviewer));
          CHKERRQ(PetscViewerASCIIPopTab(viewer));
        }
        CHKERRQ(PetscViewerFlush(sviewer));
        CHKERRQ(PetscViewerRestoreSubViewer(viewer,mpjac->psubcomm->child,&sviewer));
        CHKERRQ(PetscViewerFlush(viewer));
        /*  extra call needed because of the two calls to PetscViewerASCIIPushSynchronized() in PetscViewerGetSubViewer() */
        CHKERRQ(PetscViewerASCIIPopSynchronized(viewer));
      } else {
        CHKERRQ(PetscViewerFlush(viewer));
      }
    } else {
      PetscInt n_global;
      CHKERRMPI(MPIU_Allreduce(&jac->n_local,&n_global,1,MPIU_INT,MPI_MAX,PetscObjectComm((PetscObject)pc)));
      CHKERRQ(PetscViewerASCIIPushSynchronized(viewer));
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Local solver information for each block is in the following KSP and PC objects:\n"));
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] number of local blocks = %D, first local block number = %D\n",
                                                rank,jac->n_local,jac->first_local);CHKERRQ(ierr);
      CHKERRQ(PetscViewerASCIIPushTab(viewer));
      CHKERRQ(PetscViewerGetSubViewer(viewer,PETSC_COMM_SELF,&sviewer));
      for (i=0; i<jac->n_local; i++) {
        CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer,"[%d] local block number %D\n",rank,i));
        CHKERRQ(KSPView(jac->ksp[i],sviewer));
        CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer,"- - - - - - - - - - - - - - - - - -\n"));
      }
      CHKERRQ(PetscViewerRestoreSubViewer(viewer,PETSC_COMM_SELF,&sviewer));
      CHKERRQ(PetscViewerASCIIPopTab(viewer));
      CHKERRQ(PetscViewerFlush(viewer));
      CHKERRQ(PetscViewerASCIIPopSynchronized(viewer));
    }
  } else if (isstring) {
    CHKERRQ(PetscViewerStringSPrintf(viewer," blks=%D",jac->n));
    CHKERRQ(PetscViewerGetSubViewer(viewer,PETSC_COMM_SELF,&sviewer));
    if (jac->ksp) CHKERRQ(KSPView(jac->ksp[0],sviewer));
    CHKERRQ(PetscViewerRestoreSubViewer(viewer,PETSC_COMM_SELF,&sviewer));
  } else if (isdraw) {
    PetscDraw draw;
    char      str[25];
    PetscReal x,y,bottom,h;

    CHKERRQ(PetscViewerDrawGetDraw(viewer,0,&draw));
    CHKERRQ(PetscDrawGetCurrentPoint(draw,&x,&y));
    CHKERRQ(PetscSNPrintf(str,25,"Number blocks %D",jac->n));
    CHKERRQ(PetscDrawStringBoxed(draw,x,y,PETSC_DRAW_RED,PETSC_DRAW_BLACK,str,NULL,&h));
    bottom = y - h;
    CHKERRQ(PetscDrawPushCurrentPoint(draw,x,bottom));
    /* warning the communicator on viewer is different then on ksp in parallel */
    if (jac->ksp) CHKERRQ(KSPView(jac->ksp[0],viewer));
    CHKERRQ(PetscDrawPopCurrentPoint(draw));
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------------*/

static PetscErrorCode  PCBJacobiGetSubKSP_BJacobi(PC pc,PetscInt *n_local,PetscInt *first_local,KSP **ksp)
{
  PC_BJacobi *jac = (PC_BJacobi*)pc->data;

  PetscFunctionBegin;
  PetscCheck(pc->setupcalled,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"Must call KSPSetUp() or PCSetUp() first");

  if (n_local) *n_local = jac->n_local;
  if (first_local) *first_local = jac->first_local;
  if (ksp) *ksp                 = jac->ksp;
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCBJacobiSetTotalBlocks_BJacobi(PC pc,PetscInt blocks,PetscInt *lens)
{
  PC_BJacobi     *jac = (PC_BJacobi*)pc->data;

  PetscFunctionBegin;
  PetscCheckFalse(pc->setupcalled > 0 && jac->n!=blocks,PetscObjectComm((PetscObject)pc),PETSC_ERR_ORDER,"Cannot alter number of blocks after PCSetUp()/KSPSetUp() has been called");
  jac->n = blocks;
  if (!lens) jac->g_lens = NULL;
  else {
    CHKERRQ(PetscMalloc1(blocks,&jac->g_lens));
    CHKERRQ(PetscLogObjectMemory((PetscObject)pc,blocks*sizeof(PetscInt)));
    CHKERRQ(PetscArraycpy(jac->g_lens,lens,blocks));
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

  PetscFunctionBegin;
  jac = (PC_BJacobi*)pc->data;

  jac->n_local = blocks;
  if (!lens) jac->l_lens = NULL;
  else {
    CHKERRQ(PetscMalloc1(blocks,&jac->l_lens));
    CHKERRQ(PetscLogObjectMemory((PetscObject)pc,blocks*sizeof(PetscInt)));
    CHKERRQ(PetscArraycpy(jac->l_lens,lens,blocks));
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

.seealso: PCASMGetSubKSP()
@*/
PetscErrorCode  PCBJacobiGetSubKSP(PC pc,PetscInt *n_local,PetscInt *first_local,KSP *ksp[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  CHKERRQ(PetscUseMethod(pc,"PCBJacobiGetSubKSP_C",(PC,PetscInt*,PetscInt*,KSP **),(pc,n_local,first_local,ksp)));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscCheckFalse(blocks <= 0,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_OUTOFRANGE,"Must have positive blocks");
  CHKERRQ(PetscTryMethod(pc,"PCBJacobiSetTotalBlocks_C",(PC,PetscInt,const PetscInt[]),(pc,blocks,lens)));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID,1);
  PetscValidIntPointer(blocks,2);
  CHKERRQ(PetscUseMethod(pc,"PCBJacobiGetTotalBlocks_C",(PC,PetscInt*, const PetscInt *[]),(pc,blocks,lens)));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscCheckFalse(blocks < 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Must have nonegative blocks");
  CHKERRQ(PetscTryMethod(pc,"PCBJacobiSetLocalBlocks_C",(PC,PetscInt,const PetscInt []),(pc,blocks,lens)));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID,1);
  PetscValidIntPointer(blocks,2);
  CHKERRQ(PetscUseMethod(pc,"PCBJacobiGetLocalBlocks_C",(PC,PetscInt*, const PetscInt *[]),(pc,blocks,lens)));
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

     See PCJACOBI for point Jacobi preconditioning, PCVPBJACOBI for variable size point block Jacobi and PCPBJACOBI for large blocks

   Level: beginner

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC,
           PCASM, PCSetUseAmat(), PCGetUseAmat(), PCBJacobiGetSubKSP(), PCBJacobiSetTotalBlocks(),
           PCBJacobiSetLocalBlocks(), PCSetModifySubMatrices(), PCJACOBI, PCVPBJACOBI, PCPBJACOBI
M*/

PETSC_EXTERN PetscErrorCode PCCreate_BJacobi(PC pc)
{
  PetscMPIInt    rank;
  PC_BJacobi     *jac;

  PetscFunctionBegin;
  CHKERRQ(PetscNewLog(pc,&jac));
  CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)pc),&rank));

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
  jac->g_lens            = NULL;
  jac->l_lens            = NULL;
  jac->psubcomm          = NULL;

  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCBJacobiGetSubKSP_C",PCBJacobiGetSubKSP_BJacobi));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCBJacobiSetTotalBlocks_C",PCBJacobiSetTotalBlocks_BJacobi));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCBJacobiGetTotalBlocks_C",PCBJacobiGetTotalBlocks_BJacobi));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCBJacobiSetLocalBlocks_C",PCBJacobiSetLocalBlocks_BJacobi));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCBJacobiGetLocalBlocks_C",PCBJacobiGetLocalBlocks_BJacobi));
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

  PetscFunctionBegin;
  CHKERRQ(KSPReset(jac->ksp[0]));
  CHKERRQ(VecDestroy(&bjac->x));
  CHKERRQ(VecDestroy(&bjac->y));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDestroy_BJacobi_Singleblock(PC pc)
{
  PC_BJacobi             *jac  = (PC_BJacobi*)pc->data;
  PC_BJacobi_Singleblock *bjac = (PC_BJacobi_Singleblock*)jac->data;

  PetscFunctionBegin;
  CHKERRQ(PCReset_BJacobi_Singleblock(pc));
  CHKERRQ(KSPDestroy(&jac->ksp[0]));
  CHKERRQ(PetscFree(jac->ksp));
  CHKERRQ(PetscFree(jac->l_lens));
  CHKERRQ(PetscFree(jac->g_lens));
  CHKERRQ(PetscFree(bjac));
  CHKERRQ(PetscFree(pc->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetUpOnBlocks_BJacobi_Singleblock(PC pc)
{
  PC_BJacobi         *jac = (PC_BJacobi*)pc->data;
  KSP                subksp = jac->ksp[0];
  KSPConvergedReason reason;

  PetscFunctionBegin;
  CHKERRQ(KSPSetUp(subksp));
  CHKERRQ(KSPGetConvergedReason(subksp,&reason));
  if (reason == KSP_DIVERGED_PC_FAILED) {
    pc->failedreason = PC_SUBPC_ERROR;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApply_BJacobi_Singleblock(PC pc,Vec x,Vec y)
{
  PC_BJacobi             *jac  = (PC_BJacobi*)pc->data;
  PC_BJacobi_Singleblock *bjac = (PC_BJacobi_Singleblock*)jac->data;

  PetscFunctionBegin;
  CHKERRQ(VecGetLocalVectorRead(x, bjac->x));
  CHKERRQ(VecGetLocalVector(y, bjac->y));
  /* Since the inner KSP matrix may point directly to the diagonal block of an MPI matrix the inner
     matrix may change even if the outer KSP/PC has not updated the preconditioner, this will trigger a rebuild
     of the inner preconditioner automatically unless we pass down the outer preconditioners reuse flag.*/
  CHKERRQ(KSPSetReusePreconditioner(jac->ksp[0],pc->reusepreconditioner));
  CHKERRQ(KSPSolve(jac->ksp[0],bjac->x,bjac->y));
  CHKERRQ(KSPCheckSolve(jac->ksp[0],pc,bjac->y));
  CHKERRQ(VecRestoreLocalVectorRead(x, bjac->x));
  CHKERRQ(VecRestoreLocalVector(y, bjac->y));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCMatApply_BJacobi_Singleblock(PC pc,Mat X,Mat Y)
{
  PC_BJacobi     *jac  = (PC_BJacobi*)pc->data;
  Mat            sX,sY;

  PetscFunctionBegin;
  /* Since the inner KSP matrix may point directly to the diagonal block of an MPI matrix the inner
     matrix may change even if the outer KSP/PC has not updated the preconditioner, this will trigger a rebuild
     of the inner preconditioner automatically unless we pass down the outer preconditioners reuse flag.*/
  CHKERRQ(KSPSetReusePreconditioner(jac->ksp[0],pc->reusepreconditioner));
  CHKERRQ(MatDenseGetLocalMatrix(X,&sX));
  CHKERRQ(MatDenseGetLocalMatrix(Y,&sY));
  CHKERRQ(KSPMatSolve(jac->ksp[0],sX,sY));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplySymmetricLeft_BJacobi_Singleblock(PC pc,Vec x,Vec y)
{
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
  CHKERRQ(VecGetArrayRead(x,&x_array));
  CHKERRQ(VecGetArray(y,&y_array));
  CHKERRQ(VecPlaceArray(bjac->x,x_array));
  CHKERRQ(VecPlaceArray(bjac->y,y_array));
  /* apply the symmetric left portion of the inner PC operator */
  /* note this by-passes the inner KSP and its options completely */
  CHKERRQ(KSPGetPC(jac->ksp[0],&subpc));
  CHKERRQ(PCApplySymmetricLeft(subpc,bjac->x,bjac->y));
  CHKERRQ(VecResetArray(bjac->x));
  CHKERRQ(VecResetArray(bjac->y));
  CHKERRQ(VecRestoreArrayRead(x,&x_array));
  CHKERRQ(VecRestoreArray(y,&y_array));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplySymmetricRight_BJacobi_Singleblock(PC pc,Vec x,Vec y)
{
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
  CHKERRQ(VecGetArrayRead(x,&x_array));
  CHKERRQ(VecGetArray(y,&y_array));
  CHKERRQ(VecPlaceArray(bjac->x,x_array));
  CHKERRQ(VecPlaceArray(bjac->y,y_array));

  /* apply the symmetric right portion of the inner PC operator */
  /* note this by-passes the inner KSP and its options completely */

  CHKERRQ(KSPGetPC(jac->ksp[0],&subpc));
  CHKERRQ(PCApplySymmetricRight(subpc,bjac->x,bjac->y));

  CHKERRQ(VecRestoreArrayRead(x,&x_array));
  CHKERRQ(VecRestoreArray(y,&y_array));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplyTranspose_BJacobi_Singleblock(PC pc,Vec x,Vec y)
{
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
  CHKERRQ(VecGetArrayRead(x,&x_array));
  CHKERRQ(VecGetArray(y,&y_array));
  CHKERRQ(VecPlaceArray(bjac->x,x_array));
  CHKERRQ(VecPlaceArray(bjac->y,y_array));
  CHKERRQ(KSPSolveTranspose(jac->ksp[0],bjac->x,bjac->y));
  CHKERRQ(KSPCheckSolve(jac->ksp[0],pc,bjac->y));
  CHKERRQ(VecResetArray(bjac->x));
  CHKERRQ(VecResetArray(bjac->y));
  CHKERRQ(VecRestoreArrayRead(x,&x_array));
  CHKERRQ(VecRestoreArray(y,&y_array));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetUp_BJacobi_Singleblock(PC pc,Mat mat,Mat pmat)
{
  PC_BJacobi             *jac = (PC_BJacobi*)pc->data;
  PetscInt               m;
  KSP                    ksp;
  PC_BJacobi_Singleblock *bjac;
  PetscBool              wasSetup = PETSC_TRUE;
  VecType                vectype;
  const char             *prefix;

  PetscFunctionBegin;
  if (!pc->setupcalled) {
    if (!jac->ksp) {
      wasSetup = PETSC_FALSE;

      CHKERRQ(KSPCreate(PETSC_COMM_SELF,&ksp));
      CHKERRQ(KSPSetErrorIfNotConverged(ksp,pc->erroriffailure));
      CHKERRQ(PetscObjectIncrementTabLevel((PetscObject)ksp,(PetscObject)pc,1));
      CHKERRQ(PetscLogObjectParent((PetscObject)pc,(PetscObject)ksp));
      CHKERRQ(KSPSetType(ksp,KSPPREONLY));
      CHKERRQ(PCGetOptionsPrefix(pc,&prefix));
      CHKERRQ(KSPSetOptionsPrefix(ksp,prefix));
      CHKERRQ(KSPAppendOptionsPrefix(ksp,"sub_"));

      pc->ops->reset               = PCReset_BJacobi_Singleblock;
      pc->ops->destroy             = PCDestroy_BJacobi_Singleblock;
      pc->ops->apply               = PCApply_BJacobi_Singleblock;
      pc->ops->matapply            = PCMatApply_BJacobi_Singleblock;
      pc->ops->applysymmetricleft  = PCApplySymmetricLeft_BJacobi_Singleblock;
      pc->ops->applysymmetricright = PCApplySymmetricRight_BJacobi_Singleblock;
      pc->ops->applytranspose      = PCApplyTranspose_BJacobi_Singleblock;
      pc->ops->setuponblocks       = PCSetUpOnBlocks_BJacobi_Singleblock;

      CHKERRQ(PetscMalloc1(1,&jac->ksp));
      jac->ksp[0] = ksp;

      CHKERRQ(PetscNewLog(pc,&bjac));
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
    CHKERRQ(MatGetSize(pmat,&m,&m));
    CHKERRQ(VecCreateSeqWithArray(PETSC_COMM_SELF,1,m,NULL,&bjac->x));
    CHKERRQ(VecCreateSeqWithArray(PETSC_COMM_SELF,1,m,NULL,&bjac->y));
    CHKERRQ(MatGetVecType(pmat,&vectype));
    CHKERRQ(VecSetType(bjac->x,vectype));
    CHKERRQ(VecSetType(bjac->y,vectype));
    CHKERRQ(PetscLogObjectParent((PetscObject)pc,(PetscObject)bjac->x));
    CHKERRQ(PetscLogObjectParent((PetscObject)pc,(PetscObject)bjac->y));
  } else {
    ksp  = jac->ksp[0];
    bjac = (PC_BJacobi_Singleblock*)jac->data;
  }
  CHKERRQ(KSPGetOptionsPrefix(ksp,&prefix));
  if (pc->useAmat) {
    CHKERRQ(KSPSetOperators(ksp,mat,pmat));
    CHKERRQ(MatSetOptionsPrefix(mat,prefix));
  } else {
    CHKERRQ(KSPSetOperators(ksp,pmat,pmat));
  }
  CHKERRQ(MatSetOptionsPrefix(pmat,prefix));
  if (!wasSetup && pc->setfromoptionscalled) {
    /* If PCSetFromOptions_BJacobi is called later, KSPSetFromOptions will be called at that time. */
    CHKERRQ(KSPSetFromOptions(ksp));
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------------------------*/
static PetscErrorCode PCReset_BJacobi_Multiblock(PC pc)
{
  PC_BJacobi            *jac  = (PC_BJacobi*)pc->data;
  PC_BJacobi_Multiblock *bjac = (PC_BJacobi_Multiblock*)jac->data;
  PetscInt              i;

  PetscFunctionBegin;
  if (bjac && bjac->pmat) {
    CHKERRQ(MatDestroyMatrices(jac->n_local,&bjac->pmat));
    if (pc->useAmat) {
      CHKERRQ(MatDestroyMatrices(jac->n_local,&bjac->mat));
    }
  }

  for (i=0; i<jac->n_local; i++) {
    CHKERRQ(KSPReset(jac->ksp[i]));
    if (bjac && bjac->x) {
      CHKERRQ(VecDestroy(&bjac->x[i]));
      CHKERRQ(VecDestroy(&bjac->y[i]));
      CHKERRQ(ISDestroy(&bjac->is[i]));
    }
  }
  CHKERRQ(PetscFree(jac->l_lens));
  CHKERRQ(PetscFree(jac->g_lens));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDestroy_BJacobi_Multiblock(PC pc)
{
  PC_BJacobi            *jac  = (PC_BJacobi*)pc->data;
  PC_BJacobi_Multiblock *bjac = (PC_BJacobi_Multiblock*)jac->data;
  PetscInt              i;

  PetscFunctionBegin;
  CHKERRQ(PCReset_BJacobi_Multiblock(pc));
  if (bjac) {
    CHKERRQ(PetscFree2(bjac->x,bjac->y));
    CHKERRQ(PetscFree(bjac->starts));
    CHKERRQ(PetscFree(bjac->is));
  }
  CHKERRQ(PetscFree(jac->data));
  for (i=0; i<jac->n_local; i++) {
    CHKERRQ(KSPDestroy(&jac->ksp[i]));
  }
  CHKERRQ(PetscFree(jac->ksp));
  CHKERRQ(PetscFree(pc->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetUpOnBlocks_BJacobi_Multiblock(PC pc)
{
  PC_BJacobi         *jac = (PC_BJacobi*)pc->data;
  PetscInt           i,n_local = jac->n_local;
  KSPConvergedReason reason;

  PetscFunctionBegin;
  for (i=0; i<n_local; i++) {
    CHKERRQ(KSPSetUp(jac->ksp[i]));
    CHKERRQ(KSPGetConvergedReason(jac->ksp[i],&reason));
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
  PetscInt              i,n_local = jac->n_local;
  PC_BJacobi_Multiblock *bjac = (PC_BJacobi_Multiblock*)jac->data;
  PetscScalar           *yin;
  const PetscScalar     *xin;

  PetscFunctionBegin;
  CHKERRQ(VecGetArrayRead(x,&xin));
  CHKERRQ(VecGetArray(y,&yin));
  for (i=0; i<n_local; i++) {
    /*
       To avoid copying the subvector from x into a workspace we instead
       make the workspace vector array point to the subpart of the array of
       the global vector.
    */
    CHKERRQ(VecPlaceArray(bjac->x[i],xin+bjac->starts[i]));
    CHKERRQ(VecPlaceArray(bjac->y[i],yin+bjac->starts[i]));

    CHKERRQ(PetscLogEventBegin(PC_ApplyOnBlocks,jac->ksp[i],bjac->x[i],bjac->y[i],0));
    CHKERRQ(KSPSolve(jac->ksp[i],bjac->x[i],bjac->y[i]));
    CHKERRQ(KSPCheckSolve(jac->ksp[i],pc,bjac->y[i]));
    CHKERRQ(PetscLogEventEnd(PC_ApplyOnBlocks,jac->ksp[i],bjac->x[i],bjac->y[i],0));

    CHKERRQ(VecResetArray(bjac->x[i]));
    CHKERRQ(VecResetArray(bjac->y[i]));
  }
  CHKERRQ(VecRestoreArrayRead(x,&xin));
  CHKERRQ(VecRestoreArray(y,&yin));
  PetscFunctionReturn(0);
}

/*
      Preconditioner for block Jacobi
*/
static PetscErrorCode PCApplyTranspose_BJacobi_Multiblock(PC pc,Vec x,Vec y)
{
  PC_BJacobi            *jac = (PC_BJacobi*)pc->data;
  PetscInt              i,n_local = jac->n_local;
  PC_BJacobi_Multiblock *bjac = (PC_BJacobi_Multiblock*)jac->data;
  PetscScalar           *yin;
  const PetscScalar     *xin;

  PetscFunctionBegin;
  CHKERRQ(VecGetArrayRead(x,&xin));
  CHKERRQ(VecGetArray(y,&yin));
  for (i=0; i<n_local; i++) {
    /*
       To avoid copying the subvector from x into a workspace we instead
       make the workspace vector array point to the subpart of the array of
       the global vector.
    */
    CHKERRQ(VecPlaceArray(bjac->x[i],xin+bjac->starts[i]));
    CHKERRQ(VecPlaceArray(bjac->y[i],yin+bjac->starts[i]));

    CHKERRQ(PetscLogEventBegin(PC_ApplyTransposeOnBlocks,jac->ksp[i],bjac->x[i],bjac->y[i],0));
    CHKERRQ(KSPSolveTranspose(jac->ksp[i],bjac->x[i],bjac->y[i]));
    CHKERRQ(KSPCheckSolve(jac->ksp[i],pc,bjac->y[i]));
    CHKERRQ(PetscLogEventEnd(PC_ApplyTransposeOnBlocks,jac->ksp[i],bjac->x[i],bjac->y[i],0));

    CHKERRQ(VecResetArray(bjac->x[i]));
    CHKERRQ(VecResetArray(bjac->y[i]));
  }
  CHKERRQ(VecRestoreArrayRead(x,&xin));
  CHKERRQ(VecRestoreArray(y,&yin));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetUp_BJacobi_Multiblock(PC pc,Mat mat,Mat pmat)
{
  PC_BJacobi            *jac = (PC_BJacobi*)pc->data;
  PetscInt              m,n_local,N,M,start,i;
  const char            *prefix;
  KSP                   ksp;
  Vec                   x,y;
  PC_BJacobi_Multiblock *bjac = (PC_BJacobi_Multiblock*)jac->data;
  PC                    subpc;
  IS                    is;
  MatReuse              scall;
  VecType               vectype;

  PetscFunctionBegin;
  CHKERRQ(MatGetLocalSize(pc->pmat,&M,&N));

  n_local = jac->n_local;

  if (pc->useAmat) {
    PetscBool same;
    CHKERRQ(PetscObjectTypeCompare((PetscObject)mat,((PetscObject)pmat)->type_name,&same));
    PetscCheck(same,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_INCOMP,"Matrices not of same type");
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

      CHKERRQ(PetscNewLog(pc,&bjac));
      CHKERRQ(PetscMalloc1(n_local,&jac->ksp));
      CHKERRQ(PetscLogObjectMemory((PetscObject)pc,sizeof(n_local*sizeof(KSP))));
      CHKERRQ(PetscMalloc2(n_local,&bjac->x,n_local,&bjac->y));
      CHKERRQ(PetscMalloc1(n_local,&bjac->starts));
      CHKERRQ(PetscLogObjectMemory((PetscObject)pc,sizeof(n_local*sizeof(PetscScalar))));

      jac->data = (void*)bjac;
      CHKERRQ(PetscMalloc1(n_local,&bjac->is));
      CHKERRQ(PetscLogObjectMemory((PetscObject)pc,sizeof(n_local*sizeof(IS))));

      for (i=0; i<n_local; i++) {
        CHKERRQ(KSPCreate(PETSC_COMM_SELF,&ksp));
        CHKERRQ(KSPSetErrorIfNotConverged(ksp,pc->erroriffailure));
        CHKERRQ(PetscObjectIncrementTabLevel((PetscObject)ksp,(PetscObject)pc,1));
        CHKERRQ(PetscLogObjectParent((PetscObject)pc,(PetscObject)ksp));
        CHKERRQ(KSPSetType(ksp,KSPPREONLY));
        CHKERRQ(KSPGetPC(ksp,&subpc));
        CHKERRQ(PCGetOptionsPrefix(pc,&prefix));
        CHKERRQ(KSPSetOptionsPrefix(ksp,prefix));
        CHKERRQ(KSPAppendOptionsPrefix(ksp,"sub_"));

        jac->ksp[i] = ksp;
      }
    } else {
      bjac = (PC_BJacobi_Multiblock*)jac->data;
    }

    start = 0;
    CHKERRQ(MatGetVecType(pmat,&vectype));
    for (i=0; i<n_local; i++) {
      m = jac->l_lens[i];
      /*
      The reason we need to generate these vectors is to serve
      as the right-hand side and solution vector for the solve on the
      block. We do not need to allocate space for the vectors since
      that is provided via VecPlaceArray() just before the call to
      KSPSolve() on the block.

      */
      CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,m,&x));
      CHKERRQ(VecCreateSeqWithArray(PETSC_COMM_SELF,1,m,NULL,&y));
      CHKERRQ(VecSetType(x,vectype));
      CHKERRQ(VecSetType(y,vectype));
      CHKERRQ(PetscLogObjectParent((PetscObject)pc,(PetscObject)x));
      CHKERRQ(PetscLogObjectParent((PetscObject)pc,(PetscObject)y));

      bjac->x[i]      = x;
      bjac->y[i]      = y;
      bjac->starts[i] = start;

      CHKERRQ(ISCreateStride(PETSC_COMM_SELF,m,start,1,&is));
      bjac->is[i] = is;
      CHKERRQ(PetscLogObjectParent((PetscObject)pc,(PetscObject)is));

      start += m;
    }
  } else {
    bjac = (PC_BJacobi_Multiblock*)jac->data;
    /*
       Destroy the blocks from the previous iteration
    */
    if (pc->flag == DIFFERENT_NONZERO_PATTERN) {
      CHKERRQ(MatDestroyMatrices(n_local,&bjac->pmat));
      if (pc->useAmat) {
        CHKERRQ(MatDestroyMatrices(n_local,&bjac->mat));
      }
      scall = MAT_INITIAL_MATRIX;
    } else scall = MAT_REUSE_MATRIX;
  }

  CHKERRQ(MatCreateSubMatrices(pmat,n_local,bjac->is,bjac->is,scall,&bjac->pmat));
  if (pc->useAmat) {
    CHKERRQ(MatCreateSubMatrices(mat,n_local,bjac->is,bjac->is,scall,&bjac->mat));
  }
  /* Return control to the user so that the submatrices can be modified (e.g., to apply
     different boundary conditions for the submatrices than for the global problem) */
  CHKERRQ(PCModifySubMatrices(pc,n_local,bjac->is,bjac->is,bjac->pmat,pc->modifysubmatricesP));

  for (i=0; i<n_local; i++) {
    CHKERRQ(PetscLogObjectParent((PetscObject)pc,(PetscObject)bjac->pmat[i]));
    CHKERRQ(KSPGetOptionsPrefix(jac->ksp[i],&prefix));
    if (pc->useAmat) {
      CHKERRQ(PetscLogObjectParent((PetscObject)pc,(PetscObject)bjac->mat[i]));
      CHKERRQ(KSPSetOperators(jac->ksp[i],bjac->mat[i],bjac->pmat[i]));
      CHKERRQ(MatSetOptionsPrefix(bjac->mat[i],prefix));
    } else {
      CHKERRQ(KSPSetOperators(jac->ksp[i],bjac->pmat[i],bjac->pmat[i]));
    }
    CHKERRQ(MatSetOptionsPrefix(bjac->pmat[i],prefix));
    if (pc->setfromoptionscalled) {
      CHKERRQ(KSPSetFromOptions(jac->ksp[i]));
    }
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------------------------*/
/*
      These are for a single block with multiple processes
*/
static PetscErrorCode PCSetUpOnBlocks_BJacobi_Multiproc(PC pc)
{
  PC_BJacobi         *jac = (PC_BJacobi*)pc->data;
  KSP                subksp = jac->ksp[0];
  KSPConvergedReason reason;

  PetscFunctionBegin;
  CHKERRQ(KSPSetUp(subksp));
  CHKERRQ(KSPGetConvergedReason(subksp,&reason));
  if (reason == KSP_DIVERGED_PC_FAILED) {
    pc->failedreason = PC_SUBPC_ERROR;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCReset_BJacobi_Multiproc(PC pc)
{
  PC_BJacobi           *jac   = (PC_BJacobi*)pc->data;
  PC_BJacobi_Multiproc *mpjac = (PC_BJacobi_Multiproc*)jac->data;

  PetscFunctionBegin;
  CHKERRQ(VecDestroy(&mpjac->ysub));
  CHKERRQ(VecDestroy(&mpjac->xsub));
  CHKERRQ(MatDestroy(&mpjac->submats));
  if (jac->ksp) CHKERRQ(KSPReset(jac->ksp[0]));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDestroy_BJacobi_Multiproc(PC pc)
{
  PC_BJacobi           *jac   = (PC_BJacobi*)pc->data;
  PC_BJacobi_Multiproc *mpjac = (PC_BJacobi_Multiproc*)jac->data;

  PetscFunctionBegin;
  CHKERRQ(PCReset_BJacobi_Multiproc(pc));
  CHKERRQ(KSPDestroy(&jac->ksp[0]));
  CHKERRQ(PetscFree(jac->ksp));
  CHKERRQ(PetscSubcommDestroy(&mpjac->psubcomm));

  CHKERRQ(PetscFree(mpjac));
  CHKERRQ(PetscFree(pc->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApply_BJacobi_Multiproc(PC pc,Vec x,Vec y)
{
  PC_BJacobi           *jac   = (PC_BJacobi*)pc->data;
  PC_BJacobi_Multiproc *mpjac = (PC_BJacobi_Multiproc*)jac->data;
  PetscScalar          *yarray;
  const PetscScalar    *xarray;
  KSPConvergedReason   reason;

  PetscFunctionBegin;
  /* place x's and y's local arrays into xsub and ysub */
  CHKERRQ(VecGetArrayRead(x,&xarray));
  CHKERRQ(VecGetArray(y,&yarray));
  CHKERRQ(VecPlaceArray(mpjac->xsub,xarray));
  CHKERRQ(VecPlaceArray(mpjac->ysub,yarray));

  /* apply preconditioner on each matrix block */
  CHKERRQ(PetscLogEventBegin(PC_ApplyOnBlocks,jac->ksp[0],mpjac->xsub,mpjac->ysub,0));
  CHKERRQ(KSPSolve(jac->ksp[0],mpjac->xsub,mpjac->ysub));
  CHKERRQ(KSPCheckSolve(jac->ksp[0],pc,mpjac->ysub));
  CHKERRQ(PetscLogEventEnd(PC_ApplyOnBlocks,jac->ksp[0],mpjac->xsub,mpjac->ysub,0));
  CHKERRQ(KSPGetConvergedReason(jac->ksp[0],&reason));
  if (reason == KSP_DIVERGED_PC_FAILED) {
    pc->failedreason = PC_SUBPC_ERROR;
  }

  CHKERRQ(VecResetArray(mpjac->xsub));
  CHKERRQ(VecResetArray(mpjac->ysub));
  CHKERRQ(VecRestoreArrayRead(x,&xarray));
  CHKERRQ(VecRestoreArray(y,&yarray));
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

  PetscFunctionBegin;
  /* apply preconditioner on each matrix block */
  CHKERRQ(MatGetLocalSize(X,&m,NULL));
  CHKERRQ(MatGetSize(X,NULL,&N));
  CHKERRQ(MatDenseGetLDA(X,&lda));
  CHKERRQ(MatDenseGetLDA(Y,&ldb));
  CHKERRQ(MatDenseGetArrayRead(X,&x));
  CHKERRQ(MatDenseGetArrayWrite(Y,&y));
  CHKERRQ(MatCreateDense(PetscObjectComm((PetscObject)jac->ksp[0]),m,PETSC_DECIDE,PETSC_DECIDE,N,(PetscScalar*)x,&sX));
  CHKERRQ(MatCreateDense(PetscObjectComm((PetscObject)jac->ksp[0]),m,PETSC_DECIDE,PETSC_DECIDE,N,y,&sY));
  CHKERRQ(MatDenseSetLDA(sX,lda));
  CHKERRQ(MatDenseSetLDA(sY,ldb));
  CHKERRQ(PetscLogEventBegin(PC_ApplyOnBlocks,jac->ksp[0],X,Y,0));
  CHKERRQ(KSPMatSolve(jac->ksp[0],sX,sY));
  CHKERRQ(KSPCheckSolve(jac->ksp[0],pc,NULL));
  CHKERRQ(PetscLogEventEnd(PC_ApplyOnBlocks,jac->ksp[0],X,Y,0));
  CHKERRQ(MatDestroy(&sY));
  CHKERRQ(MatDestroy(&sX));
  CHKERRQ(MatDenseRestoreArrayWrite(Y,&y));
  CHKERRQ(MatDenseRestoreArrayRead(X,&x));
  CHKERRQ(KSPGetConvergedReason(jac->ksp[0],&reason));
  if (reason == KSP_DIVERGED_PC_FAILED) {
    pc->failedreason = PC_SUBPC_ERROR;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetUp_BJacobi_Multiproc(PC pc)
{
  PC_BJacobi           *jac   = (PC_BJacobi*)pc->data;
  PC_BJacobi_Multiproc *mpjac = (PC_BJacobi_Multiproc*)jac->data;
  PetscInt             m,n;
  MPI_Comm             comm,subcomm=0;
  const char           *prefix;
  PetscBool            wasSetup = PETSC_TRUE;
  VecType              vectype;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)pc,&comm));
  PetscCheckFalse(jac->n_local > 1,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Only a single block in a subcommunicator is supported");
  jac->n_local = 1; /* currently only a single block is supported for a subcommunicator */
  if (!pc->setupcalled) {
    wasSetup  = PETSC_FALSE;
    CHKERRQ(PetscNewLog(pc,&mpjac));
    jac->data = (void*)mpjac;

    /* initialize datastructure mpjac */
    if (!jac->psubcomm) {
      /* Create default contiguous subcommunicatiors if user does not provide them */
      CHKERRQ(PetscSubcommCreate(comm,&jac->psubcomm));
      CHKERRQ(PetscSubcommSetNumber(jac->psubcomm,jac->n));
      CHKERRQ(PetscSubcommSetType(jac->psubcomm,PETSC_SUBCOMM_CONTIGUOUS));
      CHKERRQ(PetscLogObjectMemory((PetscObject)pc,sizeof(PetscSubcomm)));
    }
    mpjac->psubcomm = jac->psubcomm;
    subcomm         = PetscSubcommChild(mpjac->psubcomm);

    /* Get matrix blocks of pmat */
    CHKERRQ(MatGetMultiProcBlock(pc->pmat,subcomm,MAT_INITIAL_MATRIX,&mpjac->submats));

    /* create a new PC that processors in each subcomm have copy of */
    CHKERRQ(PetscMalloc1(1,&jac->ksp));
    CHKERRQ(KSPCreate(subcomm,&jac->ksp[0]));
    CHKERRQ(KSPSetErrorIfNotConverged(jac->ksp[0],pc->erroriffailure));
    CHKERRQ(PetscObjectIncrementTabLevel((PetscObject)jac->ksp[0],(PetscObject)pc,1));
    CHKERRQ(PetscLogObjectParent((PetscObject)pc,(PetscObject)jac->ksp[0]));
    CHKERRQ(KSPSetOperators(jac->ksp[0],mpjac->submats,mpjac->submats));
    CHKERRQ(KSPGetPC(jac->ksp[0],&mpjac->pc));

    CHKERRQ(PCGetOptionsPrefix(pc,&prefix));
    CHKERRQ(KSPSetOptionsPrefix(jac->ksp[0],prefix));
    CHKERRQ(KSPAppendOptionsPrefix(jac->ksp[0],"sub_"));
    CHKERRQ(KSPGetOptionsPrefix(jac->ksp[0],&prefix));
    CHKERRQ(MatSetOptionsPrefix(mpjac->submats,prefix));

    /* create dummy vectors xsub and ysub */
    CHKERRQ(MatGetLocalSize(mpjac->submats,&m,&n));
    CHKERRQ(VecCreateMPIWithArray(subcomm,1,n,PETSC_DECIDE,NULL,&mpjac->xsub));
    CHKERRQ(VecCreateMPIWithArray(subcomm,1,m,PETSC_DECIDE,NULL,&mpjac->ysub));
    CHKERRQ(MatGetVecType(mpjac->submats,&vectype));
    CHKERRQ(VecSetType(mpjac->xsub,vectype));
    CHKERRQ(VecSetType(mpjac->ysub,vectype));
    CHKERRQ(PetscLogObjectParent((PetscObject)pc,(PetscObject)mpjac->xsub));
    CHKERRQ(PetscLogObjectParent((PetscObject)pc,(PetscObject)mpjac->ysub));

    pc->ops->setuponblocks = PCSetUpOnBlocks_BJacobi_Multiproc;
    pc->ops->reset         = PCReset_BJacobi_Multiproc;
    pc->ops->destroy       = PCDestroy_BJacobi_Multiproc;
    pc->ops->apply         = PCApply_BJacobi_Multiproc;
    pc->ops->matapply      = PCMatApply_BJacobi_Multiproc;
  } else { /* pc->setupcalled */
    subcomm = PetscSubcommChild(mpjac->psubcomm);
    if (pc->flag == DIFFERENT_NONZERO_PATTERN) {
      /* destroy old matrix blocks, then get new matrix blocks */
      if (mpjac->submats) CHKERRQ(MatDestroy(&mpjac->submats));
      CHKERRQ(MatGetMultiProcBlock(pc->pmat,subcomm,MAT_INITIAL_MATRIX,&mpjac->submats));
    } else {
      CHKERRQ(MatGetMultiProcBlock(pc->pmat,subcomm,MAT_REUSE_MATRIX,&mpjac->submats));
    }
    CHKERRQ(KSPSetOperators(jac->ksp[0],mpjac->submats,mpjac->submats));
  }

  if (!wasSetup && pc->setfromoptionscalled) {
    CHKERRQ(KSPSetFromOptions(jac->ksp[0]));
  }
  PetscFunctionReturn(0);
}
