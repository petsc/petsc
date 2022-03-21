
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
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)pc),&rank));
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)pc),&size));
  PetscCall(MatGetLocalSize(pc->pmat,&M,&N));
  PetscCall(MatGetBlockSize(pc->pmat,&bs));

  if (jac->n > 0 && jac->n < size) {
    PetscCall(PCSetUp_BJacobi_Multiproc(pc));
    PetscFunctionReturn(0);
  }

  /* --------------------------------------------------------------------------
      Determines the number of blocks assigned to each processor
  -----------------------------------------------------------------------------*/

  /*   local block count  given */
  if (jac->n_local > 0 && jac->n < 0) {
    PetscCallMPI(MPIU_Allreduce(&jac->n_local,&jac->n,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)pc)));
    if (jac->l_lens) { /* check that user set these correctly */
      sum = 0;
      for (i=0; i<jac->n_local; i++) {
        PetscCheckFalse(jac->l_lens[i]/bs*bs !=jac->l_lens[i],PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Mat blocksize doesn't match block Jacobi layout");
        sum += jac->l_lens[i];
      }
      PetscCheckFalse(sum != M,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local lens set incorrectly");
    } else {
      PetscCall(PetscMalloc1(jac->n_local,&jac->l_lens));
      for (i=0; i<jac->n_local; i++) jac->l_lens[i] = bs*((M/bs)/jac->n_local + (((M/bs) % jac->n_local) > i));
    }
  } else if (jac->n > 0 && jac->n_local < 0) { /* global block count given */
    /* global blocks given: determine which ones are local */
    if (jac->g_lens) {
      /* check if the g_lens is has valid entries */
      for (i=0; i<jac->n; i++) {
        PetscCheck(jac->g_lens[i],PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Zero block not allowed");
        PetscCheckFalse(jac->g_lens[i]/bs*bs != jac->g_lens[i],PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Mat blocksize doesn't match block Jacobi layout");
      }
      if (size == 1) {
        jac->n_local = jac->n;
        PetscCall(PetscMalloc1(jac->n_local,&jac->l_lens));
        PetscCall(PetscArraycpy(jac->l_lens,jac->g_lens,jac->n_local));
        /* check that user set these correctly */
        sum = 0;
        for (i=0; i<jac->n_local; i++) sum += jac->l_lens[i];
        PetscCheckFalse(sum != M,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Global lens set incorrectly");
      } else {
        PetscCall(MatGetOwnershipRange(pc->pmat,&start,&end));
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
        PetscCall(PetscMalloc1(jac->n_local,&jac->l_lens));
        PetscCall(PetscArraycpy(jac->l_lens,jac->g_lens+i_start,jac->n_local));
      }
    } else { /* no global blocks given, determine then using default layout */
      jac->n_local = jac->n/size + ((jac->n % size) > rank);
      PetscCall(PetscMalloc1(jac->n_local,&jac->l_lens));
      for (i=0; i<jac->n_local; i++) {
        jac->l_lens[i] = ((M/bs)/jac->n_local + (((M/bs) % jac->n_local) > i))*bs;
        PetscCheck(jac->l_lens[i],PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Too many blocks given");
      }
    }
  } else if (jac->n < 0 && jac->n_local < 0) { /* no blocks given */
    jac->n         = size;
    jac->n_local   = 1;
    PetscCall(PetscMalloc1(1,&jac->l_lens));
    jac->l_lens[0] = M;
  } else { /* jac->n > 0 && jac->n_local > 0 */
    if (!jac->l_lens) {
      PetscCall(PetscMalloc1(jac->n_local,&jac->l_lens));
      for (i=0; i<jac->n_local; i++) jac->l_lens[i] = bs*((M/bs)/jac->n_local + (((M/bs) % jac->n_local) > i));
    }
  }
  PetscCheckFalse(jac->n_local < 1,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Number of blocks is less than number of processors");

  /* -------------------------
      Determines mat and pmat
  ---------------------------*/
  PetscCall(MatHasOperation(pc->mat,MATOP_GET_DIAGONAL_BLOCK,&hasop));
  if (!hasop && size == 1) {
    mat  = pc->mat;
    pmat = pc->pmat;
  } else {
    if (pc->useAmat) {
      /* use block from Amat matrix, not Pmat for local MatMult() */
      PetscCall(MatGetDiagonalBlock(pc->mat,&mat));
    }
    if (pc->pmat != pc->mat || !pc->useAmat) {
      PetscCall(MatGetDiagonalBlock(pc->pmat,&pmat));
    } else pmat = mat;
  }

  /* ------
     Setup code depends on the number of blocks
  */
  if (jac->n_local == 1) {
    PetscCall(PCSetUp_BJacobi_Singleblock(pc,mat,pmat));
  } else {
    PetscCall(PCSetUp_BJacobi_Multiblock(pc,mat,pmat));
  }
  PetscFunctionReturn(0);
}

/* Default destroy, if it has never been setup */
static PetscErrorCode PCDestroy_BJacobi(PC pc)
{
  PC_BJacobi     *jac = (PC_BJacobi*)pc->data;

  PetscFunctionBegin;
  PetscCall(PetscFree(jac->g_lens));
  PetscCall(PetscFree(jac->l_lens));
  PetscCall(PetscFree(pc->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetFromOptions_BJacobi(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PC_BJacobi     *jac = (PC_BJacobi*)pc->data;
  PetscInt       blocks,i;
  PetscBool      flg;

  PetscFunctionBegin;
  PetscCall(PetscOptionsHead(PetscOptionsObject,"Block Jacobi options"));
  PetscCall(PetscOptionsInt("-pc_bjacobi_blocks","Total number of blocks","PCBJacobiSetTotalBlocks",jac->n,&blocks,&flg));
  if (flg) PetscCall(PCBJacobiSetTotalBlocks(pc,blocks,NULL));
  PetscCall(PetscOptionsInt("-pc_bjacobi_local_blocks","Local number of blocks","PCBJacobiSetLocalBlocks",jac->n_local,&blocks,&flg));
  if (flg) PetscCall(PCBJacobiSetLocalBlocks(pc,blocks,NULL));
  if (jac->ksp) {
    /* The sub-KSP has already been set up (e.g., PCSetUp_BJacobi_Singleblock), but KSPSetFromOptions was not called
     * unless we had already been called. */
    for (i=0; i<jac->n_local; i++) {
      PetscCall(KSPSetFromOptions(jac->ksp[i]));
    }
  }
  PetscCall(PetscOptionsTail());
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
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERSTRING,&isstring));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERDRAW,&isdraw));
  if (iascii) {
    if (pc->useAmat) {
      PetscCall(PetscViewerASCIIPrintf(viewer,"  using Amat local matrix, number of blocks = %D\n",jac->n));
    }
    PetscCall(PetscViewerASCIIPrintf(viewer,"  number of blocks = %D\n",jac->n));
    PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)pc),&rank));
    PetscCall(PetscViewerGetFormat(viewer,&format));
    if (format != PETSC_VIEWER_ASCII_INFO_DETAIL) {
      PetscCall(PetscViewerASCIIPrintf(viewer,"  Local solver information for first block is in the following KSP and PC objects on rank 0:\n"));
      PetscCall(PCGetOptionsPrefix(pc,&prefix));
      PetscCall(PetscViewerASCIIPrintf(viewer,"  Use -%sksp_view ::ascii_info_detail to display information for all blocks\n",prefix?prefix:""));
      if (jac->ksp && !jac->psubcomm) {
        PetscCall(PetscViewerGetSubViewer(viewer,PETSC_COMM_SELF,&sviewer));
        if (rank == 0) {
          PetscCall(PetscViewerASCIIPushTab(viewer));
          PetscCall(KSPView(jac->ksp[0],sviewer));
          PetscCall(PetscViewerASCIIPopTab(viewer));
        }
        PetscCall(PetscViewerFlush(sviewer));
        PetscCall(PetscViewerRestoreSubViewer(viewer,PETSC_COMM_SELF,&sviewer));
        PetscCall(PetscViewerFlush(viewer));
        /*  extra call needed because of the two calls to PetscViewerASCIIPushSynchronized() in PetscViewerGetSubViewer() */
        PetscCall(PetscViewerASCIIPopSynchronized(viewer));
      } else if (mpjac && jac->ksp && mpjac->psubcomm) {
        PetscCall(PetscViewerGetSubViewer(viewer,mpjac->psubcomm->child,&sviewer));
        if (!mpjac->psubcomm->color) {
          PetscCall(PetscViewerASCIIPushTab(viewer));
          PetscCall(KSPView(*(jac->ksp),sviewer));
          PetscCall(PetscViewerASCIIPopTab(viewer));
        }
        PetscCall(PetscViewerFlush(sviewer));
        PetscCall(PetscViewerRestoreSubViewer(viewer,mpjac->psubcomm->child,&sviewer));
        PetscCall(PetscViewerFlush(viewer));
        /*  extra call needed because of the two calls to PetscViewerASCIIPushSynchronized() in PetscViewerGetSubViewer() */
        PetscCall(PetscViewerASCIIPopSynchronized(viewer));
      } else {
        PetscCall(PetscViewerFlush(viewer));
      }
    } else {
      PetscInt n_global;
      PetscCallMPI(MPIU_Allreduce(&jac->n_local,&n_global,1,MPIU_INT,MPI_MAX,PetscObjectComm((PetscObject)pc)));
      PetscCall(PetscViewerASCIIPushSynchronized(viewer));
      PetscCall(PetscViewerASCIIPrintf(viewer,"  Local solver information for each block is in the following KSP and PC objects:\n"));
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] number of local blocks = %D, first local block number = %D\n",
                                                rank,jac->n_local,jac->first_local);PetscCall(ierr);
      PetscCall(PetscViewerASCIIPushTab(viewer));
      PetscCall(PetscViewerGetSubViewer(viewer,PETSC_COMM_SELF,&sviewer));
      for (i=0; i<jac->n_local; i++) {
        PetscCall(PetscViewerASCIISynchronizedPrintf(viewer,"[%d] local block number %D\n",rank,i));
        PetscCall(KSPView(jac->ksp[i],sviewer));
        PetscCall(PetscViewerASCIISynchronizedPrintf(viewer,"- - - - - - - - - - - - - - - - - -\n"));
      }
      PetscCall(PetscViewerRestoreSubViewer(viewer,PETSC_COMM_SELF,&sviewer));
      PetscCall(PetscViewerASCIIPopTab(viewer));
      PetscCall(PetscViewerFlush(viewer));
      PetscCall(PetscViewerASCIIPopSynchronized(viewer));
    }
  } else if (isstring) {
    PetscCall(PetscViewerStringSPrintf(viewer," blks=%D",jac->n));
    PetscCall(PetscViewerGetSubViewer(viewer,PETSC_COMM_SELF,&sviewer));
    if (jac->ksp) PetscCall(KSPView(jac->ksp[0],sviewer));
    PetscCall(PetscViewerRestoreSubViewer(viewer,PETSC_COMM_SELF,&sviewer));
  } else if (isdraw) {
    PetscDraw draw;
    char      str[25];
    PetscReal x,y,bottom,h;

    PetscCall(PetscViewerDrawGetDraw(viewer,0,&draw));
    PetscCall(PetscDrawGetCurrentPoint(draw,&x,&y));
    PetscCall(PetscSNPrintf(str,25,"Number blocks %D",jac->n));
    PetscCall(PetscDrawStringBoxed(draw,x,y,PETSC_DRAW_RED,PETSC_DRAW_BLACK,str,NULL,&h));
    bottom = y - h;
    PetscCall(PetscDrawPushCurrentPoint(draw,x,bottom));
    /* warning the communicator on viewer is different then on ksp in parallel */
    if (jac->ksp) PetscCall(KSPView(jac->ksp[0],viewer));
    PetscCall(PetscDrawPopCurrentPoint(draw));
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
    PetscCall(PetscMalloc1(blocks,&jac->g_lens));
    PetscCall(PetscLogObjectMemory((PetscObject)pc,blocks*sizeof(PetscInt)));
    PetscCall(PetscArraycpy(jac->g_lens,lens,blocks));
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
    PetscCall(PetscMalloc1(blocks,&jac->l_lens));
    PetscCall(PetscLogObjectMemory((PetscObject)pc,blocks*sizeof(PetscInt)));
    PetscCall(PetscArraycpy(jac->l_lens,lens,blocks));
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
  PetscCall(PetscUseMethod(pc,"PCBJacobiGetSubKSP_C",(PC,PetscInt*,PetscInt*,KSP **),(pc,n_local,first_local,ksp)));
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
  PetscCall(PetscTryMethod(pc,"PCBJacobiSetTotalBlocks_C",(PC,PetscInt,const PetscInt[]),(pc,blocks,lens)));
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
  PetscCall(PetscUseMethod(pc,"PCBJacobiGetTotalBlocks_C",(PC,PetscInt*, const PetscInt *[]),(pc,blocks,lens)));
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
  PetscCall(PetscTryMethod(pc,"PCBJacobiSetLocalBlocks_C",(PC,PetscInt,const PetscInt []),(pc,blocks,lens)));
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
  PetscCall(PetscUseMethod(pc,"PCBJacobiGetLocalBlocks_C",(PC,PetscInt*, const PetscInt *[]),(pc,blocks,lens)));
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
  PetscCall(PetscNewLog(pc,&jac));
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)pc),&rank));

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

  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCBJacobiGetSubKSP_C",PCBJacobiGetSubKSP_BJacobi));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCBJacobiSetTotalBlocks_C",PCBJacobiSetTotalBlocks_BJacobi));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCBJacobiGetTotalBlocks_C",PCBJacobiGetTotalBlocks_BJacobi));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCBJacobiSetLocalBlocks_C",PCBJacobiSetLocalBlocks_BJacobi));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCBJacobiGetLocalBlocks_C",PCBJacobiGetLocalBlocks_BJacobi));
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
  PetscCall(KSPReset(jac->ksp[0]));
  PetscCall(VecDestroy(&bjac->x));
  PetscCall(VecDestroy(&bjac->y));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDestroy_BJacobi_Singleblock(PC pc)
{
  PC_BJacobi             *jac  = (PC_BJacobi*)pc->data;
  PC_BJacobi_Singleblock *bjac = (PC_BJacobi_Singleblock*)jac->data;

  PetscFunctionBegin;
  PetscCall(PCReset_BJacobi_Singleblock(pc));
  PetscCall(KSPDestroy(&jac->ksp[0]));
  PetscCall(PetscFree(jac->ksp));
  PetscCall(PetscFree(jac->l_lens));
  PetscCall(PetscFree(jac->g_lens));
  PetscCall(PetscFree(bjac));
  PetscCall(PetscFree(pc->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetUpOnBlocks_BJacobi_Singleblock(PC pc)
{
  PC_BJacobi         *jac = (PC_BJacobi*)pc->data;
  KSP                subksp = jac->ksp[0];
  KSPConvergedReason reason;

  PetscFunctionBegin;
  PetscCall(KSPSetUp(subksp));
  PetscCall(KSPGetConvergedReason(subksp,&reason));
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
  PetscCall(VecGetLocalVectorRead(x, bjac->x));
  PetscCall(VecGetLocalVector(y, bjac->y));
  /* Since the inner KSP matrix may point directly to the diagonal block of an MPI matrix the inner
     matrix may change even if the outer KSP/PC has not updated the preconditioner, this will trigger a rebuild
     of the inner preconditioner automatically unless we pass down the outer preconditioners reuse flag.*/
  PetscCall(KSPSetReusePreconditioner(jac->ksp[0],pc->reusepreconditioner));
  PetscCall(KSPSolve(jac->ksp[0],bjac->x,bjac->y));
  PetscCall(KSPCheckSolve(jac->ksp[0],pc,bjac->y));
  PetscCall(VecRestoreLocalVectorRead(x, bjac->x));
  PetscCall(VecRestoreLocalVector(y, bjac->y));
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
  PetscCall(KSPSetReusePreconditioner(jac->ksp[0],pc->reusepreconditioner));
  PetscCall(MatDenseGetLocalMatrix(X,&sX));
  PetscCall(MatDenseGetLocalMatrix(Y,&sY));
  PetscCall(KSPMatSolve(jac->ksp[0],sX,sY));
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
  PetscCall(VecGetArrayRead(x,&x_array));
  PetscCall(VecGetArray(y,&y_array));
  PetscCall(VecPlaceArray(bjac->x,x_array));
  PetscCall(VecPlaceArray(bjac->y,y_array));
  /* apply the symmetric left portion of the inner PC operator */
  /* note this by-passes the inner KSP and its options completely */
  PetscCall(KSPGetPC(jac->ksp[0],&subpc));
  PetscCall(PCApplySymmetricLeft(subpc,bjac->x,bjac->y));
  PetscCall(VecResetArray(bjac->x));
  PetscCall(VecResetArray(bjac->y));
  PetscCall(VecRestoreArrayRead(x,&x_array));
  PetscCall(VecRestoreArray(y,&y_array));
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
  PetscCall(VecGetArrayRead(x,&x_array));
  PetscCall(VecGetArray(y,&y_array));
  PetscCall(VecPlaceArray(bjac->x,x_array));
  PetscCall(VecPlaceArray(bjac->y,y_array));

  /* apply the symmetric right portion of the inner PC operator */
  /* note this by-passes the inner KSP and its options completely */

  PetscCall(KSPGetPC(jac->ksp[0],&subpc));
  PetscCall(PCApplySymmetricRight(subpc,bjac->x,bjac->y));

  PetscCall(VecRestoreArrayRead(x,&x_array));
  PetscCall(VecRestoreArray(y,&y_array));
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
  PetscCall(VecGetArrayRead(x,&x_array));
  PetscCall(VecGetArray(y,&y_array));
  PetscCall(VecPlaceArray(bjac->x,x_array));
  PetscCall(VecPlaceArray(bjac->y,y_array));
  PetscCall(KSPSolveTranspose(jac->ksp[0],bjac->x,bjac->y));
  PetscCall(KSPCheckSolve(jac->ksp[0],pc,bjac->y));
  PetscCall(VecResetArray(bjac->x));
  PetscCall(VecResetArray(bjac->y));
  PetscCall(VecRestoreArrayRead(x,&x_array));
  PetscCall(VecRestoreArray(y,&y_array));
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

      PetscCall(KSPCreate(PETSC_COMM_SELF,&ksp));
      PetscCall(KSPSetErrorIfNotConverged(ksp,pc->erroriffailure));
      PetscCall(PetscObjectIncrementTabLevel((PetscObject)ksp,(PetscObject)pc,1));
      PetscCall(PetscLogObjectParent((PetscObject)pc,(PetscObject)ksp));
      PetscCall(KSPSetType(ksp,KSPPREONLY));
      PetscCall(PCGetOptionsPrefix(pc,&prefix));
      PetscCall(KSPSetOptionsPrefix(ksp,prefix));
      PetscCall(KSPAppendOptionsPrefix(ksp,"sub_"));

      pc->ops->reset               = PCReset_BJacobi_Singleblock;
      pc->ops->destroy             = PCDestroy_BJacobi_Singleblock;
      pc->ops->apply               = PCApply_BJacobi_Singleblock;
      pc->ops->matapply            = PCMatApply_BJacobi_Singleblock;
      pc->ops->applysymmetricleft  = PCApplySymmetricLeft_BJacobi_Singleblock;
      pc->ops->applysymmetricright = PCApplySymmetricRight_BJacobi_Singleblock;
      pc->ops->applytranspose      = PCApplyTranspose_BJacobi_Singleblock;
      pc->ops->setuponblocks       = PCSetUpOnBlocks_BJacobi_Singleblock;

      PetscCall(PetscMalloc1(1,&jac->ksp));
      jac->ksp[0] = ksp;

      PetscCall(PetscNewLog(pc,&bjac));
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
    PetscCall(MatGetSize(pmat,&m,&m));
    PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF,1,m,NULL,&bjac->x));
    PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF,1,m,NULL,&bjac->y));
    PetscCall(MatGetVecType(pmat,&vectype));
    PetscCall(VecSetType(bjac->x,vectype));
    PetscCall(VecSetType(bjac->y,vectype));
    PetscCall(PetscLogObjectParent((PetscObject)pc,(PetscObject)bjac->x));
    PetscCall(PetscLogObjectParent((PetscObject)pc,(PetscObject)bjac->y));
  } else {
    ksp  = jac->ksp[0];
    bjac = (PC_BJacobi_Singleblock*)jac->data;
  }
  PetscCall(KSPGetOptionsPrefix(ksp,&prefix));
  if (pc->useAmat) {
    PetscCall(KSPSetOperators(ksp,mat,pmat));
    PetscCall(MatSetOptionsPrefix(mat,prefix));
  } else {
    PetscCall(KSPSetOperators(ksp,pmat,pmat));
  }
  PetscCall(MatSetOptionsPrefix(pmat,prefix));
  if (!wasSetup && pc->setfromoptionscalled) {
    /* If PCSetFromOptions_BJacobi is called later, KSPSetFromOptions will be called at that time. */
    PetscCall(KSPSetFromOptions(ksp));
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
    PetscCall(MatDestroyMatrices(jac->n_local,&bjac->pmat));
    if (pc->useAmat) {
      PetscCall(MatDestroyMatrices(jac->n_local,&bjac->mat));
    }
  }

  for (i=0; i<jac->n_local; i++) {
    PetscCall(KSPReset(jac->ksp[i]));
    if (bjac && bjac->x) {
      PetscCall(VecDestroy(&bjac->x[i]));
      PetscCall(VecDestroy(&bjac->y[i]));
      PetscCall(ISDestroy(&bjac->is[i]));
    }
  }
  PetscCall(PetscFree(jac->l_lens));
  PetscCall(PetscFree(jac->g_lens));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDestroy_BJacobi_Multiblock(PC pc)
{
  PC_BJacobi            *jac  = (PC_BJacobi*)pc->data;
  PC_BJacobi_Multiblock *bjac = (PC_BJacobi_Multiblock*)jac->data;
  PetscInt              i;

  PetscFunctionBegin;
  PetscCall(PCReset_BJacobi_Multiblock(pc));
  if (bjac) {
    PetscCall(PetscFree2(bjac->x,bjac->y));
    PetscCall(PetscFree(bjac->starts));
    PetscCall(PetscFree(bjac->is));
  }
  PetscCall(PetscFree(jac->data));
  for (i=0; i<jac->n_local; i++) {
    PetscCall(KSPDestroy(&jac->ksp[i]));
  }
  PetscCall(PetscFree(jac->ksp));
  PetscCall(PetscFree(pc->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetUpOnBlocks_BJacobi_Multiblock(PC pc)
{
  PC_BJacobi         *jac = (PC_BJacobi*)pc->data;
  PetscInt           i,n_local = jac->n_local;
  KSPConvergedReason reason;

  PetscFunctionBegin;
  for (i=0; i<n_local; i++) {
    PetscCall(KSPSetUp(jac->ksp[i]));
    PetscCall(KSPGetConvergedReason(jac->ksp[i],&reason));
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
  PetscCall(VecGetArrayRead(x,&xin));
  PetscCall(VecGetArray(y,&yin));
  for (i=0; i<n_local; i++) {
    /*
       To avoid copying the subvector from x into a workspace we instead
       make the workspace vector array point to the subpart of the array of
       the global vector.
    */
    PetscCall(VecPlaceArray(bjac->x[i],xin+bjac->starts[i]));
    PetscCall(VecPlaceArray(bjac->y[i],yin+bjac->starts[i]));

    PetscCall(PetscLogEventBegin(PC_ApplyOnBlocks,jac->ksp[i],bjac->x[i],bjac->y[i],0));
    PetscCall(KSPSolve(jac->ksp[i],bjac->x[i],bjac->y[i]));
    PetscCall(KSPCheckSolve(jac->ksp[i],pc,bjac->y[i]));
    PetscCall(PetscLogEventEnd(PC_ApplyOnBlocks,jac->ksp[i],bjac->x[i],bjac->y[i],0));

    PetscCall(VecResetArray(bjac->x[i]));
    PetscCall(VecResetArray(bjac->y[i]));
  }
  PetscCall(VecRestoreArrayRead(x,&xin));
  PetscCall(VecRestoreArray(y,&yin));
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
  PetscCall(VecGetArrayRead(x,&xin));
  PetscCall(VecGetArray(y,&yin));
  for (i=0; i<n_local; i++) {
    /*
       To avoid copying the subvector from x into a workspace we instead
       make the workspace vector array point to the subpart of the array of
       the global vector.
    */
    PetscCall(VecPlaceArray(bjac->x[i],xin+bjac->starts[i]));
    PetscCall(VecPlaceArray(bjac->y[i],yin+bjac->starts[i]));

    PetscCall(PetscLogEventBegin(PC_ApplyTransposeOnBlocks,jac->ksp[i],bjac->x[i],bjac->y[i],0));
    PetscCall(KSPSolveTranspose(jac->ksp[i],bjac->x[i],bjac->y[i]));
    PetscCall(KSPCheckSolve(jac->ksp[i],pc,bjac->y[i]));
    PetscCall(PetscLogEventEnd(PC_ApplyTransposeOnBlocks,jac->ksp[i],bjac->x[i],bjac->y[i],0));

    PetscCall(VecResetArray(bjac->x[i]));
    PetscCall(VecResetArray(bjac->y[i]));
  }
  PetscCall(VecRestoreArrayRead(x,&xin));
  PetscCall(VecRestoreArray(y,&yin));
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
  PetscCall(MatGetLocalSize(pc->pmat,&M,&N));

  n_local = jac->n_local;

  if (pc->useAmat) {
    PetscBool same;
    PetscCall(PetscObjectTypeCompare((PetscObject)mat,((PetscObject)pmat)->type_name,&same));
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

      PetscCall(PetscNewLog(pc,&bjac));
      PetscCall(PetscMalloc1(n_local,&jac->ksp));
      PetscCall(PetscLogObjectMemory((PetscObject)pc,sizeof(n_local*sizeof(KSP))));
      PetscCall(PetscMalloc2(n_local,&bjac->x,n_local,&bjac->y));
      PetscCall(PetscMalloc1(n_local,&bjac->starts));
      PetscCall(PetscLogObjectMemory((PetscObject)pc,sizeof(n_local*sizeof(PetscScalar))));

      jac->data = (void*)bjac;
      PetscCall(PetscMalloc1(n_local,&bjac->is));
      PetscCall(PetscLogObjectMemory((PetscObject)pc,sizeof(n_local*sizeof(IS))));

      for (i=0; i<n_local; i++) {
        PetscCall(KSPCreate(PETSC_COMM_SELF,&ksp));
        PetscCall(KSPSetErrorIfNotConverged(ksp,pc->erroriffailure));
        PetscCall(PetscObjectIncrementTabLevel((PetscObject)ksp,(PetscObject)pc,1));
        PetscCall(PetscLogObjectParent((PetscObject)pc,(PetscObject)ksp));
        PetscCall(KSPSetType(ksp,KSPPREONLY));
        PetscCall(KSPGetPC(ksp,&subpc));
        PetscCall(PCGetOptionsPrefix(pc,&prefix));
        PetscCall(KSPSetOptionsPrefix(ksp,prefix));
        PetscCall(KSPAppendOptionsPrefix(ksp,"sub_"));

        jac->ksp[i] = ksp;
      }
    } else {
      bjac = (PC_BJacobi_Multiblock*)jac->data;
    }

    start = 0;
    PetscCall(MatGetVecType(pmat,&vectype));
    for (i=0; i<n_local; i++) {
      m = jac->l_lens[i];
      /*
      The reason we need to generate these vectors is to serve
      as the right-hand side and solution vector for the solve on the
      block. We do not need to allocate space for the vectors since
      that is provided via VecPlaceArray() just before the call to
      KSPSolve() on the block.

      */
      PetscCall(VecCreateSeq(PETSC_COMM_SELF,m,&x));
      PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF,1,m,NULL,&y));
      PetscCall(VecSetType(x,vectype));
      PetscCall(VecSetType(y,vectype));
      PetscCall(PetscLogObjectParent((PetscObject)pc,(PetscObject)x));
      PetscCall(PetscLogObjectParent((PetscObject)pc,(PetscObject)y));

      bjac->x[i]      = x;
      bjac->y[i]      = y;
      bjac->starts[i] = start;

      PetscCall(ISCreateStride(PETSC_COMM_SELF,m,start,1,&is));
      bjac->is[i] = is;
      PetscCall(PetscLogObjectParent((PetscObject)pc,(PetscObject)is));

      start += m;
    }
  } else {
    bjac = (PC_BJacobi_Multiblock*)jac->data;
    /*
       Destroy the blocks from the previous iteration
    */
    if (pc->flag == DIFFERENT_NONZERO_PATTERN) {
      PetscCall(MatDestroyMatrices(n_local,&bjac->pmat));
      if (pc->useAmat) {
        PetscCall(MatDestroyMatrices(n_local,&bjac->mat));
      }
      scall = MAT_INITIAL_MATRIX;
    } else scall = MAT_REUSE_MATRIX;
  }

  PetscCall(MatCreateSubMatrices(pmat,n_local,bjac->is,bjac->is,scall,&bjac->pmat));
  if (pc->useAmat) {
    PetscCall(MatCreateSubMatrices(mat,n_local,bjac->is,bjac->is,scall,&bjac->mat));
  }
  /* Return control to the user so that the submatrices can be modified (e.g., to apply
     different boundary conditions for the submatrices than for the global problem) */
  PetscCall(PCModifySubMatrices(pc,n_local,bjac->is,bjac->is,bjac->pmat,pc->modifysubmatricesP));

  for (i=0; i<n_local; i++) {
    PetscCall(PetscLogObjectParent((PetscObject)pc,(PetscObject)bjac->pmat[i]));
    PetscCall(KSPGetOptionsPrefix(jac->ksp[i],&prefix));
    if (pc->useAmat) {
      PetscCall(PetscLogObjectParent((PetscObject)pc,(PetscObject)bjac->mat[i]));
      PetscCall(KSPSetOperators(jac->ksp[i],bjac->mat[i],bjac->pmat[i]));
      PetscCall(MatSetOptionsPrefix(bjac->mat[i],prefix));
    } else {
      PetscCall(KSPSetOperators(jac->ksp[i],bjac->pmat[i],bjac->pmat[i]));
    }
    PetscCall(MatSetOptionsPrefix(bjac->pmat[i],prefix));
    if (pc->setfromoptionscalled) {
      PetscCall(KSPSetFromOptions(jac->ksp[i]));
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
  PetscCall(KSPSetUp(subksp));
  PetscCall(KSPGetConvergedReason(subksp,&reason));
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
  PetscCall(VecDestroy(&mpjac->ysub));
  PetscCall(VecDestroy(&mpjac->xsub));
  PetscCall(MatDestroy(&mpjac->submats));
  if (jac->ksp) PetscCall(KSPReset(jac->ksp[0]));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDestroy_BJacobi_Multiproc(PC pc)
{
  PC_BJacobi           *jac   = (PC_BJacobi*)pc->data;
  PC_BJacobi_Multiproc *mpjac = (PC_BJacobi_Multiproc*)jac->data;

  PetscFunctionBegin;
  PetscCall(PCReset_BJacobi_Multiproc(pc));
  PetscCall(KSPDestroy(&jac->ksp[0]));
  PetscCall(PetscFree(jac->ksp));
  PetscCall(PetscSubcommDestroy(&mpjac->psubcomm));

  PetscCall(PetscFree(mpjac));
  PetscCall(PetscFree(pc->data));
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
  PetscCall(VecGetArrayRead(x,&xarray));
  PetscCall(VecGetArray(y,&yarray));
  PetscCall(VecPlaceArray(mpjac->xsub,xarray));
  PetscCall(VecPlaceArray(mpjac->ysub,yarray));

  /* apply preconditioner on each matrix block */
  PetscCall(PetscLogEventBegin(PC_ApplyOnBlocks,jac->ksp[0],mpjac->xsub,mpjac->ysub,0));
  PetscCall(KSPSolve(jac->ksp[0],mpjac->xsub,mpjac->ysub));
  PetscCall(KSPCheckSolve(jac->ksp[0],pc,mpjac->ysub));
  PetscCall(PetscLogEventEnd(PC_ApplyOnBlocks,jac->ksp[0],mpjac->xsub,mpjac->ysub,0));
  PetscCall(KSPGetConvergedReason(jac->ksp[0],&reason));
  if (reason == KSP_DIVERGED_PC_FAILED) {
    pc->failedreason = PC_SUBPC_ERROR;
  }

  PetscCall(VecResetArray(mpjac->xsub));
  PetscCall(VecResetArray(mpjac->ysub));
  PetscCall(VecRestoreArrayRead(x,&xarray));
  PetscCall(VecRestoreArray(y,&yarray));
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
  PetscCall(MatGetLocalSize(X,&m,NULL));
  PetscCall(MatGetSize(X,NULL,&N));
  PetscCall(MatDenseGetLDA(X,&lda));
  PetscCall(MatDenseGetLDA(Y,&ldb));
  PetscCall(MatDenseGetArrayRead(X,&x));
  PetscCall(MatDenseGetArrayWrite(Y,&y));
  PetscCall(MatCreateDense(PetscObjectComm((PetscObject)jac->ksp[0]),m,PETSC_DECIDE,PETSC_DECIDE,N,(PetscScalar*)x,&sX));
  PetscCall(MatCreateDense(PetscObjectComm((PetscObject)jac->ksp[0]),m,PETSC_DECIDE,PETSC_DECIDE,N,y,&sY));
  PetscCall(MatDenseSetLDA(sX,lda));
  PetscCall(MatDenseSetLDA(sY,ldb));
  PetscCall(PetscLogEventBegin(PC_ApplyOnBlocks,jac->ksp[0],X,Y,0));
  PetscCall(KSPMatSolve(jac->ksp[0],sX,sY));
  PetscCall(KSPCheckSolve(jac->ksp[0],pc,NULL));
  PetscCall(PetscLogEventEnd(PC_ApplyOnBlocks,jac->ksp[0],X,Y,0));
  PetscCall(MatDestroy(&sY));
  PetscCall(MatDestroy(&sX));
  PetscCall(MatDenseRestoreArrayWrite(Y,&y));
  PetscCall(MatDenseRestoreArrayRead(X,&x));
  PetscCall(KSPGetConvergedReason(jac->ksp[0],&reason));
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
  PetscCall(PetscObjectGetComm((PetscObject)pc,&comm));
  PetscCheckFalse(jac->n_local > 1,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Only a single block in a subcommunicator is supported");
  jac->n_local = 1; /* currently only a single block is supported for a subcommunicator */
  if (!pc->setupcalled) {
    wasSetup  = PETSC_FALSE;
    PetscCall(PetscNewLog(pc,&mpjac));
    jac->data = (void*)mpjac;

    /* initialize datastructure mpjac */
    if (!jac->psubcomm) {
      /* Create default contiguous subcommunicatiors if user does not provide them */
      PetscCall(PetscSubcommCreate(comm,&jac->psubcomm));
      PetscCall(PetscSubcommSetNumber(jac->psubcomm,jac->n));
      PetscCall(PetscSubcommSetType(jac->psubcomm,PETSC_SUBCOMM_CONTIGUOUS));
      PetscCall(PetscLogObjectMemory((PetscObject)pc,sizeof(PetscSubcomm)));
    }
    mpjac->psubcomm = jac->psubcomm;
    subcomm         = PetscSubcommChild(mpjac->psubcomm);

    /* Get matrix blocks of pmat */
    PetscCall(MatGetMultiProcBlock(pc->pmat,subcomm,MAT_INITIAL_MATRIX,&mpjac->submats));

    /* create a new PC that processors in each subcomm have copy of */
    PetscCall(PetscMalloc1(1,&jac->ksp));
    PetscCall(KSPCreate(subcomm,&jac->ksp[0]));
    PetscCall(KSPSetErrorIfNotConverged(jac->ksp[0],pc->erroriffailure));
    PetscCall(PetscObjectIncrementTabLevel((PetscObject)jac->ksp[0],(PetscObject)pc,1));
    PetscCall(PetscLogObjectParent((PetscObject)pc,(PetscObject)jac->ksp[0]));
    PetscCall(KSPSetOperators(jac->ksp[0],mpjac->submats,mpjac->submats));
    PetscCall(KSPGetPC(jac->ksp[0],&mpjac->pc));

    PetscCall(PCGetOptionsPrefix(pc,&prefix));
    PetscCall(KSPSetOptionsPrefix(jac->ksp[0],prefix));
    PetscCall(KSPAppendOptionsPrefix(jac->ksp[0],"sub_"));
    PetscCall(KSPGetOptionsPrefix(jac->ksp[0],&prefix));
    PetscCall(MatSetOptionsPrefix(mpjac->submats,prefix));

    /* create dummy vectors xsub and ysub */
    PetscCall(MatGetLocalSize(mpjac->submats,&m,&n));
    PetscCall(VecCreateMPIWithArray(subcomm,1,n,PETSC_DECIDE,NULL,&mpjac->xsub));
    PetscCall(VecCreateMPIWithArray(subcomm,1,m,PETSC_DECIDE,NULL,&mpjac->ysub));
    PetscCall(MatGetVecType(mpjac->submats,&vectype));
    PetscCall(VecSetType(mpjac->xsub,vectype));
    PetscCall(VecSetType(mpjac->ysub,vectype));
    PetscCall(PetscLogObjectParent((PetscObject)pc,(PetscObject)mpjac->xsub));
    PetscCall(PetscLogObjectParent((PetscObject)pc,(PetscObject)mpjac->ysub));

    pc->ops->setuponblocks = PCSetUpOnBlocks_BJacobi_Multiproc;
    pc->ops->reset         = PCReset_BJacobi_Multiproc;
    pc->ops->destroy       = PCDestroy_BJacobi_Multiproc;
    pc->ops->apply         = PCApply_BJacobi_Multiproc;
    pc->ops->matapply      = PCMatApply_BJacobi_Multiproc;
  } else { /* pc->setupcalled */
    subcomm = PetscSubcommChild(mpjac->psubcomm);
    if (pc->flag == DIFFERENT_NONZERO_PATTERN) {
      /* destroy old matrix blocks, then get new matrix blocks */
      if (mpjac->submats) PetscCall(MatDestroy(&mpjac->submats));
      PetscCall(MatGetMultiProcBlock(pc->pmat,subcomm,MAT_INITIAL_MATRIX,&mpjac->submats));
    } else {
      PetscCall(MatGetMultiProcBlock(pc->pmat,subcomm,MAT_REUSE_MATRIX,&mpjac->submats));
    }
    PetscCall(KSPSetOperators(jac->ksp[0],mpjac->submats,mpjac->submats));
  }

  if (!wasSetup && pc->setfromoptionscalled) {
    PetscCall(KSPSetFromOptions(jac->ksp[0]));
  }
  PetscFunctionReturn(0);
}
