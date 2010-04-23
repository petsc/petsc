#define PETSCKSP_DLL

/*
   Defines a block Jacobi preconditioner.
*/
#include "private/pcimpl.h"              /*I "petscpc.h" I*/
#include "../src/ksp/pc/impls/bjacobi/bjacobi.h"

static PetscErrorCode PCSetUp_BJacobi_Singleblock(PC,Mat,Mat);
static PetscErrorCode PCSetUp_BJacobi_Multiblock(PC,Mat,Mat);

#undef __FUNCT__  
#define __FUNCT__ "PCSetUp_BJacobi"
static PetscErrorCode PCSetUp_BJacobi(PC pc)
{
  PC_BJacobi     *jac = (PC_BJacobi*)pc->data;
  Mat            mat = pc->mat,pmat = pc->pmat;
  PetscErrorCode ierr,(*f)(Mat,PetscTruth*,MatReuse,Mat*);
  PetscInt       N,M,start,i,sum,end;
  PetscInt       bs,i_start=-1,i_end=-1;
  PetscMPIInt    rank,size;
  const char     *pprefix,*mprefix;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(((PetscObject)pc)->comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(((PetscObject)pc)->comm,&size);CHKERRQ(ierr);
  ierr = MatGetLocalSize(pc->pmat,&M,&N);CHKERRQ(ierr);
  ierr = MatGetBlockSize(pc->pmat,&bs);CHKERRQ(ierr);

  /* ----------
      Determines the number of blocks assigned to each processor 
  */

  /*   local block count  given */
  if (jac->n_local > 0 && jac->n < 0) {
    ierr = MPI_Allreduce(&jac->n_local,&jac->n,1,MPIU_INT,MPI_SUM,((PetscObject)pc)->comm);CHKERRQ(ierr);
    if (jac->l_lens) { /* check that user set these correctly */
      sum = 0;
      for (i=0; i<jac->n_local; i++) {
        if (jac->l_lens[i]/bs*bs !=jac->l_lens[i]) {
          SETERRQ(PETSC_ERR_ARG_SIZ,"Mat blocksize doesn't match block Jacobi layout");
        }
        sum += jac->l_lens[i];
      }
      if (sum != M) SETERRQ(PETSC_ERR_ARG_SIZ,"Local lens sent incorrectly");
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
        if (!jac->g_lens[i]) SETERRQ(PETSC_ERR_ARG_SIZ,"Zero block not allowed");
        if (jac->g_lens[i]/bs*bs != jac->g_lens[i]) {
          SETERRQ(PETSC_ERR_ARG_SIZ,"Mat blocksize doesn't match block Jacobi layout");
        }
      }
      if (size == 1) {
        jac->n_local = jac->n;
        ierr         = PetscMalloc(jac->n_local*sizeof(PetscInt),&jac->l_lens);CHKERRQ(ierr);
        ierr         = PetscMemcpy(jac->l_lens,jac->g_lens,jac->n_local*sizeof(PetscInt));CHKERRQ(ierr);
        /* check that user set these correctly */
        sum = 0;
        for (i=0; i<jac->n_local; i++) sum += jac->l_lens[i];
        if (sum != M) SETERRQ(PETSC_ERR_ARG_SIZ,"Global lens sent incorrectly");
      } else {
        ierr = MatGetOwnershipRange(pc->pmat,&start,&end);CHKERRQ(ierr);
        /* loop over blocks determing first one owned by me */
        sum = 0;
        for (i=0; i<jac->n+1; i++) {
          if (sum == start) { i_start = i; goto start_1;}
          if (i < jac->n) sum += jac->g_lens[i];
        }
        SETERRQ(PETSC_ERR_ARG_SIZ,"Block sizes\n\
                   used in PCBJacobiSetTotalBlocks()\n\
                   are not compatible with parallel matrix layout");
 start_1: 
        for (i=i_start; i<jac->n+1; i++) {
          if (sum == end) { i_end = i; goto end_1; }
          if (i < jac->n) sum += jac->g_lens[i];
        }          
        SETERRQ(PETSC_ERR_ARG_SIZ,"Block sizes\n\
                      used in PCBJacobiSetTotalBlocks()\n\
                      are not compatible with parallel matrix layout");
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
        if (!jac->l_lens[i]) SETERRQ(PETSC_ERR_ARG_SIZ,"Too many blocks given");
      }
    }
  } else if (jac->n < 0 && jac->n_local < 0) { /* no blocks given */
    jac->n         = size;
    jac->n_local   = 1;
    ierr           = PetscMalloc(sizeof(PetscInt),&jac->l_lens);CHKERRQ(ierr);
    jac->l_lens[0] = M;
  }
  if (jac->n_local < 1) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Number of blocks is less than number of processors");

  ierr = MPI_Comm_size(((PetscObject)pc)->comm,&size);CHKERRQ(ierr);
  ierr = PetscObjectQueryFunction((PetscObject)pc->mat,"MatGetDiagonalBlock_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (size == 1 && !f) {
    mat  = pc->mat;
    pmat = pc->pmat;
  } else {
    PetscTruth iscopy;
    MatReuse   scall;

    if (jac->use_true_local) {
      scall = MAT_INITIAL_MATRIX;
      if (pc->setupcalled) {
        if (pc->flag == SAME_NONZERO_PATTERN) {
          if (jac->tp_mat) {
            scall = MAT_REUSE_MATRIX;
            mat   = jac->tp_mat;
          }
        } else {
          if (jac->tp_mat)  {
            ierr = MatDestroy(jac->tp_mat);CHKERRQ(ierr);
          }
        }
      }
      if (!f) {
        SETERRQ(PETSC_ERR_SUP,"This matrix does not support getting diagonal block");
      }
      ierr = (*f)(pc->mat,&iscopy,scall,&mat);CHKERRQ(ierr);
      /* make submatrix have same prefix as entire matrix */
      ierr = PetscObjectGetOptionsPrefix((PetscObject)pc->mat,&mprefix);CHKERRQ(ierr);
      ierr = PetscObjectSetOptionsPrefix((PetscObject)mat,mprefix);CHKERRQ(ierr);
      if (iscopy) {
        jac->tp_mat = mat;
      }
    }
    if (pc->pmat != pc->mat || !jac->use_true_local) {
      scall = MAT_INITIAL_MATRIX;
      if (pc->setupcalled) {
        if (pc->flag == SAME_NONZERO_PATTERN) {
          if (jac->tp_pmat) {
            scall = MAT_REUSE_MATRIX;
            pmat   = jac->tp_pmat;
          }
        } else {
          if (jac->tp_pmat)  {
            ierr = MatDestroy(jac->tp_pmat);CHKERRQ(ierr);
          }
        }
      }
      ierr = PetscObjectQueryFunction((PetscObject)pc->pmat,"MatGetDiagonalBlock_C",(void (**)(void))&f);CHKERRQ(ierr);
      if (!f) {
        const char *type;
        ierr = PetscObjectGetType((PetscObject) pc->pmat,&type);CHKERRQ(ierr);
        SETERRQ1(PETSC_ERR_SUP,"This matrix type, %s, does not support getting diagonal block", type);
      }
      ierr = (*f)(pc->pmat,&iscopy,scall,&pmat);CHKERRQ(ierr);
      /* make submatrix have same prefix as entire matrix */
      ierr = PetscObjectGetOptionsPrefix((PetscObject)pc->pmat,&pprefix);CHKERRQ(ierr);
      ierr = PetscObjectSetOptionsPrefix((PetscObject)pmat,pprefix);CHKERRQ(ierr);
      if (iscopy) {
        jac->tp_pmat = pmat;
      }
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
  ierr = PetscFree(jac);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCSetFromOptions_BJacobi"

static PetscErrorCode PCSetFromOptions_BJacobi(PC pc)
{
  PC_BJacobi     *jac = (PC_BJacobi*)pc->data;
  PetscErrorCode ierr;
  PetscInt       blocks;
  PetscTruth     flg;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("Block Jacobi options");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-pc_bjacobi_blocks","Total number of blocks","PCBJacobiSetTotalBlocks",jac->n,&blocks,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PCBJacobiSetTotalBlocks(pc,blocks,PETSC_NULL);CHKERRQ(ierr); 
    }
    flg  = PETSC_FALSE;
    ierr = PetscOptionsTruth("-pc_bjacobi_truelocal","Use the true matrix, not preconditioner matrix to define matrix vector product in sub-problems","PCBJacobiSetUseTrueLocal",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
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
  PC_BJacobi     *jac = (PC_BJacobi*)pc->data;
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  PetscInt       i;
  PetscTruth     iascii,isstring;
  PetscViewer    sviewer;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_STRING,&isstring);CHKERRQ(ierr);
  if (iascii) {
    if (jac->use_true_local) {
      ierr = PetscViewerASCIIPrintf(viewer,"  block Jacobi: using true local matrix, number of blocks = %D\n",jac->n);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"  block Jacobi: number of blocks = %D\n",jac->n);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(((PetscObject)pc)->comm,&rank);CHKERRQ(ierr);
    if (jac->same_local_solves) {
      ierr = PetscViewerASCIIPrintf(viewer,"  Local solve is same for all blocks, in the following KSP and PC objects:\n");CHKERRQ(ierr);
      ierr = PetscViewerGetSingleton(viewer,&sviewer);CHKERRQ(ierr);
      if (!rank && jac->ksp) {
        ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
        ierr = KSPView(jac->ksp[0],sviewer);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
      }   
      ierr = PetscViewerRestoreSingleton(viewer,&sviewer);CHKERRQ(ierr);
    } else {
      PetscInt n_global; 
      ierr = MPI_Allreduce(&jac->n_local,&n_global,1,MPIU_INT,MPI_MAX,((PetscObject)pc)->comm);CHKERRQ(ierr);
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
    }
  } else if (isstring) {
    ierr = PetscViewerStringSPrintf(viewer," blks=%D",jac->n);CHKERRQ(ierr);
    ierr = PetscViewerGetSingleton(viewer,&sviewer);CHKERRQ(ierr);
    if (jac->ksp) {ierr = KSPView(jac->ksp[0],sviewer);CHKERRQ(ierr);}
    ierr = PetscViewerRestoreSingleton(viewer,&sviewer);CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported for block Jacobi",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------------*/  

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCBJacobiSetUseTrueLocal_BJacobi"
PetscErrorCode PETSCKSP_DLLEXPORT PCBJacobiSetUseTrueLocal_BJacobi(PC pc)
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
PetscErrorCode PETSCKSP_DLLEXPORT PCBJacobiGetSubKSP_BJacobi(PC pc,PetscInt *n_local,PetscInt *first_local,KSP **ksp)
{
  PC_BJacobi   *jac = (PC_BJacobi*)pc->data;;

  PetscFunctionBegin;
  if (!pc->setupcalled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Must call KSPSetUp() or PCSetUp() first");

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
PetscErrorCode PETSCKSP_DLLEXPORT PCBJacobiSetTotalBlocks_BJacobi(PC pc,PetscInt blocks,PetscInt *lens)
{
  PC_BJacobi     *jac = (PC_BJacobi*)pc->data; 
  PetscErrorCode ierr;

  PetscFunctionBegin;

  if (pc->setupcalled > 0 && jac->n!=blocks) SETERRQ(PETSC_ERR_ORDER,"Cannot alter number of blocks after PCSetUp()/KSPSetUp() has been called"); 
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
PetscErrorCode PETSCKSP_DLLEXPORT PCBJacobiGetTotalBlocks_BJacobi(PC pc, PetscInt *blocks, const PetscInt *lens[])
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
PetscErrorCode PETSCKSP_DLLEXPORT PCBJacobiSetLocalBlocks_BJacobi(PC pc,PetscInt blocks,const PetscInt lens[])
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
PetscErrorCode PETSCKSP_DLLEXPORT PCBJacobiGetLocalBlocks_BJacobi(PC pc, PetscInt *blocks, const PetscInt *lens[])
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

   Collective on PC

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
PetscErrorCode PETSCKSP_DLLEXPORT PCBJacobiSetUseTrueLocal(PC pc)
{
  PetscErrorCode ierr,(*f)(PC);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCBJacobiSetUseTrueLocal_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc);CHKERRQ(ierr);
  } 

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

   Level: advanced

.keywords:  block, Jacobi, get, sub, KSP, context

.seealso: PCBJacobiGetSubKSP()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCBJacobiGetSubKSP(PC pc,PetscInt *n_local,PetscInt *first_local,KSP *ksp[])
{
  PetscErrorCode ierr,(*f)(PC,PetscInt *,PetscInt *,KSP **);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCBJacobiGetSubKSP_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,n_local,first_local,ksp);CHKERRQ(ierr);
  } else {
    SETERRQ(PETSC_ERR_ARG_WRONG,"Cannot get subsolvers for this preconditioner");
  }
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
PetscErrorCode PETSCKSP_DLLEXPORT PCBJacobiSetTotalBlocks(PC pc,PetscInt blocks,const PetscInt lens[])
{
  PetscErrorCode ierr,(*f)(PC,PetscInt,const PetscInt[]);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  if (blocks <= 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Must have positive blocks");
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCBJacobiSetTotalBlocks_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,blocks,lens);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCBJacobiGetTotalBlocks"
/*@C
   PCBJacobiGetTotalBlocks - Gets the global number of blocks for the block
   Jacobi preconditioner.

   Collective on PC

   Input Parameter:
.  pc - the preconditioner context

   Output parameters:
+  blocks - the number of blocks
-  lens - integer array containing the size of each block

   Level: intermediate

.keywords:  get, number, Jacobi, global, total, blocks

.seealso: PCBJacobiSetUseTrueLocal(), PCBJacobiGetLocalBlocks()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCBJacobiGetTotalBlocks(PC pc, PetscInt *blocks, const PetscInt *lens[])
{
  PetscErrorCode ierr,(*f)(PC,PetscInt*, const PetscInt *[]);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_COOKIE,1);
  PetscValidIntPointer(blocks,2);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCBJacobiGetTotalBlocks_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,blocks,lens);CHKERRQ(ierr);
  } 
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
PetscErrorCode PETSCKSP_DLLEXPORT PCBJacobiSetLocalBlocks(PC pc,PetscInt blocks,const PetscInt lens[])
{
  PetscErrorCode ierr,(*f)(PC,PetscInt,const PetscInt []);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  if (blocks < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Must have nonegative blocks");
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCBJacobiSetLocalBlocks_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,blocks,lens);CHKERRQ(ierr);
  } 
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
PetscErrorCode PETSCKSP_DLLEXPORT PCBJacobiGetLocalBlocks(PC pc, PetscInt *blocks, const PetscInt *lens[])
{
  PetscErrorCode ierr,(*f)(PC,PetscInt*, const PetscInt *[]);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_COOKIE,1);
  PetscValidIntPointer(blocks,2);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCBJacobiGetLocalBlocks_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,blocks,lens);CHKERRQ(ierr);
  } 
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

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC,
           PCASM, PCBJacobiSetUseTrueLocal(), PCBJacobiGetSubKSP(), PCBJacobiSetTotalBlocks(),
           PCBJacobiSetLocalBlocks(), PCSetModifySubmatrices()
M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCreate_BJacobi"
PetscErrorCode PETSCKSP_DLLEXPORT PCCreate_BJacobi(PC pc)
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
  jac->ksp              = 0;
  jac->use_true_local    = PETSC_FALSE;
  jac->same_local_solves = PETSC_TRUE;
  jac->g_lens            = 0;
  jac->l_lens            = 0;
  jac->tp_mat            = 0;
  jac->tp_pmat           = 0;

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
#define __FUNCT__ "PCDestroy_BJacobi_Singleblock"
PetscErrorCode PCDestroy_BJacobi_Singleblock(PC pc)
{
  PC_BJacobi             *jac = (PC_BJacobi*)pc->data;
  PC_BJacobi_Singleblock *bjac = (PC_BJacobi_Singleblock*)jac->data;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  /*
        If the on processor block had to be generated via a MatGetDiagonalBlock()
     that creates a copy, this frees the space
  */
  if (jac->tp_mat) {
    ierr = MatDestroy(jac->tp_mat);CHKERRQ(ierr);
  }
  if (jac->tp_pmat) {
    ierr = MatDestroy(jac->tp_pmat);CHKERRQ(ierr);
  }

  ierr = KSPDestroy(jac->ksp[0]);CHKERRQ(ierr);
  ierr = PetscFree(jac->ksp);CHKERRQ(ierr);
  ierr = VecDestroy(bjac->x);CHKERRQ(ierr);
  ierr = VecDestroy(bjac->y);CHKERRQ(ierr);
  ierr = PetscFree(jac->l_lens);CHKERRQ(ierr);
  ierr = PetscFree(jac->g_lens);CHKERRQ(ierr);
  ierr = PetscFree(bjac);CHKERRQ(ierr);
  ierr = PetscFree(jac);CHKERRQ(ierr);
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
  Vec                    x,y;
  PC_BJacobi_Singleblock *bjac;
  PetscTruth             wasSetup;

  PetscFunctionBegin;

  /* set default direct solver with no Krylov method */
  if (!pc->setupcalled) {
    const char *prefix;
    wasSetup = PETSC_FALSE;
    ierr = KSPCreate(PETSC_COMM_SELF,&ksp);CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)ksp,(PetscObject)pc,1);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(pc,ksp);CHKERRQ(ierr);
    ierr = KSPSetType(ksp,KSPPREONLY);CHKERRQ(ierr);
    ierr = PCGetOptionsPrefix(pc,&prefix);CHKERRQ(ierr);
    ierr = KSPSetOptionsPrefix(ksp,prefix);CHKERRQ(ierr);
    ierr = KSPAppendOptionsPrefix(ksp,"sub_");CHKERRQ(ierr);
    /*
      The reason we need to generate these vectors is to serve 
      as the right-hand side and solution vector for the solve on the 
      block. We do not need to allocate space for the vectors since
      that is provided via VecPlaceArray() just before the call to 
      KSPSolve() on the block.
    */
    ierr = MatGetSize(pmat,&m,&m);CHKERRQ(ierr);
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,m,PETSC_NULL,&x);CHKERRQ(ierr);
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,m,PETSC_NULL,&y);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(pc,x);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(pc,y);CHKERRQ(ierr);

    pc->ops->destroy             = PCDestroy_BJacobi_Singleblock;
    pc->ops->apply               = PCApply_BJacobi_Singleblock;
    pc->ops->applysymmetricleft  = PCApplySymmetricLeft_BJacobi_Singleblock;
    pc->ops->applysymmetricright = PCApplySymmetricRight_BJacobi_Singleblock;
    pc->ops->applytranspose      = PCApplyTranspose_BJacobi_Singleblock;
    pc->ops->setuponblocks       = PCSetUpOnBlocks_BJacobi_Singleblock;

    ierr = PetscNewLog(pc,PC_BJacobi_Singleblock,&bjac);CHKERRQ(ierr);
    bjac->x      = x;
    bjac->y      = y;

    ierr = PetscMalloc(sizeof(KSP),&jac->ksp);CHKERRQ(ierr);
    jac->ksp[0] = ksp;
    jac->data    = (void*)bjac;
  } else {
    wasSetup = PETSC_TRUE;
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
#define __FUNCT__ "PCDestroy_BJacobi_Multiblock"
PetscErrorCode PCDestroy_BJacobi_Multiblock(PC pc)
{
  PC_BJacobi            *jac = (PC_BJacobi*)pc->data;
  PC_BJacobi_Multiblock *bjac = (PC_BJacobi_Multiblock*)jac->data;
  PetscErrorCode        ierr;
  PetscInt              i;

  PetscFunctionBegin;
  ierr = MatDestroyMatrices(jac->n_local,&bjac->pmat);CHKERRQ(ierr);
  if (jac->use_true_local) {
    ierr = MatDestroyMatrices(jac->n_local,&bjac->mat);CHKERRQ(ierr);
  }

  /*
        If the on processor block had to be generated via a MatGetDiagonalBlock()
     that creates a copy, this frees the space
  */
  if (jac->tp_mat) {
    ierr = MatDestroy(jac->tp_mat);CHKERRQ(ierr);
  }
  if (jac->tp_pmat) {
    ierr = MatDestroy(jac->tp_pmat);CHKERRQ(ierr);
  }

  for (i=0; i<jac->n_local; i++) {
    ierr = KSPDestroy(jac->ksp[i]);CHKERRQ(ierr);
    ierr = VecDestroy(bjac->x[i]);CHKERRQ(ierr);
    ierr = VecDestroy(bjac->y[i]);CHKERRQ(ierr);
    ierr = ISDestroy(bjac->is[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(jac->ksp);CHKERRQ(ierr);
  ierr = PetscFree2(bjac->x,bjac->y);CHKERRQ(ierr);
  ierr = PetscFree(bjac->starts);CHKERRQ(ierr);
  ierr = PetscFree(bjac->is);CHKERRQ(ierr);
  ierr = PetscFree(bjac);CHKERRQ(ierr);
  ierr = PetscFree(jac->l_lens);CHKERRQ(ierr);
  ierr = PetscFree(jac->g_lens);CHKERRQ(ierr);
  ierr = PetscFree(jac);CHKERRQ(ierr);
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

    ierr = PetscLogEventBegin(PC_SetUpOnBlocks,jac->ksp[i],bjac->x[i],bjac->y[i],0);CHKERRQ(ierr);
    ierr = KSPSolve(jac->ksp[i],bjac->x[i],bjac->y[i]);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(PC_SetUpOnBlocks,jac->ksp[i],bjac->x[i],bjac->y[i],0);CHKERRQ(ierr);

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
  MatReuse               scall = MAT_REUSE_MATRIX;

  PetscFunctionBegin;
  ierr = MatGetLocalSize(pc->pmat,&M,&N);CHKERRQ(ierr);

  n_local = jac->n_local;

  if (jac->use_true_local) {
    PetscTruth same;
    ierr = PetscTypeCompare((PetscObject)mat,((PetscObject)pmat)->type_name,&same);CHKERRQ(ierr);
    if (!same) SETERRQ(PETSC_ERR_ARG_INCOMP,"Matrices not of same type");
  }

  if (!pc->setupcalled) {
    scall                  = MAT_INITIAL_MATRIX;
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

    start = 0;
    for (i=0; i<n_local; i++) {
      ierr = KSPCreate(PETSC_COMM_SELF,&ksp);CHKERRQ(ierr);
      ierr = PetscObjectIncrementTabLevel((PetscObject)ksp,(PetscObject)pc,1);CHKERRQ(ierr);
      ierr = PetscLogObjectParent(pc,ksp);CHKERRQ(ierr);
      ierr = KSPSetType(ksp,KSPPREONLY);CHKERRQ(ierr);
      ierr = KSPGetPC(ksp,&subpc);CHKERRQ(ierr);
      ierr = PCGetOptionsPrefix(pc,&prefix);CHKERRQ(ierr);
      ierr = KSPSetOptionsPrefix(ksp,prefix);CHKERRQ(ierr);
      ierr = KSPAppendOptionsPrefix(ksp,"sub_");CHKERRQ(ierr);

      m = jac->l_lens[i];

      /*
      The reason we need to generate these vectors is to serve 
      as the right-hand side and solution vector for the solve on the 
      block. We do not need to allocate space for the vectors since
      that is provided via VecPlaceArray() just before the call to 
      KSPSolve() on the block.

      */
      ierr = VecCreateSeq(PETSC_COMM_SELF,m,&x);CHKERRQ(ierr);
      ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,m,PETSC_NULL,&y);CHKERRQ(ierr);
      ierr = PetscLogObjectParent(pc,x);CHKERRQ(ierr);
      ierr = PetscLogObjectParent(pc,y);CHKERRQ(ierr);
      bjac->x[i]      = x;
      bjac->y[i]      = y;
      bjac->starts[i] = start;
      jac->ksp[i]    = ksp;

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
    if(pc->setfromoptionscalled){
      ierr = KSPSetFromOptions(jac->ksp[i]);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}
