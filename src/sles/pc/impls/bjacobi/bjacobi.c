#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: bjacobi.c,v 1.126 1999/02/19 17:02:30 bsmith Exp bsmith $";
#endif
/*
   Defines a block Jacobi preconditioner.
*/
#include "src/mat/matimpl.h"
#include "src/sles/pc/pcimpl.h"              /*I "pc.h" I*/
#include "src/sles/pc/impls/bjacobi/bjacobi.h"

static int PCSetUp_BJacobi_Singleblock(PC,Mat,Mat);
static int PCSetUp_BJacobi_Multiblock(PC,Mat,Mat);

#undef __FUNC__  
#define __FUNC__ "PCSetUp_BJacobi"
static int PCSetUp_BJacobi(PC pc)
{
  PC_BJacobi      *jac = (PC_BJacobi *) pc->data;
  Mat             mat = pc->mat, pmat = pc->pmat;
  int             ierr, N, M, start, i, rank, size,sum, end;
  int             bs, i_start=-1, i_end=-1;

  PetscFunctionBegin;
  MPI_Comm_rank(pc->comm,&rank);
  MPI_Comm_size(pc->comm,&size);
  ierr = MatGetLocalSize(pc->pmat,&M,&N); CHKERRQ(ierr);
  ierr = MatGetBlockSize(pc->pmat,&bs); CHKERRQ(ierr);

  /* ----------
      Determines the number of blocks assigned to each processor 
  */

  /*   local block count  given */
  if (jac->n_local > 0 && jac->n < 0) {
    ierr = MPI_Allreduce(&jac->n_local,&jac->n,1,MPI_INT,MPI_SUM,pc->comm);CHKERRQ(ierr);
    if (jac->l_lens) { /* check that user set these correctly */
      sum = 0;
      for (i=0; i<jac->n_local; i++) {
        if (jac->l_lens[i]/bs*bs !=jac->l_lens[i]) {
          SETERRQ(PETSC_ERR_ARG_SIZ,0,"Mat blocksize doesn't match block Jacobi layout");
        }
        sum += jac->l_lens[i];
      }
      if (sum != M) SETERRQ(PETSC_ERR_ARG_SIZ,0,"Local lens sent incorrectly");
    }
  } else if (jac->n > 0 && jac->n_local < 0) { /* global block count given */
    /* global blocks given: determine which ones are local */
    if (jac->g_lens) {
      /* check if the g_lens is has valid entries */
      for (i=0; i<jac->n; i++) {
        if (!jac->g_lens[i]) SETERRQ(PETSC_ERR_ARG_SIZ,0,"Zero block not allowed");
        if (jac->g_lens[i]/bs*bs != jac->g_lens[i]) {
          SETERRQ(PETSC_ERR_ARG_SIZ,0,"Mat blocksize doesn't match block Jacobi layout");
        }
      }
      if (size == 1) {
        jac->n_local = jac->n;
        jac->l_lens  = (int *) PetscMalloc(jac->n_local*sizeof(int));CHKPTRQ(jac->l_lens);
        PetscMemcpy(jac->l_lens,jac->g_lens,jac->n_local*sizeof(int));
        /* check that user set these correctly */
        sum = 0;
        for (i=0; i<jac->n_local; i++) sum += jac->l_lens[i];
        if (sum != M) SETERRQ(PETSC_ERR_ARG_SIZ,0,"Global lens sent incorrectly");
      } else {
        ierr = MatGetOwnershipRange(pc->pmat,&start,&end);CHKERRQ(ierr);
        /* loop over blocks determing first one owned by me */
        sum = 0;
        for (i=0; i<jac->n+1; i++) {
          if (sum == start) { i_start = i; goto start_1;}
          if (i < jac->n) sum += jac->g_lens[i];
        }
        SETERRQ(PETSC_ERR_ARG_SIZ,0,"Block sizes\n\
                   used in PCBJacobiSetTotalBlocks()\n\
                   are not compatible with parallel matrix layout");
 start_1: 
        for (i=i_start; i<jac->n+1; i++) {
          if (sum == end) { i_end = i; goto end_1; }
          if (i < jac->n) sum += jac->g_lens[i];
        }          
        SETERRQ(PETSC_ERR_ARG_SIZ,0,"Block sizes\n\
                      used in PCBJacobiSetTotalBlocks()\n\
                      are not compatible with parallel matrix layout");
 end_1: 
        jac->n_local = i_end - i_start;
        jac->l_lens = (int *) PetscMalloc(jac->n_local*sizeof(int));CHKPTRQ(jac->l_lens); 
        PetscMemcpy(jac->l_lens,jac->g_lens+i_start,jac->n_local*sizeof(int));
      }
    } else { /* no global blocks given, determine then using default layout */
      jac->n_local = jac->n/size + ((jac->n % size) > rank);
      jac->l_lens  = (int *) PetscMalloc(jac->n_local*sizeof(int));CHKPTRQ(jac->l_lens);
      for (i=0; i<jac->n_local; i++) {
        jac->l_lens[i] = ((M/bs)/jac->n_local + (((M/bs) % jac->n_local) > i))*bs;
        if (!jac->l_lens[i]) SETERRQ(PETSC_ERR_ARG_SIZ,0,"Too many blocks given");
      }
    }
  } else if (jac->n < 0 && jac->n_local < 0) { /* no blocks given */
    jac->n         = size;
    jac->n_local   = 1;
    jac->l_lens    = (int *) PetscMalloc(sizeof(int));CHKPTRQ(jac->l_lens);
    jac->l_lens[0] = M;
  }

  MPI_Comm_size(pc->comm,&size);
  if (size == 1) {
    mat  = pc->mat;
    pmat = pc->pmat;
  } else {
    PetscTruth iscopy;
    MatReuse   scall;
    int        (*f)(Mat,PetscTruth*,MatReuse,Mat*);

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
      ierr = PetscObjectQueryFunction((PetscObject)pc->mat,"MatGetDiagonalBlock_C",(void**)&f);CHKERRQ(ierr);
      if (!f) {
        SETERRQ(PETSC_ERR_SUP,0,"This matrix does not support getting diagonal block");
      }
      ierr = (*f)(pc->mat,&iscopy,scall,&mat);CHKERRQ(ierr);
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
      ierr = PetscObjectQueryFunction((PetscObject)pc->pmat,"MatGetDiagonalBlock_C",(void**)&f);CHKERRQ(ierr);
      if (!f) {
        SETERRQ(PETSC_ERR_SUP,0,"This matrix does not support getting diagonal block");
      }
      ierr = (*f)(pc->pmat,&iscopy,scall,&pmat);CHKERRQ(ierr);
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
#undef __FUNC__  
#define __FUNC__ "PCDestroy_BJacobi"
static int PCDestroy_BJacobi(PC pc)
{
  PC_BJacobi *jac = (PC_BJacobi *) pc->data;

  PetscFunctionBegin;
  if (jac->g_lens) PetscFree(jac->g_lens);
  if (jac->l_lens) PetscFree(jac->l_lens);
  PetscFree(jac);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCSetFromOptions_BJacobi"
static int PCSetFromOptions_BJacobi(PC pc)
{
  int        blocks,flg,ierr;

  PetscFunctionBegin;
  ierr = OptionsGetInt(pc->prefix,"-pc_bjacobi_blocks",&blocks,&flg); CHKERRQ(ierr);
  if (flg) {
    ierr = PCBJacobiSetTotalBlocks(pc,blocks,PETSC_NULL); CHKERRQ(ierr); 
  }
  ierr = OptionsHasName(pc->prefix,"-pc_bjacobi_truelocal",&flg); CHKERRQ(ierr);
  if (flg) {
    ierr = PCBJacobiSetUseTrueLocal(pc); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCPrintHelp_BJacobi"
static int PCPrintHelp_BJacobi(PC pc,char *p)
{
  PetscFunctionBegin;
  (*PetscHelpPrintf)(pc->comm," Options for PCBJACOBI preconditioner:\n");
  (*PetscHelpPrintf)(pc->comm," %spc_bjacobi_blocks <blks>: total blocks in preconditioner\n",p);
  (*PetscHelpPrintf)(pc->comm, " %spc_bjacobi_truelocal: use blocks from the local linear\
 system matrix \n      instead of the preconditioning matrix\n",p);
  (*PetscHelpPrintf)(pc->comm," %ssub : prefix to control options for individual blocks.\
 Add before the \n      usual KSP and PC option names (e.g., %ssub_ksp_type\
 <kspmethod>)\n",p,p);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCView_BJacobi"
static int PCView_BJacobi(PC pc,Viewer viewer)
{
  PC_BJacobi       *jac = (PC_BJacobi *) pc->data;
  int              rank, ierr, i;
  ViewerType       vtype;

  PetscFunctionBegin;
  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  if (PetscTypeCompare(vtype,ASCII_VIEWER)) {
    if (jac->use_true_local) {
      ViewerASCIIPrintf(viewer,"  block Jacobi: using true local matrix, number of blocks = %d\n", jac->n);
    }
    ViewerASCIIPrintf(viewer,"  block Jacobi: number of blocks = %d\n", jac->n);
    MPI_Comm_rank(pc->comm,&rank);
    if (jac->same_local_solves) {
      ViewerASCIIPrintf(viewer,"  Local solve is same for all blocks, in the following KSP and PC objects:\n");
      if (!rank && jac->sles) {
        ierr = ViewerASCIIPushTab(viewer);CHKERRQ(ierr);
        ierr = SLESView(jac->sles[0],viewer); CHKERRQ(ierr);
        ierr = ViewerASCIIPopTab(viewer);CHKERRQ(ierr);
      }   
    } else {
      FILE *fd;

      ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
      ViewerASCIIPrintf(viewer,"  Local solve info for each block is in the following KSP and PC objects:\n");
      PetscSequentialPhaseBegin(pc->comm,1);
      PetscFPrintf(PETSC_COMM_SELF,fd,"Proc %d: number of local blocks = %d, first local block number = %d\n",
                   rank,jac->n_local,jac->first_local);
      for (i=0; i<jac->n_local; i++) {
        PetscFPrintf(PETSC_COMM_SELF,fd,"Proc %d: local block number %d\n",rank,i);
           /* This shouldn't really be STDOUT */
        ierr = SLESView(jac->sles[i],VIEWER_STDOUT_SELF); CHKERRQ(ierr);
        if (i != jac->n_local-1) PetscFPrintf(PETSC_COMM_SELF,fd,"- - - - - - - - - - - - - - - - - -\n");
      }
      fflush(fd);
      PetscSequentialPhaseEnd(pc->comm,1);
    }
  } else if (PetscTypeCompare(vtype,STRING_VIEWER)) {
    ierr = ViewerStringSPrintf(viewer," blks=%d",jac->n);CHKERRQ(ierr);
    if (jac->sles) {ierr = SLESView(jac->sles[0],viewer); CHKERRQ(ierr);}
  } else {
    SETERRQ(1,1,"Viewer type not supported for this object");
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------------*/  

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "PCBJacobiSetUseTrueLocal_BJacobi"
int PCBJacobiSetUseTrueLocal_BJacobi(PC pc)
{
  PC_BJacobi   *jac;

  PetscFunctionBegin;
  jac                 = (PC_BJacobi *) pc->data;
  jac->use_true_local = 1;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "PCBJacobiGetSubSLES_BJacobi"
int PCBJacobiGetSubSLES_BJacobi(PC pc,int *n_local,int *first_local,SLES **sles)
{
  PC_BJacobi   *jac;

  PetscFunctionBegin;
  if (!pc->setupcalled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Must call SLESSetUp first");

  jac                           = (PC_BJacobi *) pc->data;
  if (n_local) *n_local         = jac->n_local;
  if (first_local) *first_local = jac->first_local;
  *sles                         = jac->sles;
  jac->same_local_solves        = 0; /* Assume that local solves are now different;
                                     not necessarily true though!  This flag is 
                                     used only for PCView_BJacobi() */
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "PCBJacobiSetTotalBlocks_BJacobi"
int PCBJacobiSetTotalBlocks_BJacobi(PC pc, int blocks,int *lens)
{
  PC_BJacobi *jac = (PC_BJacobi *) pc->data; 

  PetscFunctionBegin;

  jac->n = blocks;
  if (!lens) {
    jac->g_lens = 0;
  } else {
    jac->g_lens = (int *) PetscMalloc(blocks*sizeof(int)); CHKPTRQ(jac->g_lens);
    PLogObjectMemory(pc,blocks*sizeof(int));
    PetscMemcpy(jac->g_lens,lens,blocks*sizeof(int));
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "PCBJacobiSetLocalBlocks_BJacobi"
int PCBJacobiSetLocalBlocks_BJacobi(PC pc, int blocks,int *lens)
{
  PC_BJacobi *jac;

  PetscFunctionBegin;
  jac = (PC_BJacobi *) pc->data; 

  jac->n_local = blocks;
  if (!lens) {
    jac->l_lens = 0;
  } else {
    jac->l_lens = (int *) PetscMalloc(blocks*sizeof(int)); CHKPTRQ(jac->l_lens);
    PLogObjectMemory(pc,blocks*sizeof(int));
    PetscMemcpy(jac->l_lens,lens,blocks*sizeof(int));
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

/* -------------------------------------------------------------------------------------*/  

#undef __FUNC__  
#define __FUNC__ "PCBJacobiSetUseTrueLocal"
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
int PCBJacobiSetUseTrueLocal(PC pc)
{
  int ierr, (*f)(PC);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCBJacobiSetUseTrueLocal_C",(void **)&f); CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc);CHKERRQ(ierr);
  } 

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCBJacobiGetSubSLES"
/*@C
   PCBJacobiGetSubSLES - Gets the local SLES contexts for all blocks on
   this processor.
   
   Note Collective

   Input Parameter:
.  pc - the preconditioner context

   Output Parameters:
+  n_local - the number of blocks on this processor, or PETSC_NULL
.  first_local - the global number of the first block on this processor, or PETSC_NULL
-  sles - the array of SLES contexts

   Notes:  
   After PCBJacobiGetSubSLES() the array of SLES contexts is not to be freed.
   
   Currently for some matrix implementations only 1 block per processor 
   is supported.
   
   You must call SLESSetUp() before calling PCBJacobiGetSubSLES().

   Level: advanced

.keywords:  block, Jacobi, get, sub, SLES, context

.seealso: PCBJacobiGetSubSLES()
@*/
int PCBJacobiGetSubSLES(PC pc,int *n_local,int *first_local,SLES **sles)
{
  int ierr, (*f)(PC,int *,int *,SLES **);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  PetscValidIntPointer(n_local);
  PetscValidIntPointer(first_local);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCBJacobiGetSubSLES_C",(void **)&f); CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,n_local,first_local,sles);CHKERRQ(ierr);
  } else {
    SETERRQ(1,1,"Cannot get subsolvers for this preconditioner");
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCBJacobiSetTotalBlocks"
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
int PCBJacobiSetTotalBlocks(PC pc, int blocks,int *lens)
{
  int ierr, (*f)(PC,int,int *);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  if (blocks <= 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Must have positive blocks");
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCBJacobiSetTotalBlocks_C",(void **)&f); CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,blocks,lens);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}
  
#undef __FUNC__  
#define __FUNC__ "PCBJacobiSetLocalBlocks"
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
int PCBJacobiSetLocalBlocks(PC pc, int blocks,int *lens)
{
  int ierr, (*f)(PC,int ,int *);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  if (blocks < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Must have nonegative blocks");
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCBJacobiSetLocalBlocks_C",(void **)&f); CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,blocks,lens);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

/* -----------------------------------------------------------------------------------*/

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "PCCreate_BJacobi"
int PCCreate_BJacobi(PC pc)
{
  int          rank,size,ierr;
  PC_BJacobi   *jac = PetscNew(PC_BJacobi); CHKPTRQ(jac);

  PetscFunctionBegin;
  PLogObjectMemory(pc,sizeof(PC_BJacobi));
  MPI_Comm_rank(pc->comm,&rank);
  MPI_Comm_size(pc->comm,&size);
  pc->apply              = 0;
  pc->applytrans         = 0;
  pc->setup              = PCSetUp_BJacobi;
  pc->destroy            = PCDestroy_BJacobi;
  pc->setfromoptions     = PCSetFromOptions_BJacobi;
  pc->printhelp          = PCPrintHelp_BJacobi;
  pc->view               = PCView_BJacobi;
  pc->applyrich          = 0;
  pc->data               = (void *) jac;
  jac->n                 = -1;
  jac->n_local           = -1;
  jac->first_local       = rank;
  jac->sles              = 0;
  jac->use_true_local    = 0;
  jac->same_local_solves = 1;
  jac->g_lens            = 0;
  jac->l_lens            = 0;
  jac->tp_mat            = 0;
  jac->tp_pmat           = 0;

  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBJacobiSetUseTrueLocal_C",
                    "PCBJacobiSetUseTrueLocal_BJacobi",
                    (void*)PCBJacobiSetUseTrueLocal_BJacobi);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBJacobiGetSubSLES_C","PCBJacobiGetSubSLES_BJacobi",
                    (void*)PCBJacobiGetSubSLES_BJacobi);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBJacobiSetTotalBlocks_C","PCBJacobiSetTotalBlocks_BJacobi",
                    (void*)PCBJacobiSetTotalBlocks_BJacobi);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBJacobiSetLocalBlocks_C","PCBJacobiSetLocalBlocks_BJacobi",
                    (void*)PCBJacobiSetLocalBlocks_BJacobi);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END

/* --------------------------------------------------------------------------------------------*/
/*
        These are for a single block per processor; works for AIJ, BAIJ; Seq and MPI
*/
#undef __FUNC__  
#define __FUNC__ "PCDestroy_BJacobi_Singleblock"
int PCDestroy_BJacobi_Singleblock(PC pc)
{
  PC_BJacobi             *jac = (PC_BJacobi *) pc->data;
  PC_BJacobi_Singleblock *bjac = (PC_BJacobi_Singleblock *) jac->data;
  int                    ierr;

  PetscFunctionBegin;
  /*
        If the on processor block had to be generated via a MatGetDiagonalBlock()
     that creates a copy (for example MPIBDiag matrices do), this frees the space
  */
  if (jac->tp_mat) {
    ierr = MatDestroy(jac->tp_mat);CHKERRQ(ierr);
  }
  if (jac->tp_pmat) {
    ierr = MatDestroy(jac->tp_pmat);CHKERRQ(ierr);
  }

  ierr = SLESDestroy(jac->sles[0]); CHKERRQ(ierr);
  PetscFree(jac->sles);
  ierr = VecDestroy(bjac->x); CHKERRQ(ierr);
  ierr = VecDestroy(bjac->y); CHKERRQ(ierr);
  if (jac->l_lens) PetscFree(jac->l_lens);
  if (jac->g_lens) PetscFree(jac->g_lens);
  PetscFree(bjac); PetscFree(jac); 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCSetUpOnBlocks_BJacobi_Singleblock"
int PCSetUpOnBlocks_BJacobi_Singleblock(PC pc)
{
  int                    ierr;
  PC_BJacobi             *jac = (PC_BJacobi *) pc->data;
  PC_BJacobi_Singleblock *bjac = (PC_BJacobi_Singleblock *) jac->data;

  PetscFunctionBegin;
  ierr = SLESSetUp(jac->sles[0],bjac->x,bjac->y); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCApply_BJacobi_Singleblock"
int PCApply_BJacobi_Singleblock(PC pc,Vec x, Vec y)
{
  int                    ierr,its;
  PC_BJacobi             *jac = (PC_BJacobi *) pc->data;
  PC_BJacobi_Singleblock *bjac = (PC_BJacobi_Singleblock *) jac->data;
  Scalar                 *x_array,*y_array;

  PetscFunctionBegin;
  /* 
      The VecPlaceArray() is to avoid having to copy the 
    y vector into the bjac->x vector. The reason for 
    the bjac->x vector is that we need a sequential vector
    for the sequential solve.
  */
  ierr = VecGetArray(x,&x_array); CHKERRQ(ierr); 
  ierr = VecGetArray(y,&y_array); CHKERRQ(ierr); 
  ierr = VecPlaceArray(bjac->x,x_array); CHKERRQ(ierr); 
  ierr = VecPlaceArray(bjac->y,y_array); CHKERRQ(ierr); 
  ierr = SLESSolve(jac->sles[0],bjac->x,bjac->y,&its); CHKERRQ(ierr); 
  ierr = VecRestoreArray(x,&x_array); CHKERRQ(ierr); 
  ierr = VecRestoreArray(y,&y_array); CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCApplyTrans_BJacobi_Singleblock"
int PCApplyTrans_BJacobi_Singleblock(PC pc,Vec x, Vec y)
{
  int                    ierr,its;
  PC_BJacobi             *jac = (PC_BJacobi *) pc->data;
  PC_BJacobi_Singleblock *bjac = (PC_BJacobi_Singleblock *) jac->data;
  Scalar                 *x_array, *y_array;

  PetscFunctionBegin;
  /* 
      The VecPlaceArray() is to avoid having to copy the 
    y vector into the bjac->x vector. The reason for 
    the bjac->x vector is that we need a sequential vector
    for the sequential solve.
  */
  ierr = VecGetArray(x,&x_array); CHKERRQ(ierr); 
  ierr = VecGetArray(y,&y_array); CHKERRQ(ierr); 
  ierr = VecPlaceArray(bjac->x,x_array); CHKERRQ(ierr); 
  ierr = VecPlaceArray(bjac->y,y_array); CHKERRQ(ierr); 
  ierr = SLESSolveTrans(jac->sles[0],bjac->x,bjac->y,&its); CHKERRQ(ierr); 
  ierr = VecRestoreArray(x,&x_array); CHKERRQ(ierr); 
  ierr = VecRestoreArray(y,&y_array); CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCSetUp_BJacobi_Singleblock"
static int PCSetUp_BJacobi_Singleblock(PC pc, Mat mat, Mat pmat)
{
  PC_BJacobi             *jac = (PC_BJacobi *) pc->data;
  int                    ierr, m;
  SLES                   sles;
  Vec                    x,y;
  PC_BJacobi_Singleblock *bjac;
  KSP                    subksp;
  PC                     subpc;

  PetscFunctionBegin;

  /* set default direct solver with no Krylov method */
  if (!pc->setupcalled) {
    char *prefix;
    ierr = SLESCreate(PETSC_COMM_SELF,&sles); CHKERRQ(ierr);
    PLogObjectParent(pc,sles);
    ierr = SLESGetKSP(sles,&subksp); CHKERRQ(ierr);
    ierr = KSPSetType(subksp,KSPPREONLY); CHKERRQ(ierr);
    ierr = SLESGetPC(sles,&subpc); CHKERRQ(ierr);
    ierr = PCSetType(subpc,PCILU); CHKERRQ(ierr);
    ierr = PCGetOptionsPrefix(pc,&prefix);CHKERRQ(ierr);
    ierr = SLESSetOptionsPrefix(sles,prefix); CHKERRQ(ierr);
    ierr = SLESAppendOptionsPrefix(sles,"sub_"); CHKERRQ(ierr);
    ierr = SLESSetFromOptions(sles); CHKERRQ(ierr);
    /*
      The reason we need to generate these vectors is to serve 
      as the right-hand side and solution vector for the solve on the 
      block. We do not need to allocate space for the vectors since
      that is provided via VecPlaceArray() just before the call to 
      SLESSolve() on the block.
    */
    ierr = MatGetSize(pmat,&m,&m); CHKERRQ(ierr);
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,m,PETSC_NULL,&x); CHKERRQ(ierr);
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,m,PETSC_NULL,&y); CHKERRQ(ierr);
    PLogObjectParent(pc,x);
    PLogObjectParent(pc,y);

    pc->destroy       = PCDestroy_BJacobi_Singleblock;
    pc->apply         = PCApply_BJacobi_Singleblock;
    pc->applytrans    = PCApplyTrans_BJacobi_Singleblock;
    pc->setuponblocks = PCSetUpOnBlocks_BJacobi_Singleblock;

    bjac         = (PC_BJacobi_Singleblock *) PetscMalloc(sizeof(PC_BJacobi_Singleblock));CHKPTRQ(bjac);
    PLogObjectMemory(pc,sizeof(PC_BJacobi_Singleblock));
    bjac->x      = x;
    bjac->y      = y;

    jac->sles    = (SLES*) PetscMalloc( sizeof(SLES) ); CHKPTRQ(jac->sles);
    jac->sles[0] = sles;
    jac->data    = (void *) bjac;
  } else {
    sles = jac->sles[0];
    bjac = (PC_BJacobi_Singleblock *)jac->data;
  }
  if (jac->use_true_local) {
    ierr = SLESSetOperators(sles,mat,pmat,pc->flag); CHKERRQ(ierr);
  }  else {
    ierr = SLESSetOperators(sles,pmat,pmat,pc->flag); CHKERRQ(ierr);
  }   
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------------------------*/

#undef __FUNC__  
#define __FUNC__ "PCDestroy_BJacobi_Multiblock"
int PCDestroy_BJacobi_Multiblock(PC pc)
{
  PC_BJacobi            *jac = (PC_BJacobi *) pc->data;
  PC_BJacobi_Multiblock *bjac = (PC_BJacobi_Multiblock *) jac->data;
  int                   i,ierr;

  PetscFunctionBegin;
  ierr = MatDestroyMatrices(jac->n_local,&bjac->pmat); CHKERRQ(ierr);
  if (jac->use_true_local) {
    ierr = MatDestroyMatrices(jac->n_local,&bjac->mat); CHKERRQ(ierr);
  }

  /*
        If the on processor block had to be generated via a MatGetDiagonalBlock()
     that creates a copy (for example MPIBDiag matrices do), this frees the space
  */
  if (jac->tp_mat) {
    ierr = MatDestroy(jac->tp_mat);CHKERRQ(ierr);
  }
  if (jac->tp_pmat) {
    ierr = MatDestroy(jac->tp_pmat);CHKERRQ(ierr);
  }

  for ( i=0; i<jac->n_local; i++ ) {
    ierr = SLESDestroy(jac->sles[i]); CHKERRQ(ierr);
    ierr = VecDestroy(bjac->x[i]); CHKERRQ(ierr);
    ierr = VecDestroy(bjac->y[i]); CHKERRQ(ierr);
    ierr = ISDestroy(bjac->is[i]); CHKERRQ(ierr);
  }
  PetscFree(jac->sles);
  PetscFree(bjac->x);
  PetscFree(bjac->starts);
  PetscFree(bjac->is);
  PetscFree(bjac);
  if (jac->l_lens) PetscFree(jac->l_lens);
  if (jac->g_lens) PetscFree(jac->g_lens);
  PetscFree(jac); 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCSetUpOnBlocks_BJacobi_Multiblock"
int PCSetUpOnBlocks_BJacobi_Multiblock(PC pc)
{
  PC_BJacobi            *jac = (PC_BJacobi *) pc->data;
  int                   ierr,i,n_local = jac->n_local;
  PC_BJacobi_Multiblock *bjac = (PC_BJacobi_Multiblock *) jac->data;

  PetscFunctionBegin;
  for ( i=0; i<n_local; i++ ) {
    ierr = SLESSetUp(jac->sles[i],bjac->x[i],bjac->y[i]); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
      Preconditioner for block Jacobi 
*/
#undef __FUNC__  
#define __FUNC__ "PCApply_BJacobi_Multiblock"
int PCApply_BJacobi_Multiblock(PC pc,Vec x, Vec y)
{
  PC_BJacobi            *jac = (PC_BJacobi *) pc->data;
  int                   ierr,its,i,n_local = jac->n_local;
  PC_BJacobi_Multiblock *bjac = (PC_BJacobi_Multiblock *) jac->data;
  Scalar                *xin,*yin;
  static int            flag = 1,SUBSlesSolve;

  PetscFunctionBegin;
  if (flag) {
    ierr = PLogEventRegister(&SUBSlesSolve,"SubSlesSolve","black:");CHKERRQ(ierr);
    flag=0;
  }
  ierr = VecGetArray(x,&xin);CHKERRQ(ierr);
  ierr = VecGetArray(y,&yin);CHKERRQ(ierr);
  for ( i=0; i<n_local; i++ ) {
    /* 
       To avoid copying the subvector from x into a workspace we instead 
       make the workspace vector array point to the subpart of the array of
       the global vector.
    */
    ierr = VecPlaceArray(bjac->x[i],xin+bjac->starts[i]);CHKERRQ(ierr);
    ierr = VecPlaceArray(bjac->y[i],yin+bjac->starts[i]);CHKERRQ(ierr);

    PLogEventBegin(SUBSlesSolve,jac->sles[i],bjac->x[i],bjac->y[i],0);
    ierr = SLESSolve(jac->sles[i],bjac->x[i],bjac->y[i],&its); CHKERRQ(ierr);
    PLogEventEnd(SUBSlesSolve,jac->sles[i],bjac->x[i],bjac->y[i],0);
  }
  ierr = VecRestoreArray(x,&xin);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&yin);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
      Preconditioner for block Jacobi 
*/
#undef __FUNC__  
#define __FUNC__ "PCApplyTrans_BJacobi_Multiblock"
int PCApplyTrans_BJacobi_Multiblock(PC pc,Vec x, Vec y)
{
  PC_BJacobi            *jac = (PC_BJacobi *) pc->data;
  int                   ierr,its,i,n_local = jac->n_local;
  PC_BJacobi_Multiblock *bjac = (PC_BJacobi_Multiblock *) jac->data;
  Scalar                *xin,*yin;
  static int            flag = 1,SUBSlesSolve;

  PetscFunctionBegin;
  if (flag) {
    ierr = PLogEventRegister(&SUBSlesSolve,"SubSlesSolveTrans","black:");CHKERRQ(ierr);
    flag=0;
  }
  ierr = VecGetArray(x,&xin);CHKERRQ(ierr);
  ierr = VecGetArray(y,&yin);CHKERRQ(ierr);
  for ( i=0; i<n_local; i++ ) {
    /* 
       To avoid copying the subvector from x into a workspace we instead 
       make the workspace vector array point to the subpart of the array of
       the global vector.
    */
    ierr = VecPlaceArray(bjac->x[i],xin+bjac->starts[i]);CHKERRQ(ierr);
    ierr = VecPlaceArray(bjac->y[i],yin+bjac->starts[i]);CHKERRQ(ierr);

    PLogEventBegin(SUBSlesSolve,jac->sles[i],bjac->x[i],bjac->y[i],0);
    ierr = SLESSolveTrans(jac->sles[i],bjac->x[i],bjac->y[i],&its); CHKERRQ(ierr);
    PLogEventEnd(SUBSlesSolve,jac->sles[i],bjac->x[i],bjac->y[i],0);
  }
  ierr = VecRestoreArray(x,&xin);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&yin);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCSetUp_BJacobi_Multiblock"
int PCSetUp_BJacobi_Multiblock(PC pc,Mat mat,Mat pmat)
{
  PC_BJacobi             *jac = (PC_BJacobi *) pc->data;
  int                    ierr, m, n_local, N, M, start, i;
  char                   *prefix;
  SLES                   sles;
  Vec                    x,y;
  PC_BJacobi_Multiblock  *bjac = (PC_BJacobi_Multiblock *) jac->data;
  KSP                    subksp;
  PC                     subpc;
  IS                     is;
  MatReuse               scall = MAT_REUSE_MATRIX;

  PetscFunctionBegin;
  ierr = MatGetLocalSize(pc->pmat,&M,&N);CHKERRQ(ierr);

  n_local = jac->n_local;

  if (jac->use_true_local) {
    if (mat->type != pmat->type) SETERRQ(PETSC_ERR_ARG_INCOMP,0,"Matrices not of same type");
  }

  /* set default direct solver with no Krylov method */
  if (!pc->setupcalled) {
    scall             = MAT_INITIAL_MATRIX;
    pc->destroy       = PCDestroy_BJacobi_Multiblock;
    pc->apply         = PCApply_BJacobi_Multiblock;
    pc->applytrans    = PCApplyTrans_BJacobi_Multiblock;
    pc->setuponblocks = PCSetUpOnBlocks_BJacobi_Multiblock;

    bjac         = (PC_BJacobi_Multiblock *) PetscMalloc(sizeof(PC_BJacobi_Multiblock));CHKPTRQ(bjac);
    PLogObjectMemory(pc,sizeof(PC_BJacobi_Multiblock));
    jac->sles    = (SLES*) PetscMalloc(n_local*sizeof(SLES)); CHKPTRQ(jac->sles);
    PLogObjectMemory(pc,sizeof(n_local*sizeof(SLES)));
    bjac->x      = (Vec*) PetscMalloc(2*n_local*sizeof(Vec)); CHKPTRQ(bjac->x);
    PLogObjectMemory(pc,sizeof(2*n_local*sizeof(Vec)));
    bjac->y      = bjac->x + n_local;
    bjac->starts = (int*) PetscMalloc(n_local*sizeof(Scalar));CHKPTRQ(bjac->starts);
    PLogObjectMemory(pc,sizeof(n_local*sizeof(Scalar)));
    
    jac->data    = (void *) bjac;
    bjac->is     = (IS *) PetscMalloc(n_local*sizeof(IS)); CHKPTRQ(bjac->is);
    PLogObjectMemory(pc,sizeof(n_local*sizeof(IS)));

    start = 0;
    for ( i=0; i<n_local; i++ ) {
      ierr = SLESCreate(PETSC_COMM_SELF,&sles); CHKERRQ(ierr);
      PLogObjectParent(pc,sles);
      ierr = SLESGetKSP(sles,&subksp); CHKERRQ(ierr);
      ierr = KSPSetType(subksp,KSPPREONLY); CHKERRQ(ierr);
      ierr = SLESGetPC(sles,&subpc); CHKERRQ(ierr);
      ierr = PCSetType(subpc,PCILU); CHKERRQ(ierr);
      ierr = PCGetOptionsPrefix(pc,&prefix); CHKERRQ(ierr);
      ierr = SLESSetOptionsPrefix(sles,prefix); CHKERRQ(ierr);
      ierr = SLESAppendOptionsPrefix(sles,"sub_"); CHKERRQ(ierr);
      ierr = SLESSetFromOptions(sles); CHKERRQ(ierr);

      m = jac->l_lens[i];

      /*
      The reason we need to generate these vectors is to serve 
      as the right-hand side and solution vector for the solve on the 
      block. We do not need to allocate space for the vectors since
      that is provided via VecPlaceArray() just before the call to 
      SLESSolve() on the block.

           For x we need a real array since it is used directly by the 
          GS version.

      */
      ierr = VecCreateSeq(PETSC_COMM_SELF,m,&x); CHKERRQ(ierr);
      ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,m,PETSC_NULL,&y); CHKERRQ(ierr);
      PLogObjectParent(pc,x);
      PLogObjectParent(pc,y);
      bjac->x[i]      = x;
      bjac->y[i]      = y;
      bjac->starts[i] = start;
      jac->sles[i]    = sles;

      ierr = ISCreateStride(PETSC_COMM_SELF,m,start,1,&is); CHKERRQ(ierr);
      bjac->is[i] = is;
      PLogObjectParent(pc,is);

      start += m;
    }
  } else {
    bjac = (PC_BJacobi_Multiblock *) jac->data;
    /* 
       Destroy the blocks from the previous iteration
    */
    if (pc->flag == DIFFERENT_NONZERO_PATTERN) {
      ierr = MatDestroyMatrices(n_local,&bjac->pmat); CHKERRQ(ierr);
      if (jac->use_true_local) {
        ierr = MatDestroyMatrices(n_local,&bjac->mat); CHKERRQ(ierr);
      }
      scall = MAT_INITIAL_MATRIX;
    }
  }

  ierr = MatGetSubMatrices(pmat,n_local,bjac->is,bjac->is,scall,&bjac->pmat);CHKERRQ(ierr);
  if (jac->use_true_local) {
    ierr = MatGetSubMatrices(mat,n_local,bjac->is,bjac->is,scall,&bjac->mat);CHKERRQ(ierr);
  }
  /* Return control to the user so that the submatrices can be modified (e.g., to apply
     different boundary conditions for the submatrices than for the global problem) */
  ierr = PCModifySubMatrices(pc,n_local,bjac->is,bjac->is,bjac->pmat,pc->modifysubmatricesP);CHKERRQ(ierr);
  for ( i=0; i<n_local; i++ ) {
    PLogObjectParent(pc,bjac->pmat[i]);
    if (jac->use_true_local) {
      PLogObjectParent(pc,bjac->mat[i]);
      ierr = SLESSetOperators(jac->sles[i],bjac->mat[i],bjac->pmat[i],pc->flag);CHKERRQ(ierr);
    } else {
      ierr = SLESSetOperators(jac->sles[i],bjac->pmat[i],bjac->pmat[i],pc->flag);CHKERRQ(ierr);
    }
  }

  PetscFunctionReturn(0);
}











