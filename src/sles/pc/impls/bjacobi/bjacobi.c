#ifndef lint
static char vcid[] = "$Id: bjacobi.c,v 1.77 1996/07/08 22:18:33 bsmith Exp curfman $";
#endif
/*
   Defines a block Jacobi preconditioner.
*/
#include "src/mat/matimpl.h"
#include "pcimpl.h"              /*I "pc.h" I*/
#include "bjacobi.h"
#include "pinclude/pviewer.h"

extern int PCSetUp_BJacobiAIJ(PC);
extern int PCSetUp_BJacobiBAIJ(PC);
extern int PCSetUp_BJacobiMPIBDiag(PC);

static int (*setups[])(PC) = {0,
                              PCSetUp_BJacobiAIJ,
                              PCSetUp_BJacobiAIJ,
                              0,
                              0,
                              0,   
                              PCSetUp_BJacobiMPIBDiag,
                              0,PCSetUp_BJacobiBAIJ,
                              PCSetUp_BJacobiBAIJ,0};

static int PCSetUp_BJacobi(PC pc)
{
  Mat pmat = pc->pmat;

  if (!setups[pmat->type]) SETERRQ(PETSC_ERR_SUP,"PCSetUp_BJacobi");
  return (*setups[pmat->type])(pc);
}

/* Default destroy, if it has never been setup */
static int PCDestroy_BJacobi(PetscObject obj)
{
  PC         pc = (PC) obj;
  PC_BJacobi *jac = (PC_BJacobi *) pc->data;
  if (jac->g_lens) PetscFree(jac->g_lens);
  if (jac->l_lens) PetscFree(jac->l_lens);
  if (jac->g_true) PetscFree(jac->g_true);
  if (jac->l_true) PetscFree(jac->l_true);
  PetscFree(jac);
  return 0;
}

static int PCSetFromOptions_BJacobi(PC pc)
{
  int        blocks,flg,ierr;

  ierr = OptionsGetInt(pc->prefix,"-pc_bjacobi_blocks",&blocks,&flg); CHKERRQ(ierr);
  if (flg) {
    ierr = PCBJacobiSetTotalBlocks(pc,blocks,PETSC_NULL,PETSC_NULL); CHKERRQ(ierr); 
  }
  ierr = OptionsHasName(pc->prefix,"-pc_bjacobi_truelocal",&flg); CHKERRQ(ierr);
  if (flg) {
    ierr = PCBJacobiSetUseTrueLocal(pc); CHKERRQ(ierr);
  }
  return 0;
}

static int PCSetFromOptions_BGS(PC pc)
{
  int        blocks,ierr,flg;

  ierr = OptionsGetInt(pc->prefix,"-pc_bgs_blocks",&blocks,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PCBGSSetTotalBlocks(pc,blocks,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  }
  ierr = OptionsHasName(pc->prefix,"-pc_bgs_truelocal",&flg); CHKERRQ(ierr);
  if (flg) {
    ierr = PCBGSSetUseTrueLocal(pc); CHKERRQ(ierr);
  }
  ierr = OptionsHasName(pc->prefix,"-pc_bgs_symmetric",&flg); CHKERRQ(ierr);
  if (flg) {
    ierr = PCBGSSetSymmetric(pc,PCBGS_SYMMETRIC_SWEEP); CHKERRQ(ierr);
  }
  return 0;
}


/*@
   PCBGSSetSymmetric - Sets the BGS preconditioner to use symmetric, 
   backward, or forward relaxation. By default forward relaxation is used.

   Input Parameters:
.  pc - the preconditioner context
.  flag - one of the following:
$    PCBGS_FORWARD_SWEEP
$    PCBGS_SYMMETRIC_SWEEP

   Options Database Keys:
$  -pc_gs_symmetric

.keywords: PC, BGS, Gauss-Seidel, set, relaxation, sweep, forward, symmetric

.seealso: PCBGSSetTotalBlocks() PCBGSSetUseTrueLocal()
@*/
int PCBGSSetSymmetric(PC pc, PCBGSType flag)
{
  PC_BJacobi *jac = (PC_BJacobi *) pc->data; 
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  if (pc->type != PCBGS) return 0;
  jac->gstype = flag;
  return 0;
}

/*@
   PCBJacobiSetUseTrueLocal - Sets a flag to indicate that the block 
   problem is associated with the linear system matrix instead of the
   default (where it is associated with the preconditioning matrix).
   That is, if the local system is solved iteratively then it iterates
   on the block from the matrix using the block from the preconditioner
   as the preconditioner for the local block.

   Input Parameters:
.  pc - the preconditioner context

   Options Database Key:
$  -pc_bjacobi_truelocal

   Note:
   For the common case in which the preconditioning and linear 
   system matrices are identical, this routine is unnecessary.

.keywords:  block, Jacobi, set, true, local, flag

.seealso: PCSetOperators(), PCBJacobiSetLocalBlocks()
@*/
int PCBJacobiSetUseTrueLocal(PC pc)
{
  PC_BJacobi   *jac;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  if (pc->type != PCBJACOBI && pc->type != PCBGS ) return 0;
  jac = (PC_BJacobi *) pc->data;
  jac->use_true_local = 1;
  return 0;
}

/*@
   PCBGSSetUseTrueLocal - Sets a flag to indicate that the block 
   problem is associated with the linear system matrix instead of the
   default (where it is associated with the preconditioning matrix).
   That is, if the local system is solved iteratively then it iterates
   on the block from the matrix using the block from the preconditioner
   as the preconditioner for the local block.

   Input Parameters:
.  pc - the preconditioner context

   Options Database Key:
$  -pc_bgs_truelocal

   Note:
   For the common case in which the preconditioning and linear 
   system matrices are identical, this routine is unnecessary.

.keywords:  block, BGS, Gauss-Seidel, set, true, local, flag

.seealso: PCSetOperators(), PCBGSSetBlocks()
@*/
int PCBGSSetUseTrueLocal(PC pc)
{
  return PCBJacobiSetUseTrueLocal(pc);
}

/*@C
   PCBJacobiGetSubSLES - Gets the local SLES contexts for all blocks on
   this processor.
   
   Input Parameter:
.  pc - the preconditioner context

   Output Parameters:
.  n_local - the number of blocks on this processor
.  first_local - the global number of the first block on this processor
.  sles - the array of SLES contexts

   Note:  
   Currently for some matrix implementations only 1 block per processor 
   is supported.
   
   You must call SLESSetUp() before calling PCBJacobiGetSubSLES().

.keywords:  block, Jacobi, get, sub, SLES, context

.seealso: PCBJacobiGetSubSLES()
@*/
int PCBJacobiGetSubSLES(PC pc,int *n_local,int *first_local,SLES **sles)
{
  PC_BJacobi   *jac;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  if (pc->type != PCBJACOBI && pc->type != PCBGS) return 0;
  if (!pc->setupcalled) SETERRQ(1,"PCBJacobiGetSubSLES:Must call SLESSetUp first");
  jac = (PC_BJacobi *) pc->data;
  *n_local = jac->n_local;
  *first_local = jac->first_local;
  *sles = jac->sles;
  jac->same_local_solves = 0; /* Assume that local solves are now different;
                                 not necessarily true though!  This flag is 
                                 used only for PCView_BJacobi */
  return 0;
}

/*@C
   PCBGSGetSubSLES - Gets the local SLES contexts for all blocks on
   this processor.
   
   Input Parameter:
.  pc - the preconditioner context

   Output Parameters:
.  n_local - the number of blocks on this processor
.  first_local - the global number of the first block on this processor
.  sles - the array of SLES contexts

   Note:  
   Currently for some matrix implementations only 1 block per processor 
   is supported.
   
   You must call SLESSetUp() before calling PCBGSGetSubSLES().

.keywords:  block, BGS, Gauss-Seidel, get, sub, SLES, context

.seealso: PCBJacobiGetSubSLES()
@*/
int PCBGSGetSubSLES(PC pc,int *n_local,int *first_local,SLES **sles)
{  
  return PCBJacobiGetSubSLES(pc, n_local, first_local, sles);
}

static int PCPrintHelp_BJacobi(PC pc,char *p)
{
  PetscPrintf(pc->comm," Options for PCBJACOBI preconditioner:\n");
  PetscPrintf(pc->comm," %spc_bjacobi_blocks <blks>: total blocks in preconditioner\n",p);
  PetscPrintf(pc->comm, " %spc_bjacobi_truelocal: use blocks from the local linear\
 system matrix \n      instead of the preconditioning matrix\n",p);
  PetscPrintf(pc->comm," %ssub : prefix to control options for individual blocks.\
 Add before the \n      usual KSP and PC option names (e.g., %ssub_ksp_type\
 <kspmethod>)\n",p,p);
  return 0;
}

static int PCPrintHelp_BGS(PC pc,char *p)
{
  PetscPrintf(pc->comm," Options for PCBGS preconditioner:\n");
  PetscPrintf(pc->comm," %spc_bgs_blocks <blks>: total blocks in preconditioner\n",p);
  PetscPrintf(pc->comm, " %spc_bgs_truelocal: use blocks from the local linear\
 system matrix \n      instead of the preconditioning matrix\n",p);
  PetscPrintf(pc->comm, " %spc_bgs_symmetric: use both a forward and backward sweep\n",p);
  PetscPrintf(pc->comm," %ssub : prefix to control options for individual blocks.\
 Add before the \n      usual KSP and PC option names (e.g., %ssub_ksp_type\
 <kspmethod>)\n",p,p);
  return 0;
}

static int PCView_BJacobi(PetscObject obj,Viewer viewer)
{
  PC               pc = (PC)obj;
  FILE             *fd;
  PC_BJacobi       *jac = (PC_BJacobi *) pc->data;
  int              rank, ierr;
  char             *c,*bgs = "Block Gauss-Seiedel", *bj ="Block Jacobi";
  ViewerType       vtype;

  if (jac->gs) c = bgs; else c = bj;
  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  if (vtype == ASCII_FILE_VIEWER || vtype == ASCII_FILES_VIEWER) {
    ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
    if (jac->use_true_local) {
      PetscFPrintf(pc->comm,fd,
         "    %s: using true local matrix, number of blocks = %d\n", c, jac->n);
    }
    PetscFPrintf(pc->comm,fd,"    %s: number of blocks = %d\n", c, jac->n);
    MPI_Comm_rank(pc->comm,&rank);
    if (jac->same_local_solves) {
      PetscFPrintf(pc->comm,fd,
      "    Local solve is same for all blocks, in the following KSP and PC objects:\n");
      if (!rank) {
        ierr = SLESView(jac->sles[0],VIEWER_STDOUT_SELF); CHKERRQ(ierr);
      }           /* now only 1 block per proc */
                /* This shouldn't really be STDOUT */
    } else {
      PetscFPrintf(pc->comm,fd,
       "    Local solve info for each block is in the following KSP and PC objects:\n");
      PetscSequentialPhaseBegin(pc->comm,1);
      fprintf(fd,"Proc %d: number of local blocks = %d, first local block number = %d\n",
      rank,jac->n_local,jac->first_local);
      ierr = SLESView(jac->sles[0],VIEWER_STDOUT_SELF); CHKERRQ(ierr);
           /* now only 1 block per proc */
           /* This shouldn't really be STDOUT */
      fflush(fd);
      PetscSequentialPhaseEnd(pc->comm,1);
    }
  } else if (vtype == STRING_VIEWER) {
    ViewerStringSPrintf(viewer," blks=%d",jac->n);
    ierr = SLESView(jac->sles[0],viewer);
  }
  return 0;
}

int PCCreate_BJacobi(PC pc)
{
  int          rank,size;
  PC_BJacobi   *jac = PetscNew(PC_BJacobi); CHKPTRQ(jac);

  MPI_Comm_rank(pc->comm,&rank);
  MPI_Comm_size(pc->comm,&size);
  pc->apply              = 0;
  pc->setup              = PCSetUp_BJacobi;
  pc->destroy            = PCDestroy_BJacobi;
  pc->setfrom            = PCSetFromOptions_BJacobi;
  pc->printhelp          = PCPrintHelp_BJacobi;
  pc->view               = PCView_BJacobi;
  pc->type               = PCBJACOBI;
  pc->data               = (void *) jac;
  jac->gs                = PETSC_FALSE;
  jac->n                 = -1;
  jac->n_local           = -1;
  jac->first_local       = rank;
  jac->sles              = 0;
  jac->use_true_local    = 0;
  jac->same_local_solves = 1;
  jac->g_lens            = 0;
  jac->l_lens            = 0;
  jac->g_true            = 0;
  jac->l_true            = 0;
  return 0;
}

int PCCreate_BGS(PC pc)
{
  int        ierr;
  PC_BJacobi *jac;

  ierr          = PCCreate_BJacobi(pc); CHKERRQ(ierr);
  jac           = (PC_BJacobi*) pc->data;
  jac->gs       = PETSC_TRUE;
  jac->gstype   = PCBGS_FORWARD_SWEEP;
  pc->setfrom   = PCSetFromOptions_BGS;
  pc->printhelp = PCPrintHelp_BGS;
  pc->type      = PCBGS;
  return 0;
}
  
/*@
   PCBJacobiSetTotalBlocks - Sets the global number of blocks for the block
   Jacobi preconditioner.

   Input Parameters:
.  pc - the preconditioner context
.  blocks - the number of blocks
.  lens - [optional] integer array containing the size of each block
.  true - [optional] integer array whose entries are USE_PRECONDITIONER_MATRIX
.          or USE_TRUE_MATRIX can only be provided if lens is provided.

   Options Database Key:
$  -pc_bjacobi_blocks blocks

   Notes:  
   Currently only a limited number of blocking configurations are supported.
   All processors sharing the PC must call this routine with the same data.

.keywords:  set, number, Jacobi, global, total, blocks

.seealso: PCBJacobiSetUseTrueLocal(), PCBJacobiSetLocalBlocks()
@*/
int PCBJacobiSetTotalBlocks(PC pc, int blocks,int *lens,int *true1)
{
  PC_BJacobi *jac = (PC_BJacobi *) pc->data; 

  PetscValidHeaderSpecific(pc,PC_COOKIE);
  if (blocks <= 0) SETERRQ(1,"PCBJacobiSetTotalBlocks:Must have positive blocks");
  if (pc->type != PCBJACOBI && pc->type != PCBGS) return 0;

  jac->n = blocks;
  if (!lens) {
    jac->g_lens = 0;
    jac->g_true = 0;
  }
  else {
    jac->g_lens = (int *) PetscMalloc(blocks*sizeof(int)); CHKPTRQ(jac->g_lens);
    PetscMemcpy(jac->g_lens,lens,blocks*sizeof(int));
    jac->g_true = (int *) PetscMalloc(blocks*sizeof(int)); CHKPTRQ(jac->g_true);
    if (true1) {
      PetscMemcpy(jac->g_true,true1,blocks*sizeof(int));      
    }
    else {
      PetscMemzero(jac->g_true,blocks*sizeof(int));
    }
  }
  return 0;
}
  
/*@
   PCBGSSetTotalBlocks - Sets the global number of blocks for the block Gauss-Seidel
   (BGS) preconditioner.

   Input Parameters:
.  pc - the preconditioner context
.  blocks - the number of blocks
.  lens - [optional] integer array containing the size of each block
.  true - [optional] integer array whose entries are USE_PRECONDITIONER_MATRIX
.          or USE_TRUE_MATRIX can only be provided if lens is provided.

   Options Database Key:
$  -pc_bgs_blocks blocks

   Notes:  
   Currently only a limited number of blocking configurations are supported.
   All processors sharing the PC must call this routine with the same data.

.keywords:  set, number, BGS, gauss-seidel, global, total, blocks

.seealso: PCBGSSetUseTrueLocal(), PCBGSSetLocalBlocks()
@*/
int PCBGSSetTotalBlocks(PC pc, int blocks,int *lens,int *true1)
{
  return PCBJacobiSetTotalBlocks(pc, blocks, lens, true1);
}
/*@
   PCBJacobiSetLocalBlocks - Sets the local number of blocks for the block
   Jacobi preconditioner.

   Input Parameters:
.  pc - the preconditioner context
.  blocks - the number of blocks
.  lens - [optional] integer array containing size of each block
.  true - [optiona] integer array whose entries are USE_PRECONDITIONER_MATRIX
.          or USE_TRUE_MATRIX can only be provided if lens is provided.

   Note:  
   Currently only a limited number of blocking configurations are supported.

.keywords: PC, set, number, Jacobi, local, blocks

.seealso: PCBJacobiSetUseTrueLocal(), PCBJacobiSetTotalBlocks()
@*/
int PCBJacobiSetLocalBlocks(PC pc, int blocks,int *lens,int *true1)
{
  PC_BJacobi *jac = (PC_BJacobi *) pc->data; 

  PetscValidHeaderSpecific(pc,PC_COOKIE);
  if (blocks < 0) SETERRQ(1,"PCBJacobiSetLocalBlocks:Must have nonegative blocks");
  if (pc->type != PCBJACOBI && pc->type != PCBGS ) return 0;

  jac->n_local = blocks;
  if (!lens) {
    jac->l_lens = 0;
    jac->l_true = 0;
  }
  else {
    jac->l_lens = (int *) PetscMalloc(blocks*sizeof(int)); CHKPTRQ(jac->l_lens);
    PetscMemcpy(jac->l_lens,lens,blocks*sizeof(int));
    jac->l_true = (int *) PetscMalloc(blocks*sizeof(int)); CHKPTRQ(jac->l_true);
    if (true1) {
      PetscMemcpy(jac->l_true,true1,blocks*sizeof(int));      
    }
    else {
      PetscMemzero(jac->l_true,blocks*sizeof(int));
    }
  }
  return 0;
}


/*@
   PCBGSSetLocalBlocks - Sets the local number of blocks for the block
   Gauss-Seidel (BGS) preconditioner.

   Input Parameters:
.  pc - the preconditioner context
.  blocks - the number of blocks
.  lens - [optional] integer array containing size of each block
.  true - [optiona] integer array whose entries are USE_PRECONDITIONER_MATRIX
.          or USE_TRUE_MATRIX can only be provided if lens is provided.

   Note:  
   Currently only a limited number of blocking configurations are supported.

.keywords: PC, set, number, BGS, Gauss-Seidel, local, blocks

.seealso: PCBGSSetUseTrueLocal(), PCBGSSetTotalBlocks()
@*/
int PCBGSSetLocalBlocks(PC pc, int blocks,int *lens,int *true1)
{
  return PCBJacobiSetLocalBlocks(pc, blocks, lens, true1);
}


