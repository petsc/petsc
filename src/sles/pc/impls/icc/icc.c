#ifndef lint
static char vcid[] = "$Id: icc.c,v 1.36 1996/04/26 00:00:26 balay Exp balay $ ";
#endif
/*
   Defines a Cholesky factorization preconditioner for any Mat implementation.
  Presently only provided for MPIRowbs format (i.e. BlockSolve).
*/

#include "pcimpl.h"          /*I "pc.h" I*/
#include "src/mat/matimpl.h"
#include "icc.h"

extern int PCSetUp_ICC_MPIRowbs(PC);

static int (*setups[])(PC) = {0,
                              0,
                              0,
                              0,
#if defined(HAVE_BLOCKSOLVE) && !defined(__cplusplus)
                              PCSetUp_ICC_MPIRowbs,
#else
                              0,
#endif
                              0,
                              0,
                              0,   
                              0,
                              0,0,0,0,0};


static int PCSetup_ICC(PC pc)
{
  PC_ICC *icc = (PC_ICC *) pc->data;
  IS     perm;
  int    ierr;

  /* Currently no reorderings are supported!
  ierr = MatGetReordering(pc->pmat,icc->ordering,&perm,&perm); CHKERRQ(ierr); */
  perm = 0;

  if (!pc->setupcalled) {
    if (setups[pc->pmat->type]) {
      ierr = (*setups[pc->pmat->type])(pc); CHKERRQ(ierr);
    }
    ierr = MatIncompleteCholeskyFactorSymbolic(pc->pmat,perm,1.0,
				icc->levels,&icc->fact); CHKERRQ(ierr);
  }
  else if (pc->flag != SAME_NONZERO_PATTERN) {
    ierr = MatDestroy(icc->fact); CHKERRQ(ierr);
    ierr = MatIncompleteCholeskyFactorSymbolic(pc->pmat,perm,1.0,
				icc->levels,&icc->fact); CHKERRQ(ierr);
  }
  ierr = MatCholeskyFactorNumeric(pc->pmat,&icc->fact); CHKERRQ(ierr);
  return 0;
}

static int PCDestroy_ICC(PetscObject obj)
{
  PC     pc = (PC) obj;
  PC_ICC *icc = (PC_ICC *) pc->data;

  MatDestroy(icc->fact);
  PetscFree(icc);
  return 0;
}

static int PCApply_ICC(PC pc,Vec x,Vec y)
{
  PC_ICC *icc = (PC_ICC *) pc->data;
  int    ierr;

  ierr = MatSolve(icc->fact,x,y); CHKERRQ(ierr);
  return 0;  
}

static int PCApplySymmetricLeft_ICC(PC pc,Vec x,Vec y)
{
  PC_ICC *icc = (PC_ICC *) pc->data;
  return MatForwardSolve(icc->fact,x,y);
}

static int PCApplySymmetricRight_ICC(PC pc,Vec x,Vec y)
{
  PC_ICC *icc = (PC_ICC *) pc->data;
  return MatBackwardSolve(icc->fact,x,y);
}

static int PCPrintHelp_ICC(PC pc,char *p)
{
  PetscPrintf(pc->comm," Options for PCICC preconditioner:\n");
  PetscPrintf(pc->comm,"  %spc_icc_factorpointwise:Do NOT use block factorization \n",p);
  PetscPrintf(pc->comm,"    (Note: This only applies to the MATMPIROWBS matrix format;\n");
  PetscPrintf(pc->comm,"    all others currently only support point factorization.\n");
  return 0;
}

static int PCGetFactoredMatrix_ICC(PC pc,Mat *mat)
{
  PC_ICC *icc = (PC_ICC *) pc->data;
  *mat = icc->fact;
  return 0;
}

static int PCSetFromOptions_ICC(PC pc)
{
  return 0;
}

int PCCreate_ICC(PC pc)
{
  PC_ICC      *icc = PetscNew(PC_ICC); CHKPTRQ(icc);
  icc->fact	   = 0;
  icc->ordering    = ORDER_ND;
  icc->levels	   = 0;
  icc->bs_iter     = 0;
  icc->implctx     = 0;
  pc->apply	   = PCApply_ICC;
  pc->setup        = PCSetup_ICC;
  pc->destroy	   = PCDestroy_ICC;
  pc->setfrom      = PCSetFromOptions_ICC;
  pc->printhelp    = PCPrintHelp_ICC;
  pc->view         = 0;
  pc->getfactoredmatrix   = PCGetFactoredMatrix_ICC;
  pc->type	   = PCICC;
  pc->data	   = (void *) icc;
  pc->applysymmetricleft  = PCApplySymmetricLeft_ICC;
  pc->applysymmetricright = PCApplySymmetricRight_ICC;
  return 0;
}


