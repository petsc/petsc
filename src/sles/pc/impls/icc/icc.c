#ifndef lint
static char vcid[] = "$Id: icc.c,v 1.18 1995/09/06 03:05:14 bsmith Exp curfman $ ";
#endif
/*
   Defines a Cholesky factorization preconditioner for any Mat implementation.
*/

#include "pcimpl.h"
#include "mat/matimpl.h"
#include "icc.h"

#if defined(HAVE_BLOCKSOLVE) && !defined(_cplusplus)
extern int PCImplCreate_ICC_MPIRowbs(PC pc);
extern int PCImplDestroy_ICC_MPIRowbs(PC pc);
#endif

static int PCSetup_ICC(PC pc)
{
  PC_ICC *icc = (PC_ICC *) pc->data;
  IS     perm;
  int    ierr;

  /* Currently no reorderings are supported!
  ierr = MatGetReordering(pc->pmat,icc->ordering,&perm,&perm); CHKERRQ(ierr); */
  perm = 0;

  if (!pc->setupcalled) {
#if defined(HAVE_BLOCKSOLVE) && !defined(_cplusplus)
    if (pc->pmat->type == MATMPIROW_BS) {
      icc->ImplCreate = PCImplCreate_ICC_MPIRowbs;
    }
#endif
    if (icc->ImplCreate) {ierr = (*icc->ImplCreate)(pc); CHKERRQ(ierr);}
    ierr = MatIncompleteCholeskyFactorSymbolic(pc->pmat,perm,1.0,
				icc->levels,&icc->fact); CHKERRQ(ierr);
  }
  else if (!(pc->flag & PMAT_SAME_NONZERO_PATTERN)) {
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
  int    ierr;

  if (icc->ImplDestroy) {ierr = (*icc->ImplDestroy)(pc); CHKERRQ(ierr);}
  MatDestroy(icc->fact);
  PETSCFREE(icc);
  return 0;
}

static int PCApply_ICC(PC pc,Vec x,Vec y)
{
  PC_ICC *icc = (PC_ICC *) pc->data;
  return MatSolve(icc->fact,x,y);
}

static int PCPrintHelp_ICC(PC pc)
{
  char *p;
  if (pc->prefix) p = pc->prefix; else p = "-";
  MPIU_printf(pc->comm," Options for PCICC preconditioner:\n");
  MPIU_printf(pc->comm,"%pc_icc_bsiter:  use BlockSolve iterative solver instead\
                  of KSP routines\n",p);
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
#if defined(HAVE_BLOCKSOLVE) && !defined(__cplusplus)
  if (OptionsHasName(pc->prefix,"-pc_icc_bsiter")) {
    PCBSIterSetBlockSolve(pc);
  }
#endif
  return 0;
}

int PCCreate_ICC(PC pc)
{
  PC_ICC      *icc = PETSCNEW(PC_ICC); CHKPTRQ(icc);
  icc->fact	   = 0;
  icc->ordering    = ORDER_ND;
  icc->levels	   = 0;
  icc->ImplCreate  = 0;
  icc->ImplDestroy = 0;
  icc->bs_iter     = 0;
  icc->implctx     = 0;
  pc->apply	   = PCApply_ICC;
  pc->setup        = PCSetup_ICC;
  pc->destroy	   = PCDestroy_ICC;
  pc->setfrom      = PCSetFromOptions_ICC;
  pc->printhelp    = PCPrintHelp_ICC;
  pc->view         = 0;
  pc->getfactmat   = PCGetFactoredMatrix_ICC;
  pc->type	   = PCICC;
  pc->data	   = (void *) icc;
  return 0;
}


