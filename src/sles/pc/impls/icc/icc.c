#ifndef lint
static char vcid[] = "$Id: ";
#endif
/*
   Defines a Cholesky factorization preconditioner for any Mat implementation.
*/

#include "pcimpl.h"
#include "mat/matimpl.h"
#include "icc.h"

extern int PCImplCreate_ICC_MPIRow(PC pc);
extern int PCImplDestroy_ICC_MPIRow(PC pc);

static int PCSetup_ICC(PC pc)
{
  PC_ICC *icc = (PC_ICC *) pc->data;
  IS     perm;
  int    ierr;

#if defined(HAVE_BLOCKSOLVE) && !defined(PETSC_COMPLEX)
  { Mat mat = pc->mat;
    if (mat->type == MATMPIROW) icc->ImplCreate = PCImplCreate_ICC_MPIRow;
  }
#endif
  if (icc->ImplCreate) {ierr = (*icc->ImplCreate)(pc); CHKERR(ierr);}

  /* Currently no reorderings are supported.
  ierr = MatGetReordering(pc->pmat,icc->ordering,&perm,&perm); CHKERR(ierr); */

  if (!pc->setupcalled) {
    ierr = MatIncompleteCholeskyFactorSymbolic(pc->pmat,perm,
				icc->levels,&icc->fact); CHKERR(ierr);
  }
  else if (!(pc->flag & MAT_SAME_NONZERO_PATTERN)) {
    ierr = MatDestroy(icc->fact); CHKERR(ierr);
    ierr = MatIncompleteCholeskyFactorSymbolic(pc->pmat,perm,
				icc->levels,&icc->fact); CHKERR(ierr);
  }
  ierr = MatCholeskyFactorNumeric(pc->pmat,&icc->fact); CHKERR(ierr);
  return 0;
}

static int PCDestroy_ICC(PetscObject obj)
{
  PC     pc = (PC) obj;
  PC_ICC *icc = (PC_ICC *) pc->data;
  int    ierr;

  if (icc->ImplDestroy) {ierr = (*icc->ImplDestroy)(pc); CHKERR(ierr);}
  MatDestroy(icc->fact);
  FREE(icc);
  PLogObjectDestroy(pc);
  PETSCHEADERDESTROY(pc);
  return 0;
}

static int PCApply_ICC(PC pc,Vec x,Vec y)
{
  PC_ICC *icc = (PC_ICC *) pc->data;
  int    ierr;

  return MatSolve(icc->fact,x,y);
}

static int PCPrintHelp_ICC(PC pc)
{
  char *p;
  if (pc->prefix) p = pc->prefix; else p = "-";
  fprintf(stderr,"%ssor_its its: number of inner SOR iterations to use\n",p);
  return 0;
}

int PCCreate_ICC(PC pc)
{
  int    ierr;
  PC_ICC *icc = NEW(PC_ICC); CHKPTR(icc);
  icc->fact		= 0;
  icc->ordering 	= ORDER_ND;
  icc->levels		= 0;
  icc->ImplCreate       = 0;
  icc->ImplDestroy	= 0;
  pc->apply		= PCApply_ICC;
  pc->setup		= PCSetup_ICC;
  pc->destroy		= PCDestroy_ICC;
  pc->type		= PCICC;
  pc->data		= (void *) icc;
  pc->printhelp		= PCPrintHelp_ICC;

  if (icc->ImplCreate) {ierr = (*icc->ImplCreate)(pc); CHKERR(ierr);}
  return 0;
}
