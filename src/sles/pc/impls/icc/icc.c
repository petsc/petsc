#ifndef lint
static char vcid[] = "$Id: $ ";
#endif
/*
   Defines a Cholesky factorization preconditioner for any Mat implementation.
*/

#include "pcimpl.h"
#include "mat/matimpl.h"
#include "icc.h"

extern int PCImplCreate_ICC_MPIRowbs(PC pc);
extern int PCImplDestroy_ICC_MPIRowbs(PC pc);

static int PCSetup_ICC(PC pc)
{
  PC_ICC *icc = (PC_ICC *) pc->data;
  IS     perm;
  int    ierr;

  /* Currently no reorderings are supported.
  ierr = MatGetReordering(pc->pmat,icc->ordering,&perm,&perm); CHKERR(ierr); */

  if (!pc->setupcalled) {
#if defined(HAVE_BLOCKSOLVE) && !defined(PETSC_COMPLEX)
    if (pc->mat->type == MATMPIROW_BS) {
      if (!icc->bs_iter) icc->ImplCreate = PCImplCreate_ICC_MPIRowbs;
    }
#endif
    if (icc->ImplCreate) {ierr = (*icc->ImplCreate)(pc); CHKERR(ierr);}
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
  PC_ICC       *icc = (PC_ICC *) pc->data;
  Mat_MPIRowbs *mbs = (Mat_MPIRowbs *) icc->pmat->data;
  int          ierr, guess = 0, block_size, max_iters, cgtol;
 
  if (!icc->bs_iter) return MatSolve(icc->fact,x,y);
  guess = 1; 

  /* Call BlockSolve ICCG solver, where permuting and scaling occurs
     _within_ BSpar_solve */
  block_size = 1;
  pre_option = PRE_STICCG;   /* preconditioning option:  ICC */
  max_iters = ctx->itctx->max_it;
  cgtol = ctx->itctx->rtol;
  its = BSpar_solve( block_size, lctx->pA, lctx->f_pA, lctx->comm_pA,
                     b, x, pre_option, cgtol,
                     max_iters, &residual, guess, lctx->procinfo); 
}

static int PCSetFromOptions_ICC(PC pc)
{
  if (OptionsHasName(0,pc->prefix,"-icc_bsiter")) {
    PCICCSetBlockSolveIter(pc);
  } 
  return 0;
}

/*@ 
   PCICCSetBlockSolveIter - Sets flag so that BlockSolve iterative solver is
   used instead of default KSP routines.

   Input Parameter:
.  pc - the preconditioner context

   Note:
   This option is valid only when the MATMPIROW_BS data structure
   is used for the preconditioning matrix.
@*/
int PCICCSetBlockSolveIter(PC pc)
{
  PC_ICC *icc;
  VALIDHEADER(pc,PC_COOKIE);
  if (pc->type != PCICC) return 0;
  icc = (PC_ICC *) pc->data;
  icc->bs_iter = 1;
  return 0;
}

static int PCPrintHelp_ICC(PC pc)
{
  char *p;
  if (pc->prefix) p = pc->prefix; else p = "-";
  fprintf(stderr,"%icc_bsiter:  use BlockSolve iterative solver instead\
                  of KSP routines\n",p);
  return 0;
}

int PCCreate_ICC(PC pc)
{
  PC_ICC *icc = NEW(PC_ICC); CHKPTR(icc);
  icc->fact	   = 0;
  icc->ordering    = ORDER_ND;
  icc->levels	   = 0;
  icc->ImplCreate  = 0;
  icc->ImplDestroy = 0;
  icc->bs_iter     = 0;
  pc->apply	   = PCApply_ICC;
  pc->setup        = PCSetup_ICC;
  pc->setfrom      = PCSetFromOptions_ICC;
  pc->printhelp    = PCPrintHelp_ICC;  
  pc->destroy	   = PCDestroy_ICC;
  pc->type	   = PCICC;
  pc->data	   = (void *) icc;
  pc->printhelp	   = PCPrintHelp_ICC;
  return 0;
}

