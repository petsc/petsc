
#ifndef lint
static char vcid[] = "$Id: snestest.c,v 1.11 1995/08/14 23:16:04 curfman Exp curfman $";
#endif

#include "draw.h"
#include "snesimpl.h"

typedef struct {
  int complete_print;
} SNES_Test;

/*
     SNESSolve_Test - Tests whether a hand computed Jacobian 
     matches one compute via finite differences.
*/
int SNESSolve_Test(SNES snes,int *its)
{
  Mat          A = snes->jacobian,B;
  Vec          x = snes->vec_sol;
  int          ierr,i;
  MatStructure flg;
  Scalar       mone = -1.0,one = 1.0;
  double       norm,gnorm;
  SNES_Test    *neP = (SNES_Test*) snes->data;

  if (A != snes->jacobian_pre) 
    SETERRQ(1,"SNESSolve_Test: Cannot test with alternative preconditioner");

  MPIU_printf(snes->comm,"Testing hand-coded Jacobian, if the ratio is\n");
  MPIU_printf(snes->comm,"O(1.e-8), the hand-coded Jacobian is probably correct.\n");
  if (!neP->complete_print) {
    MPIU_printf(snes->comm,"Run with -snes_test_display to show difference\n");
    MPIU_printf(snes->comm,"of hand-coded and finite difference Jacobian.\n");
  }

  for ( i=0; i<3; i++ ) {
    if (i == 0) {ierr = SNESComputeInitialGuess(snes,x); CHKERRQ(ierr);}
    else if (i == 1) {ierr = VecSet(&mone,x); CHKERRQ(ierr);}
    else {ierr = VecSet(&one,x); CHKERRQ(ierr);}
 
    /* compute both versions of Jacobian */
    ierr = SNESComputeJacobian(snes,x,&A,&A,&flg);CHKERRQ(ierr);
    if (i == 0) {ierr = MatConvert(A,MATSAME,&B); CHKERRQ(ierr);}
    ierr = SNESDefaultComputeJacobian(snes,x,&B,&B,&flg,snes->funP);
    CHKERRQ(ierr);
    if (neP->complete_print) {
      MPIU_printf(snes->comm,"Finite difference Jacobian\n");
      ierr = MatView(B,SYNC_STDOUT_VIEWER); CHKERRQ(ierr);
    }
    /* compare */
    ierr = MatAXPY(&mone,A,B); CHKERRQ(ierr);
    ierr = MatNorm(B,NORM_FROBENIUS,&norm); CHKERRQ(ierr);
    ierr = MatNorm(A,NORM_FROBENIUS,&gnorm); CHKERRQ(ierr);
    if (neP->complete_print) {
      MPIU_printf(snes->comm,"Hand-coded Jacobian\n");
      ierr = MatView(A,SYNC_STDOUT_VIEWER); CHKERRQ(ierr);
    }
    MPIU_printf(snes->comm,"Norm of matrix ratio %g difference %g\n",
                           norm/gnorm,norm);
  }
  ierr = MatDestroy(B); CHKERRQ(ierr);
  return 0;
}
/* ------------------------------------------------------------ */
int SNESDestroy_Test(PetscObject obj)
{
  SNES snes = (SNES) obj;
  SLESDestroy(snes->sles);
  PLogObjectDestroy(obj);
  PETSCHEADERDESTROY(obj);
  return 0;
}

static int SNESPrintHelp_Test(SNES snes)
{
  MPIU_printf(snes->comm,"Test code to compute Jacobian\n");
  MPIU_printf(snes->comm,"-snes_test_display - display difference between\n");
  return 0;
}

static int SNESSetFromOptions_Test(SNES snes)
{
  SNES_Test *ls = (SNES_Test *)snes->data;
  if (OptionsHasName(0,"-snes_test_display")) {
    ls->complete_print = 1;
  }
  return 0;
}

/* ------------------------------------------------------------ */
int SNESCreate_Test(SNES  snes )
{
  SNES_Test *neP;

  if (snes->method_class != SNES_NONLINEAR_EQUATIONS) SETERRQ(1,
    "SNESCreate_Test: Valid for SNES_NONLINEAR_EQUATIONS problems only");
  snes->type		= SNES_NTEST;
  snes->setup		= 0;
  snes->solve		= SNESSolve_Test;
  snes->destroy		= SNESDestroy_Test;
  snes->converged	= SNESDefaultConverged;
  snes->printhelp       = SNESPrintHelp_Test;
  snes->setfromoptions  = SNESSetFromOptions_Test;

  neP			= PETSCNEW(SNES_Test);   CHKPTRQ(neP);
  PLogObjectMemory(snes,sizeof(SNES_Test));
  snes->data    	= (void *) neP;
  neP->complete_print   = 0;
  return 0;
}




