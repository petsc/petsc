
#ifndef lint
static char vcid[] = "$Id: snestest.c,v 1.6 1995/05/18 22:48:08 bsmith Exp bsmith $";
#endif

#include "draw.h"
#include "snesimpl.h"

typedef struct {
  int complete_print;
} SNES_Test;

/*
     SNESSolve_Test - Tests whether a hand computed Jacobian 
        matches one compute via finite differences

  Input Parameters:

  Output Parameters:

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

  if (A != snes->jacobian_pre) SETERR(1,"Cannot test with alternative pre");

  MPIU_printf(snes->comm,"Testing handcoded Jacobian, if the ratio is\n");
  MPIU_printf(snes->comm,"O(1.e-8) it is probably correct.\n");
  if (!neP->complete_print) {
    MPIU_printf(snes->comm,"Run with -snes_test_display to show difference\n");
    MPIU_printf(snes->comm,"of hand coding and finite difference Jacobian.\n");
  }

  for ( i=0; i<3; i++ ) {
    if (i == 0) {ierr = SNESComputeInitialGuess(snes,x); CHKERR(ierr);}
    else if (i == 1) {VecSet(&mone,x);}
    else {VecSet(&one,x);}
 
    /* compute both versions of Jacobian */
    ierr = (*snes->ComputeJacobian)(snes,x,&A,&A,&flg,snes->jacP);CHKERR(ierr);
    if (i == 0) MatConvert(A,MATSAME,&B); 
    ierr = SNESDefaultComputeJacobian(snes,x,&B,&B,&flg,snes->funP);
    CHKERR(ierr);
    if (neP->complete_print) {
      MPIU_printf(snes->comm,"Finite difference Jacobian\n");
      MatView(B,SYNC_STDOUT_VIEWER);
    }
    /* compare */
    ierr = MatAXPY(&mone,A,B);
    MatNorm(B,NORM_FROBENIUS,&norm);
    MatNorm(A,NORM_FROBENIUS,&gnorm);
    if (neP->complete_print) {
      MPIU_printf(snes->comm,"Hand-coded Jacobian\n");
      MatView(A,SYNC_STDOUT_VIEWER);
    }
    MPIU_printf(snes->comm,"Norm of matrix ratio %g difference %g\n",
                           norm/gnorm,norm);
  }
  MatDestroy(B);
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
  fprintf(stderr,"Test code to compute Jacobian\n");
  fprintf(stderr,"-snes_test_display - display difference between\n");
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

  snes->type		= SNES_NTEST;
  snes->setup		= 0;
  snes->solve		= SNESSolve_Test;
  snes->destroy		= SNESDestroy_Test;
  snes->Converged	= SNESDefaultConverged;
  snes->printhelp       = SNESPrintHelp_Test;
  snes->setfromoptions  = SNESSetFromOptions_Test;

  neP			= NEW(SNES_Test);   CHKPTR(neP);
  snes->data    	= (void *) neP;
  neP->complete_print   = 0;
  return 0;
}




