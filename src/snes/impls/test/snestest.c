/*$Id: snestest.c,v 1.50 2000/08/14 20:02:01 bsmith Exp bsmith $*/

#include "src/snes/snesimpl.h"

typedef struct {
  PetscTruth complete_print;
} SNES_Test;

/*
     SNESSolve_Test - Tests whether a hand computed Jacobian 
     matches one compute via finite differences.
*/
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"SNESSolve_Test"
int SNESSolve_Test(SNES snes,int *its)
{
  Mat          A = snes->jacobian,B;
  Vec          x = snes->vec_sol;
  int          ierr,i;
  MatStructure flg;
  Scalar       mone = -1.0,one = 1.0;
  double       norm,gnorm;
  SNES_Test    *neP = (SNES_Test*)snes->data;

  PetscFunctionBegin;
  *its = 0;

  if (A != snes->jacobian_pre) {
    SETERRQ(PETSC_ERR_ARG_WRONG,0,"Cannot test with alternative preconditioner");
  }

  ierr = PetscPrintf(snes->comm,"Testing hand-coded Jacobian, if the ratio is\n");CHKERRQ(ierr);
  ierr = PetscPrintf(snes->comm,"O(1.e-8), the hand-coded Jacobian is probably correct.\n");CHKERRQ(ierr);
  if (!neP->complete_print) {
    ierr = PetscPrintf(snes->comm,"Run with -snes_test_display to show difference\n");CHKERRQ(ierr);
    ierr = PetscPrintf(snes->comm,"of hand-coded and finite difference Jacobian.\n");CHKERRQ(ierr);
  }

  for (i=0; i<3; i++) {
    if (i == 1) {ierr = VecSet(&mone,x);CHKERRQ(ierr);}
    else if (i == 2) {ierr = VecSet(&one,x);CHKERRQ(ierr);}
 
    /* compute both versions of Jacobian */
    ierr = SNESComputeJacobian(snes,x,&A,&A,&flg);CHKERRQ(ierr);
    if (!i) {ierr = MatConvert(A,MATSAME,&B);CHKERRQ(ierr);}
    ierr = SNESDefaultComputeJacobian(snes,x,&B,&B,&flg,snes->funP);CHKERRQ(ierr);
    if (neP->complete_print) {
      ierr = PetscPrintf(snes->comm,"Finite difference Jacobian\n");CHKERRQ(ierr);
      ierr = MatView(B,VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    }
    /* compare */
    ierr = MatAXPY(&mone,A,B);CHKERRQ(ierr);
    ierr = MatNorm(B,NORM_FROBENIUS,&norm);CHKERRQ(ierr);
    ierr = MatNorm(A,NORM_FROBENIUS,&gnorm);CHKERRQ(ierr);
    if (neP->complete_print) {
      ierr = PetscPrintf(snes->comm,"Hand-coded Jacobian\n");CHKERRQ(ierr);
      ierr = MatView(A,VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    }
    ierr = PetscPrintf(snes->comm,"Norm of matrix ratio %g difference %g\n",norm/gnorm,norm);CHKERRQ(ierr);
  }
  ierr = MatDestroy(B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* ------------------------------------------------------------ */
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"SNESDestroy_Test"
int SNESDestroy_Test(SNES snes)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"SNESSetFromOptions_Test"
static int SNESSetFromOptions_Test(SNES snes)
{
  SNES_Test  *ls = (SNES_Test *)snes->data;
  int        ierr;

  PetscFunctionBegin;

  ierr = OptionsBegin(snes->comm,snes->prefix,"Hand-coded Jacobian tester");CHKERRQ(ierr);
    ierr = OptionsName("-snes_test_display","Display difference between approximate and handcoded Jacobian","None",&ls->complete_print);CHKERRQ(ierr);
  ierr = OptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------ */
EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"SNESCreate_Test"
int SNESCreate_Test(SNES  snes)
{
  SNES_Test *neP;

  PetscFunctionBegin;
  if (snes->method_class != SNES_NONLINEAR_EQUATIONS) {
    SETERRQ(PETSC_ERR_ARG_WRONG,0,"SNES_NONLINEAR_EQUATIONS only");
  }
  snes->setup		= 0;
  snes->solve		= SNESSolve_Test;
  snes->destroy		= SNESDestroy_Test;
  snes->converged	= SNESConverged_EQ_LS;
  snes->setfromoptions  = SNESSetFromOptions_Test;

  neP			= PetscNew(SNES_Test);CHKPTRQ(neP);
  PLogObjectMemory(snes,sizeof(SNES_Test));
  snes->data    	= (void*)neP;
  neP->complete_print   = PETSC_FALSE;
  PetscFunctionReturn(0);
}
EXTERN_C_END




