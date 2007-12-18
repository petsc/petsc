#define PETSCSNES_DLL

#include "src/snes/impls/picard/picard.h"

/*
  SNESDestroy_Picard - Destroys the private SNES_Picard context that was created with SNESCreate_Picard().

  Input Parameter:
. snes - the SNES context

  Application Interface Routine: SNESDestroy()
*/
#undef __FUNCT__  
#define __FUNCT__ "SNESDestroy_Picard"
PetscErrorCode SNESDestroy_Picard(SNES snes)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (snes->vec_sol_update) {
    ierr = VecDestroy(snes->vec_sol_update);CHKERRQ(ierr);
  }
  ierr = PetscFree(snes->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   SNESSetUp_Picard - Sets up the internal data structures for the later use
   of the SNESPICARD nonlinear solver.

   Input Parameters:
+  snes - the SNES context
-  x - the solution vector

   Application Interface Routine: SNESSetUp()
 */
#undef __FUNCT__  
#define __FUNCT__ "SNESSetUp_Picard"
PetscErrorCode SNESSetUp_Picard(SNES snes)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDuplicate(snes->vec_sol, &snes->vec_sol_update);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  SNESSetFromOptions_Picard - Sets various parameters for the SNESLS method.

  Input Parameter:
. snes - the SNES context

  Application Interface Routine: SNESSetFromOptions()
*/
#undef __FUNCT__  
#define __FUNCT__ "SNESSetFromOptions_Picard"
static PetscErrorCode SNESSetFromOptions_Picard(SNES snes)
{
  SNES_Picard   *ls = (SNES_Picard *)snes->data;
  const char    *types[] = {"basic"};
  PetscInt       indx;
  PetscTruth     flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("SNES Picard options");CHKERRQ(ierr);
    ierr = PetscOptionsEList("-snes_picard","Picard Type","SNESLineSearchSet",types,1,"basic",&indx,&flg);CHKERRQ(ierr);
    if (flg) {
      ls->type = indx;
    }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  SNESView_Picard - Prints info from the SNESPICARD data structure.

  Input Parameters:
+ SNES - the SNES context
- viewer - visualization context

  Application Interface Routine: SNESView()
*/
#undef __FUNCT__  
#define __FUNCT__ "SNESView_Picard"
static PetscErrorCode SNESView_Picard(SNES snes, PetscViewer viewer)
{
  SNES_Picard   *ls = (SNES_Picard *)snes->data;
  const char    *cstr;
  PetscTruth     iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject) viewer, PETSC_VIEWER_ASCII, &iascii);CHKERRQ(ierr);
  if (iascii) {
    switch(ls->type) {
    case 0:
      cstr = "basic";
      break;
    default:
      cstr = "unknown";
    }
    ierr = PetscViewerASCIIPrintf(viewer,"  picard variant: %s\n", cstr);CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_ERR_SUP, "Viewer type %s not supported for SNES Picard", ((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

/*
  SNESSolve_Picard - Solves a nonlinear system with the Picard method.

  Input Parameters:
. snes - the SNES context

  Output Parameter:
. outits - number of iterations until termination

  Application Interface Routine: SNESSolve()
*/
#undef __FUNCT__  
#define __FUNCT__ "SNESSolve_Picard"
PetscErrorCode SNESSolve_Picard(SNES snes)
{ 
  /* SNES_Picard   *neP = (SNES_Picard *) snes->data; */
  Vec            X, Y, F;
  PetscReal      alpha = 1.0;
  PetscReal      fnorm;
  PetscInt       maxits, i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  snes->reason = SNES_CONVERGED_ITERATING;

  maxits = snes->max_its;	     /* maximum number of iterations */
  X      = snes->vec_sol;	     /* X^n */
  Y      = snes->vec_sol_update; /* \tilde X */
  F      = snes->vec_func;       /* residual vector */

  ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
  snes->iter = 0;
  snes->norm = 0;
  ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
  ierr = SNESComputeFunction(snes,X,F);CHKERRQ(ierr);
  if (snes->domainerror) {
    snes->reason = SNES_DIVERGED_FUNCTION_DOMAIN;
    PetscFunctionReturn(0);
  }
  ierr = VecNorm(F, NORM_2, &fnorm);CHKERRQ(ierr);CHKFPQ(fnorm); /* fnorm <- ||F||  */
  ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
  snes->norm = fnorm;
  ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
  SNESLogConvHistory(snes,fnorm,0);
  SNESMonitor(snes,0,fnorm);

  /* set parameter for default relative tolerance convergence test */
  snes->ttol = fnorm*snes->rtol;
  /* test convergence */
  ierr = (*snes->ops->converged)(snes,0,0.0,0.0,fnorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
  if (snes->reason) PetscFunctionReturn(0);

  for(i = 0; i < maxits; i++) {
    /* Update guess */
    ierr = VecWAXPY(Y, -1.0, F, X);CHKERRQ(ierr);
    /* X^{n+1} = (1 - \alpha) X^n + \alpha Y */
    ierr = VecAXPBY(X, alpha, 1 - alpha, Y);CHKERRQ(ierr);
    /* Compute F(X^{new}) */
    ierr = SNESComputeFunction(snes, X, F);CHKERRQ(ierr);
    ierr = VecNorm(F, NORM_2, &fnorm);CHKERRQ(ierr);CHKFPQ(fnorm);
    if (snes->nfuncs >= snes->max_funcs) {
      snes->reason = SNES_DIVERGED_FUNCTION_COUNT;
      break;
    }
    if (snes->domainerror) {
      snes->reason = SNES_DIVERGED_FUNCTION_DOMAIN;
      PetscFunctionReturn(0);
    }
    /* Monitor convergence */
    ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
    snes->iter = i+1;
    snes->norm = fnorm;
    ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
    SNESLogConvHistory(snes,snes->norm,0);
    SNESMonitor(snes,snes->iter,snes->norm);
    /* Test for convergence */
    ierr = (*snes->ops->converged)(snes,snes->iter,0.0,0.0,fnorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
    if (snes->reason) break;
  }
  if (i == maxits) {
    ierr = PetscInfo1(snes, "Maximum number of iterations has been reached: %D\n", maxits);CHKERRQ(ierr);
    if (!snes->reason) snes->reason = SNES_DIVERGED_MAX_IT;
  }
  PetscFunctionReturn(0);
}

/*MC
  SNESPICARD - Picard nonlinear solver that uses successive substitutions

  Level: beginner

.seealso:  SNESCreate(), SNES, SNESSetType(), SNESLS, SNESTR
M*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "SNESCreate_Picard"
PetscErrorCode PETSCSNES_DLLEXPORT SNESCreate_Picard(SNES snes)
{
  SNES_Picard   *neP;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  snes->ops->destroy	    = SNESDestroy_Picard;
  snes->ops->setup	        = SNESSetUp_Picard;
  snes->ops->setfromoptions = SNESSetFromOptions_Picard;
  snes->ops->view           = SNESView_Picard;
  snes->ops->solve	        = SNESSolve_Picard;

  ierr = PetscNewLog(snes, SNES_Picard, &neP);CHKERRQ(ierr);
  snes->data = (void*)neP;
  neP->type  = 0;

  PetscFunctionReturn(0);
}
EXTERN_C_END
