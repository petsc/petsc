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
    snes->vec_sol_update = PETSC_NULL;
  }
  if (snes->nwork) {
    ierr = VecDestroyVecs(snes->work, snes->nwork);CHKERRQ(ierr);
    snes->nwork = 0;
    snes->work  = PETSC_NULL;
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
  if (!snes->vec_sol_update) {
    ierr = VecDuplicate(snes->vec_sol, &snes->vec_sol_update);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(snes, snes->vec_sol_update);CHKERRQ(ierr);
  }
  if (!snes->work) {
    snes->nwork = 1;
    ierr = VecDuplicateVecs(snes->vec_sol, snes->nwork, &snes->work);CHKERRQ(ierr);
    ierr = PetscLogObjectParents(snes,snes->nwork, snes->work);CHKERRQ(ierr);
  }
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
  const char    *types[] = {"basic", "quadratic", "cubic"};
  PetscInt       indx = 0;
  PetscTruth     flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("SNES Picard options");CHKERRQ(ierr);
    ierr = PetscOptionsEList("-snes_picard","Picard Type","SNESLineSearchSet",types,3,"basic",&indx,&flg);CHKERRQ(ierr);
    ls->type = indx;
    if (flg) {
      switch (indx) {
      case 0:
        ierr = SNESLineSearchSet(snes,SNESLineSearchNo,PETSC_NULL);CHKERRQ(ierr);
        break;
      case 1:
        ierr = SNESLineSearchSet(snes,SNESLineSearchQuadratic,PETSC_NULL);CHKERRQ(ierr);
        break;
      case 2:
        ierr = SNESLineSearchSet(snes,SNESLineSearchCubic,PETSC_NULL);CHKERRQ(ierr);
        break;
      }
    }
    ls->alpha = 1.0;
    ierr = PetscOptionsReal("-snes_picard_alpha","Momentum parameter","SNES",ls->alpha,&ls->alpha,&flg);CHKERRQ(ierr);
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
  SNES_Picard   *neP = (SNES_Picard *) snes->data;
  Vec            X, Y, F, W;
  PetscReal      alpha = neP->alpha;
  PetscReal      fnorm;
  PetscInt       maxits, i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  snes->reason = SNES_CONVERGED_ITERATING;

  maxits = snes->max_its;	     /* maximum number of iterations */
  X      = snes->vec_sol;	     /* X^n */
  Y      = snes->vec_sol_update; /* \tilde X */
  F      = snes->vec_func;       /* residual vector */
  W      = snes->work[0];        /* work vector */

  ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
  snes->iter = 0;
  snes->norm = 0;
  ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
  ierr = SNESComputeFunction(snes,X,F);CHKERRQ(ierr);
  if (snes->domainerror) {
    snes->reason = SNES_DIVERGED_FUNCTION_DOMAIN;
    PetscFunctionReturn(0);
  }
  ierr = VecNorm(F, NORM_2, &fnorm);CHKERRQ(ierr); /* fnorm <- ||F||  */
  if PetscIsInfOrNanReal(fnorm) SETERRQ(PETSC_ERR_FP,"Infinite or not-a-number generated in norm");
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
    PetscTruth lsSuccess = PETSC_TRUE;

    /* Call general purpose update function */
    if (snes->ops->update) {
      ierr = (*snes->ops->update)(snes, snes->iter);CHKERRQ(ierr);
    }
    if (neP->type == 0) {
      ierr = PetscPrintf(snes->hdr.comm, "Fixed alpha = %g\n", alpha);CHKERRQ(ierr);
      /* Update guess Y = X^n - F(X^n) */
      ierr = VecWAXPY(Y, -1.0, F, X);CHKERRQ(ierr);
      /* X^{n+1} = (1 - \alpha) X^n + \alpha Y */
      ierr = VecAXPBY(X, alpha, 1 - alpha, Y);CHKERRQ(ierr);
      /* Compute F(X^{new}) */
      ierr = SNESComputeFunction(snes, X, F);CHKERRQ(ierr);
      ierr = VecNorm(F, NORM_2, &fnorm);CHKERRQ(ierr);
      if PetscIsInfOrNanReal(fnorm) SETERRQ(PETSC_ERR_FP,"Infinite or not-a-number generated norm");
    } else {
      /* Compute a (scaled) negative update in the line search routine: 
         Y <- X - lambda*Y 
         and evaluate G = function(Y) (depends on the line search). */
#if 0
      ierr = (*neP->LineSearch)(snes, PETSC_NULL/*neP->lsP*/, X, F, F/*G*/, F/*Y*/, X/*W*/, fnorm, 0.0, &dummyNorm,  &fnorm,  &lsSuccess);CHKERRQ(ierr);
#else
      /* Put this in function after it works */
      PetscReal norms[4];
      PetscInt  i;

      norms[0] = fnorm;
      /* Calculate the solution increment, Y = X^n - F(X^n) */
      ierr = VecWAXPY(Y, -1.0, F, X);CHKERRQ(ierr);
      /* Calculate trial solutions */
#if 1
      for(alpha = 0.5, i = 1; alpha <= 1.0; alpha += 0.5, ++i) {
        /* Calculate X^{n+1} = (1 - \alpha) X^n + \alpha Y */
        ierr = VecCopy(X, W);CHKERRQ(ierr);
        ierr = VecAXPBY(W, alpha, 1 - alpha, Y);CHKERRQ(ierr);
        ierr = SNESComputeFunction(snes, W, F);CHKERRQ(ierr);
        ierr = VecNorm(F, NORM_2, &norms[i]);CHKERRQ(ierr);
        norms[i] = PetscSqr(norms[i]);
      }
      /* Fit a quadratic:
           If we have x_{0,1,2} = 0, 0.5, 1.0 which generate norms y_{0,1,2}
           a =  2 y_3 + 2 y_1 - 4 y_2
           b = -3 y_1 + 4 y_2 -   y_3
           c =    y_1
           x_min = -b/2a = (3 y_1 - 4 y_2 + y_3)/(4 y_3 + 4 y_1 - 8 y_2)
      */
      if (2.0*norms[2] + 2.0*norms[0] - 4.0*norms[1] >= 0.0) {
        alpha = (3.0*norms[0] - 4.0*norms[1] + norms[2])/(4.0*norms[2] + 4.0*norms[0] - 8.0*norms[1]);
        alpha = PetscMin(alpha, 1.0);
        alpha = PetscMax(alpha, 0.0);
        alpha *= 0.9;
      } else {
        alpha = 1.0;
      }
      ierr = PetscPrintf(snes->hdr.comm, "norms[0] = %g, norms[1] = %g, norms[2] = %g\n", norms[0], norms[1], norms[2]);CHKERRQ(ierr);
      ierr = PetscPrintf(snes->hdr.comm, "Choose alpha = %g\n", alpha);CHKERRQ(ierr);
      ierr = VecAXPBY(X, alpha, 1 - alpha, Y);CHKERRQ(ierr);
      ierr = SNESComputeFunction(snes, X, F);CHKERRQ(ierr);
      ierr = VecNorm(F, NORM_2, &fnorm);CHKERRQ(ierr);
#else
      for(alpha = 0.333333, i = 1; alpha <= 1.0; alpha += 0.333333, ++i) {
        /* Calculate X^{n+1} = (1 - \alpha) X^n + \alpha Y */
        ierr = VecWAXPBY(W, X, alpha, 1 - alpha, Y);CHKERRQ(ierr);
        ierr = SNESComputeFunction(snes, W, F);CHKERRQ(ierr);
        ierr = VecNorm(F, NORM_2, &norms[i]);CHKERRQ(ierr);
      }
#endif
#endif      
    }
    if (!lsSuccess) {
      if (++snes->numFailures >= snes->maxFailures) {
        snes->reason = SNES_DIVERGED_LS_FAILURE;
        break;
      }
    }
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

typedef PetscErrorCode (*FCN1)(SNES,Vec,Vec,void*,PetscTruth*);                 /* force argument to next function to not be extern C*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "SNESLineSearchSetPreCheck_Picard"
PetscErrorCode PETSCSNES_DLLEXPORT SNESLineSearchSetPreCheck_Picard(SNES snes, FCN1 func, void *checkctx)
{
  PetscFunctionBegin;
  ((SNES_Picard *)(snes->data))->precheckstep = func;
  ((SNES_Picard *)(snes->data))->precheck     = checkctx;
  PetscFunctionReturn(0);
}
EXTERN_C_END

typedef PetscErrorCode (*FCN2)(SNES,void*,Vec,Vec,Vec,Vec,Vec,PetscReal,PetscReal,PetscReal*,PetscReal*,PetscTruth*); /* force argument to next function to not be extern C*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "SNESLineSearchSet_Picard"
PetscErrorCode PETSCSNES_DLLEXPORT SNESLineSearchSet_Picard(SNES snes, FCN2 func, void *lsctx)
{
  PetscFunctionBegin;
  ((SNES_Picard *)(snes->data))->LineSearch = func;
  ((SNES_Picard *)(snes->data))->lsP        = lsctx;
  PetscFunctionReturn(0);
}
EXTERN_C_END

typedef PetscErrorCode (*FCN3)(SNES,Vec,Vec,Vec,void*,PetscTruth*,PetscTruth*); /* force argument to next function to not be extern C*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "SNESLineSearchSetPostCheck_Picard"
PetscErrorCode PETSCSNES_DLLEXPORT SNESLineSearchSetPostCheck_Picard(SNES snes, FCN3 func, void *checkctx)
{
  PetscFunctionBegin;
  ((SNES_Picard *)(snes->data))->postcheckstep = func;
  ((SNES_Picard *)(snes->data))->postcheck     = checkctx;
  PetscFunctionReturn(0);
}
EXTERN_C_END

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
  snes->data = (void*) neP;
  neP->type  = 0;
  neP->alpha		 = 1.e-4;
  neP->maxstep		 = 1.e8;
  neP->steptol       = 1.e-12;
  neP->LineSearch    = SNESLineSearchNo;
  neP->lsP           = PETSC_NULL;
  neP->postcheckstep = PETSC_NULL;
  neP->postcheck     = PETSC_NULL;
  neP->precheckstep  = PETSC_NULL;
  neP->precheck      = PETSC_NULL;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)snes,"SNESLineSearchSet_C",
					   "SNESLineSearchSet_Picard",
					   SNESLineSearchSet_Picard);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)snes,"SNESLineSearchSetPostCheck_C",
					   "SNESLineSearchSetPostCheck_Picard",
					   SNESLineSearchSetPostCheck_Picard);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)snes,"SNESLineSearchSetPreCheck_C",
					   "SNESLineSearchSetPreCheck_Picard",
					   SNESLineSearchSetPreCheck_Picard);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
