#include <petsc/private/linesearchimpl.h>
#include <petscsnes.h>

static PetscErrorCode  SNESLineSearchApply_L2(SNESLineSearch linesearch)
{
  PetscBool      changed_y, changed_w;
  PetscErrorCode ierr;
  Vec            X;
  Vec            F;
  Vec            Y;
  Vec            W;
  SNES           snes;
  PetscReal      gnorm;
  PetscReal      ynorm;
  PetscReal      xnorm;
  PetscReal      steptol, maxstep, rtol, atol, ltol;
  PetscViewer    monitor;
  PetscReal      lambda, lambda_old, lambda_mid, lambda_update, delLambda;
  PetscReal      fnrm, fnrm_old, fnrm_mid;
  PetscReal      delFnrm, delFnrm_old, del2Fnrm;
  PetscInt       i, max_its;
  PetscErrorCode (*objective)(SNES,Vec,PetscReal*,void*);

  PetscFunctionBegin;
  ierr = SNESLineSearchGetVecs(linesearch, &X, &F, &Y, &W, NULL);CHKERRQ(ierr);
  ierr = SNESLineSearchGetNorms(linesearch, &xnorm, &gnorm, &ynorm);CHKERRQ(ierr);
  ierr = SNESLineSearchGetLambda(linesearch, &lambda);CHKERRQ(ierr);
  ierr = SNESLineSearchGetSNES(linesearch, &snes);CHKERRQ(ierr);
  ierr = SNESLineSearchSetReason(linesearch, SNES_LINESEARCH_SUCCEEDED);CHKERRQ(ierr);
  ierr = SNESLineSearchGetTolerances(linesearch, &steptol, &maxstep, &rtol, &atol, &ltol, &max_its);CHKERRQ(ierr);
  ierr = SNESLineSearchGetDefaultMonitor(linesearch, &monitor);CHKERRQ(ierr);

  ierr = SNESGetObjective(snes,&objective,NULL);CHKERRQ(ierr);

  /* precheck */
  ierr       = SNESLineSearchPreCheck(linesearch,X,Y,&changed_y);CHKERRQ(ierr);
  lambda_old = 0.0;
  if (!objective) {
    fnrm_old = gnorm*gnorm;
  } else {
    ierr = SNESComputeObjective(snes,X,&fnrm_old);CHKERRQ(ierr);
  }
  lambda_mid = 0.5*(lambda + lambda_old);

  for (i = 0; i < max_its; i++) {

    while (PETSC_TRUE) {
      ierr = VecCopy(X, W);CHKERRQ(ierr);
      ierr = VecAXPY(W, -lambda_mid, Y);CHKERRQ(ierr);
      if (linesearch->ops->viproject) {
        ierr = (*linesearch->ops->viproject)(snes, W);CHKERRQ(ierr);
      }
      if (!objective) {
        /* compute the norm at the midpoint */
        ierr = (*linesearch->ops->snesfunc)(snes, W, F);CHKERRQ(ierr);
        if (linesearch->ops->vinorm) {
          fnrm_mid = gnorm;
          ierr     = (*linesearch->ops->vinorm)(snes, F, W, &fnrm_mid);CHKERRQ(ierr);
        } else {
          ierr = VecNorm(F,NORM_2,&fnrm_mid);CHKERRQ(ierr);
        }

        /* compute the norm at the new endpoit */
        ierr = VecCopy(X, W);CHKERRQ(ierr);
        ierr = VecAXPY(W, -lambda, Y);CHKERRQ(ierr);
        if (linesearch->ops->viproject) {
          ierr = (*linesearch->ops->viproject)(snes, W);CHKERRQ(ierr);
        }
        ierr = (*linesearch->ops->snesfunc)(snes, W, F);CHKERRQ(ierr);
        if (linesearch->ops->vinorm) {
          fnrm = gnorm;
          ierr = (*linesearch->ops->vinorm)(snes, F, W, &fnrm);CHKERRQ(ierr);
        } else {
          ierr = VecNorm(F,NORM_2,&fnrm);CHKERRQ(ierr);
        }
        fnrm_mid = fnrm_mid*fnrm_mid;
        fnrm = fnrm*fnrm;
      } else {
        /* compute the objective at the midpoint */
        ierr = VecCopy(X, W);CHKERRQ(ierr);
        ierr = VecAXPY(W, -lambda_mid, Y);CHKERRQ(ierr);
        ierr = SNESComputeObjective(snes,W,&fnrm_mid);CHKERRQ(ierr);

        /* compute the objective at the new endpoint */
        ierr = VecCopy(X, W);CHKERRQ(ierr);
        ierr = VecAXPY(W, -lambda, Y);CHKERRQ(ierr);
        ierr = SNESComputeObjective(snes,W,&fnrm);CHKERRQ(ierr);
      }
      if (!PetscIsInfOrNanReal(fnrm)) break;
      if (monitor) {
        ierr = PetscViewerASCIIAddTab(monitor,((PetscObject)linesearch)->tablevel);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(monitor,"    Line search: objective function at lambdas = %g is Inf or Nan, cutting lambda\n",(double)lambda);CHKERRQ(ierr);
        ierr = PetscViewerASCIISubtractTab(monitor,((PetscObject)linesearch)->tablevel);CHKERRQ(ierr);
      }
      if (lambda <= steptol) {
        ierr = SNESLineSearchSetReason(linesearch, SNES_LINESEARCH_FAILED_REDUCT);CHKERRQ(ierr);
        PetscFunctionReturn(0);
      }
      maxstep = .95*lambda; /* forbid the search from ever going back to the "failed" length that generates Nan or Inf */
      lambda  = .5*(lambda + lambda_old);
      lambda_mid = .5*(lambda + lambda_old);
    }

    delLambda   = lambda - lambda_old;
    /* compute f'() at the end points using second order one sided differencing */
    delFnrm     = (3.*fnrm - 4.*fnrm_mid + 1.*fnrm_old) / delLambda;
    delFnrm_old = (-3.*fnrm_old + 4.*fnrm_mid -1.*fnrm) / delLambda;
    /* compute f''() at the midpoint using centered differencing */
    del2Fnrm    = (delFnrm - delFnrm_old) / delLambda;

    if (monitor) {
      ierr = PetscViewerASCIIAddTab(monitor,((PetscObject)linesearch)->tablevel);CHKERRQ(ierr);
      if (!objective) {
        ierr = PetscViewerASCIIPrintf(monitor,"    Line search: lambdas = [%g, %g, %g], fnorms = [%g, %g, %g]\n",(double)lambda, (double)lambda_mid, (double)lambda_old, (double)PetscSqrtReal(fnrm), (double)PetscSqrtReal(fnrm_mid), (double)PetscSqrtReal(fnrm_old));CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIIPrintf(monitor,"    Line search: lambdas = [%g, %g, %g], obj = [%g, %g, %g]\n",(double)lambda, (double)lambda_mid, (double)lambda_old, (double)fnrm, (double)fnrm_mid, (double)fnrm_old);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIISubtractTab(monitor,((PetscObject)linesearch)->tablevel);CHKERRQ(ierr);
    }

    /* compute the secant (Newton) update -- always go downhill */
    if (del2Fnrm > 0.) lambda_update = lambda - delFnrm / del2Fnrm;
    else if (del2Fnrm < 0.) lambda_update = lambda + delFnrm / del2Fnrm;
    else break;

    if (lambda_update < steptol) lambda_update = 0.5*(lambda + lambda_old);

    if (PetscIsInfOrNanReal(lambda_update)) break;

    if (lambda_update > maxstep) break;

    /* update the endpoints and the midpoint of the bracketed secant region */
    lambda_old = lambda;
    lambda     = lambda_update;
    fnrm_old   = fnrm;
    lambda_mid = 0.5*(lambda + lambda_old);
  }
  /* construct the solution */
  ierr = VecCopy(X, W);CHKERRQ(ierr);
  ierr = VecAXPY(W, -lambda, Y);CHKERRQ(ierr);
  if (linesearch->ops->viproject) {
    ierr = (*linesearch->ops->viproject)(snes, W);CHKERRQ(ierr);
  }

  /* postcheck */
  ierr = SNESLineSearchPostCheck(linesearch,X,Y,W,&changed_y,&changed_w);CHKERRQ(ierr);
  if (changed_y) {
    ierr = VecAXPY(X, -lambda, Y);CHKERRQ(ierr);
    if (linesearch->ops->viproject) {
      ierr = (*linesearch->ops->viproject)(snes, X);CHKERRQ(ierr);
    }
  } else {
    ierr = VecCopy(W, X);CHKERRQ(ierr);
  }
  ierr = (*linesearch->ops->snesfunc)(snes,X,F);CHKERRQ(ierr);

  ierr = SNESLineSearchSetLambda(linesearch, lambda);CHKERRQ(ierr);
  ierr = SNESLineSearchComputeNorms(linesearch);CHKERRQ(ierr);
  ierr = SNESLineSearchGetNorms(linesearch, &xnorm, &gnorm, &ynorm);CHKERRQ(ierr);

  if (monitor) {
    ierr = PetscViewerASCIIAddTab(monitor,((PetscObject)linesearch)->tablevel);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(monitor,"    Line search terminated: lambda = %g, fnorms = %g\n", (double)lambda, (double)gnorm);CHKERRQ(ierr);
    ierr = PetscViewerASCIISubtractTab(monitor,((PetscObject)linesearch)->tablevel);CHKERRQ(ierr);
  }
  if (lambda <= steptol) {
    ierr = SNESLineSearchSetReason(linesearch, SNES_LINESEARCH_FAILED_REDUCT);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*MC
   SNESLINESEARCHL2 - Secant search in the L2 norm of the function or the objective function, if it is provided with SNESSetObjective().

   Attempts to solve min_lambda f(x + lambda y) using the secant method with the initial bracketing of lambda between [0,damping]. Differences of f()
   are used to approximate the first and second derivative of f() with respect to lambda, f'() and f''(). The secant method is run for maxit iterations.

   When an objective function is provided f(w) is the objective function otherwise f(w) = ||F(w)||^2. x is the current step and y is the search direction.

   This has no checks on whether the secant method is actually converging.

   Options Database Keys:
+  -snes_linesearch_max_it <maxit> - maximum number of iterations, default is 1
.  -snes_linesearch_maxstep <length> - the algorithm insures that a step length is never longer than this value
.  -snes_linesearch_damping <damping> - initial step is scaled back by this factor, default is 1.0
-  -snes_linesearch_minlambda <minlambda> - minimum allowable lambda

   Level: advanced

   Developer Notes:
    A better name for this method might be SNESLINESEARCHSECANT, L2 is not descriptive

.seealso: SNESLINESEARCHBT, SNESLINESEARCHCP, SNESLineSearch, SNESLineSearchCreate(), SNESLineSearchSetType()
M*/
PETSC_EXTERN PetscErrorCode SNESLineSearchCreate_L2(SNESLineSearch linesearch)
{
  PetscFunctionBegin;
  linesearch->ops->apply          = SNESLineSearchApply_L2;
  linesearch->ops->destroy        = NULL;
  linesearch->ops->setfromoptions = NULL;
  linesearch->ops->reset          = NULL;
  linesearch->ops->view           = NULL;
  linesearch->ops->setup          = NULL;

  linesearch->max_its = 1;
  PetscFunctionReturn(0);
}
