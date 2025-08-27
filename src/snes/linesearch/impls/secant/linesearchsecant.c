#include <petsc/private/linesearchimpl.h>
#include <petscsnes.h>

static PetscErrorCode SNESLineSearchApply_Secant(SNESLineSearch linesearch)
{
  PetscBool        changed_y, changed_w;
  Vec              X;
  Vec              F;
  Vec              Y;
  Vec              W;
  SNES             snes;
  PetscReal        gnorm;
  PetscReal        ynorm;
  PetscReal        xnorm;
  PetscReal        minlambda, maxlambda, atol, ltol;
  PetscViewer      monitor;
  PetscReal        lambda, lambda_old, lambda_mid, lambda_update, delLambda;
  PetscReal        fnrm, fnrm_old, fnrm_mid;
  PetscReal        delFnrm, delFnrm_old, del2Fnrm;
  PetscInt         i, max_it;
  SNESObjectiveFn *objective;

  PetscFunctionBegin;
  PetscCall(SNESLineSearchGetVecs(linesearch, &X, &F, &Y, &W, NULL));
  PetscCall(SNESLineSearchGetNorms(linesearch, &xnorm, &gnorm, &ynorm));
  PetscCall(SNESLineSearchGetLambda(linesearch, &lambda));
  PetscCall(SNESLineSearchGetSNES(linesearch, &snes));
  PetscCall(SNESLineSearchSetReason(linesearch, SNES_LINESEARCH_SUCCEEDED));
  PetscCall(SNESLineSearchGetTolerances(linesearch, &minlambda, &maxlambda, NULL, &atol, &ltol, &max_it));
  PetscCall(SNESLineSearchGetDefaultMonitor(linesearch, &monitor));

  PetscCall(SNESGetObjective(snes, &objective, NULL));

  /* precheck */
  PetscCall(SNESLineSearchPreCheck(linesearch, X, Y, &changed_y));
  lambda_old = 0.0;
  if (!objective) {
    fnrm_old = gnorm * gnorm;
  } else {
    PetscCall(SNESComputeObjective(snes, X, &fnrm_old));
  }
  lambda_mid = 0.5 * (lambda + lambda_old);

  for (i = 0; i < max_it; i++) {
    /* check whether new lambda is NaN or Inf - if so, iteratively shrink towards lambda_old */
    while (PETSC_TRUE) {
      PetscCall(VecWAXPY(W, -lambda_mid, Y, X));
      if (linesearch->ops->viproject) PetscCall((*linesearch->ops->viproject)(snes, W));
      if (!objective) {
        /* compute the norm at the midpoint */
        PetscCall((*linesearch->ops->snesfunc)(snes, W, F));
        if (linesearch->ops->vinorm) {
          fnrm_mid = gnorm;
          PetscCall((*linesearch->ops->vinorm)(snes, F, W, &fnrm_mid));
        } else {
          PetscCall(VecNorm(F, NORM_2, &fnrm_mid));
        }

        /* compute the norm at the new endpoint */
        PetscCall(VecWAXPY(W, -lambda, Y, X));
        if (linesearch->ops->viproject) PetscCall((*linesearch->ops->viproject)(snes, W));
        PetscCall((*linesearch->ops->snesfunc)(snes, W, F));
        if (linesearch->ops->vinorm) {
          fnrm = gnorm;
          PetscCall((*linesearch->ops->vinorm)(snes, F, W, &fnrm));
        } else {
          PetscCall(VecNorm(F, NORM_2, &fnrm));
        }
        fnrm_mid = fnrm_mid * fnrm_mid;
        fnrm     = fnrm * fnrm;
      } else {
        /* compute the objective at the midpoint */
        PetscCall(SNESComputeObjective(snes, W, &fnrm_mid));

        /* compute the objective at the new endpoint */
        PetscCall(VecWAXPY(W, -lambda, Y, X));
        PetscCall(SNESComputeObjective(snes, W, &fnrm));
      }

      /* if new endpoint is viable, exit */
      if (!PetscIsInfOrNanReal(fnrm)) break;
      if (monitor) {
        PetscCall(PetscViewerASCIIAddTab(monitor, ((PetscObject)linesearch)->tablevel));
        PetscCall(PetscViewerASCIIPrintf(monitor, "    Line search: objective function at lambda = %g is Inf or Nan, cutting lambda\n", (double)lambda));
        PetscCall(PetscViewerASCIISubtractTab(monitor, ((PetscObject)linesearch)->tablevel));
      }

      /* if smallest allowable lambda gives NaN or Inf, exit line search */
      if (lambda <= minlambda) {
        if (monitor) {
          PetscCall(PetscViewerASCIIAddTab(monitor, ((PetscObject)linesearch)->tablevel));
          PetscCall(PetscViewerASCIIPrintf(monitor, "    Line search: objective function at lambda = %g <= lambda_min = %g is Inf or Nan, can not further cut lambda\n", (double)lambda, (double)lambda));
          PetscCall(PetscViewerASCIISubtractTab(monitor, ((PetscObject)linesearch)->tablevel));
        }
        PetscCall(SNESLineSearchSetReason(linesearch, SNES_LINESEARCH_FAILED_REDUCT));
        PetscFunctionReturn(PETSC_SUCCESS);
      }

      /* forbid the search from ever going back to the "failed" length that generates Nan or Inf */
      maxlambda = .95 * lambda;

      /* shrink lambda towards the previous one which was viable */
      lambda     = .5 * (lambda + lambda_old);
      lambda_mid = .5 * (lambda + lambda_old);
    }

    /* monitoring output */
    if (monitor) {
      PetscCall(PetscViewerASCIIAddTab(monitor, ((PetscObject)linesearch)->tablevel));
      if (!objective) {
        PetscCall(PetscViewerASCIIPrintf(monitor, "    Line search: lambdas = [%g, %g, %g], fnorms = [%g, %g, %g]\n", (double)lambda, (double)lambda_mid, (double)lambda_old, (double)PetscSqrtReal(fnrm), (double)PetscSqrtReal(fnrm_mid), (double)PetscSqrtReal(fnrm_old)));
      } else {
        PetscCall(PetscViewerASCIIPrintf(monitor, "    Line search: lambdas = [%g, %g, %g], obj = [%g, %g, %g]\n", (double)lambda, (double)lambda_mid, (double)lambda_old, (double)fnrm, (double)fnrm_mid, (double)fnrm_old));
      }
      PetscCall(PetscViewerASCIISubtractTab(monitor, ((PetscObject)linesearch)->tablevel));
    }

    /* determine change of lambda */
    delLambda = lambda - lambda_old;

    /* check change of lambda tolerance */
    if (PetscAbsReal(delLambda) < ltol) {
      if (monitor) {
        PetscCall(PetscViewerASCIIAddTab(monitor, ((PetscObject)linesearch)->tablevel));
        PetscCall(PetscViewerASCIIPrintf(monitor, "    Line search: abs(delLambda) = %g < ltol = %g\n", (double)PetscAbsReal(delLambda), (double)ltol));
        PetscCall(PetscViewerASCIISubtractTab(monitor, ((PetscObject)linesearch)->tablevel));
      }
      break;
    }

    /* compute f'() at the end points using second order one sided differencing */
    delFnrm     = (3. * fnrm - 4. * fnrm_mid + 1. * fnrm_old) / delLambda;
    delFnrm_old = (-3. * fnrm_old + 4. * fnrm_mid - 1. * fnrm) / delLambda;

    /* compute f''() at the midpoint using centered differencing */
    del2Fnrm = (delFnrm - delFnrm_old) / delLambda;

    /* check absolute tolerance */
    if (PetscAbsReal(delFnrm) <= atol) {
      if (monitor) {
        PetscCall(PetscViewerASCIIAddTab(monitor, ((PetscObject)linesearch)->tablevel));
        PetscCall(PetscViewerASCIIPrintf(monitor, "    Line search: abs(delFnrm) = %g <= atol = %g\n", (double)PetscAbsReal(delFnrm), (double)atol));
        PetscCall(PetscViewerASCIISubtractTab(monitor, ((PetscObject)linesearch)->tablevel));
      }
      break;
    }

    /* compute the secant (Newton) update -- always go downhill */
    if (del2Fnrm > 0.) lambda_update = lambda - delFnrm / del2Fnrm;
    else if (del2Fnrm < 0.) lambda_update = lambda + delFnrm / del2Fnrm;
    else {
      if (monitor) {
        PetscCall(PetscViewerASCIIAddTab(monitor, ((PetscObject)linesearch)->tablevel));
        PetscCall(PetscViewerASCIIPrintf(monitor, "    Line search: del2Fnrm = 0\n"));
        PetscCall(PetscViewerASCIISubtractTab(monitor, ((PetscObject)linesearch)->tablevel));
      }
      break;
    }

    /* prevent secant method from stepping out of bounds */
    if (lambda_update < minlambda) lambda_update = 0.5 * (lambda + lambda_old);
    if (lambda_update > maxlambda) {
      lambda_update = maxlambda;
      break;
    }

    /* if lambda is NaN or Inf, do not accept update but exit with previous lambda */
    if (PetscIsInfOrNanReal(lambda_update)) {
      if (monitor) {
        PetscCall(PetscViewerASCIIAddTab(monitor, ((PetscObject)linesearch)->tablevel));
        PetscCall(PetscViewerASCIIPrintf(monitor, "    Line search: lambda_update is NaN or Inf\n"));
        PetscCall(PetscViewerASCIISubtractTab(monitor, ((PetscObject)linesearch)->tablevel));
      }
      PetscCall(SNESLineSearchSetReason(linesearch, SNES_LINESEARCH_FAILED_NANORINF));
      break;
    }

    /* update the endpoints and the midpoint of the bracketed secant region */
    lambda_old = lambda;
    lambda     = lambda_update;
    fnrm_old   = fnrm;
    lambda_mid = 0.5 * (lambda + lambda_old);

    if ((i == max_it - 1) && monitor) {
      PetscCall(PetscViewerASCIIAddTab(monitor, ((PetscObject)linesearch)->tablevel));
      PetscCall(PetscViewerASCIIPrintf(monitor, "    Line search: reached maximum number of iterations!\n"));
      PetscCall(PetscViewerASCIISubtractTab(monitor, ((PetscObject)linesearch)->tablevel));
    }
  }

  /* construct the solution */
  PetscCall(VecWAXPY(W, -lambda, Y, X));
  if (linesearch->ops->viproject) PetscCall((*linesearch->ops->viproject)(snes, W));

  /* postcheck */
  PetscCall(SNESLineSearchSetLambda(linesearch, lambda));
  PetscCall(SNESLineSearchPostCheck(linesearch, X, Y, W, &changed_y, &changed_w));
  if (changed_y) {
    if (!changed_w) PetscCall(VecWAXPY(W, -lambda, Y, X));
    if (linesearch->ops->viproject) PetscCall((*linesearch->ops->viproject)(snes, W));
  }
  PetscCall(VecCopy(W, X));
  PetscCall((*linesearch->ops->snesfunc)(snes, X, F));

  PetscCall(SNESLineSearchComputeNorms(linesearch));

  if (monitor) {
    PetscCall(SNESLineSearchGetNorms(linesearch, NULL, &gnorm, NULL));
    PetscCall(PetscViewerASCIIAddTab(monitor, ((PetscObject)linesearch)->tablevel));
    PetscCall(PetscViewerASCIIPrintf(monitor, "    Line search terminated: lambda = %g, fnorm = %g\n", (double)lambda, (double)gnorm));
    PetscCall(PetscViewerASCIISubtractTab(monitor, ((PetscObject)linesearch)->tablevel));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   SNESLINESEARCHSECANT - Secant search in the L2 norm of the function or the objective function.

   Attempts to solve $ \min_{\lambda} f(x_k + \lambda Y_k) $ using the secant method with the initial bracketing of $ \lambda $ between [0,damping].
   $f(x_k + \lambda Y_k)$ is either the squared L2-norm of the function $||F(x_k + \lambda Y_k)||_2^2$,
   or the objective function $G(x_k + \lambda Y_k)$ if it is provided with `SNESSetObjective()`
   Differences of $f()$ are used to approximate the first and second derivative of $f()$ with respect to
   $\lambda$, $f'()$ and $f''()$.

   When an objective function is provided $f(w)$ is the objective function otherwise $f(w) = ||F(w)||^2$.
   $x$ is the current step and $y$ is the search direction.

   Options Database Keys:
+  -snes_linesearch_max_it <1>         - maximum number of iterations within the line search
.  -snes_linesearch_damping <1.0>      - initial `lambda` on entry to the line search
.  -snes_linesearch_minlambda <1e\-12> - minimum allowable `lambda`
.  -snes_linesearch_maxlambda <1.0>    - maximum `lambda` (scaling of solution update) allowed
.  -snes_linesearch_atol <1e\-15>      - absolute tolerance for the secant method $ f'() < atol $
-  -snes_linesearch_ltol <1e\-8>       - minimum absolute change in `lambda` allowed

   Level: advanced

.seealso: [](ch_snes), `SNESLINESEARCHBT`, `SNESLINESEARCHCP`, `SNESLineSearch`, `SNESLineSearchType`, `SNESLineSearchCreate()`, `SNESLineSearchSetType()`
M*/
PETSC_EXTERN PetscErrorCode SNESLineSearchCreate_Secant(SNESLineSearch linesearch)
{
  PetscFunctionBegin;
  linesearch->ops->apply          = SNESLineSearchApply_Secant;
  linesearch->ops->destroy        = NULL;
  linesearch->ops->setfromoptions = NULL;
  linesearch->ops->reset          = NULL;
  linesearch->ops->view           = NULL;
  linesearch->ops->setup          = NULL;

  linesearch->max_it = 1;
  PetscFunctionReturn(PETSC_SUCCESS);
}
