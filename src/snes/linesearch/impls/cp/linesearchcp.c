#include <petsc/private/linesearchimpl.h>
#include <petscsnes.h>

static PetscErrorCode SNESLineSearchApply_CP(SNESLineSearch linesearch)
{
  PetscBool   changed_y, changed_w;
  Vec         X, Y, F, W;
  SNES        snes;
  PetscReal   xnorm, ynorm, gnorm, minlambda, maxlambda, rtol, atol, ltol;
  PetscReal   lambda, lambda_old, lambda_update, delLambda;
  PetscScalar fty, fty_init, fty_old, fty_mid1, fty_mid2, s;
  PetscInt    i, max_it;
  PetscViewer monitor;

  PetscFunctionBegin;
  PetscCall(SNESLineSearchGetVecs(linesearch, &X, &F, &Y, &W, NULL));
  PetscCall(SNESLineSearchGetNorms(linesearch, &xnorm, &gnorm, &ynorm));
  PetscCall(SNESLineSearchGetSNES(linesearch, &snes));
  PetscCall(SNESLineSearchGetLambda(linesearch, &lambda));
  PetscCall(SNESLineSearchGetTolerances(linesearch, &minlambda, &maxlambda, &rtol, &atol, &ltol, &max_it));
  PetscCall(SNESLineSearchSetReason(linesearch, SNES_LINESEARCH_SUCCEEDED));
  PetscCall(SNESLineSearchGetDefaultMonitor(linesearch, &monitor));

  /* precheck */
  PetscCall(SNESLineSearchPreCheck(linesearch, X, Y, &changed_y));
  lambda_old = 0.0;

  /* evaluate initial point */
  if (linesearch->ops->vidirderiv) {
    PetscCall((*linesearch->ops->vidirderiv)(snes, F, X, Y, &fty_old));
  } else {
    PetscCall(VecDot(F, Y, &fty_old));
  }
  if (PetscAbsScalar(fty_old) < atol * ynorm) {
    if (monitor) {
      PetscCall(PetscViewerASCIIAddTab(monitor, ((PetscObject)linesearch)->tablevel));
      PetscCall(PetscViewerASCIIPrintf(monitor, "    Line search terminated at initial point because dot(F,Y) = %g < atol*||y|| = %g\n", (double)PetscAbsScalar(fty_old), (double)(atol * ynorm)));
      PetscCall(PetscViewerASCIISubtractTab(monitor, ((PetscObject)linesearch)->tablevel));
    }
    PetscCall(SNESSetConvergedReason(linesearch->snes, SNES_CONVERGED_FNORM_ABS));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  fty_init = fty_old;

  for (i = 0; i < max_it; i++) {
    /* compute the norm at lambda */
    PetscCall(VecWAXPY(W, -lambda, Y, X));
    if (linesearch->ops->viproject) PetscCall((*linesearch->ops->viproject)(snes, W));
    PetscCall((*linesearch->ops->snesfunc)(snes, W, F));
    if (linesearch->ops->vidirderiv) {
      PetscCall((*linesearch->ops->vidirderiv)(snes, F, W, Y, &fty));
    } else {
      PetscCall(VecDot(F, Y, &fty));
    }

    /* compute change of lambda */
    delLambda = lambda - lambda_old;

    /* check change of lambda tolerance */
    if (PetscAbsReal(delLambda) < ltol) {
      if (monitor) {
        PetscCall(PetscViewerASCIIAddTab(monitor, ((PetscObject)linesearch)->tablevel));
        PetscCall(PetscViewerASCIIPrintf(monitor, "    Line search: abs(dlambda) = %g < ltol = %g\n", (double)PetscAbsReal(delLambda), (double)ltol));
        PetscCall(PetscViewerASCIISubtractTab(monitor, ((PetscObject)linesearch)->tablevel));
      }
      break;
    }

    /* check relative tolerance */
    if (PetscAbsScalar(fty) / PetscAbsScalar(fty_init) < rtol) {
      if (monitor) {
        PetscCall(PetscViewerASCIIAddTab(monitor, ((PetscObject)linesearch)->tablevel));
        PetscCall(PetscViewerASCIIPrintf(monitor, "    Line search: abs(fty/fty_init) = %g <= rtol  = %g\n", (double)(PetscAbsScalar(fty) / PetscAbsScalar(fty_init)), (double)rtol));
        PetscCall(PetscViewerASCIISubtractTab(monitor, ((PetscObject)linesearch)->tablevel));
      }
      break;
    }

    /* check absolute tolerance */
    if (PetscAbsScalar(fty) < atol * ynorm && i > 0) {
      if (monitor) {
        PetscCall(PetscViewerASCIIAddTab(monitor, ((PetscObject)linesearch)->tablevel));
        PetscCall(PetscViewerASCIIPrintf(monitor, "    Line search: abs(fty)/||y|| = %g <= atol = %g\n", (double)(PetscAbsScalar(fty) / ynorm), (double)atol));
        PetscCall(PetscViewerASCIISubtractTab(monitor, ((PetscObject)linesearch)->tablevel));
      }
      break;
    }

    /* print iteration information */
    if (monitor) {
      PetscCall(PetscViewerASCIIAddTab(monitor, ((PetscObject)linesearch)->tablevel));
      PetscCall(PetscViewerASCIIPrintf(monitor, "    Line search: lambdas = [%g, %g], ftys = [%g, %g]\n", (double)lambda, (double)lambda_old, (double)PetscRealPart(fty), (double)PetscRealPart(fty_old)));
      PetscCall(PetscViewerASCIISubtractTab(monitor, ((PetscObject)linesearch)->tablevel));
    }

    /* approximate the second derivative */
    if (linesearch->order == SNES_LINESEARCH_ORDER_LINEAR) {
      /* first order finite difference approximation */
      s = (fty - fty_old) / delLambda;
    } else if (linesearch->order == SNES_LINESEARCH_ORDER_QUADRATIC) {
      /* evaluate function at midpoint 0.5 * (lambda + lambda_old) */
      PetscCall(VecWAXPY(W, -0.5 * (lambda + lambda_old), Y, X));
      if (linesearch->ops->viproject) PetscCall((*linesearch->ops->viproject)(snes, W));
      PetscCall((*linesearch->ops->snesfunc)(snes, W, F));
      if (linesearch->ops->vidirderiv) {
        PetscCall((*linesearch->ops->vidirderiv)(snes, F, W, Y, &fty_mid1));
      } else {
        PetscCall(VecDot(F, Y, &fty_mid1));
      }

      /* second order finite difference approximation */
      s = (3. * fty - 4. * fty_mid1 + fty_old) / delLambda;

    } else {
      /* evaluate function at midpoint 0.5 * (lambda + lambda_old) */
      PetscCall(VecWAXPY(W, -0.5 * (lambda + lambda_old), Y, X));
      if (linesearch->ops->viproject) PetscCall((*linesearch->ops->viproject)(snes, W));
      PetscCall((*linesearch->ops->snesfunc)(snes, W, F));
      if (linesearch->ops->vidirderiv) {
        PetscCall((*linesearch->ops->vidirderiv)(snes, F, W, Y, &fty_mid1));
      } else {
        PetscCall(VecDot(F, Y, &fty_mid1));
      }

      /* evaluate function at lambda + 0.5 * (lambda - lambda_old) */
      PetscCall(VecWAXPY(W, -(lambda + 0.5 * (lambda - lambda_old)), Y, X));
      if (linesearch->ops->viproject) PetscCall((*linesearch->ops->viproject)(snes, W));
      PetscCall((*linesearch->ops->snesfunc)(snes, W, F));
      if (linesearch->ops->vidirderiv) {
        PetscCall((*linesearch->ops->vidirderiv)(snes, F, W, Y, &fty_mid2));
      } else {
        PetscCall(VecDot(F, Y, &fty_mid2));
      }

      /* third order finite difference approximation */
      s = (2. * fty_mid2 + 3. * fty - 6. * fty_mid1 + fty_old) / (3. * delLambda);
    }

    /* compute secant update (modifying the search direction if necessary) */
    if (PetscRealPart(s) > 0.) s = -s;
    if (s == 0.0) {
      if (monitor) {
        PetscCall(PetscViewerASCIIAddTab(monitor, ((PetscObject)linesearch)->tablevel));
        PetscCall(PetscViewerASCIIPrintf(monitor, "    Line search: del2Fnrm = 0\n"));
        PetscCall(PetscViewerASCIISubtractTab(monitor, ((PetscObject)linesearch)->tablevel));
      }
      break;
    }
    lambda_update = lambda - PetscRealPart(fty / s);

    /* if step is too small, go the opposite direction */
    if (lambda_update < minlambda) lambda_update = lambda + PetscRealPart(fty / s);
    /* if secant method would step out of bounds, exit with the respective bound */
    if (lambda_update > maxlambda) {
      lambda = maxlambda;
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

    /* compute the new state of the line search */
    lambda_old = lambda;
    lambda     = lambda_update;
    fty_old    = fty;
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
  PetscCall(SNESLineSearchGetNorms(linesearch, &xnorm, &gnorm, &ynorm));

  if (monitor) {
    PetscCall(PetscViewerASCIIAddTab(monitor, ((PetscObject)linesearch)->tablevel));
    PetscCall(PetscViewerASCIIPrintf(monitor, "    Line search terminated: lambda = %g, fnorms = %g\n", (double)lambda, (double)gnorm));
    PetscCall(PetscViewerASCIISubtractTab(monitor, ((PetscObject)linesearch)->tablevel));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   SNESLINESEARCHCP - Critical point line search. This line search assumes that there exists some
   artificial $G(x)$ for which the `SNESFunctionFn` $F(x) = grad G(x)$. Therefore, this line search seeks
   to find roots of the directional derivative via a secant method, that is $F(x_k - \lambda Y_k) \cdot Y_k / ||Y_k|| = 0$.

   Options Database Keys:
+  -snes_linesearch_minlambda <1e\-12> - the minimum acceptable `lambda` (scaling of solution update)
.  -snes_linesearch_maxlambda <1.0>    - the algorithm ensures that `lambda` is never larger than this value
.  -snes_linesearch_damping <1.0>      - initial `lambda` on entry to the line search
.  -snes_linesearch_order <1>          - order of the approximation in the secant method, must be 1, 2, or 3
.  -snes_linesearch_max_it <1>         - the maximum number of secant iterations performed
.  -snes_linesearch_rtol <1e\-8>       - relative tolerance for the directional derivative
.  -snes_linesearch_atol <1e\-15>      - absolute tolerance for the directional derivative
-  -snes_linesearch_ltol <1e\-8>       - minimum absolute change in `lambda` allowed

   Level: advanced

   Notes:
   This method does NOT use the objective function if it is provided with `SNESSetObjective()`.

   This method is the preferred line search for `SNESQN` and `SNESNCG`.

.seealso: [](ch_snes), `SNESLineSearch`, `SNESLineSearchType`, `SNESLineSearchCreate()`, `SNESLineSearchSetType()`, `SNESLINESEARCHBISECTION`
M*/
PETSC_EXTERN PetscErrorCode SNESLineSearchCreate_CP(SNESLineSearch linesearch)
{
  PetscFunctionBegin;
  linesearch->ops->apply          = SNESLineSearchApply_CP;
  linesearch->ops->destroy        = NULL;
  linesearch->ops->setfromoptions = NULL;
  linesearch->ops->reset          = NULL;
  linesearch->ops->view           = NULL;
  linesearch->ops->setup          = NULL;
  linesearch->order               = SNES_LINESEARCH_ORDER_LINEAR;

  linesearch->max_it = 1;
  PetscFunctionReturn(PETSC_SUCCESS);
}
