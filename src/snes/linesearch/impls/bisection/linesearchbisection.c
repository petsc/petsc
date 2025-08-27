#include <petsc/private/linesearchimpl.h>
#include <petsc/private/snesimpl.h>

static PetscErrorCode SNESLineSearchApply_Bisection(SNESLineSearch linesearch)
{
  PetscBool   changed_y, changed_w;
  Vec         X, F, Y, W, G;
  SNES        snes;
  PetscReal   ynorm;
  PetscReal   lambda_left, lambda, lambda_right, lambda_old;
  PetscScalar fty_left, fty, fty_initial;
  PetscViewer monitor;
  PetscReal   rtol, atol, ltol;
  PetscInt    it, max_it;

  PetscFunctionBegin;
  PetscCall(SNESLineSearchGetVecs(linesearch, &X, &F, &Y, &W, &G));
  PetscCall(SNESLineSearchGetLambda(linesearch, &lambda));
  PetscCall(SNESLineSearchGetSNES(linesearch, &snes));
  PetscCall(SNESLineSearchGetTolerances(linesearch, NULL, NULL, &rtol, &atol, &ltol, &max_it));
  PetscCall(SNESLineSearchGetDefaultMonitor(linesearch, &monitor));

  /* pre-check */
  PetscCall(SNESLineSearchPreCheck(linesearch, X, Y, &changed_y));

  /* compute ynorm to normalize search direction */
  PetscCall(VecNorm(Y, NORM_2, &ynorm));

  /* initialize interval for bisection */
  lambda_left  = 0.0;
  lambda_right = lambda;

  /* compute fty at left end of interval */
  if (linesearch->ops->vidirderiv) {
    PetscCall((*linesearch->ops->vidirderiv)(snes, F, X, Y, &fty_left));
  } else {
    PetscCall(VecDot(F, Y, &fty_left));
  }
  fty_initial = fty_left;

  /* compute fty at right end of interval (initial lambda) */
  PetscCall(VecWAXPY(W, -lambda, Y, X));
  if (linesearch->ops->viproject) PetscCall((*linesearch->ops->viproject)(snes, W));
  PetscCall((*linesearch->ops->snesfunc)(snes, W, G));
  if (snes->nfuncs >= snes->max_funcs && snes->max_funcs >= 0) {
    PetscCall(PetscInfo(snes, "Exceeded maximum function evaluations during line search!\n"));
    snes->reason = SNES_DIVERGED_FUNCTION_COUNT;
    PetscCall(SNESLineSearchSetReason(linesearch, SNES_LINESEARCH_FAILED_FUNCTION));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if (linesearch->ops->vidirderiv) {
    PetscCall((*linesearch->ops->vidirderiv)(snes, G, W, Y, &fty));
  } else {
    PetscCall(VecDot(G, Y, &fty));
  }
  /* check whether sign changes in interval */
  if (!PetscIsInfOrNanScalar(fty) && (PetscRealPart(fty_left * fty) > 0.0)) {
    /* no change of sign: accept full step */
    if (monitor) {
      PetscCall(PetscViewerASCIIAddTab(monitor, ((PetscObject)linesearch)->tablevel));
      PetscCall(PetscViewerASCIIPrintf(monitor, "      Line search: sign of fty does not change in step interval, accepting full step\n"));
      PetscCall(PetscViewerASCIISubtractTab(monitor, ((PetscObject)linesearch)->tablevel));
    }
  } else {
    /* change of sign: iteratively bisect interval */
    lambda_old = 0.0;
    it         = 0;

    while (PETSC_TRUE) {
      /* check for NaN or Inf */
      if (PetscIsInfOrNanScalar(fty)) {
        if (monitor) {
          PetscCall(PetscViewerASCIIAddTab(monitor, ((PetscObject)linesearch)->tablevel));
          PetscCall(PetscViewerASCIIPrintf(monitor, "      Line search fty is NaN or Inf!\n"));
          PetscCall(PetscViewerASCIISubtractTab(monitor, ((PetscObject)linesearch)->tablevel));
        }
        PetscCall(SNESLineSearchSetReason(linesearch, SNES_LINESEARCH_FAILED_NANORINF));
        PetscFunctionReturn(PETSC_SUCCESS);
        break;
      }

      /* check absolute tolerance */
      if (PetscAbsScalar(fty) <= atol * ynorm) {
        if (monitor) {
          PetscCall(PetscViewerASCIIAddTab(monitor, ((PetscObject)linesearch)->tablevel));
          PetscCall(PetscViewerASCIIPrintf(monitor, "      Line search: abs(fty)/||y|| = %g <= atol = %g\n", (double)(PetscAbsScalar(fty) / ynorm), (double)atol));
          PetscCall(PetscViewerASCIISubtractTab(monitor, ((PetscObject)linesearch)->tablevel));
        }
        break;
      }

      /* check relative tolerance */
      if (PetscAbsScalar(fty) / PetscAbsScalar(fty_initial) <= rtol) {
        if (monitor) {
          PetscCall(PetscViewerASCIIAddTab(monitor, ((PetscObject)linesearch)->tablevel));
          PetscCall(PetscViewerASCIIPrintf(monitor, "      Line search: abs(fty/fty_initial) = %g <= rtol  = %g\n", (double)(PetscAbsScalar(fty) / PetscAbsScalar(fty_initial)), (double)rtol));
          PetscCall(PetscViewerASCIISubtractTab(monitor, ((PetscObject)linesearch)->tablevel));
        }
        break;
      }

      /* check maximum number of iterations */
      if (it > max_it) {
        if (monitor) {
          PetscCall(PetscViewerASCIIAddTab(monitor, ((PetscObject)linesearch)->tablevel));
          PetscCall(PetscViewerASCIIPrintf(monitor, "      Line search: maximum iterations reached\n"));
          PetscCall(PetscViewerASCIISubtractTab(monitor, ((PetscObject)linesearch)->tablevel));
        }
        PetscCall(SNESLineSearchSetReason(linesearch, SNES_LINESEARCH_FAILED_REDUCT));
        PetscFunctionReturn(PETSC_SUCCESS);
        break;
      }

      /* check change of lambda tolerance */
      if (PetscAbsReal(lambda - lambda_old) < ltol) {
        if (monitor) {
          PetscCall(PetscViewerASCIIAddTab(monitor, ((PetscObject)linesearch)->tablevel));
          PetscCall(PetscViewerASCIIPrintf(monitor, "      Line search: abs(dlambda) = %g < ltol = %g\n", (double)PetscAbsReal(lambda - lambda_old), (double)ltol));
          PetscCall(PetscViewerASCIISubtractTab(monitor, ((PetscObject)linesearch)->tablevel));
        }
        break;
      }

      /* determine direction of bisection (not necessary for 0th iteration) */
      if (it > 0) {
        if (PetscRealPart(fty * fty_left) <= 0.0) {
          lambda_right = lambda;
        } else {
          lambda_left = lambda;
          /* also update fty_left for direction check in next iteration */
          fty_left = fty;
        }
      }

      /* bisect interval */
      lambda_old = lambda;
      lambda     = 0.5 * (lambda_left + lambda_right);

      /* compute fty at new lambda */
      PetscCall(VecWAXPY(W, -lambda, Y, X));
      if (linesearch->ops->viproject) PetscCall((*linesearch->ops->viproject)(snes, W));
      PetscCall((*linesearch->ops->snesfunc)(snes, W, G));
      if (snes->nfuncs >= snes->max_funcs && snes->max_funcs >= 0) {
        PetscCall(PetscInfo(snes, "Exceeded maximum function evaluations during line search!\n"));
        snes->reason = SNES_DIVERGED_FUNCTION_COUNT;
        PetscCall(SNESLineSearchSetReason(linesearch, SNES_LINESEARCH_FAILED_FUNCTION));
        PetscFunctionReturn(PETSC_SUCCESS);
      }
      if (linesearch->ops->vidirderiv) {
        PetscCall((*linesearch->ops->vidirderiv)(snes, G, W, Y, &fty));
      } else {
        PetscCall(VecDot(G, Y, &fty));
      }

      /* print iteration information */
      if (monitor) {
        PetscCall(PetscViewerASCIIAddTab(monitor, ((PetscObject)linesearch)->tablevel));
        PetscCall(PetscViewerASCIIPrintf(monitor, "      %3" PetscInt_FMT " Line search: fty/||y|| = %g, lambda = %g\n", it, (double)(PetscRealPart(fty) / ynorm), (double)lambda));
        PetscCall(PetscViewerASCIISubtractTab(monitor, ((PetscObject)linesearch)->tablevel));
      }

      /* count up iteration */
      it++;
    }
  }

  /* post-check */
  PetscCall(SNESLineSearchSetLambda(linesearch, lambda));
  PetscCall(SNESLineSearchPostCheck(linesearch, X, Y, W, &changed_y, &changed_w));
  if (changed_y) {
    if (!changed_w) PetscCall(VecWAXPY(W, -lambda, Y, X));
    if (linesearch->ops->viproject) PetscCall((*linesearch->ops->viproject)(snes, W));
  }

  /* update solution*/
  PetscCall(VecCopy(W, X));
  PetscCall((*linesearch->ops->snesfunc)(snes, X, F));
  PetscCall(SNESLineSearchComputeNorms(linesearch));

  /* finalization */
  PetscCall(SNESLineSearchSetReason(linesearch, SNES_LINESEARCH_SUCCEEDED));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   SNESLINESEARCHBISECTION - Bisection line search.
   Similar to the critical point line search, `SNESLINESEARCHCP`, the bisection line search assumes that there exists some $G(x)$ for which the `SNESFunctionFn` $F(x) = grad G(x)$.
   This line search seeks to find the root of the directional derivative, that is $F(x_k - \lambda Y_k) \cdot Y_k / ||Y_k|| = 0$, along the search direction $Y_k$ through bisection.

   Options Database Keys:
+  -snes_linesearch_max_it <50>   - maximum number of bisection iterations for the line search
.  -snes_linesearch_damping <1.0> - initial `lambda` on entry to the line search
.  -snes_linesearch_rtol <1e\-8>  - relative tolerance for the directional derivative
.  -snes_linesearch_atol <1e\-6>  - absolute tolerance for the directional derivative
-  -snes_linesearch_ltol <1e\-6>  - minimum absolute change in `lambda` allowed

   Level: intermediate

   Notes:
   `lambda` is the scaling of the search direction (vector) that is computed by this algorithm.
   If there is no change of sign in the directional derivative from $\lambda=0$ to the initial `lambda` (the damping), then the initial `lambda` will be used.
   Hence, this line search will always give a `lambda` in the interval $[0, damping]$.
   This method does NOT use the objective function if it is provided with `SNESSetObjective()`.

.seealso: [](ch_snes), `SNESLineSearch`, `SNESLineSearchType`, `SNESLineSearchCreate()`, `SNESLineSearchSetType()`, `SNESLINESEARCHCP`
M*/
PETSC_EXTERN PetscErrorCode SNESLineSearchCreate_Bisection(SNESLineSearch linesearch)
{
  PetscFunctionBegin;
  linesearch->ops->apply          = SNESLineSearchApply_Bisection;
  linesearch->ops->destroy        = NULL;
  linesearch->ops->setfromoptions = NULL;
  linesearch->ops->reset          = NULL;
  linesearch->ops->view           = NULL;
  linesearch->ops->setup          = NULL;

  /* set default option values */
  linesearch->max_it = 50;
  linesearch->rtol   = 1e-8;
  linesearch->atol   = 1e-6;
  linesearch->ltol   = 1e-6;
  PetscFunctionReturn(PETSC_SUCCESS);
}
