#include <petsc/private/linesearchimpl.h> /*I  "petscsnes.h"  I*/
#include <petsc/private/snesimpl.h>

typedef struct {
  PetscReal alpha; /* sufficient decrease parameter */
} SNESLineSearch_BT;

/*@
  SNESLineSearchBTSetAlpha - Sets the descent parameter, `alpha`, in the `SNESLINESEARCHBT` `SNESLineSearch` variant.

  Input Parameters:
+ linesearch - linesearch context
- alpha      - The descent parameter

  Level: intermediate

.seealso: [](ch_snes), `SNESLineSearch`, `SNESLineSearchSetLambda()`, `SNESLineSearchGetTolerances()`, `SNESLINESEARCHBT`, `SNESLineSearchBTGetAlpha()`
@*/
PetscErrorCode SNESLineSearchBTSetAlpha(SNESLineSearch linesearch, PetscReal alpha)
{
  SNESLineSearch_BT *bt = (SNESLineSearch_BT *)linesearch->data;
  PetscBool          isbt;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch, SNESLINESEARCH_CLASSID, 1);
  PetscCall(PetscObjectTypeCompare((PetscObject)linesearch, SNESLINESEARCHBT, &isbt));
  if (isbt) bt->alpha = alpha;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  SNESLineSearchBTGetAlpha - Gets the descent parameter, `alpha`, in the `SNESLINESEARCHBT` variant that was set with `SNESLineSearchBTSetAlpha()`

  Input Parameter:
. linesearch - linesearch context

  Output Parameter:
. alpha - The descent parameter

  Level: intermediate

.seealso: [](ch_snes), `SNESLineSearch`, `SNESLineSearchGetLambda()`, `SNESLineSearchGetTolerances()`, `SNESLINESEARCHBT`, `SNESLineSearchBTSetAlpha()`
@*/
PetscErrorCode SNESLineSearchBTGetAlpha(SNESLineSearch linesearch, PetscReal *alpha)
{
  SNESLineSearch_BT *bt = (SNESLineSearch_BT *)linesearch->data;
  PetscBool          isbt;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch, SNESLINESEARCH_CLASSID, 1);
  PetscCall(PetscObjectTypeCompare((PetscObject)linesearch, SNESLINESEARCHBT, &isbt));
  PetscCheck(isbt, PetscObjectComm((PetscObject)linesearch), PETSC_ERR_USER, "Not for type %s", ((PetscObject)linesearch)->type_name);
  *alpha = bt->alpha;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESLineSearchApply_BT(SNESLineSearch linesearch)
{
  SNESLineSearch_BT *bt = (SNESLineSearch_BT *)linesearch->data;
  PetscBool          changed_y, changed_w;
  Vec                X, F, Y, W, G;
  SNES               snes;
  PetscReal          fnorm, xnorm, ynorm, gnorm;
  PetscReal          lambda, lambdatemp, lambdaprev, minlambda, initslope, alpha, stol;
  PetscReal          t1, t2, a, b, d;
  PetscReal          f;
  PetscReal          g, gprev;
  PetscViewer        monitor;
  PetscInt           max_it, count;
  Mat                jac;
  SNESObjectiveFn   *objective;
  const char *const  ordStr[] = {"Linear", "Quadratic", "Cubic"};

  PetscFunctionBegin;
  PetscCall(SNESLineSearchGetVecs(linesearch, &X, &F, &Y, &W, &G));
  PetscCall(SNESLineSearchGetNorms(linesearch, NULL, &fnorm, NULL));
  PetscCall(SNESLineSearchGetLambda(linesearch, &lambda));
  PetscCall(SNESLineSearchGetSNES(linesearch, &snes));
  PetscCall(SNESLineSearchGetDefaultMonitor(linesearch, &monitor));
  PetscCall(SNESLineSearchGetTolerances(linesearch, &minlambda, NULL, NULL, NULL, NULL, &max_it));
  PetscCall(SNESGetTolerances(snes, NULL, NULL, &stol, NULL, NULL));
  PetscCall(SNESGetObjective(snes, &objective, NULL));
  alpha = bt->alpha;

  PetscCall(SNESGetJacobian(snes, &jac, NULL, NULL, NULL));
  PetscCheck(jac || objective, PetscObjectComm((PetscObject)linesearch), PETSC_ERR_USER, "SNESLineSearchBT requires a Jacobian matrix");

  PetscCall(SNESLineSearchPreCheck(linesearch, X, Y, &changed_y));
  PetscCall(SNESLineSearchSetReason(linesearch, SNES_LINESEARCH_SUCCEEDED));

  PetscCall(VecNormBegin(Y, NORM_2, &ynorm));
  PetscCall(VecNormBegin(X, NORM_2, &xnorm));
  PetscCall(VecNormEnd(Y, NORM_2, &ynorm));
  PetscCall(VecNormEnd(X, NORM_2, &xnorm));

  if (ynorm == 0.0) {
    if (monitor) {
      PetscCall(PetscViewerASCIIAddTab(monitor, ((PetscObject)linesearch)->tablevel));
      PetscCall(PetscViewerASCIIPrintf(monitor, "    Line search: Initial direction and size is 0\n"));
      PetscCall(PetscViewerASCIISubtractTab(monitor, ((PetscObject)linesearch)->tablevel));
    }
    PetscCall(VecCopy(X, W));
    PetscCall(VecCopy(F, G));
    PetscCall(SNESLineSearchSetNorms(linesearch, xnorm, fnorm, ynorm));
    PetscCall(SNESLineSearchSetReason(linesearch, SNES_LINESEARCH_FAILED_REDUCT));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* if the SNES has an objective set, use that instead of the function value */
  if (objective) {
    PetscCall(SNESComputeObjective(snes, X, &f));
  } else {
    f = 0.5 * PetscSqr(fnorm);
  }

  /* compute the initial slope */
  if (objective) {
    /* slope comes from the function (assumed to be the gradient of the objective) */
    PetscCall(VecDotRealPart(Y, F, &initslope));
  } else {
    /* slope comes from the normal equations */
    PetscCall(MatMult(jac, Y, W));
    PetscCall(VecDotRealPart(F, W, &initslope));
    if (initslope > 0.0) initslope = -initslope;
    if (initslope == 0.0) initslope = -1.0;
  }

  while (PETSC_TRUE) {
    PetscCall(VecWAXPY(W, -lambda, Y, X));
    if (linesearch->ops->viproject) PetscCall((*linesearch->ops->viproject)(snes, W));
    if (snes->nfuncs >= snes->max_funcs && snes->max_funcs >= 0) {
      PetscCall(PetscInfo(snes, "Exceeded maximum function evaluations, while checking full step length!\n"));
      snes->reason = SNES_DIVERGED_FUNCTION_COUNT;
      PetscCall(SNESLineSearchSetReason(linesearch, SNES_LINESEARCH_FAILED_FUNCTION));
      PetscFunctionReturn(PETSC_SUCCESS);
    }

    if (objective) {
      PetscCall(SNESComputeObjective(snes, W, &g));
    } else {
      PetscCall((*linesearch->ops->snesfunc)(snes, W, G));
      if (linesearch->ops->vinorm) {
        gnorm = fnorm;
        PetscCall((*linesearch->ops->vinorm)(snes, G, W, &gnorm));
      } else {
        PetscCall(VecNorm(G, NORM_2, &gnorm));
      }
      g = 0.5 * PetscSqr(gnorm);
    }
    PetscCall(SNESLineSearchMonitor(linesearch));

    if (!PetscIsInfOrNanReal(g)) break;
    if (monitor) {
      PetscCall(PetscViewerASCIIAddTab(monitor, ((PetscObject)linesearch)->tablevel));
      PetscCall(PetscViewerASCIIPrintf(monitor, "    Line search: objective function at lambdas = %g is Inf or Nan, cutting lambda\n", (double)lambda));
      PetscCall(PetscViewerASCIISubtractTab(monitor, ((PetscObject)linesearch)->tablevel));
    }
    if (lambda <= minlambda) SNESCheckFunctionNorm(snes, g);
    lambda *= .5;
  }

  if (!objective) PetscCall(PetscInfo(snes, "Initial fnorm %14.12e gnorm %14.12e\n", (double)fnorm, (double)gnorm));
  if (g <= f + lambda * alpha * initslope) { /* Sufficient reduction or step tolerance convergence */
    if (monitor) {
      PetscCall(PetscViewerASCIIAddTab(monitor, ((PetscObject)linesearch)->tablevel));
      if (!objective) {
        PetscCall(PetscViewerASCIIPrintf(monitor, "    Line search: Using full step: fnorm %14.12e gnorm %14.12e\n", (double)fnorm, (double)gnorm));
      } else {
        PetscCall(PetscViewerASCIIPrintf(monitor, "    Line search: Using full step: old obj %14.12e new obj %14.12e\n", (double)f, (double)g));
      }
      PetscCall(PetscViewerASCIISubtractTab(monitor, ((PetscObject)linesearch)->tablevel));
    }
  } else {
    if (stol * xnorm > ynorm) {
      /* Since the full step didn't give sufficient decrease and the step is tiny, exit */
      PetscCall(SNESLineSearchSetNorms(linesearch, xnorm, fnorm, ynorm));
      PetscCall(SNESLineSearchSetReason(linesearch, SNES_LINESEARCH_SUCCEEDED));
      if (monitor) {
        PetscCall(PetscViewerASCIIAddTab(monitor, ((PetscObject)linesearch)->tablevel));
        PetscCall(PetscViewerASCIIPrintf(monitor, "    Line search: Ended due to ynorm < stol*xnorm (%14.12e < %14.12e).\n", (double)ynorm, (double)(stol * xnorm)));
        PetscCall(PetscViewerASCIISubtractTab(monitor, ((PetscObject)linesearch)->tablevel));
      }
      PetscFunctionReturn(PETSC_SUCCESS);
    }
    /* Here to avoid -Wmaybe-uninitiliazed warnings */
    lambdaprev = lambda;
    gprev      = g;
    if (linesearch->order != SNES_LINESEARCH_ORDER_LINEAR) {
      /* Fit points with quadratic */
      lambdatemp = -initslope * PetscSqr(lambda) / (2.0 * (g - f - lambda * initslope));
      lambda     = PetscClipInterval(lambdatemp, .1 * lambda, .5 * lambda);

      PetscCall(VecWAXPY(W, -lambda, Y, X));
      if (linesearch->ops->viproject) PetscCall((*linesearch->ops->viproject)(snes, W));
      if (snes->nfuncs >= snes->max_funcs && snes->max_funcs >= 0) {
        PetscCall(PetscInfo(snes, "Exceeded maximum function evaluations, while attempting quadratic backtracking! %" PetscInt_FMT " \n", snes->nfuncs));
        snes->reason = SNES_DIVERGED_FUNCTION_COUNT;
        PetscCall(SNESLineSearchSetReason(linesearch, SNES_LINESEARCH_FAILED_FUNCTION));
        PetscFunctionReturn(PETSC_SUCCESS);
      }
      if (objective) {
        PetscCall(SNESComputeObjective(snes, W, &g));
      } else {
        PetscCall((*linesearch->ops->snesfunc)(snes, W, G));
        if (linesearch->ops->vinorm) {
          gnorm = fnorm;
          PetscCall((*linesearch->ops->vinorm)(snes, G, W, &gnorm));
        } else {
          PetscCall(VecNorm(G, NORM_2, &gnorm));
        }
        g = 0.5 * PetscSqr(gnorm);
      }
      if (PetscIsInfOrNanReal(g)) {
        PetscCall(SNESLineSearchSetReason(linesearch, SNES_LINESEARCH_FAILED_NANORINF));
        PetscCall(PetscInfo(snes, "Aborted due to Nan or Inf in function evaluation\n"));
        PetscFunctionReturn(PETSC_SUCCESS);
      }
      if (monitor) {
        PetscCall(PetscViewerASCIIAddTab(monitor, ((PetscObject)linesearch)->tablevel));
        if (!objective) {
          PetscCall(PetscViewerASCIIPrintf(monitor, "    Line search: gnorm after quadratic fit %14.12e\n", (double)gnorm));
        } else {
          PetscCall(PetscViewerASCIIPrintf(monitor, "    Line search: obj after quadratic fit %14.12e\n", (double)g));
        }
        PetscCall(PetscViewerASCIISubtractTab(monitor, ((PetscObject)linesearch)->tablevel));
      }
    }
    if (linesearch->order != SNES_LINESEARCH_ORDER_LINEAR && g <= f + lambda * alpha * initslope) { /* sufficient reduction */
      if (monitor) {
        PetscCall(PetscViewerASCIIAddTab(monitor, ((PetscObject)linesearch)->tablevel));
        PetscCall(PetscViewerASCIIPrintf(monitor, "    Line search: Quadratically determined step, lambda=%18.16e\n", (double)lambda));
        PetscCall(PetscViewerASCIISubtractTab(monitor, ((PetscObject)linesearch)->tablevel));
      }
    } else {
      for (count = 0; count < max_it; count++) {
        if (lambda <= minlambda) {
          if (monitor) {
            PetscCall(PetscViewerASCIIAddTab(monitor, ((PetscObject)linesearch)->tablevel));
            PetscCall(PetscViewerASCIIPrintf(monitor, "    Line search: unable to find good step length! After %" PetscInt_FMT " tries \n", count));
            if (!objective) {
              PetscCall(PetscViewerASCIIPrintf(monitor, "    Line search: fnorm=%18.16e, gnorm=%18.16e, ynorm=%18.16e, minlambda=%18.16e, lambda=%18.16e, initial slope=%18.16e\n", (double)fnorm, (double)gnorm, (double)ynorm, (double)minlambda, (double)lambda, (double)initslope));
            } else {
              PetscCall(PetscViewerASCIIPrintf(monitor, "    Line search: obj(0)=%18.16e, obj=%18.16e, ynorm=%18.16e, minlambda=%18.16e, lambda=%18.16e, initial slope=%18.16e\n", (double)f, (double)g, (double)ynorm, (double)minlambda, (double)lambda, (double)initslope));
            }
            PetscCall(PetscViewerASCIISubtractTab(monitor, ((PetscObject)linesearch)->tablevel));
          }
          PetscCall(SNESLineSearchSetReason(linesearch, SNES_LINESEARCH_FAILED_REDUCT));
          PetscFunctionReturn(PETSC_SUCCESS);
        }
        if (linesearch->order == SNES_LINESEARCH_ORDER_CUBIC) {
          /* Fit points with cubic */
          t1 = g - f - lambda * initslope;
          t2 = gprev - f - lambdaprev * initslope;
          a  = (t1 / (lambda * lambda) - t2 / (lambdaprev * lambdaprev)) / (lambda - lambdaprev);
          b  = (-lambdaprev * t1 / (lambda * lambda) + lambda * t2 / (lambdaprev * lambdaprev)) / (lambda - lambdaprev);
          d  = b * b - 3 * a * initslope;
          if (d < 0.0) d = 0.0;
          if (a == 0.0) lambdatemp = -initslope / (2.0 * b);
          else lambdatemp = (-b + PetscSqrtReal(d)) / (3.0 * a);
        } else if (linesearch->order == SNES_LINESEARCH_ORDER_QUADRATIC) {
          lambdatemp = -initslope * PetscSqr(lambda) / (2.0 * (g - f - lambda * initslope));
        } else if (linesearch->order == SNES_LINESEARCH_ORDER_LINEAR) { /* Just backtrack */
          lambdatemp = .5 * lambda;
        } else SETERRQ(PetscObjectComm((PetscObject)linesearch), PETSC_ERR_SUP, "Line search order %" PetscInt_FMT " for type bt", linesearch->order);
        lambdaprev = lambda;
        gprev      = g;

        lambda = PetscClipInterval(lambdatemp, .1 * lambda, .5 * lambda);
        PetscCall(VecWAXPY(W, -lambda, Y, X));
        if (linesearch->ops->viproject) PetscCall((*linesearch->ops->viproject)(snes, W));
        if (snes->nfuncs >= snes->max_funcs && snes->max_funcs >= 0) {
          PetscCall(PetscInfo(snes, "Exceeded maximum function evaluations, while looking for good step length! %" PetscInt_FMT " \n", count));
          if (!objective) PetscCall(PetscInfo(snes, "fnorm=%18.16e, gnorm=%18.16e, ynorm=%18.16e, lambda=%18.16e, initial slope=%18.16e\n", (double)fnorm, (double)gnorm, (double)ynorm, (double)lambda, (double)initslope));
          PetscCall(SNESLineSearchSetReason(linesearch, SNES_LINESEARCH_FAILED_FUNCTION));
          snes->reason = SNES_DIVERGED_FUNCTION_COUNT;
          PetscFunctionReturn(PETSC_SUCCESS);
        }
        if (objective) {
          PetscCall(SNESComputeObjective(snes, W, &g));
        } else {
          PetscCall((*linesearch->ops->snesfunc)(snes, W, G));
          if (linesearch->ops->vinorm) {
            gnorm = fnorm;
            PetscCall((*linesearch->ops->vinorm)(snes, G, W, &gnorm));
          } else {
            PetscCall(VecNorm(G, NORM_2, &gnorm));
          }
          g = 0.5 * PetscSqr(gnorm);
        }
        if (PetscIsInfOrNanReal(g)) {
          PetscCall(SNESLineSearchSetReason(linesearch, SNES_LINESEARCH_FAILED_NANORINF));
          PetscCall(PetscInfo(snes, "Aborted due to Nan or Inf in function evaluation\n"));
          PetscFunctionReturn(PETSC_SUCCESS);
        }
        if (g <= f + lambda * alpha * initslope) { /* is reduction enough? */
          if (monitor) {
            PetscCall(PetscViewerASCIIAddTab(monitor, ((PetscObject)linesearch)->tablevel));
            if (!objective) {
              PetscCall(PetscViewerASCIIPrintf(monitor, "    Line search: %s step, current gnorm %14.12e lambda=%18.16e\n", ordStr[linesearch->order - 1], (double)gnorm, (double)lambda));
              PetscCall(PetscViewerASCIISubtractTab(monitor, ((PetscObject)linesearch)->tablevel));
            } else {
              PetscCall(PetscViewerASCIIPrintf(monitor, "    Line search: %s step, obj %14.12e lambda=%18.16e\n", ordStr[linesearch->order - 1], (double)g, (double)lambda));
              PetscCall(PetscViewerASCIISubtractTab(monitor, ((PetscObject)linesearch)->tablevel));
            }
          }
          break;
        } else if (monitor) {
          PetscCall(PetscViewerASCIIAddTab(monitor, ((PetscObject)linesearch)->tablevel));
          if (!objective) {
            PetscCall(PetscViewerASCIIPrintf(monitor, "    Line search: %s step no good, shrinking lambda, current gnorm %12.12e lambda=%18.16e\n", ordStr[linesearch->order - 1], (double)gnorm, (double)lambda));
            PetscCall(PetscViewerASCIISubtractTab(monitor, ((PetscObject)linesearch)->tablevel));
          } else {
            PetscCall(PetscViewerASCIIPrintf(monitor, "    Line search: %s step no good, shrinking lambda, obj %12.12e lambda=%18.16e\n", ordStr[linesearch->order - 1], (double)g, (double)lambda));
            PetscCall(PetscViewerASCIISubtractTab(monitor, ((PetscObject)linesearch)->tablevel));
          }
        }
      }
    }
  }

  /* postcheck */
  PetscCall(SNESLineSearchSetLambda(linesearch, lambda));
  PetscCall(SNESLineSearchPostCheck(linesearch, X, Y, W, &changed_y, &changed_w));
  if (changed_y) {
    if (!changed_w) PetscCall(VecWAXPY(W, -lambda, Y, X));
    if (linesearch->ops->viproject) PetscCall((*linesearch->ops->viproject)(snes, W));
  }
  if (changed_y || changed_w || objective) { /* recompute the function norm if the step has changed or the objective isn't the norm */
    PetscCall((*linesearch->ops->snesfunc)(snes, W, G));
    if (linesearch->ops->vinorm) {
      gnorm = fnorm;
      PetscCall((*linesearch->ops->vinorm)(snes, G, W, &gnorm));
    } else {
      PetscCall(VecNorm(G, NORM_2, &gnorm));
    }
    PetscCall(VecNorm(Y, NORM_2, &ynorm));
    if (PetscIsInfOrNanReal(gnorm)) {
      PetscCall(SNESLineSearchSetReason(linesearch, SNES_LINESEARCH_FAILED_NANORINF));
      PetscCall(PetscInfo(snes, "Aborted due to Nan or Inf in function evaluation\n"));
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  }

  /* copy the solution over */
  PetscCall(VecCopy(W, X));
  PetscCall(VecCopy(G, F));
  PetscCall(VecNorm(X, NORM_2, &xnorm));
  PetscCall(SNESLineSearchSetNorms(linesearch, xnorm, gnorm, ynorm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESLineSearchView_BT(SNESLineSearch linesearch, PetscViewer viewer)
{
  PetscBool          isascii;
  SNESLineSearch_BT *bt = (SNESLineSearch_BT *)linesearch->data;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    if (linesearch->order == SNES_LINESEARCH_ORDER_CUBIC) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "  interpolation: cubic\n"));
    } else if (linesearch->order == SNES_LINESEARCH_ORDER_QUADRATIC) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "  interpolation: quadratic\n"));
    }
    PetscCall(PetscViewerASCIIPrintf(viewer, "  alpha=%e\n", (double)bt->alpha));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESLineSearchDestroy_BT(SNESLineSearch linesearch)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(linesearch->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESLineSearchSetFromOptions_BT(SNESLineSearch linesearch, PetscOptionItems PetscOptionsObject)
{
  SNESLineSearch_BT *bt = (SNESLineSearch_BT *)linesearch->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "SNESLineSearch BT options");
  PetscCall(PetscOptionsReal("-snes_linesearch_alpha", "Descent tolerance", "SNESLineSearchBT", bt->alpha, &bt->alpha, NULL));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   SNESLINESEARCHBT - Backtracking line search {cite}`dennis:83`.

   This line search finds the minimum of a polynomial fitting either $1/2 ||F(x_k + \lambda Y_k)||_2^2$,
   or the objective function $G(x_k + \lambda Y_k)$ if it is provided with `SNESSetObjective()`.
   If this fit does not satisfy the sufficient decrease conditions, the interval shrinks
   and the fit is reattempted at most `max_it` times or until $\lambda$ is below `minlambda`.

   Options Database Keys:
+  -snes_linesearch_alpha <1e\-4>      - slope descent parameter
.  -snes_linesearch_damping <1.0>      - initial `lambda` on entry to the line search
.  -snes_linesearch_max_it <40>        - maximum number of shrinking iterations in the line search
.  -snes_linesearch_minlambda <1e\-12> - minimum `lambda` (scaling of solution update) allowed
-  -snes_linesearch_order <3>          - order of the polynomial fit, must be 1, 2, or 3. With order 1, it performs a simple backtracking without any curve fitting

   Level: advanced

   Note:
   This line search will always produce a step that is less than or equal to, in length, the full step size.

.seealso: [](ch_snes), `SNESLineSearch`, `SNESLineSearchType`, `SNESLineSearchCreate()`, `SNESLineSearchSetType()`
M*/
PETSC_EXTERN PetscErrorCode SNESLineSearchCreate_BT(SNESLineSearch linesearch)
{
  SNESLineSearch_BT *bt;

  PetscFunctionBegin;
  linesearch->ops->apply          = SNESLineSearchApply_BT;
  linesearch->ops->destroy        = SNESLineSearchDestroy_BT;
  linesearch->ops->setfromoptions = SNESLineSearchSetFromOptions_BT;
  linesearch->ops->reset          = NULL;
  linesearch->ops->view           = SNESLineSearchView_BT;
  linesearch->ops->setup          = NULL;

  PetscCall(PetscNew(&bt));

  linesearch->data   = (void *)bt;
  linesearch->max_it = 40;
  linesearch->order  = SNES_LINESEARCH_ORDER_CUBIC;
  bt->alpha          = 1e-4;
  PetscFunctionReturn(PETSC_SUCCESS);
}
