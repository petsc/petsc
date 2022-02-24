#include <petsc/private/linesearchimpl.h>
#include <petscsnes.h>

static PetscErrorCode SNESLineSearchApply_CP(SNESLineSearch linesearch)
{
  PetscBool      changed_y, changed_w;
  Vec            X, Y, F, W;
  SNES           snes;
  PetscReal      xnorm, ynorm, gnorm, steptol, atol, rtol, ltol, maxstep;
  PetscReal      lambda, lambda_old, lambda_update, delLambda;
  PetscScalar    fty, fty_init, fty_old, fty_mid1, fty_mid2, s;
  PetscInt       i, max_its;
  PetscViewer    monitor;

  PetscFunctionBegin;
  CHKERRQ(SNESLineSearchGetVecs(linesearch, &X, &F, &Y, &W, NULL));
  CHKERRQ(SNESLineSearchGetNorms(linesearch, &xnorm, &gnorm, &ynorm));
  CHKERRQ(SNESLineSearchGetSNES(linesearch, &snes));
  CHKERRQ(SNESLineSearchGetLambda(linesearch, &lambda));
  CHKERRQ(SNESLineSearchGetTolerances(linesearch, &steptol, &maxstep, &rtol, &atol, &ltol, &max_its));
  CHKERRQ(SNESLineSearchSetReason(linesearch, SNES_LINESEARCH_SUCCEEDED));
  CHKERRQ(SNESLineSearchGetDefaultMonitor(linesearch, &monitor));

  /* precheck */
  CHKERRQ(SNESLineSearchPreCheck(linesearch,X,Y,&changed_y));
  lambda_old = 0.0;

  CHKERRQ(VecDot(F,Y,&fty_old));
  if (PetscAbsScalar(fty_old) < atol * ynorm) {
    if (monitor) {
      CHKERRQ(PetscViewerASCIIAddTab(monitor,((PetscObject)linesearch)->tablevel));
      CHKERRQ(PetscViewerASCIIPrintf(monitor,"    Line search terminated at initial point because dot(F,Y) = %g < atol*||y|| = %g\n",(double)PetscAbsScalar(fty_old), (double)atol*ynorm));
      CHKERRQ(PetscViewerASCIISubtractTab(monitor,((PetscObject)linesearch)->tablevel));
    }
    CHKERRQ(SNESSetConvergedReason(linesearch->snes,SNES_CONVERGED_FNORM_ABS));
    PetscFunctionReturn(0);
  }

  fty_init = fty_old;

  for (i = 0; i < max_its; i++) {
    /* compute the norm at lambda */
    CHKERRQ(VecCopy(X, W));
    CHKERRQ(VecAXPY(W, -lambda, Y));
    if (linesearch->ops->viproject) {
      CHKERRQ((*linesearch->ops->viproject)(snes, W));
    }
    CHKERRQ((*linesearch->ops->snesfunc)(snes,W,F));
    CHKERRQ(VecDot(F,Y,&fty));

    delLambda = lambda - lambda_old;

    /* check for convergence */
    if (PetscAbsReal(delLambda) < steptol*lambda) break;
    if (PetscAbsScalar(fty) / PetscAbsScalar(fty_init) < rtol) break;
    if (PetscAbsScalar(fty) < atol * ynorm && i > 0) break;
    if (monitor) {
      CHKERRQ(PetscViewerASCIIAddTab(monitor,((PetscObject)linesearch)->tablevel));
      CHKERRQ(PetscViewerASCIIPrintf(monitor,"    Line search: lambdas = [%g, %g], ftys = [%g, %g]\n",(double)lambda, (double)lambda_old, (double)PetscRealPart(fty), (double)PetscRealPart(fty_old)));
      CHKERRQ(PetscViewerASCIISubtractTab(monitor,((PetscObject)linesearch)->tablevel));
    }

    /* compute the search direction */
    if (linesearch->order == SNES_LINESEARCH_ORDER_LINEAR) {
      s = (fty - fty_old) / delLambda;
    } else if (linesearch->order == SNES_LINESEARCH_ORDER_QUADRATIC) {
      CHKERRQ(VecCopy(X, W));
      CHKERRQ(VecAXPY(W, -0.5*(lambda + lambda_old), Y));
      if (linesearch->ops->viproject) {
        CHKERRQ((*linesearch->ops->viproject)(snes, W));
      }
      CHKERRQ((*linesearch->ops->snesfunc)(snes,W,F));
      CHKERRQ(VecDot(F, Y, &fty_mid1));
      s    = (3.*fty - 4.*fty_mid1 + fty_old) / delLambda;
    } else {
      CHKERRQ(VecCopy(X, W));
      CHKERRQ(VecAXPY(W, -0.5*(lambda + lambda_old), Y));
      if (linesearch->ops->viproject) {
        CHKERRQ((*linesearch->ops->viproject)(snes, W));
      }
      CHKERRQ((*linesearch->ops->snesfunc)(snes,W,F));
      CHKERRQ(VecDot(F, Y, &fty_mid1));
      CHKERRQ(VecCopy(X, W));
      CHKERRQ(VecAXPY(W, -(lambda + 0.5*(lambda - lambda_old)), Y));
      if (linesearch->ops->viproject) {
        CHKERRQ((*linesearch->ops->viproject)(snes, W));
      }
      CHKERRQ((*linesearch->ops->snesfunc)(snes, W, F));
      CHKERRQ(VecDot(F, Y, &fty_mid2));
      s    = (2.*fty_mid2 + 3.*fty - 6.*fty_mid1 + fty_old) / (3.*delLambda);
    }
    /* if the solve is going in the wrong direction, fix it */
    if (PetscRealPart(s) > 0.) s = -s;
    if (s == 0.0) break;
    lambda_update =  lambda - PetscRealPart(fty / s);

    /* switch directions if we stepped out of bounds */
    if (lambda_update < steptol) lambda_update = lambda + PetscRealPart(fty / s);

    if (PetscIsInfOrNanReal(lambda_update)) break;
    if (lambda_update > maxstep) break;

    /* compute the new state of the line search */
    lambda_old = lambda;
    lambda     = lambda_update;
    fty_old    = fty;
  }
  /* construct the solution */
  CHKERRQ(VecCopy(X, W));
  CHKERRQ(VecAXPY(W, -lambda, Y));
  if (linesearch->ops->viproject) {
    CHKERRQ((*linesearch->ops->viproject)(snes, W));
  }
  /* postcheck */
  CHKERRQ(SNESLineSearchPostCheck(linesearch,X,Y,W,&changed_y,&changed_w));
  if (changed_y) {
    CHKERRQ(VecAXPY(X, -lambda, Y));
    if (linesearch->ops->viproject) {
      CHKERRQ((*linesearch->ops->viproject)(snes, X));
    }
  } else {
    CHKERRQ(VecCopy(W, X));
  }
  CHKERRQ((*linesearch->ops->snesfunc)(snes,X,F));

  CHKERRQ(SNESLineSearchComputeNorms(linesearch));
  CHKERRQ(SNESLineSearchGetNorms(linesearch, &xnorm, &gnorm, &ynorm));

  CHKERRQ(SNESLineSearchSetLambda(linesearch, lambda));

  if (monitor) {
    CHKERRQ(PetscViewerASCIIAddTab(monitor,((PetscObject)linesearch)->tablevel));
    CHKERRQ(PetscViewerASCIIPrintf(monitor,"    Line search terminated: lambda = %g, fnorms = %g\n", (double)lambda, (double)gnorm));
    CHKERRQ(PetscViewerASCIISubtractTab(monitor,((PetscObject)linesearch)->tablevel));
  }
  if (lambda <= steptol) {
    CHKERRQ(SNESLineSearchSetReason(linesearch, SNES_LINESEARCH_FAILED_REDUCT));
  }
  PetscFunctionReturn(0);
}

/*MC
   SNESLINESEARCHCP - Critical point line search. This line search assumes that there exists some
   artificial G(x) for which the SNESFunction F(x) = grad G(x).  Therefore, this line search seeks
   to find roots of dot(F, Y) via a secant method.

   Options Database Keys:
+  -snes_linesearch_minlambda <minlambda> - the minimum acceptable lambda
.  -snes_linesearch_maxstep <length> - the algorithm insures that a step length is never longer than this value
.  -snes_linesearch_damping <damping> - initial trial step length is scaled by this factor, default is 1.0
-  -snes_linesearch_max_it <max_it> - the maximum number of secant steps performed.

   Notes:
   This method does NOT use the objective function if it is provided with SNESSetObjective().

   This method is the preferred line search for SNESQN and SNESNCG.

   Level: advanced

.seealso: SNESLineSearchCreate(), SNESLineSearchSetType()
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

  linesearch->max_its = 1;
  PetscFunctionReturn(0);
}
