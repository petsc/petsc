#include <petsc-private/linesearchimpl.h>
#include <petscsnes.h>

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchApply_CP"
static PetscErrorCode SNESLineSearchApply_CP(SNESLineSearch linesearch)
{
  PetscBool      changed_y, changed_w, domainerror;
  PetscErrorCode ierr;
  Vec             X, Y, F, W;
  SNES            snes;
  PetscReal       xnorm, ynorm, gnorm, steptol, atol, rtol, ltol, maxstep;

  PetscReal       lambda, lambda_old, lambda_update, delLambda;
  PetscScalar     fty, fty_init, fty_old, fty_mid1, fty_mid2, s;
  PetscInt        i, max_its;

  PetscViewer     monitor;

  PetscFunctionBegin;

  ierr = SNESLineSearchGetVecs(linesearch, &X, &F, &Y, &W, PETSC_NULL);CHKERRQ(ierr);
  ierr = SNESLineSearchGetNorms(linesearch, &xnorm, &gnorm, &ynorm);CHKERRQ(ierr);
  ierr = SNESLineSearchGetSNES(linesearch, &snes);CHKERRQ(ierr);
  ierr = SNESLineSearchGetLambda(linesearch, &lambda);CHKERRQ(ierr);
  ierr = SNESLineSearchGetTolerances(linesearch, &steptol, &maxstep, &rtol, &atol, &ltol, &max_its);CHKERRQ(ierr);
  ierr = SNESLineSearchSetSuccess(linesearch, PETSC_TRUE);CHKERRQ(ierr);
  ierr = SNESLineSearchGetMonitor(linesearch, &monitor);CHKERRQ(ierr);

  /* precheck */
  ierr = SNESLineSearchPreCheck(linesearch,X,Y,&changed_y);CHKERRQ(ierr);
  lambda_old = 0.0;
  ierr = VecDot(F, Y, &fty_old);CHKERRQ(ierr);
  fty_init = fty_old;

  for (i = 0; i < max_its; i++) {

    /* compute the norm at lambda */
    ierr = VecCopy(X, W);CHKERRQ(ierr);
    ierr = VecAXPY(W, -lambda, Y);CHKERRQ(ierr);
    if (linesearch->ops->viproject) {
      ierr = (*linesearch->ops->viproject)(snes, W);CHKERRQ(ierr);
    }
    ierr = SNESComputeFunction(snes, W, F);CHKERRQ(ierr);

    ierr = VecDot(F, Y, &fty);CHKERRQ(ierr);

    delLambda    = lambda - lambda_old;

    /* check for convergence */
    if (PetscAbsReal(delLambda) < steptol*lambda) break;
    if (PetscAbsScalar(fty) / PetscAbsScalar(fty_init) < rtol) break;
    if (PetscAbsScalar(fty) < atol && i > 0) break;
    if (monitor) {
      ierr = PetscViewerASCIIAddTab(monitor,((PetscObject)linesearch)->tablevel);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(monitor,"    Line search: lambdas = [%g, %g], ftys = [%g, %g]\n",(double)lambda, (double)lambda_old, (double)PetscRealPart(fty), (double)PetscRealPart(fty_old));CHKERRQ(ierr);
      ierr = PetscViewerASCIISubtractTab(monitor,((PetscObject)linesearch)->tablevel);CHKERRQ(ierr);
    }

    /* compute the search direction */
    if (linesearch->order == SNES_LINESEARCH_ORDER_LINEAR) {
      s = (fty - fty_old) / delLambda;
    } else if (linesearch->order == SNES_LINESEARCH_ORDER_QUADRATIC) {
      ierr = VecCopy(X, W);CHKERRQ(ierr);
      ierr = VecAXPY(W, -0.5*(lambda + lambda_old), Y);CHKERRQ(ierr);
      if (linesearch->ops->viproject) {
        ierr = (*linesearch->ops->viproject)(snes, W);CHKERRQ(ierr);
      }
      ierr = SNESComputeFunction(snes, W, F);CHKERRQ(ierr);
      ierr = VecDot(F, Y, &fty_mid1);CHKERRQ(ierr);
      s = (3.*fty - 4.*fty_mid1 + fty_old) / delLambda;
    } else {
      ierr = VecCopy(X, W);CHKERRQ(ierr);
      ierr = VecAXPY(W, -0.5*(lambda + lambda_old), Y);CHKERRQ(ierr);
      if (linesearch->ops->viproject) {
        ierr = (*linesearch->ops->viproject)(snes, W);CHKERRQ(ierr);
      }
      ierr = SNESComputeFunction(snes, W, F);CHKERRQ(ierr);
      ierr = VecDot(F, Y, &fty_mid1);CHKERRQ(ierr);
      ierr = VecCopy(X, W);CHKERRQ(ierr);
      ierr = VecAXPY(W, -(lambda + 0.5*(lambda - lambda_old)), Y);CHKERRQ(ierr);
      if (linesearch->ops->viproject) {
        ierr = (*linesearch->ops->viproject)(snes, W);CHKERRQ(ierr);
      }
      ierr = SNESComputeFunction(snes, W, F);CHKERRQ(ierr);
      ierr = VecDot(F, Y, &fty_mid2);CHKERRQ(ierr);
      s = (2.*fty_mid2 + 3.*fty - 6.*fty_mid1 + fty_old) / (3.*delLambda);
    }
    /* if the solve is going in the wrong direction, fix it */
    if (PetscRealPart(s) > 0.) s = -s;
    lambda_update =  lambda - PetscRealPart(fty / s);

    /* switch directions if we stepped out of bounds */
    if (lambda_update < steptol) {
      lambda_update = lambda + PetscRealPart(fty / s);
    }

    if (PetscIsInfOrNanScalar(lambda_update)) break;
    if (lambda_update > maxstep) {
      break;
    }

    /* compute the new state of the line search */
    lambda_old = lambda;
    lambda = lambda_update;
    fty_old = fty;
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
  ierr = SNESComputeFunction(snes,X,F);CHKERRQ(ierr);
  ierr = SNESGetFunctionDomainError(snes, &domainerror);CHKERRQ(ierr);
  if (domainerror) {
    ierr = SNESLineSearchSetSuccess(linesearch, PETSC_FALSE);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  ierr = SNESLineSearchComputeNorms(linesearch);CHKERRQ(ierr);
  ierr = SNESLineSearchGetNorms(linesearch, &xnorm, &gnorm, &ynorm);CHKERRQ(ierr);

  ierr = SNESLineSearchSetLambda(linesearch, lambda);CHKERRQ(ierr);

  if (monitor) {
    ierr = PetscViewerASCIIAddTab(monitor,((PetscObject)linesearch)->tablevel);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(monitor,"    Line search terminated: lambda = %g, fnorms = %g\n", (double)lambda, (double)gnorm);CHKERRQ(ierr);
    ierr = PetscViewerASCIISubtractTab(monitor,((PetscObject)linesearch)->tablevel);CHKERRQ(ierr);
  }
  if (lambda <= steptol) {
    ierr = SNESLineSearchSetSuccess(linesearch, PETSC_FALSE);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchCreate_CP"
/*MC
   SNESLINESEARCHCP - Critical point line search. This line search assumes that there exists some
   artificial G(x) for which the SNESFunction F(x) = grad G(x).  Therefore, this line search seeks
   to find roots of dot(F, Y) via a secant method.

   Options Database Keys:
+  -snes_linesearch_minlambda - the minimum acceptable lambda
.  -snes_linesearch_damping - initial trial step length
-  -snes_linesearch_max_it  - the maximum number of secant steps performed.

   Notes:
   This method is the preferred line search for SNESQN and SNESNCG.

   Level: advanced

.keywords: SNES, SNESLineSearch, damping

.seealso: SNESLineSearchCreate(), SNESLineSearchSetType()
M*/
PETSC_EXTERN_C PetscErrorCode SNESLineSearchCreate_CP(SNESLineSearch linesearch)
{
  PetscFunctionBegin;
  linesearch->ops->apply          = SNESLineSearchApply_CP;
  linesearch->ops->destroy        = PETSC_NULL;
  linesearch->ops->setfromoptions = PETSC_NULL;
  linesearch->ops->reset          = PETSC_NULL;
  linesearch->ops->view           = PETSC_NULL;
  linesearch->ops->setup          = PETSC_NULL;
  linesearch->order = SNES_LINESEARCH_ORDER_LINEAR;

  linesearch->max_its =            1;

  PetscFunctionReturn(0);
}
