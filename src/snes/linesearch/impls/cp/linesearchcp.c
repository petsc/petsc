#include <private/linesearchimpl.h>
#include <petscsnes.h>

#undef __FUNCT__
#define __FUNCT__ "LineSearchApply_CP"

/*@C
   LineSearchCP - This routine is not a line search at all;
   it simply uses the full step.  Thus, this routine is intended
   to serve as a template and is not recommended for general use.

   Logically Collective on SNES and Vec

   Input Parameters:
+  snes - nonlinear context
.  lsctx - optional context for line search (not used here)
.  x - current iterate
.  f - residual evaluated at x
.  y - search direction
.  fnorm - 2-norm of f
-  xnorm - norm of x if known, otherwise 0

   Output Parameters:
+  g - residual evaluated at new iterate y
.  w - new iterate
.  gnorm - 2-norm of g
.  ynorm - 2-norm of search length
-  flag - PETSC_TRUE on success, PETSC_FALSE on failure

   Options Database Key:
.  -snes_ls basic - Activates SNESLineSearchNo()

   Level: advanced

.keywords: SNES, nonlinear, line search, cubic

.seealso: SNESLineSearchCubic(), SNESLineSearchQuadratic(),
          SNESLineSearchSet(), SNESLineSearchNoNorms()
@*/
PetscErrorCode  PetscLineSearchApply_CP(PetscLineSearch linesearch)
{

  PetscBool      changed_y, changed_w, domainerror;
  PetscErrorCode ierr;
  Vec             X, Y, F, W;
  SNES            snes;
  PetscReal       xnorm, ynorm, gnorm, steptol, atol, rtol, ltol, maxstep;

  PetscReal       lambda, lambda_old, lambda_update, delLambda;
  PetscScalar     fty, fty_old;
  PetscInt        i, max_its;

  PetscViewer     monitor;

  PetscFunctionBegin;

  ierr = PetscLineSearchGetVecs(linesearch, &X, &F, &Y, &W, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscLineSearchGetNorms(linesearch, &xnorm, &gnorm, &ynorm);CHKERRQ(ierr);
  ierr = PetscLineSearchGetSNES(linesearch, &snes);CHKERRQ(ierr);
  ierr = PetscLineSearchGetLambda(linesearch, &lambda);CHKERRQ(ierr);
  ierr = PetscLineSearchGetTolerances(linesearch, &steptol, &maxstep, &rtol, &atol, &ltol, &max_its);CHKERRQ(ierr);
  ierr = PetscLineSearchSetSuccess(linesearch, PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscLineSearchGetMonitor(linesearch, &monitor);CHKERRQ(ierr);

  /* precheck */
  ierr = PetscLineSearchPreCheck(linesearch, &changed_y);CHKERRQ(ierr);
  lambda_old = 0.0;
  ierr = VecDot(F, Y, &fty_old);CHKERRQ(ierr);
  for (i = 0; i < max_its; i++) {

    /* compute the norm at lambda */
    ierr = VecCopy(X, W);CHKERRQ(ierr);
    ierr = VecAXPY(W, -lambda, Y);CHKERRQ(ierr);
    ierr = SNESComputeFunction(snes, W, F);CHKERRQ(ierr);

    ierr = VecDot(F, Y, &fty);CHKERRQ(ierr);

    delLambda    = lambda - lambda_old;
    if (PetscAbsReal(delLambda) < steptol) break;
    if (monitor) {
      ierr = PetscViewerASCIIAddTab(monitor,((PetscObject)linesearch)->tablevel);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(monitor,"    Line search: lambdas = [%g, %g], ftys = [%g, %g]\n",
                                    lambda, lambda_old, PetscRealPart(fty), PetscRealPart(fty_old));CHKERRQ(ierr);
      ierr = PetscViewerASCIISubtractTab(monitor,((PetscObject)linesearch)->tablevel);CHKERRQ(ierr);
    }

    /* compute the search direction */
    lambda_update =  PetscRealPart((fty*lambda_old - fty_old*lambda) / (fty - fty_old));
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

  /* postcheck */
  ierr = PetscLineSearchPostCheck(linesearch, &changed_y, &changed_w);CHKERRQ(ierr);
  if (changed_y) {
    ierr = VecAXPY(X, -lambda, Y);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(W, X);CHKERRQ(ierr);
  }
  ierr = SNESComputeFunction(snes,X,F);CHKERRQ(ierr);
  ierr = SNESGetFunctionDomainError(snes, &domainerror);CHKERRQ(ierr);
  if (domainerror) {
    ierr = PetscLineSearchSetSuccess(linesearch, PETSC_FALSE);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  ierr = PetscLineSearchComputeNorms(linesearch);CHKERRQ(ierr);
  ierr = PetscLineSearchGetNorms(linesearch, &xnorm, &gnorm, &ynorm);CHKERRQ(ierr);

  ierr = PetscLineSearchSetLambda(linesearch, lambda);CHKERRQ(ierr);

  if (monitor) {
    ierr = PetscViewerASCIIAddTab(monitor,((PetscObject)linesearch)->tablevel);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(monitor,"    Line search terminated: lambda = %g, fnorms = %g\n", lambda, gnorm);CHKERRQ(ierr);
    ierr = PetscViewerASCIISubtractTab(monitor,((PetscObject)linesearch)->tablevel);CHKERRQ(ierr);
  }
  if (lambda <= steptol) {
    ierr = PetscLineSearchSetSuccess(linesearch, PETSC_FALSE);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PetscLineSearchCreate_CP"
PetscErrorCode PetscLineSearchCreate_CP(PetscLineSearch linesearch)
{
  PetscFunctionBegin;
  linesearch->ops->apply          = PetscLineSearchApply_CP;
  linesearch->ops->destroy        = PETSC_NULL;
  linesearch->ops->setfromoptions = PETSC_NULL;
  linesearch->ops->reset          = PETSC_NULL;
  linesearch->ops->view           = PETSC_NULL;
  linesearch->ops->setup          = PETSC_NULL;
  PetscFunctionReturn(0);
}
EXTERN_C_END
