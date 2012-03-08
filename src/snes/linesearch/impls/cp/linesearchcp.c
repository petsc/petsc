#include <private/linesearchimpl.h>
#include <private/snesimpl.h>

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
PetscErrorCode  LineSearchApply_CP(LineSearch linesearch)
{

  PetscBool      changed_y, changed_w;
  PetscErrorCode ierr;
  Vec             X = linesearch->vec_sol;
  Vec             F  = linesearch->vec_func;
  Vec             Y  = linesearch->vec_update;
  Vec             W =  linesearch->vec_sol_new;
  SNES            snes  = linesearch->snes;
  PetscReal       *gnorm  = &linesearch->fnorm;
  PetscReal       *ynorm  = &linesearch->ynorm;
  PetscReal       *xnorm =  &linesearch->xnorm;

  PetscReal       lambda, lambda_old, lambda_update, delLambda;
  PetscScalar     fty, fty_old;
  PetscInt        i;

  PetscFunctionBegin;
  /* precheck */
  ierr = LineSearchPreCheck(linesearch, &changed_y);CHKERRQ(ierr);
  lambda = linesearch->lambda;
  lambda_old = 0.0;
  ierr = VecDot(F, Y, &fty_old);CHKERRQ(ierr);
  linesearch->success = PETSC_TRUE;
  for (i = 0; i < linesearch->max_its; i++) {

    /* compute the norm at lambda */
    ierr = VecCopy(X, W);CHKERRQ(ierr);
    ierr = VecAXPY(W, -lambda, Y);CHKERRQ(ierr);
    ierr = SNESComputeFunction(snes, W, F);CHKERRQ(ierr);

    ierr = VecDot(F, Y, &fty);CHKERRQ(ierr);

    delLambda    = lambda - lambda_old;
    if (PetscAbsReal(delLambda) < linesearch->steptol) break;
    if (linesearch->monitor) {
      ierr = PetscViewerASCIIAddTab(linesearch->monitor,((PetscObject)linesearch)->tablevel);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(linesearch->monitor,"    Line search: lambdas = [%g, %g], ftys = [%g, %g]\n",
                                    lambda, lambda_old, PetscRealPart(fty), PetscRealPart(fty_old));CHKERRQ(ierr);
      ierr = PetscViewerASCIISubtractTab(linesearch->monitor,((PetscObject)linesearch)->tablevel);CHKERRQ(ierr);
    }

    /* compute the search direction */
    lambda_update =  PetscRealPart((fty*lambda_old - fty_old*lambda) / (fty - fty_old));
    if (PetscIsInfOrNanScalar(lambda_update)) break;
    if (lambda_update > linesearch->maxstep) {
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
  ierr = LineSearchPostCheck(linesearch, &changed_y, &changed_w);CHKERRQ(ierr);
  if (changed_y) {
    ierr = VecAXPY(X, -lambda, Y);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(W, X);CHKERRQ(ierr);
  }
  ierr = SNESComputeFunction(snes,X,F);CHKERRQ(ierr);
  if (snes->domainerror) {
    linesearch->success = PETSC_FALSE;
    PetscFunctionReturn(0);
  }

  ierr = VecNormBegin(F, NORM_2, gnorm);CHKERRQ(ierr);
  ierr = VecNormBegin(X, NORM_2, xnorm);CHKERRQ(ierr);
  ierr = VecNormBegin(Y, NORM_2, ynorm);CHKERRQ(ierr);
  ierr = VecNormEnd(F, NORM_2, gnorm);CHKERRQ(ierr);
  ierr = VecNormEnd(X, NORM_2, xnorm);CHKERRQ(ierr);
  ierr = VecNormEnd(Y, NORM_2, ynorm);CHKERRQ(ierr);

  linesearch->lambda = lambda;
  if (linesearch->monitor) {
    ierr = PetscViewerASCIIAddTab(linesearch->monitor,((PetscObject)linesearch)->tablevel);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(linesearch->monitor,"    Line search terminated: lambda = %g, fnorms = %g\n", lambda, linesearch->fnorm);CHKERRQ(ierr);
    ierr = PetscViewerASCIISubtractTab(linesearch->monitor,((PetscObject)linesearch)->tablevel);CHKERRQ(ierr);
  }
  if (lambda <= linesearch->steptol) {
    linesearch->success = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "LineSearchCreate_CP"
PetscErrorCode LineSearchCreate_CP(LineSearch linesearch)
{
  PetscFunctionBegin;
  linesearch->ops->apply          = LineSearchApply_CP;
  linesearch->ops->destroy        = PETSC_NULL;
  linesearch->ops->setfromoptions = PETSC_NULL;
  linesearch->ops->reset          = PETSC_NULL;
  linesearch->ops->view           = PETSC_NULL;
  linesearch->ops->setup          = PETSC_NULL;
  PetscFunctionReturn(0);
}
EXTERN_C_END
