#include <private/linesearchimpl.h>
#include <private/snesimpl.h>

#undef __FUNCT__
#define __FUNCT__ "LineSearchApply_L2"

/*@C
   LineSearchL2 - This routine is not a line search at all;
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
PetscErrorCode  LineSearchApply_L2(LineSearch linesearch)
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

  PetscReal        lambda, lambda_old, lambda_mid, lambda_update, delLambda;
  PetscReal        fnrm, fnrm_old, fnrm_mid;
  PetscReal        delFnrm, delFnrm_old, del2Fnrm;
  PetscInt         i;

  PetscFunctionBegin;
  /* precheck */
  ierr = LineSearchPreCheck(linesearch, &changed_y);CHKERRQ(ierr);

  lambda = linesearch->lambda;
  lambda_old = 0.0;
  fnrm_old = (*gnorm)*(*gnorm);
  lambda_mid = 0.5*(lambda + lambda_old);
  linesearch->success = PETSC_TRUE;
  for (i = 0; i < linesearch->max_its; i++) {

  /* compute the norm at the midpoint */
  ierr = VecCopy(X, W);CHKERRQ(ierr);
  ierr = VecAXPY(W, -lambda_mid, Y);CHKERRQ(ierr);
  ierr = SNESComputeFunction(snes, W, F);CHKERRQ(ierr);
  ierr = VecDot(F, F, &fnrm_mid);CHKERRQ(ierr);

  /* compute the norm at lambda */
  ierr = VecCopy(X, W);CHKERRQ(ierr);
  ierr = VecAXPY(W, -lambda, Y);CHKERRQ(ierr);
  ierr = SNESComputeFunction(snes, W, F);CHKERRQ(ierr);
  ierr = VecDot(F, F, &fnrm);CHKERRQ(ierr);

  /* this gives us the derivatives at the endpoints -- compute them from here

   a = x - a0

   p_0(x) = (x / dA - 1)(2x / dA - 1)
   p_1(x) = 4(x / dA)(1 - x / dA)
   p_2(x) = (x / dA)(2x / dA - 1)

   dp_0[0] / dx = 3 / dA
   dp_1[0] / dx = -4 / dA
   dp_2[0] / dx = 1 / dA

   dp_0[dA] / dx = -1 / dA
   dp_1[dA] / dx = 4 / dA
   dp_2[dA] / dx = -3 / dA

   d^2p_0[0] / dx^2 =  4 / dA^2
   d^2p_1[0] / dx^2 = -8 / dA^2
   d^2p_2[0] / dx^2 =  4 / dA^2
     */

    delLambda    = lambda - lambda_old;
    delFnrm      = (3.*fnrm - 4.*fnrm_mid + 1.*fnrm_old) / delLambda;
    delFnrm_old  = (-3.*fnrm_old + 4.*fnrm_mid -1.*fnrm) / delLambda;
    del2Fnrm = (delFnrm - delFnrm_old) / delLambda;

    /* check for positive curvature -- looking for that root wouldn't be a good thing. */
    while ((PetscRealPart(del2Fnrm) < 0.0) && (fabs(delLambda) > snes->steptol)) {
      fnrm_old = fnrm_mid;
      lambda_old = lambda_mid;
      lambda_mid = 0.5*(lambda_old + lambda);
      ierr = VecCopy(X, W);CHKERRQ(ierr);
      ierr = VecAXPY(W, -lambda_mid, Y);CHKERRQ(ierr);
      ierr = SNESComputeFunction(snes, W, F);CHKERRQ(ierr);
      ierr = VecDot(F, F, &fnrm_mid);CHKERRQ(ierr);
      delLambda    = lambda - lambda_old;
      delFnrm      = (3.*fnrm - 4.*fnrm_mid + 1.*fnrm_old) / delLambda;
      delFnrm_old  = (-3.*fnrm_old + 4.*fnrm_mid -1.*fnrm) / delLambda;
      del2Fnrm = (delFnrm - delFnrm_old) / delLambda;
    }

    if (linesearch->monitor) {
      ierr = PetscViewerASCIIAddTab(linesearch->monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(linesearch->monitor,"    Line search: lambdas = [%g, %g, %g], fnorms = [%g, %g, %g]\n",
                                    lambda, lambda_mid, lambda_old, PetscSqrtReal(PetscRealPart(fnrm)), PetscSqrtReal(PetscRealPart(fnrm_mid)), PetscSqrtReal(PetscRealPart(fnrm_old)));CHKERRQ(ierr);
      ierr = PetscViewerASCIISubtractTab(linesearch->monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
    }

    /* compute the search direction */
    lambda_update = lambda - PetscRealPart(delFnrm)*delLambda / PetscRealPart(delFnrm - delFnrm_old);
    if (PetscIsInfOrNanScalar(lambda_update)) break;
    if (lambda_update > snes->maxstep) {
      break;
    }

    /* compute the new state of the line search */
    lambda_old = lambda;
    lambda = lambda_update;
    fnrm_old = fnrm;
    lambda_mid = 0.5*(lambda + lambda_old);
  }
  /* postcheck */
  ierr = LineSearchPostCheck(linesearch, &changed_y, &changed_w);CHKERRQ(ierr);
  if (changed_y) {
    ierr = VecAXPY(X,-snes->damping,Y);CHKERRQ(ierr);
  }
  ierr = SNESComputeFunction(snes,X,F);CHKERRQ(ierr);
  if (snes->domainerror) {
    linesearch->success = PETSC_FALSE;
    PetscFunctionReturn(0);
  }
  ierr = VecAXPY(X, -lambda, Y);CHKERRQ(ierr);
  ierr = SNESComputeFunction(snes, X, F);CHKERRQ(ierr);
  ierr = VecNormBegin(F, NORM_2, gnorm);CHKERRQ(ierr);
  ierr = VecNormBegin(X, NORM_2, xnorm);CHKERRQ(ierr);
  ierr = VecNormBegin(Y, NORM_2, ynorm);CHKERRQ(ierr);
  ierr = VecNormEnd(F, NORM_2, gnorm);CHKERRQ(ierr);
  ierr = VecNormEnd(X, NORM_2, xnorm);CHKERRQ(ierr);
  ierr = VecNormEnd(Y, NORM_2, ynorm);CHKERRQ(ierr);

  linesearch->lambda = lambda;
  if (linesearch->monitor) {
    ierr = PetscViewerASCIIAddTab(linesearch->monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(linesearch->monitor,"    Line search terminated: lambda = %g, fnorms = %g\n", lambda, linesearch->fnorm);CHKERRQ(ierr);
    ierr = PetscViewerASCIISubtractTab(linesearch->monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
  }
  if (lambda <= snes->steptol) {
    linesearch->success = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "LineSearchCreate_L2"
PetscErrorCode LineSearchCreate_L2(LineSearch linesearch)
{
  PetscFunctionBegin;
  linesearch->ops->apply          = LineSearchApply_L2;
  linesearch->ops->destroy        = PETSC_NULL;
  linesearch->ops->setfromoptions = PETSC_NULL;
  linesearch->ops->reset          = PETSC_NULL;
  linesearch->ops->view           = PETSC_NULL;
  linesearch->ops->setup          = PETSC_NULL;
  PetscFunctionReturn(0);
}
EXTERN_C_END
