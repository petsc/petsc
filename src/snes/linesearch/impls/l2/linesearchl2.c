#include <private/linesearchimpl.h>
#include <petscsnes.h>

#undef __FUNCT__
#define __FUNCT__ "PetscLineSearchApply_L2"

/*@C
   PetscLineSearchL2 - This routine is not a line search at all;
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
PetscErrorCode  PetscLineSearchApply_L2(PetscLineSearch linesearch)
{

  PetscBool      changed_y, changed_w;
  PetscErrorCode ierr;
  Vec             X;
  Vec             F;
  Vec             Y;
  Vec             W;
  SNES            snes;
  PetscReal       gnorm;
  PetscReal       ynorm;
  PetscReal       xnorm;
  PetscReal       steptol, maxstep, rtol, atol, ltol;

  PetscViewer     monitor;
  PetscBool       domainerror;
  PetscReal       lambda, lambda_old, lambda_mid, lambda_update, delLambda;
  PetscReal       fnrm, fnrm_old, fnrm_mid;
  PetscReal       delFnrm, delFnrm_old, del2Fnrm;
  PetscInt        i, max_its;

  PetscFunctionBegin;

  ierr = PetscLineSearchGetVecs(linesearch, &X, &F, &Y, &W, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscLineSearchGetNorms(linesearch, &xnorm, &gnorm, &ynorm);CHKERRQ(ierr);
  ierr = PetscLineSearchGetLambda(linesearch, &lambda);CHKERRQ(ierr);
  ierr = PetscLineSearchGetSNES(linesearch, &snes);CHKERRQ(ierr);
  ierr = PetscLineSearchSetSuccess(linesearch, PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscLineSearchGetTolerances(linesearch, &steptol, &maxstep, &rtol, &atol, &ltol, &max_its);CHKERRQ(ierr);
  ierr = PetscLineSearchGetMonitor(linesearch, &monitor);CHKERRQ(ierr);

  /* precheck */
  ierr = PetscLineSearchPreCheck(linesearch, &changed_y);CHKERRQ(ierr);
  lambda_old = 0.0;
  fnrm_old = gnorm*gnorm;
  lambda_mid = 0.5*(lambda + lambda_old);

  for (i = 0; i < max_its; i++) {

  /* compute the norm at the midpoint */
  ierr = VecCopy(X, W);CHKERRQ(ierr);
  ierr = VecAXPY(W, -lambda_mid, Y);CHKERRQ(ierr);
  if (linesearch->ops->viproject) {
    ierr = (*linesearch->ops->viproject)(snes, W);CHKERRQ(ierr);
  }
  ierr = SNESComputeFunction(snes, W, F);CHKERRQ(ierr);
  if (linesearch->ops->vinorm) {
    fnrm_mid = gnorm;
    ierr = (*linesearch->ops->vinorm)(snes, F, W, &fnrm_mid);CHKERRQ(ierr);
  } else {
    ierr = VecNorm(F, NORM_2, &fnrm_mid);CHKERRQ(ierr);
  }
  fnrm_mid = fnrm_mid*fnrm_mid;

  /* compute the norm at lambda */
  ierr = VecCopy(X, W);CHKERRQ(ierr);
  ierr = VecAXPY(W, -lambda, Y);CHKERRQ(ierr);
  if (linesearch->ops->viproject) {
    ierr = (*linesearch->ops->viproject)(snes, W);CHKERRQ(ierr);
  }
  ierr = SNESComputeFunction(snes, W, F);CHKERRQ(ierr);
  if (linesearch->ops->vinorm) {
    fnrm = gnorm;
    ierr = (*linesearch->ops->vinorm)(snes, F, W, &fnrm);CHKERRQ(ierr);
  } else {
    ierr = VecNorm(F, NORM_2, &fnrm);CHKERRQ(ierr);
  }
  fnrm = fnrm*fnrm;

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

    while ((del2Fnrm < 0.0) && (fabs(delLambda) > steptol)) {
      fnrm_old = fnrm_mid;
      lambda_old = lambda_mid;
      lambda_mid = 0.5*(lambda_old + lambda);
      ierr = VecCopy(X, W);CHKERRQ(ierr);
      ierr = VecAXPY(W, -lambda_mid, Y);CHKERRQ(ierr);
      if (linesearch->ops->viproject) {
        ierr = (*linesearch->ops->viproject)(snes, W);CHKERRQ(ierr);
      }
      ierr = SNESComputeFunction(snes, W, F);CHKERRQ(ierr);
      if (linesearch->ops->vinorm) {
        fnrm_mid = gnorm;
        ierr = (*linesearch->ops->vinorm)(snes, F, W, &fnrm_mid);CHKERRQ(ierr);
      } else {
        ierr = VecNorm(F, NORM_2, &fnrm_mid);CHKERRQ(ierr);
      }
      fnrm_mid = fnrm_mid*fnrm_mid;
      delLambda    = lambda - lambda_old;
      delFnrm      = (3.*fnrm - 4.*fnrm_mid + 1.*fnrm_old) / delLambda;
      delFnrm_old  = (-3.*fnrm_old + 4.*fnrm_mid -1.*fnrm) / delLambda;
      del2Fnrm = (delFnrm - delFnrm_old) / delLambda;
    }
 
    if (monitor) {
      ierr = PetscViewerASCIIAddTab(monitor,((PetscObject)linesearch)->tablevel);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(monitor,"    Line search: lambdas = [%g, %g, %g], fnorms = [%g, %g, %g]\n",
                                    lambda, lambda_mid, lambda_old, PetscSqrtReal(fnrm), PetscSqrtReal(fnrm_mid), PetscSqrtReal(fnrm_old));CHKERRQ(ierr);
      ierr = PetscViewerASCIISubtractTab(monitor,((PetscObject)linesearch)->tablevel);CHKERRQ(ierr);
    }

    /* compute the search direction */
    lambda_update = lambda - delFnrm*delLambda / (delFnrm - delFnrm_old);
    if (PetscIsInfOrNanScalar(lambda_update)) break;
    if (lambda_update > maxstep) {
      break;
    }

    /* compute the new state of the line search */
    lambda_old = lambda;
    lambda = lambda_update;
    fnrm_old = fnrm;
    lambda_mid = 0.5*(lambda + lambda_old);
  }
  /* construct the solution */
  ierr = VecCopy(X, W);CHKERRQ(ierr);
  ierr = VecAXPY(W, -lambda, Y);CHKERRQ(ierr);
  if (linesearch->ops->viproject) {
    ierr = (*linesearch->ops->viproject)(snes, W);CHKERRQ(ierr);
  }

  /* postcheck */
  ierr = PetscLineSearchPostCheck(linesearch, &changed_y, &changed_w);CHKERRQ(ierr);
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
    ierr = PetscLineSearchSetSuccess(linesearch, PETSC_FALSE);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  ierr = PetscLineSearchSetLambda(linesearch, lambda);CHKERRQ(ierr);
  ierr = PetscLineSearchComputeNorms(linesearch);CHKERRQ(ierr);
  ierr = PetscLineSearchGetNorms(linesearch, &xnorm, &gnorm, &ynorm);CHKERRQ(ierr);

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
#define __FUNCT__ "PetscLineSearchCreate_L2"
PetscErrorCode PetscLineSearchCreate_L2(PetscLineSearch linesearch)
{
  PetscFunctionBegin;
  linesearch->ops->apply          = PetscLineSearchApply_L2;
  linesearch->ops->destroy        = PETSC_NULL;
  linesearch->ops->setfromoptions = PETSC_NULL;
  linesearch->ops->reset          = PETSC_NULL;
  linesearch->ops->view           = PETSC_NULL;
  linesearch->ops->setup          = PETSC_NULL;
  PetscFunctionReturn(0);
}
EXTERN_C_END
