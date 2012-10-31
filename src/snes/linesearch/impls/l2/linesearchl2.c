#include <petsc-private/linesearchimpl.h>
#include <petscsnes.h>

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchApply_L2"
static PetscErrorCode  SNESLineSearchApply_L2(SNESLineSearch linesearch)
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

  SNESObjective   obj;

  PetscFunctionBegin;

  ierr = SNESLineSearchGetVecs(linesearch, &X, &F, &Y, &W, PETSC_NULL);CHKERRQ(ierr);
  ierr = SNESLineSearchGetNorms(linesearch, &xnorm, &gnorm, &ynorm);CHKERRQ(ierr);
  ierr = SNESLineSearchGetLambda(linesearch, &lambda);CHKERRQ(ierr);
  ierr = SNESLineSearchGetSNES(linesearch, &snes);CHKERRQ(ierr);
  ierr = SNESLineSearchSetSuccess(linesearch, PETSC_TRUE);CHKERRQ(ierr);
  ierr = SNESLineSearchGetTolerances(linesearch, &steptol, &maxstep, &rtol, &atol, &ltol, &max_its);CHKERRQ(ierr);
  ierr = SNESLineSearchGetMonitor(linesearch, &monitor);CHKERRQ(ierr);

  ierr = SNESGetObjective(snes,&obj,PETSC_NULL);CHKERRQ(ierr);

  /* precheck */
  ierr = SNESLineSearchPreCheck(linesearch,X,Y,&changed_y);CHKERRQ(ierr);
  lambda_old = 0.0;
  if (!obj) {
    fnrm_old = gnorm*gnorm;
  } else {
    ierr = SNESComputeObjective(snes,X,&fnrm_old);CHKERRQ(ierr);
  }
  lambda_mid = 0.5*(lambda + lambda_old);

  for (i = 0; i < max_its; i++) {

    /* compute the norm at the midpoint */
    ierr = VecCopy(X, W);CHKERRQ(ierr);
    ierr = VecAXPY(W, -lambda_mid, Y);CHKERRQ(ierr);
    if (linesearch->ops->viproject) {
      ierr = (*linesearch->ops->viproject)(snes, W);CHKERRQ(ierr);
    }
    if (!obj) {
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
    } else {
      /* compute the objective at the midpoint */
      ierr = VecCopy(X, W);CHKERRQ(ierr);
      ierr = VecAXPY(W, -lambda_mid, Y);CHKERRQ(ierr);
      ierr = SNESComputeObjective(snes,W,&fnrm_mid);CHKERRQ(ierr);

      /* compute the objective at the midpoint */
      ierr = VecCopy(X, W);CHKERRQ(ierr);
      ierr = VecAXPY(W, -lambda, Y);CHKERRQ(ierr);
      ierr = SNESComputeObjective(snes,W,&fnrm);CHKERRQ(ierr);
    }
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

    if (monitor) {
      ierr = PetscViewerASCIIAddTab(monitor,((PetscObject)linesearch)->tablevel);CHKERRQ(ierr);
      if (!obj) {
        ierr = PetscViewerASCIIPrintf(monitor,"    Line search: lambdas = [%g, %g, %g], fnorms = [%g, %g, %g]\n",
                                      (double)lambda, (double)lambda_mid, (double)lambda_old, (double)PetscSqrtReal(fnrm), (double)PetscSqrtReal(fnrm_mid), (double)PetscSqrtReal(fnrm_old));CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIIPrintf(monitor,"    Line search: lambdas = [%g, %g, %g], obj = [%g, %g, %g]\n",
                                      (double)lambda, (double)lambda_mid, (double)lambda_old, (double)fnrm, (double)fnrm_mid, (double)fnrm_old);CHKERRQ(ierr);

      }
      ierr = PetscViewerASCIISubtractTab(monitor,((PetscObject)linesearch)->tablevel);CHKERRQ(ierr);
    }

    /* compute the search direction -- always go downhill */
    if (del2Fnrm > 0.) {
      lambda_update = lambda - delFnrm / del2Fnrm;
    } else {
      lambda_update = lambda + delFnrm / del2Fnrm;
    }

    if (lambda_update < steptol) {
      lambda_update = 0.5*(lambda + lambda_old);
    }

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

  ierr = SNESLineSearchSetLambda(linesearch, lambda);CHKERRQ(ierr);
  ierr = SNESLineSearchComputeNorms(linesearch);CHKERRQ(ierr);
  ierr = SNESLineSearchGetNorms(linesearch, &xnorm, &gnorm, &ynorm);CHKERRQ(ierr);

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
#define __FUNCT__ "SNESLineSearchCreate_L2"
/*MC
   SNESLINESEARCHL2 - Secant search in the L2 norm of the function.

   The function norm is evaluated at points in [0, damping] to construct
   a polynomial fitting.  This fitting is used to construct a new lambda
   based upon secant descent.  The process is repeated on the new
   interval, [lambda, lambda_old], max_it - 1 times.

   Options Database Keys:
+  -snes_linesearch_max_it<1> - maximum number of iterations
.  -snes_linesearch_damping<1.0> - initial steplength
-  -snes_linesearch_minlambda - minimum allowable lambda

   Level: advanced

.keywords: SNES, nonlinear, line search, norm, secant

.seealso: SNESLineSearchBT, SNESLineSearchCP, SNESLineSearch
M*/
PETSC_EXTERN_C PetscErrorCode SNESLineSearchCreate_L2(SNESLineSearch linesearch)
{
  PetscFunctionBegin;
  linesearch->ops->apply          = SNESLineSearchApply_L2;
  linesearch->ops->destroy        = PETSC_NULL;
  linesearch->ops->setfromoptions = PETSC_NULL;
  linesearch->ops->reset          = PETSC_NULL;
  linesearch->ops->view           = PETSC_NULL;
  linesearch->ops->setup          = PETSC_NULL;

  linesearch->max_its             = 1;

  PetscFunctionReturn(0);
}
