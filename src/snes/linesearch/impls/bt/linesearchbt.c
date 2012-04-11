#include <petsc-private/linesearchimpl.h> /*I  "petscsnes.h"  I*/
#include <petsc-private/snesimpl.h>

typedef struct {
  PetscReal        alpha; /* sufficient decrease parameter */
} SNESLineSearch_BT;

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchBTSetAlpha"
/*@
   SNESLineSearchBTSetAlpha - Sets the descent parameter, alpha, in the BT linesearch variant.

   Input Parameters:
+  linesearch - linesearch context
-  alpha - The descent parameter

   Level: intermediate

.seealso: SNESLineSearchSetLambda(), SNESLineSearchGetTolerances() SNESLINESEARCHBT
@*/
PetscErrorCode SNESLineSearchBTSetAlpha(SNESLineSearch linesearch, PetscReal alpha)
{
  SNESLineSearch_BT  *bt;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch,SNESLINESEARCH_CLASSID,1);
  bt = (SNESLineSearch_BT *)linesearch->data;
  bt->alpha = alpha;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchBTGetAlpha"
/*@
   SNESLineSearchBTGetAlpha - Gets the descent parameter, alpha, in the BT linesearch variant.

   Input Parameters:
.  linesearch - linesearch context

   Output Parameters:
.  alpha - The descent parameter

   Level: intermediate

.seealso: SNESLineSearchGetLambda(), SNESLineSearchGetTolerances() SNESLINESEARCHBT
@*/
PetscErrorCode SNESLineSearchBTGetAlpha(SNESLineSearch linesearch, PetscReal *alpha)
{
  SNESLineSearch_BT  *bt;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch,SNESLINESEARCH_CLASSID,1);
  bt = (SNESLineSearch_BT *)linesearch->data;
  *alpha = bt->alpha;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchApply_BT"
static PetscErrorCode  SNESLineSearchApply_BT(SNESLineSearch linesearch)
{
  PetscBool      changed_y, changed_w;
  PetscErrorCode ierr;
  Vec            X, F, Y, W, G;
  SNES           snes;
  PetscReal      fnorm, xnorm, ynorm, gnorm, gnormprev;
  PetscReal      lambda, lambdatemp, lambdaprev, minlambda, maxstep, rellength, initslope, alpha;
  PetscReal      t1, t2, a, b, d, steptol;
#if defined(PETSC_USE_COMPLEX)
  PetscScalar    cinitslope;
#endif
  PetscBool      domainerror;
  PetscViewer    monitor;
  PetscInt       max_its, count;
  SNESLineSearch_BT  *bt;
  Mat            jac;


  PetscFunctionBegin;

  ierr = SNESLineSearchGetVecs(linesearch, &X, &F, &Y, &W, &G);CHKERRQ(ierr);
  ierr = SNESLineSearchGetNorms(linesearch, &xnorm, &fnorm, &ynorm);CHKERRQ(ierr);
  ierr = SNESLineSearchGetLambda(linesearch, &lambda);CHKERRQ(ierr);
  ierr = SNESLineSearchGetSNES(linesearch, &snes);CHKERRQ(ierr);
  ierr = SNESLineSearchGetMonitor(linesearch, &monitor);CHKERRQ(ierr);
  ierr = SNESLineSearchGetTolerances(linesearch, &steptol, &maxstep, PETSC_NULL, PETSC_NULL, PETSC_NULL, &max_its);
  bt = (SNESLineSearch_BT *)linesearch->data;

  alpha = bt->alpha;

  ierr = SNESGetJacobian(snes, &jac, PETSC_NULL, PETSC_NULL, PETSC_NULL);CHKERRQ(ierr);
  if (!jac) {
    SETERRQ(((PetscObject)linesearch)->comm, PETSC_ERR_USER, "SNESLineSearchBT requires a Jacobian matrix");
  }
  /* precheck */
  ierr = SNESLineSearchPreCheck(linesearch, &changed_y);CHKERRQ(ierr);
  ierr = SNESLineSearchSetSuccess(linesearch, PETSC_TRUE);CHKERRQ(ierr);

  ierr = VecNorm(Y, NORM_2, &ynorm);CHKERRQ(ierr);
  if (ynorm == 0.0) {
    if (monitor) {
      ierr = PetscViewerASCIIAddTab(monitor,((PetscObject)linesearch)->tablevel);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(monitor,"    Line search: Initial direction and size is 0\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIISubtractTab(monitor,((PetscObject)linesearch)->tablevel);CHKERRQ(ierr);
    }
    ierr   = VecCopy(X,W);CHKERRQ(ierr);
    ierr   = VecCopy(F,G);CHKERRQ(ierr);
    ierr = SNESLineSearchSetSuccess(linesearch, PETSC_FALSE);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  if (ynorm > maxstep) {	/* Step too big, so scale back */
    if (monitor) {
      ierr = PetscViewerASCIIAddTab(monitor,((PetscObject)linesearch)->tablevel);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(monitor,"    Line search: Scaling step by %14.12e old ynorm %14.12e\n", (maxstep/ynorm),ynorm);CHKERRQ(ierr);
      ierr = PetscViewerASCIISubtractTab(monitor,((PetscObject)linesearch)->tablevel);CHKERRQ(ierr);
    }
    ierr = VecScale(Y,maxstep/(ynorm));CHKERRQ(ierr);
    ynorm = maxstep;
  }
  ierr      = VecMaxPointwiseDivide(Y,X,&rellength);CHKERRQ(ierr);
  minlambda = steptol/rellength;
  ierr      = MatMult(jac,Y,W);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr      = VecDot(F,W,&cinitslope);CHKERRQ(ierr);
  initslope = PetscRealPart(cinitslope);
#else
  ierr      = VecDot(F,W,&initslope);CHKERRQ(ierr);
#endif
  if (initslope > 0.0)  initslope = -initslope;
  if (initslope == 0.0) initslope = -1.0;

  ierr = VecWAXPY(W,-lambda,Y,X);CHKERRQ(ierr);
  if (linesearch->ops->viproject) {
    ierr = (*linesearch->ops->viproject)(snes, W);CHKERRQ(ierr);
  }
  if (snes->nfuncs >= snes->max_funcs) {
    ierr  = PetscInfo(snes,"Exceeded maximum function evaluations, while checking full step length!\n");CHKERRQ(ierr);
    snes->reason = SNES_DIVERGED_FUNCTION_COUNT;
    ierr = SNESLineSearchSetSuccess(linesearch, PETSC_FALSE);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = SNESComputeFunction(snes,W,G);CHKERRQ(ierr);
  ierr = SNESGetFunctionDomainError(snes, &domainerror);CHKERRQ(ierr);
  if (domainerror) {
    ierr = SNESLineSearchSetSuccess(linesearch, PETSC_FALSE);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  if (linesearch->ops->vinorm) {
    gnorm = fnorm;
    ierr = (*linesearch->ops->vinorm)(snes, G, W, &gnorm);CHKERRQ(ierr);
  } else {
    ierr = VecNorm(G,NORM_2,&gnorm);CHKERRQ(ierr);
  }

  if (PetscIsInfOrNanReal(gnorm)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FP,"User provided compute function generated a Not-a-Number");
  ierr = PetscInfo2(snes,"Initial fnorm %14.12e gnorm %14.12e\n", fnorm, gnorm);CHKERRQ(ierr);
  if (.5*gnorm*gnorm <= .5*fnorm*fnorm + lambda*alpha*initslope) { /* Sufficient reduction */
    if (monitor) {
      ierr = PetscViewerASCIIAddTab(monitor,((PetscObject)linesearch)->tablevel);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(monitor,"    Line search: Using full step: fnorm %14.12e gnorm %14.12e\n", fnorm, gnorm);CHKERRQ(ierr);
      ierr = PetscViewerASCIISubtractTab(monitor,((PetscObject)linesearch)->tablevel);CHKERRQ(ierr);
    }
  } else {
    /* Fit points with quadratic */
    lambdatemp = -initslope/(gnorm*gnorm - fnorm*fnorm - 2.0*lambda*initslope);
    lambdaprev = lambda;
    gnormprev  = gnorm;
    if (lambdatemp > .5*lambda)  lambdatemp = .5*lambda;
    if (lambdatemp <= .1*lambda) lambda = .1*lambda;
    else                         lambda = lambdatemp;

    ierr  = VecWAXPY(W,-lambda,Y,X);CHKERRQ(ierr);
    if (linesearch->ops->viproject) {
      ierr = (*linesearch->ops->viproject)(snes, W);CHKERRQ(ierr);
    }
    if (snes->nfuncs >= snes->max_funcs) {
      ierr  = PetscInfo1(snes,"Exceeded maximum function evaluations, while attempting quadratic backtracking! %D \n",snes->nfuncs);CHKERRQ(ierr);
      snes->reason = SNES_DIVERGED_FUNCTION_COUNT;
      ierr = SNESLineSearchSetSuccess(linesearch, PETSC_FALSE);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
    ierr = SNESComputeFunction(snes,W,G);CHKERRQ(ierr);
    ierr = SNESGetFunctionDomainError(snes, &domainerror);CHKERRQ(ierr);
    if (domainerror) {
      PetscFunctionReturn(0);
    }
    if (linesearch->ops->vinorm) {
      gnorm = fnorm;
      ierr = (*linesearch->ops->vinorm)(snes, G, W, &gnorm);CHKERRQ(ierr);
    } else {
      ierr = VecNorm(G,NORM_2,&gnorm);CHKERRQ(ierr);
    }
    if (PetscIsInfOrNanReal(gnorm)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FP,"User provided compute function generated a Not-a-Number");
    if (monitor) {
      ierr = PetscViewerASCIIAddTab(monitor,((PetscObject)linesearch)->tablevel);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(monitor,"    Line search: gnorm after quadratic fit %14.12e\n",gnorm);CHKERRQ(ierr);
      ierr = PetscViewerASCIISubtractTab(monitor,((PetscObject)linesearch)->tablevel);CHKERRQ(ierr);
    }
    if (.5*gnorm*gnorm < .5*fnorm*fnorm + lambda*alpha*initslope) { /* sufficient reduction */
      if (monitor) {
        ierr = PetscViewerASCIIAddTab(monitor,((PetscObject)linesearch)->tablevel);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(monitor,"    Line search: Quadratically determined step, lambda=%18.16e\n",(double)lambda);CHKERRQ(ierr);
        ierr = PetscViewerASCIISubtractTab(monitor,((PetscObject)linesearch)->tablevel);CHKERRQ(ierr);
      }
    } else {
      /* Fit points with cubic */
      for (count = 0; count < max_its; count++) {
        if (lambda <= minlambda) {
          if (monitor) {
            ierr = PetscViewerASCIIAddTab(monitor,((PetscObject)linesearch)->tablevel);CHKERRQ(ierr);
            ierr = PetscViewerASCIIPrintf(monitor,"    Line search: unable to find good step length! After %D tries \n",count);CHKERRQ(ierr);
            ierr = PetscViewerASCIIPrintf(monitor,
                                          "    Line search: fnorm=%18.16e, gnorm=%18.16e, ynorm=%18.16e, minlambda=%18.16e, lambda=%18.16e, initial slope=%18.16e\n",
                                          fnorm, gnorm, ynorm, minlambda, lambda, initslope);CHKERRQ(ierr);
            ierr = PetscViewerASCIISubtractTab(monitor,((PetscObject)linesearch)->tablevel);CHKERRQ(ierr);
          }
          ierr = SNESLineSearchSetSuccess(linesearch, PETSC_FALSE);CHKERRQ(ierr);
          PetscFunctionReturn(0);
        }
        if (linesearch->order == SNES_LINESEARCH_ORDER_CUBIC) {
          t1 = .5*(gnorm*gnorm - fnorm*fnorm) - lambda*initslope;
          t2 = .5*(gnormprev*gnormprev  - fnorm*fnorm) - lambdaprev*initslope;
          a  = (t1/(lambda*lambda) - t2/(lambdaprev*lambdaprev))/(lambda-lambdaprev);
          b  = (-lambdaprev*t1/(lambda*lambda) + lambda*t2/(lambdaprev*lambdaprev))/(lambda-lambdaprev);
          d  = b*b - 3*a*initslope;
          if (d < 0.0) d = 0.0;
          if (a == 0.0) {
            lambdatemp = -initslope/(2.0*b);
          } else {
            lambdatemp = (-b + PetscSqrtReal(d))/(3.0*a);
          }
        } else if (linesearch->order == SNES_LINESEARCH_ORDER_QUADRATIC) {
          lambdatemp = -initslope/(gnorm*gnorm - fnorm*fnorm - 2.0*initslope);
        } else {
          SETERRQ(((PetscObject)linesearch)->comm, PETSC_ERR_SUP, "unsupported line search order for type bt");
        }
          lambdaprev = lambda;
          gnormprev  = gnorm;
        if (lambdatemp > .5*lambda)  lambdatemp = .5*lambda;
        if (lambdatemp <= .1*lambda) lambda     = .1*lambda;
        else                         lambda     = lambdatemp;
        ierr  = VecWAXPY(W,-lambda,Y,X);CHKERRQ(ierr);
        if (snes->nfuncs >= snes->max_funcs) {
          ierr = PetscInfo1(snes,"Exceeded maximum function evaluations, while looking for good step length! %D \n",count);CHKERRQ(ierr);
          ierr = PetscInfo5(snes,"fnorm=%18.16e, gnorm=%18.16e, ynorm=%18.16e, lambda=%18.16e, initial slope=%18.16e\n",
                            fnorm,gnorm,ynorm,lambda,initslope);CHKERRQ(ierr);
          ierr = SNESLineSearchSetSuccess(linesearch, PETSC_FALSE);CHKERRQ(ierr);
          snes->reason = SNES_DIVERGED_FUNCTION_COUNT;
          PetscFunctionReturn(0);
        }
        ierr = SNESComputeFunction(snes,W,G);CHKERRQ(ierr);
        ierr = SNESGetFunctionDomainError(snes, &domainerror);CHKERRQ(ierr);
        if (domainerror) {
          ierr = SNESLineSearchSetSuccess(linesearch, PETSC_FALSE);CHKERRQ(ierr);
          PetscFunctionReturn(0);
        }
        if (linesearch->ops->vinorm) {
          gnorm = fnorm;
          ierr = (*linesearch->ops->vinorm)(snes, G, W, &gnorm);CHKERRQ(ierr);
        } else {
          ierr = VecNorm(G,NORM_2,&gnorm);CHKERRQ(ierr);
        }
        if (PetscIsInfOrNanReal(gnorm)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FP,"User provided compute function generated a Not-a-Number");
        if (.5*gnorm*gnorm < .5*fnorm*fnorm + lambda*alpha*initslope) { /* is reduction enough? */
          if (monitor) {
            ierr = PetscViewerASCIIAddTab(monitor,((PetscObject)linesearch)->tablevel);CHKERRQ(ierr);
            if (linesearch->order == SNES_LINESEARCH_ORDER_CUBIC) {
              ierr = PetscViewerASCIIPrintf(monitor,"    Line search: Cubically determined step, current gnorm %14.12e lambda=%18.16e\n",gnorm,lambda);CHKERRQ(ierr);
            } else {
              ierr = PetscViewerASCIIPrintf(monitor,"    Line search: Quadratically determined step, current gnorm %14.12e lambda=%18.16e\n",gnorm,lambda);CHKERRQ(ierr);
            }
            ierr = PetscViewerASCIISubtractTab(monitor,((PetscObject)linesearch)->tablevel);CHKERRQ(ierr);
          }
          break;
        } else {
          if (monitor) {
            ierr = PetscViewerASCIIAddTab(monitor,((PetscObject)linesearch)->tablevel);CHKERRQ(ierr);
            if (linesearch->order == SNES_LINESEARCH_ORDER_CUBIC) {
              ierr = PetscViewerASCIIPrintf(monitor,"    Line search: Cubic step no good, shrinking lambda, current gnorm %12.12e lambda=%18.16e\n",gnorm,lambda);CHKERRQ(ierr);
            } else {
              ierr = PetscViewerASCIIPrintf(monitor,"    Line search: Quadratic step no good, shrinking lambda, current gnorm %12.12e lambda=%18.16e\n",gnorm,lambda);CHKERRQ(ierr);
            }
            ierr = PetscViewerASCIISubtractTab(monitor,((PetscObject)linesearch)->tablevel);CHKERRQ(ierr);
          }
        }
      }
    }
  }

  /* postcheck */
  ierr = SNESLineSearchPostCheck(linesearch, &changed_y, &changed_w);CHKERRQ(ierr);
  if (changed_y) {
    ierr = VecWAXPY(W,-lambda,Y,X);CHKERRQ(ierr);
    if (linesearch->ops->viproject) {
      ierr = (*linesearch->ops->viproject)(snes, W);CHKERRQ(ierr);
    }
  }
  if (changed_y || changed_w) { /* recompute the function if the step has changed */
    ierr = SNESComputeFunction(snes,W,G);CHKERRQ(ierr);
    ierr = SNESGetFunctionDomainError(snes, &domainerror);CHKERRQ(ierr);
    if (domainerror) {
      ierr = SNESLineSearchSetSuccess(linesearch, PETSC_FALSE);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
    if (linesearch->ops->vinorm) {
      gnorm = fnorm;
      ierr = (*linesearch->ops->vinorm)(snes, G, W, &gnorm);CHKERRQ(ierr);
    } else {
      ierr = VecNorm(G,NORM_2,&gnorm);CHKERRQ(ierr);
    }
    ierr = VecNorm(Y,NORM_2,&ynorm);CHKERRQ(ierr);
    if (PetscIsInfOrNanReal(gnorm)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FP,"User provided compute function generated a Not-a-Number");

  }

  /* copy the solution over */
  ierr = VecCopy(W, X);CHKERRQ(ierr);
  ierr = VecCopy(G, F);CHKERRQ(ierr);
  ierr = VecNorm(X, NORM_2, &xnorm);CHKERRQ(ierr);
  ierr = SNESLineSearchSetLambda(linesearch, lambda);CHKERRQ(ierr);
  ierr = SNESLineSearchSetNorms(linesearch, xnorm, gnorm, ynorm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchView_BT"
PetscErrorCode SNESLineSearchView_BT(SNESLineSearch linesearch, PetscViewer viewer)
{
  PetscErrorCode    ierr;
  PetscBool         iascii;
  SNESLineSearch_BT *bt;
  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  bt = (SNESLineSearch_BT*)linesearch->data;
  if (iascii) {
    if (linesearch->order == SNES_LINESEARCH_ORDER_CUBIC) {
    ierr = PetscViewerASCIIPrintf(viewer, "  interpolation: cubic\n");CHKERRQ(ierr);
    } else if (linesearch->order == SNES_LINESEARCH_ORDER_QUADRATIC) {
    ierr = PetscViewerASCIIPrintf(viewer, "  interpolation: quadratic\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer, "  alpha=%G\n", bt->alpha);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchDestroy_BT"
static PetscErrorCode SNESLineSearchDestroy_BT(SNESLineSearch linesearch)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(linesearch->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchSetFromOptions_BT"
static PetscErrorCode SNESLineSearchSetFromOptions_BT(SNESLineSearch linesearch)
{

  PetscErrorCode       ierr;
  SNESLineSearch_BT    *bt;
  PetscFunctionBegin;

  bt = (SNESLineSearch_BT*)linesearch->data;

  ierr = PetscOptionsHead("SNESLineSearch BT options");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-snes_linesearch_alpha",   "Descent tolerance",        "SNESLineSearchBT", bt->alpha, &bt->alpha, PETSC_NULL);CHKERRQ(ierr);

  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchCreate_BT"
/*MC
   SNESLINESEARCHBT - Backtracking line search.

   This line search finds the minimum of a polynomial fitting of the L2 norm of the
   function. If this fit does not satisfy the conditions for progress, the interval shrinks
   and the fit is reattempted at most max_it times or until lambda is below minlambda.

   Options Database Keys:
+  -snes_linesearch_alpha<1e-4> - slope descent parameter
.  -snes_linesearch_damping<1.0> - initial step length
.  -snes_linesearch_max_it<40> - maximum number of shrinking step
.  -snes_linesearch_minlambda<1e-12> - minimum step length allowed
-  -snes_linesearch_order<cubic,quadratic> - order of the approximation

   Level: advanced

   Notes:
   This line search is taken from "Numerical Methods for Unconstrained
   Optimization and Nonlinear Equations" by Dennis and Schnabel, page 325.

.keywords: SNES, SNESLineSearch, damping

.seealso: SNESLineSearchCreate(), SNESLineSearchSetType()
M*/
PETSC_EXTERN_C PetscErrorCode SNESLineSearchCreate_BT(SNESLineSearch linesearch)
{

  SNESLineSearch_BT  *bt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  linesearch->ops->apply          = SNESLineSearchApply_BT;
  linesearch->ops->destroy        = SNESLineSearchDestroy_BT;
  linesearch->ops->setfromoptions = SNESLineSearchSetFromOptions_BT;
  linesearch->ops->reset          = PETSC_NULL;
  linesearch->ops->view           = SNESLineSearchView_BT;
  linesearch->ops->setup          = PETSC_NULL;

  ierr = PetscNewLog(linesearch, SNESLineSearch_BT, &bt);CHKERRQ(ierr);
  linesearch->data = (void *)bt;
  linesearch->max_its = 40;
  linesearch->order = SNES_LINESEARCH_ORDER_CUBIC;
  bt->alpha = 1e-4;

  PetscFunctionReturn(0);
}
