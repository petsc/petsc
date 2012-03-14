#include <private/linesearchimpl.h>
#include <private/snesimpl.h>

typedef enum {LINESEARCH_BT_QUADRATIC, LINESEARCH_BT_CUBIC} LineSearchBTOrder;

typedef struct {
  PetscReal        alpha; /* sufficient decrease parameter */
  LineSearchBTOrder order;
} LineSearch_BT;

/*MC
   LineSearchBT - Backtracking line searches.

   These linesearches try a polynomial fit for the L2 norm of the error
   using the gradient.  Failing that, they step back and try again.

   Level: advanced

.keywords: SNES, LineSearch, damping

.seealso: LineSearchCreate(), LineSearchSetType()
M*/

#undef __FUNCT__
#define __FUNCT__ "LineSearchApply_BT"

PetscErrorCode  LineSearchApply_BT(LineSearch linesearch)
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
  LineSearch_BT  *bt;
  Mat            jac;


  PetscFunctionBegin;

  ierr = LineSearchGetVecs(linesearch, &X, &F, &Y, &W, &G);CHKERRQ(ierr);
  ierr = LineSearchGetNorms(linesearch, &xnorm, &fnorm, &ynorm);CHKERRQ(ierr);
  ierr = LineSearchGetLambda(linesearch, &lambda);CHKERRQ(ierr);
  ierr = LineSearchGetSNES(linesearch, &snes);CHKERRQ(ierr);
  ierr = LineSearchGetMonitor(linesearch, &monitor);CHKERRQ(ierr);
  ierr = LineSearchGetTolerances(linesearch, &steptol, &maxstep, PETSC_NULL, PETSC_NULL, PETSC_NULL, &max_its);
  bt = (LineSearch_BT *)linesearch->data;

  alpha = bt->alpha;

  ierr = SNESGetJacobian(snes, &jac, PETSC_NULL, PETSC_NULL, PETSC_NULL);CHKERRQ(ierr);
  if (!jac) {
    SETERRQ(((PetscObject)linesearch)->comm, PETSC_ERR_USER, "LineSearchBT requires a Jacobian matrix");
  }

  ierr = PetscLogEventBegin(LineSearch_Apply,snes,X,F,G);CHKERRQ(ierr);

  /* precheck */
  ierr = LineSearchPreCheck(linesearch, &changed_y);CHKERRQ(ierr);
  ierr = LineSearchSetSuccess(linesearch, PETSC_TRUE);CHKERRQ(ierr);

  ierr = VecNorm(Y, NORM_2, &ynorm);CHKERRQ(ierr);
  if (ynorm == 0.0) {
    if (monitor) {
      ierr = PetscViewerASCIIAddTab(monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(monitor,"    Line search: Initial direction and size is 0\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIISubtractTab(monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
    }
    ierr   = VecCopy(X,W);CHKERRQ(ierr);
    ierr   = VecCopy(F,G);CHKERRQ(ierr);
    ierr = LineSearchSetSuccess(linesearch, PETSC_FALSE);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  if (ynorm > maxstep) {	/* Step too big, so scale back */
    if (monitor) {
      ierr = PetscViewerASCIIAddTab(monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(monitor,"    Line search: Scaling step by %14.12e old ynorm %14.12e\n", (maxstep/ynorm),ynorm);CHKERRQ(ierr);
      ierr = PetscViewerASCIISubtractTab(monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
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

  ierr = VecWAXPY(W,-1.0,Y,X);CHKERRQ(ierr);
  if (snes->nfuncs >= snes->max_funcs) {
    ierr  = PetscInfo(snes,"Exceeded maximum function evaluations, while checking full step length!\n");CHKERRQ(ierr);
    snes->reason = SNES_DIVERGED_FUNCTION_COUNT;
    ierr = LineSearchSetSuccess(linesearch, PETSC_FALSE);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = SNESComputeFunction(snes,W,G);CHKERRQ(ierr);
  if (snes->domainerror) {
    ierr = PetscLogEventEnd(SNES_LineSearch,snes,X,F,G);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = VecNorm(G,NORM_2,&gnorm);CHKERRQ(ierr);
  if (PetscIsInfOrNanReal(gnorm)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FP,"User provided compute function generated a Not-a-Number");
  ierr = PetscInfo2(snes,"Initial fnorm %14.12e gnorm %14.12e\n", fnorm, gnorm);CHKERRQ(ierr);
  if (.5*gnorm*gnorm <= .5*fnorm*fnorm + lambda*alpha*initslope) { /* Sufficient reduction */
    if (monitor) {
      ierr = PetscViewerASCIIAddTab(monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(monitor,"    Line search: Using full step: fnorm %14.12e gnorm %14.12e\n", fnorm, gnorm);CHKERRQ(ierr);
      ierr = PetscViewerASCIISubtractTab(monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
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
    if (snes->nfuncs >= snes->max_funcs) {
      ierr  = PetscInfo1(snes,"Exceeded maximum function evaluations, while attempting quadratic backtracking! %D \n",snes->nfuncs);CHKERRQ(ierr);
      snes->reason = SNES_DIVERGED_FUNCTION_COUNT;
      ierr = LineSearchSetSuccess(linesearch, PETSC_FALSE);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
    ierr = SNESComputeFunction(snes,W,G);CHKERRQ(ierr);
    if (snes->domainerror) {
      ierr = PetscLogEventEnd(SNES_LineSearch,snes,X,F,G);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
    ierr = VecNorm(G,NORM_2,&gnorm);CHKERRQ(ierr);
    if (PetscIsInfOrNanReal(gnorm)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FP,"User provided compute function generated a Not-a-Number");
    if (monitor) {
      ierr = PetscViewerASCIIAddTab(monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(monitor,"    Line search: gnorm after quadratic fit %14.12e\n",gnorm);CHKERRQ(ierr);
      ierr = PetscViewerASCIISubtractTab(monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
    }
    if (.5*gnorm*gnorm < .5*fnorm*fnorm + lambda*alpha*initslope) { /* sufficient reduction */
      if (monitor) {
        ierr = PetscViewerASCIIAddTab(monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(monitor,"    Line search: Quadratically determined step, lambda=%18.16e\n",(double)lambda);CHKERRQ(ierr);
        ierr = PetscViewerASCIISubtractTab(monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
      }
    } else {
      /* Fit points with cubic */
      for (count = 0; count < max_its; count++) {
        if (lambda <= minlambda) {
          if (monitor) {
            ierr = PetscViewerASCIIAddTab(monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
            ierr = PetscViewerASCIIPrintf(monitor,"    Line search: unable to find good step length! After %D tries \n",count);CHKERRQ(ierr);
            ierr = PetscViewerASCIIPrintf(monitor,
                                          "    Line search: fnorm=%18.16e, gnorm=%18.16e, ynorm=%18.16e, minlambda=%18.16e, lambda=%18.16e, initial slope=%18.16e\n",
                                          fnorm, gnorm, ynorm, minlambda, lambda, initslope);CHKERRQ(ierr);
            ierr = PetscViewerASCIISubtractTab(monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
          }
          ierr = LineSearchSetSuccess(linesearch, PETSC_FALSE);CHKERRQ(ierr);
          PetscFunctionReturn(0);
        }
        if (bt->order == LINESEARCH_BT_CUBIC) {
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
        } else if (bt->order == LINESEARCH_BT_QUADRATIC) {
          lambdatemp = -initslope/(gnorm*gnorm - fnorm*fnorm - 2.0*initslope);
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
          ierr = LineSearchSetSuccess(linesearch, PETSC_FALSE);CHKERRQ(ierr);
          snes->reason = SNES_DIVERGED_FUNCTION_COUNT;
          PetscFunctionReturn(0);
        }
        ierr = SNESComputeFunction(snes,W,G);CHKERRQ(ierr);
        if (snes->domainerror) {
          ierr = PetscLogEventEnd(SNES_LineSearch,snes,X,F,G);CHKERRQ(ierr);
          PetscFunctionReturn(0);
        }
        ierr = VecNorm(G,NORM_2,&gnorm);CHKERRQ(ierr);
        if (PetscIsInfOrNanReal(gnorm)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FP,"User provided compute function generated a Not-a-Number");
        if (.5*gnorm*gnorm < .5*fnorm*fnorm + lambda*alpha*initslope) { /* is reduction enough? */
          if (monitor) {
            ierr = PetscViewerASCIIAddTab(monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
            if (bt->order == LINESEARCH_BT_CUBIC) {
              ierr = PetscViewerASCIIPrintf(monitor,"    Line search: Cubically determined step, current gnorm %14.12e lambda=%18.16e\n",gnorm,lambda);CHKERRQ(ierr);
            } else {
              ierr = PetscViewerASCIIPrintf(monitor,"    Line search: Quadratically determined step, current gnorm %14.12e lambda=%18.16e\n",gnorm,lambda);CHKERRQ(ierr);
            }
            ierr = PetscViewerASCIISubtractTab(monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
          }
          break;
        } else {
          if (monitor) {
            ierr = PetscViewerASCIIAddTab(monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
            if (bt->order == LINESEARCH_BT_CUBIC) {
              ierr = PetscViewerASCIIPrintf(monitor,"    Line search: Cubic step no good, shrinking lambda, current gnorm %12.12e lambda=%18.16e\n",gnorm,lambda);CHKERRQ(ierr);
            } else {
              ierr = PetscViewerASCIIPrintf(monitor,"    Line search: Quadratic step no good, shrinking lambda, current gnorm %12.12e lambda=%18.16e\n",gnorm,lambda);CHKERRQ(ierr);
            }
            ierr = PetscViewerASCIISubtractTab(monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
          }
        }
      }
    }
  }

  /* postcheck */
  ierr = LineSearchPostCheck(linesearch, &changed_y, &changed_w);CHKERRQ(ierr);
  if (changed_y) {
    ierr = VecWAXPY(W,-lambda,Y,X);CHKERRQ(ierr);
  }
  ierr = SNESComputeFunction(snes,W,F);CHKERRQ(ierr);
  ierr = SNESGetFunctionDomainError(snes, &domainerror);CHKERRQ(ierr);
  if (domainerror) {
    ierr = LineSearchSetSuccess(linesearch, PETSC_FALSE);
    PetscFunctionReturn(0);
  }

  /* copy the solution over */
  ierr = VecCopy(W, X);CHKERRQ(ierr);
  ierr = LineSearchSetLambda(linesearch, lambda);CHKERRQ(ierr);
  ierr = LineSearchComputeNorms(linesearch);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "LineSearchDestroy_BT"
PetscErrorCode LineSearchDestroy_BT(LineSearch linesearch) {

  PetscFunctionBegin;
  PetscFunctionReturn(0);

}


#undef __FUNCT__
#define __FUNCT__ "LineSearchSetFromOptions_BT"
static PetscErrorCode LineSearchSetFromOptions_BT(LineSearch linesearch)
{

  PetscErrorCode ierr;
  LineSearch_BT    *bt;
  const char       *orders[] = {"quadratic", "cubic"};
  PetscInt         indx = 0;
  PetscBool        flg;
  PetscFunctionBegin;

  bt = (LineSearch_BT*)linesearch->data;

  ierr = PetscOptionsHead("LineSearch BT options");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-linesearch_bt_alpha",   "Descent tolerance",        "LineSearchBT", bt->alpha, &bt->alpha, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEList("-linesearch_bt_order",  "Order of approximation",   "LineSearchBT", orders,2,"quadratic",&indx,&flg);CHKERRQ(ierr);
  if (flg) {
    switch (indx) {
    case 0: bt->order = LINESEARCH_BT_QUADRATIC;
      break;
    case 1: bt->order = LINESEARCH_BT_CUBIC;
      break;
    }
  }

  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "LineSearchCreate_BT"
PetscErrorCode LineSearchCreate_BT(LineSearch linesearch)
{

  LineSearch_BT  *bt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  linesearch->ops->apply          = LineSearchApply_BT;
  linesearch->ops->destroy        = LineSearchDestroy_BT;
  linesearch->ops->setfromoptions = LineSearchSetFromOptions_BT;
  linesearch->ops->reset          = PETSC_NULL;
  linesearch->ops->view           = PETSC_NULL;
  linesearch->ops->setup          = PETSC_NULL;

  ierr = PetscNewLog(linesearch, LineSearch_BT, &bt);CHKERRQ(ierr);
  linesearch->data = (void *)bt;
  linesearch->max_its = 40;
  bt->order = LINESEARCH_BT_CUBIC;
  bt->alpha = 1e-4;

  PetscFunctionReturn(0);
}
EXTERN_C_END
