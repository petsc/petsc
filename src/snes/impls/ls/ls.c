
#include <../src/snes/impls/ls/lsimpl.h>

/*
     Checks if J^T F = 0 which implies we've found a local minimum of the norm of the function,
    || F(u) ||_2 but not a zero, F(u) = 0. In the case when one cannot compute J^T F we use the fact that
    0 = (J^T F)^T W = F^T J W iff W not in the null space of J. Thanks for Jorge More 
    for this trick. One assumes that the probability that W is in the null space of J is very, very small.
*/ 
#undef __FUNCT__  
#define __FUNCT__ "SNESLSCheckLocalMin_Private"
PetscErrorCode SNESLSCheckLocalMin_Private(SNES snes,Mat A,Vec F,Vec W,PetscReal fnorm,PetscBool  *ismin)
{
  PetscReal      a1;
  PetscErrorCode ierr;
  PetscBool      hastranspose;

  PetscFunctionBegin;
  *ismin = PETSC_FALSE;
  ierr = MatHasOperation(A,MATOP_MULT_TRANSPOSE,&hastranspose);CHKERRQ(ierr);
  if (hastranspose) {
    /* Compute || J^T F|| */
    ierr = MatMultTranspose(A,F,W);CHKERRQ(ierr);
    ierr = VecNorm(W,NORM_2,&a1);CHKERRQ(ierr);
    ierr = PetscInfo1(snes,"|| J^T F|| %14.12e near zero implies found a local minimum\n",(double)(a1/fnorm));CHKERRQ(ierr);
    if (a1/fnorm < 1.e-4) *ismin = PETSC_TRUE;
  } else {
    Vec         work;
    PetscScalar result;
    PetscReal   wnorm;

    ierr = VecSetRandom(W,PETSC_NULL);CHKERRQ(ierr);
    ierr = VecNorm(W,NORM_2,&wnorm);CHKERRQ(ierr);
    ierr = VecDuplicate(W,&work);CHKERRQ(ierr);
    ierr = MatMult(A,W,work);CHKERRQ(ierr);
    ierr = VecDot(F,work,&result);CHKERRQ(ierr);
    ierr = VecDestroy(&work);CHKERRQ(ierr);
    a1   = PetscAbsScalar(result)/(fnorm*wnorm);
    ierr = PetscInfo1(snes,"(F^T J random)/(|| F ||*||random|| %14.12e near zero implies found a local minimum\n",(double)a1);CHKERRQ(ierr);
    if (a1 < 1.e-4) *ismin = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

/*
     Checks if J^T(F - J*X) = 0 
*/ 
#undef __FUNCT__  
#define __FUNCT__ "SNESLSCheckResidual_Private"
PetscErrorCode SNESLSCheckResidual_Private(SNES snes,Mat A,Vec F,Vec X,Vec W1,Vec W2)
{
  PetscReal      a1,a2;
  PetscErrorCode ierr;
  PetscBool      hastranspose;

  PetscFunctionBegin;
  ierr = MatHasOperation(A,MATOP_MULT_TRANSPOSE,&hastranspose);CHKERRQ(ierr);
  if (hastranspose) {
    ierr = MatMult(A,X,W1);CHKERRQ(ierr);
    ierr = VecAXPY(W1,-1.0,F);CHKERRQ(ierr);

    /* Compute || J^T W|| */
    ierr = MatMultTranspose(A,W1,W2);CHKERRQ(ierr);
    ierr = VecNorm(W1,NORM_2,&a1);CHKERRQ(ierr);
    ierr = VecNorm(W2,NORM_2,&a2);CHKERRQ(ierr);
    if (a1 != 0.0) {
      ierr = PetscInfo1(snes,"||J^T(F-Ax)||/||F-AX|| %14.12e near zero implies inconsistent rhs\n",(double)(a2/a1));CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchCubic_LS"
/*@C
   SNESLineSearchCubic - Performs a cubic line search (default line search method).

   Collective on SNES

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
-  flag - PETSC_TRUE if line search succeeds; PETSC_FALSE on failure.

   Options Database Key:
+  -snes_ls cubic - Activates SNESLineSearchCubic()
.   -snes_ls_alpha <alpha> - Sets alpha
.   -snes_ls_maxstep <maxstep> - Sets the maximum stepsize the line search will use (if the 2-norm(y) > maxstep then scale y to be y = (maxstep/2-norm(y)) *y)
-   -snes_ls_minlambda <minlambda> - Sets the minimum lambda the line search will use minlambda/ max_i ( y[i]/x[i] )

    
   Notes:
   This line search is taken from "Numerical Methods for Unconstrained 
   Optimization and Nonlinear Equations" by Dennis and Schnabel, page 325.

   Level: advanced

.keywords: SNES, nonlinear, line search, cubic

.seealso: SNESLineSearchQuadratic(), SNESLineSearchNo(), SNESLineSearchSet(), SNESLineSearchNoNorms()
@*/
PetscErrorCode  SNESLineSearchCubic_LS(SNES snes,void *lsctx,Vec x,Vec f,Vec y,PetscReal fnorm,PetscReal xnorm,Vec g,Vec w,PetscReal *ynorm,PetscReal *gnorm,PetscBool  *flag)
{
  /* 
     Note that for line search purposes we work with with the related
     minimization problem:
        min  z(x):  R^n -> R,
     where z(x) = .5 * fnorm*fnorm, and fnorm = || f ||_2.
   */
        
  PetscReal      initslope,lambdaprev,gnormprev,a,b,d,t1,t2,rellength;
  PetscReal      minlambda,lambda,lambdatemp;
#if defined(PETSC_USE_COMPLEX)
  PetscScalar    cinitslope;
#endif
  PetscErrorCode ierr;
  PetscInt       count;
  PetscBool      changed_w = PETSC_FALSE,changed_y = PETSC_FALSE;
  MPI_Comm       comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)snes,&comm);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(SNES_LineSearch,snes,x,f,g);CHKERRQ(ierr);
  *flag   = PETSC_TRUE;

  ierr = VecNorm(y,NORM_2,ynorm);CHKERRQ(ierr);
  if (*ynorm == 0.0) {
    if (snes->ls_monitor) {
      ierr = PetscViewerASCIIAddTab(snes->ls_monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(snes->ls_monitor,"    Line search: Initial direction and size is 0\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIISubtractTab(snes->ls_monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
    }
    *gnorm = fnorm;
    ierr   = VecCopy(x,w);CHKERRQ(ierr);
    ierr   = VecCopy(f,g);CHKERRQ(ierr);
    *flag  = PETSC_FALSE;
    goto theend1;
  }
  if (*ynorm > snes->maxstep) {	/* Step too big, so scale back */
    if (snes->ls_monitor) {
      ierr = PetscViewerASCIIAddTab(snes->ls_monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(snes->ls_monitor,"    Line search: Scaling step by %14.12e old ynorm %14.12e\n",(double)(snes->maxstep/(*ynorm)),(double)(*ynorm));CHKERRQ(ierr);
      ierr = PetscViewerASCIISubtractTab(snes->ls_monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
    }
    ierr = VecScale(y,snes->maxstep/(*ynorm));CHKERRQ(ierr);
    *ynorm = snes->maxstep;
  }
  ierr      = VecMaxPointwiseDivide(y,x,&rellength);CHKERRQ(ierr);
  minlambda = snes->steptol/rellength;
  ierr      = MatMult(snes->jacobian,y,w);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr      = VecDot(f,w,&cinitslope);CHKERRQ(ierr);
  initslope = PetscRealPart(cinitslope);
#else
  ierr      = VecDot(f,w,&initslope);CHKERRQ(ierr);
#endif
  if (initslope > 0.0)  initslope = -initslope;
  if (initslope == 0.0) initslope = -1.0;

  ierr = VecWAXPY(w,-1.0,y,x);CHKERRQ(ierr);
  if (snes->nfuncs >= snes->max_funcs) {
    ierr  = PetscInfo(snes,"Exceeded maximum function evaluations, while checking full step length!\n");CHKERRQ(ierr);
    *flag = PETSC_FALSE;
    snes->reason = SNES_DIVERGED_FUNCTION_COUNT;
    goto theend1;
  }
  ierr = SNESComputeFunction(snes,w,g);CHKERRQ(ierr);
  if (snes->domainerror) {
    ierr = PetscLogEventEnd(SNES_LineSearch,snes,x,f,g);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = VecNorm(g,NORM_2,gnorm);CHKERRQ(ierr);
  if (PetscIsInfOrNanReal(*gnorm)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FP,"User provided compute function generated a Not-a-Number");
  ierr = PetscInfo2(snes,"Initial fnorm %14.12e gnorm %14.12e\n",(double)fnorm,(double)*gnorm);CHKERRQ(ierr);
  if (.5*(*gnorm)*(*gnorm) <= .5*fnorm*fnorm + snes->ls_alpha*initslope) { /* Sufficient reduction */
    if (snes->ls_monitor) {
      ierr = PetscViewerASCIIAddTab(snes->ls_monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(snes->ls_monitor,"    Line search: Using full step: fnorm %14.12e gnorm %14.12e\n",(double)fnorm,(double)*gnorm);CHKERRQ(ierr);
      ierr = PetscViewerASCIISubtractTab(snes->ls_monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
    }
    goto theend1;
  }

  /* Fit points with quadratic */
  lambda     = 1.0;
  lambdatemp = -initslope/((*gnorm)*(*gnorm) - fnorm*fnorm - 2.0*initslope);
  lambdaprev = lambda;
  gnormprev  = *gnorm;
  if (lambdatemp > .5*lambda)  lambdatemp = .5*lambda;
  if (lambdatemp <= .1*lambda) lambda = .1*lambda; 
  else                         lambda = lambdatemp;

  ierr  = VecWAXPY(w,-lambda,y,x);CHKERRQ(ierr);
  if (snes->nfuncs >= snes->max_funcs) {
    ierr  = PetscInfo1(snes,"Exceeded maximum function evaluations, while attempting quadratic backtracking! %D \n",snes->nfuncs);CHKERRQ(ierr);
    *flag = PETSC_FALSE;
    snes->reason = SNES_DIVERGED_FUNCTION_COUNT;
    goto theend1;
  }
  ierr = SNESComputeFunction(snes,w,g);CHKERRQ(ierr);
  if (snes->domainerror) {
    ierr = PetscLogEventEnd(SNES_LineSearch,snes,x,f,g);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = VecNorm(g,NORM_2,gnorm);CHKERRQ(ierr);
  if (PetscIsInfOrNanReal(*gnorm)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FP,"User provided compute function generated a Not-a-Number");
  if (snes->ls_monitor) {
    ierr = PetscViewerASCIIAddTab(snes->ls_monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(snes->ls_monitor,"    Line search: gnorm after quadratic fit %14.12e\n",(double)(*gnorm));CHKERRQ(ierr);
    ierr = PetscViewerASCIISubtractTab(snes->ls_monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
  }
  if (.5*(*gnorm)*(*gnorm) < .5*fnorm*fnorm + lambda*snes->ls_alpha*initslope) { /* sufficient reduction */
    if (snes->ls_monitor) {
      ierr = PetscViewerASCIIAddTab(snes->ls_monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(snes->ls_monitor,"    Line search: Quadratically determined step, lambda=%18.16e\n",(double)lambda);CHKERRQ(ierr);
      ierr = PetscViewerASCIISubtractTab(snes->ls_monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
    }
    goto theend1;
  }

  /* Fit points with cubic */
  count = 1;
  while (PETSC_TRUE) {
    if (lambda <= minlambda) { 
      if (snes->ls_monitor) {
        ierr = PetscViewerASCIIAddTab(snes->ls_monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
	ierr = PetscViewerASCIIPrintf(snes->ls_monitor,"    Line search: unable to find good step length! After %D tries \n",count);CHKERRQ(ierr);
	ierr = PetscViewerASCIIPrintf(snes->ls_monitor,"    Line search: fnorm=%18.16e, gnorm=%18.16e, ynorm=%18.16e, minlambda=%18.16e, lambda=%18.16e, initial slope=%18.16e\n",(double)fnorm,(double)(*gnorm),(double)(*ynorm),(double)minlambda,(double)lambda,(double)initslope);CHKERRQ(ierr);
        ierr = PetscViewerASCIISubtractTab(snes->ls_monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
      }
      *flag = PETSC_FALSE; 
      break;
    }
    t1 = .5*((*gnorm)*(*gnorm) - fnorm*fnorm) - lambda*initslope;
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
    lambdaprev = lambda;
    gnormprev  = *gnorm;
    if (lambdatemp > .5*lambda)  lambdatemp = .5*lambda;
    if (lambdatemp <= .1*lambda) lambda     = .1*lambda;
    else                         lambda     = lambdatemp;
    ierr  = VecWAXPY(w,-lambda,y,x);CHKERRQ(ierr);
    if (snes->nfuncs >= snes->max_funcs) {
      ierr = PetscInfo1(snes,"Exceeded maximum function evaluations, while looking for good step length! %D \n",count);CHKERRQ(ierr);
      ierr = PetscInfo5(snes,"fnorm=%18.16e, gnorm=%18.16e, ynorm=%18.16e, lambda=%18.16e, initial slope=%18.16e\n",(double)fnorm,(double)(*gnorm),(double)(*ynorm),(double)lambda,(double)initslope);CHKERRQ(ierr);
      *flag = PETSC_FALSE;
      snes->reason = SNES_DIVERGED_FUNCTION_COUNT;
      break;
    }
    ierr = SNESComputeFunction(snes,w,g);CHKERRQ(ierr);
    if (snes->domainerror) {
      ierr = PetscLogEventEnd(SNES_LineSearch,snes,x,f,g);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
    ierr = VecNorm(g,NORM_2,gnorm);CHKERRQ(ierr);
    if (PetscIsInfOrNanReal(*gnorm)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FP,"User provided compute function generated a Not-a-Number");
    if (.5*(*gnorm)*(*gnorm) < .5*fnorm*fnorm + lambda*snes->ls_alpha*initslope) { /* is reduction enough? */
      if (snes->ls_monitor) {
	ierr = PetscPrintf(comm,"    Line search: Cubically determined step, current gnorm %14.12e lambda=%18.16e\n",(double)(*gnorm),(double)lambda);CHKERRQ(ierr);
      }
      break;
    } else {
      if (snes->ls_monitor) {
        ierr = PetscPrintf(comm,"    Line search: Cubic step no good, shrinking lambda, current gnorm %12.12e lambda=%18.16e\n",(double)(*gnorm),(double)lambda);CHKERRQ(ierr);
      }
    }
    count++;
  }
  theend1:
  /* Optional user-defined check for line search step validity */
  if (snes->ops->postcheckstep && *flag) {
    ierr = (*snes->ops->postcheckstep)(snes,x,y,w,snes->postcheck,&changed_y,&changed_w);CHKERRQ(ierr);
    if (changed_y) {
      ierr = VecWAXPY(w,-1.0,y,x);CHKERRQ(ierr);
    }
    if (changed_y || changed_w) { /* recompute the function if the step has changed */
      ierr = SNESComputeFunction(snes,w,g);CHKERRQ(ierr);
      if (snes->domainerror) {
        ierr = PetscLogEventEnd(SNES_LineSearch,snes,x,f,g);CHKERRQ(ierr);
        PetscFunctionReturn(0);
      }
      ierr = VecNormBegin(g,NORM_2,gnorm);CHKERRQ(ierr);
      if (PetscIsInfOrNanReal(*gnorm)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FP,"User provided compute function generated a Not-a-Number");
      ierr = VecNormBegin(y,NORM_2,ynorm);CHKERRQ(ierr);
      ierr = VecNormEnd(g,NORM_2,gnorm);CHKERRQ(ierr);
      ierr = VecNormEnd(y,NORM_2,ynorm);CHKERRQ(ierr);
    }
  }
  ierr = PetscLogEventEnd(SNES_LineSearch,snes,x,f,g);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */

#undef __FUNCT__  
#define __FUNCT__ "SNESLineSearchQuadratic_LS"
/*@C
   SNESLineSearchQuadratic_LS - Performs a quadratic line search.

   Collective on SNES and Vec

   Input Parameters:
+  snes - the SNES context
.  lsctx - optional context for line search (not used here)
.  x - current iterate
.  f - residual evaluated at x
.  y - search direction 
.  fnorm - 2-norm of f
-  xnorm - norm of x if known, otherwise 0

   Output Parameters:
+  g - residual evaluated at new iterate w
.  w - new iterate (x + lambda*y)
.  gnorm - 2-norm of g
.  ynorm - 2-norm of search length
-  flag - PETSC_TRUE if line search succeeds; PETSC_FALSE on failure.

   Options Database Keys:
+  -snes_ls quadratic - Activates SNESLineSearchQuadratic()
.   -snes_ls_alpha <alpha> - Sets alpha
.   -snes_ls_maxstep <maxstep> - Sets the maximum stepsize the line search will use (if the 2-norm(y) > maxstep then scale y to be y = (maxstep/2-norm(y)) *y)
-   -snes_ls_minlambda <minlambda> - Sets the minimum lambda the line search will use minlambda/ max_i ( y[i]/x[i] )

   Notes:
   Use SNESLineSearchSet() to set this routine within the SNESLS method.  

   Level: advanced

.keywords: SNES, nonlinear, quadratic, line search

.seealso: SNESLineSearchCubic(), SNESLineSearchNo(), SNESLineSearchSet(), SNESLineSearchNoNorms()
@*/
PetscErrorCode  SNESLineSearchQuadratic_LS(SNES snes,void *lsctx,Vec x,Vec f,Vec y,PetscReal fnorm,PetscReal xnorm,Vec g,Vec w,PetscReal *ynorm,PetscReal *gnorm,PetscBool  *flag)
{
  /* 
     Note that for line search purposes we work with with the related
     minimization problem:
        min  z(x):  R^n -> R,
     where z(x) = .5 * fnorm*fnorm,and fnorm = || f ||_2.
   */
  PetscReal      initslope,minlambda,lambda,lambdatemp,rellength;
#if defined(PETSC_USE_COMPLEX)
  PetscScalar    cinitslope;
#endif
  PetscErrorCode ierr;
  PetscInt       count;
  PetscBool      changed_w = PETSC_FALSE,changed_y = PETSC_FALSE;

  PetscFunctionBegin;
  ierr    = PetscLogEventBegin(SNES_LineSearch,snes,x,f,g);CHKERRQ(ierr);
  *flag   = PETSC_TRUE;

  ierr = VecNorm(y,NORM_2,ynorm);CHKERRQ(ierr);
  if (*ynorm == 0.0) {
    if (snes->ls_monitor) {
      ierr = PetscViewerASCIIAddTab(snes->ls_monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(snes->ls_monitor,"Line search: Direction and size is 0\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIISubtractTab(snes->ls_monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
    }
    *gnorm = fnorm;
    ierr   = VecCopy(x,w);CHKERRQ(ierr);
    ierr   = VecCopy(f,g);CHKERRQ(ierr);
    *flag  = PETSC_FALSE;
    goto theend2;
  }
  if (*ynorm > snes->maxstep) {	/* Step too big, so scale back */
    ierr   = VecScale(y,snes->maxstep/(*ynorm));CHKERRQ(ierr);
    *ynorm = snes->maxstep;
  }
  ierr      = VecMaxPointwiseDivide(y,x,&rellength);CHKERRQ(ierr);
  minlambda = snes->steptol/rellength;
  ierr = MatMult(snes->jacobian,y,w);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr      = VecDot(f,w,&cinitslope);CHKERRQ(ierr);
  initslope = PetscRealPart(cinitslope);
#else
  ierr = VecDot(f,w,&initslope);CHKERRQ(ierr);
#endif
  if (initslope > 0.0)  initslope = -initslope;
  if (initslope == 0.0) initslope = -1.0;
  ierr = PetscInfo1(snes,"Initslope %14.12e \n",(double)initslope);CHKERRQ(ierr);

  ierr = VecWAXPY(w,-1.0,y,x);CHKERRQ(ierr);
  if (snes->nfuncs >= snes->max_funcs) {
    ierr  = PetscInfo(snes,"Exceeded maximum function evaluations, while checking full step length!\n");CHKERRQ(ierr);
    *flag = PETSC_FALSE;
    snes->reason = SNES_DIVERGED_FUNCTION_COUNT;
    goto theend2;
  }
  ierr = SNESComputeFunction(snes,w,g);CHKERRQ(ierr);
  if (snes->domainerror) {
    ierr = PetscLogEventEnd(SNES_LineSearch,snes,x,f,g);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = VecNorm(g,NORM_2,gnorm);CHKERRQ(ierr);
  if (PetscIsInfOrNanReal(*gnorm)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FP,"User provided compute function generated a Not-a-Number");
  if (.5*(*gnorm)*(*gnorm) <= .5*fnorm*fnorm + snes->ls_alpha*initslope) { /* Sufficient reduction */
    if (snes->ls_monitor) {
      ierr = PetscViewerASCIIAddTab(snes->ls_monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(snes->ls_monitor,"    Line Search: Using full step\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIISubtractTab(snes->ls_monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
    }
    goto theend2;
  }

  /* Fit points with quadratic */
  lambda = 1.0;
  count = 1;
  while (PETSC_TRUE) {
    if (lambda <= minlambda) { /* bad luck; use full step */
      if (snes->ls_monitor) {
        ierr = PetscViewerASCIIAddTab(snes->ls_monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(snes->ls_monitor,"Line search: Unable to find good step length! %D \n",count);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(snes->ls_monitor,"Line search: fnorm=%14.12e, gnorm=%14.12e, ynorm=%14.12e, lambda=%14.12e, initial slope=%14.12e\n",(double)fnorm,(double)(*gnorm),(double)(*ynorm),(double)lambda,(double)initslope);CHKERRQ(ierr);
        ierr = PetscViewerASCIISubtractTab(snes->ls_monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
      }
      ierr = VecCopy(x,w);CHKERRQ(ierr);
      *flag = PETSC_FALSE;
      break;
    }
    lambdatemp = -initslope/((*gnorm)*(*gnorm) - fnorm*fnorm - 2.0*initslope);
    if (lambdatemp > .5*lambda)  lambdatemp = .5*lambda;
    if (lambdatemp <= .1*lambda) lambda     = .1*lambda; 
    else                         lambda     = lambdatemp;
    
    ierr = VecWAXPY(w,-lambda,y,x);CHKERRQ(ierr);
    if (snes->nfuncs >= snes->max_funcs) {
      ierr  = PetscInfo1(snes,"Exceeded maximum function evaluations, while looking for good step length! %D \n",count);CHKERRQ(ierr);
      ierr  = PetscInfo5(snes,"fnorm=%18.16e, gnorm=%18.16e, ynorm=%18.16e, lambda=%18.16e, initial slope=%18.16e\n",(double)fnorm,(double)*gnorm,(double)*ynorm,(double)lambda,(double)initslope);CHKERRQ(ierr);
      *flag = PETSC_FALSE;
      snes->reason = SNES_DIVERGED_FUNCTION_COUNT;
      break;
    }
    ierr = SNESComputeFunction(snes,w,g);CHKERRQ(ierr);
    if (snes->domainerror) {
      ierr = PetscLogEventEnd(SNES_LineSearch,snes,x,f,g);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
    ierr = VecNorm(g,NORM_2,gnorm);CHKERRQ(ierr);
    if (PetscIsInfOrNanReal(*gnorm)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FP,"User provided compute function generated a Not-a-Number");
    if (.5*(*gnorm)*(*gnorm) < .5*fnorm*fnorm + lambda*snes->ls_alpha*initslope) { /* sufficient reduction */
      if (snes->ls_monitor) {
        ierr = PetscViewerASCIIAddTab(snes->ls_monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(snes->ls_monitor,"    Line Search: Quadratically determined step, lambda=%14.12e\n",(double)lambda);CHKERRQ(ierr);
        ierr = PetscViewerASCIISubtractTab(snes->ls_monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
      }
      break;
    }
    count++;
  }
  theend2:
  /* Optional user-defined check for line search step validity */
  if (snes->ops->postcheckstep) {
    ierr = (*snes->ops->postcheckstep)(snes,x,y,w,snes->postcheck,&changed_y,&changed_w);CHKERRQ(ierr);
    if (changed_y) {
      ierr = VecWAXPY(w,-1.0,y,x);CHKERRQ(ierr);
    }
    if (changed_y || changed_w) { /* recompute the function if the step has changed */
      ierr = SNESComputeFunction(snes,w,g);
      if (snes->domainerror) {
        ierr = PetscLogEventEnd(SNES_LineSearch,snes,x,f,g);CHKERRQ(ierr);
        PetscFunctionReturn(0);
      }
      ierr = VecNormBegin(g,NORM_2,gnorm);CHKERRQ(ierr);
      ierr = VecNormBegin(y,NORM_2,ynorm);CHKERRQ(ierr);
      ierr = VecNormEnd(g,NORM_2,gnorm);CHKERRQ(ierr);
      ierr = VecNormEnd(y,NORM_2,ynorm);CHKERRQ(ierr);
      if (PetscIsInfOrNanReal(*gnorm)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FP,"User provided compute function generated a Not-a-Number");
    }
  }
  ierr = PetscLogEventEnd(SNES_LineSearch,snes,x,f,g);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchSetType_LS"
PetscErrorCode  SNESLineSearchSetType_LS(SNES snes, SNESLineSearchType type)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  switch (type) {
  case SNES_LS_BASIC:
    ierr = SNESLineSearchSet(snes,SNESLineSearchNo,PETSC_NULL);CHKERRQ(ierr);
    break;
  case SNES_LS_BASIC_NONORMS:
    ierr = SNESLineSearchSet(snes,SNESLineSearchNoNorms,PETSC_NULL);CHKERRQ(ierr);
    break;
  case SNES_LS_QUADRATIC:
    ierr = SNESLineSearchSet(snes,SNESLineSearchQuadratic_LS,PETSC_NULL);CHKERRQ(ierr);
    break;
  case SNES_LS_CUBIC:
    ierr = SNESLineSearchSet(snes,SNESLineSearchCubic_LS,PETSC_NULL);CHKERRQ(ierr);
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP,"Unknown line search type.");
    break;
  }
  snes->ls_type = type;
  PetscFunctionReturn(0);
}
EXTERN_C_END


/*  -------------------------------------------------------------------- 

     This file implements a truncated Newton method with a line search,
     for solving a system of nonlinear equations, using the KSP, Vec, 
     and Mat interfaces for linear solvers, vectors, and matrices, 
     respectively.

     The following basic routines are required for each nonlinear solver:
          SNESCreate_XXX()          - Creates a nonlinear solver context
          SNESSetFromOptions_XXX()  - Sets runtime options
          SNESSolve_XXX()           - Solves the nonlinear system
          SNESDestroy_XXX()         - Destroys the nonlinear solver context
     The suffix "_XXX" denotes a particular implementation, in this case
     we use _LS (e.g., SNESCreate_LS, SNESSolve_LS) for solving
     systems of nonlinear equations with a line search (LS) method.
     These routines are actually called via the common user interface
     routines SNESCreate(), SNESSetFromOptions(), SNESSolve(), and 
     SNESDestroy(), so the application code interface remains identical 
     for all nonlinear solvers.

     Another key routine is:
          SNESSetUp_XXX()           - Prepares for the use of a nonlinear solver
     by setting data structures and options.   The interface routine SNESSetUp()
     is not usually called directly by the user, but instead is called by
     SNESSolve() if necessary.

     Additional basic routines are:
          SNESView_XXX()            - Prints details of runtime options that
                                      have actually been used.
     These are called by application codes via the interface routines
     SNESView().

     The various types of solvers (preconditioners, Krylov subspace methods,
     nonlinear solvers, timesteppers) are all organized similarly, so the
     above description applies to these categories also.  

    -------------------------------------------------------------------- */
/*
   SNESSolve_LS - Solves a nonlinear system with a truncated Newton
   method with a line search.

   Input Parameters:
.  snes - the SNES context

   Output Parameter:
.  outits - number of iterations until termination

   Application Interface Routine: SNESSolve()

   Notes:
   This implements essentially a truncated Newton method with a
   line search.  By default a cubic backtracking line search 
   is employed, as described in the text "Numerical Methods for
   Unconstrained Optimization and Nonlinear Equations" by Dennis 
   and Schnabel.
*/
#undef __FUNCT__  
#define __FUNCT__ "SNESSolve_LS"
PetscErrorCode SNESSolve_LS(SNES snes)
{ 
  PetscErrorCode     ierr;
  PetscInt           maxits,i,lits;
  PetscBool          lssucceed;
  MatStructure       flg = DIFFERENT_NONZERO_PATTERN;
  PetscReal          fnorm,gnorm,xnorm=0,ynorm;
  Vec                Y,X,F,G,W;
  KSPConvergedReason kspreason;

  PetscFunctionBegin;
  snes->numFailures            = 0;
  snes->numLinearSolveFailures = 0;
  snes->reason                 = SNES_CONVERGED_ITERATING;

  maxits	= snes->max_its;	/* maximum number of iterations */
  X		= snes->vec_sol;	/* solution vector */
  F		= snes->vec_func;	/* residual vector */
  Y		= snes->work[0];	/* work vectors */
  G		= snes->work[1];
  W		= snes->work[2];

  ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
  snes->iter = 0;
  snes->norm = 0.0;
  ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
  ierr = SNESComputeFunction(snes,X,F);CHKERRQ(ierr);
  if (snes->domainerror) {
    snes->reason = SNES_DIVERGED_FUNCTION_DOMAIN;
    PetscFunctionReturn(0);
  }
  ierr = VecNormBegin(F,NORM_2,&fnorm);CHKERRQ(ierr);	/* fnorm <- ||F||  */
  ierr = VecNormBegin(X,NORM_2,&xnorm);CHKERRQ(ierr);	/* xnorm <- ||x||  */
  ierr = VecNormEnd(F,NORM_2,&fnorm);CHKERRQ(ierr);
  ierr = VecNormEnd(X,NORM_2,&xnorm);CHKERRQ(ierr);
  if (PetscIsInfOrNanReal(fnorm)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FP,"User provided compute function generated a Not-a-Number");
  ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
  snes->norm = fnorm;
  ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
  SNESLogConvHistory(snes,fnorm,0);
  ierr = SNESMonitor(snes,0,fnorm);CHKERRQ(ierr);

  /* set parameter for default relative tolerance convergence test */
  snes->ttol = fnorm*snes->rtol;
  /* test convergence */
  ierr = (*snes->ops->converged)(snes,0,0.0,0.0,fnorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
  if (snes->reason) PetscFunctionReturn(0);

  for (i=0; i<maxits; i++) {

    /* Call general purpose update function */
    if (snes->ops->update) {
      ierr = (*snes->ops->update)(snes, snes->iter);CHKERRQ(ierr);
    }

    /* Solve J Y = F, where J is Jacobian matrix */
    ierr = SNESComputeJacobian(snes,X,&snes->jacobian,&snes->jacobian_pre,&flg);CHKERRQ(ierr);
    ierr = KSPSetOperators(snes->ksp,snes->jacobian,snes->jacobian_pre,flg);CHKERRQ(ierr);
    ierr = SNES_KSPSolve(snes,snes->ksp,F,Y);CHKERRQ(ierr);
    ierr = KSPGetConvergedReason(snes->ksp,&kspreason);CHKERRQ(ierr);
    if (kspreason < 0) {
      if (++snes->numLinearSolveFailures >= snes->maxLinearSolveFailures) {
        ierr = PetscInfo2(snes,"iter=%D, number linear solve failures %D greater than current SNES allowed, stopping solve\n",snes->iter,snes->numLinearSolveFailures);CHKERRQ(ierr);
        snes->reason = SNES_DIVERGED_LINEAR_SOLVE;
        break;
      }
    }
    ierr = KSPGetIterationNumber(snes->ksp,&lits);CHKERRQ(ierr);
    snes->linear_its += lits;
    ierr = PetscInfo2(snes,"iter=%D, linear solve iterations=%D\n",snes->iter,lits);CHKERRQ(ierr);

    if (snes->ops->precheckstep) {
      PetscBool  changed_y = PETSC_FALSE;
      ierr = (*snes->ops->precheckstep)(snes,X,Y,snes->precheck,&changed_y);CHKERRQ(ierr);
    }

    if (PetscLogPrintInfo){
      ierr = SNESLSCheckResidual_Private(snes,snes->jacobian,F,Y,G,W);CHKERRQ(ierr);
    }

    /* Compute a (scaled) negative update in the line search routine: 
         Y <- X - lambda*Y 
       and evaluate G = function(Y) (depends on the line search). 
    */
    ierr = VecCopy(Y,snes->vec_sol_update);CHKERRQ(ierr);
    ynorm = 1; gnorm = fnorm;
    ierr = (*snes->ops->linesearch)(snes,snes->lsP,X,F,Y,fnorm,xnorm,G,W,&ynorm,&gnorm,&lssucceed);CHKERRQ(ierr);
    ierr = PetscInfo4(snes,"fnorm=%18.16e, gnorm=%18.16e, ynorm=%18.16e, lssucceed=%d\n",(double)fnorm,(double)gnorm,(double)ynorm,(int)lssucceed);CHKERRQ(ierr);
    if (snes->reason == SNES_DIVERGED_FUNCTION_COUNT) break;
    if (snes->domainerror) {
      snes->reason = SNES_DIVERGED_FUNCTION_DOMAIN;
      PetscFunctionReturn(0);
    }
    if (!lssucceed) {
      if (++snes->numFailures >= snes->maxFailures) {
	PetscBool  ismin;
        snes->reason = SNES_DIVERGED_LINE_SEARCH;
        ierr = SNESLSCheckLocalMin_Private(snes,snes->jacobian,G,W,gnorm,&ismin);CHKERRQ(ierr);
        if (ismin) snes->reason = SNES_DIVERGED_LOCAL_MIN;
        break;
      }
    }
    /* Update function and solution vectors */
    fnorm = gnorm;
    ierr = VecCopy(G,F);CHKERRQ(ierr);
    ierr = VecCopy(W,X);CHKERRQ(ierr);
    /* Monitor convergence */
    ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
    snes->iter = i+1;
    snes->norm = fnorm;
    ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
    SNESLogConvHistory(snes,snes->norm,lits);
    ierr = SNESMonitor(snes,snes->iter,snes->norm);CHKERRQ(ierr);
    /* Test for convergence, xnorm = || X || */
    if (snes->ops->converged != SNESSkipConverged) { ierr = VecNorm(X,NORM_2,&xnorm);CHKERRQ(ierr); }
    ierr = (*snes->ops->converged)(snes,snes->iter,xnorm,ynorm,fnorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
    if (snes->reason) break;
  }
  if (i == maxits) {
    ierr = PetscInfo1(snes,"Maximum number of iterations has been reached: %D\n",maxits);CHKERRQ(ierr);
    if(!snes->reason) snes->reason = SNES_DIVERGED_MAX_IT;
  }
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
/*
   SNESSetUp_LS - Sets up the internal data structures for the later use
   of the SNESLS nonlinear solver.

   Input Parameter:
.  snes - the SNES context
.  x - the solution vector

   Application Interface Routine: SNESSetUp()

   Notes:
   For basic use of the SNES solvers, the user need not explicitly call
   SNESSetUp(), since these actions will automatically occur during
   the call to SNESSolve().
 */
#undef __FUNCT__  
#define __FUNCT__ "SNESSetUp_LS"
PetscErrorCode SNESSetUp_LS(SNES snes)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SNESDefaultGetWork(snes,3);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */

#undef __FUNCT__
#define __FUNCT__ "SNESReset_LS"
PetscErrorCode SNESReset_LS(SNES snes)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (snes->work) {ierr = VecDestroyVecs(snes->nwork,&snes->work);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/*
   SNESDestroy_LS - Destroys the private SNES_LS context that was created
   with SNESCreate_LS().

   Input Parameter:
.  snes - the SNES context

   Application Interface Routine: SNESDestroy()
 */
#undef __FUNCT__
#define __FUNCT__ "SNESDestroy_LS"
PetscErrorCode SNESDestroy_LS(SNES snes)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SNESReset_LS(snes);CHKERRQ(ierr);
  ierr = PetscFree(snes->data);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */

/*
   SNESView_LS - Prints info from the SNESLS data structure.

   Input Parameters:
.  SNES - the SNES context
.  viewer - visualization context

   Application Interface Routine: SNESView()
*/
#undef __FUNCT__  
#define __FUNCT__ "SNESView_LS"
static PetscErrorCode SNESView_LS(SNES snes,PetscViewer viewer)
{
  const char     *cstr;
  PetscErrorCode ierr;
  PetscBool      iascii;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    cstr = SNESLineSearchTypeName(snes->ls_type);
    ierr = PetscViewerASCIIPrintf(viewer,"  line search variant: %s\n",cstr);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  alpha=%14.12e, maxstep=%14.12e, minlambda=%14.12e\n",(double)snes->ls_alpha,(double)snes->maxstep,(double)snes->steptol);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  damping factor=%14.12e\n",(double)snes->damping);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   SNESSetFromOptions_LS - Sets various parameters for the SNESLS method.

   Input Parameter:
.  snes - the SNES context

   Application Interface Routine: SNESSetFromOptions()
*/
#undef __FUNCT__  
#define __FUNCT__ "SNESSetFromOptions_LS"
static PetscErrorCode SNESSetFromOptions_LS(SNES snes)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("SNES Line search options");CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*MC
      SNESLS - Newton based nonlinear solver that uses a line search

   Options Database:
+   -snes_ls [cubic,quadratic,basic,basicnonorms] - Selects line search
.   -snes_ls_alpha <alpha> - Sets alpha used in determining if reduction in function norm is sufficient
.   -snes_ls_maxstep <maxstep> - Sets the maximum stepsize the line search will use (if the 2-norm(y) > maxstep then scale y to be y = (maxstep/2-norm(y)) *y)
.   -snes_ls_minlambda <minlambda>  - Sets the minimum lambda the line search will use  minlambda / max_i ( y[i]/x[i] )
.   -snes_ls_monitor - print information about progress of line searches 
-   -snes_ls_damping - damping factor used if -snes_ls is basic or basicnonorms


    Notes: This is the default nonlinear solver in SNES

   Level: beginner

.seealso:  SNESCreate(), SNES, SNESSetType(), SNESTR, SNESLineSearchSet(), 
           SNESLineSearchSetPostCheck(), SNESLineSearchNo(), SNESLineSearchCubic(), SNESLineSearchQuadratic(), 
          SNESLineSearchSet(), SNESLineSearchNoNorms(), SNESLineSearchSetPreCheck(), SNESLineSearchSetParams(), SNESLineSearchGetParams()

M*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "SNESCreate_LS"
PetscErrorCode  SNESCreate_LS(SNES snes)
{
  PetscErrorCode ierr;
  SNES_LS        *neP;

  PetscFunctionBegin;
  snes->ops->setup           = SNESSetUp_LS;
  snes->ops->solve           = SNESSolve_LS;
  snes->ops->destroy         = SNESDestroy_LS;
  snes->ops->setfromoptions  = SNESSetFromOptions_LS;
  snes->ops->view            = SNESView_LS;
  snes->ops->reset           = SNESReset_LS;

  snes->usesksp                      = PETSC_TRUE;
  snes->usespc                       = PETSC_FALSE;
  ierr                               = PetscNewLog(snes,SNES_LS,&neP);CHKERRQ(ierr);
  snes->data                         = (void*)neP;
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)snes,"SNESLineSearchSetType_C","SNESLineSearchSetType_LS",SNESLineSearchSetType_LS);CHKERRQ(ierr);
  ierr = SNESLineSearchSetType(snes, SNES_LS_CUBIC);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
