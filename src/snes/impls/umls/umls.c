/*$Id: umls.c,v 1.98 2000/06/09 14:53:32 bsmith Exp bsmith $*/

#include "src/snes/impls/umls/umls.h"             /*I "petscsnes.h" I*/

EXTERN int SNESStep(SNES,double*,double*,double*,double*,
                    double*,double*,double*,double*,double*);

/*
   Implements Newton's Method with a line search approach
   for solving unconstrained minimization problems.

   Note:
   The line search algorithm is taken from More and Thuente,
   "Line search algorithms with guaranteed sufficient decrease",
   Argonne National Laboratory, Technical Report MCS-P330-1092.
*/

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"SNESSolve_UM_LS"
static int SNESSolve_UM_LS(SNES snes,int *outits)
{
  SNES_UM_LS          *neP = (SNES_UM_LS*)snes->data;
  int                 maxits,success,iters,i,global_dim,ierr,kspmaxit;
  double              snorm,*f,*gnorm,two = 2.0,tnorm;
  Scalar              neg_one = -1.0;
  Vec                 G,X,RHS,S,W;
  SLES                sles;
  KSP                 ksp;
  MatStructure        flg = DIFFERENT_NONZERO_PATTERN;
  SNESConvergedReason reason;
  KSPConvergedReason  kreason;

  PetscFunctionBegin;
  snes->reason  = SNES_CONVERGED_ITERATING;

  maxits	= snes->max_its;        /* maximum number of iterations */
  X		= snes->vec_sol; 	/* solution vector */
  G		= snes->vec_func;	/* gradient vector */
  RHS		= snes->work[0]; 	/* work vectors */
  S		= snes->work[1];	/* step vector */
  W		= snes->work[2];	/* work vector */
  f		= &(snes->fc);		/* function to minimize */
  gnorm		= &(snes->norm);	/* gradient norm */

  ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
  snes->iter = 0;
  ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
  ierr = SNESComputeMinimizationFunction(snes,X,f);CHKERRQ(ierr); /* f(X) */
  ierr = SNESComputeGradient(snes,X,G);CHKERRQ(ierr);     /* G(X) <- gradient */

  ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
  ierr = VecNorm(G,NORM_2,gnorm);CHKERRQ(ierr);         /* gnorm = || G || */
  ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
  SNESLogConvHistory(snes,*gnorm,0);
  SNESMonitor(snes,0,*gnorm);

  ierr = SNESGetSLES(snes,&sles);CHKERRQ(ierr);
  ierr = SLESGetKSP(sles,&ksp);CHKERRQ(ierr);
  ierr = VecGetSize(X,&global_dim);CHKERRQ(ierr);
  kspmaxit = neP->max_kspiter_factor * ((int)sqrt((double)global_dim));
  ierr = KSPSetTolerances(ksp,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,kspmaxit);CHKERRQ(ierr);

  for (i=0; i<maxits; i++) {
    ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
    snes->iter = i+1;
    ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
    neP->gamma = neP->gamma_factor*(*gnorm);
    success = 0;
    ierr = VecCopy(G,RHS);CHKERRQ(ierr);
    ierr = VecScale(&neg_one,RHS);CHKERRQ(ierr);
    ierr = SNESComputeHessian(snes,X,&snes->jacobian,&snes->jacobian_pre,&flg);CHKERRQ(ierr);
    ierr = SLESSetOperators(snes->sles,snes->jacobian,snes->jacobian_pre,flg);CHKERRQ(ierr);
    while (!success) {
      ierr = SLESSolve(snes->sles,RHS,S,&iters);CHKERRQ(ierr);
      snes->linear_its += iters;
      ierr = KSPGetConvergedReason(ksp,&kreason);CHKERRQ(ierr);
      if ((int)kreason < 0 || (iters >= kspmaxit)) { /* Modify diagonal of Hessian */
        neP->gamma_factor *= two; 
        neP->gamma = neP->gamma_factor*(*gnorm); 
#if !defined(PETSC_USE_COMPLEX)
        PLogInfo(snes,"SNESSolve_UM_LS:  modify diagonal (assume same nonzero structure), gamma_factor=%g, gamma=%g\n",
                 neP->gamma_factor,neP->gamma);
#else
        PLogInfo(snes,"SNESSolve_UM_LS:  modify diagonal (asuume same nonzero structure), gamma_factor=%g, gamma=%g\n",
                 neP->gamma_factor,PetscRealPart(neP->gamma));
#endif
        ierr = MatShift(&neP->gamma,snes->jacobian);CHKERRQ(ierr);
        if ((snes->jacobian_pre != snes->jacobian) && (flg != SAME_PRECONDITIONER)){
          ierr = MatShift(&neP->gamma,snes->jacobian_pre);CHKERRQ(ierr);
        }
        /* We currently assume that all diagonal elements were allocated in
         original matrix, so that nonzero pattern is same ... should fix this */
        ierr = SLESSetOperators(snes->sles,snes->jacobian,snes->jacobian_pre,
               SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      } else {
        success = 1;
      }
    }   
    ierr = VecNorm(S,NORM_2,&snorm);CHKERRQ(ierr);

    /* Line search */
    neP->step = 1.0;
    ierr = (*neP->LineSearch)(snes,X,G,S,W,f,&(neP->step),&tnorm,&(neP->line));
    ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
    snes->norm = tnorm;
    ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
    if (neP->line != 1) snes->nfailures++;CHKERRQ(ierr);

    SNESLogConvHistory(snes,*gnorm,iters);
    SNESMonitor(snes,i+1,*gnorm);
    PLogInfo(snes,"SNESSolve_UM_LS: %d:  f=%g, gnorm=%g, snorm=%g, step=%g, KSP iters=%d\n",
             snes->iter,*f,*gnorm,snorm,neP->step,iters);

    /* Test for convergence */
    ierr = (*snes->converged)(snes,snorm,*gnorm,*f,&reason,snes->cnvP);CHKERRQ(ierr);
    if (reason) break;
    neP->gamma_factor /= two;
  }
  /* Verify solution is in correct location */
  if (X != snes->vec_sol) {
    ierr = VecCopy(X,snes->vec_sol);CHKERRQ(ierr);
    snes->vec_sol_always = snes->vec_sol;
    snes->vec_func_always = snes->vec_func;
  }
  if (i == maxits) {
    PLogInfo(snes,"SNESSolve_UM_LS: Maximum number of iterations reached: %d\n",maxits);
    i--;
    reason = SNES_DIVERGED_MAX_IT;
  }
  ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
  snes->reason = reason;
  ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
  *outits = i+1;
  PetscFunctionReturn(0);
}
/* ---------------------------------------------------------- */
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"SNESSetUp_UM_LS"
static int SNESSetUp_UM_LS(SNES snes)
{
  int ierr;

  PetscFunctionBegin;
  snes->nwork = 4;
  ierr = VecDuplicateVecs(snes->vec_sol,snes->nwork,&snes->work);CHKERRQ(ierr);
  PLogObjectParents(snes,snes->nwork,snes->work);
  snes->vec_sol_update_always = snes->work[3];
  PetscFunctionReturn(0);
}
/*------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"SNESDestroy_UM_LS"
static int SNESDestroy_UM_LS(SNES snes)
{
  int  ierr;

  PetscFunctionBegin;
  if (snes->nwork) {
    ierr =  VecDestroyVecs(snes->work,snes->nwork);CHKERRQ(ierr);
  }
  ierr = PetscFree(snes->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/*------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"SNESSetFromOptions_UM_LS"
static int SNESSetFromOptions_UM_LS(SNES snes)
{
  SNES_UM_LS *ctx = (SNES_UM_LS *)snes->data;
  int        ierr;

  PetscFunctionBegin;
  ierr = OptionsBegin(snes->comm,snes->prefix,"SNES trust region options for minimization");CHKERRQ(ierr);
    ierr = OptionsDouble("-snes_um_ls_gamma_factor","Damping parameter","None",ctx->gamma_factor,&ctx->gamma_factor,0);CHKERRQ(ierr);
    ierr = OptionsInt("-snes_um_ls_maxfev","Max function evaluation in line search","None",ctx->maxfev,&ctx->maxfev,0);CHKERRQ(ierr);
    ierr = OptionsDouble("-snes_um_ls_ftol","Tolerance for sufficient decrease","None",ctx->ftol,&ctx->ftol,0);CHKERRQ(ierr);
    ierr = OptionsDouble("-snes_um_ls_gtol","Tolerance for curvature condition","None",ctx->gtol,&ctx->gtol,0);CHKERRQ(ierr);
    ierr = OptionsDouble("-snes_um_ls_rtol","Relative tolerance for acceptable step","None",ctx->rtol,&ctx->rtol,0);CHKERRQ(ierr);
    ierr = OptionsDouble("-snes_um_ls_stepmin","Lower bound for step","None",ctx->stepmin,&ctx->stepmin,0);CHKERRQ(ierr);
    ierr = OptionsDouble("-snes_um_ls_stepmax","upper bound for step","None",ctx->stepmax,&ctx->stepmax,0);CHKERRQ(ierr);
  ierr = OptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"SNESView_UM_LS"
static int SNESView_UM_LS(SNES snes,Viewer viewer)
{
  SNES_UM_LS *ls = (SNES_UM_LS *)snes->data;
  int        ierr;
  PetscTruth isascii;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,ASCII_VIEWER,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = ViewerASCIIPrintf(viewer,"  gamma_f=%g, maxf=%d, maxkspf=%d, ftol=%g, rtol=%g, gtol=%g\n",
                      ls->gamma_factor,ls->maxfev,ls->max_kspiter_factor,ls->ftol,ls->rtol,ls->gtol);CHKERRQ(ierr);
  } else {
    SETERRQ1(1,1,"Viewer type %s not supported for SNES UM LS",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}
/* ---------------------------------------------------------- */
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"SNESConverged_UM_LS"
/*@C
   SNESConverged_UM_LS - Monitors the convergence of the SNESSolve_UM_LS()
   routine (default). 

   Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  xnorm - 2-norm of current iterate
.  gnorm - 2-norm of current gradient
.  f - objective function value
-  dummy - unused dummy context

   Output Parameter:
.   reason - one of
$   SNES_CONVERGED_FNORM_ABS         (F < F_minabs),
$   SNES_CONVERGED_GNORM_ABS         (grad F < grad),
$   SNES_DIVERGED_FUNCTION_COUNT     (nfunc > max_func),
$   SNES_DIVERGED_LS_FAILURE         (line search attempt failed)
$   SNES_DIVERGED_FNORM_NAN          (f = NaN),
$   SNES_CONVERGED_ITERATING         otherwise

   where
+  atol     - absolute gradient norm tolerance, set with SNESSetTolerances()
.  fmin     - lower bound on function value, set with SNESSetMinimizationFunctionTolerance()
.  max_func - maximum number of function evaluations, set with SNESSetTolerances()
-  nfunc    - number of function evaluations

   Level: intermediate

@*/
int SNESConverged_UM_LS(SNES snes,double xnorm,double gnorm,double f,SNESConvergedReason *reason,void *dummy)
{
  SNES_UM_LS *neP = (SNES_UM_LS*)snes->data;

  PetscFunctionBegin;

  if (f != f) {
    PLogInfo(snes,"SNESConverged_UM_LS:Failed to converged, function is NaN\n");
    *reason = SNES_DIVERGED_FNORM_NAN;
  } else if (f < snes->fmin) {
    PLogInfo(snes,"SNESConverged_UM_LS: Converged due to function value %g < minimum function value %g\n",
             f,snes->fmin);
    *reason = SNES_CONVERGED_FNORM_ABS ;
  } else if (gnorm < snes->atol) {
    PLogInfo(snes,"SNESConverged_UM_LS: Converged due to gradient norm %g < %g\n",gnorm,snes->atol);
    *reason = SNES_CONVERGED_GNORM_ABS;
  } else if (snes->nfuncs > snes->max_funcs) {
    PLogInfo(snes,"SNESConverged_UM_LS: Exceeded maximum number of function evaluations: %d > %d\n",
             snes->nfuncs,snes->max_funcs);
    *reason = SNES_DIVERGED_FUNCTION_COUNT;
  } else if (neP->line != 1) {
    PLogInfo(snes,"SNESConverged_UM_LS: Line search failed for above reason\n");
    *reason = SNES_DIVERGED_LS_FAILURE;
  } else {
    *reason = SNES_CONVERGED_ITERATING;
  }
  PetscFunctionReturn(0);
}
/* ---------------------------------------------------------- */
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"SNESMoreLineSearch"
/* @ SNESMoreLineSearch - This routine performs a line search algorithm,
     taken from More and Thuente, "Line search algorithms with 
     guaranteed sufficient decrease", Argonne National Laboratory", 
     Technical Report MCS-P330-1092.

   Input Parameters:
+  snes - SNES context
.  X - current iterate (on output X contains new iterate, X + step*S)
.  S - search direction
.  f - objective function evaluated at X
.  G - gradient evaluated at X
.  W - work vector
-  step - initial estimate of step length

   Output parameters:
+  f - objective function evaluated at new iterate, X + step*S
.  G - gradient evaluated at new iterate, X + step*S
.  X - new iterate
.  gnorm - 2-norm of G
-  step - final step length

   Info is set to one of:
.   1 - the line search succeeds; the sufficient decrease
   condition and the directional derivative condition hold

   negative number if an input parameter is invalid
.   -1 -  step < 0 
.   -2 -  ftol < 0 
.   -3 -  rtol < 0 
.   -4 -  gtol < 0 
.   -5 -  stepmin < 0 
.   -6 -  stepmax < stepmin
-   -7 -  maxfev < 0

   positive number > 1 if the line search otherwise terminates
+    2 -  Relative width of the interval of uncertainty is 
         at most rtol.
.    3 -  Maximum number of function evaluations (maxfev) has 
         been reached.
.    4 -  Step is at the lower bound, stepmin.
.    5 -  Step is at the upper bound, stepmax.
.    6 -  Rounding errors may prevent further progress. 
         There may not be a step that satisfies the 
         sufficient decrease and curvature conditions.  
         Tolerances may be too small.
+    7 -  Search direction is not a descent direction.

   Notes:
   This routine is used within the SNESUMLS method.
@ */
int SNESMoreLineSearch(SNES snes,Vec X,Vec G,Vec S,Vec W,double *f,
                  double *step,double *gnorm,int *info)
{
  SNES_UM_LS *neP = (SNES_UM_LS*)snes->data;
  double     zero = 0.0,two = 2.0,p5 = 0.5,p66 = 0.66,xtrapf = 4.0;
  double     finit,width,width1,dginit,fm,fxm,fym,dgm,dgxm,dgym;
  double     dgx,dgy,dg,fx,fy,stx,sty,dgtest,ftest1;
  int        ierr,i,stage1;
#if defined(PETSC_USE_COMPLEX)
  Scalar    cdginit,cdg,cstep = 0.0;
#endif

  PetscFunctionBegin;
  /* neP->stepmin - lower bound for step */
  /* neP->stepmax - upper bound for step */
  /* neP->rtol 	  - relative tolerance for an acceptable step */
  /* neP->ftol 	  - tolerance for sufficient decrease condition */
  /* neP->gtol 	  - tolerance for curvature condition */
  /* neP->nfev 	  - number of function evaluations */
  /* neP->maxfev  - maximum number of function evaluations */

  /* Check input parameters for errors */
  if (*step < zero) {
    PLogInfo(snes,"SNESMoreLineSearch:Line search error: step (%g) < 0\n",*step);
    *info = -1; PetscFunctionReturn(0);
  } else if (neP->ftol < zero) {
    PLogInfo(snes,"SNESMoreLineSearch:Line search error: ftol (%g) < 0\n,neP->ftol");
    *info = -2; PetscFunctionReturn(0);
  } else if (neP->rtol < zero) {
    PLogInfo(snes,"SNESMoreLineSearch:Line search error: rtol (%g) < 0\n",neP->rtol);
    *info = -3; PetscFunctionReturn(0);
  } else if (neP->gtol < zero) {
    PLogInfo(snes,"SNESMoreLineSearch:Line search error: gtol (%g) < 0\n",neP->gtol);
    *info = -4; PetscFunctionReturn(0);
  } else if (neP->stepmin < zero) {
    PLogInfo(snes,"SNESMoreLineSearch:Line search error: stepmin (%g) < 0\n,neP->stepmin");
    *info = -5; PetscFunctionReturn(0);
  } else if (neP->stepmax < neP->stepmin) {
    PLogInfo(snes,"SNESMoreLineSearch:Line search error: stepmax (%g) < stepmin (%g)\n",
       neP->stepmax,neP->stepmin);
    *info = -6; PetscFunctionReturn(0);
  } else if (neP->maxfev < zero) {
    PLogInfo(snes,"SNESMoreLineSearch:Line search error: maxfev (%d) < 0\n",neP->maxfev);
    *info = -7; PetscFunctionReturn(0);
  }

  /* Check that search direction is a descent direction */
#if defined(PETSC_USE_COMPLEX)
  ierr = VecDot(G,S,&cdginit);CHKERRQ(ierr); dginit = PetscRealPart(cdginit);
#else
  ierr = VecDot(G,S,&dginit);CHKERRQ(ierr);  /* dginit = G^T S */
#endif
  if (dginit >= zero) {
    PLogInfo(snes,"SNESMoreLineSearch:Search direction not a descent direction\n");
    *info = 7; PetscFunctionReturn(0);
  }

  /* Initialization */
  neP->bracket = 0;
  *info	  = 0;
  stage1  = 1;
  finit   = *f;
  dgtest  = neP->ftol * dginit;
  width   = neP->stepmax - neP->stepmin;
  width1  = width * two;
  ierr = VecCopy(X,W);CHKERRQ(ierr);
  /* Variable dictionary:  
     stx, fx, dgx - the step, function, and derivative at the best step
     sty, fy, dgy - the step, function, and derivative at the other endpoint 
                   of the interval of uncertainty
     step, f, dg - the step, function, and derivative at the current step */

  stx = zero;
  fx  = finit;
  dgx = dginit;
  sty = zero;
  fy  = finit;
  dgy = dginit;
 
  neP->nfev = 0;
  for (i=0; i< neP->maxfev; i++) {
    /* Set min and max steps to correspond to the interval of uncertainty */
    if (neP->bracket) {
      neP->stepmin = PetscMin(stx,sty); 
      neP->stepmax = PetscMax(stx,sty); 
    } else {
      neP->stepmin = stx;
      neP->stepmax = *step + xtrapf * (*step - stx);
    }

    /* Force the step to be within the bounds */
    *step = PetscMax(*step,neP->stepmin);
    *step = PetscMin(*step,neP->stepmax);

    /* If an unusual termination is to occur, then let step be the lowest
       point obtained thus far */
    if (((neP->bracket) && (*step <= neP->stepmin || *step >= neP->stepmax)) ||
        ((neP->bracket) && (neP->stepmax - neP->stepmin <= neP->rtol * neP->stepmax)) ||
        (neP->nfev >= neP->maxfev - 1) || (neP->infoc == 0)) 
      *step = stx;

#if defined(PETSC_USE_COMPLEX)
    cstep = *step;
    ierr = VecWAXPY(&cstep,S,W,X);CHKERRQ(ierr);
#else
    ierr = VecWAXPY(step,S,W,X);CHKERRQ(ierr); 	/* X = W + step*S */
#endif
    ierr = SNESComputeMinimizationFunction(snes,X,f);CHKERRQ(ierr);
    neP->nfev++;
    ierr = SNESComputeGradient(snes,X,G);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
    ierr = VecDot(G,S,&cdg);CHKERRQ(ierr); dg = PetscRealPart(cdg);
#else
    ierr = VecDot(G,S,&dg);CHKERRQ(ierr);	        /* dg = G^T S */
#endif
    ftest1 = finit + *step * dgtest;
  
    /* Convergence testing */
    if (((neP->bracket) && (*step <= neP->stepmin||*step >= neP->stepmax)) || (!neP->infoc)) {
      *info = 6;
      PLogInfo(snes,"SNESMoreLineSearch:Rounding errors may prevent further progress.  May not be a step satisfying\n");
      PLogInfo(snes,"SNESMoreLineSearch:sufficient decrease and curvature conditions. Tolerances may be too small.\n");
    }
    if ((*step == neP->stepmax) && (*f <= ftest1) && (dg <= dgtest)) {
      PLogInfo(snes,"SNESMoreLineSearch:Step is at the upper bound, stepmax (%g)\n",neP->stepmax);
      *info = 5;
    }
    if ((*step == neP->stepmin) && (*f >= ftest1) && (dg >= dgtest)) {
      PLogInfo(snes,"SNESMoreLineSearch:Step is at the lower bound, stepmin (%g)\n",neP->stepmin);
      *info = 4;
    }
    if (neP->nfev >= neP->maxfev) {
      PLogInfo(snes,"SNESMoreLineSearch:Number of line search function evals (%d) > maximum (%d)\n",neP->nfev,neP->maxfev);
      *info = 3;
    }
    if ((neP->bracket) && (neP->stepmax - neP->stepmin <= neP->rtol*neP->stepmax)){
      PLogInfo(snes,"SNESMoreLineSearch:Relative width of interval of uncertainty is at most rtol (%g)\n",neP->rtol);
      *info = 2;
    }
    if ((*f <= ftest1) && (PetscAbsDouble(dg) <= neP->gtol*(-dginit))) {
      PLogInfo(snes,"SNESMoreLineSearch:Line search success: Sufficient decrease and directional deriv conditions hold\n");
      *info = 1;
    }
    if (*info) break;

    /* In the first stage, we seek a step for which the modified function
        has a nonpositive value and nonnegative derivative */
    if ((stage1) && (*f <= ftest1) && (dg >= dginit * PetscMin(neP->ftol,neP->gtol)))
      stage1 = 0;

    /* A modified function is used to predict the step only if we
       have not obtained a step for which the modified function has a 
       nonpositive function value and nonnegative derivative, and if a
       lower function value has been obtained but the decrease is not
       sufficient */

    if ((stage1) && (*f <= fx) && (*f > ftest1)) {
      fm   = *f - *step * dgtest;	/* Define modified function */
      fxm  = fx - stx * dgtest;	        /* and derivatives */
      fym  = fy - sty * dgtest;
      dgm  = dg - dgtest;
      dgxm = dgx - dgtest;
      dgym = dgy - dgtest;

      /* Update the interval of uncertainty and compute the new step */
      ierr = SNESStep(snes,&stx,&fxm,&dgxm,&sty,&fym,&dgym,step,&fm,&dgm);CHKERRQ(ierr);

      fx  = fxm + stx * dgtest;	/* Reset the function and */
      fy  = fym + sty * dgtest;	/* gradient values */
      dgx = dgxm + dgtest; 
      dgy = dgym + dgtest; 
    } else {
      /* Update the interval of uncertainty and compute the new step */
      ierr = SNESStep(snes,&stx,&fx,&dgx,&sty,&fy,&dgy,step,f,&dg);CHKERRQ(ierr);
    }

   /* Force a sufficient decrease in the interval of uncertainty */
   if (neP->bracket) {
     if (PetscAbsDouble(sty - stx) >= p66 * width1) *step = stx + p5*(sty - stx);
       width1 = width;
       width = PetscAbsDouble(sty - stx);
     }
   }

  /* Finish computations */
  PLogInfo(snes,"SNESMoreLineSearch:%d function evals in line search, step = %10.4f\n",neP->nfev,neP->step);
  ierr = VecNorm(G,NORM_2,gnorm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"SNESLineSearchGetDampingParameter_UM_LS"
int SNESLineSearchGetDampingParameter_UM_LS(SNES snes,Scalar *damp)
{
  SNES_UM_LS *neP;

  PetscFunctionBegin;
  neP = (SNES_UM_LS*)snes->data;
  *damp = neP->gamma;
  PetscFunctionReturn(0);
}
EXTERN_C_END

/* ---------------------------------------------------------- */
EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"SNESCreate_UM_LS"
int SNESCreate_UM_LS(SNES snes)
{
  SNES_UM_LS *neP;
  SLES      sles;
  PC        pc;
  int       ierr;

  PetscFunctionBegin;
  if (snes->method_class != SNES_UNCONSTRAINED_MINIMIZATION) {
    SETERRQ(PETSC_ERR_ARG_WRONG,0,"For SNES_UNCONSTRAINED_MINIMIZATION only");
  }
  snes->setup		  = SNESSetUp_UM_LS;
  snes->solve		  = SNESSolve_UM_LS;
  snes->destroy		  = SNESDestroy_UM_LS;
  snes->converged	  = SNESConverged_UM_LS;
  snes->view              = SNESView_UM_LS;
  snes->setfromoptions    = SNESSetFromOptions_UM_LS;
  snes->nwork             = 0;

  neP			  = PetscNew(SNES_UM_LS);CHKPTRQ(neP);
  PLogObjectMemory(snes,sizeof(SNES_UM_LS));
  snes->data	          = (void*)neP;
  neP->LineSearch	  = SNESMoreLineSearch; 
  neP->gamma		  = 0.0;
  neP->gamma_factor	  = 0.005;
  neP->max_kspiter_factor = 5;
  neP->step		  = 1.0; 
  neP->ftol		  = 0.001;
  neP->rtol		  = 1.0e-10;
  neP->gtol		  = 0.90;
  neP->stepmin		  = 1.0e-20;
  neP->stepmax		  = 1.0e+20;
  neP->nfev		  = 0; 
  neP->bracket		  = 0; 
  neP->infoc              = 1;
  neP->maxfev		  = 30;

  /* Set default preconditioner to be Jacobi, to override SLES default. */
  ierr = SNESGetSLES(snes,&sles);CHKERRQ(ierr);
  ierr = SLESGetPC(sles,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCJACOBI);CHKERRQ(ierr);

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)snes,"SNESLineSearchGetDampingParameter_C",
                                    "SNESLineSearchGetDampingParameter_UM_LS",
                                     SNESLineSearchGetDampingParameter_UM_LS);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END


#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"SNESLineSearchGetDampingParameter"
/* @
   SNESLineSearchGetDampingParameter - Gets the damping parameter used within
   the line search method SNESUMLS for unconstrained minimization.

   Input Parameter:
.  type - SNES method

   Output Parameter:
.  damp - the damping parameter

.keywords: SNES, nonlinear, get, line search, damping parameter
@ */
int SNESLineSearchGetDampingParameter(SNES snes,Scalar *damp)
{
  int ierr,(*f)(SNES,Scalar *);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE);

  ierr = PetscObjectQueryFunction((PetscObject)snes,"SNESLineSearchGetDampingParameter_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(snes,damp);CHKERRQ(ierr);
  } else {
    SETERRQ(1,1,"Can only get line search damping when line search algorithm used");
  }
  PetscFunctionReturn(0);
}
