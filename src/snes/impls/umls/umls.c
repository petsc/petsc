#ifndef lint
static char vcid[] = "$Id: umls.c,v 1.13 1995/11/01 19:12:15 bsmith Exp bsmith $";
#endif

#include <math.h>
#include "umls.h"
#include "pinclude/pviewer.h"

extern int SNESStep(SNES,double*,double*,double*,double*,
                    double*,double*,double*,double*,double*);

/*
   Implements Newton's Method with a line search approach
   for solving unconstrained minimization problems.

   Note:
   The line search algorithm is taken from More and Thuente,
   "Line search algorithms with guaranteed sufficient decrease",
   Argonne National Laboratory", Technical Report MCS-P330-1092.
*/
static int SNESSolve_UMLS(SNES snes,int *outits)
{
  SNES_UMLS    *neP = (SNES_UMLS *) snes->data;
  int          maxits, success, iters, history_len, i, global_dim, ierr, kspmaxit;
  double       *history, snorm, *f, *gnorm, two = 2.0;
  Scalar       neg_one = -1.0;
  Vec          G, X, RHS, S, W;
  SLES         sles;
  KSP          ksp;
  MatStructure flg = ALLMAT_DIFFERENT_NONZERO_PATTERN;

  history	= snes->conv_hist;      /* convergence history */
  history_len	= snes->conv_hist_len;  /* convergence history length */
  maxits	= snes->max_its;        /* maximum number of iterations */
  X		= snes->vec_sol; 	/* solution vector */
  G		= snes->vec_func;	/* gradient vector */
  RHS		= snes->work[0]; 	/* work vectors */
  S		= snes->work[1];	/* step vector */
  W		= snes->work[2];	/* work vector */
  f		= &(snes->fc);		/* function to minimize */
  gnorm		= &(snes->norm);	/* gradient norm */

  ierr = SNESComputeInitialGuess(snes,X); CHKERRQ(ierr);/* X <- X_0 */
  ierr = SNESComputeMinimizationFunction(snes,X,f); CHKERRQ(ierr); /* f(X) */
  ierr = SNESComputeGradient(snes,X,G); CHKERRQ(ierr);  /* G(X) <- gradient */
  ierr = VecNorm(G,NORM_2,gnorm);   CHKERRQ(ierr);             /* gnorm = || G || */
  if (history && history_len > 0) history[0] = *gnorm;
  if (snes->monitor){(*snes->monitor)(snes,0,*gnorm,snes->monP); CHKERRQ(ierr);}

  ierr = SNESGetSLES(snes,&sles); CHKERRQ(ierr);
  ierr = SLESGetKSP(sles,&ksp); CHKERRQ(ierr);
  ierr = VecGetSize(X,&global_dim); CHKERRQ(ierr);
  kspmaxit = neP->max_kspiter_factor * ((int) sqrt((double) global_dim));
  ierr = KSPSetTolerances(ksp,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,kspmaxit);
         CHKERRQ(ierr);

  for ( i=0; i<maxits; i++ ) {
    snes->iter = i+1;
    neP->gamma = neP->gamma_factor*(*gnorm);
    success = 0;
    ierr = VecCopy(G,RHS); CHKERRQ(ierr);
    ierr = VecScale(&neg_one,RHS); CHKERRQ(ierr);
    while (!success) {
      ierr = SNESComputeHessian(snes,X,&snes->jacobian,&snes->jacobian_pre,&flg);
             CHKERRQ(ierr);
      /* Modify diagonal elements of Hessian */
      ierr = SLESSetOperators(snes->sles,snes->jacobian,snes->jacobian_pre,flg);
             CHKERRQ(ierr);
      ierr = SLESSolve(snes->sles,RHS,S,&iters); CHKERRQ(ierr);
      ierr = VecNorm(S,NORM_2,&snorm); CHKERRQ(ierr);
      if ((iters < 0) || (iters >= kspmaxit)) {
        neP->gamma_factor *= two; 
        neP->gamma = neP->gamma_factor*(*gnorm); 
      } else {
        success = 1;
      }
    }   
    /* Line search */
    ierr = (*neP->LineSearch)(snes,X,G,S,W,f,&(neP->step),gnorm,&(neP->line));
    if (neP->line != 1) snes->nfailures++;
    CHKERRQ(ierr);

    if (history && history_len > i+1) history[i+1] = *gnorm;
    if (snes->monitor) {ierr = (*snes->monitor)(snes,i+1,*gnorm,snes->monP); CHKERRQ(ierr);}
    PLogInfo((PetscObject)snes,"%d:  f=%g, gnorm=%g, snorm=%g, step=%g, KSP iters=%d\n",
             snes->iter, *f, *gnorm, snorm, neP->step, iters );

    /* Test for convergence */
    if ((*snes->converged)(snes,snorm,*gnorm,*f,snes->cnvP)) break;
    neP->gamma_factor /= two;
  }
  /* Verify solution is in correct location */
  if (X != snes->vec_sol) {
    ierr = VecCopy(X,snes->vec_sol); CHKERRQ(ierr);
    snes->vec_sol_always = snes->vec_sol;
    snes->vec_func_always = snes->vec_func;
  }
  if (i == maxits) {
    PLogInfo((PetscObject)snes,"SNES: Maximum number of iterations reached: %d\n",maxits);
    i--;
  }
  *outits = i+1;
  return 0;
}
/* ---------------------------------------------------------- */
static int SNESSetUp_UMLS(SNES snes)
{
  int ierr;
  snes->nwork = 4;
  ierr = VecGetVecs(snes->vec_sol,snes->nwork,&snes->work); CHKERRQ(ierr);
  PLogObjectParents(snes,snes->nwork,snes->work);
  snes->vec_sol_update_always = snes->work[3];
  return 0;
}
/*------------------------------------------------------------*/
static int SNESDestroy_UMLS(PetscObject obj )
{
  SNES snes = (SNES) obj;
  VecFreeVecs(snes->work,snes->nwork);
  PetscFree(snes->data);
  return 0;
}
/*------------------------------------------------------------*/
static int SNESSetFromOptions_UMLS(SNES snes)
{
  SNES_UMLS *ctx = (SNES_UMLS *)snes->data;
  double    tmp;
  int       itmp;

  if (OptionsGetDouble(snes->prefix,"-gamma_factor",&tmp)) {ctx->gamma_factor = tmp;}
  if (OptionsGetInt(snes->prefix,"-maxfev",&itmp)) {ctx->maxfev = itmp;}
  if (OptionsGetDouble(snes->prefix,"-ftol",&tmp)) {ctx->ftol = tmp;}
  if (OptionsGetDouble(snes->prefix,"-gtol",&tmp)) {ctx->gtol = tmp;}
  if (OptionsGetDouble(snes->prefix,"-rtol",&tmp)) {ctx->rtol = tmp;}
  if (OptionsGetDouble(snes->prefix,"-stepmin",&tmp)) {ctx->stepmin = tmp;}
  if (OptionsGetDouble(snes->prefix,"-stepmax",&tmp)) {ctx->stepmax = tmp;}
  return 0;
}
/*------------------------------------------------------------*/
static int SNESPrintHelp_UMLS(SNES snes)
{
  SNES_UMLS *ctx = (SNES_UMLS *)snes->data;
  char      *p;

  if (snes->prefix) p = snes->prefix; else p = "-";
  MPIU_printf(snes->comm," method umls (unconstrained minimization):\n");
  MPIU_printf(snes->comm,"   %ssnes_line_search_gamma_f gamma_f (default %g) damping parameter\n",
    p,ctx->gamma_factor);
  MPIU_printf(snes->comm,"   %ssnes_line_search_maxf maxf (default %d) max function evals in line search\n",p,ctx->maxfev);
  MPIU_printf(snes->comm,"   %ssnes_line_search_maxkspf (default %d) computes max KSP iters\n",p,ctx->max_kspiter_factor);
  MPIU_printf(snes->comm,"   %ssnes_line_search_ftol ftol (default %g) tol for sufficient decrease\n",p,ctx->ftol);
  MPIU_printf(snes->comm,"   %ssnes_line_search_rtol rtol (default %g) relative tol for acceptable step\n",p,ctx->rtol);
  MPIU_printf(snes->comm,"   %ssnes_line_search_gtol gtol (default %g) tol for curvature condition\n",p,ctx->gtol);
  MPIU_printf(snes->comm,"   %ssnes_line_search_stepmin stepmin (default %g) lower bound for step\n",p,ctx->stepmin);
  MPIU_printf(snes->comm,"   %ssnes_line_search_stepmax stepmax (default %g) upper bound for step\n",p,ctx->stepmax);
  return 0;
}
/*------------------------------------------------------------*/
static int SNESView_UMLS(PetscObject obj,Viewer viewer)
{
  SNES      snes = (SNES)obj;
  SNES_UMLS *ls = (SNES_UMLS *)snes->data;
  FILE      *fd;
  int       ierr;

  ierr = ViewerFileGetPointer_Private(viewer,&fd); CHKERRQ(ierr);
  MPIU_fprintf(snes->comm,fd,"    gamma_f=%g, maxf=%d, maxkspf=%d,stepmin=%g,stepmax=%g\n",
               ls->gamma_factor,ls->maxfev,ls->max_kspiter_factor,ls->stepmin,ls->stepmax);
  MPIU_fprintf(snes->comm,fd,"    ftol=%g, rtol=%g,gtol=%g\n",ls->ftol,ls->rtol,ls->gtol);
  return 0;
}
/* ---------------------------------------------------------- */
/*@ 
   SNESConverged_UMLS - Default test for monitoring the 
   convergence of the SNESSolve_UMLS() routine. 

   Input Parameters:
.  snes - the SNES context
.  xnorm - 2-norm of current iterate
.  gnorm - 2-norm of current gradient
.  f - objective function value

   Returns:
$   1  if  ( f < fmin ),
$   2  if  ( gnorm < atol ),
$  -1  if  ( nfunc > max_func ),
$  -2  if  ( gnorm < epsmch ),
$  -3  if  line search attempt failed,
$   0  otherwise,

   where
$  atol     - absolute function tolerance,
$             set with SNESSetAbsoluteTolerance()
$  epsmch   - machine epsilon
$  fmin     - lower bound on function value,
$             set with SNESSetMinFunctionTolerance()
$  max_func - maximum number of function evaluations,
$             set with SNESSetMaxFunctionEvaluations()
$  nfunc    - number of function evaluations
@*/
int SNESConverged_UMLS(SNES snes,double xnorm,double gnorm,double f,
                       void *dummy)
{
  SNES_UMLS *neP = (SNES_UMLS *) snes->data;
  double    epsmch = 1.0e-14;   /* This must be fixed */

  /* Test for successful convergence */
  if (f < snes->fmin) {
    PLogInfo((PetscObject)snes,
      "SNES: Converged due to function value %g < minimum function value %g\n",f,snes->fmin);
    return 1;
  }
  if (gnorm < snes->atol) {
    PLogInfo((PetscObject)snes,"SNES:Converged due to gradient norm %g<%g\n",gnorm,snes->atol);
    return 2;
  }
  /* Test for termination and stringent tolerances. (failure and stop) */
 if (snes->nfuncs > snes->max_funcs) {
    PLogInfo((PetscObject)snes,
             "SNES: Exceeded maximum number of function evaluations: %d>%d\n",
             snes->nfuncs,snes->max_funcs );
    return -1;
  } 
  if (gnorm < epsmch) {
    PLogInfo((PetscObject)snes,"SNES: Gradient norm %g < minimum tolerance %g\n",gnorm,epsmch);
    return -2;
  }
  if (neP->line != 1) {
    PLogInfo((PetscObject)snes,"SNES: Line search failed for above reason\n");
    return -3;
  }
  return 0;
}
/* ---------------------------------------------------------- */
/* @ SNESMoreLineSearch - This routine performs a line search algorithm,
     taken from More and Thuente, "Line search algorithms with 
     guaranteed sufficient decrease", Argonne National Laboratory", 
     Technical Report MCS-P330-1092.

   Input Parameters:
.  snes - SNES context
.  X - current iterate (on output X contains new iterate, X + step*S)
.  S - search direction
.  f - objective function evaluated at X
.  G - gradient evaluated at X
.  W - work vector
.  step - initial estimate of step length

   Output parameters:
.  f - objective function evaluated at new iterate, X + step*S
.  G - gradient evaluated at new iterate, X + step*S
.  X - new iterate
.  gnorm - 2-norm of G
.  step - final step length

   Info is set to one of:
$  1 if the line search succeeds:  the sufficient decrease
$    condition and the directional derivative condition hold
$
$  negative number if the input parameters are invalid:
$   -1:  step < 0
$   -2:  ftol < 0 
$   -3:  rtol < 0 
$   -4:  gtol < 0 
$   -5:  stepmin < 0 
$   -6:  stepmax < stepmin 
$   -7:  maxfev < 0 
$
$  positive number > 1 if the line search otherwise terminates:
$    2:  Relative width of the interval of uncertainty is 
$        at most rtol.
$    3:  Maximum number of function evaluations (maxfev) has 
$        been reached.
$    4:  Step is at the lower bound, stepmin.
$    5:  Step is at the upper bound, stepmax.
$    6:  Rounding errors may prevent further progress. 
$        There may not be a step that satisfies the 
$        sufficient decrease and curvature conditions.  
$        Tolerances may be too small.
$    7:  Search direction is not a descent direction.

   Notes:
   This routine is used within the SNES_UM_NLS method.
@ */
int SNESMoreLineSearch(SNES snes,Vec X,Vec G,Vec S,Vec W,double *f,
                  double *step,double *gnorm,int *info)
{
  SNES_UMLS *neP = (SNES_UMLS *) snes->data;
  double    zero = 0.0, two = 2.0, p5 = 0.5, p66 = 0.66, xtrapf = 4.0;
  double    finit, width, width1, dginit,fm, fxm, fym, dgm, dgxm, dgym;
  double    dgx, dgy, dg, fx, fy, stx, sty, dgtest, ftest1;
  int       ierr, i, stage1;

 /* This is not correctly coded for complex version */
#if defined(PETSC_COMPLEX)
  Scalar    cdginit, cstep, cdg;
#endif

  /* neP->stepmin - lower bound for step */
  /* neP->stepmax - upper bound for step */
  /* neP->rtol 	  - relative tolerance for an acceptable step */
  /* neP->ftol 	  - tolerance for sufficient decrease condition */
  /* neP->gtol 	  - tolerance for curvature condition */
  /* neP->nfev 	  - number of function evaluations */
  /* neP->maxfev  - maximum number of function evaluations */

  /* Check input parameters for errors */
  if (*step < zero) {
    PLogInfo((PetscObject)snes,"Line search error: step (%g) < 0\n",*step);
    *info = -1; return 0;
  } else if (neP->ftol < zero) {
    PLogInfo((PetscObject)snes,"Line search error: ftol (%g) < 0\n,neP->ftol");
    *info = -2; return 0;
  } else if (neP->rtol < zero) {
    PLogInfo((PetscObject)snes,"Line search error: rtol (%g) < 0\n",neP->rtol);
    *info = -3; return 0;
  } else if (neP->gtol < zero) {
    PLogInfo((PetscObject)snes,"Line search error: gtol (%g) < 0\n",neP->gtol);
    *info = -4; return 0;
  } else if (neP->stepmin < zero) {
    PLogInfo((PetscObject)snes,"Line search error: stepmin (%g) < 0\n,neP->stepmin");
    *info = -5; return 0;
  } else if (neP->stepmax < neP->stepmin) {
    PLogInfo((PetscObject)snes,"Line search error: stepmax (%g) < stepmin (%g)\n",
       neP->stepmax,neP->stepmin);
    *info = -6; return 0;
  } else if (neP->maxfev < zero) {
    PLogInfo((PetscObject)snes,"Line search error: maxfev (%d) < 0\n",neP->maxfev);
    *info = -7; return 0;
  }

  /* Check that search direction is a descent direction */
#if defined(PETSC_COMPLEX)
  ierr = VecDot(G,S,&cdginit); CHKERRQ(ierr); dginit = real(cdginit);
#else
  ierr = VecDot(G,S,&dginit); CHKERRQ(ierr);  /* dginit = G^T S */
#endif
  if (dginit >= zero) {
    PLogInfo((PetscObject)snes,"Line search error:Search direction not a descent direction\n");
    *info = 7; return 0;
  }

  /* Initialization */
  *info	  = 0;
  stage1  = 1;
  finit   = *f;
  dgtest  = neP->ftol * dginit;
  width   = neP->stepmax - neP->stepmin;
  width1  = width * two;
  ierr = VecCopy(X,W); CHKERRQ(ierr);
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

#if defined(PETSC_COMPLEX)
    ierr = VecWAXPY(&cstep,S,W,X); CHKERRQ(ierr); *step = real(cstep);
#else
    ierr = VecWAXPY(step,S,W,X); CHKERRQ(ierr); 	/* X = W + step*S */
#endif
    ierr = SNESComputeMinimizationFunction(snes,X,f); CHKERRQ(ierr);
    neP->nfev++;
    ierr = SNESComputeGradient(snes,X,G); CHKERRQ(ierr);
#if defined(PETSC_COMPLEX)
    ierr = VecDot(G,S,&cdg); CHKERRQ(ierr); dg = real(cdg);
#else
    ierr = VecDot(G,S,&dg); CHKERRQ(ierr);	        /* dg = G^T S */
#endif
    ftest1 = finit + *step * dgtest;
  
    /* Convergence testing */
    if (((neP->bracket)&&(*step <= neP->stepmin||*step >= neP->stepmax))||(!neP->infoc)) {
      *info = 6;
      PLogInfo((PetscObject)snes,
        "Rounding errors may prevent further progress.  May not be a step satisfying\n");
      PLogInfo((PetscObject)snes,
        "sufficient decrease and curvature conditions. Tolerances may be too small.\n");
    }
    if ((*step == neP->stepmax) && (*f <= ftest1) && (dg <= dgtest)) {
      PLogInfo((PetscObject)snes,"Step is at the upper bound, stepmax (%g)\n",neP->stepmax);
      *info = 5;
    }
    if ((*step == neP->stepmin) && (*f >= ftest1) && (dg >= dgtest)) {
      PLogInfo((PetscObject)snes,"Step is at the lower bound, stepmin (%g)\n",neP->stepmin);
      *info = 4;
    }
    if (neP->nfev >= neP->maxfev) {
      PLogInfo((PetscObject)snes,
        "Number of line search function evals (%d) > maximum (%d)\n",neP->nfev,neP->maxfev);
      *info = 3;
    }
    if ((neP->bracket) && (neP->stepmax - neP->stepmin <= neP->rtol*neP->stepmax)){
      PLogInfo((PetscObject)snes,
        "Relative width of interval of uncertainty is at most rtol (%g)\n",neP->rtol);
      *info = 2;
    }
    if ((*f <= ftest1) && (PetscAbsScalar(dg) <= neP->gtol*(-dginit))) {
      PLogInfo((PetscObject)snes,
        "Line search success: Sufficient decrease and directional deriv conditions hold\n");
      *info = 1;
    }
    if (*info) break;

    /* In the first stage, we seek a step for which the modified function
        has a nonpositive value and nonnegative derivative */
    if ((stage1) && (*f <= ftest1) && (dg >= dginit * PetscMin(neP->ftol, neP->gtol)))
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
      ierr = SNESStep(snes,&stx,&fxm,&dgxm,&sty,&fym,&dgym,step,&fm,&dgm); 
      CHKERRQ(ierr);

      fx  = fxm + stx * dgtest;	/* Reset the function and */
      fy  = fym + sty * dgtest;	/* gradient values */
      dgx = dgxm + dgtest; 
      dgy = dgym + dgtest; 
    } else {
      /* Update the interval of uncertainty and compute the new step */
      ierr = SNESStep(snes,&stx,&fx,&dgx,&sty,&fy,&dgy,step,f,&dg); CHKERRQ(ierr);
    }

   /* Force a sufficient decrease in the interval of uncertainty */
   if (neP->bracket) {
     if (PetscAbsScalar(sty - stx) >= p66 * width1) *step = stx + p5*(sty - stx);
       width1 = width;
       width = PetscAbsScalar(sty - stx);
     }
   }

  /* Finish computations */
  PLogInfo((PetscObject)snes,"%d function evals in line search, step = %10.4f\n", 
           neP->nfev,neP->step);
  ierr = VecNorm(G,NORM_2,gnorm); CHKERRQ(ierr);
  return 0;
}
/* ---------------------------------------------------------- */
int SNESCreate_UMLS(SNES snes)
{
  SNES_UMLS *neP;

  if (snes->method_class != SNES_UNCONSTRAINED_MINIMIZATION) SETERRQ(1,
    "SNESCreate_UMLS:For SNES_UNCONSTRAINED_MINIMIZATION only");
  snes->type 		  = SNES_UM_NLS;
  snes->setup		  = SNESSetUp_UMLS;
  snes->solve		  = SNESSolve_UMLS;
  snes->destroy		  = SNESDestroy_UMLS;
  snes->converged	  = SNESConverged_UMLS;
  snes->monitor           = SNESDefaultMonitor;
  snes->printhelp         = SNESPrintHelp_UMLS;
  snes->view              = SNESView_UMLS;
  snes->setfromoptions    = SNESSetFromOptions_UMLS;

  neP			  = PetscNew(SNES_UMLS); CHKPTRQ(neP);
  PLogObjectMemory(snes,sizeof(SNES_UMLS));
  snes->data	          = (void *) neP;
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
  return 0;
}

/* @
   SNESGetLineSearchDampingParameter - Gets the damping parameter used within
   the line search method SNES_UM_NLS for unconstrained minimization.

   Input Parameter:
.  method - SNES method

   Output Parameter:
.  damp - the damping parameter

.keywords: SNES, nonlinear, get, line search, damping parameter
@ */
int SNESGetLineSearchDampingParameter(SNES snes,double *damp)
{
  SNES_UMLS *neP;
  PETSCVALIDHEADERSPECIFIC(snes,SNES_COOKIE);
  neP = (SNES_UMLS *) snes->data;
  *damp = neP->gamma;
  return 0;
}
