
#include "src/snes/impls/ls/ls.h"

/*
     Checks if J^T F = 0 which implies we've found a local minimum of the function,
    but not a zero. In the case when one cannot compute J^T F we use the fact that
    0 = (J^T F)^T W = F^T J W iff W not in the null space of J. Thanks for Jorge More 
    for this trick.
*/ 
#undef __FUNCT__  
#define __FUNCT__ "SNESLSCheckLocalMin_Private"
PetscErrorCode SNESLSCheckLocalMin_Private(Mat A,Vec F,Vec W,PetscReal fnorm,PetscTruth *ismin)
{
  PetscReal  a1;
  PetscErrorCode ierr;
  PetscTruth hastranspose;

  PetscFunctionBegin;
  *ismin = PETSC_FALSE;
  ierr = MatHasOperation(A,MATOP_MULT_TRANSPOSE,&hastranspose);CHKERRQ(ierr);
  if (hastranspose) {
    /* Compute || J^T F|| */
    ierr = MatMultTranspose(A,F,W);CHKERRQ(ierr);
    ierr = VecNorm(W,NORM_2,&a1);CHKERRQ(ierr);
    PetscLogInfo(0,"SNESSolve_LS: || J^T F|| %g near zero implies found a local minimum\n",a1/fnorm);
    if (a1/fnorm < 1.e-4) *ismin = PETSC_TRUE;
  } else {
    Vec       work;
    PetscScalar    result;
    PetscReal wnorm;

    ierr = VecSetRandom(PETSC_NULL,W);CHKERRQ(ierr);
    ierr = VecNorm(W,NORM_2,&wnorm);CHKERRQ(ierr);
    ierr = VecDuplicate(W,&work);CHKERRQ(ierr);
    ierr = MatMult(A,W,work);CHKERRQ(ierr);
    ierr = VecDot(F,work,&result);CHKERRQ(ierr);
    ierr = VecDestroy(work);CHKERRQ(ierr);
    a1   = PetscAbsScalar(result)/(fnorm*wnorm);
    PetscLogInfo(0,"SNESSolve_LS: (F^T J random)/(|| F ||*||random|| %g near zero implies found a local minimum\n",a1);
    if (a1 < 1.e-4) *ismin = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

/*
     Checks if J^T(F - J*X) = 0 
*/ 
#undef __FUNCT__  
#define __FUNCT__ "SNESLSCheckResidual_Private"
PetscErrorCode SNESLSCheckResidual_Private(Mat A,Vec F,Vec X,Vec W1,Vec W2)
{
  PetscReal     a1,a2;
  PetscErrorCode ierr;
  PetscTruth    hastranspose;
  PetscScalar   mone = -1.0;

  PetscFunctionBegin;
  ierr = MatHasOperation(A,MATOP_MULT_TRANSPOSE,&hastranspose);CHKERRQ(ierr);
  if (hastranspose) {
    ierr = MatMult(A,X,W1);CHKERRQ(ierr);
    ierr = VecAXPY(&mone,F,W1);CHKERRQ(ierr);

    /* Compute || J^T W|| */
    ierr = MatMultTranspose(A,W1,W2);CHKERRQ(ierr);
    ierr = VecNorm(W1,NORM_2,&a1);CHKERRQ(ierr);
    ierr = VecNorm(W2,NORM_2,&a2);CHKERRQ(ierr);
    if (a1 != 0) {
      PetscLogInfo(0,"SNESSolve_LS: ||J^T(F-Ax)||/||F-AX|| %g near zero implies inconsistent rhs\n",a2/a1);
    }
  }
  PetscFunctionReturn(0);
}

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
  SNES_LS        *neP = (SNES_LS*)snes->data;
  PetscErrorCode ierr;
  PetscInt       maxits,i,lits;
  PetscTruth     lssucceed;
  MatStructure   flg = DIFFERENT_NONZERO_PATTERN;
  PetscReal      fnorm,gnorm,xnorm,ynorm;
  Vec            Y,X,F,G,W,TMP;
  KSP            ksp;

  PetscFunctionBegin;
  ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
  snes->reason  = SNES_CONVERGED_ITERATING;

  maxits	= snes->max_its;	/* maximum number of iterations */
  X		= snes->vec_sol;	/* solution vector */
  F		= snes->vec_func;	/* residual vector */
  Y		= snes->work[0];	/* work vectors */
  G		= snes->work[1];
  W		= snes->work[2];

  ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
  snes->iter = 0;
  ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
  ierr = SNESComputeFunction(snes,X,F);CHKERRQ(ierr);  /*  F(X)      */
  ierr = VecNorm(F,NORM_2,&fnorm);CHKERRQ(ierr);	/* fnorm <- ||F||  */
  ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
  snes->norm = fnorm;
  ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
  SNESLogConvHistory(snes,fnorm,0);
  SNESMonitor(snes,0,fnorm);

  if (fnorm < snes->abstol) {snes->reason = SNES_CONVERGED_FNORM_ABS; PetscFunctionReturn(0);}

  /* set parameter for default relative tolerance convergence test */
  snes->ttol = fnorm*snes->rtol;

  for (i=0; i<maxits; i++) {

    /* Call general purpose update function */
    if (snes->update) {
      ierr = (*snes->update)(snes, snes->iter);CHKERRQ(ierr);
    }

    /* Solve J Y = F, where J is Jacobian matrix */
    ierr = SNESComputeJacobian(snes,X,&snes->jacobian,&snes->jacobian_pre,&flg);CHKERRQ(ierr);
    ierr = KSPSetOperators(snes->ksp,snes->jacobian,snes->jacobian_pre,flg);CHKERRQ(ierr);
    ierr = KSPSolve(snes->ksp,F,Y);CHKERRQ(ierr);
    ierr = KSPGetIterationNumber(ksp,&lits);CHKERRQ(ierr);

    if (PetscLogPrintInfo){
      ierr = SNESLSCheckResidual_Private(snes->jacobian,F,Y,G,W);CHKERRQ(ierr);
    }

    /* should check what happened to the linear solve? */
    snes->linear_its += lits;
    PetscLogInfo(snes,"SNESSolve_LS: iter=%D, linear solve iterations=%D\n",snes->iter,lits);

    /* Compute a (scaled) negative update in the line search routine: 
         Y <- X - lambda*Y 
       and evaluate G(Y) = function(Y)) 
    */
    ierr = VecCopy(Y,snes->vec_sol_update_always);CHKERRQ(ierr);
    ierr = (*neP->LineSearch)(snes,neP->lsP,X,F,G,Y,W,fnorm,&ynorm,&gnorm,&lssucceed);CHKERRQ(ierr);
    PetscLogInfo(snes,"SNESSolve_LS: fnorm=%18.16e, gnorm=%18.16e, ynorm=%18.16e, lssucceed=%d\n",fnorm,gnorm,ynorm,(int)lssucceed);

    TMP = F; F = G; snes->vec_func_always = F; G = TMP;
    TMP = X; X = Y; snes->vec_sol_always = X;  Y = TMP;
    fnorm = gnorm;

    ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
    snes->iter = i+1;
    snes->norm = fnorm;
    ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
    SNESLogConvHistory(snes,fnorm,lits);
    SNESMonitor(snes,i+1,fnorm);

    if (!lssucceed) {
      PetscTruth ismin;

      if (++snes->numFailures >= snes->maxFailures) {
        snes->reason = SNES_DIVERGED_LS_FAILURE;
        ierr = SNESLSCheckLocalMin_Private(snes->jacobian,F,W,fnorm,&ismin);CHKERRQ(ierr);
        if (ismin) snes->reason = SNES_DIVERGED_LOCAL_MIN;
        break;
      }
    } 

    /* Test for convergence */
    if (snes->converged) {
      ierr = VecNorm(X,NORM_2,&xnorm);CHKERRQ(ierr);	/* xnorm = || X || */
      ierr = (*snes->converged)(snes,xnorm,ynorm,fnorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
      if (snes->reason) {
        break;
      }
    }
  }
  if (X != snes->vec_sol) {
    ierr = VecCopy(X,snes->vec_sol);CHKERRQ(ierr);
  }
  if (F != snes->vec_func) {
    ierr = VecCopy(F,snes->vec_func);CHKERRQ(ierr);
  }
  snes->vec_sol_always  = snes->vec_sol;
  snes->vec_func_always = snes->vec_func;
  if (i == maxits) {
    PetscLogInfo(snes,"SNESSolve_LS: Maximum number of iterations has been reached: %D\n",maxits);
    snes->reason = SNES_DIVERGED_MAX_IT;
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
  snes->nwork = 4;
  ierr = VecDuplicateVecs(snes->vec_sol,snes->nwork,&snes->work);CHKERRQ(ierr);
  ierr = PetscLogObjectParents(snes,snes->nwork,snes->work);CHKERRQ(ierr);
  snes->vec_sol_update_always = snes->work[3];
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
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
  if (snes->nwork) {
    ierr = VecDestroyVecs(snes->work,snes->nwork);CHKERRQ(ierr);
  }
  ierr = PetscFree(snes->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "SNESNoLineSearch"

/*@C
   SNESNoLineSearch - This routine is not a line search at all; 
   it simply uses the full Newton step.  Thus, this routine is intended 
   to serve as a template and is not recommended for general use.  

   Collective on SNES and Vec

   Input Parameters:
+  snes - nonlinear context
.  lsctx - optional context for line search (not used here)
.  x - current iterate
.  f - residual evaluated at x
.  y - search direction (contains new iterate on output)
.  w - work vector
-  fnorm - 2-norm of f

   Output Parameters:
+  g - residual evaluated at new iterate y
.  y - new iterate (contains search direction on input)
.  gnorm - 2-norm of g
.  ynorm - 2-norm of search length
-  flag - PETSC_TRUE on success, PETSC_FALSE on failure

   Options Database Key:
.  -snes_ls basic - Activates SNESNoLineSearch()

   Level: advanced

.keywords: SNES, nonlinear, line search, cubic

.seealso: SNESCubicLineSearch(), SNESQuadraticLineSearch(), 
          SNESSetLineSearch(), SNESNoLineSearchNoNorms()
@*/
PetscErrorCode SNESNoLineSearch(SNES snes,void *lsctx,Vec x,Vec f,Vec g,Vec y,Vec w,PetscReal fnorm,PetscReal *ynorm,PetscReal *gnorm,PetscTruth *flag)
{
  PetscErrorCode ierr;
  PetscScalar    mone = -1.0;
  SNES_LS        *neP = (SNES_LS*)snes->data;
  PetscTruth     change_y = PETSC_FALSE;

  PetscFunctionBegin;
  *flag = PETSC_TRUE; 
  ierr = PetscLogEventBegin(SNES_LineSearch,snes,x,f,g);CHKERRQ(ierr);
  ierr = VecNorm(y,NORM_2,ynorm);CHKERRQ(ierr);  /* ynorm = || y || */
  ierr = VecAYPX(&mone,x,y);CHKERRQ(ierr);            /* y <- y - x      */
  if (neP->CheckStep) {
   ierr = (*neP->CheckStep)(snes,neP->checkP,y,&change_y);CHKERRQ(ierr);
  }
  ierr = SNESComputeFunction(snes,y,g);CHKERRQ(ierr); /* Compute F(y)    */
  ierr = VecNorm(g,NORM_2,gnorm);CHKERRQ(ierr);  /* gnorm = || g || */
  ierr = PetscLogEventEnd(SNES_LineSearch,snes,x,f,g);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "SNESNoLineSearchNoNorms"

/*@C
   SNESNoLineSearchNoNorms - This routine is not a line search at 
   all; it simply uses the full Newton step. This version does not
   even compute the norm of the function or search direction; this
   is intended only when you know the full step is fine and are
   not checking for convergence of the nonlinear iteration (for
   example, you are running always for a fixed number of Newton steps).

   Collective on SNES and Vec

   Input Parameters:
+  snes - nonlinear context
.  lsctx - optional context for line search (not used here)
.  x - current iterate
.  f - residual evaluated at x
.  y - search direction (contains new iterate on output)
.  w - work vector
-  fnorm - 2-norm of f

   Output Parameters:
+  g - residual evaluated at new iterate y
.  gnorm - not changed
.  ynorm - not changed
-  flag - set to PETSC_TRUE indicating a successful line search

   Options Database Key:
.  -snes_ls basicnonorms - Activates SNESNoLineSearchNoNorms()

   Notes:
   SNESNoLineSearchNoNorms() must be used in conjunction with
   either the options
$     -snes_no_convergence_test -snes_max_it <its>
   or alternatively a user-defined custom test set via
   SNESSetConvergenceTest(); or a -snes_max_it of 1, 
   otherwise, the SNES solver will generate an error.

   During the final iteration this will not evaluate the function at
   the solution point. This is to save a function evaluation while
   using pseudo-timestepping.

   The residual norms printed by monitoring routines such as
   SNESDefaultMonitor() (as activated via -snes_monitor) will not be 
   correct, since they are not computed.

   Level: advanced

.keywords: SNES, nonlinear, line search, cubic

.seealso: SNESCubicLineSearch(), SNESQuadraticLineSearch(), 
          SNESSetLineSearch(), SNESNoLineSearch()
@*/
PetscErrorCode SNESNoLineSearchNoNorms(SNES snes,void *lsctx,Vec x,Vec f,Vec g,Vec y,Vec w,PetscReal fnorm,PetscReal *ynorm,PetscReal *gnorm,PetscTruth *flag)
{
  PetscErrorCode ierr;
  PetscScalar mone = -1.0;
  SNES_LS     *neP = (SNES_LS*)snes->data;
  PetscTruth  change_y = PETSC_FALSE;

  PetscFunctionBegin;
  *flag = PETSC_TRUE; 
  ierr = PetscLogEventBegin(SNES_LineSearch,snes,x,f,g);CHKERRQ(ierr);
  ierr = VecAYPX(&mone,x,y);CHKERRQ(ierr);            /* y <- y - x      */
  if (neP->CheckStep) {
   ierr = (*neP->CheckStep)(snes,neP->checkP,y,&change_y);CHKERRQ(ierr);
  }
  
  /* don't evaluate function the last time through */
  if (snes->iter < snes->max_its-1) {
    ierr = SNESComputeFunction(snes,y,g);CHKERRQ(ierr); /* Compute F(y)    */
  }
  ierr = PetscLogEventEnd(SNES_LineSearch,snes,x,f,g);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "SNESCubicLineSearch"
/*@C
   SNESCubicLineSearch - Performs a cubic line search (default line search method).

   Collective on SNES

   Input Parameters:
+  snes - nonlinear context
.  lsctx - optional context for line search (not used here)
.  x - current iterate
.  f - residual evaluated at x
.  y - search direction (contains new iterate on output)
.  w - work vector
-  fnorm - 2-norm of f

   Output Parameters:
+  g - residual evaluated at new iterate y
.  y - new iterate (contains search direction on input)
.  gnorm - 2-norm of g
.  ynorm - 2-norm of search length
-  flag - PETSC_TRUE if line search succeeds; PETSC_FALSE on failure.

   Options Database Key:
$  -snes_ls cubic - Activates SNESCubicLineSearch()

   Notes:
   This line search is taken from "Numerical Methods for Unconstrained 
   Optimization and Nonlinear Equations" by Dennis and Schnabel, page 325.

   Level: advanced

.keywords: SNES, nonlinear, line search, cubic

.seealso: SNESQuadraticLineSearch(), SNESNoLineSearch(), SNESSetLineSearch(), SNESNoLineSearchNoNorms()
@*/
PetscErrorCode SNESCubicLineSearch(SNES snes,void *lsctx,Vec x,Vec f,Vec g,Vec y,Vec w,PetscReal fnorm,PetscReal *ynorm,PetscReal *gnorm,PetscTruth *flag)
{
  /* 
     Note that for line search purposes we work with with the related
     minimization problem:
        min  z(x):  R^n -> R,
     where z(x) = .5 * fnorm*fnorm, and fnorm = || f ||_2.
   */
        
  PetscReal   steptol,initslope,lambdaprev,gnormprev,a,b,d,t1,t2,rellength;
  PetscReal   maxstep,minlambda,alpha,lambda,lambdatemp,lambdaneg;
#if defined(PETSC_USE_COMPLEX)
  PetscScalar cinitslope,clambda;
#endif
  PetscErrorCode ierr;
  PetscInt count;
  SNES_LS     *neP = (SNES_LS*)snes->data;
  PetscScalar mone = -1.0,scale;
  PetscTruth  change_y = PETSC_FALSE;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(SNES_LineSearch,snes,x,f,g);CHKERRQ(ierr);
  *flag   = PETSC_TRUE;
  alpha   = neP->alpha;
  maxstep = neP->maxstep;
  steptol = neP->steptol;

  ierr = VecNorm(y,NORM_2,ynorm);CHKERRQ(ierr);
  if (*ynorm == 0.0) {
    PetscLogInfo(snes,"SNESCubicLineSearch: Search direction and size is 0\n");
    *gnorm = fnorm;
    ierr   = VecCopy(x,y);CHKERRQ(ierr);
    ierr   = VecCopy(f,g);CHKERRQ(ierr);
    *flag  = PETSC_FALSE;
    goto theend1;
  }
  if (*ynorm > maxstep) {	/* Step too big, so scale back */
    scale = maxstep/(*ynorm);
#if defined(PETSC_USE_COMPLEX)
    PetscLogInfo(snes,"SNESCubicLineSearch: Scaling step by %g old ynorm %g\n",PetscRealPart(scale),*ynorm);
#else
    PetscLogInfo(snes,"SNESCubicLineSearch: Scaling step by %g old ynorm %g\n",scale,*ynorm);
#endif
    ierr = VecScale(&scale,y);CHKERRQ(ierr);
    *ynorm = maxstep;
  }
  ierr      = VecMaxPointwiseDivide(y,x,&rellength);CHKERRQ(ierr);
  minlambda = steptol/rellength;
  ierr = MatMult(snes->jacobian,y,w);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = VecDot(f,w,&cinitslope);CHKERRQ(ierr);
  initslope = PetscRealPart(cinitslope);
#else
  ierr = VecDot(f,w,&initslope);CHKERRQ(ierr);
#endif
  if (initslope > 0.0) initslope = -initslope;
  if (initslope == 0.0) initslope = -1.0;

  ierr = VecCopy(y,w);CHKERRQ(ierr);
  ierr = VecAYPX(&mone,x,w);CHKERRQ(ierr);
  ierr = SNESComputeFunction(snes,w,g);CHKERRQ(ierr);
  ierr = VecNorm(g,NORM_2,gnorm);CHKERRQ(ierr);
  if (.5*(*gnorm)*(*gnorm) <= .5*fnorm*fnorm + alpha*initslope) { /* Sufficient reduction */
    ierr = VecCopy(w,y);CHKERRQ(ierr);
    PetscLogInfo(snes,"SNESCubicLineSearch: Using full step\n");
    goto theend1;
  }

  /* Fit points with quadratic */
  lambda = 1.0;
  lambdatemp = -initslope/((*gnorm)*(*gnorm) - fnorm*fnorm - 2.0*initslope);
  lambdaprev = lambda;
  gnormprev = *gnorm;
  if (lambdatemp > .5*lambda)  lambdatemp = .5*lambda;
  if (lambdatemp <= .1*lambda) lambda = .1*lambda; 
  else                         lambda = lambdatemp;
  ierr   = VecCopy(x,w);CHKERRQ(ierr);
  lambdaneg = -lambda;
#if defined(PETSC_USE_COMPLEX)
  clambda = lambdaneg; ierr = VecAXPY(&clambda,y,w);CHKERRQ(ierr);
#else
  ierr = VecAXPY(&lambdaneg,y,w);CHKERRQ(ierr);
#endif
  ierr = SNESComputeFunction(snes,w,g);CHKERRQ(ierr);
  ierr = VecNorm(g,NORM_2,gnorm);CHKERRQ(ierr);
  if (.5*(*gnorm)*(*gnorm) < .5*fnorm*fnorm + lambda*alpha*initslope) { /* sufficient reduction */
    ierr = VecCopy(w,y);CHKERRQ(ierr);
    PetscLogInfo(snes,"SNESCubicLineSearch: Quadratically determined step, lambda=%18.16e\n",lambda);
    goto theend1;
  }

  /* Fit points with cubic */
  count = 1;
  while (count < 10000) {
    if (lambda <= minlambda) { /* bad luck; use full step */
      PetscLogInfo(snes,"SNESCubicLineSearch:Unable to find good step length! %D \n",count);
      PetscLogInfo(snes,"SNESCubicLineSearch:fnorm=%18.16e, gnorm=%18.16e, ynorm=%18.16e, lambda=%18.16e, initial slope=%18.16e\n",fnorm,*gnorm,*ynorm,lambda,initslope);
      ierr = VecCopy(x,y);CHKERRQ(ierr);
      *flag = PETSC_FALSE; break;
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
      lambdatemp = (-b + sqrt(d))/(3.0*a);
    }
    lambdaprev = lambda;
    gnormprev  = *gnorm;
    if (lambdatemp > .5*lambda)  lambdatemp = .5*lambda;
    if (lambdatemp <= .1*lambda) lambda     = .1*lambda;
    else                         lambda     = lambdatemp;
    ierr = VecCopy(x,w);CHKERRQ(ierr);
    lambdaneg = -lambda;
#if defined(PETSC_USE_COMPLEX)
    clambda = lambdaneg;
    ierr = VecAXPY(&clambda,y,w);CHKERRQ(ierr);
#else
    ierr = VecAXPY(&lambdaneg,y,w);CHKERRQ(ierr);
#endif
    if (snes->nfuncs > snes->max_funcs) {
      PetscLogInfo(snes,"SNESCubicLineSearch:Exceeded maximum function evaluations, but unable to find good step length! %D \n",count);
      PetscLogInfo(snes,"SNESCubicLineSearch:fnorm=%18.16e, gnorm=%18.16e, ynorm=%18.16e, lambda=%18.16e, initial slope=%18.16e\n",fnorm,*gnorm,*ynorm,lambda,initslope);
      ierr = VecCopy(x,y);CHKERRQ(ierr);
      *flag = PETSC_FALSE;
      break;
    } else {
      ierr = SNESComputeFunction(snes,w,g);CHKERRQ(ierr);
    }
    ierr = VecNorm(g,NORM_2,gnorm);CHKERRQ(ierr);
    if (.5*(*gnorm)*(*gnorm) < .5*fnorm*fnorm + lambda*alpha*initslope) { /* is reduction enough? */
      ierr = VecCopy(w,y);CHKERRQ(ierr);
      PetscLogInfo(snes,"SNESCubicLineSearch: Cubically determined step, lambda=%18.16e\n",lambda);
      goto theend1;
    } else {
      PetscLogInfo(snes,"SNESCubicLineSearch: Cubic step no good, shrinking lambda,  lambda=%18.16e\n",lambda);
    }
    count++;
  }
  if (count >= 10000) {
    SETERRQ(PETSC_ERR_LIB, "Lambda was decreased more than 10,000 times, so something is probably wrong with the function evaluation");
  }
  theend1:
  /* Optional user-defined check for line search step validity */
  if (neP->CheckStep) {
    ierr = (*neP->CheckStep)(snes,neP->checkP,y,&change_y);CHKERRQ(ierr);
    if (change_y) { /* recompute the function if the step has changed */
      ierr = SNESComputeFunction(snes,y,g);CHKERRQ(ierr);
      ierr = VecNormBegin(y,NORM_2,ynorm);CHKERRQ(ierr);
      ierr = VecNormBegin(g,NORM_2,gnorm);CHKERRQ(ierr);
      ierr = VecNormEnd(y,NORM_2,ynorm);CHKERRQ(ierr);
      ierr = VecNormEnd(g,NORM_2,gnorm);CHKERRQ(ierr);
    }
  }
  ierr = PetscLogEventEnd(SNES_LineSearch,snes,x,f,g);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "SNESQuadraticLineSearch"
/*@C
   SNESQuadraticLineSearch - Performs a quadratic line search.

   Collective on SNES and Vec

   Input Parameters:
+  snes - the SNES context
.  lsctx - optional context for line search (not used here)
.  x - current iterate
.  f - residual evaluated at x
.  y - search direction (contains new iterate on output)
.  w - work vector
-  fnorm - 2-norm of f

   Output Parameters:
+  g - residual evaluated at new iterate y
.  y - new iterate (contains search direction on input)
.  gnorm - 2-norm of g
.  ynorm - 2-norm of search length
-  flag - PETSC_TRUE if line search succeeds; PETSC_FALSE on failure.

   Options Database Key:
.  -snes_ls quadratic - Activates SNESQuadraticLineSearch()

   Notes:
   Use SNESSetLineSearch() to set this routine within the SNESLS method.  

   Level: advanced

.keywords: SNES, nonlinear, quadratic, line search

.seealso: SNESCubicLineSearch(), SNESNoLineSearch(), SNESSetLineSearch(), SNESNoLineSearchNoNorms()
@*/
PetscErrorCode SNESQuadraticLineSearch(SNES snes,void *lsctx,Vec x,Vec f,Vec g,Vec y,Vec w,PetscReal fnorm,PetscReal *ynorm,PetscReal *gnorm,PetscTruth *flag)
{
  /* 
     Note that for line search purposes we work with with the related
     minimization problem:
        min  z(x):  R^n -> R,
     where z(x) = .5 * fnorm*fnorm,and fnorm = || f ||_2.
   */
  PetscReal   steptol,initslope,maxstep,minlambda,alpha,lambda,lambdatemp,lambdaneg,rellength;
#if defined(PETSC_USE_COMPLEX)
  PetscScalar cinitslope,clambda;
#endif
  PetscErrorCode ierr;
  PetscInt count;
  SNES_LS     *neP = (SNES_LS*)snes->data;
  PetscScalar mone = -1.0,scale;
  PetscTruth  change_y = PETSC_FALSE;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(SNES_LineSearch,snes,x,f,g);CHKERRQ(ierr);
  *flag   = PETSC_TRUE;
  alpha   = neP->alpha;
  maxstep = neP->maxstep;
  steptol = neP->steptol;

  ierr = VecNorm(y,NORM_2,ynorm);CHKERRQ(ierr);
  if (*ynorm == 0.0) {
    PetscLogInfo(snes,"SNESQuadraticLineSearch: Search direction and size is 0\n");
    *gnorm = fnorm;
    ierr   = VecCopy(x,y);CHKERRQ(ierr);
    ierr   = VecCopy(f,g);CHKERRQ(ierr);
    *flag  = PETSC_FALSE;
    goto theend2;
  }
  if (*ynorm > maxstep) {	/* Step too big, so scale back */
    scale = maxstep/(*ynorm);
    ierr = VecScale(&scale,y);CHKERRQ(ierr);
    *ynorm = maxstep;
  }
  ierr      = VecMaxPointwiseDivide(y,x,&rellength);CHKERRQ(ierr);
  minlambda = steptol/rellength;
  ierr = MatMult(snes->jacobian,y,w);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = VecDot(f,w,&cinitslope);CHKERRQ(ierr);
  initslope = PetscRealPart(cinitslope);
#else
  ierr = VecDot(f,w,&initslope);CHKERRQ(ierr);
#endif
  if (initslope > 0.0) initslope = -initslope;
  if (initslope == 0.0) initslope = -1.0;

  ierr = VecCopy(y,w);CHKERRQ(ierr);
  ierr = VecAYPX(&mone,x,w);CHKERRQ(ierr);
  ierr = SNESComputeFunction(snes,w,g);CHKERRQ(ierr);
  ierr = VecNorm(g,NORM_2,gnorm);CHKERRQ(ierr);
  if (.5*(*gnorm)*(*gnorm) <= .5*fnorm*fnorm + alpha*initslope) { /* Sufficient reduction */
    ierr = VecCopy(w,y);CHKERRQ(ierr);
    PetscLogInfo(snes,"SNESQuadraticLineSearch: Using full step\n");
    goto theend2;
  }

  /* Fit points with quadratic */
  lambda = 1.0;
  count = 1;
  while (PETSC_TRUE) {
    if (lambda <= minlambda) { /* bad luck; use full step */
      PetscLogInfo(snes,"SNESQuadraticLineSearch:Unable to find good step length! %D \n",count);
      PetscLogInfo(snes,"SNESQuadraticLineSearch:fnorm=%g, gnorm=%g, ynorm=%g, lambda=%g, initial slope=%g\n",fnorm,*gnorm,*ynorm,lambda,initslope);
      ierr = VecCopy(x,y);CHKERRQ(ierr);
      *flag = PETSC_FALSE; break;
    }
    lambdatemp = -initslope/((*gnorm)*(*gnorm) - fnorm*fnorm - 2.0*initslope);
    if (lambdatemp > .5*lambda)  lambdatemp = .5*lambda;
    if (lambdatemp <= .1*lambda) lambda     = .1*lambda; 
    else                         lambda     = lambdatemp;
    ierr = VecCopy(x,w);CHKERRQ(ierr);
    lambdaneg = -lambda;
#if defined(PETSC_USE_COMPLEX)
    clambda = lambdaneg; ierr = VecAXPY(&clambda,y,w);CHKERRQ(ierr);
#else
    ierr = VecAXPY(&lambdaneg,y,w);CHKERRQ(ierr);
#endif
    if (snes->nfuncs > snes->max_funcs) {
      PetscLogInfo(snes,"SNESCubicLineSearch:Exceeded maximum function evaluations, but unable to find good step length! %D \n",count);
      PetscLogInfo(snes,"SNESCubicLineSearch:fnorm=%18.16e, gnorm=%18.16e, ynorm=%18.16e, lambda=%18.16e, initial slope=%18.16e\n",fnorm,*gnorm,*ynorm,lambda,initslope);
      ierr = VecCopy(x,y);CHKERRQ(ierr);
      *flag = PETSC_FALSE;
      break;
    } else {
      ierr = SNESComputeFunction(snes,w,g);CHKERRQ(ierr);
    }
    ierr = VecNorm(g,NORM_2,gnorm);CHKERRQ(ierr);
    if (.5*(*gnorm)*(*gnorm) < .5*fnorm*fnorm + lambda*alpha*initslope) { /* sufficient reduction */
      ierr = VecCopy(w,y);CHKERRQ(ierr);
      PetscLogInfo(snes,"SNESQuadraticLineSearch:Quadratically determined step, lambda=%g\n",lambda);
      break;
    }
    count++;
  }
  theend2:
  /* Optional user-defined check for line search step validity */
  if (neP->CheckStep) {
    ierr = (*neP->CheckStep)(snes,neP->checkP,y,&change_y);CHKERRQ(ierr);
    if (change_y) { /* recompute the function if the step has changed */
      ierr = SNESComputeFunction(snes,y,g);CHKERRQ(ierr);
      ierr = VecNormBegin(y,NORM_2,ynorm);CHKERRQ(ierr);
      ierr = VecNormBegin(g,NORM_2,gnorm);CHKERRQ(ierr);
      ierr = VecNormEnd(y,NORM_2,ynorm);CHKERRQ(ierr);
      ierr = VecNormEnd(g,NORM_2,gnorm);CHKERRQ(ierr);
    }
  }
  ierr = PetscLogEventEnd(SNES_LineSearch,snes,x,f,g);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "SNESSetLineSearch"
/*@C
   SNESSetLineSearch - Sets the line search routine to be used
   by the method SNESLS.

   Input Parameters:
+  snes - nonlinear context obtained from SNESCreate()
.  lsctx - optional user-defined context for use by line search 
-  func - pointer to int function

   Collective on SNES

   Available Routines:
+  SNESCubicLineSearch() - default line search
.  SNESQuadraticLineSearch() - quadratic line search
.  SNESNoLineSearch() - the full Newton step (actually not a line search)
-  SNESNoLineSearchNoNorms() - the full Newton step (calculating no norms; faster in parallel)

    Options Database Keys:
+   -snes_ls [cubic,quadratic,basic,basicnonorms] - Selects line search
.   -snes_ls_alpha <alpha> - Sets alpha
.   -snes_ls_maxstep <max> - Sets maxstep
-   -snes_ls_steptol <steptol> - Sets steptol, this is the minimum step size that the line search code
                   will accept; min p[i]/x[i] < steptol. The -snes_stol <stol> is the minimum step length
                   the default convergence test will use and is based on 2-norm(p) < stol*2-norm(x)

   Calling sequence of func:
.vb
   func (SNES snes,void *lsctx,Vec x,Vec f,Vec g,Vec y,Vec w,
         PetscReal fnorm,PetscReal *ynorm,PetscReal *gnorm,PetscTruth *flag)
.ve

    Input parameters for func:
+   snes - nonlinear context
.   lsctx - optional user-defined context for line search
.   x - current iterate
.   f - residual evaluated at x
.   y - search direction (contains new iterate on output)
.   w - work vector
-   fnorm - 2-norm of f

    Output parameters for func:
+   g - residual evaluated at new iterate y
.   y - new iterate (contains search direction on input)
.   gnorm - 2-norm of g
.   ynorm - 2-norm of search length
-   flag - set to PETSC_TRUE if the line search succeeds; PETSC_FALSE on failure.

    Level: advanced

.keywords: SNES, nonlinear, set, line search, routine

.seealso: SNESCubicLineSearch(), SNESQuadraticLineSearch(), SNESNoLineSearch(), SNESNoLineSearchNoNorms(), 
          SNESSetLineSearchCheck(), SNESSetLineSearchParams(), SNESGetLineSearchParams()
@*/
PetscErrorCode SNESSetLineSearch(SNES snes,PetscErrorCode (*func)(SNES,void*,Vec,Vec,Vec,Vec,Vec,PetscReal,PetscReal*,PetscReal*,PetscTruth*),void *lsctx)
{
  PetscErrorCode ierr,(*f)(SNES,PetscErrorCode (*)(SNES,void*,Vec,Vec,Vec,Vec,Vec,PetscReal,PetscReal*,PetscReal*,PetscTruth*),void*);

  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)snes,"SNESSetLineSearch_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(snes,func,lsctx);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

typedef PetscErrorCode (*FCN2)(SNES,void*,Vec,Vec,Vec,Vec,Vec,PetscReal,PetscReal*,PetscReal*,PetscTruth*); /* force argument to next function to not be extern C*/
/* -------------------------------------------------------------------------- */
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "SNESSetLineSearch_LS"
PetscErrorCode SNESSetLineSearch_LS(SNES snes,FCN2 func,void *lsctx)
{
  PetscFunctionBegin;
  ((SNES_LS *)(snes->data))->LineSearch = func;
  ((SNES_LS *)(snes->data))->lsP        = lsctx;
  PetscFunctionReturn(0);
}
EXTERN_C_END
/* -------------------------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "SNESSetLineSearchCheck"
/*@C
   SNESSetLineSearchCheck - Sets a routine to check the validity of new iterate computed
   by the line search routine in the Newton-based method SNESLS.

   Input Parameters:
+  snes - nonlinear context obtained from SNESCreate()
.  func - pointer to int function
-  checkctx - optional user-defined context for use by step checking routine 

   Collective on SNES

   Calling sequence of func:
.vb
   int func (SNES snes, void *checkctx, Vec x, PetscTruth *flag)
.ve
   where func returns an error code of 0 on success and a nonzero
   on failure.

   Input parameters for func:
+  snes - nonlinear context
.  checkctx - optional user-defined context for use by step checking routine 
-  x - current candidate iterate

   Output parameters for func:
+  x - current iterate (possibly modified)
-  flag - flag indicating whether x has been modified (either
           PETSC_TRUE of PETSC_FALSE)

   Level: advanced

   Notes:
   SNESNoLineSearch() and SNESNoLineSearchNoNorms() accept the new
   iterate computed by the line search checking routine.  In particular,
   these routines (1) compute a candidate iterate u_{i+1}, (2) pass control 
   to the checking routine, and then (3) compute the corresponding nonlinear
   function f(u_{i+1}) with the (possibly altered) iterate u_{i+1}.

   SNESQuadraticLineSearch() and SNESCubicLineSearch() also accept the
   new iterate computed by the line search checking routine.  In particular,
   these routines (1) compute a candidate iterate u_{i+1} as well as a
   candidate nonlinear function f(u_{i+1}), (2) pass control to the checking 
   routine, and then (3) force a re-evaluation of f(u_{i+1}) if any changes 
   were made to the candidate iterate in the checking routine (as indicated 
   by flag=PETSC_TRUE).  The overhead of this function re-evaluation can be
   very costly, so use this feature with caution!

.keywords: SNES, nonlinear, set, line search check, step check, routine

.seealso: SNESSetLineSearch()
@*/
PetscErrorCode SNESSetLineSearchCheck(SNES snes,PetscErrorCode (*func)(SNES,void*,Vec,PetscTruth*),void *checkctx)
{
  PetscErrorCode ierr,(*f)(SNES,PetscErrorCode (*)(SNES,void*,Vec,PetscTruth*),void*);

  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)snes,"SNESSetLineSearchCheck_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(snes,func,checkctx);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
typedef PetscErrorCode (*FCN)(SNES,void*,Vec,PetscTruth*); /* force argument to next function to not be extern C*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "SNESSetLineSearchCheck_LS"
PetscErrorCode SNESSetLineSearchCheck_LS(SNES snes,FCN func,void *checkctx)
{
  PetscFunctionBegin;
  ((SNES_LS *)(snes->data))->CheckStep = func;
  ((SNES_LS *)(snes->data))->checkP    = checkctx;
  PetscFunctionReturn(0);
}
EXTERN_C_END
/* -------------------------------------------------------------------------- */
/*
   SNESPrintHelp_LS - Prints all options for the SNES_LS method.

   Input Parameter:
.  snes - the SNES context

   Application Interface Routine: SNESPrintHelp()
*/
#undef __FUNCT__  
#define __FUNCT__ "SNESPrintHelp_LS"
static PetscErrorCode SNESPrintHelp_LS(SNES snes,char *p)
{
  SNES_LS *ls = (SNES_LS *)snes->data;

  PetscFunctionBegin;
  (*PetscHelpPrintf)(snes->comm," method SNES_LS (ls) for systems of nonlinear equations:\n");
  (*PetscHelpPrintf)(snes->comm,"   %ssnes_ls [cubic,quadratic,basic,basicnonorms,...]\n",p);
  (*PetscHelpPrintf)(snes->comm,"   %ssnes_ls_alpha <alpha> (default %g)\n",p,ls->alpha);
  (*PetscHelpPrintf)(snes->comm,"   %ssnes_ls_maxstep <max> (default %g)\n",p,ls->maxstep);
  (*PetscHelpPrintf)(snes->comm,"   %ssnes_ls_steptol <tol> (default %g)\n",p,ls->steptol);
  PetscFunctionReturn(0);
}

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
  SNES_LS    *ls = (SNES_LS *)snes->data;
  const char *cstr;
  PetscErrorCode ierr;
  PetscTruth iascii;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    if (ls->LineSearch == SNESNoLineSearch)             cstr = "SNESNoLineSearch";
    else if (ls->LineSearch == SNESQuadraticLineSearch) cstr = "SNESQuadraticLineSearch";
    else if (ls->LineSearch == SNESCubicLineSearch)     cstr = "SNESCubicLineSearch";
    else                                                cstr = "unknown";
    ierr = PetscViewerASCIIPrintf(viewer,"  line search variant: %s\n",cstr);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  alpha=%g, maxstep=%g, steptol=%g\n",ls->alpha,ls->maxstep,ls->steptol);CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported for SNES EQ LS",((PetscObject)viewer)->type_name);
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
  SNES_LS    *ls = (SNES_LS *)snes->data;
  const char *lses[] = {"basic","basicnonorms","quadratic","cubic"};
  PetscErrorCode ierr;
  PetscInt indx;
  PetscTruth flg;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("SNES Line search options");CHKERRQ(ierr);
    ierr = PetscOptionsReal("-snes_ls_alpha","Function norm must decrease by","None",ls->alpha,&ls->alpha,0);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-snes_ls_maxstep","Step must be less than","None",ls->maxstep,&ls->maxstep,0);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-snes_ls_steptol","Step must be greater than","None",ls->steptol,&ls->steptol,0);CHKERRQ(ierr);

    ierr = PetscOptionsEList("-snes_ls","Line search used","SNESSetLineSearch",lses,4,"cubic",&indx,&flg);CHKERRQ(ierr);
    if (flg) {
      switch (indx) {
      case 0:
        ierr = SNESSetLineSearch(snes,SNESNoLineSearch,PETSC_NULL);CHKERRQ(ierr);
        break;
      case 1:
        ierr = SNESSetLineSearch(snes,SNESNoLineSearchNoNorms,PETSC_NULL);CHKERRQ(ierr);
        break;
      case 2:
        ierr = SNESSetLineSearch(snes,SNESQuadraticLineSearch,PETSC_NULL);CHKERRQ(ierr);
        break;
      case 3:
        ierr = SNESSetLineSearch(snes,SNESCubicLineSearch,PETSC_NULL);CHKERRQ(ierr);
        break;
      }
    }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
/*MC
      SNESLS - Newton based nonlinear solver that uses a line search

   Options Database:
+   -snes_ls [cubic,quadratic,basic,basicnonorms] - Selects line search
.   -snes_ls_alpha <alpha> - Sets alpha
.   -snes_ls_maxstep <max> - Sets maxstep
-   -snes_ls_steptol <steptol> - Sets steptol, this is the minimum step size that the line search code
                   will accept; min p[i]/x[i] < steptol. The -snes_stol <stol> is the minimum step length
                   the default convergence test will use and is based on 2-norm(p) < stol*2-norm(x)

    Notes: This is the default nonlinear solver in SNES

   Level: beginner

.seealso:  SNESCreate(), SNES, SNESSetType(), SNESTR, SNESSetLineSearch(), 
           SNESSetLineSearchCheck(), SNESNoLineSearch(), SNESCubicLineSearch(), SNESQuadraticLineSearch(), 
          SNESSetLineSearch(), SNESNoLineSearchNoNorms()

M*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "SNESCreate_LS"
PetscErrorCode SNESCreate_LS(SNES snes)
{
  PetscErrorCode ierr;
  SNES_LS *neP;

  PetscFunctionBegin;
  snes->setup		= SNESSetUp_LS;
  snes->solve		= SNESSolve_LS;
  snes->destroy		= SNESDestroy_LS;
  snes->converged	= SNESConverged_LS;
  snes->printhelp       = SNESPrintHelp_LS;
  snes->setfromoptions  = SNESSetFromOptions_LS;
  snes->view            = SNESView_LS;
  snes->nwork           = 0;

  ierr                  = PetscNew(SNES_LS,&neP);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(snes,sizeof(SNES_LS));CHKERRQ(ierr);
  snes->data    	= (void*)neP;
  neP->alpha		= 1.e-4;
  neP->maxstep		= 1.e8;
  neP->steptol		= 1.e-12;
  neP->LineSearch       = SNESCubicLineSearch;
  neP->lsP              = PETSC_NULL;
  neP->CheckStep        = PETSC_NULL;
  neP->checkP           = PETSC_NULL;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)snes,"SNESSetLineSearch_C","SNESSetLineSearch_LS",SNESSetLineSearch_LS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)snes,"SNESSetLineSearchCheck_C","SNESSetLineSearchCheck_LS",SNESSetLineSearchCheck_LS);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END



