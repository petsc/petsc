#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ls.c,v 1.124 1999/03/01 04:49:15 bsmith Exp curfman $";
#endif

#include "src/snes/impls/ls/ls.h"

/*  -------------------------------------------------------------------- 

     This file implements a truncated Newton method with a line search,
     for solving a system of nonlinear equations, using the SLES, Vec, 
     and Mat interfaces for linear solvers, vectors, and matrices, 
     respectively.

     The following basic routines are required for each nonlinear solver:
          SNESCreate_XXX()          - Creates a nonlinear solver context
          SNESSetFromOptions_XXX()  - Sets runtime options
          SNESSolve_XXX()           - Solves the nonlinear system
          SNESDestroy_XXX()         - Destroys the nonlinear solver context
     The suffix "_XXX" denotes a particular implementation, in this case
     we use _EQ_LS (e.g., SNESCreate_EQ_LS, SNESSolve_EQ_LS) for solving
     systems of nonlinear equations (EQ) with a line search (LS) method.
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
          SNESPrintHelp_XXX()       - Prints nonlinear solver runtime options
          SNESView_XXX()            - Prints details of runtime options that
                                      have actually been used.
     These are called by application codes via the interface routines
     SNESPrintHelp() and SNESView().

     The various types of solvers (preconditioners, Krylov subspace methods,
     nonlinear solvers, timesteppers) are all organized similarly, so the
     above description applies to these categories also.  

    -------------------------------------------------------------------- */
/*
   SNESSolve_EQ_LS - Solves a nonlinear system with a truncated Newton
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
#undef __FUNC__  
#define __FUNC__ "SNESSolve_EQ_LS"
int SNESSolve_EQ_LS(SNES snes,int *outits)
{
  SNES_LS       *neP = (SNES_LS *) snes->data;
  int           maxits, i, ierr, lits, lsfail;
  MatStructure  flg = DIFFERENT_NONZERO_PATTERN;
  double        fnorm, gnorm, xnorm, ynorm;
  Vec           Y, X, F, G, W, TMP;

  PetscFunctionBegin;
  maxits	= snes->max_its;	/* maximum number of iterations */
  X		= snes->vec_sol;	/* solution vector */
  F		= snes->vec_func;	/* residual vector */
  Y		= snes->work[0];	/* work vectors */
  G		= snes->work[1];
  W		= snes->work[2];

  ierr = SNESComputeFunction(snes,X,F); CHKERRQ(ierr);  /*  F(X)      */
  ierr = VecNorm(F,NORM_2,&fnorm); CHKERRQ(ierr);	/* fnorm <- ||F||  */
  PetscAMSTakeAccess(snes);
  snes->iter = 0;
  snes->norm = fnorm;
  PetscAMSGrantAccess(snes);
  SNESLogConvHistory(snes,fnorm,0);
  SNESMonitor(snes,0,fnorm);

  if (fnorm < snes->atol) {*outits = 0; PetscFunctionReturn(0);}

  /* set parameter for default relative tolerance convergence test */
  snes->ttol = fnorm*snes->rtol;

  for ( i=0; i<maxits; i++ ) {

    /* Solve J Y = F, where J is Jacobian matrix */
    ierr = SNESComputeJacobian(snes,X,&snes->jacobian,&snes->jacobian_pre,&flg); CHKERRQ(ierr);
    ierr = SLESSetOperators(snes->sles,snes->jacobian,snes->jacobian_pre,flg); CHKERRQ(ierr);
    ierr = SLESSolve(snes->sles,F,Y,&lits); CHKERRQ(ierr);
    snes->linear_its += PetscAbsInt(lits);
    PLogInfo(snes,"SNESSolve_EQ_LS: iter=%d, linear solve iterations=%d\n",snes->iter,lits);

    /* Compute a (scaled) negative update in the line search routine: 
         Y <- X - lambda*Y 
       and evaluate G(Y) = function(Y)) 
    */
    ierr = VecCopy(Y,snes->vec_sol_update_always); CHKERRQ(ierr);
    ierr = (*neP->LineSearch)(snes,neP->lsP,X,F,G,Y,W,fnorm,&ynorm,&gnorm,&lsfail); CHKERRQ(ierr);
    PLogInfo(snes,"SNESSolve_EQ_LS: fnorm=%g, gnorm=%g, ynorm=%g, lsfail=%d\n",fnorm,gnorm,ynorm,lsfail);
    if (lsfail) snes->nfailures++;

    TMP = F; F = G; snes->vec_func_always = F; G = TMP;
    TMP = X; X = Y; snes->vec_sol_always = X;  Y = TMP;
    fnorm = gnorm;

    PetscAMSTakeAccess(snes);
    snes->iter = i+1;
    snes->norm = fnorm;
    PetscAMSGrantAccess(snes);
    SNESLogConvHistory(snes,fnorm,lits);
    SNESMonitor(snes,i+1,fnorm);

    /* Test for convergence */
    if (snes->converged) {
      ierr = VecNorm(X,NORM_2,&xnorm); CHKERRQ(ierr);	/* xnorm = || X || */
      if ((*snes->converged)(snes,xnorm,ynorm,fnorm,snes->cnvP)) {
        break;
      }
    }
  }
  if (X != snes->vec_sol) {
    ierr = VecCopy(X,snes->vec_sol); CHKERRQ(ierr);
    snes->vec_sol_always  = snes->vec_sol;
    snes->vec_func_always = snes->vec_func;
  }
  if (i == maxits) {
    PLogInfo(snes,"SNESSolve_EQ_LS: Maximum number of iterations has been reached: %d\n",maxits);
    i--;
  }
  *outits = i+1;
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
/*
   SNESSetUp_EQ_LS - Sets up the internal data structures for the later use
   of the SNES_EQ_LS nonlinear solver.

   Input Parameter:
.  snes - the SNES context
.  x - the solution vector

   Application Interface Routine: SNESSetUp()

   Notes:
   For basic use of the SNES solvers the user need not explicitly call
   SNESSetUp(), since these actions will automatically occur during
   the call to SNESSolve().
 */
#undef __FUNC__  
#define __FUNC__ "SNESSetUp_EQ_LS"
int SNESSetUp_EQ_LS(SNES snes)
{
  int ierr;

  PetscFunctionBegin;
  snes->nwork = 4;
  ierr = VecDuplicateVecs(snes->vec_sol,snes->nwork,&snes->work);CHKERRQ(ierr);
  PLogObjectParents(snes,snes->nwork,snes->work);
  snes->vec_sol_update_always = snes->work[3];
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
/*
   SNESDestroy_EQ_LS - Destroys the private SNES_LS context that was created
   with SNESCreate_EQ_LS().

   Input Parameter:
.  snes - the SNES context

   Application Interface Routine: SNESDestroy()
 */
#undef __FUNC__  
#define __FUNC__ "SNESDestroy_EQ_LS"
int SNESDestroy_EQ_LS(SNES snes)
{
  int  ierr;

  PetscFunctionBegin;
  if (snes->nwork) {
    ierr = VecDestroyVecs(snes->work,snes->nwork); CHKERRQ(ierr);
  }
  PetscFree(snes->data);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
#undef __FUNC__  
#define __FUNC__ "SNESNoLineSearch"

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
-  flag - set to 0, indicating a successful line search

   Options Database Key:
.  -snes_eq_ls basic - Activates SNESNoLineSearch()

   Level: advanced

.keywords: SNES, nonlinear, line search, cubic

.seealso: SNESCubicLineSearch(), SNESQuadraticLineSearch(), 
          SNESSetLineSearch(), SNESNoLineSearchNoNorms()
@*/
int SNESNoLineSearch(SNES snes, void *lsctx, Vec x, Vec f, Vec g, Vec y, Vec w,
                     double fnorm, double *ynorm, double *gnorm,int *flag )
{
  int    ierr;
  Scalar mone = -1.0;

  PetscFunctionBegin;
  *flag = 0;
  PLogEventBegin(SNES_LineSearch,snes,x,f,g);
  ierr = VecNorm(y,NORM_2,ynorm); CHKERRQ(ierr);       /* ynorm = || y || */
  ierr = VecAYPX(&mone,x,y); CHKERRQ(ierr);            /* y <- y - x      */
  ierr = SNESComputeFunction(snes,y,g); CHKERRQ(ierr); /* Compute F(y)    */
  ierr = VecNorm(g,NORM_2,gnorm); CHKERRQ(ierr);       /* gnorm = || g || */
  PLogEventEnd(SNES_LineSearch,snes,x,f,g);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
#undef __FUNC__  
#define __FUNC__ "SNESNoLineSearchNoNorms"

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
-  flag - set to 0, indicating a successful line search

   Options Database Key:
.  -snes_eq_ls basicnonorms - Activates SNESNoLineSearchNoNorms()

   Level: advanced

.keywords: SNES, nonlinear, line search, cubic

.seealso: SNESCubicLineSearch(), SNESQuadraticLineSearch(), 
          SNESSetLineSearch(), SNESNoLineSearch()
@*/
int SNESNoLineSearchNoNorms(SNES snes, void *lsctx, Vec x, Vec f, Vec g, Vec y, Vec w,
                     double fnorm, double *ynorm, double *gnorm,int *flag )
{
  int    ierr;
  Scalar mone = -1.0;

  PetscFunctionBegin;
  *flag = 0;
  PLogEventBegin(SNES_LineSearch,snes,x,f,g);
  ierr = VecAYPX(&mone,x,y); CHKERRQ(ierr);            /* y <- y - x      */
  ierr = SNESComputeFunction(snes,y,g); CHKERRQ(ierr); /* Compute F(y)    */
  PLogEventEnd(SNES_LineSearch,snes,x,f,g);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
#undef __FUNC__  
#define __FUNC__ "SNESCubicLineSearch"
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
-  flag - 0 if line search succeeds; -1 on failure.

   Options Database Key:
$  -snes_eq_ls cubic - Activates SNESCubicLineSearch()

   Notes:
   This line search is taken from "Numerical Methods for Unconstrained 
   Optimization and Nonlinear Equations" by Dennis and Schnabel, page 325.

   Level: advanced

.keywords: SNES, nonlinear, line search, cubic

.seealso: SNESQuadraticLineSearch(), SNESNoLineSearch(), SNESSetLineSearch(), SNESNoLineSearchNoNorms()
@*/
int SNESCubicLineSearch(SNES snes,void *lsctx,Vec x,Vec f,Vec g,Vec y,Vec w,
                        double fnorm,double *ynorm,double *gnorm,int *flag)
{
  /* 
     Note that for line search purposes we work with with the related
     minimization problem:
        min  z(x):  R^n -> R,
     where z(x) = .5 * fnorm*fnorm, and fnorm = || f ||_2.
   */
        
  double  steptol, initslope, lambdaprev, gnormprev, a, b, d, t1, t2;
  double  maxstep, minlambda, alpha, lambda, lambdatemp, lambdaneg;
#if defined(USE_PETSC_COMPLEX)
  Scalar  cinitslope, clambda;
#endif
  int     ierr, count;
  SNES_LS *neP = (SNES_LS *) snes->data;
  Scalar  mone = -1.0,scale;

  PetscFunctionBegin;
  PLogEventBegin(SNES_LineSearch,snes,x,f,g);
  *flag   = 0;
  alpha   = neP->alpha;
  maxstep = neP->maxstep;
  steptol = neP->steptol;

  ierr = VecNorm(y,NORM_2,ynorm); CHKERRQ(ierr);
  if (*ynorm < snes->atol) {
    PLogInfo(snes,"SNESCubicLineSearch: Search direction and size are nearly 0\n");
    *gnorm = fnorm;
    ierr = VecCopy(x,y); CHKERRQ(ierr);
    ierr = VecCopy(f,g); CHKERRQ(ierr);
    goto theend1;
  }
  if (*ynorm > maxstep) {	/* Step too big, so scale back */
    scale = maxstep/(*ynorm);
#if defined(USE_PETSC_COMPLEX)
    PLogInfo(snes,"SNESCubicLineSearch: Scaling step by %g\n",PetscReal(scale));
#else
    PLogInfo(snes,"SNESCubicLineSearch: Scaling step by %g\n",scale);
#endif
    ierr = VecScale(&scale,y); CHKERRQ(ierr);
    *ynorm = maxstep;
  }
  minlambda = steptol/(*ynorm);
  ierr = MatMult(snes->jacobian,y,w); CHKERRQ(ierr);
#if defined(USE_PETSC_COMPLEX)
  ierr = VecDot(f,w,&cinitslope); CHKERRQ(ierr);
  initslope = PetscReal(cinitslope);
#else
  ierr = VecDot(f,w,&initslope); CHKERRQ(ierr);
#endif
  if (initslope > 0.0) initslope = -initslope;
  if (initslope == 0.0) initslope = -1.0;

  ierr = VecCopy(y,w); CHKERRQ(ierr);
  ierr = VecAYPX(&mone,x,w); CHKERRQ(ierr);
  ierr = SNESComputeFunction(snes,w,g); CHKERRQ(ierr);
  ierr = VecNorm(g,NORM_2,gnorm); 
  if ((*gnorm)*(*gnorm)*0.5 <= fnorm*fnorm*0.5 + alpha*initslope) { /* Sufficient reduction */
    ierr = VecCopy(w,y); CHKERRQ(ierr);
    PLogInfo(snes,"SNESCubicLineSearch: Using full step\n");
    goto theend1;
  }

  /* Fit points with quadratic */
  lambda = 1.0; count = 0;
  lambdatemp = -initslope/((*gnorm)*(*gnorm) - fnorm*fnorm - 2.0*initslope);
  lambdaprev = lambda;
  gnormprev = *gnorm;
  if (lambdatemp <= .1*lambda) lambda = .1*lambda; 
  else lambda = lambdatemp;
  ierr   = VecCopy(x,w); CHKERRQ(ierr);
  lambdaneg = -lambda;
#if defined(USE_PETSC_COMPLEX)
  clambda = lambdaneg; ierr = VecAXPY(&clambda,y,w); CHKERRQ(ierr);
#else
  ierr = VecAXPY(&lambdaneg,y,w); CHKERRQ(ierr);
#endif
  ierr = SNESComputeFunction(snes,w,g); CHKERRQ(ierr);
  ierr = VecNorm(g,NORM_2,gnorm); CHKERRQ(ierr);
  if ((*gnorm)*(*gnorm)*0.5 <= fnorm*fnorm*0.5 + alpha*initslope) { /* sufficient reduction */
    ierr = VecCopy(w,y); CHKERRQ(ierr);
    PLogInfo(snes,"SNESCubicLineSearch: Quadratically determined step, lambda=%g\n",lambda);
    goto theend1;
  }

  /* Fit points with cubic */
  count = 1;
  while (1) {
    if (lambda <= minlambda) { /* bad luck; use full step */
      PLogInfo(snes,"SNESCubicLineSearch:Unable to find good step length! %d \n",count);
      PLogInfo(snes,"SNESCubicLineSearch:fnorm=%g, gnorm=%g, ynorm=%g, lambda=%g, initial slope=%g\n",
               fnorm,*gnorm,*ynorm,lambda,initslope);
      ierr = VecCopy(w,y); CHKERRQ(ierr);
      *flag = -1; break;
    }
    t1 = ((*gnorm)*(*gnorm) - fnorm*fnorm)*0.5 - lambda*initslope;
    t2 = (gnormprev*gnormprev  - fnorm*fnorm)*0.5 - lambdaprev*initslope;
    a  = (t1/(lambda*lambda) - t2/(lambdaprev*lambdaprev))/(lambda-lambdaprev);
    b  = (-lambdaprev*t1/(lambda*lambda) + lambda*t2/(lambdaprev*lambdaprev))/(lambda-lambdaprev);
    d  = b*b - 3*a*initslope;
    if (d < 0.0) d = 0.0;
    if (a == 0.0) {
      lambdatemp = -initslope/(2.0*b);
    } else {
      lambdatemp = (-b + sqrt(d))/(3.0*a);
    }
    if (lambdatemp > .5*lambda) {
      lambdatemp = .5*lambda;
    }
    lambdaprev = lambda;
    gnormprev = *gnorm;
    if (lambdatemp <= .1*lambda) {
      lambda = .1*lambda;
    }
    else lambda = lambdatemp;
    ierr = VecCopy( x, w ); CHKERRQ(ierr);
    lambdaneg = -lambda;
#if defined(USE_PETSC_COMPLEX)
    clambda = lambdaneg;
    ierr = VecAXPY(&clambda,y,w); CHKERRQ(ierr);
#else
    ierr = VecAXPY(&lambdaneg,y,w); CHKERRQ(ierr);
#endif
    ierr = SNESComputeFunction(snes,w,g); CHKERRQ(ierr);
    ierr = VecNorm(g,NORM_2,gnorm); CHKERRQ(ierr);
    if ((*gnorm)*(*gnorm)*0.5 <= fnorm*fnorm*0.5 + alpha*initslope) { /* is reduction enough? */
      ierr = VecCopy(w,y); CHKERRQ(ierr);
      PLogInfo(snes,"SNESCubicLineSearch: Cubically determined step, lambda=%g\n",lambda);
      goto theend1;
    } else {
      PLogInfo(snes,"SNESCubicLineSearch: Cubic step no good, shrinking lambda,  lambda=%g\n",lambda);
    }
    count++;
  }
  theend1:
  PLogEventEnd(SNES_LineSearch,snes,x,f,g);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
#undef __FUNC__  
#define __FUNC__ "SNESQuadraticLineSearch"
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
-  flag - 0 if line search succeeds; -1 on failure.

   Options Database Key:
.  -snes_eq_ls quadratic - Activates SNESQuadraticLineSearch()

   Notes:
   Use SNESSetLineSearch() to set this routine within the SNES_EQ_LS method.  

   Level: advanced

.keywords: SNES, nonlinear, quadratic, line search

.seealso: SNESCubicLineSearch(), SNESNoLineSearch(), SNESSetLineSearch(), SNESNoLineSearchNoNorms()
@*/
int SNESQuadraticLineSearch(SNES snes, void *lsctx, Vec x, Vec f, Vec g, Vec y, Vec w,
                           double fnorm, double *ynorm, double *gnorm,int *flag)
{
  /* 
     Note that for line search purposes we work with with the related
     minimization problem:
        min  z(x):  R^n -> R,
     where z(x) = .5 * fnorm*fnorm, and fnorm = || f ||_2.
   */
  double  steptol,initslope,maxstep,minlambda,alpha,lambda,lambdatemp;
#if defined(USE_PETSC_COMPLEX)
  Scalar  cinitslope,clambda;
#endif
  int     ierr,count;
  SNES_LS *neP = (SNES_LS *) snes->data;
  Scalar  mone = -1.0,scale;

  PetscFunctionBegin;
  PLogEventBegin(SNES_LineSearch,snes,x,f,g);
  *flag   = 0;
  alpha   = neP->alpha;
  maxstep = neP->maxstep;
  steptol = neP->steptol;

  VecNorm(y, NORM_2,ynorm );
  if (*ynorm < snes->atol) {
    PLogInfo(snes,"SNESQuadraticLineSearch: Search direction and size is 0\n");
    *gnorm = fnorm;
    ierr = VecCopy(x,y); CHKERRQ(ierr);
    ierr = VecCopy(f,g); CHKERRQ(ierr);
    goto theend2;
  }
  if (*ynorm > maxstep) {	/* Step too big, so scale back */
    scale = maxstep/(*ynorm);
    ierr = VecScale(&scale,y); CHKERRQ(ierr);
    *ynorm = maxstep;
  }
  minlambda = steptol/(*ynorm);
  ierr = MatMult(snes->jacobian,y,w); CHKERRQ(ierr);
#if defined(USE_PETSC_COMPLEX)
  ierr = VecDot(f,w,&cinitslope); CHKERRQ(ierr);
  initslope = PetscReal(cinitslope);
#else
  ierr = VecDot(f,w,&initslope); CHKERRQ(ierr);
#endif
  if (initslope > 0.0) initslope = -initslope;
  if (initslope == 0.0) initslope = -1.0;

  ierr = VecCopy(y,w); CHKERRQ(ierr);
  ierr = VecAYPX(&mone,x,w); CHKERRQ(ierr);
  ierr = SNESComputeFunction(snes,w,g); CHKERRQ(ierr);
  ierr = VecNorm(g,NORM_2,gnorm); CHKERRQ(ierr);
  if ((*gnorm)*(*gnorm)*0.5 <= fnorm*fnorm*0.5 + alpha*initslope) { /* Sufficient reduction */
    ierr = VecCopy(w,y); CHKERRQ(ierr);
    PLogInfo(snes,"SNESQuadraticLineSearch: Using full step\n");
    goto theend2;
  }

  /* Fit points with quadratic */
  lambda = 1.0; count = 0;
  count = 1;
  while (1) {
    if (lambda <= minlambda) { /* bad luck; use full step */
      PLogInfo(snes,"SNESQuadraticLineSearch:Unable to find good step length! %d \n",count);
      PLogInfo(snes,"SNESQuadraticLineSearch:fnorm=%g, gnorm=%g, ynorm=%g, lambda=%g, initial slope=%g\n",
               fnorm,*gnorm,*ynorm,lambda,initslope);
      ierr = VecCopy(w,y); CHKERRQ(ierr);
      *flag = -1; break;
    }
    lambdatemp = -initslope/((*gnorm)*(*gnorm) - fnorm*fnorm - 2.0*initslope);
    if (lambdatemp <= .1*lambda) { 
      lambda = .1*lambda; 
    } else lambda = lambdatemp;
    ierr = VecCopy(x,w); CHKERRQ(ierr);
    lambda = -lambda;
#if defined(USE_PETSC_COMPLEX)
    clambda = lambda; ierr = VecAXPY(&clambda,y,w); CHKERRQ(ierr);
#else
    ierr = VecAXPY(&lambda,y,w); CHKERRQ(ierr);
#endif
    ierr = SNESComputeFunction(snes,w,g); CHKERRQ(ierr);
    ierr = VecNorm(g,NORM_2,gnorm); CHKERRQ(ierr);
    if ((*gnorm)*(*gnorm)*0.5 <= fnorm*fnorm*0.5 + alpha*initslope) { /* sufficient reduction */
      ierr = VecCopy(w,y); CHKERRQ(ierr);
      PLogInfo(snes,"SNESQuadraticLineSearch:Quadratically determined step, lambda=%g\n",lambda);
      break;
    }
    count++;
  }
  theend2:
  PLogEventEnd(SNES_LineSearch,snes,x,f,g);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
#undef __FUNC__  
#define __FUNC__ "SNESSetLineSearch"
/*@C
   SNESSetLineSearch - Sets the line search routine to be used
   by the method SNES_EQ_LS.

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
+   -snes_eq_ls [cubic,quadratic,basic,basicnonorms] - Selects line search
.   -snes_eq_ls_alpha <alpha> - Sets alpha
.   -snes_eq_ls_maxstep <max> - Sets maxstep
-   -snes_eq_ls_steptol <steptol> - Sets steptol

   Calling sequence of func:
.vb
   func (SNES snes, void *lsctx, Vec x, Vec f, Vec g, Vec y, Vec w,
         double fnorm, double *ynorm, double *gnorm, *flag)
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
-   flag - set to 0 if the line search succeeds; a nonzero integer 
           on failure.

    Level: advanced

.keywords: SNES, nonlinear, set, line search, routine

.seealso: SNESCubicLineSearch(), SNESQuadraticLineSearch(), SNESNoLineSearch(), SNESNoLineSearchNoNorms()
@*/
int SNESSetLineSearch(SNES snes,int (*func)(SNES,void*,Vec,Vec,Vec,Vec,Vec,double,double*,double*,int*),void *lsctx)
{
  int ierr, (*f)(SNES,int (*)(SNES,void*,Vec,Vec,Vec,Vec,Vec,double,double*,double*,int*),void*);

  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)snes,"SNESSetLineSearch_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(snes,func,lsctx);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "SNESSetLineSearch_LS"
int SNESSetLineSearch_LS(SNES snes,int (*func)(SNES,void*,Vec,Vec,Vec,Vec,Vec,
                         double,double*,double*,int*),void *lsctx)
{
  PetscFunctionBegin;
  ((SNES_LS *)(snes->data))->LineSearch = func;
  ((SNES_LS *)(snes->data))->lsP = lsctx;
  PetscFunctionReturn(0);
}
EXTERN_C_END
/* -------------------------------------------------------------------------- */
/*
   SNESPrintHelp_EQ_LS - Prints all options for the SNES_EQ_LS method.

   Input Parameter:
.  snes - the SNES context

   Application Interface Routine: SNESPrintHelp()
*/
#undef __FUNC__  
#define __FUNC__ "SNESPrintHelp_EQ_LS"
static int SNESPrintHelp_EQ_LS(SNES snes,char *p)
{
  SNES_LS *ls = (SNES_LS *)snes->data;

  PetscFunctionBegin;
  (*PetscHelpPrintf)(snes->comm," method SNES_EQ_LS (ls) for systems of nonlinear equations:\n");
  (*PetscHelpPrintf)(snes->comm,"   %ssnes_eq_ls [cubic,quadratic,basic,basicnonorms,...]\n",p);
  (*PetscHelpPrintf)(snes->comm,"   %ssnes_eq_ls_alpha <alpha> (default %g)\n",p,ls->alpha);
  (*PetscHelpPrintf)(snes->comm,"   %ssnes_eq_ls_maxstep <max> (default %g)\n",p,ls->maxstep);
  (*PetscHelpPrintf)(snes->comm,"   %ssnes_eq_ls_steptol <tol> (default %g)\n",p,ls->steptol);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
/*
   SNESView_EQ_LS - Prints info from the SNES_EQ_LS data structure.

   Input Parameters:
.  SNES - the SNES context
.  viewer - visualization context

   Application Interface Routine: SNESView()
*/
#undef __FUNC__  
#define __FUNC__ "SNESView_EQ_LS"
static int SNESView_EQ_LS(SNES snes,Viewer viewer)
{
  SNES_LS    *ls = (SNES_LS *)snes->data;
  char       *cstr;
  int        ierr;
  ViewerType vtype;

  PetscFunctionBegin;
  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  if (PetscTypeCompare(vtype,ASCII_VIEWER)) {
    if (ls->LineSearch == SNESNoLineSearch)             cstr = "SNESNoLineSearch";
    else if (ls->LineSearch == SNESQuadraticLineSearch) cstr = "SNESQuadraticLineSearch";
    else if (ls->LineSearch == SNESCubicLineSearch)     cstr = "SNESCubicLineSearch";
    else                                                cstr = "unknown";
    ViewerASCIIPrintf(viewer,"  line search variant: %s\n",cstr);
    ViewerASCIIPrintf(viewer,"  alpha=%g, maxstep=%g, steptol=%g\n",ls->alpha,ls->maxstep,ls->steptol);
  } else {
    SETERRQ(1,1,"Viewer type not supported for this object");
  }
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
/*
   SNESSetFromOptions_EQ_LS - Sets various parameters for the SNES_EQ_LS method.

   Input Parameter:
.  snes - the SNES context

   Application Interface Routine: SNESSetFromOptions()
*/
#undef __FUNC__  
#define __FUNC__ "SNESSetFromOptions_EQ_LS"
static int SNESSetFromOptions_EQ_LS(SNES snes)
{
  SNES_LS *ls = (SNES_LS *)snes->data;
  char    ver[16];
  double  tmp;
  int     ierr,flg;

  PetscFunctionBegin;
  ierr = OptionsGetDouble(snes->prefix,"-snes_eq_ls_alpha",&tmp, &flg);CHKERRQ(ierr);
  if (flg) {
    ls->alpha = tmp;
  }
  ierr = OptionsGetDouble(snes->prefix,"-snes_eq_ls_maxstep",&tmp, &flg);CHKERRQ(ierr);
  if (flg) {
    ls->maxstep = tmp;
  }
  ierr = OptionsGetDouble(snes->prefix,"-snes_eq_ls_steptol",&tmp, &flg);CHKERRQ(ierr);
  if (flg) {
    ls->steptol = tmp;
  }
  ierr = OptionsGetString(snes->prefix,"-snes_eq_ls",ver,16, &flg); CHKERRQ(ierr);
  if (flg) {
    if (!PetscStrcmp(ver,"basic")) {
      ierr = SNESSetLineSearch(snes,SNESNoLineSearch,PETSC_NULL);CHKERRQ(ierr);
    } else if (!PetscStrcmp(ver,"basicnonorms")) {
      ierr = SNESSetLineSearch(snes,SNESNoLineSearchNoNorms,PETSC_NULL);CHKERRQ(ierr);
    } else if (!PetscStrcmp(ver,"quadratic")) {
      ierr = SNESSetLineSearch(snes,SNESQuadraticLineSearch,PETSC_NULL);CHKERRQ(ierr);
    } else if (!PetscStrcmp(ver,"cubic")) {
      ierr = SNESSetLineSearch(snes,SNESCubicLineSearch,PETSC_NULL);CHKERRQ(ierr);
    }
    else {SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Unknown line search");}
  }
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
/*
   SNESCreate_EQ_LS - Creates a nonlinear solver context for the SNES_EQ_LS method,
   SNES_LS, and sets this as the private data within the generic nonlinear solver
   context, SNES, that was created within SNESCreate().


   Input Parameter:
.  snes - the SNES context

   Application Interface Routine: SNESCreate()
 */
EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "SNESCreate_EQ_LS"
int SNESCreate_EQ_LS(SNES snes)
{
  int     ierr;
  SNES_LS *neP;

  PetscFunctionBegin;
  if (snes->method_class != SNES_NONLINEAR_EQUATIONS) {
    SETERRQ(PETSC_ERR_ARG_WRONG,0,"For SNES_NONLINEAR_EQUATIONS only");
  }

  snes->setup		= SNESSetUp_EQ_LS;
  snes->solve		= SNESSolve_EQ_LS;
  snes->destroy		= SNESDestroy_EQ_LS;
  snes->converged	= SNESConverged_EQ_LS;
  snes->printhelp       = SNESPrintHelp_EQ_LS;
  snes->setfromoptions  = SNESSetFromOptions_EQ_LS;
  snes->view            = SNESView_EQ_LS;
  snes->nwork           = 0;

  neP			= PetscNew(SNES_LS);   CHKPTRQ(neP);
  PLogObjectMemory(snes,sizeof(SNES_LS));
  snes->data    	= (void *) neP;
  neP->alpha		= 1.e-4;
  neP->maxstep		= 1.e8;
  neP->steptol		= 1.e-12;
  neP->LineSearch       = SNESCubicLineSearch;

  ierr = PetscObjectComposeFunction((PetscObject)snes,"SNESSetLineSearch_C","SNESSetLineSearch_LS",
                    (void*)SNESSetLineSearch_LS);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END



