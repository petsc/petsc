#ifndef lint
static char vcid[] = "$Id: ls.c,v 1.21 1995/06/03 04:25:44 bsmith Exp bsmith $";
#endif

#include <math.h>
#include "ls.h"

/*
     Implements a line search variant of Newton's Method 
    for solving systems of nonlinear equations.  

    Input parameters:
.   snes - nonlinear context obtained from SNESCreate()

    Output Parameters:
.   its  - Number of global iterations until termination.

    Notes:
    See SNESCreate() and SNESSetUp() for information on the definition and
    initialization of the nonlinear solver context.  

    This implements essentially a truncated Newton method with a
    line search.  By default a cubic backtracking line search 
    is employed, as described in the text "Numerical Methods for
    Unconstrained Optimization and Nonlinear Equations" by Dennis 
    and Schnabel.  See the examples in src/snes/examples.
*/

int SNESSolve_LS( SNES snes, int *outits )
{
  SNES_LS      *neP = (SNES_LS *) snes->data;
  int          maxits, i, history_len,ierr,lits;
  MatStructure flg = ALLMAT_DIFFERENT_NONZERO_PATTERN;
  double       fnorm, gnorm, xnorm, ynorm, *history;
  Vec          Y, X, F, G, W, TMP;

  history	= snes->conv_hist;	/* convergence history */
  history_len	= snes->conv_hist_len;	/* convergence history length */
  maxits	= snes->max_its;	/* maximum number of iterations */
  X		= snes->vec_sol;		/* solution vector */
  F		= snes->vec_func;		/* residual vector */
  Y		= snes->work[0];		/* work vectors */
  G		= snes->work[1];
  W		= snes->work[2];

  ierr = SNESComputeInitialGuess(snes,X); CHKERRQ(ierr);  /* X <- X_0 */
  VecNorm( X, &xnorm );		       /* xnorm = || X || */
  ierr = SNESComputeFunction(snes,X,F); CHKERRQ(ierr); /* (+/-) F(X) */
  VecNorm(F, &fnorm );	        	/* fnorm <- || F || */  
  snes->norm = fnorm;
  if (history && history_len > 0) history[0] = fnorm;
  if (snes->Monitor) (*snes->Monitor)(snes,0,fnorm,snes->monP);
        
  for ( i=0; i<maxits; i++ ) {
       snes->iter = i+1;

       /* Solve J Y = -F, where J is Jacobian matrix */
       ierr = SNESComputeJacobian(snes,X,&snes->jacobian,&snes->jacobian_pre,
                                &flg,snes->jacP); CHKERRQ(ierr);
       ierr = SLESSetOperators(snes->sles,snes->jacobian,snes->jacobian_pre,flg);
       ierr = SLESSolve(snes->sles,F,Y,&lits); CHKERRQ(ierr);
       ierr = (*neP->LineSearch)(snes, X, F, G, Y, W, fnorm, &ynorm, &gnorm );
       CHKERRQ(ierr);

       TMP = F; F = G; snes->vec_func_always = F; G = TMP;
       TMP = X; X = Y; snes->vec_sol_always = X; Y = TMP;
       fnorm = gnorm;

       snes->norm = fnorm;
       if (history && history_len > i+1) history[i+1] = fnorm;
       VecNorm( X, &xnorm );		/* xnorm = || X || */
       if (snes->Monitor) (*snes->Monitor)(snes,i+1,fnorm,snes->monP);

       /* Test for convergence */
       if ((*snes->Converged)( snes, xnorm, ynorm, fnorm,snes->cnvP )) {
           if (X != snes->vec_sol) {
             VecCopy( X, snes->vec_sol );
             snes->vec_sol_always = snes->vec_sol;
             snes->vec_func_always = snes->vec_func;
           }
           break;
       }
  }
  if (i == maxits) i--;
  *outits = i+1;
  return 0;
}
/* ------------------------------------------------------------ */
int SNESSetUp_LS(SNES snes )
{
  int ierr;
  snes->nwork = 3;
  ierr = VecGetVecs( snes->vec_sol, snes->nwork,&snes->work ); CHKERRQ(ierr);
  PLogObjectParents(snes,snes->nwork,snes->work ); 
  return 0;
}
/* ------------------------------------------------------------ */
int SNESDestroy_LS(PetscObject obj)
{
  SNES snes = (SNES) obj;
  VecFreeVecs(snes->work, snes->nwork );
  PETSCFREE(snes->data);
  return 0;
}
/*@ 
   SNESDefaultMonitor - Default SNES monitoring routine.  Prints the 
   residual norm at each iteration.

   Input Parameters:
.  snes - the SNES context
.  its - iteration number
.  fnorm - 2-norm residual value (may be estimated)
.  dummy - unused context

   Notes:
   f is either the residual or its negative, depending on the user's
   preference, as set with SNESSetFunction().

.keywords: SNES, nonlinear, default, monitor, residual, norm

.seealso: SNESSetMonitor()
@*/
int SNESDefaultMonitor(SNES snes,int its, double fnorm,void *dummy)
{
  MPIU_printf(snes->comm, "iter = %d, Function norm %g \n",its,fnorm);
  return 0;
}

int SNESDefaultSMonitor(SNES snes,int its, double fnorm,void *dummy)
{
  if (fnorm > 1.e-9 || fnorm == 0.0) {
    MPIU_printf(snes->comm, "iter = %d, Function norm %g \n",its,fnorm);
  }
  else if (fnorm > 1.e-11){
    MPIU_printf(snes->comm, "iter = %d, Function norm %5.3e \n",its,fnorm);
  }
  else {
    MPIU_printf(snes->comm, "iter = %d, Function norm < 1.e-11\n",its);
  }
  return 0;
}

/*@ 
   SNESDefaultConverged - Default test for monitoring the convergence 
   of the SNES solvers.

   Input Parameters:
.  snes - the SNES context
.  xnorm - 2-norm of current iterate
.  pnorm - 2-norm of current step 
.  fnorm - 2-norm of residual
.  dummy - unused context

   Returns:
$  2  if  ( fnorm < atol ),
$  3  if  ( pnorm < xtol*xnorm ),
$ -2  if  ( nres > max_res ),
$  0  otherwise,

   where
$    atol    - absolute residual norm tolerance,
$              set with SNESSetAbsoluteTolerance()
$    max_res - maximum number of residual evaluations,
$              set with SNESSetMaxResidualEvaluations()
$    nres    - number of residual evaluations
$    xtol    - relative residual norm tolerance,
$              set with SNESSetRelativeTolerance()

.keywords: SNES, nonlinear, default, converged, convergence

.seealso: SNESSetConvergenceTest()
@*/
int SNESDefaultConverged(SNES snes,double xnorm,double pnorm,double fnorm,
                         void *dummy)
{
  if (fnorm < snes->atol) {
    PLogInfo((PetscObject)snes,
      "Converged due to absolute residual norm %g < %g\n",fnorm,snes->atol);
    return 2;
  }
  if (pnorm < snes->xtol*(xnorm)) {
    PLogInfo((PetscObject)snes,
      "Converged due to small update length  %g < %g*%g\n",
       pnorm,snes->xtol,xnorm);
    return 3;
  }
  if (snes->nfuncs > snes->max_funcs) {
    PLogInfo((PetscObject)snes,
      "Exceeded maximum number of residual evaluations: %d > %d\n",
       snes->nfuncs, snes->max_funcs );
    return -2;
  }  
  return 0;
}

/* ------------------------------------------------------------ */
/*ARGSUSED*/
/*@
   SNESNoLineSearch - This routine is not a line search at all; 
   it simply uses the full Newton step.  Thus, this routine is intended 
   to serve as a template and is not recommended for general use.  

   Input Parameters:
.  snes - nonlinear context
.  x - current iterate
.  f - residual evaluated at x
.  y - search direction (contains new iterate on output)
.  w - work vector
.  fnorm - 2-norm of f

   Output Parameters:
.  g - residual evaluated at new iterate y
.  y - new iterate (contains search direction on input)
.  gnorm - 2-norm of g
.  ynorm - 2-norm of search length

   Options Database Key:
$  -snes_line_search basic

   Returns:
   1, indicating success of the step.

.keywords: SNES, nonlinear, line search, cubic

.seealso: SNESCubicLineSearch(), SNESQuadraticLineSearch(), 
.seealso: SNESSetLineSearchRoutine()
@*/
int SNESNoLineSearch(SNES snes, Vec x, Vec f, Vec g, Vec y, Vec w,
                             double fnorm, double *ynorm, double *gnorm )
{
  int    ierr;
  Scalar one = 1.0;
  PLogEventBegin(SNES_LineSearch,snes,x,f,g);
  VecNorm(y, ynorm );	/* ynorm = || y ||    */
  VecAXPY(&one, x, y );	/* y <- x + y         */
  ierr = SNESComputeFunction(snes,y,g); CHKERRQ(ierr);
  VecNorm( g, gnorm ); 	/* gnorm = || g ||    */
  PLogEventEnd(SNES_LineSearch,snes,x,f,g);
  return 1;
}
/* ------------------------------------------------------------------ */
/*@
   SNESCubicLineSearch - This routine performs a cubic line search and
   is the default line search method.

   Input Parameters:
.  snes - nonlinear context
.  x - current iterate
.  f - residual evaluated at x
.  y - search direction (contains new iterate on output)
.  w - work vector
.  fnorm - 2-norm of f

   Output parameters:
.  g - residual evaluated at new iterate y
.  y - new iterate (contains search direction on input)
.  gnorm - 2-norm of g
.  ynorm - 2-norm of search length

   Returns:
   1 if the line search succeeds; 0 if the line search fails.

   Options Database Key:
$  -snes_line_search cubic

   Notes:
   This line search is taken from "Numerical Methods for Unconstrained 
   Optimization and Nonlinear Equations" by Dennis and Schnabel, page 325.

.keywords: SNES, nonlinear, line search, cubic

.seealso: SNESNoLineSearch(), SNESNoLineSearch(), SNESSetLineSearchRoutine()
@*/
int SNESCubicLineSearch(SNES snes, Vec x, Vec f, Vec g, Vec y, Vec w,
                              double fnorm, double *ynorm, double *gnorm )
{
  double  steptol, initslope;
  double  lambdaprev, gnormprev;
  double  a, b, d, t1, t2;
#if defined(PETSC_COMPLEX)
  Scalar  cinitslope,clambda;
#endif
  int     ierr,count;
  SNES_LS *neP = (SNES_LS *) snes->data;
  Scalar  one = 1.0,scale;
  double  maxstep,minlambda,alpha,lambda,lambdatemp;

  PLogEventBegin(SNES_LineSearch,snes,x,f,g);
  alpha   = neP->alpha;
  maxstep = neP->maxstep;
  steptol = neP->steptol;

  VecNorm(y, ynorm );
  if (*ynorm > maxstep) {	/* Step too big, so scale back */
    scale = maxstep/(*ynorm);
#if defined(PETSC_COMPLEX)
    PLogInfo((PetscObject)snes,"Scaling step by %g\n",real(scale));
#else
    PLogInfo((PetscObject)snes,"Scaling step by %g\n",scale);
#endif
    VecScale(&scale, y ); 
    *ynorm = maxstep;
  }
  minlambda = steptol/(*ynorm);
#if defined(PETSC_COMPLEX)
  VecDot(f, y, &cinitslope ); 
  initslope = real(cinitslope);
#else
  VecDot(f, y, &initslope ); 
#endif
  if (initslope > 0.0) initslope = -initslope;
  if (initslope == 0.0) initslope = -1.0;

  VecCopy(y, w );
  VecAXPY(&one, x, w );
  ierr = SNESComputeFunction(snes,w,g); CHKERRQ(ierr);
  VecNorm(g, gnorm ); 
  if (*gnorm <= fnorm + alpha*initslope) {	/* Sufficient reduction */
      VecCopy(w, y );
      PLogInfo((PetscObject)snes,"Using full step\n");
      PLogEventEnd(SNES_LineSearch,snes,x,f,g);
      return 0;
  }

  /* Fit points with quadratic */
  lambda = 1.0; count = 0;
  lambdatemp = -initslope/(2.0*(*gnorm - fnorm - initslope));
  lambdaprev = lambda;
  gnormprev = *gnorm;
  if (lambdatemp <= .1*lambda) { 
      lambda = .1*lambda; 
  } else lambda = lambdatemp;
  VecCopy(x, w );
#if defined(PETSC_COMPLEX)
  clambda = lambda; VecAXPY(&clambda, y, w );
#else
  VecAXPY(&lambda, y, w );
#endif
  ierr = SNESComputeFunction(snes,w,g); CHKERRQ(ierr);
  VecNorm(g, gnorm ); 
  if (*gnorm <= fnorm + alpha*initslope) {      /* sufficient reduction */
      VecCopy(w, y );
      PLogInfo((PetscObject)snes,"Quadratically determined step, lambda %g\n",lambda);
      PLogEventEnd(SNES_LineSearch,snes,x,f,g);
      return 0;
  }

  /* Fit points with cubic */
  count = 1;
  while (1) {
      if (lambda <= minlambda) { /* bad luck; use full step */
           PLogInfo((PetscObject)snes,"Unable to find good step length! %d \n",count);
           PLogInfo((PetscObject)snes, "f %g fnew %g ynorm %g lambda %g \n",
                   fnorm,*gnorm, *ynorm,lambda);
           VecCopy(w, y );
           PLogEventEnd(SNES_LineSearch,snes,x,f,g);
           return -1;
      }
      t1 = *gnorm - fnorm - lambda*initslope;
      t2 = gnormprev  - fnorm - lambdaprev*initslope;
      a = (t1/(lambda*lambda) - 
                t2/(lambdaprev*lambdaprev))/(lambda-lambdaprev);
      b = (-lambdaprev*t1/(lambda*lambda) + 
                lambda*t2/(lambdaprev*lambdaprev))/(lambda-lambdaprev);
      d = b*b - 3*a*initslope;
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
      VecCopy( x, w );
#if defined(PETSC_COMPLEX)
      VecAXPY(&clambda, y, w );
#else
      VecAXPY(&lambda, y, w );
#endif
      ierr = SNESComputeFunction(snes,w,g); CHKERRQ(ierr);
      VecNorm(g, gnorm ); 
      if (*gnorm <= fnorm + alpha*initslope) {      /* is reduction enough */
         VecCopy(w, y );
         PLogInfo((PetscObject)snes,"Cubically determined step, lambda %g\n",lambda);
         PLogEventEnd(SNES_LineSearch,snes,x,f,g);
         return 0;
      }
      count++;
   }
  PLogEventEnd(SNES_LineSearch,snes,x,f,g);
  return 0;
}
/*@
   SNESQuadraticLineSearch - This routine performs a cubic line search.

   Input Parameters:
.  snes - the SNES context
.  x - current iterate
.  f - residual evaluated at x
.  y - search direction (contains new iterate on output)
.  w - work vector
.  fnorm - 2-norm of f

   Output Parameters:
.  g - residual evaluated at new iterate y
.  y - new iterate (contains search direction on input)
.  gnorm - 2-norm of g
.  ynorm - 2-norm of search length

   Returns:
   1 if the line search succeeds; 0 if the line search fails.

   Options Database Key:
$  -snes_line_search quadratic

   Notes:
   Use SNESSetLineSearchRoutine()
   to set this routine within the SNES_NLS method.  

.keywords: SNES, nonlinear, quadratic, line search

.seealso: SNESCubicLineSearch(), SNESNoLineSearch(), SNESSetLineSearchRoutine()
@*/
int SNESQuadraticLineSearch(SNES snes, Vec x, Vec f, Vec g, Vec y, Vec w,
                              double fnorm, double *ynorm, double *gnorm )
{
  double  steptol, initslope;
  double  lambdaprev, gnormprev;
#if defined(PETSC_COMPLEX)
  Scalar  cinitslope,clambda;
#endif
  int     ierr,count;
  SNES_LS *neP = (SNES_LS *) snes->data;
  Scalar  one = 1.0,scale;
  double  maxstep,minlambda,alpha,lambda,lambdatemp;

  PLogEventBegin(SNES_LineSearch,snes,x,f,g);
  alpha   = neP->alpha;
  maxstep = neP->maxstep;
  steptol = neP->steptol;

  VecNorm(y, ynorm );
  if (*ynorm > maxstep) {	/* Step too big, so scale back */
    scale = maxstep/(*ynorm);
    VecScale(&scale, y ); 
    *ynorm = maxstep;
  }
  minlambda = steptol/(*ynorm);
#if defined(PETSC_COMPLEX)
  VecDot(f, y, &cinitslope ); 
  initslope = real(cinitslope);
#else
  VecDot(f, y, &initslope ); 
#endif
  if (initslope > 0.0) initslope = -initslope;
  if (initslope == 0.0) initslope = -1.0;

  VecCopy(y, w );
  VecAXPY(&one, x, w );
  ierr = SNESComputeFunction(snes,w,g); CHKERRQ(ierr);
  VecNorm(g, gnorm ); 
  if (*gnorm <= fnorm + alpha*initslope) {	/* Sufficient reduction */
      VecCopy(w, y );
      PLogInfo((PetscObject)snes,"Using full step\n");
      PLogEventEnd(SNES_LineSearch,snes,x,f,g);
      return 0;
  }

  /* Fit points with quadratic */
  lambda = 1.0; count = 0;
  count = 1;
  while (1) {
    if (lambda <= minlambda) { /* bad luck; use full step */
      PLogInfo((PetscObject)snes,"Unable to find good step length! %d \n",count);
      PLogInfo((PetscObject)snes, "f %g fnew %g ynorm %g lambda %g \n",
                   fnorm,*gnorm, *ynorm,lambda);
      VecCopy(w, y );
      PLogEventEnd(SNES_LineSearch,snes,x,f,g);
      return 0;
    }
    lambdatemp = -initslope/(2.0*(*gnorm - fnorm - initslope));
    lambdaprev = lambda;
    gnormprev = *gnorm;
    if (lambdatemp <= .1*lambda) { 
      lambda = .1*lambda; 
    } else lambda = lambdatemp;
    VecCopy(x, w );
#if defined(PETSC_COMPLEX)
    clambda = lambda; VecAXPY(&clambda, y, w );
#else
    VecAXPY(&lambda, y, w );
#endif
    ierr = SNESComputeFunction(snes,w,g); CHKERRQ(ierr);
    VecNorm(g, gnorm ); 
    if (*gnorm <= fnorm + alpha*initslope) {      /* sufficient reduction */
      VecCopy(w, y );
      PLogInfo((PetscObject)snes,"Quadratically determined step, lambda %g\n",lambda);
      PLogEventEnd(SNES_LineSearch,snes,x,f,g);
      return 0;
    }
    count++;
  }

  PLogEventEnd(SNES_LineSearch,snes,x,f,g);
  return 0;
}
/* ------------------------------------------------------------ */
/*@C
   SNESSetLineSearchRoutine - Sets the line search routine to be used
   by the method SNES_LS.

   Input Parameters:
.  snes - nonlinear context obtained from SNESCreate()
.  func - pointer to int function

   Available Routines:
.  SNESCubicLineSearch() - default line search
.  SNESQuadraticLineSearch() - quadratic line search
.  SNESNoLineSearch() - the full Newton step (actually not a line search)

    Options Database Keys:
$   -snes_line_search [basic,quadratic,cubic]

   Calling sequence of func:
   func (SNES snes, Vec x, Vec f, Vec g, Vec y,
         Vec w, double fnorm, double *ynorm, double *gnorm)

    Input parameters for func:
.   snes - nonlinear context
.   x - current iterate
.   f - residual evaluated at x
.   y - search direction (contains new iterate on output)
.   w - work vector
.   fnorm - 2-norm of f

    Output parameters for func:
.   g - residual evaluated at new iterate y
.   y - new iterate (contains search direction on input)
.   gnorm - 2-norm of g
.   ynorm - 2-norm of search length

    Returned by func:
    1 if the line search succeeds; 0 if the line search fails.

.keywords: SNES, nonlinear, set, line search, routine

.seealso: SNESNoLineSearch(), SNESQuadraticLineSearch(), SNESCubicLineSearch()
@*/
int SNESSetLineSearchRoutine(SNES snes,int (*func)(SNES,Vec,Vec,Vec,Vec,Vec,
                             double,double *,double*) )
{
  if ((snes)->type == SNES_NLS)
    ((SNES_LS *)(snes->data))->LineSearch = func;
  return 0;
}

static int SNESPrintHelp_LS(SNES snes)
{
  SNES_LS *ls = (SNES_LS *)snes->data;
  fprintf(stderr,"-snes_line_search [basic,quadratic,cubic]\n");
  fprintf(stderr,"-snes_line_search_alpha alpha (default %g)\n",ls->alpha);
  fprintf(stderr,"-snes_line_search_maxstep max (default %g)\n",ls->maxstep);
  fprintf(stderr,"-snes_line_search_steptol tol (default %g)\n",ls->steptol);
  return 0;
}

static int SNESSetFromOptions_LS(SNES snes)
{
  SNES_LS *ls = (SNES_LS *)snes->data;
  char    ver[16];
  double  tmp;

  if (OptionsGetDouble(snes->prefix,"-snes_line_search_alpa",&tmp)) {
    ls->alpha = tmp;
  }
  if (OptionsGetDouble(snes->prefix,"-snes_line_search_maxstep",&tmp)) {
    ls->maxstep = tmp;
  }
  if (OptionsGetDouble(snes->prefix,"-snes_line_search_steptol",&tmp)) {
    ls->steptol = tmp;
  }
  if (OptionsGetString(snes->prefix,"-snes_line_search",ver,16)) {
    if (!strcmp(ver,"basic")) {
      SNESSetLineSearchRoutine(snes,SNESNoLineSearch);
    }
    else if (!strcmp(ver,"quadratic")) {
      SNESSetLineSearchRoutine(snes,SNESQuadraticLineSearch);
    }
    else if (!strcmp(ver,"cubic")) {
      SNESSetLineSearchRoutine(snes,SNESCubicLineSearch);
    }
    else {SETERRQ(1,"Unknown line search?");}
  }
  return 0;
}

/* ------------------------------------------------------------ */
int SNESCreate_LS(SNES  snes )
{
  SNES_LS *neP;

  snes->type		= SNES_NLS;
  snes->setup		= SNESSetUp_LS;
  snes->solve		= SNESSolve_LS;
  snes->destroy		= SNESDestroy_LS;
  snes->Converged	= SNESDefaultConverged;
  snes->printhelp       = SNESPrintHelp_LS;
  snes->setfromoptions  = SNESSetFromOptions_LS;

  neP			= PETSCNEW(SNES_LS);   CHKPTRQ(neP);
  snes->data    	= (void *) neP;
  neP->alpha		= 1.e-4;
  neP->maxstep		= 1.e8;
  neP->steptol		= 1.e-12;
  neP->LineSearch       = SNESCubicLineSearch;
  return 0;
}




