#ifndef lint
static char vcid[] = "$Id: ls.c,v 1.35 1995/07/27 03:01:17 curfman Exp curfman $";
#endif

#include <math.h>
#include "ls.h"
#include "pviewer.h"
#if defined(HAVE_STRING_H)
#include <string.h>
#endif

/*
     Implements a line search variant of Newton's Method 
    for solving systems of nonlinear equations.  

    Input parameters:
.   snes - nonlinear context obtained from SNESCreate()

    Output Parameters:
.   outits  - Number of global iterations until termination.

    Notes:
    This implements essentially a truncated Newton method with a
    line search.  By default a cubic backtracking line search 
    is employed, as described in the text "Numerical Methods for
    Unconstrained Optimization and Nonlinear Equations" by Dennis 
    and Schnabel.  See the examples in src/snes/examples.
*/

int SNESSolve_LS(SNES snes,int *outits)
{
  SNES_LS      *neP = (SNES_LS *) snes->data;
  int          maxits, i, history_len, ierr, lits, lsfail;
  MatStructure flg = ALLMAT_DIFFERENT_NONZERO_PATTERN;
  double       fnorm, gnorm, xnorm, ynorm, *history;
  Vec          Y, X, F, G, W, TMP;
  SLES         sles;
  KSP          ksp;

  history	= snes->conv_hist;	/* convergence history */
  history_len	= snes->conv_hist_len;	/* convergence history length */
  maxits	= snes->max_its;	/* maximum number of iterations */
  X		= snes->vec_sol;	/* solution vector */
  F		= snes->vec_func;	/* residual vector */
  Y		= snes->work[0];	/* work vectors */
  G		= snes->work[1];
  W		= snes->work[2];

  ierr = SNESComputeInitialGuess(snes,X); CHKERRQ(ierr); /* X <- X_0 */
  VecNorm(X,&xnorm);		                         /* xnorm = || X || */
  ierr = SNESComputeFunction(snes,X,F); CHKERRQ(ierr);   /* (+/-) F(X) */
  VecNorm(F,&fnorm);	        	                 /* fnorm <- ||F|| */
  snes->norm = fnorm;
  if (history && history_len > 0) history[0] = fnorm;
  if (snes->monitor) 
    {ierr = (*snes->monitor)(snes,0,fnorm,snes->monP); CHKERRQ(ierr);}

  /* Set the KSP stopping criteria to use the Eisenstat-Walker method */
  if (snes->ksp_ewconv) {
    ierr = SNESGetSLES(snes,&sles); CHKERRQ(ierr);
    ierr = SLESGetKSP(sles,&ksp); CHKERRQ(ierr);
    ierr = KSPSetConvergenceTest(ksp,SNES_KSP_EW_Converged_Private,
           (void *)snes);  CHKERRQ(ierr);
  }
          
  for ( i=0; i<maxits; i++ ) {
       snes->iter = i+1;

       /* Solve J Y = -F, where J is Jacobian matrix */
       ierr = SNESComputeJacobian(snes,X,&snes->jacobian,&snes->jacobian_pre,
                                &flg); CHKERRQ(ierr);
       ierr = SLESSetOperators(snes->sles,snes->jacobian,snes->jacobian_pre,
                                flg); CHKERRQ(ierr);
       ierr = SLESSolve(snes->sles,F,Y,&lits); CHKERRQ(ierr);
       ierr = VecCopy(Y,snes->vec_sol_update_always); CHKERRQ(ierr);
       ierr = (*neP->LineSearch)(snes,X,F,G,Y,W,fnorm,&ynorm,&gnorm,&lsfail);
       if (lsfail) snes->nfailures++;
       CHKERRQ(ierr);

       TMP = F; F = G; snes->vec_func_always = F; G = TMP;
       TMP = X; X = Y; snes->vec_sol_always = X; Y = TMP;
       fnorm = gnorm;

       snes->norm = fnorm;
       if (history && history_len > i+1) history[i+1] = fnorm;
       VecNorm(X,&xnorm);		/* xnorm = || X || */
       if (snes->monitor) 
         {(*snes->monitor)(snes,i+1,fnorm,snes->monP); CHKERRQ(ierr);}

       /* Test for convergence */
       if ((*snes->converged)(snes,xnorm,ynorm,fnorm,snes->cnvP)) {
           if (X != snes->vec_sol) {
             VecCopy(X,snes->vec_sol);
             snes->vec_sol_always = snes->vec_sol;
             snes->vec_func_always = snes->vec_func;
           }
           break;
       }
  }
  if (i == maxits) {
    PLogInfo((PetscObject)snes,
      "Maximum number of iterations has been reached: %d\n",maxits);
    i--;
  }
  *outits = i+1;
  return 0;
}
/* ------------------------------------------------------------ */
int SNESSetUp_LS(SNES snes )
{
  int ierr;
  snes->nwork = 4;
  ierr = VecGetVecs(snes->vec_sol,snes->nwork,&snes->work); CHKERRQ(ierr);
  PLogObjectParents(snes,snes->nwork,snes->work ); 
  snes->vec_sol_update_always = snes->work[3];
  return 0;
}
/* ------------------------------------------------------------ */
int SNESDestroy_LS(PetscObject obj)
{
  SNES snes = (SNES) obj;
  VecFreeVecs(snes->work,snes->nwork);
  PETSCFREE(snes->data);
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
.  flag - set to 0, indicating a successful line search

   Options Database Key:
$  -snes_line_search basic

.keywords: SNES, nonlinear, line search, cubic

.seealso: SNESCubicLineSearch(), SNESQuadraticLineSearch(), 
          SNESSetLineSearchRoutine()
@*/
int SNESNoLineSearch(SNES snes, Vec x, Vec f, Vec g, Vec y, Vec w,
              double fnorm, double *ynorm, double *gnorm,int *flag )
{
  int    ierr;
  Scalar one = 1.0;
  *flag = 0;
  PLogEventBegin(SNES_LineSearch,snes,x,f,g);
  VecNorm(y,ynorm);                            /* ynorm = || y || */
  VecAXPY(&one,x,y);	                       /* y <- x + y      */
  ierr = SNESComputeFunction(snes,y,g); CHKERRQ(ierr);
  VecNorm(g,gnorm); 	                       /* gnorm = || g || */
  PLogEventEnd(SNES_LineSearch,snes,x,f,g);
  return 0;
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
.  flag - 0 if line search succeeds; -1 on failure.

   Options Database Key:
$  -snes_line_search cubic

   Notes:
   This line search is taken from "Numerical Methods for Unconstrained 
   Optimization and Nonlinear Equations" by Dennis and Schnabel, page 325.

.keywords: SNES, nonlinear, line search, cubic

.seealso: SNESNoLineSearch(), SNESNoLineSearch(), SNESSetLineSearchRoutine()
@*/
int SNESCubicLineSearch(SNES snes, Vec x, Vec f, Vec g, Vec y, Vec w,
                double fnorm, double *ynorm, double *gnorm,int *flag)
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
  *flag = 0;
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
      goto theend;
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
      goto theend;
  }

  /* Fit points with cubic */
  count = 1;
  while (1) {
      if (lambda <= minlambda) { /* bad luck; use full step */
           PLogInfo((PetscObject)snes,"Unable to find good step length! %d \n",count);
           PLogInfo((PetscObject)snes, "f %g fnew %g ynorm %g lambda %g \n",
                   fnorm,*gnorm, *ynorm,lambda);
           VecCopy(w, y );
           *flag = -1; break;
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
         *flag = -1; break;
      }
      count++;
   }
  theend:
  PLogEventEnd(SNES_LineSearch,snes,x,f,g);
  return 0;
}
/* ------------------------------------------------------------------ */
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
.  flag - 0 if line search succeeds; -1 on failure.

   Options Database Key:
$  -snes_line_search quadratic

   Notes:
   Use SNESSetLineSearchRoutine()
   to set this routine within the SNES_NLS method.  

.keywords: SNES, nonlinear, quadratic, line search

.seealso: SNESCubicLineSearch(), SNESNoLineSearch(), SNESSetLineSearchRoutine()
@*/
int SNESQuadraticLineSearch(SNES snes, Vec x, Vec f, Vec g, Vec y, Vec w,
                    double fnorm, double *ynorm, double *gnorm,int *flag)
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
  *flag = 0;
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
      goto theend;
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
      *flag = -1; break;
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
      break;
    }
    count++;
  }
  theend:
  PLogEventEnd(SNES_LineSearch,snes,x,f,g);
  return 0;
}
/* ------------------------------------------------------------ */
/*@
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
         Vec w, double fnorm, double *ynorm, 
         double *gnorm, *flag)

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
.   flag - set to 0 if the line search succeeds; a nonzero integer 
           on failure.

.keywords: SNES, nonlinear, set, line search, routine

.seealso: SNESNoLineSearch(), SNESQuadraticLineSearch(), SNESCubicLineSearch()
@*/
int SNESSetLineSearchRoutine(SNES snes,int (*func)(SNES,Vec,Vec,Vec,Vec,Vec,
                             double,double *,double*,int*) )
{
  if ((snes)->type == SNES_NLS)
    ((SNES_LS *)(snes->data))->LineSearch = func;
  return 0;
}
/* ------------------------------------------------------------------ */
static int SNESPrintHelp_LS(SNES snes)
{
  SNES_LS *ls = (SNES_LS *)snes->data;
  char    *p;
  if (snes->prefix) p = snes->prefix; else p = "-";
  MPIU_printf(snes->comm," method ls (system of nonlinear equations):\n");
  MPIU_printf(snes->comm,"   %ssnes_line_search [basic,quadratic,cubic]\n",p);
  MPIU_printf(snes->comm,"   %ssnes_line_search_alpha alpha (default %g)\n",p,ls->alpha);
  MPIU_printf(snes->comm,"   %ssnes_line_search_maxstep max (default %g)\n",p,ls->maxstep);
  MPIU_printf(snes->comm,"   %ssnes_line_search_steptol tol (default %g)\n",p,ls->steptol);
  return 0;
}
/* ------------------------------------------------------------------ */
static int SNESView_LS(PetscObject obj,Viewer viewer)
{
  SNES    snes = (SNES)obj;
  SNES_LS *ls = (SNES_LS *)snes->data;
  FILE    *fd = ViewerFileGetPointer_Private(viewer);
  char    *cstring;
  
  if (ls->LineSearch == SNESNoLineSearch) 
    cstring = "SNESNoLineSearch";
  else if (ls->LineSearch == SNESQuadraticLineSearch) 
    cstring = "SNESQuadraticLineSearch";
  else if (ls->LineSearch == SNESCubicLineSearch) 
    cstring = "SNESCubicLineSearch";
  else
    cstring = "unknown";
  MPIU_fprintf(snes->comm,fd,"    line search variant: %s\n",cstring);
  MPIU_fprintf(snes->comm,fd,"    alpha=%g, maxstep=%g, steptol=%g\n",
     ls->alpha,ls->maxstep,ls->steptol);
  return 0;
}
/* ------------------------------------------------------------------ */
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
    else {SETERRQ(1,"SNESSetFromOptions_LS: Unknown line search");}
  }
  return 0;
}
/* ------------------------------------------------------------ */
int SNESCreate_LS(SNES  snes )
{
  SNES_LS *neP;

  if (snes->method_class != SNES_NONLINEAR_EQUATIONS) SETERRQ(1,
    "SNESCreate_LS: Valid for SNES_NONLINEAR_EQUATIONS problems only");
  snes->type		= SNES_EQ_NLS;
  snes->setup		= SNESSetUp_LS;
  snes->solve		= SNESSolve_LS;
  snes->destroy		= SNESDestroy_LS;
  snes->converged	= SNESDefaultConverged;
  snes->printhelp       = SNESPrintHelp_LS;
  snes->setfromoptions  = SNESSetFromOptions_LS;
  snes->view            = SNESView_LS;

  neP			= PETSCNEW(SNES_LS);   CHKPTRQ(neP);
  snes->data    	= (void *) neP;
  neP->alpha		= 1.e-4;
  neP->maxstep		= 1.e8;
  neP->steptol		= 1.e-12;
  neP->LineSearch       = SNESCubicLineSearch;
  return 0;
}




