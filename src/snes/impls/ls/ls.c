#ifndef lint
static char vcid[] = "$Id: ls.c,v 1.61 1996/03/10 17:29:35 bsmith Exp bsmith $";
#endif

#include <math.h>
#include "ls.h"
#include "pinclude/pviewer.h"

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
    and Schnabel.
*/

int SNESSolve_LS(SNES snes,int *outits)
{
  SNES_LS      *neP = (SNES_LS *) snes->data;
  int          maxits, i, history_len, ierr, lits, lsfail;
  MatStructure flg = DIFFERENT_NONZERO_PATTERN;
  double       fnorm, gnorm, xnorm, ynorm, *history;
  Vec          Y, X, F, G, W, TMP;

  history	= snes->conv_hist;	/* convergence history */
  history_len	= snes->conv_hist_len;	/* convergence history length */
  maxits	= snes->max_its;	/* maximum number of iterations */
  X		= snes->vec_sol;	/* solution vector */
  F		= snes->vec_func;	/* residual vector */
  Y		= snes->work[0];	/* work vectors */
  G		= snes->work[1];
  W		= snes->work[2];

  ierr = VecNorm(X,NORM_2,&xnorm); CHKERRQ(ierr);               /* xnorm = || X || */
  ierr = SNESComputeFunction(snes,X,F); CHKERRQ(ierr);          /*  F(X)      */
  ierr = VecNorm(F,NORM_2,&fnorm); CHKERRQ(ierr);	        /* fnorm <- ||F||  */
  snes->norm = fnorm;
  if (history && history_len > 0) history[0] = fnorm;
  if (snes->monitor) {ierr = (*snes->monitor)(snes,0,fnorm,snes->monP); CHKERRQ(ierr);}

  /* set parameter for default relative tolerance convergence test */
  snes->ttol = fnorm*snes->rtol;

  for ( i=0; i<maxits; i++ ) {
    snes->iter = i+1;

    /* Solve J Y = -F, where J is Jacobian matrix */
    ierr = SNESComputeJacobian(snes,X,&snes->jacobian,&snes->jacobian_pre,&flg);CHKERRQ(ierr);
    ierr = SLESSetOperators(snes->sles,snes->jacobian,snes->jacobian_pre,flg); CHKERRQ(ierr);
    ierr = SLESSolve(snes->sles,F,Y,&lits); CHKERRQ(ierr);
    ierr = VecCopy(Y,snes->vec_sol_update_always); CHKERRQ(ierr);
    ierr = (*neP->LineSearch)(snes,X,F,G,Y,W,fnorm,&ynorm,&gnorm,&lsfail);CHKERRQ(ierr);
    if (lsfail) snes->nfailures++;

    TMP = F; F = G; snes->vec_func_always = F; G = TMP;
    TMP = X; X = Y; snes->vec_sol_always = X; Y = TMP;
    fnorm = gnorm;

    snes->norm = fnorm;
    if (history && history_len > i+1) history[i+1] = fnorm;
    ierr = VecNorm(X,NORM_2,&xnorm); CHKERRQ(ierr);	/* xnorm = || X || */
    if (snes->monitor) {ierr = (*snes->monitor)(snes,i+1,fnorm,snes->monP);CHKERRQ(ierr);}

    /* Test for convergence */
    if ((*snes->converged)(snes,xnorm,ynorm,fnorm,snes->cnvP)) {
      if (X != snes->vec_sol) {
        ierr = VecCopy(X,snes->vec_sol); CHKERRQ(ierr);
        snes->vec_sol_always = snes->vec_sol;
        snes->vec_func_always = snes->vec_func;
      }
      break;
    }
  }
  if (i == maxits) {
    PLogInfo((PetscObject)snes,
      "SNESSolve_LS: Maximum number of iterations has been reached: %d\n",maxits);
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
  ierr = VecDuplicateVecs(snes->vec_sol,snes->nwork,&snes->work); CHKERRQ(ierr);
  PLogObjectParents(snes,snes->nwork,snes->work);
  snes->vec_sol_update_always = snes->work[3];
  return 0;
}
/* ------------------------------------------------------------ */
int SNESDestroy_LS(PetscObject obj)
{
  SNES snes = (SNES) obj;
  int  ierr;
  ierr = VecDestroyVecs(snes->work,snes->nwork); CHKERRQ(ierr);
  PetscFree(snes->data);
  return 0;
}
/* ------------------------------------------------------------ */
/*ARGSUSED*/
/*@C
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
  Scalar mone = -1.0;

  *flag = 0;
  PLogEventBegin(SNES_LineSearch,snes,x,f,g);
  ierr = VecNorm(y,NORM_2,ynorm); CHKERRQ(ierr);              /* ynorm = || y || */
  ierr = VecAYPX(&mone,x,y); CHKERRQ(ierr);                   /* y <- x + y      */
  ierr = SNESComputeFunction(snes,y,g); CHKERRQ(ierr);        /*  F(y)      */
  ierr = VecNorm(g,NORM_2,gnorm); CHKERRQ(ierr);              /* gnorm = || g || */
  PLogEventEnd(SNES_LineSearch,snes,x,f,g);
  return 0;
}
/* ------------------------------------------------------------------ */
/*@C
   SNESCubicLineSearch - This routine performs a cubic line search and
   is the default line search method.

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
  double  steptol, initslope,lambdaprev, gnormprev,a, b, d, t1, t2;
  double  maxstep,minlambda,alpha,lambda,lambdatemp;
#if defined(PETSC_COMPLEX)
  Scalar  cinitslope,clambda;
#endif
  int     ierr, count;
  SNES_LS *neP = (SNES_LS *) snes->data;
  Scalar  mone = -1.0,scale;

  PLogEventBegin(SNES_LineSearch,snes,x,f,g);
  *flag   = 0;
  alpha   = neP->alpha;
  maxstep = neP->maxstep;
  steptol = neP->steptol;

  ierr = VecNorm(y,NORM_2,ynorm); CHKERRQ(ierr);
  if (*ynorm > maxstep) {	/* Step too big, so scale back */
    scale = maxstep/(*ynorm);
#if defined(PETSC_COMPLEX)
    PLogInfo((PetscObject)snes,"SNESCubicLineSearch: Scaling step by %g\n",real(scale));
#else
    PLogInfo((PetscObject)snes,"SNESCubicLineSearch: Scaling step by %g\n",scale);
#endif
    ierr = VecScale(&scale,y); CHKERRQ(ierr);
    *ynorm = maxstep;
  }
  minlambda = steptol/(*ynorm);
  ierr = MatMult(snes->jacobian,y,w); CHKERRQ(ierr);
#if defined(PETSC_COMPLEX)
  ierr = VecDot(f,w,&cinitslope); CHKERRQ(ierr);
  initslope = real(cinitslope);
#else
  ierr = VecDot(f,w,&initslope); CHKERRQ(ierr);
#endif
  if (initslope > 0.0) initslope = -initslope;
  if (initslope == 0.0) initslope = -1.0;

  ierr = VecCopy(y,w); CHKERRQ(ierr);
  ierr = VecAYPX(&mone,x,w); CHKERRQ(ierr);
  ierr = SNESComputeFunction(snes,w,g); CHKERRQ(ierr);
  ierr = VecNorm(g,NORM_2,gnorm); 
  if (*gnorm <= fnorm + alpha*initslope) {	/* Sufficient reduction */
    ierr = VecCopy(w,y); CHKERRQ(ierr);
    PLogInfo((PetscObject)snes,"SNESCubicLineSearch: Using full step\n");
    goto theend;
  }

  /* Fit points with quadratic */
  lambda = 1.0; count = 0;
  lambdatemp = -initslope/((*gnorm)*(*gnorm) - fnorm*fnorm - 2.0*initslope);
  lambdaprev = lambda;
  gnormprev = *gnorm;
  if (lambdatemp <= .1*lambda) lambda = .1*lambda; 
  else lambda = lambdatemp;
  ierr   = VecCopy(x,w); CHKERRQ(ierr);
  lambda = -lambda;
#if defined(PETSC_COMPLEX)
  clambda = lambda; ierr = VecAXPY(&clambda,y,w); CHKERRQ(ierr);
#else
  ierr = VecAXPY(&lambda,y,w); CHKERRQ(ierr);
#endif
  ierr = SNESComputeFunction(snes,w,g); CHKERRQ(ierr);
  ierr = VecNorm(g,NORM_2,gnorm); CHKERRQ(ierr);
  if (*gnorm <= fnorm + alpha*initslope) {      /* sufficient reduction */
    ierr = VecCopy(w,y); CHKERRQ(ierr);
    PLogInfo((PetscObject)snes,
             "SNESCubicLineSearch: Quadratically determined step, lambda %g\n",lambda);
    goto theend;
  }

  /* Fit points with cubic */
  count = 1;
  while (1) {
    if (lambda <= minlambda) { /* bad luck; use full step */
      PLogInfo((PetscObject)snes,
         "SNESCubicLineSearch:Unable to find good step length! %d \n",count);
      PLogInfo((PetscObject)snes, 
         "SNESCubicLineSearch:f %g fnew %g ynorm %g lambda %g initial slope %g\n",fnorm,*gnorm,*ynorm,lambda,initslope);
      ierr = VecCopy(w,y); CHKERRQ(ierr);
      *flag = -1; break;
    }
    t1 = *gnorm - fnorm - lambda*initslope;
    t2 = gnormprev  - fnorm - lambdaprev*initslope;
    a = (t1/(lambda*lambda) - t2/(lambdaprev*lambdaprev))/(lambda-lambdaprev);
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
    ierr = VecCopy( x, w ); CHKERRQ(ierr);
    lambda = -lambda;
#if defined(PETSC_COMPLEX)
    clambda = lambda;
    ierr = VecAXPY(&clambda,y,w); CHKERRQ(ierr);
#else
    ierr = VecAXPY(&lambda,y,w); CHKERRQ(ierr);
#endif
    ierr = SNESComputeFunction(snes,w,g); CHKERRQ(ierr);
    ierr = VecNorm(g,NORM_2,gnorm); CHKERRQ(ierr);
    if (*gnorm <= fnorm + alpha*initslope) {      /* is reduction enough */
      ierr = VecCopy(w,y); CHKERRQ(ierr);
      PLogInfo((PetscObject)snes,
                "SNESCubicLineSearch: Cubically determined step, lambda %g\n",lambda);
      *flag = -1; break;
    }
    count++;
  }
  theend:
  PLogEventEnd(SNES_LineSearch,snes,x,f,g);
  return 0;
}
/* ------------------------------------------------------------------ */
/*@C
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
   to set this routine within the SNES_EQ_NLS method.  

.keywords: SNES, nonlinear, quadratic, line search

.seealso: SNESCubicLineSearch(), SNESNoLineSearch(), SNESSetLineSearchRoutine()
@*/
int SNESQuadraticLineSearch(SNES snes, Vec x, Vec f, Vec g, Vec y, Vec w,
                           double fnorm, double *ynorm, double *gnorm,int *flag)
{
  double  steptol,initslope,lambdaprev,gnormprev,maxstep,minlambda,alpha,lambda,lambdatemp;
#if defined(PETSC_COMPLEX)
  Scalar  cinitslope,clambda;
#endif
  int     ierr,count;
  SNES_LS *neP = (SNES_LS *) snes->data;
  Scalar  mone = -1.0,scale;

  PLogEventBegin(SNES_LineSearch,snes,x,f,g);
  *flag = 0;
  alpha   = neP->alpha;
  maxstep = neP->maxstep;
  steptol = neP->steptol;

  VecNorm(y, NORM_2,ynorm );
  if (*ynorm > maxstep) {	/* Step too big, so scale back */
    scale = maxstep/(*ynorm);
    ierr = VecScale(&scale,y); CHKERRQ(ierr);
    *ynorm = maxstep;
  }
  minlambda = steptol/(*ynorm);
  ierr = MatMult(snes->jacobian,y,w); CHKERRQ(ierr);
#if defined(PETSC_COMPLEX)
  ierr = VecDot(f,w,&cinitslope); CHKERRQ(ierr);
  initslope = real(cinitslope);
#else
  ierr = VecDot(f,w,&initslope); CHKERRQ(ierr);
#endif
  if (initslope > 0.0) initslope = -initslope;
  if (initslope == 0.0) initslope = -1.0;

  ierr = VecCopy(y,w); CHKERRQ(ierr);
  ierr = VecAYPX(&mone,x,w); CHKERRQ(ierr);
  ierr = SNESComputeFunction(snes,w,g); CHKERRQ(ierr);
  ierr = VecNorm(g,NORM_2,gnorm); CHKERRQ(ierr);
  if (*gnorm <= fnorm + alpha*initslope) {	/* Sufficient reduction */
    ierr = VecCopy(w,y); CHKERRQ(ierr);
    PLogInfo((PetscObject)snes,"SNESQuadraticLineSearch: Using full step\n");
    goto theend;
  }

  /* Fit points with quadratic */
  lambda = 1.0; count = 0;
  count = 1;
  while (1) {
    if (lambda <= minlambda) { /* bad luck; use full step */
      PLogInfo((PetscObject)snes,
          "SNESQuadraticLineSearch:Unable to find good step length! %d \n",count);
      PLogInfo((PetscObject)snes, 
      "SNESQuadraticLineSearch:f %g fnew %g ynorm %g lambda %g\n",fnorm,*gnorm,*ynorm,lambda);
      ierr = VecCopy(w,y); CHKERRQ(ierr);
      *flag = -1; break;
    }
    lambdatemp = -initslope/((*gnorm)*(*gnorm) - fnorm*fnorm - 2.0*initslope);
    lambdaprev = lambda;
    gnormprev = *gnorm;
    if (lambdatemp <= .1*lambda) { 
      lambda = .1*lambda; 
    } else lambda = lambdatemp;
    ierr = VecCopy(x,w); CHKERRQ(ierr);
    lambda = -lambda;
#if defined(PETSC_COMPLEX)
    clambda = lambda; ierr = VecAXPY(&clambda,y,w); CHKERRQ(ierr);
#else
    ierr = VecAXPY(&lambda,y,w); CHKERRQ(ierr);
#endif
    ierr = SNESComputeFunction(snes,w,g); CHKERRQ(ierr);
    ierr = VecNorm(g,NORM_2,gnorm); CHKERRQ(ierr);
    if (*gnorm <= fnorm + alpha*initslope) {      /* sufficient reduction */
      ierr = VecCopy(w,y); CHKERRQ(ierr);
      PLogInfo((PetscObject)snes,
        "SNESQuadraticLineSearch:Quadratically determined step, lambda %g\n",lambda);
      break;
    }
    count++;
  }
  theend:
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
                             double,double*,double*,int*))
{
  if ((snes)->type == SNES_EQ_NLS) ((SNES_LS *)(snes->data))->LineSearch = func;
  return 0;
}
/* ------------------------------------------------------------------ */
static int SNESPrintHelp_LS(SNES snes,char *p)
{
  SNES_LS *ls = (SNES_LS *)snes->data;

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
  SNES       snes = (SNES)obj;
  SNES_LS    *ls = (SNES_LS *)snes->data;
  FILE       *fd;
  char       *cstr;
  int        ierr;
  ViewerType vtype;

  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  if (vtype  == ASCII_FILE_VIEWER || vtype == ASCII_FILES_VIEWER) {  
    ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
    if (ls->LineSearch == SNESNoLineSearch) cstr = "SNESNoLineSearch";
    else if (ls->LineSearch == SNESQuadraticLineSearch) cstr = "SNESQuadraticLineSearch";
    else if (ls->LineSearch == SNESCubicLineSearch) cstr = "SNESCubicLineSearch";
    else cstr = "unknown";
    MPIU_fprintf(snes->comm,fd,"    line search variant: %s\n",cstr);
    MPIU_fprintf(snes->comm,fd,"    alpha=%g, maxstep=%g, steptol=%g\n",
                 ls->alpha,ls->maxstep,ls->steptol);
  }
  return 0;
}
/* ------------------------------------------------------------------ */
static int SNESSetFromOptions_LS(SNES snes)
{
  SNES_LS *ls = (SNES_LS *)snes->data;
  char    ver[16];
  double  tmp;
  int     ierr,flg;

  ierr = OptionsGetDouble(snes->prefix,"-snes_line_search_alpa",&tmp, &flg);CHKERRQ(ierr);
  if (flg) {
    ls->alpha = tmp;
  }
  ierr = OptionsGetDouble(snes->prefix,"-snes_line_search_maxstep",&tmp, &flg);CHKERRQ(ierr);
  if (flg) {
    ls->maxstep = tmp;
  }
  ierr = OptionsGetDouble(snes->prefix,"-snes_line_search_steptol",&tmp, &flg);CHKERRQ(ierr);
  if (flg) {
    ls->steptol = tmp;
  }
  ierr = OptionsGetString(snes->prefix,"-snes_line_search",ver,16, &flg); CHKERRQ(ierr);
  if (flg) {
    if (!PetscStrcmp(ver,"basic")) {
      SNESSetLineSearchRoutine(snes,SNESNoLineSearch);
    }
    else if (!PetscStrcmp(ver,"quadratic")) {
      SNESSetLineSearchRoutine(snes,SNESQuadraticLineSearch);
    }
    else if (!PetscStrcmp(ver,"cubic")) {
      SNESSetLineSearchRoutine(snes,SNESCubicLineSearch);
    }
    else {SETERRQ(1,"SNESSetFromOptions_LS:Unknown line search");}
  }
  return 0;
}
/* ------------------------------------------------------------ */
int SNESCreate_LS(SNES  snes )
{
  SNES_LS *neP;

  if (snes->method_class != SNES_NONLINEAR_EQUATIONS) 
    SETERRQ(1,"SNESCreate_LS:For SNES_NONLINEAR_EQUATIONS only");
  snes->type		= SNES_EQ_NLS;
  snes->setup		= SNESSetUp_LS;
  snes->solve		= SNESSolve_LS;
  snes->destroy		= SNESDestroy_LS;
  snes->converged	= SNESDefaultConverged;
  snes->printhelp       = SNESPrintHelp_LS;
  snes->setfromoptions  = SNESSetFromOptions_LS;
  snes->view            = SNESView_LS;

  neP			= PetscNew(SNES_LS);   CHKPTRQ(neP);
  PLogObjectMemory(snes,sizeof(SNES_LS));
  snes->data    	= (void *) neP;
  neP->alpha		= 1.e-4;
  neP->maxstep		= 1.e8;
  neP->steptol		= 1.e-12;
  neP->LineSearch       = SNESCubicLineSearch;
  return 0;
}




