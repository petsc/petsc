#ifndef lint
static char vcid[] = "$Id: snes.c,v 1.23 1995/10/19 22:28:49 curfman Exp bsmith $";
#endif

#include "draw.h"          /*I "draw.h"  I*/
#include "snesimpl.h"      /*I "snes.h"  I*/
#include "sys/nreg.h"      /*I  "sys/nreg.h"  I*/
#include "pinclude/pviewer.h"
#include <math.h>

extern int SNESGetMethodFromOptions_Private(SNES,SNESMethod*);
extern int SNESPrintMethods_Private(char*,char*);

/*@ 
   SNESView - Prints the SNES data structure.

   Input Parameters:
.  SNES - the SNES context
.  viewer - visualization context

   Options Database Key:
$  -snes_view : calls SNESView() at end of SNESSolve()

   Notes:
   The available visualization contexts include
$     STDOUT_VIEWER_SELF - standard output (default)
$     STDOUT_VIEWER_WORLD - synchronized standard
$       output where only the first processor opens
$       the file.  All other processors send their 
$       data to the first processor to print. 

   The user can open alternative vistualization contexts with
$    ViewerFileOpenASCII() - output to a specified file

.keywords: SNES, view

.seealso: ViewerFileOpenASCII()
@*/
int SNESView(SNES snes,Viewer viewer)
{
  PetscObject         vobj = (PetscObject) viewer;
  SNES_KSP_EW_ConvCtx *kctx;
  FILE                *fd;
  int                 ierr;
  SLES                sles;
  char                *method;

  if (vobj->cookie == VIEWER_COOKIE && (vobj->type == ASCII_FILE_VIEWER ||
                                        vobj->type == ASCII_FILES_VIEWER)) {
    ierr = ViewerFileGetPointer_Private(viewer,&fd); CHKERRQ(ierr);
    MPIU_fprintf(snes->comm,fd,"SNES Object:\n");
    SNESGetMethodName((SNESMethod)snes->type,&method);
    MPIU_fprintf(snes->comm,fd,"  method: %s\n",method);
    if (snes->view) (*snes->view)((PetscObject)snes,viewer);
    MPIU_fprintf(snes->comm,fd,
      "  maximum iterations=%d, maximum function evaluations=%d\n",
      snes->max_its,snes->max_funcs);
    MPIU_fprintf(snes->comm,fd,
    "  tolerances: relative=%g, absolute=%g, truncation=%g, solution=%g\n",
      snes->rtol, snes->atol, snes->trunctol, snes->xtol);
    if (snes->method_class == SNES_UNCONSTRAINED_MINIMIZATION)
      MPIU_fprintf(snes->comm,fd,"  min function tolerance=%g\n",snes->fmin);
    if (snes->ksp_ewconv) {
      kctx = (SNES_KSP_EW_ConvCtx *)snes->kspconvctx;
      if (kctx) {
        MPIU_fprintf(snes->comm,fd,
     "  Eisenstat-Walker computation of KSP relative tolerance (version %d)\n",
        kctx->version);
        MPIU_fprintf(snes->comm,fd,
          "    rtol_0=%g, rtol_max=%g, threshold=%g\n",kctx->rtol_0,
          kctx->rtol_max,kctx->threshold);
        MPIU_fprintf(snes->comm,fd,"    gamma=%g, alpha=%g, alpha2=%g\n",
          kctx->gamma,kctx->alpha,kctx->alpha2);
      }
    }
    SNESGetSLES(snes,&sles);
    ierr = SLESView(sles,viewer); CHKERRQ(ierr);
  }
  return 0;
}

/*@
   SNESSetFromOptions - Sets various SLES parameters from user options.

   Input Parameter:
.  snes - the SNES context

.keywords: SNES, nonlinear, set, options, database

.seealso: SNESPrintHelp()
@*/
int SNESSetFromOptions(SNES snes)
{
  SNESMethod method;
  double tmp;
  SLES   sles;
  int    ierr;
  int    version   = PETSC_DEFAULT;
  double rtol_0    = PETSC_DEFAULT;
  double rtol_max  = PETSC_DEFAULT;
  double gamma2    = PETSC_DEFAULT;
  double alpha     = PETSC_DEFAULT;
  double alpha2    = PETSC_DEFAULT;
  double threshold = PETSC_DEFAULT;

  PETSCVALIDHEADERSPECIFIC(snes,SNES_COOKIE);
  if (snes->setup_called)
    SETERRQ(1,"SNESSetFromOptions:Must call prior to SNESSetUp!");
  if (SNESGetMethodFromOptions_Private(snes,&method)) {
    SNESSetMethod(snes,method);
  }
  if (OptionsHasName(0,"-help"))  SNESPrintHelp(snes);
  if (OptionsGetDouble(snes->prefix,"-snes_stol",&tmp)) {
    SNESSetSolutionTolerance(snes,tmp);
  }
  if (OptionsGetDouble(snes->prefix,"-snes_ttol",&tmp)) {
    SNESSetTruncationTolerance(snes,tmp);
  }
  if (OptionsGetDouble(snes->prefix,"-snes_atol",&tmp)) {
    SNESSetAbsoluteTolerance(snes,tmp);
  }
  if (OptionsGetDouble(snes->prefix,"-snes_trtol",&tmp)) {
    SNESSetTrustRegionTolerance(snes,tmp);
  }
  if (OptionsGetDouble(snes->prefix,"-snes_rtol",&tmp)) {
    SNESSetRelativeTolerance(snes,tmp);
  }
  if (OptionsGetDouble(snes->prefix,"-snes_fmin",&tmp)) {
    SNESSetMinFunctionTolerance(snes,tmp);
  }
  OptionsGetInt(snes->prefix,"-snes_max_it",&snes->max_its);
  OptionsGetInt(snes->prefix,"-snes_max_funcs",&snes->max_funcs);
  if (OptionsHasName(snes->prefix,"-snes_ksp_ew_conv")) {
    snes->ksp_ewconv = 1;
  }
  OptionsGetInt(snes->prefix,"-snes_ksp_ew_version",&version);
  OptionsGetDouble(snes->prefix,"-snes_ksp_ew_rtol0",&rtol_0);
  OptionsGetDouble(snes->prefix,"-snes_ksp_ew_rtolmax",&rtol_max);
  OptionsGetDouble(snes->prefix,"-snes_ksp_ew_gamma",&gamma2);
  OptionsGetDouble(snes->prefix,"-snes_ksp_ew_alpha",&alpha);
  OptionsGetDouble(snes->prefix,"-snes_ksp_ew_alpha2",&alpha2);
  OptionsGetDouble(snes->prefix,"-snes_ksp_ew_threshold",&threshold);
  ierr = SNES_KSP_SetParametersEW(snes,version,rtol_0,rtol_max,gamma2,alpha,
                            alpha2,threshold); CHKERRQ(ierr);
  if (OptionsHasName(snes->prefix,"-snes_monitor")) {
    SNESSetMonitor(snes,SNESDefaultMonitor,0);
  }
  if (OptionsHasName(snes->prefix,"-snes_smonitor")) {
    SNESSetMonitor(snes,SNESDefaultSMonitor,0);
  }
  if (OptionsHasName(snes->prefix,"-snes_xmonitor")){
    int       rank = 0;
    DrawLGCtx lg;
    MPI_Initialized(&rank);
    if (rank) MPI_Comm_rank(snes->comm,&rank);
    if (!rank) {
      ierr = SNESLGMonitorCreate(0,0,0,0,300,300,&lg); CHKERRQ(ierr);
      ierr = SNESSetMonitor(snes,SNESLGMonitor,(void *)lg); CHKERRQ(ierr);
      PLogObjectParent(snes,lg);
    }
  }
  if (OptionsHasName(snes->prefix,"-snes_fd") && 
    snes->method_class == SNES_NONLINEAR_EQUATIONS) {
    ierr = SNESSetJacobian(snes,snes->jacobian,snes->jacobian_pre,
           SNESDefaultComputeJacobian,snes->funP); CHKERRQ(ierr);
  }
  if (OptionsHasName(snes->prefix,"-snes_mf") &&
    snes->method_class == SNES_NONLINEAR_EQUATIONS) {
    Mat J;
    ierr = SNESDefaultMatrixFreeMatCreate(snes,snes->vec_sol,&J);CHKERRQ(ierr);
    ierr = SNESSetJacobian(snes,J,J,0,snes->funP); CHKERRQ(ierr);
    PLogObjectParent(snes,J);
    snes->mfshell = J;
  }
  ierr = SNESGetSLES(snes,&sles); CHKERRQ(ierr);
  ierr = SLESSetFromOptions(sles); CHKERRQ(ierr);
  if (!snes->setfromoptions) return 0;
  return (*snes->setfromoptions)(snes);
}

/*@
   SNESPrintHelp - Prints all options for the SNES component.

   Input Parameter:
.  snes - the SNES context

   Options Database Keys:
$  -help, -h

.keywords: SNES, nonlinear, help

.seealso: SLESSetFromOptions()
@*/
int SNESPrintHelp(SNES snes)
{
  char    *prefix = "-";
  SNES_KSP_EW_ConvCtx *kctx = (SNES_KSP_EW_ConvCtx *)snes->kspconvctx;
  if (snes->prefix) prefix = snes->prefix;
  PETSCVALIDHEADERSPECIFIC(snes,SNES_COOKIE);
  MPIU_printf(snes->comm,"SNES options ----------------------------\n");
  SNESPrintMethods_Private(prefix,"snes_method");
  MPIU_printf(snes->comm," %ssnes_monitor: use default SNES monitor\n",prefix);
  MPIU_printf(snes->comm," %ssnes_view: view SNES info after each nonlinear solve\n",prefix);
  MPIU_printf(snes->comm," %ssnes_max_it its (default %d)\n",prefix,snes->max_its);
  MPIU_printf(snes->comm," %ssnes_stol tol (default %g)\n",prefix,snes->xtol);
  MPIU_printf(snes->comm," %ssnes_atol tol (default %g)\n",prefix,snes->atol);
  MPIU_printf(snes->comm," %ssnes_rtol tol (default %g)\n",prefix,snes->rtol);
  MPIU_printf(snes->comm," %ssnes_ttol tol (default %g)\n",prefix,snes->trunctol);
  MPIU_printf(snes->comm,
   " options for solving systems of nonlinear equations only:\n");
  MPIU_printf(snes->comm,"   %ssnes_fd: use finite differences for Jacobian\n",prefix);
  MPIU_printf(snes->comm,"   %ssnes_mf: use matrix-free Jacobian\n",prefix);
  MPIU_printf(snes->comm,"   %ssnes_ksp_ew_conv: use Eisenstat-Walker computation of KSP rtol. Params are:\n",prefix);
  MPIU_printf(snes->comm,
   "     %ssnes_ksp_ew_version version (1 or 2, default is %d)\n",
   prefix,kctx->version);
  MPIU_printf(snes->comm,
   "     %ssnes_ksp_ew_rtol0 rtol0 (0 <= rtol0 < 1, default %g)\n",
   prefix,kctx->rtol_0);
  MPIU_printf(snes->comm,
   "     %ssnes_ksp_ew_rtolmax rtolmax (0 <= rtolmax < 1, default %g)\n",
   prefix,kctx->rtol_max);
  MPIU_printf(snes->comm,
   "     %ssnes_ksp_ew_gamma gamma (0 <= gamma <= 1, default %g)\n",
   prefix,kctx->gamma);
  MPIU_printf(snes->comm,
   "     %ssnes_ksp_ew_alpha alpha (1 < alpha <= 2, default %g)\n",
   prefix,kctx->alpha);
  MPIU_printf(snes->comm,
   "     %ssnes_ksp_ew_alpha2 alpha2 (default %g)\n",
   prefix,kctx->alpha2);
  MPIU_printf(snes->comm,
   "     %ssnes_ksp_ew_threshold threshold (0 < threshold < 1, default %g)\n",
   prefix,kctx->threshold);
  MPIU_printf(snes->comm,
   " options for solving unconstrained minimization problems only:\n");
  MPIU_printf(snes->comm,"   %ssnes_fmin tol (default %g)\n",prefix,snes->fmin);
  MPIU_printf(snes->comm," Run program with %ssnes_method method -help for help on ",prefix);
  MPIU_printf(snes->comm,"a particular method\n");
  if (snes->printhelp) (*snes->printhelp)(snes);
  return 0;
}
/*@
   SNESSetApplicationContext - Sets the optional user-defined context for 
   the nonlinear solvers.  

   Input Parameters:
.  snes - the SNES context
.  usrP - optional user context

.keywords: SNES, nonlinear, set, application, context

.seealso: SNESGetApplicationContext()
@*/
int SNESSetApplicationContext(SNES snes,void *usrP)
{
  PETSCVALIDHEADERSPECIFIC(snes,SNES_COOKIE);
  snes->user		= usrP;
  return 0;
}
/*@C
   SNESGetApplicationContext - Gets the user-defined context for the 
   nonlinear solvers.  

   Input Parameter:
.  snes - SNES context

   Output Parameter:
.  usrP - user context

.keywords: SNES, nonlinear, get, application, context

.seealso: SNESSetApplicationContext()
@*/
int SNESGetApplicationContext( SNES snes,  void **usrP )
{
  PETSCVALIDHEADERSPECIFIC(snes,SNES_COOKIE);
  *usrP = snes->user;
  return 0;
}
/*@
   SNESGetIterationNumber - Gets the current iteration number of the
   nonlinear solver.

   Input Parameter:
.  snes - SNES context

   Output Parameter:
.  iter - iteration number

.keywords: SNES, nonlinear, get, iteration, number
@*/
int SNESGetIterationNumber(SNES snes,int* iter)
{
  PETSCVALIDHEADERSPECIFIC(snes,SNES_COOKIE);
  *iter = snes->iter;
  return 0;
}
/*@
   SNESGetFunctionNorm - Gets the norm of the current function that was set
   with SNESSSetFunction().

   Input Parameter:
.  snes - SNES context

   Output Parameter:
.  fnorm - 2-norm of function

   Note:
   SNESGetFunctionNorm() is valid for SNES_NONLINEAR_EQUATIONS methods only.

.keywords: SNES, nonlinear, get, function, norm
@*/
int SNESGetFunctionNorm(SNES snes,Scalar *fnorm)
{
  PETSCVALIDHEADERSPECIFIC(snes,SNES_COOKIE);
  if (snes->method_class != SNES_NONLINEAR_EQUATIONS) SETERRQ(1,
    "SNESGetFunctionNorm:For SNES_NONLINEAR_EQUATIONS only");
  *fnorm = snes->norm;
  return 0;
}
/*@
   SNESGetGradientNorm - Gets the norm of the current gradient that was set
   with SNESSSetGradient().

   Input Parameter:
.  snes - SNES context

   Output Parameter:
.  fnorm - 2-norm of gradient

   Note:
   SNESGetGradientNorm() is valid for SNES_UNCONSTRAINED_MINIMIZATION 
   methods only.

.keywords: SNES, nonlinear, get, gradient, norm
@*/
int SNESGetGradientNorm(SNES snes,Scalar *gnorm)
{
  PETSCVALIDHEADERSPECIFIC(snes,SNES_COOKIE);
  if (snes->method_class != SNES_UNCONSTRAINED_MINIMIZATION) SETERRQ(1,
    "SNESGetGradientNorm:For SNES_UNCONSTRAINED_MINIMIZATION only");
  *gnorm = snes->norm;
  return 0;
}
/*@
   SNESGetNumberUnsuccessfulSteps - Gets the number of unsuccessful steps
   attempted by the nonlinear solver.

   Input Parameter:
.  snes - SNES context

   Output Parameter:
.  nfails - number of unsuccessful steps attempted

.keywords: SNES, nonlinear, get, number, unsuccessful, steps
@*/
int SNESGetNumberUnsuccessfulSteps(SNES snes,int* nfails)
{
  PETSCVALIDHEADERSPECIFIC(snes,SNES_COOKIE);
  *nfails = snes->nfailures;
  return 0;
}
/*@C
   SNESGetSLES - Returns the SLES context for a SNES solver.

   Input Parameter:
.  snes - the SNES context

   Output Parameter:
.  sles - the SLES context

   Notes:
   The user can then directly manipulate the SLES context to set various
   options, etc.  Likewise, the user can then extract and manipulate the 
   KSP and PC contexts as well.

.keywords: SNES, nonlinear, get, SLES, context

.seealso: SLESGetPC(), SLESGetKSP()
@*/
int SNESGetSLES(SNES snes,SLES *sles)
{
  PETSCVALIDHEADERSPECIFIC(snes,SNES_COOKIE);
  *sles = snes->sles;
  return 0;
}
/* -----------------------------------------------------------*/
/*@C
   SNESCreate - Creates a nonlinear solver context.

   Input Parameter:
.  comm - MPI communicator
.  type - type of method, one of
$    SNES_NONLINEAR_EQUATIONS 
$      (for systems of nonlinear equations)
$    SNES_UNCONSTRAINED_MINIMIZATION
$      (for unconstrained minimization)

   Output Parameter:
.  outsnes - the new SNES context

.keywords: SNES, nonlinear, create, context

.seealso: SNESSetUp(), SNESSolve(), SNESDestroy()
@*/
int SNESCreate(MPI_Comm comm,SNESType type,SNES *outsnes)
{
  int  ierr;
  SNES snes;
  SNES_KSP_EW_ConvCtx *kctx;
  *outsnes = 0;
  PETSCHEADERCREATE(snes,_SNES,SNES_COOKIE,SNES_UNKNOWN_METHOD,comm);
  PLogObjectCreate(snes);
  snes->max_its           = 50;
  snes->max_funcs	  = 1000;
  snes->norm		  = 0.0;
  snes->rtol		  = 1.e-8;
  snes->atol		  = 1.e-10;
  snes->xtol		  = 1.e-8;
  snes->trunctol	  = 1.e-12;
  snes->nfuncs            = 0;
  snes->nfailures         = 0;
  snes->monitor           = 0;
  snes->data              = 0;
  snes->view              = 0;
  snes->computeumfunction = 0;
  snes->umfunP            = 0;
  snes->fc                = 0;
  snes->deltatol          = 1.e-12;
  snes->fmin              = -1.e30;
  snes->method_class      = type;
  snes->set_method_called = 0;
  snes->setup_called      = 0;
  snes->ksp_ewconv        = 0;

  /* Create context to compute Eisenstat-Walker relative tolerance for KSP */
  kctx = PETSCNEW(SNES_KSP_EW_ConvCtx); CHKPTRQ(kctx);
  snes->kspconvctx  = (void*)kctx;
  kctx->version     = 2;
  kctx->rtol_0      = .3; /* Eisenstat and Walker suggest rtol_0=.5, but 
                             this was too large for some test cases */
  kctx->rtol_last   = 0;
  kctx->rtol_max    = .9;
  kctx->gamma       = 1.0;
  kctx->alpha2      = .5*(1.0 + sqrt(5.0));
  kctx->alpha       = kctx->alpha2;
  kctx->threshold   = .1;
  kctx->lresid_last = 0;
  kctx->norm_last   = 0;

  ierr = SLESCreate(comm,&snes->sles); CHKERRQ(ierr);
  PLogObjectParent(snes,snes->sles)
  *outsnes = snes;
  return 0;
}

/* --------------------------------------------------------------- */
/*@C
   SNESSetFunction - Sets the function evaluation routine and function 
   vector for use by the SNES routines in solving systems of nonlinear
   equations.

   Input Parameters:
.  snes - the SNES context
.  func - function evaluation routine
.  resid_neg - indicator whether func evaluates f or -f. 
   If resid_neg is NEGATIVE_FUNCTION_VALUE, then func evaluates -f; otherwise, 
   func evaluates f.
.  ctx - optional user-defined function context 
.  r - vector to store function value

   Calling sequence of func:
.  func (SNES, Vec x, Vec f, void *ctx);

.  x - input vector
.  f - function vector or its negative
.  ctx - optional user-defined context for private data for the 
         function evaluation routine (may be null)

   Notes:
   The Newton-like methods typically solve linear systems of the form
$      f'(x) x = -f(x),
$  where f'(x) denotes the Jacobian matrix and f(x) is the function.
   By setting resid_neg = 1, the user can supply -f(x) directly.

   SNESSetFunction() is valid for SNES_NONLINEAR_EQUATIONS methods only.
   Analogous routines for SNES_UNCONSTRAINED_MINIMIZATION methods are
   SNESSetMinimizationFunction() and SNESSetGradient();

.keywords: SNES, nonlinear, set, function

.seealso: SNESGetFunction(), SNESSetJacobian(), SNESSetSolution()
@*/
int SNESSetFunction( SNES snes, Vec r, int (*func)(SNES,Vec,Vec,void*),
                     void *ctx,SNESFunctionSign rneg)
{
  PETSCVALIDHEADERSPECIFIC(snes,SNES_COOKIE);
  if (snes->method_class != SNES_NONLINEAR_EQUATIONS) SETERRQ(1,
    "SNESSetFunction:For SNES_NONLINEAR_EQUATIONS only");
  snes->computefunction     = func; 
  snes->rsign               = (int) rneg;
  snes->vec_func            = snes->vec_func_always = r;
  snes->funP                = ctx;
  return 0;
}

/*@
   SNESComputeFunction - Computes the function that has been set with
   SNESSetFunction().  

   Input Parameters:
.  snes - the SNES context
.  x - input vector

   Output Parameter:
.  y - function vector or its negative, as set by SNESSetFunction()

   Notes:
   SNESComputeFunction() is valid for SNES_NONLINEAR_EQUATIONS methods only.
   Analogous routines for SNES_UNCONSTRAINED_MINIMIZATION methods are
   SNESComputeMinimizationFunction() and SNESComputeGradient();

.keywords: SNES, nonlinear, compute, function

.seealso: SNESSetFunction()
@*/
int SNESComputeFunction(SNES snes,Vec x, Vec y)
{
  int    ierr;
  Scalar mone = -1.0;

  PLogEventBegin(SNES_FunctionEval,snes,x,y,0);
  ierr = (*snes->computefunction)(snes,x,y,snes->funP); CHKERRQ(ierr);
  if (!snes->rsign) {
    ierr = VecScale(&mone,y); CHKERRQ(ierr);
  }
  PLogEventEnd(SNES_FunctionEval,snes,x,y,0);
  return 0;
}

/*@C
   SNESSetMinimizationFunction - Sets the function evaluation routine for 
   unconstrained minimization.

   Input Parameters:
.  snes - the SNES context
.  func - function evaluation routine
.  ctx - optional user-defined function context 

   Calling sequence of func:
.  func (SNES snes,Vec x,double *f,void *ctx);

.  x - input vector
.  f - function
.  ctx - optional user-defined context for private data for the 
         function evaluation routine (may be null)

   Notes:
   SNESSetMinimizationFunction() is valid for SNES_UNCONSTRAINED_MINIMIZATION
   methods only. An analogous routine for SNES_NONLINEAR_EQUATIONS methods is
   SNESSetFunction().

.keywords: SNES, nonlinear, set, minimization, function

.seealso:  SNESGetMinimizationFunction(), SNESSetHessian(), SNESSetGradient(), 
           SNESSetSolution()
@*/
int SNESSetMinimizationFunction(SNES snes,int (*func)(SNES,Vec,double*,void*),
                      void *ctx)
{
  PETSCVALIDHEADERSPECIFIC(snes,SNES_COOKIE);
  if (snes->method_class != SNES_UNCONSTRAINED_MINIMIZATION) SETERRQ(1,
    "SNESSetMinimizationFunction:Only for SNES_UNCONSTRAINED_MINIMIZATION");
  snes->computeumfunction   = func; 
  snes->umfunP              = ctx;
  return 0;
}

/*@
   SNESComputeMinimizationFunction - Computes the function that has been
   set with SNESSetMinimizationFunction().

   Input Parameters:
.  snes - the SNES context
.  x - input vector

   Output Parameter:
.  y - function value

   Notes:
   SNESComputeMinimizationFunction() is valid only for 
   SNES_UNCONSTRAINED_MINIMIZATION methods. An analogous routine for 
   SNES_NONLINEAR_EQUATIONS methods is SNESComputeFunction().
@*/
int SNESComputeMinimizationFunction(SNES snes,Vec x,double *y)
{
  int    ierr;
  if (snes->method_class != SNES_UNCONSTRAINED_MINIMIZATION) SETERRQ(1,
    "SNESComputeMinimizationFunction:Only for SNES_UNCONSTRAINED_MINIMIZATION");
  PLogEventBegin(SNES_MinimizationFunctionEval,snes,x,y,0);
  ierr = (*snes->computeumfunction)(snes,x,y,snes->umfunP); CHKERRQ(ierr);
  PLogEventEnd(SNES_MinimizationFunctionEval,snes,x,y,0);
  return 0;
}

/*@C
   SNESSetGradient - Sets the gradient evaluation routine and gradient
   vector for use by the SNES routines.

   Input Parameters:
.  snes - the SNES context
.  func - function evaluation routine
.  ctx - optional user-defined function context 
.  r - vector to store gradient value

   Calling sequence of func:
.  func (SNES, Vec x, Vec g, void *ctx);

.  x - input vector
.  g - gradient vector
.  ctx - optional user-defined context for private data for the 
         function evaluation routine (may be null)

   Notes:
   SNESSetMinimizationFunction() is valid for SNES_UNCONSTRAINED_MINIMIZATION
   methods only. An analogous routine for SNES_NONLINEAR_EQUATIONS methods is
   SNESSetFunction().

.keywords: SNES, nonlinear, set, function

.seealso: SNESGetGradient(), SNESSetHessian(), SNESSetMinimizationFunction(),
          SNESSetSolution()
@*/
int SNESSetGradient(SNES snes,Vec r,int (*func)(SNES,Vec,Vec,void*),
                     void *ctx)
{
  PETSCVALIDHEADERSPECIFIC(snes,SNES_COOKIE);
  if (snes->method_class != SNES_UNCONSTRAINED_MINIMIZATION) SETERRQ(1,
    "SNESSetGradient:For SNES_UNCONSTRAINED_MINIMIZATION only");
  snes->computefunction     = func;
  snes->vec_func            = snes->vec_func_always = r;
  snes->funP                = ctx;
  return 0;
}

/*@
   SNESComputeGradient - Computes the gradient that has been
   set with SNESSetGradient().

   Input Parameters:
.  snes - the SNES context
.  x - input vector

   Output Parameter:
.  y - gradient vector

   Notes:
   SNESComputeGradient() is valid only for 
   SNES_UNCONSTRAINED_MINIMIZATION methods. An analogous routine for 
   SNES_NONLINEAR_EQUATIONS methods is SNESComputeFunction().
@*/
int SNESComputeGradient(SNES snes,Vec x, Vec y)
{
  int    ierr;
  if (snes->method_class != SNES_UNCONSTRAINED_MINIMIZATION) SETERRQ(1,
    "SNESComputeGradient:For SNES_UNCONSTRAINED_MINIMIZATION only");
  PLogEventBegin(SNES_GradientEval,snes,x,y,0);
  ierr = (*snes->computefunction)(snes,x,y,snes->funP); CHKERRQ(ierr);
  PLogEventEnd(SNES_GradientEval,snes,x,y,0);
  return 0;
}

int SNESComputeJacobian(SNES snes,Vec X,Mat *A,Mat *B,MatStructure *flg)
{
  int    ierr;
  if (snes->method_class != SNES_NONLINEAR_EQUATIONS) SETERRQ(1,
    "SNESComputeJacobian: For SNES_NONLINEAR_EQUATIONS only");
  if (!snes->computejacobian) return 0;
  PLogEventBegin(SNES_JacobianEval,snes,X,*A,*B);
  ierr = (*snes->computejacobian)(snes,X,A,B,flg,snes->jacP); CHKERRQ(ierr);
  PLogEventEnd(SNES_JacobianEval,snes,X,*A,*B);
  return 0;
}

int SNESComputeHessian(SNES snes,Vec X,Mat *A,Mat *B,MatStructure *flg)
{
  int    ierr;
  if (snes->method_class != SNES_UNCONSTRAINED_MINIMIZATION) SETERRQ(1,
    "SNESComputeHessian:For SNES_UNCONSTRAINED_MINIMIZATION only");
  if (!snes->computejacobian) return 0;
  PLogEventBegin(SNES_HessianEval,snes,X,*A,*B);
  ierr = (*snes->computejacobian)(snes,X,A,B,flg,snes->jacP); CHKERRQ(ierr);
  PLogEventEnd(SNES_HessianEval,snes,X,*A,*B);
  return 0;
}

/*@C
   SNESSetJacobian - Sets the function to compute Jacobian as well as the
   location to store it.

   Input Parameters:
.  snes - the SNES context
.  A - Jacobian matrix
.  B - preconditioner matrix (usually same as the Jacobian)
.  func - Jacobian evaluation routine
.  ctx - optional user-defined context for private data for the 
         Jacobian evaluation routine (may be null)

   Calling sequence of func:
.  func (SNES,Vec x,Mat *A,Mat *B,int *flag,void *ctx);

.  x - input vector
.  A - Jacobian matrix
.  B - preconditioner matrix, usually the same as A
.  flag - flag indicating information about matrix structure,
   same as flag in SLESSetOperators()
.  ctx - optional user-defined Jacobian context

   Notes: 
   The function func() takes Mat * as the matrix arguments rather than Mat.  
   This allows the Jacobian evaluation routine to replace A and/or B with a 
   completely new new matrix structure (not just different matrix elements)
   when appropriate, for instance, if the nonzero structure is changing
   throughout the global iterations.

.keywords: SNES, nonlinear, set, Jacobian, matrix

.seealso: SNESSetFunction(), SNESSetSolution()
@*/
int SNESSetJacobian(SNES snes,Mat A,Mat B,int (*func)(SNES,Vec,Mat*,Mat*,
                    MatStructure*,void*),void *ctx)
{
  PETSCVALIDHEADERSPECIFIC(snes,SNES_COOKIE);
  if (snes->method_class != SNES_NONLINEAR_EQUATIONS) SETERRQ(1,
    "SNESSetJacobian:For SNES_NONLINEAR_EQUATIONS only");
  snes->computejacobian = func;
  snes->jacP            = ctx;
  snes->jacobian        = A;
  snes->jacobian_pre    = B;
  return 0;
}
/*@C
   SNESSetHessian - Sets the function to compute Hessian as well as the
   location to store it.

   Input Parameters:
.  snes - the SNES context
.  A - Hessian matrix
.  B - preconditioner matrix (usually same as the Hessian)
.  func - Jacobian evaluation routine
.  ctx - optional user-defined context for private data for the 
         Hessian evaluation routine (may be null)

   Calling sequence of func:
.  func (SNES,Vec x,Mat *A,Mat *B,int *flag,void *ctx);

.  x - input vector
.  A - Hessian matrix
.  B - preconditioner matrix, usually the same as A
.  flag - flag indicating information about matrix structure,
   same as flag in SLESSetOperators()
.  ctx - optional user-defined Hessian context

   Notes: 
   The function func() takes Mat * as the matrix arguments rather than Mat.  
   This allows the Hessian evaluation routine to replace A and/or B with a 
   completely new new matrix structure (not just different matrix elements)
   when appropriate, for instance, if the nonzero structure is changing
   throughout the global iterations.

.keywords: SNES, nonlinear, set, Hessian, matrix

.seealso: SNESSetMinimizationFunction(), SNESSetSolution(), SNESSetGradient()
@*/
int SNESSetHessian(SNES snes,Mat A,Mat B,int (*func)(SNES,Vec,Mat*,Mat*,
                    MatStructure*,void*),void *ctx)
{
  PETSCVALIDHEADERSPECIFIC(snes,SNES_COOKIE);
  if (snes->method_class != SNES_UNCONSTRAINED_MINIMIZATION) SETERRQ(1,
    "SNESSetHessian:For SNES_UNCONSTRAINED_MINIMIZATION only");
  snes->computejacobian = func;
  snes->jacP            = ctx;
  snes->jacobian        = A;
  snes->jacobian_pre    = B;
  return 0;
}

/* ----- Routines to initialize and destroy a nonlinear solver ---- */

/*@
   SNESSetUp - Sets up the internal data structures for the later use
   of a nonlinear solver.  Call SNESSetUp() after calling SNESCreate()
   and optional routines of the form SNESSetXXX(), but before calling 
   SNESSolve().  

   Input Parameter:
.  snes - the SNES context

.keywords: SNES, nonlinear, setup

.seealso: SNESCreate(), SNESSolve(), SNESDestroy()
@*/
int SNESSetUp(SNES snes)
{
  int ierr;
  PETSCVALIDHEADERSPECIFIC(snes,SNES_COOKIE);
  if (!snes->vec_sol)
    SETERRQ(1,"SNESSetUp:Must call SNESSetSolution() first");

  if ((snes->method_class == SNES_NONLINEAR_EQUATIONS)) {
    if (!snes->set_method_called)
      {ierr = SNESSetMethod(snes,SNES_EQ_NLS); CHKERRQ(ierr);}
    if (!snes->vec_func) SETERRQ(1,
      "SNESSetUp:Must call SNESSetFunction() first");
    if (!snes->computefunction) SETERRQ(1,
      "SNESSetUp:Must call SNESSetFunction() first");
    if (!snes->jacobian) SETERRQ(1,
      "SNESSetUp:Must call SNESSetJacobian() first");

    /* Set the KSP stopping criterion to use the Eisenstat-Walker method */
    if (snes->ksp_ewconv && snes->type != SNES_EQ_NTR) {
      SLES sles; KSP ksp;
      ierr = SNESGetSLES(snes,&sles); CHKERRQ(ierr);
      ierr = SLESGetKSP(sles,&ksp); CHKERRQ(ierr);
      ierr = KSPSetConvergenceTest(ksp,SNES_KSP_EW_Converged_Private,
             (void *)snes); CHKERRQ(ierr);
    }
  } else if ((snes->method_class == SNES_UNCONSTRAINED_MINIMIZATION)) {
    if (!snes->set_method_called)
      {ierr = SNESSetMethod(snes,SNES_UM_NTR); CHKERRQ(ierr);}
    if (!snes->vec_func) SETERRQ(1,
     "SNESSetUp:Must call SNESSetGradient() first");
    if (!snes->computefunction) SETERRQ(1,
      "SNESSetUp:Must call SNESSetGradient() first");
    if (!snes->computeumfunction) SETERRQ(1,
      "SNESSetUp:Must call SNESSetMinimizationFunction() first");
    if (!snes->jacobian) SETERRQ(1,
      "SNESSetUp:Must call SNESSetHessian() first");
  } else SETERRQ(1,"SNESSetUp:Unknown method class");
  if (snes->setup) {ierr = (*snes->setup)(snes); CHKERRQ(ierr);}
  snes->setup_called = 1;
  return 0;
}

/*@C
   SNESDestroy - Destroys the nonlinear solver context that was created
   with SNESCreate().

   Input Parameter:
.  snes - the SNES context

.keywords: SNES, nonlinear, destroy

.seealso: SNESCreate(), SNESSetUp(), SNESSolve()
@*/
int SNESDestroy(SNES snes)
{
  int ierr;
  PETSCVALIDHEADERSPECIFIC(snes,SNES_COOKIE);
  ierr = (*(snes)->destroy)((PetscObject)snes); CHKERRQ(ierr);
  if (snes->kspconvctx) PETSCFREE(snes->kspconvctx);
  if (snes->mfshell) MatDestroy(snes->mfshell);
  ierr = SLESDestroy(snes->sles); CHKERRQ(ierr);
  PLogObjectDestroy((PetscObject)snes);
  PETSCHEADERDESTROY((PetscObject)snes);
  return 0;
}

/* ----------- Routines to set solver parameters ---------- */

/*@
   SNESSetMaxIterations - Sets the maximum number of global iterations to use.

   Input Parameters:
.  snes - the SNES context
.  maxits - maximum number of iterations to use

   Options Database Key:
$  -snes_max_it  maxits

   Note:
   The default maximum number of iterations is 50.

.keywords: SNES, nonlinear, set, maximum, iterations

.seealso: SNESSetMaxFunctionEvaluations()
@*/
int SNESSetMaxIterations(SNES snes,int maxits)
{
  PETSCVALIDHEADERSPECIFIC(snes,SNES_COOKIE);
  snes->max_its = maxits;
  return 0;
}

/*@
   SNESSetMaxFunctionEvaluations - Sets the maximum number of function
   evaluations to use.

   Input Parameters:
.  snes - the SNES context
.  maxf - maximum number of function evaluations

   Options Database Key:
$  -snes_max_funcs maxf

   Note:
   The default maximum number of function evaluations is 1000.

.keywords: SNES, nonlinear, set, maximum, function, evaluations

.seealso: SNESSetMaxIterations()
@*/
int SNESSetMaxFunctionEvaluations(SNES snes,int maxf)
{
  PETSCVALIDHEADERSPECIFIC(snes,SNES_COOKIE);
  snes->max_funcs = maxf;
  return 0;
}

/*@
   SNESSetRelativeTolerance - Sets the relative convergence tolerance.  

   Input Parameters:
.  snes - the SNES context
.  rtol - tolerance
   
   Options Database Key: 
$    -snes_rtol tol

.keywords: SNES, nonlinear, set, relative, convergence, tolerance
 
.seealso: SNESSetAbsoluteTolerance(), SNESSetSolutionTolerance(),
           SNESSetTruncationTolerance()
@*/
int SNESSetRelativeTolerance(SNES snes,double rtol)
{
  PETSCVALIDHEADERSPECIFIC(snes,SNES_COOKIE);
  snes->rtol = rtol;
  return 0;
}

/*@
   SNESSetTrustRegionTolerance - Sets the trust region parameter tolerance.  

   Input Parameters:
.  snes - the SNES context
.  tol - tolerance
   
   Options Database Key: 
$    -snes_trtol tol

.keywords: SNES, nonlinear, set, trust region, tolerance
 
.seealso: SNESSetAbsoluteTolerance(), SNESSetSolutionTolerance(),
           SNESSetTruncationTolerance()
@*/
int SNESSetTrustRegionTolerance(SNES snes,double tol)
{
  PETSCVALIDHEADERSPECIFIC(snes,SNES_COOKIE);
  snes->deltatol = tol;
  return 0;
}

/*@
   SNESSetAbsoluteTolerance - Sets the absolute convergence tolerance.  

   Input Parameters:
.  snes - the SNES context
.  atol - tolerance

   Options Database Key: 
$    -snes_atol tol

.keywords: SNES, nonlinear, set, absolute, convergence, tolerance

.seealso: SNESSetRelativeTolerance(), SNESSetSolutionTolerance(),
           SNESSetTruncationTolerance()
@*/
int SNESSetAbsoluteTolerance(SNES snes,double atol)
{
  PETSCVALIDHEADERSPECIFIC(snes,SNES_COOKIE);
  snes->atol = atol;
  return 0;
}

/*@
   SNESSetTruncationTolerance - Sets the tolerance that may be used by the
   step routines to control the accuracy of the step computation.

   Input Parameters:
.  snes - the SNES context
.  tol - tolerance

   Options Database Key: 
$    -snes_ttol tol

   Notes:
   If the step computation involves an application of the inverse
   Jacobian (or Hessian), this parameter may be used to control the 
   accuracy of that application. 

.keywords: SNES, nonlinear, set, truncation, tolerance

.seealso: SNESSetRelativeTolerance(), SNESSetSolutionTolerance(),
          SNESSetAbsoluteTolerance()
@*/
int SNESSetTruncationTolerance(SNES snes,double tol)
{
  PETSCVALIDHEADERSPECIFIC(snes,SNES_COOKIE);
  snes->trunctol = tol;
  return 0;
}

/*@
   SNESSetSolutionTolerance - Sets the convergence tolerance in terms of 
   the norm of the change in the solution between steps.

   Input Parameters:
.  snes - the SNES context
.  tol - tolerance

   Options Database Key: 
$    -snes_stol tol

.keywords: SNES, nonlinear, set, solution, tolerance

.seealso: SNESSetTruncationTolerance(), SNESSetRelativeTolerance(),
          SNESSetAbsoluteTolerance()
@*/
int SNESSetSolutionTolerance( SNES snes, double tol )
{
  PETSCVALIDHEADERSPECIFIC(snes,SNES_COOKIE);
  snes->xtol = tol;
  return 0;
}

/*@
   SNESSetMinFunctionTolerance - Sets the minimum allowable function tolerance
   for unconstrained minimization solvers.
   
   Input Parameters:
.  snes - the SNES context
.  ftol - minimum function tolerance

   Options Database Key: 
$    -snes_fmin ftol

   Note:
   SNESSetMinFunctionTolerance() is valid for SNES_UNCONSTRAINED_MINIMIZATION
   methods only.

.keywords: SNES, nonlinear, set, minimum, convergence, function, tolerance

.seealso: SNESSetRelativeTolerance(), SNESSetSolutionTolerance(),
           SNESSetTruncationTolerance()
@*/
int SNESSetMinFunctionTolerance(SNES snes,double ftol)
{
  PETSCVALIDHEADERSPECIFIC(snes,SNES_COOKIE);
  snes->fmin = ftol;
  return 0;
}



/* ---------- Routines to set various aspects of nonlinear solver --------- */

/*@C
   SNESSetSolution - Sets the initial guess routine and solution vector
   for use by the SNES routines.

   Input Parameters:
.  snes - the SNES context
.  x - the solution vector
.  func - optional routine to compute an initial guess (may be null)
.  ctx - optional user-defined context for private data for the 
         initial guess routine (may be null)

   Calling sequence of func:
   int guess(Vec x, void *ctx)

.  x - input vector
.  ctx - optional user-defined initial guess context 

   Note:
   If no initial guess routine is indicated, an initial guess of zero 
   will be used.

.keywords: SNES, nonlinear, set, solution, initial guess

.seealso: SNESGetSolution(), SNESSetJacobian(), SNESSetFunction()
@*/
int SNESSetSolution(SNES snes,Vec x,int (*func)(SNES,Vec,void*),void *ctx)
{
  PETSCVALIDHEADERSPECIFIC(snes,SNES_COOKIE);
  snes->vec_sol             = snes->vec_sol_always = x;
  snes->computeinitialguess = func;
  snes->gusP                = ctx;
  return 0;
}

/* ------------ Routines to set performance monitoring options ----------- */

/*@C
   SNESSetMonitor - Sets the function that is to be used at every
   iteration of the nonlinear solver to display the iteration's 
   progress.   

   Input Parameters:
.  snes - the SNES context
.  func - monitoring routine
.  mctx - optional user-defined context for private data for the 
          monitor routine (may be null)

   Calling sequence of func:
   int func((SNES snes,int its, Vec x,Vec f,
             double norm,void *mctx)

$    snes - the SNES context
$    its - iteration number
$    mctx - optional monitoring context
$
$ SNES_NONLINEAR_EQUATIONS methods:
$    norm - 2-norm function value (may be estimated)
$
$ SNES_UNCONSTRAINED_MINIMIZATION methods:
$    norm - 2-norm gradient value (may be estimated)

.keywords: SNES, nonlinear, set, monitor

.seealso: SNESDefaultMonitor()
@*/
int SNESSetMonitor( SNES snes, int (*func)(SNES,int,double,void*), 
                    void *mctx )
{
  snes->monitor = func;
  snes->monP    = (void*)mctx;
  return 0;
}

/*@C
   SNESSetConvergenceTest - Sets the function that is to be used 
   to test for convergence of the nonlinear iterative solution.   

   Input Parameters:
.  snes - the SNES context
.  func - routine to test for convergence
.  cctx - optional context for private data for the convergence routine 
          (may be null)

   Calling sequence of func:
   int func (SNES snes,double xnorm,double gnorm,
             double f,void *cctx)

$    snes - the SNES context
$    cctx - optional convergence context
$    xnorm - 2-norm of current iterate
$
$ SNES_NONLINEAR_EQUATIONS methods:
$    gnorm - 2-norm of current step 
$    f - 2-norm of function
$
$ SNES_UNCONSTRAINED_MINIMIZATION methods:
$    gnorm - 2-norm of current gradient
$    f - function value

.keywords: SNES, nonlinear, set, convergence, test

.seealso: SNESDefaultConverged()
@*/
int SNESSetConvergenceTest(SNES snes,
          int (*func)(SNES,double,double,double,void*),void *cctx)
{
  (snes)->converged = func;
  (snes)->cnvP      = cctx;
  return 0;
}

/*
   SNESScaleStep_Private - Scales a step so that its length is less than the
   positive parameter delta.

    Input Parameters:
.   snes - the SNES context
.   y - approximate solution of linear system
.   fnorm - 2-norm of current function
.   delta - trust region size

    Output Parameters:
.   gpnorm - predicted function norm at the new point, assuming local 
    linearization.  The value is zero if the step lies within the trust 
    region, and exceeds zero otherwise.
.   ynorm - 2-norm of the step

    Note:
    For non-trust region methods such as SNES_EQ_NLS, the parameter delta 
    is set to be the maximum allowable step size.  

.keywords: SNES, nonlinear, scale, step
*/
int SNESScaleStep_Private(SNES snes,Vec y,double *fnorm,double *delta, 
                  double *gpnorm,double *ynorm)
{
  double norm;
  Scalar cnorm;
  VecNorm(y,NORM_2, &norm );
  if (norm > *delta) {
     norm = *delta/norm;
     *gpnorm = (1.0 - norm)*(*fnorm);
     cnorm = norm;
     VecScale( &cnorm, y );
     *ynorm = *delta;
  } else {
     *gpnorm = 0.0;
     *ynorm = norm;
  }
  return 0;
}

/*@
   SNESSolve - Solves a nonlinear system.  Call SNESSolve after calling 
   SNESCreate(), optional routines of the form SNESSetXXX(), and SNESSetUp().

   Input Parameter:
.  snes - the SNES context

   Output Parameter:
   its - number of iterations until termination

.keywords: SNES, nonlinear, solve

.seealso: SNESCreate(), SNESSetUp(), SNESDestroy()
@*/
int SNESSolve(SNES snes,int *its)
{
  int ierr;
  PLogEventBegin(SNES_Solve,snes,0,0,0);
  ierr = (*(snes)->solve)(snes,its); CHKERRQ(ierr);
  PLogEventEnd(SNES_Solve,snes,0,0,0);
  if (OptionsHasName(0,"-snes_view")) {
    SNESView(snes,STDOUT_VIEWER_WORLD); CHKERRQ(ierr);
  }
  return 0;
}

/* --------- Internal routines for SNES Package --------- */

/*
   SNESComputeInitialGuess - Manages computation of initial approximation.
 */
int SNESComputeInitialGuess( SNES snes,Vec  x )
{
  int    ierr;
  Scalar zero = 0.0;
  if (snes->computeinitialguess) {
    ierr = (*snes->computeinitialguess)(snes, x, snes->gusP); CHKERRQ(ierr);
  }
  else {
    ierr = VecSet(&zero,x); CHKERRQ(ierr);
  }
  return 0;
}

/* ------------------------------------------------------------------ */

NRList *__NLList;

/*@
   SNESSetMethod - Sets the method for the nonlinear solver.  

   Input Parameters:
.  snes - the SNES context
.  method - a known method

   Notes:
   See "petsc/include/snes.h" for available methods (for instance)
$  Systems of nonlinear equations:
$    SNES_EQ_NLS - Newton's method with line search
$    SNES_EQ_NTR - Newton's method with trust region
$  Unconstrained minimization:
$    SNES_UM_NTR - Newton's method with trust region 
$    SNES_UM_NLS - Newton's method with line search

  Options Database Command:
$ -snes_method  <method>
$    Use -help for a list of available methods
$    (for instance, ls or tr)

.keysords: SNES, set, method
@*/
int SNESSetMethod(SNES snes,SNESMethod method)
{
  int (*r)(SNES);
  PETSCVALIDHEADERSPECIFIC(snes,SNES_COOKIE);
  /* Get the function pointers for the iterative method requested */
  if (!__NLList) {SNESRegisterAll();}
  if (!__NLList) {SETERRQ(1,"SNESSetMethod:Could not get methods");}
  r =  (int (*)(SNES))NRFindRoutine( __NLList, (int)method, (char *)0 );
  if (!r) {SETERRQ(1,"SNESSetMethod:Unknown method");}
  if (snes->data) PETSCFREE(snes->data);
  snes->set_method_called = 1;
  return (*r)(snes);
}

/* --------------------------------------------------------------------- */
/*@C
   SNESRegister - Adds the method to the nonlinear solver package, given 
   a function pointer and a nonlinear solver name of the type SNESMethod.

   Input Parameters:
.  name - for instance SNES_EQ_NLS, SNES_EQ_NTR, ...
.  sname - corfunPonding string for name
.  create - routine to create method context

.keywords: SNES, nonlinear, register

.seealso: SNESRegisterAll(), SNESRegisterDestroy()
@*/
int SNESRegister(int name, char *sname, int (*create)(SNES))
{
  int ierr;
  if (!__NLList) {ierr = NRCreate(&__NLList); CHKERRQ(ierr);}
  NRRegister( __NLList, name, sname, (int (*)(void*))create );
  return 0;
}
/* --------------------------------------------------------------------- */
/*@C
   SNESRegisterDestroy - Frees the list of nonlinear solvers that were
   registered by SNESRegister().

.keywords: SNES, nonlinear, register, destroy

.seealso: SNESRegisterAll(), SNESRegisterAll()
@*/
int SNESRegisterDestroy()
{
  if (__NLList) {
    NRDestroy( __NLList );
    __NLList = 0;
  }
  return 0;
}

/*
   SNESGetMethodFromOptions_Private - Sets the selected method from the 
   options database.

   Input Parameter:
.  ctx - the SNES context

   Output Parameter:
.  method -  solver method

   Returns:
   Returns 1 if the method is found; 0 otherwise.

   Options Database Key:
$  -snes_method  method
*/
int SNESGetMethodFromOptions_Private(SNES ctx,SNESMethod *method)
{
  char sbuf[50];
  if (OptionsGetString(ctx->prefix,"-snes_method", sbuf, 50 )) {
    if (!__NLList) SNESRegisterAll();
    *method = (SNESMethod)NRFindID( __NLList, sbuf );
    return 1;
  }
  return 0;
}

/*@
   SNESGetMethodFromContext - Gets the nonlinear solver method from an 
   active SNES context.

   Input Parameter:
.  snes - the SNES context

   Output parameters:
.  method - the method ID

.keywords: SNES, nonlinear, get, method, context, type

.seealso: SNESGetMethodName()
@*/
int SNESGetMethodFromContext(SNES snes, SNESMethod *method)
{
  PETSCVALIDHEADERSPECIFIC(snes,SNES_COOKIE);
  *method = (SNESMethod) snes->type;
  return 0;
}

/*@C
   SNESGetMethodName - Gets the SNES method name (as a string) from
   the method type.

   Input Parameter:
.  method - SNES method

   Output Parameter:
.  name - name of SNES method

.keywords: SNES, nonlinear, get, method, name
@*/
int SNESGetMethodName(SNESMethod method,char **name)
{
  int ierr;
  if (!__NLList) {ierr = SNESRegisterAll(); CHKERRQ(ierr);}
  *name = NRFindName( __NLList, (int) method );
  return 0;
}

#include <stdio.h>
/*
   SNESPrintMethods_Private - Prints the SNES methods available from the 
   options database.

   Input Parameters:
.  prefix - prefix (usually "-")
.  name - the options database name (by default "snes_method") 
*/
int SNESPrintMethods_Private(char* prefix,char *name)
{
  FuncList *entry;
  if (!__NLList) {SNESRegisterAll();}
  entry = __NLList->head;
  fprintf(stderr," %s%s (one of)",prefix,name);
  while (entry) {
    fprintf(stderr," %s",entry->name);
    entry = entry->next;
  }
  fprintf(stderr,"\n");
  return 0;
}

/*@C
   SNESGetSolution - Returns the vector where the approximate solution is
   stored.

   Input Parameter:
.  snes - the SNES context

   Output Parameter:
.  x - the solution

.keywords: SNES, nonlinear, get, solution

.seealso: SNESSetSolution(), SNESGetFunction(), SNESGetSolutionUpdate()
@*/
int SNESGetSolution(SNES snes,Vec *x)
{
  PETSCVALIDHEADERSPECIFIC(snes,SNES_COOKIE);
  *x = snes->vec_sol_always;
  return 0;
}  

/*@C
   SNESGetSolutionUpdate - Returns the vector where the solution update is
   stored. 

   Input Parameter:
.  snes - the SNES context

   Output Parameter:
.  x - the solution update

   Notes:
   This vector is implementation dependent.

.keywords: SNES, nonlinear, get, solution, update

.seealso: SNESGetSolution(), SNESGetFunction
@*/
int SNESGetSolutionUpdate(SNES snes,Vec *x)
{
  PETSCVALIDHEADERSPECIFIC(snes,SNES_COOKIE);
  *x = snes->vec_sol_update_always;
  return 0;
}

/*@C
   SNESGetFunction - Returns the vector where the function is
   stored.  Actually usually returns the vector where the negative of 
   the function is stored.

   Input Parameter:
.  snes - the SNES context

   Output Parameter:
.  r - the function (or its negative)

   Notes:
   SNESGetFunction() is valid for SNES_NONLINEAR_EQUATIONS methods only
   Analogous routines for SNES_UNCONSTRAINED_MINIMIZATION methods are
   SNESGetMinimizationFunction() and SNESGetGradient();

.keywords: SNES, nonlinear, get function

.seealso: SNESSetFunction(), SNESGetSolution()
@*/
int SNESGetFunction(SNES snes,Vec *r)
{
  PETSCVALIDHEADERSPECIFIC(snes,SNES_COOKIE);
  if (snes->method_class != SNES_NONLINEAR_EQUATIONS) SETERRQ(1,
    "SNESGetFunction:For SNES_NONLINEAR_EQUATIONS only");
  *r = snes->vec_func_always;
  return 0;
}  

/*@C
   SNESGetGradient - Returns the vector where the gradient is
   stored.  Actually usually returns the vector where the negative of 
   the function is stored.

   Input Parameter:
.  snes - the SNES context

   Output Parameter:
.  r - the gradient

   Notes:
   SNESGetGradient() is valid for SNES_UNCONSTRAINED_MINIMIZATION methods 
   only.  An analogous routine for SNES_NONLINEAR_EQUATIONS methods is
   SNESGetFunction().

.keywords: SNES, nonlinear, get, gradient

.seealso: SNESGetMinimizationFunction(), SNESGetSolution()
@*/
int SNESGetGradient(SNES snes,Vec *r)
{
  PETSCVALIDHEADERSPECIFIC(snes,SNES_COOKIE);
  if (snes->method_class != SNES_UNCONSTRAINED_MINIMIZATION) SETERRQ(1,
    "SNESGetGradient:For SNES_UNCONSTRAINED_MINIMIZATION only");
  *r = snes->vec_func_always;
  return 0;
}  

/*@
   SNESGetMinimizationFunction - Returns the scalar function value for 
   unconstrained minimization problems.

   Input Parameter:
.  snes - the SNES context

   Output Parameter:
.  r - the function

   Notes:
   SNESGetMinimizationFunction() is valid for SNES_UNCONSTRAINED_MINIMIZATION
   methods only.  An analogous routine for SNES_NONLINEAR_EQUATIONS methods is
   SNESGetFunction().

.keywords: SNES, nonlinear, get, function

.seealso: SNESGetGradient(), SNESGetSolution()
@*/
int SNESGetMinimizationFunction(SNES snes,double *r)
{
  PETSCVALIDHEADERSPECIFIC(snes,SNES_COOKIE);
  if (snes->method_class != SNES_UNCONSTRAINED_MINIMIZATION) SETERRQ(1,
    "SNESGetMinimizationFunction:For SNES_UNCONSTRAINED_MINIMIZATION only");
  *r = snes->fc;
  return 0;
}  





