#ifndef lint
static char vcid[] = "$Id: snes.c,v 1.75 1996/07/10 00:33:04 curfman Exp curfman $";
#endif

#include "draw.h"          /*I "draw.h"  I*/
#include "snesimpl.h"      /*I "snes.h"  I*/
#include "src/sys/nreg.h"      
#include "pinclude/pviewer.h"
#include <math.h>

extern int SNESGetTypeFromOptions_Private(SNES,SNESType*,int*);
extern int SNESPrintTypes_Private(MPI_Comm,char*,char*);

/*@ 
   SNESView - Prints the SNES data structure.

   Input Parameters:
.  SNES - the SNES context
.  viewer - visualization context

   Options Database Key:
$  -snes_view : calls SNESView() at end of SNESSolve()

   Notes:
   The available visualization contexts include
$     VIEWER_STDOUT_SELF - standard output (default)
$     VIEWER_STDOUT_WORLD - synchronized standard
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
  SNES_KSP_EW_ConvCtx *kctx;
  FILE                *fd;
  int                 ierr;
  SLES                sles;
  char                *method;
  ViewerType          vtype;

  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  if (vtype == ASCII_FILE_VIEWER || vtype == ASCII_FILES_VIEWER) {
    ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
    PetscFPrintf(snes->comm,fd,"SNES Object:\n");
    SNESGetType(snes,PETSC_NULL,&method);
    PetscFPrintf(snes->comm,fd,"  method: %s\n",method);
    if (snes->view) (*snes->view)((PetscObject)snes,viewer);
    PetscFPrintf(snes->comm,fd,
      "  maximum iterations=%d, maximum function evaluations=%d\n",
      snes->max_its,snes->max_funcs);
    PetscFPrintf(snes->comm,fd,
    "  tolerances: relative=%g, absolute=%g, truncation=%g, solution=%g\n",
      snes->rtol, snes->atol, snes->trunctol, snes->xtol);
    if (snes->method_class == SNES_UNCONSTRAINED_MINIMIZATION)
      PetscFPrintf(snes->comm,fd,"  min function tolerance=%g\n",snes->fmin);
    if (snes->ksp_ewconv) {
      kctx = (SNES_KSP_EW_ConvCtx *)snes->kspconvctx;
      if (kctx) {
        PetscFPrintf(snes->comm,fd,
     "  Eisenstat-Walker computation of KSP relative tolerance (version %d)\n",
        kctx->version);
        PetscFPrintf(snes->comm,fd,
          "    rtol_0=%g, rtol_max=%g, threshold=%g\n",kctx->rtol_0,
          kctx->rtol_max,kctx->threshold);
        PetscFPrintf(snes->comm,fd,"    gamma=%g, alpha=%g, alpha2=%g\n",
          kctx->gamma,kctx->alpha,kctx->alpha2);
      }
    }
  } else if (vtype == STRING_VIEWER) {
    SNESGetType(snes,PETSC_NULL,&method);
    ViewerStringSPrintf(viewer," %-3.3s",method);
  }
  SNESGetSLES(snes,&sles);
  ierr = SLESView(sles,viewer); CHKERRQ(ierr);
  return 0;
}

/*@
   SNESSetFromOptions - Sets various SNES and SLES parameters from user options.

   Input Parameter:
.  snes - the SNES context

.keywords: SNES, nonlinear, set, options, database

.seealso: SNESPrintHelp(), SNESSetOptionsPrefix()
@*/
int SNESSetFromOptions(SNES snes)
{
  SNESType method;
  double   tmp;
  SLES     sles;
  int      ierr, flg;
  int      version   = PETSC_DEFAULT;
  double   rtol_0    = PETSC_DEFAULT;
  double   rtol_max  = PETSC_DEFAULT;
  double   gamma2    = PETSC_DEFAULT;
  double   alpha     = PETSC_DEFAULT;
  double   alpha2    = PETSC_DEFAULT;
  double   threshold = PETSC_DEFAULT;

  PetscValidHeaderSpecific(snes,SNES_COOKIE);
  if (snes->setup_called)SETERRQ(1,"SNESSetFromOptions:Must call prior to SNESSetUp!");
  ierr = SNESGetTypeFromOptions_Private(snes,&method,&flg); CHKERRQ(ierr);
  if (flg) {
    ierr = SNESSetType(snes,method); CHKERRQ(ierr);
  }
  else if (!snes->set_method_called) {
    if (snes->method_class == SNES_NONLINEAR_EQUATIONS) {
      ierr = SNESSetType(snes,SNES_EQ_LS); CHKERRQ(ierr);
    }
    else {
      ierr = SNESSetType(snes,SNES_UM_TR); CHKERRQ(ierr);
    }
  }

  ierr = OptionsHasName(PETSC_NULL,"-help", &flg); CHKERRQ(ierr);
  if (flg) { SNESPrintHelp(snes); }
  ierr = OptionsGetDouble(snes->prefix,"-snes_stol",&tmp, &flg); CHKERRQ(ierr);
  if (flg) { SNESSetTolerances(snes,PETSC_DEFAULT,PETSC_DEFAULT,tmp,PETSC_DEFAULT,PETSC_DEFAULT); }
  ierr = OptionsGetDouble(snes->prefix,"-snes_atol",&tmp, &flg); CHKERRQ(ierr);
  if (flg) { SNESSetTolerances(snes,tmp,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT); }
  ierr = OptionsGetDouble(snes->prefix,"-snes_rtol",&tmp, &flg);  CHKERRQ(ierr);
  if (flg) { SNESSetTolerances(snes,PETSC_DEFAULT,tmp,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT); }
  ierr = OptionsGetInt(snes->prefix,"-snes_max_it",&snes->max_its, &flg); CHKERRQ(ierr);
  ierr = OptionsGetInt(snes->prefix,"-snes_max_funcs",&snes->max_funcs, &flg);CHKERRQ(ierr);
  ierr = OptionsGetDouble(snes->prefix,"-snes_trtol",&tmp, &flg); CHKERRQ(ierr);
  if (flg) { SNESSetTrustRegionTolerance(snes,tmp); }
  ierr = OptionsGetDouble(snes->prefix,"-snes_fmin",&tmp, &flg); CHKERRQ(ierr);
  if (flg) { SNESSetMinimizationFunctionTolerance(snes,tmp); }
  ierr = OptionsHasName(snes->prefix,"-snes_ksp_ew_conv", &flg); CHKERRQ(ierr);
  if (flg) { snes->ksp_ewconv = 1; }
  ierr = OptionsGetInt(snes->prefix,"-snes_ksp_ew_version",&version,&flg); CHKERRQ(ierr);
  ierr = OptionsGetDouble(snes->prefix,"-snes_ksp_ew_rtol0",&rtol_0,&flg); CHKERRQ(ierr);
  ierr = OptionsGetDouble(snes->prefix,"-snes_ksp_ew_rtolmax",&rtol_max,&flg); CHKERRQ(ierr);
  ierr = OptionsGetDouble(snes->prefix,"-snes_ksp_ew_gamma",&gamma2,&flg); CHKERRQ(ierr);
  ierr = OptionsGetDouble(snes->prefix,"-snes_ksp_ew_alpha",&alpha,&flg); CHKERRQ(ierr);
  ierr = OptionsGetDouble(snes->prefix,"-snes_ksp_ew_alpha2",&alpha2,&flg); CHKERRQ(ierr);
  ierr = OptionsGetDouble(snes->prefix,"-snes_ksp_ew_threshold",&threshold,&flg);  CHKERRQ(ierr);
  ierr = SNES_KSP_SetParametersEW(snes,version,rtol_0,rtol_max,gamma2,alpha,
                                  alpha2,threshold); CHKERRQ(ierr);
  ierr = OptionsHasName(snes->prefix,"-snes_monitor", &flg);  CHKERRQ(ierr);
  if (flg) { SNESSetMonitor(snes,SNESDefaultMonitor,0); }
  ierr = OptionsHasName(snes->prefix,"-snes_smonitor", &flg);  CHKERRQ(ierr);
   if (flg) { SNESSetMonitor(snes,SNESDefaultSMonitor,0); }
  ierr = OptionsHasName(snes->prefix,"-snes_xmonitor", &flg);  CHKERRQ(ierr);
  if (flg) {
    int       rank = 0;
    DrawLG lg;
    MPI_Initialized(&rank);
    if (rank) MPI_Comm_rank(snes->comm,&rank);
    if (!rank) {
      ierr = SNESLGMonitorCreate(0,0,0,0,300,300,&lg); CHKERRQ(ierr);
      ierr = SNESSetMonitor(snes,SNESLGMonitor,(void *)lg); CHKERRQ(ierr);
      snes->xmonitor = lg;
      PLogObjectParent(snes,lg);
    }
  }
  ierr = OptionsHasName(snes->prefix,"-snes_fd", &flg);  CHKERRQ(ierr);
  if (flg && snes->method_class == SNES_NONLINEAR_EQUATIONS) {
    ierr = SNESSetJacobian(snes,snes->jacobian,snes->jacobian_pre,
                 SNESDefaultComputeJacobian,snes->funP); CHKERRQ(ierr);
    PLogInfo(snes,
      "SNESSetFromOptions: Setting default finite difference Jacobian matrix\n");
  }
  else if (flg && snes->method_class == SNES_UNCONSTRAINED_MINIMIZATION) {
    ierr = SNESSetHessian(snes,snes->jacobian,snes->jacobian_pre,
                 SNESDefaultComputeHessian,snes->funP); CHKERRQ(ierr);
    PLogInfo(snes,
      "SNESSetFromOptions: Setting default finite difference Hessian matrix\n");
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

.seealso: SNESSetFromOptions()
@*/
int SNESPrintHelp(SNES snes)
{
  char                p[64];
  SNES_KSP_EW_ConvCtx *kctx;

  PetscValidHeaderSpecific(snes,SNES_COOKIE);

  PetscStrcpy(p,"-");
  if (snes->prefix) PetscStrcat(p, snes->prefix);

  kctx = (SNES_KSP_EW_ConvCtx *)snes->kspconvctx;

  PetscPrintf(snes->comm,"SNES options ----------------------------\n");
  SNESPrintTypes_Private(snes->comm,p,"snes_type");
  PetscPrintf(snes->comm," %ssnes_monitor: use default SNES monitor\n",p);
  PetscPrintf(snes->comm," %ssnes_view: view SNES info after each nonlinear solve\n",p);
  PetscPrintf(snes->comm," %ssnes_max_it <its>: max iterations (default %d)\n",p,snes->max_its);
  PetscPrintf(snes->comm," %ssnes_max_funcs <maxf>: max function evals (default %d)\n",p,snes->max_funcs);
  PetscPrintf(snes->comm," %ssnes_stol <stol>: successive step tolerance (default %g)\n",p,snes->xtol);
  PetscPrintf(snes->comm," %ssnes_atol <atol>: absolute tolerance (default %g)\n",p,snes->atol);
  PetscPrintf(snes->comm," %ssnes_rtol <rtol>: relative tolerance (default %g)\n",p,snes->rtol);
  PetscPrintf(snes->comm," %ssnes_trtol <trtol>: trust region parameter tolerance (default %g)\n",p,snes->deltatol);
  if (snes->type == SNES_NONLINEAR_EQUATIONS) {
    PetscPrintf(snes->comm,
     " options for solving systems of nonlinear equations only:\n");
    PetscPrintf(snes->comm,"   %ssnes_fd: use finite differences for Jacobian\n",p);
    PetscPrintf(snes->comm,"   %ssnes_mf: use matrix-free Jacobian\n",p);
    PetscPrintf(snes->comm,"   %ssnes_ksp_ew_conv: use Eisenstat-Walker computation of KSP rtol. Params are:\n",p);
    PetscPrintf(snes->comm,
     "     %ssnes_ksp_ew_version <version> (1 or 2, default is %d)\n",p,kctx->version);
    PetscPrintf(snes->comm,
     "     %ssnes_ksp_ew_rtol0 <rtol0> (0 <= rtol0 < 1, default %g)\n",p,kctx->rtol_0);
    PetscPrintf(snes->comm,
     "     %ssnes_ksp_ew_rtolmax <rtolmax> (0 <= rtolmax < 1, default %g)\n",p,kctx->rtol_max);
    PetscPrintf(snes->comm,
     "     %ssnes_ksp_ew_gamma <gamma> (0 <= gamma <= 1, default %g)\n",p,kctx->gamma);
    PetscPrintf(snes->comm,
     "     %ssnes_ksp_ew_alpha <alpha> (1 < alpha <= 2, default %g)\n",p,kctx->alpha);
    PetscPrintf(snes->comm,
     "     %ssnes_ksp_ew_alpha2 <alpha2> (default %g)\n",p,kctx->alpha2);
    PetscPrintf(snes->comm,
     "     %ssnes_ksp_ew_threshold <threshold> (0 < threshold < 1, default %g)\n",p,kctx->threshold);
  }
  else if (snes->type == SNES_UNCONSTRAINED_MINIMIZATION) {
    PetscPrintf(snes->comm,
     " options for solving unconstrained minimization problems only:\n");
    PetscPrintf(snes->comm,"   %ssnes_fmin <ftol>: minimum function tolerance (default %g)\n",p,snes->fmin);
    PetscPrintf(snes->comm,"   %ssnes_fd: use finite differences for Hessian\n",p);
    PetscPrintf(snes->comm,"   %ssnes_mf: use matrix-free Hessian\n",p);
  }
  PetscPrintf(snes->comm," Run program with -help %ssnes_type <method> for help on ",p);
  PetscPrintf(snes->comm,"a particular method\n");
  if (snes->printhelp) (*snes->printhelp)(snes,p);
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
  PetscValidHeaderSpecific(snes,SNES_COOKIE);
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
  PetscValidHeaderSpecific(snes,SNES_COOKIE);
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
  PetscValidHeaderSpecific(snes,SNES_COOKIE);
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
   A related routine for SNES_UNCONSTRAINED_MINIMIZATION methods is
   SNESGetGradientNorm().

.keywords: SNES, nonlinear, get, function, norm

.seealso: SNESSetFunction()
@*/
int SNESGetFunctionNorm(SNES snes,Scalar *fnorm)
{
  PetscValidHeaderSpecific(snes,SNES_COOKIE);
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
   methods only.  A related routine for SNES_NONLINEAR_EQUATIONS methods
   is SNESGetFunctionNorm().

.keywords: SNES, nonlinear, get, gradient, norm

.seelso: SNESSetGradient()
@*/
int SNESGetGradientNorm(SNES snes,Scalar *gnorm)
{
  PetscValidHeaderSpecific(snes,SNES_COOKIE);
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
  PetscValidHeaderSpecific(snes,SNES_COOKIE);
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
  PetscValidHeaderSpecific(snes,SNES_COOKIE);
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

.seealso: SNESSolve(), SNESDestroy()
@*/
int SNESCreate(MPI_Comm comm,SNESProblemType type,SNES *outsnes)
{
  int                 ierr;
  SNES                snes;
  SNES_KSP_EW_ConvCtx *kctx;

  *outsnes = 0;
  if (type != SNES_UNCONSTRAINED_MINIMIZATION && type != SNES_NONLINEAR_EQUATIONS)
    SETERRQ(1,"SNESCreate:incorrect method type"); 
  PetscHeaderCreate(snes,_SNES,SNES_COOKIE,SNES_UNKNOWN_METHOD,comm);
  PLogObjectCreate(snes);
  snes->max_its           = 50;
  snes->max_funcs	  = 1000;
  snes->norm		  = 0.0;
  if (type == SNES_UNCONSTRAINED_MINIMIZATION) {
    snes->rtol		  = 1.e-8;
    snes->ttol            = 0.0;
    snes->atol		  = 1.e-10;
  }
  else {
    snes->rtol		  = 1.e-8;
    snes->ttol            = 0.0;
    snes->atol		  = 1.e-50;
  }
  snes->xtol		  = 1.e-8;
  snes->trunctol	  = 1.e-12; /* no longer used */
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
  snes->type              = -1;
  snes->xmonitor          = 0;
  snes->vwork             = 0;
  snes->nwork             = 0;

  /* Create context to compute Eisenstat-Walker relative tolerance for KSP */
  kctx = PetscNew(SNES_KSP_EW_ConvCtx); CHKPTRQ(kctx);
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
.  ctx - optional user-defined function context 
.  r - vector to store function value

   Calling sequence of func:
.  func (SNES, Vec x, Vec f, void *ctx);

.  x - input vector
.  f - vector function
.  ctx - optional user-defined context for private data for the 
         function evaluation routine (may be null)

   Notes:
   The Newton-like methods typically solve linear systems of the form
$      f'(x) x = -f(x),
$  where f'(x) denotes the Jacobian matrix and f(x) is the function.

   SNESSetFunction() is valid for SNES_NONLINEAR_EQUATIONS methods only.
   Analogous routines for SNES_UNCONSTRAINED_MINIMIZATION methods are
   SNESSetMinimizationFunction() and SNESSetGradient();

.keywords: SNES, nonlinear, set, function

.seealso: SNESGetFunction(), SNESComputeFunction(), SNESSetJacobian()
@*/
int SNESSetFunction( SNES snes, Vec r, int (*func)(SNES,Vec,Vec,void*),void *ctx)
{
  PetscValidHeaderSpecific(snes,SNES_COOKIE);
  if (snes->method_class != SNES_NONLINEAR_EQUATIONS) SETERRQ(1,
    "SNESSetFunction:For SNES_NONLINEAR_EQUATIONS only");
  snes->computefunction     = func; 
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
.  y - function vector, as set by SNESSetFunction()

n   Notes:
   SNESComputeFunction() is valid for SNES_NONLINEAR_EQUATIONS methods only.
   Analogous routines for SNES_UNCONSTRAINED_MINIMIZATION methods are
   SNESComputeMinimizationFunction() and SNESComputeGradient();

.keywords: SNES, nonlinear, compute, function

.seealso: SNESSetFunction(), SNESGetFunction()
@*/
int SNESComputeFunction(SNES snes,Vec x, Vec y)
{
  int    ierr;

  if (snes->method_class != SNES_NONLINEAR_EQUATIONS) SETERRQ(1,
    "SNESComputeFunction: For SNES_NONLINEAR_EQUATIONS only");
  PLogEventBegin(SNES_FunctionEval,snes,x,y,0);
  ierr = (*snes->computefunction)(snes,x,y,snes->funP); CHKERRQ(ierr);
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

.seealso:  SNESGetMinimizationFunction(), SNESComputeMinimizationFunction(),
           SNESSetHessian(), SNESSetGradient()
@*/
int SNESSetMinimizationFunction(SNES snes,int (*func)(SNES,Vec,double*,void*),
                      void *ctx)
{
  PetscValidHeaderSpecific(snes,SNES_COOKIE);
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

.keywords: SNES, nonlinear, compute, minimization, function

.seealso: SNESSetMinimizationFunction(), SNESGetMinimizationFunction(),
          SNESComputeGradient(), SNESComputeHessian()
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

.seealso: SNESGetGradient(), SNESComputeGradient(), SNESSetHessian(),
          SNESSetMinimizationFunction(),
@*/
int SNESSetGradient(SNES snes,Vec r,int (*func)(SNES,Vec,Vec,void*),
                     void *ctx)
{
  PetscValidHeaderSpecific(snes,SNES_COOKIE);
  if (snes->method_class != SNES_UNCONSTRAINED_MINIMIZATION) SETERRQ(1,
    "SNESSetGradient:For SNES_UNCONSTRAINED_MINIMIZATION only");
  snes->computefunction     = func;
  snes->vec_func            = snes->vec_func_always = r;
  snes->funP                = ctx;
  return 0;
}

/*@
   SNESComputeGradient - Computes the gradient that has been set with
   SNESSetGradient().

   Input Parameters:
.  snes - the SNES context
.  x - input vector

   Output Parameter:
.  y - gradient vector

   Notes:
   SNESComputeGradient() is valid only for 
   SNES_UNCONSTRAINED_MINIMIZATION methods. An analogous routine for 
   SNES_NONLINEAR_EQUATIONS methods is SNESComputeFunction().

.keywords: SNES, nonlinear, compute, gradient

.seealso:  SNESSetGradient(), SNESGetGradient(), 
           SNESComputeMinimizationFunction(), SNESComputeHessian()
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

/*@
   SNESComputeJacobian - Computes the Jacobian matrix that has been
   set with SNESSetJacobian().

   Input Parameters:
.  snes - the SNES context
.  x - input vector

   Output Parameters:
.  A - Jacobian matrix
.  B - optional preconditioning matrix
.  flag - flag indicating matrix structure

   Notes: 
   Most users should not need to explicitly call this routine, as it
   is used internally within the nonlinear solvers. 

   See SLESSetOperators() for information about setting the flag parameter.

   SNESComputeJacobian() is valid only for SNES_NONLINEAR_EQUATIONS
   methods. An analogous routine for SNES_UNCONSTRAINED_MINIMIZATION 
   methods is SNESComputeJacobian().

.keywords: SNES, compute, Jacobian, matrix

.seealso:  SNESSetJacobian(), SLESSetOperators()
@*/
int SNESComputeJacobian(SNES snes,Vec X,Mat *A,Mat *B,MatStructure *flg)
{
  int    ierr;
  if (snes->method_class != SNES_NONLINEAR_EQUATIONS) SETERRQ(1,
    "SNESComputeJacobian: For SNES_NONLINEAR_EQUATIONS only");
  if (!snes->computejacobian) return 0;
  PLogEventBegin(SNES_JacobianEval,snes,X,*A,*B);
  *flg = DIFFERENT_NONZERO_PATTERN;
  ierr = (*snes->computejacobian)(snes,X,A,B,flg,snes->jacP); CHKERRQ(ierr);
  PLogEventEnd(SNES_JacobianEval,snes,X,*A,*B);
  /* make sure user returned a correct Jacobian and preconditioner */
  PetscValidHeaderSpecific(*A,MAT_COOKIE);
  PetscValidHeaderSpecific(*B,MAT_COOKIE);  
  return 0;
}

/*@
   SNESComputeHessian - Computes the Hessian matrix that has been
   set with SNESSetHessian().

   Input Parameters:
.  snes - the SNES context
.  x - input vector

   Output Parameters:
.  A - Hessian matrix
.  B - optional preconditioning matrix
.  flag - flag indicating matrix structure

   Notes: 
   Most users should not need to explicitly call this routine, as it
   is used internally within the nonlinear solvers. 

   See SLESSetOperators() for information about setting the flag parameter.

   SNESComputeHessian() is valid only for 
   SNES_UNCONSTRAINED_MINIMIZATION methods. An analogous routine for 
   SNES_NONLINEAR_EQUATIONS methods is SNESComputeJacobian().

.keywords: SNES, compute, Hessian, matrix

.seealso:  SNESSetHessian(), SLESSetOperators(), SNESComputeGradient(),
           SNESComputeMinimizationFunction()
@*/
int SNESComputeHessian(SNES snes,Vec x,Mat *A,Mat *B,MatStructure *flag)
{
  int    ierr;
  if (snes->method_class != SNES_UNCONSTRAINED_MINIMIZATION) SETERRQ(1,
    "SNESComputeHessian:For SNES_UNCONSTRAINED_MINIMIZATION only");
  if (!snes->computejacobian) return 0;
  PLogEventBegin(SNES_HessianEval,snes,x,*A,*B);
  *flag = DIFFERENT_NONZERO_PATTERN;
  ierr = (*snes->computejacobian)(snes,x,A,B,flag,snes->jacP); CHKERRQ(ierr);
  PLogEventEnd(SNES_HessianEval,snes,x,*A,*B);
  /* make sure user returned a correct Jacobian and preconditioner */
  PetscValidHeaderSpecific(*A,MAT_COOKIE);
  PetscValidHeaderSpecific(*B,MAT_COOKIE);  
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
.  flag - flag indicating information about the preconditioner matrix
   structure (same as flag in SLESSetOperators())
.  ctx - optional user-defined Jacobian context

   Notes: 
   See SLESSetOperators() for information about setting the flag input
   parameter in the routine func().  Be sure to read this information!

   The routine func() takes Mat * as the matrix arguments rather than Mat.  
   This allows the Jacobian evaluation routine to replace A and/or B with a 
   completely new new matrix structure (not just different matrix elements)
   when appropriate, for instance, if the nonzero structure is changing
   throughout the global iterations.

.keywords: SNES, nonlinear, set, Jacobian, matrix

.seealso: SLESSetOperators(), SNESSetFunction()
@*/
int SNESSetJacobian(SNES snes,Mat A,Mat B,int (*func)(SNES,Vec,Mat*,Mat*,
                    MatStructure*,void*),void *ctx)
{
  PetscValidHeaderSpecific(snes,SNES_COOKIE);
  if (snes->method_class != SNES_NONLINEAR_EQUATIONS) SETERRQ(1,
    "SNESSetJacobian:For SNES_NONLINEAR_EQUATIONS only");
  snes->computejacobian = func;
  snes->jacP            = ctx;
  snes->jacobian        = A;
  snes->jacobian_pre    = B;
  return 0;
}

/*@
   SNESGetJacobian - Returns the Jacobian matrix and optionally the user 
   provided context for evaluating the Jacobian.

   Input Parameter:
.  snes - the nonlinear solver context

   Output Parameters:
.  A - location to stash Jacobian matrix (or PETSC_NULL)
.  B - location to stash preconditioner matrix (or PETSC_NULL)
.  ctx - location to stash Jacobian ctx (or PETSC_NULL)

.seealso: SNESSetJacobian(), SNESComputeJacobian()
@*/
int SNESGetJacobian(SNES snes,Mat *A,Mat *B, void **ctx)
{
  if (snes->method_class != SNES_NONLINEAR_EQUATIONS) SETERRQ(1,
    "SNESSetJacobian:For SNES_NONLINEAR_EQUATIONS only");
  if (A)   *A = snes->jacobian;
  if (B)   *B = snes->jacobian_pre;
  if (ctx) *ctx = snes->jacP;
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
.  flag - flag indicating information about the preconditioner matrix
   structure (same as flag in SLESSetOperators())
.  ctx - optional user-defined Hessian context

   Notes: 
   See SLESSetOperators() for information about setting the flag input
   parameter in the routine func().  Be sure to read this information!

   The function func() takes Mat * as the matrix arguments rather than Mat.  
   This allows the Hessian evaluation routine to replace A and/or B with a 
   completely new new matrix structure (not just different matrix elements)
   when appropriate, for instance, if the nonzero structure is changing
   throughout the global iterations.

.keywords: SNES, nonlinear, set, Hessian, matrix

.seealso: SNESSetMinimizationFunction(), SNESSetGradient(), SLESSetOperators()
@*/
int SNESSetHessian(SNES snes,Mat A,Mat B,int (*func)(SNES,Vec,Mat*,Mat*,
                    MatStructure*,void*),void *ctx)
{
  PetscValidHeaderSpecific(snes,SNES_COOKIE);
  if (snes->method_class != SNES_UNCONSTRAINED_MINIMIZATION) SETERRQ(1,
    "SNESSetHessian:For SNES_UNCONSTRAINED_MINIMIZATION only");
  snes->computejacobian = func;
  snes->jacP            = ctx;
  snes->jacobian        = A;
  snes->jacobian_pre    = B;
  return 0;
}

/*@
   SNESGetHessian - Returns the Hessian matrix and optionally the user 
   provided context for evaluating the Hessian.

   Input Parameter:
.  snes - the nonlinear solver context

   Output Parameters:
.  A - location to stash Hessian matrix (or PETSC_NULL)
.  B - location to stash preconditioner matrix (or PETSC_NULL)
.  ctx - location to stash Hessian ctx (or PETSC_NULL)

.seealso: SNESSetHessian(), SNESComputeHessian()
@*/
int SNESGetHessian(SNES snes,Mat *A,Mat *B, void **ctx)
{
  if (snes->method_class != SNES_UNCONSTRAINED_MINIMIZATION) SETERRQ(1,
    "SNESSetHessian:For SNES_UNCONSTRAINED_MINIMIZATION only");
  if (A)   *A = snes->jacobian;
  if (B)   *B = snes->jacobian_pre;
  if (ctx) *ctx = snes->jacP;
  return 0;
}

/* ----- Routines to initialize and destroy a nonlinear solver ---- */

/*@
   SNESSetUp - Sets up the internal data structures for the later use
   of a nonlinear solver.

   Input Parameter:
.  snes - the SNES context
.  x - the solution vector

   Notes:
   For basic use of the SNES solvers the user need not explicitly call
   SNESSetUp(), since these actions will automatically occur during
   the call to SNESSolve().  However, if one wishes to control this
   phase separately, SNESSetUp() should be called after SNESCreate()
   and optional routines of the form SNESSetXXX(), but before SNESSolve().  

.keywords: SNES, nonlinear, setup

.seealso: SNESCreate(), SNESSolve(), SNESDestroy()
@*/
int SNESSetUp(SNES snes,Vec x)
{
  int ierr, flg;
  PetscValidHeaderSpecific(snes,SNES_COOKIE);
  PetscValidHeaderSpecific(x,VEC_COOKIE);
  snes->vec_sol = snes->vec_sol_always = x;

  ierr = OptionsHasName(snes->prefix,"-snes_mf", &flg);  CHKERRQ(ierr); 
  if (flg) {
    Mat J;
    ierr = SNESDefaultMatrixFreeMatCreate(snes,snes->vec_sol,&J);CHKERRQ(ierr);
    PLogObjectParent(snes,J);
    snes->mfshell = J;
    if (snes->method_class == SNES_NONLINEAR_EQUATIONS) {
      ierr = SNESSetJacobian(snes,J,J,0,snes->funP); CHKERRQ(ierr);
      PLogInfo(snes,"SNESSetUp: Setting default matrix-free Jacobian routines\n");
    }
    else if (snes->method_class == SNES_UNCONSTRAINED_MINIMIZATION) {
      ierr = SNESSetHessian(snes,J,J,0,snes->funP); CHKERRQ(ierr);
      PLogInfo(snes,"SNESSetUp: Setting default matrix-free Hessian routines\n");
    } else SETERRQ(1,"SNESSetUp:Method class doesn't support matrix-free option");
  }
  if ((snes->method_class == SNES_NONLINEAR_EQUATIONS)) {
    if (!snes->vec_func) SETERRQ(1,"SNESSetUp:Must call SNESSetFunction() first");
    if (!snes->computefunction) SETERRQ(1,"SNESSetUp:Must call SNESSetFunction() first");
    if (!snes->jacobian) SETERRQ(1,"SNESSetUp:Must call SNESSetJacobian() first");
    if (snes->vec_func == snes->vec_sol) SETERRQ(1,"SNESSetUp:Solution vector cannot be function vector");

    /* Set the KSP stopping criterion to use the Eisenstat-Walker method */
    if (snes->ksp_ewconv && snes->type != SNES_EQ_TR) {
      SLES sles; KSP ksp;
      ierr = SNESGetSLES(snes,&sles); CHKERRQ(ierr);
      ierr = SLESGetKSP(sles,&ksp); CHKERRQ(ierr);
      ierr = KSPSetConvergenceTest(ksp,SNES_KSP_EW_Converged_Private,
             (void *)snes); CHKERRQ(ierr);
    }
  } else if ((snes->method_class == SNES_UNCONSTRAINED_MINIMIZATION)) {
    if (!snes->vec_func) SETERRQ(1,"SNESSetUp:Must call SNESSetGradient() first");
    if (!snes->computefunction) SETERRQ(1,"SNESSetUp:Must call SNESSetGradient() first");
    if (!snes->computeumfunction) 
      SETERRQ(1,"SNESSetUp:Must call SNESSetMinimizationFunction() first");
    if (!snes->jacobian) SETERRQ(1,"SNESSetUp:Must call SNESSetHessian() first");
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

.seealso: SNESCreate(), SNESSolve()
@*/
int SNESDestroy(SNES snes)
{
  int ierr;
  PetscValidHeaderSpecific(snes,SNES_COOKIE);
  ierr = (*(snes)->destroy)((PetscObject)snes); CHKERRQ(ierr);
  if (snes->kspconvctx) PetscFree(snes->kspconvctx);
  if (snes->mfshell) MatDestroy(snes->mfshell);
  ierr = SLESDestroy(snes->sles); CHKERRQ(ierr);
  if (snes->xmonitor) SNESLGMonitorDestroy(snes->xmonitor);
  if (snes->vwork) VecDestroyVecs(snes->vwork,snes->nvwork);
  PLogObjectDestroy((PetscObject)snes);
  PetscHeaderDestroy((PetscObject)snes);
  return 0;
}

/* ----------- Routines to set solver parameters ---------- */

/*@
   SNESSetTolerances - Sets various parameters used in convergence tests.

   Input Parameters:
.  snes - the SNES context
.  atol - tolerance

   Options Database Key: 
$    -snes_atol <atol> - absolute convergence tolerance
$    -snes_rtol <rtol> - relative convergence tolerance
$    -snes_stol <stol> - convergence tolerance in terms of 
$          the norm of the change in the solution between steps
$    -snes_max_it <maxit> - maximum number of iterations
$    -snes_max_funcs <maxf> - maximum number of function evaluations

   Notes:
   The default maximum number of iterations is 50.
   The default maximum number of function evaluations is 1000.

.keywords: SNES, nonlinear, set, absolute, convergence, tolerance

.seealso: SNESSetTrustRegionTolerance(), SNESSetMinimizationFunctionTolerance()
@*/
int SNESSetTolerances(SNES snes,double atol,double rtol,double stol,int maxit,int maxf)
{
  PetscValidHeaderSpecific(snes,SNES_COOKIE);
  if (atol != PETSC_DEFAULT)  snes->atol      = atol;
  if (rtol != PETSC_DEFAULT)  snes->rtol      = rtol;
  if (stol != PETSC_DEFAULT)  snes->xtol      = stol;
  if (maxit != PETSC_DEFAULT) snes->max_its   = maxit;
  if (maxf != PETSC_DEFAULT)  snes->max_funcs = maxf;
  return 0;
}

/*@
   SNESSetTrustRegionTolerance - Sets the trust region parameter tolerance.  

   Input Parameters:
.  snes - the SNES context
.  tol - tolerance
   
   Options Database Key: 
$    -snes_trtol <tol>

.keywords: SNES, nonlinear, set, trust region, tolerance
 
.seealso: SNESSetTolerances(), SNESSetMinimizationFunctionTolerance()
@*/
int SNESSetTrustRegionTolerance(SNES snes,double tol)
{
  PetscValidHeaderSpecific(snes,SNES_COOKIE);
  snes->deltatol = tol;
  return 0;
}

/*@
   SNESSetMinimizationFunctionTolerance - Sets the minimum allowable function tolerance
   for unconstrained minimization solvers.
   
   Input Parameters:
.  snes - the SNES context
.  ftol - minimum function tolerance

   Options Database Key: 
$    -snes_fmin <ftol>

   Note:
   SNESSetMinimizationFunctionTolerance() is valid for SNES_UNCONSTRAINED_MINIMIZATION
   methods only.

.keywords: SNES, nonlinear, set, minimum, convergence, function, tolerance

.seealso: SNESSetTolerances(), SNESSetTrustRegionTolerance()
@*/
int SNESSetMinimizationFunctionTolerance(SNES snes,double ftol)
{
  PetscValidHeaderSpecific(snes,SNES_COOKIE);
  snes->fmin = ftol;
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
   int func(SNES snes,int its, Vec x,Vec f,double norm,void *mctx)

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

.seealso: SNESConverged_EQ_LS(), SNESConverged_EQ_TR(), 
          SNESConverged_UM_LS(), SNESConverged_UM_TR()
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
    For non-trust region methods such as SNES_EQ_LS, the parameter delta 
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
   SNESCreate() and optional routines of the form SNESSetXXX().

   Input Parameter:
.  snes - the SNES context
.  x - the solution vector

   Output Parameter:
   its - number of iterations until termination

   Note:
   The user should initialize the vector, x, with the initial guess
   for the nonlinear solve prior to calling SNESSolve.  In particular,
   to employ an initial guess of zero, the user should explicitly set
   this vector to zero by calling VecSet().

.keywords: SNES, nonlinear, solve

.seealso: SNESCreate(), SNESDestroy()
@*/
int SNESSolve(SNES snes,Vec x,int *its)
{
  int ierr, flg;

  PetscValidHeaderSpecific(snes,SNES_COOKIE);
  if (!snes->setup_called) {ierr = SNESSetUp(snes,x); CHKERRQ(ierr);}
  else {snes->vec_sol = snes->vec_sol_always = x;}
  PLogEventBegin(SNES_Solve,snes,0,0,0);
  ierr = (*(snes)->solve)(snes,its); CHKERRQ(ierr);
  PLogEventEnd(SNES_Solve,snes,0,0,0);
  ierr = OptionsHasName(PETSC_NULL,"-snes_view", &flg); CHKERRQ(ierr);
  if (flg) { ierr = SNESView(snes,VIEWER_STDOUT_WORLD); CHKERRQ(ierr); }
  return 0;
}

/* --------- Internal routines for SNES Package --------- */
static NRList *__SNESList = 0;

/*@
   SNESSetType - Sets the method for the nonlinear solver.  

   Input Parameters:
.  snes - the SNES context
.  method - a known method

   Notes:
   See "petsc/include/snes.h" for available methods (for instance)
$  Systems of nonlinear equations:
$    SNES_EQ_LS - Newton's method with line search
$    SNES_EQ_TR - Newton's method with trust region
$  Unconstrained minimization:
$    SNES_UM_TR - Newton's method with trust region 
$    SNES_UM_LS - Newton's method with line search

  Options Database Command:
$ -snes_type  <method>
$    Use -help for a list of available methods
$    (for instance, ls or tr)

.keysords: SNES, set, method
@*/
int SNESSetType(SNES snes,SNESType method)
{
  int (*r)(SNES);
  PetscValidHeaderSpecific(snes,SNES_COOKIE);
  if (snes->type == (int) method) return 0;

  /* Get the function pointers for the iterative method requested */
  if (!__SNESList) {SNESRegisterAll();}
  if (!__SNESList) {SETERRQ(1,"SNESSetType:Could not get methods");}
  r =  (int (*)(SNES))NRFindRoutine( __SNESList, (int)method, (char *)0 );
  if (!r) {SETERRQ(1,"SNESSetType:Unknown method");}
  if (snes->data) PetscFree(snes->data);
  snes->set_method_called = 1;
  return (*r)(snes);
}

/* --------------------------------------------------------------------- */
/*@C
   SNESRegister - Adds the method to the nonlinear solver package, given 
   a function pointer and a nonlinear solver name of the type SNESType.

   Input Parameters:
.  name - for instance SNES_EQ_LS, SNES_EQ_TR, ...
.  sname - corresponding string for name
.  create - routine to create method context

.keywords: SNES, nonlinear, register

.seealso: SNESRegisterAll(), SNESRegisterDestroy()
@*/
int SNESRegister(int name, char *sname, int (*create)(SNES))
{
  int ierr;
  if (!__SNESList) {ierr = NRCreate(&__SNESList); CHKERRQ(ierr);}
  NRRegister( __SNESList, name, sname, (int (*)(void*))create );
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
  if (__SNESList) {
    NRDestroy( __SNESList );
    __SNESList = 0;
  }
  return 0;
}

/*
   SNESGetTypeFromOptions_Private - Sets the selected method from the 
   options database.

   Input Parameter:
.  ctx - the SNES context

   Output Parameter:
.  method -  solver method

   Returns:
   Returns 1 if the method is found; 0 otherwise.

   Options Database Key:
$  -snes_type  method
*/
int SNESGetTypeFromOptions_Private(SNES ctx,SNESType *method,int *flg)
{
  int ierr;
  char sbuf[50];
  ierr = OptionsGetString(ctx->prefix,"-snes_type", sbuf, 50, flg); CHKERRQ(ierr);
  if (*flg) {
    if (!__SNESList) {ierr = SNESRegisterAll(); CHKERRQ(ierr);}
    *method = (SNESType)NRFindID( __SNESList, sbuf );
  }
  return 0;
}

/*@C
   SNESGetType - Gets the SNES method type and name (as a string).

   Input Parameter:
.  snes - nonlinear solver context

   Output Parameter:
.  method - SNES method (or use PETSC_NULL)
.  name - name of SNES method (or use PETSC_NULL)

.keywords: SNES, nonlinear, get, method, name
@*/
int SNESGetType(SNES snes, SNESType *method,char **name)
{
  int ierr;
  if (!__SNESList) {ierr = SNESRegisterAll(); CHKERRQ(ierr);}
  if (method) *method = (SNESType) snes->type;
  if (name)  *name = NRFindName( __SNESList, (int) snes->type );
  return 0;
}

#include <stdio.h>
/*
   SNESPrintTypes_Private - Prints the SNES methods available from the 
   options database.

   Input Parameters:
.  comm   - communicator (usually MPI_COMM_WORLD)
.  prefix - prefix (usually "-")
.  name   - the options database name (by default "snes_type") 
*/
int SNESPrintTypes_Private(MPI_Comm comm,char* prefix,char *name)
{
  FuncList *entry;
  if (!__SNESList) {SNESRegisterAll();}
  entry = __SNESList->head;
  PetscPrintf(comm," %s%s (one of)",prefix,name);
  while (entry) {
    PetscPrintf(comm," %s",entry->name);
    entry = entry->next;
  }
  PetscPrintf(comm,"\n");
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

.seealso: SNESGetFunction(), SNESGetGradient(), SNESGetSolutionUpdate()
@*/
int SNESGetSolution(SNES snes,Vec *x)
{
  PetscValidHeaderSpecific(snes,SNES_COOKIE);
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
  PetscValidHeaderSpecific(snes,SNES_COOKIE);
  *x = snes->vec_sol_update_always;
  return 0;
}

/*@C
   SNESGetFunction - Returns the vector where the function is stored.

   Input Parameter:
.  snes - the SNES context

   Output Parameter:
.  r - the function

   Notes:
   SNESGetFunction() is valid for SNES_NONLINEAR_EQUATIONS methods only
   Analogous routines for SNES_UNCONSTRAINED_MINIMIZATION methods are
   SNESGetMinimizationFunction() and SNESGetGradient();

.keywords: SNES, nonlinear, get, function

.seealso: SNESSetFunction(), SNESGetSolution(), SNESGetMinimizationFunction(),
          SNESGetGradient()
@*/
int SNESGetFunction(SNES snes,Vec *r)
{
  PetscValidHeaderSpecific(snes,SNES_COOKIE);
  if (snes->method_class != SNES_NONLINEAR_EQUATIONS) SETERRQ(1,
    "SNESGetFunction:For SNES_NONLINEAR_EQUATIONS only");
  *r = snes->vec_func_always;
  return 0;
}  

/*@C
   SNESGetGradient - Returns the vector where the gradient is stored.

   Input Parameter:
.  snes - the SNES context

   Output Parameter:
.  r - the gradient

   Notes:
   SNESGetGradient() is valid for SNES_UNCONSTRAINED_MINIMIZATION methods 
   only.  An analogous routine for SNES_NONLINEAR_EQUATIONS methods is
   SNESGetFunction().

.keywords: SNES, nonlinear, get, gradient

.seealso: SNESGetMinimizationFunction(), SNESGetSolution(), SNESGetFunction()
@*/
int SNESGetGradient(SNES snes,Vec *r)
{
  PetscValidHeaderSpecific(snes,SNES_COOKIE);
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

.seealso: SNESGetGradient(), SNESGetSolution(), SNESGetFunction()
@*/
int SNESGetMinimizationFunction(SNES snes,double *r)
{
  PetscValidHeaderSpecific(snes,SNES_COOKIE);
  if (snes->method_class != SNES_UNCONSTRAINED_MINIMIZATION) SETERRQ(1,
    "SNESGetMinimizationFunction:For SNES_UNCONSTRAINED_MINIMIZATION only");
  *r = snes->fc;
  return 0;
}  

/*@C
   SNESSetOptionsPrefix - Sets the prefix used for searching for all 
   SNES options in the database.

   Input Parameter:
.  snes - the SNES context
.  prefix - the prefix to prepend to all option names

.keywords: SNES, set, options, prefix, database

.seealso: SNESSetFromOptions()
@*/
int SNESSetOptionsPrefix(SNES snes,char *prefix)
{
  int ierr;

  PetscValidHeaderSpecific(snes,SNES_COOKIE);
  ierr = PetscObjectSetPrefix((PetscObject)snes, prefix); CHKERRQ(ierr);
  ierr = SLESSetOptionsPrefix(snes->sles,prefix);CHKERRQ(ierr);
  return 0;
}

/*@C
   SNESAppendOptionsPrefix - Append to the prefix used for searching for all 
   SNES options in the database.

   Input Parameter:
.  snes - the SNES context
.  prefix - the prefix to prepend to all option names

.keywords: SNES, append, options, prefix, database

.seealso: SNESGetOptionsPrefix()
@*/
int SNESAppendOptionsPrefix(SNES snes,char *prefix)
{
  int ierr;
  
  PetscValidHeaderSpecific(snes,SNES_COOKIE);
  ierr = PetscObjectAppendPrefix((PetscObject)snes, prefix); CHKERRQ(ierr);
  ierr = SLESAppendOptionsPrefix(snes->sles,prefix);CHKERRQ(ierr);
  return 0;
}

/*@
   SNESGetOptionsPrefix - Sets the prefix used for searching for all 
   SNES options in the database.

   Input Parameter:
.  snes - the SNES context

   Output Parameter:
.  prefix - pointer to the prefix string used

.keywords: SNES, get, options, prefix, database

.seealso: SNESAppendOptionsPrefix()
@*/
int SNESGetOptionsPrefix(SNES snes,char **prefix)
{
  int ierr;

  PetscValidHeaderSpecific(snes,SNES_COOKIE);
  ierr = PetscObjectGetPrefix((PetscObject)snes, prefix); CHKERRQ(ierr);
  return 0;
}





