#define PETSCSNES_DLL

#include "src/snes/snesimpl.h"      /*I "petscsnes.h"  I*/

PetscTruth SNESRegisterAllCalled = PETSC_FALSE;
PetscFList SNESList              = PETSC_NULL;

/* Logging support */
PetscCookie PETSCSNES_DLLEXPORT SNES_COOKIE = 0;
PetscEvent  SNES_Solve = 0, SNES_LineSearch = 0, SNES_FunctionEval = 0, SNES_JacobianEval = 0;

#undef __FUNCT__  
#define __FUNCT__ "SNESView"
/*@C
   SNESView - Prints the SNES data structure.

   Collective on SNES

   Input Parameters:
+  SNES - the SNES context
-  viewer - visualization context

   Options Database Key:
.  -snes_view - Calls SNESView() at end of SNESSolve()

   Notes:
   The available visualization contexts include
+     PETSC_VIEWER_STDOUT_SELF - standard output (default)
-     PETSC_VIEWER_STDOUT_WORLD - synchronized standard
         output where only the first processor opens
         the file.  All other processors send their 
         data to the first processor to print. 

   The user can open an alternative visualization context with
   PetscViewerASCIIOpen() - output to a specified file.

   Level: beginner

.keywords: SNES, view

.seealso: PetscViewerASCIIOpen()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESView(SNES snes,PetscViewer viewer)
{
  SNES_KSP_EW_ConvCtx *kctx;
  PetscErrorCode      ierr;
  KSP                 ksp;
  SNESType            type;
  PetscTruth          iascii,isstring;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_(snes->comm); 
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_COOKIE,2);
  PetscCheckSameComm(snes,1,viewer,2);

  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_STRING,&isstring);CHKERRQ(ierr);
  if (iascii) {
    if (snes->prefix) {
      ierr = PetscViewerASCIIPrintf(viewer,"SNES Object:(%s)\n",snes->prefix);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"SNES Object:\n");CHKERRQ(ierr);
    }
    ierr = SNESGetType(snes,&type);CHKERRQ(ierr);
    if (type) {
      ierr = PetscViewerASCIIPrintf(viewer,"  type: %s\n",type);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  type: not set yet\n");CHKERRQ(ierr);
    }
    if (snes->view) {
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = (*snes->view)(snes,viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"  maximum iterations=%D, maximum function evaluations=%D\n",snes->max_its,snes->max_funcs);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  tolerances: relative=%g, absolute=%g, solution=%g\n",
                 snes->rtol,snes->abstol,snes->xtol);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  total number of linear solver iterations=%D\n",snes->linear_its);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  total number of function evaluations=%D\n",snes->nfuncs);CHKERRQ(ierr);
    if (snes->ksp_ewconv) {
      kctx = (SNES_KSP_EW_ConvCtx *)snes->kspconvctx;
      if (kctx) {
        ierr = PetscViewerASCIIPrintf(viewer,"  Eisenstat-Walker computation of KSP relative tolerance (version %D)\n",kctx->version);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"    rtol_0=%g, rtol_max=%g, threshold=%g\n",kctx->rtol_0,kctx->rtol_max,kctx->threshold);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"    gamma=%g, alpha=%g, alpha2=%g\n",kctx->gamma,kctx->alpha,kctx->alpha2);CHKERRQ(ierr);
      }
    }
  } else if (isstring) {
    ierr = SNESGetType(snes,&type);CHKERRQ(ierr);
    ierr = PetscViewerStringSPrintf(viewer," %-3.3s",type);CHKERRQ(ierr);
  }
  ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  ierr = KSPView(ksp,viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  We retain a list of functions that also take SNES command 
  line options. These are called at the end SNESSetFromOptions()
*/
#define MAXSETFROMOPTIONS 5
static PetscInt numberofsetfromoptions;
static PetscErrorCode (*othersetfromoptions[MAXSETFROMOPTIONS])(SNES);

#undef __FUNCT__  
#define __FUNCT__ "SNESAddOptionsChecker"
/*@C
  SNESAddOptionsChecker - Adds an additional function to check for SNES options.

  Not Collective

  Input Parameter:
. snescheck - function that checks for options

  Level: developer

.seealso: SNESSetFromOptions()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESAddOptionsChecker(PetscErrorCode (*snescheck)(SNES))
{
  PetscFunctionBegin;
  if (numberofsetfromoptions >= MAXSETFROMOPTIONS) {
    SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE, "Too many options checkers, only %D allowed", MAXSETFROMOPTIONS);
  }
  othersetfromoptions[numberofsetfromoptions++] = snescheck;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESSetFromOptions"
/*@
   SNESSetFromOptions - Sets various SNES and KSP parameters from user options.

   Collective on SNES

   Input Parameter:
.  snes - the SNES context

   Options Database Keys:
+  -snes_type <type> - ls, tr, umls, umtr, test
.  -snes_stol - convergence tolerance in terms of the norm
                of the change in the solution between steps
.  -snes_atol <abstol> - absolute tolerance of residual norm
.  -snes_rtol <rtol> - relative decrease in tolerance norm from initial
.  -snes_max_it <max_it> - maximum number of iterations
.  -snes_max_funcs <max_funcs> - maximum number of function evaluations
.  -snes_max_fail <max_fail> - maximum number of failures
.  -snes_trtol <trtol> - trust region tolerance
.  -snes_no_convergence_test - skip convergence test in nonlinear 
                               solver; hence iterations will continue until max_it
                               or some other criterion is reached. Saves expense
                               of convergence test
.  -snes_monitor - prints residual norm at each iteration 
.  -snes_vecmonitor - plots solution at each iteration
.  -snes_vecmonitor_update - plots update to solution at each iteration 
.  -snes_xmonitor - plots residual norm at each iteration 
.  -snes_fd - use finite differences to compute Jacobian; very slow, only for testing
.  -snes_mf_ksp_monitor - if using matrix-free multiply then print h at each KSP iteration
-  -snes_print_converged_reason - print the reason for convergence/divergence after each solve

    Options Database for Eisenstat-Walker method:
+  -snes_ksp_ew_conv - use Eisenstat-Walker method for determining linear system convergence
.  -snes_ksp_ew_version ver - version of  Eisenstat-Walker method
.  -snes_ksp_ew_rtol0 <rtol0> - Sets rtol0
.  -snes_ksp_ew_rtolmax <rtolmax> - Sets rtolmax
.  -snes_ksp_ew_gamma <gamma> - Sets gamma
.  -snes_ksp_ew_alpha <alpha> - Sets alpha
.  -snes_ksp_ew_alpha2 <alpha2> - Sets alpha2 
-  -snes_ksp_ew_threshold <threshold> - Sets threshold

   Notes:
   To see all options, run your program with the -help option or consult
   the users manual.

   Level: beginner

.keywords: SNES, nonlinear, set, options, database

.seealso: SNESSetOptionsPrefix()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESSetFromOptions(SNES snes)
{
  KSP                 ksp;
  SNES_KSP_EW_ConvCtx *kctx = (SNES_KSP_EW_ConvCtx *)snes->kspconvctx;
  PetscTruth          flg;
  PetscErrorCode      ierr;
  PetscInt            i;
  const char          *deft;
  char                type[256];

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);

  ierr = PetscOptionsBegin(snes->comm,snes->prefix,"Nonlinear solver (SNES) options","SNES");CHKERRQ(ierr); 
    if (snes->type_name) {
      deft = snes->type_name;
    } else {  
      deft = SNESLS;
    }

    if (!SNESRegisterAllCalled) {ierr = SNESRegisterAll(PETSC_NULL);CHKERRQ(ierr);}
    ierr = PetscOptionsList("-snes_type","Nonlinear solver method","SNESSetType",SNESList,deft,type,256,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = SNESSetType(snes,type);CHKERRQ(ierr);
    } else if (!snes->type_name) {
      ierr = SNESSetType(snes,deft);CHKERRQ(ierr);
    }
    ierr = PetscOptionsName("-snes_view","Print detailed information on solver used","SNESView",0);CHKERRQ(ierr);

    ierr = PetscOptionsReal("-snes_stol","Stop if step length less then","SNESSetTolerances",snes->xtol,&snes->xtol,0);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-snes_atol","Stop if function norm less then","SNESSetTolerances",snes->abstol,&snes->abstol,0);CHKERRQ(ierr);

    ierr = PetscOptionsReal("-snes_rtol","Stop if decrease in function norm less then","SNESSetTolerances",snes->rtol,&snes->rtol,0);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-snes_max_it","Maximum iterations","SNESSetTolerances",snes->max_its,&snes->max_its,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-snes_max_funcs","Maximum function evaluations","SNESSetTolerances",snes->max_funcs,&snes->max_funcs,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-snes_max_fail","Maximum failures","SNESSetTolerances",snes->maxFailures,&snes->maxFailures,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsName("-snes_converged_reason","Print reason for converged or diverged","SNESSolve",&flg);CHKERRQ(ierr);
    if (flg) {
      snes->printreason = PETSC_TRUE;
    }

    ierr = PetscOptionsName("-snes_ksp_ew_conv","Use Eisentat-Walker linear system convergence test","SNES_KSP_SetParametersEW",&snes->ksp_ewconv);CHKERRQ(ierr);

    ierr = PetscOptionsInt("-snes_ksp_ew_version","Version 1 or 2","SNES_KSP_SetParametersEW",kctx->version,&kctx->version,0);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-snes_ksp_ew_rtol0","0 <= rtol0 < 1","SNES_KSP_SetParametersEW",kctx->rtol_0,&kctx->rtol_0,0);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-snes_ksp_ew_rtolmax","0 <= rtolmax < 1","SNES_KSP_SetParametersEW",kctx->rtol_max,&kctx->rtol_max,0);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-snes_ksp_ew_gamma","0 <= gamma <= 1","SNES_KSP_SetParametersEW",kctx->gamma,&kctx->gamma,0);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-snes_ksp_ew_alpha","1 < alpha <= 2","SNES_KSP_SetParametersEW",kctx->alpha,&kctx->alpha,0);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-snes_ksp_ew_alpha2","alpha2","SNES_KSP_SetParametersEW",kctx->alpha2,&kctx->alpha2,0);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-snes_ksp_ew_threshold","0 < threshold < 1","SNES_KSP_SetParametersEW",kctx->threshold,&kctx->threshold,0);CHKERRQ(ierr);

    ierr = PetscOptionsName("-snes_no_convergence_test","Don't test for convergence","None",&flg);CHKERRQ(ierr);
    if (flg) {snes->converged = 0;}
    ierr = PetscOptionsName("-snes_cancelmonitors","Remove all monitors","SNESClearMonitor",&flg);CHKERRQ(ierr);
    if (flg) {ierr = SNESClearMonitor(snes);CHKERRQ(ierr);}
    ierr = PetscOptionsName("-snes_monitor","Monitor norm of function","SNESDefaultMonitor",&flg);CHKERRQ(ierr);
    if (flg) {ierr = SNESSetMonitor(snes,SNESDefaultMonitor,0,0);CHKERRQ(ierr);}
    ierr = PetscOptionsName("-snes_ratiomonitor","Monitor norm of function","SNESSetRatioMonitor",&flg);CHKERRQ(ierr);
    if (flg) {ierr = SNESSetRatioMonitor(snes);CHKERRQ(ierr);}
    ierr = PetscOptionsName("-snes_smonitor","Monitor norm of function (fewer digits)","SNESDefaultSMonitor",&flg);CHKERRQ(ierr);
    if (flg) {ierr = SNESSetMonitor(snes,SNESDefaultSMonitor,0,0);CHKERRQ(ierr);}
    ierr = PetscOptionsName("-snes_vecmonitor","Plot solution at each iteration","SNESVecViewMonitor",&flg);CHKERRQ(ierr);
    if (flg) {ierr = SNESSetMonitor(snes,SNESVecViewMonitor,0,0);CHKERRQ(ierr);}
    ierr = PetscOptionsName("-snes_vecmonitor_update","Plot correction at each iteration","SNESVecViewUpdateMonitor",&flg);CHKERRQ(ierr);
    if (flg) {ierr = SNESSetMonitor(snes,SNESVecViewUpdateMonitor,0,0);CHKERRQ(ierr);}
    ierr = PetscOptionsName("-snes_vecmonitor_residual","Plot residual at each iteration","SNESVecViewResidualMonitor",&flg);CHKERRQ(ierr);
    if (flg) {ierr = SNESSetMonitor(snes,SNESVecViewResidualMonitor,0,0);CHKERRQ(ierr);}
    ierr = PetscOptionsName("-snes_xmonitor","Plot function norm at each iteration","SNESLGMonitor",&flg);CHKERRQ(ierr);
    if (flg) {ierr = SNESSetMonitor(snes,SNESLGMonitor,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);}

    ierr = PetscOptionsName("-snes_fd","Use finite differences (slow) to compute Jacobian","SNESDefaultComputeJacobian",&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = SNESSetJacobian(snes,snes->jacobian,snes->jacobian_pre,SNESDefaultComputeJacobian,snes->funP);CHKERRQ(ierr);
      ierr = PetscLogInfo((snes,"SNESSetFromOptions: Setting default finite difference Jacobian matrix\n"));CHKERRQ(ierr);
    }

    for(i = 0; i < numberofsetfromoptions; i++) {
      ierr = (*othersetfromoptions[i])(snes);CHKERRQ(ierr);
    }

    if (snes->setfromoptions) {
      ierr = (*snes->setfromoptions)(snes);CHKERRQ(ierr);
    }

  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  PetscFunctionReturn(0); 
}


#undef __FUNCT__  
#define __FUNCT__ "SNESSetApplicationContext"
/*@
   SNESSetApplicationContext - Sets the optional user-defined context for 
   the nonlinear solvers.  

   Collective on SNES

   Input Parameters:
+  snes - the SNES context
-  usrP - optional user context

   Level: intermediate

.keywords: SNES, nonlinear, set, application, context

.seealso: SNESGetApplicationContext()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESSetApplicationContext(SNES snes,void *usrP)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  snes->user		= usrP;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESGetApplicationContext"
/*@C
   SNESGetApplicationContext - Gets the user-defined context for the 
   nonlinear solvers.  

   Not Collective

   Input Parameter:
.  snes - SNES context

   Output Parameter:
.  usrP - user context

   Level: intermediate

.keywords: SNES, nonlinear, get, application, context

.seealso: SNESSetApplicationContext()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESGetApplicationContext(SNES snes,void **usrP)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  *usrP = snes->user;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESGetIterationNumber"
/*@
   SNESGetIterationNumber - Gets the number of nonlinear iterations completed
   at this time.

   Not Collective

   Input Parameter:
.  snes - SNES context

   Output Parameter:
.  iter - iteration number

   Notes:
   For example, during the computation of iteration 2 this would return 1.

   This is useful for using lagged Jacobians (where one does not recompute the 
   Jacobian at each SNES iteration). For example, the code
.vb
      ierr = SNESGetIterationNumber(snes,&it);
      if (!(it % 2)) {
        [compute Jacobian here]
      }
.ve
   can be used in your ComputeJacobian() function to cause the Jacobian to be
   recomputed every second SNES iteration.

   Level: intermediate

.keywords: SNES, nonlinear, get, iteration, number
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESGetIterationNumber(SNES snes,PetscInt* iter)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  PetscValidIntPointer(iter,2);
  *iter = snes->iter;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESGetFunctionNorm"
/*@
   SNESGetFunctionNorm - Gets the norm of the current function that was set
   with SNESSSetFunction().

   Collective on SNES

   Input Parameter:
.  snes - SNES context

   Output Parameter:
.  fnorm - 2-norm of function

   Level: intermediate

.keywords: SNES, nonlinear, get, function, norm

.seealso: SNESGetFunction()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESGetFunctionNorm(SNES snes,PetscScalar *fnorm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  PetscValidScalarPointer(fnorm,2);
  *fnorm = snes->norm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESGetNumberUnsuccessfulSteps"
/*@
   SNESGetNumberUnsuccessfulSteps - Gets the number of unsuccessful steps
   attempted by the nonlinear solver.

   Not Collective

   Input Parameter:
.  snes - SNES context

   Output Parameter:
.  nfails - number of unsuccessful steps attempted

   Notes:
   This counter is reset to zero for each successive call to SNESSolve().

   Level: intermediate

.keywords: SNES, nonlinear, get, number, unsuccessful, steps
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESGetNumberUnsuccessfulSteps(SNES snes,PetscInt* nfails)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  PetscValidIntPointer(nfails,2);
  *nfails = snes->numFailures;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESSetMaximumUnsuccessfulSteps"
/*@
   SNESSetMaximumUnsuccessfulSteps - Sets the maximum number of unsuccessful steps
   attempted by the nonlinear solver before it gives up.

   Not Collective

   Input Parameters:
+  snes     - SNES context
-  maxFails - maximum of unsuccessful steps

   Level: intermediate

.keywords: SNES, nonlinear, set, maximum, unsuccessful, steps
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESSetMaximumUnsuccessfulSteps(SNES snes, PetscInt maxFails)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  snes->maxFailures = maxFails;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESGetMaximumUnsuccessfulSteps"
/*@
   SNESGetMaximumUnsuccessfulSteps - Gets the maximum number of unsuccessful steps
   attempted by the nonlinear solver before it gives up.

   Not Collective

   Input Parameter:
.  snes     - SNES context

   Output Parameter:
.  maxFails - maximum of unsuccessful steps

   Level: intermediate

.keywords: SNES, nonlinear, get, maximum, unsuccessful, steps
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESGetMaximumUnsuccessfulSteps(SNES snes, PetscInt *maxFails)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  PetscValidIntPointer(maxFails,2);
  *maxFails = snes->maxFailures;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESGetNumberLinearIterations"
/*@
   SNESGetNumberLinearIterations - Gets the total number of linear iterations
   used by the nonlinear solver.

   Not Collective

   Input Parameter:
.  snes - SNES context

   Output Parameter:
.  lits - number of linear iterations

   Notes:
   This counter is reset to zero for each successive call to SNESSolve().

   Level: intermediate

.keywords: SNES, nonlinear, get, number, linear, iterations
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESGetNumberLinearIterations(SNES snes,PetscInt* lits)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  PetscValidIntPointer(lits,2);
  *lits = snes->linear_its;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESGetKSP"
/*@C
   SNESGetKSP - Returns the KSP context for a SNES solver.

   Not Collective, but if SNES object is parallel, then KSP object is parallel

   Input Parameter:
.  snes - the SNES context

   Output Parameter:
.  ksp - the KSP context

   Notes:
   The user can then directly manipulate the KSP context to set various
   options, etc.  Likewise, the user can then extract and manipulate the 
   KSP and PC contexts as well.

   Level: beginner

.keywords: SNES, nonlinear, get, KSP, context

.seealso: KSPGetPC()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESGetKSP(SNES snes,KSP *ksp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  PetscValidPointer(ksp,2);
  *ksp = snes->ksp;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESPublish_Petsc"
static PetscErrorCode SNESPublish_Petsc(PetscObject obj)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

/* -----------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "SNESCreate"
/*@C
   SNESCreate - Creates a nonlinear solver context.

   Collective on MPI_Comm

   Input Parameters:
+  comm - MPI communicator

   Output Parameter:
.  outsnes - the new SNES context

   Options Database Keys:
+   -snes_mf - Activates default matrix-free Jacobian-vector products,
               and no preconditioning matrix
.   -snes_mf_operator - Activates default matrix-free Jacobian-vector
               products, and a user-provided preconditioning matrix
               as set by SNESSetJacobian()
-   -snes_fd - Uses (slow!) finite differences to compute Jacobian

   Level: beginner

.keywords: SNES, nonlinear, create, context

.seealso: SNESSolve(), SNESDestroy(), SNES
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESCreate(MPI_Comm comm,SNES *outsnes)
{
  PetscErrorCode      ierr;
  SNES                snes;
  SNES_KSP_EW_ConvCtx *kctx;

  PetscFunctionBegin;
  PetscValidPointer(outsnes,2);
  *outsnes = PETSC_NULL;
#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = SNESInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif

  ierr = PetscHeaderCreate(snes,_p_SNES,PetscInt,SNES_COOKIE,0,"SNES",comm,SNESDestroy,SNESView);CHKERRQ(ierr);
  snes->bops->publish     = SNESPublish_Petsc;
  snes->max_its           = 50;
  snes->max_funcs	  = 10000;
  snes->norm		  = 0.0;
  snes->rtol		  = 1.e-8;
  snes->ttol              = 0.0;
  snes->abstol		  = 1.e-50;
  snes->xtol		  = 1.e-8;
  snes->deltatol	  = 1.e-12;
  snes->nfuncs            = 0;
  snes->numFailures       = 0;
  snes->maxFailures       = 1;
  snes->linear_its        = 0;
  snes->numbermonitors    = 0;
  snes->data              = 0;
  snes->view              = 0;
  snes->setupcalled       = 0;
  snes->ksp_ewconv        = PETSC_FALSE;
  snes->vwork             = 0;
  snes->nwork             = 0;
  snes->conv_hist_len     = 0;
  snes->conv_hist_max     = 0;
  snes->conv_hist         = PETSC_NULL;
  snes->conv_hist_its     = PETSC_NULL;
  snes->conv_hist_reset   = PETSC_TRUE;
  snes->reason            = SNES_CONVERGED_ITERATING;

  /* Create context to compute Eisenstat-Walker relative tolerance for KSP */
  ierr = PetscNew(SNES_KSP_EW_ConvCtx,&kctx);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(snes,sizeof(SNES_KSP_EW_ConvCtx));CHKERRQ(ierr);
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

  ierr = KSPCreate(comm,&snes->ksp);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(snes,snes->ksp);CHKERRQ(ierr);

  *outsnes = snes;
  ierr = PetscPublishAll(snes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESSetFunction"
/*@C
   SNESSetFunction - Sets the function evaluation routine and function 
   vector for use by the SNES routines in solving systems of nonlinear
   equations.

   Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  func - function evaluation routine
.  r - vector to store function value
-  ctx - [optional] user-defined context for private data for the 
         function evaluation routine (may be PETSC_NULL)

   Calling sequence of func:
$    func (SNES snes,Vec x,Vec f,void *ctx);

.  f - function vector
-  ctx - optional user-defined function context 

   Notes:
   The Newton-like methods typically solve linear systems of the form
$      f'(x) x = -f(x),
   where f'(x) denotes the Jacobian matrix and f(x) is the function.

   Level: beginner

.keywords: SNES, nonlinear, set, function

.seealso: SNESGetFunction(), SNESComputeFunction(), SNESSetJacobian()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESSetFunction(SNES snes,Vec r,PetscErrorCode (*func)(SNES,Vec,Vec,void*),void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  PetscValidHeaderSpecific(r,VEC_COOKIE,2);
  PetscCheckSameComm(snes,1,r,2);

  snes->computefunction     = func; 
  snes->vec_func            = snes->vec_func_always = r;
  snes->funP                = ctx;
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "SNESSetRhs"
/*@C
   SNESSetRhs - Sets the vector for solving F(x) = rhs. If rhs is not set
   it assumes a zero right hand side.

   Collective on SNES

   Input Parameters:
+  snes - the SNES context
-  rhs - the right hand side vector or PETSC_NULL for a zero right hand side

   Level: intermediate

.keywords: SNES, nonlinear, set, function, right hand side

.seealso: SNESGetFunction(), SNESComputeFunction(), SNESSetJacobian(), SNESSetFunction()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESSetRhs(SNES snes,Vec rhs)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  if (rhs) {
    PetscValidHeaderSpecific(rhs,VEC_COOKIE,2);
    PetscCheckSameComm(snes,1,rhs,2);
    ierr = PetscObjectReference((PetscObject)rhs);CHKERRQ(ierr);
  }
  if (snes->afine) {
    ierr = VecDestroy(snes->afine);CHKERRQ(ierr);
  }
  snes->afine = rhs;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESComputeFunction"
/*@
   SNESComputeFunction - Calls the function that has been set with
                         SNESSetFunction().  

   Collective on SNES

   Input Parameters:
+  snes - the SNES context
-  x - input vector

   Output Parameter:
.  y - function vector, as set by SNESSetFunction()

   Notes:
   SNESComputeFunction() is typically used within nonlinear solvers
   implementations, so most users would not generally call this routine
   themselves.

   Level: developer

.keywords: SNES, nonlinear, compute, function

.seealso: SNESSetFunction(), SNESGetFunction()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESComputeFunction(SNES snes,Vec x,Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  PetscValidHeaderSpecific(x,VEC_COOKIE,2);
  PetscValidHeaderSpecific(y,VEC_COOKIE,3);
  PetscCheckSameComm(snes,1,x,2);
  PetscCheckSameComm(snes,1,y,3);

  ierr = PetscLogEventBegin(SNES_FunctionEval,snes,x,y,0);CHKERRQ(ierr);
  PetscStackPush("SNES user function");
  ierr = (*snes->computefunction)(snes,x,y,snes->funP);
  PetscStackPop;
  if (PetscExceptionValue(ierr)) {
    PetscErrorCode pierr = PetscLogEventEnd(SNES_FunctionEval,snes,x,y,0);CHKERRQ(pierr);
  }
  CHKERRQ(ierr);
  if (snes->afine) {
    PetscScalar mone = -1.0;
    ierr = VecAXPY(y,mone,snes->afine);CHKERRQ(ierr);
  }
  snes->nfuncs++;
  ierr = PetscLogEventEnd(SNES_FunctionEval,snes,x,y,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESComputeJacobian"
/*@
   SNESComputeJacobian - Computes the Jacobian matrix that has been
   set with SNESSetJacobian().

   Collective on SNES and Mat

   Input Parameters:
+  snes - the SNES context
-  x - input vector

   Output Parameters:
+  A - Jacobian matrix
.  B - optional preconditioning matrix
-  flag - flag indicating matrix structure

   Notes: 
   Most users should not need to explicitly call this routine, as it 
   is used internally within the nonlinear solvers. 

   See KSPSetOperators() for important information about setting the
   flag parameter.

   Level: developer

.keywords: SNES, compute, Jacobian, matrix

.seealso:  SNESSetJacobian(), KSPSetOperators()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESComputeJacobian(SNES snes,Vec X,Mat *A,Mat *B,MatStructure *flg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  PetscValidHeaderSpecific(X,VEC_COOKIE,2);
  PetscValidPointer(flg,5);
  PetscCheckSameComm(snes,1,X,2);
  if (!snes->computejacobian) PetscFunctionReturn(0);
  ierr = PetscLogEventBegin(SNES_JacobianEval,snes,X,*A,*B);CHKERRQ(ierr);
  *flg = DIFFERENT_NONZERO_PATTERN;
  PetscStackPush("SNES user Jacobian function");
  ierr = (*snes->computejacobian)(snes,X,A,B,flg,snes->jacP);CHKERRQ(ierr);
  PetscStackPop;
  ierr = PetscLogEventEnd(SNES_JacobianEval,snes,X,*A,*B);CHKERRQ(ierr);
  /* make sure user returned a correct Jacobian and preconditioner */
  PetscValidHeaderSpecific(*A,MAT_COOKIE,3);
  PetscValidHeaderSpecific(*B,MAT_COOKIE,4);  
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESSetJacobian"
/*@C
   SNESSetJacobian - Sets the function to compute Jacobian as well as the
   location to store the matrix.

   Collective on SNES and Mat

   Input Parameters:
+  snes - the SNES context
.  A - Jacobian matrix
.  B - preconditioner matrix (usually same as the Jacobian)
.  func - Jacobian evaluation routine
-  ctx - [optional] user-defined context for private data for the 
         Jacobian evaluation routine (may be PETSC_NULL)

   Calling sequence of func:
$     func (SNES snes,Vec x,Mat *A,Mat *B,int *flag,void *ctx);

+  x - input vector
.  A - Jacobian matrix
.  B - preconditioner matrix, usually the same as A
.  flag - flag indicating information about the preconditioner matrix
   structure (same as flag in KSPSetOperators())
-  ctx - [optional] user-defined Jacobian context

   Notes: 
   See KSPSetOperators() for important information about setting the flag
   output parameter in the routine func().  Be sure to read this information!

   The routine func() takes Mat * as the matrix arguments rather than Mat.  
   This allows the Jacobian evaluation routine to replace A and/or B with a 
   completely new new matrix structure (not just different matrix elements)
   when appropriate, for instance, if the nonzero structure is changing
   throughout the global iterations.

   Level: beginner

.keywords: SNES, nonlinear, set, Jacobian, matrix

.seealso: KSPSetOperators(), SNESSetFunction(), , MatSNESMFComputeJacobian(), SNESDefaultComputeJacobianColor()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESSetJacobian(SNES snes,Mat A,Mat B,PetscErrorCode (*func)(SNES,Vec,Mat*,Mat*,MatStructure*,void*),void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  if (A) PetscValidHeaderSpecific(A,MAT_COOKIE,2);
  if (B) PetscValidHeaderSpecific(B,MAT_COOKIE,3);
  if (A) PetscCheckSameComm(snes,1,A,2);
  if (B) PetscCheckSameComm(snes,1,B,2);
  if (func) snes->computejacobian = func;
  if (ctx)  snes->jacP            = ctx;
  if (A) {
    if (snes->jacobian) {ierr = MatDestroy(snes->jacobian);CHKERRQ(ierr);}
    snes->jacobian = A;
    ierr           = PetscObjectReference((PetscObject)A);CHKERRQ(ierr);
  }
  if (B) {
    if (snes->jacobian_pre) {ierr = MatDestroy(snes->jacobian_pre);CHKERRQ(ierr);}
    snes->jacobian_pre = B;
    ierr               = PetscObjectReference((PetscObject)B);CHKERRQ(ierr);
  }
  ierr = KSPSetOperators(snes->ksp,A,B,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESGetJacobian"
/*@C
   SNESGetJacobian - Returns the Jacobian matrix and optionally the user 
   provided context for evaluating the Jacobian.

   Not Collective, but Mat object will be parallel if SNES object is

   Input Parameter:
.  snes - the nonlinear solver context

   Output Parameters:
+  A - location to stash Jacobian matrix (or PETSC_NULL)
.  B - location to stash preconditioner matrix (or PETSC_NULL)
.  func - location to put Jacobian function (or PETSC_NULL)
-  ctx - location to stash Jacobian ctx (or PETSC_NULL)

   Level: advanced

.seealso: SNESSetJacobian(), SNESComputeJacobian()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESGetJacobian(SNES snes,Mat *A,Mat *B,PetscErrorCode (**func)(SNES,Vec,Mat*,Mat*,MatStructure*,void*),void **ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  if (A)    *A    = snes->jacobian;
  if (B)    *B    = snes->jacobian_pre;
  if (func) *func = snes->computejacobian;
  if (ctx)  *ctx  = snes->jacP;
  PetscFunctionReturn(0);
}

/* ----- Routines to initialize and destroy a nonlinear solver ---- */
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESDefaultMatrixFreeCreate2(SNES,Vec,Mat*);

#undef __FUNCT__  
#define __FUNCT__ "SNESSetUp"
/*@
   SNESSetUp - Sets up the internal data structures for the later use
   of a nonlinear solver.

   Collective on SNES

   Input Parameters:
.  snes - the SNES context

   Notes:
   For basic use of the SNES solvers the user need not explicitly call
   SNESSetUp(), since these actions will automatically occur during
   the call to SNESSolve().  However, if one wishes to control this
   phase separately, SNESSetUp() should be called after SNESCreate()
   and optional routines of the form SNESSetXXX(), but before SNESSolve().  

   Level: advanced

.keywords: SNES, nonlinear, setup

.seealso: SNESCreate(), SNESSolve(), SNESDestroy()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESSetUp(SNES snes)
{
  PetscErrorCode ierr;
  PetscTruth     flg, iseqtr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);

  ierr = PetscOptionsHasName(snes->prefix,"-snes_mf_operator",&flg);CHKERRQ(ierr); 
  /*
      This version replaces the user provided Jacobian matrix with a
      matrix-free version but still employs the user-provided preconditioner matrix
  */
  if (flg) {
    Mat J;
    ierr = MatCreateSNESMF(snes,snes->vec_sol,&J);CHKERRQ(ierr);
    ierr = MatSNESMFSetFromOptions(J);CHKERRQ(ierr);
    ierr = PetscLogInfo((snes,"SNESSetUp: Setting default matrix-free operator routines\n"));CHKERRQ(ierr);
    ierr = SNESSetJacobian(snes,J,0,0,0);CHKERRQ(ierr);
    ierr = MatDestroy(J);CHKERRQ(ierr);
  }

#if !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_SINGLE) && !defined(PETSC_USE_MAT_SINGLE)
  ierr = PetscOptionsHasName(snes->prefix,"-snes_mf_operator2",&flg);CHKERRQ(ierr); 
  if (flg) {
    Mat J;
    ierr = SNESDefaultMatrixFreeCreate2(snes,snes->vec_sol,&J);CHKERRQ(ierr);
    ierr = SNESSetJacobian(snes,J,0,0,0);CHKERRQ(ierr);
    ierr = MatDestroy(J);CHKERRQ(ierr);
  }
#endif

  ierr = PetscOptionsHasName(snes->prefix,"-snes_mf",&flg);CHKERRQ(ierr); 
  /*
      This version replaces both the user-provided Jacobian and the user-
      provided preconditioner matrix with the default matrix free version.
   */
  if (flg) {
    Mat  J;
    KSP ksp;
    PC   pc;

    ierr = MatCreateSNESMF(snes,snes->vec_sol,&J);CHKERRQ(ierr);
    ierr = MatSNESMFSetFromOptions(J);CHKERRQ(ierr);
    ierr = PetscLogInfo((snes,"SNESSetUp: Setting default matrix-free operator and preconditioner routines\n"));CHKERRQ(ierr);
    ierr = SNESSetJacobian(snes,J,J,MatSNESMFComputeJacobian,snes->funP);CHKERRQ(ierr);
    ierr = MatDestroy(J);CHKERRQ(ierr);

    /* force no preconditioner */
    ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    ierr = PetscTypeCompare((PetscObject)pc,PCSHELL,&flg);CHKERRQ(ierr);
    if (!flg) {
      ierr = PCSetType(pc,PCNONE);CHKERRQ(ierr);
    }
  }

  if (!snes->vec_func) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Must call SNESSetFunction() first");
  if (!snes->computefunction) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Must call SNESSetFunction() first");
  if (!snes->jacobian) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Must call SNESSetJacobian() first \n or use -snes_mf option");
  if (snes->vec_func == snes->vec_sol) {  
    SETERRQ(PETSC_ERR_ARG_IDN,"Solution vector cannot be function vector");
  }

  /* Set the KSP stopping criterion to use the Eisenstat-Walker method */
  ierr = PetscTypeCompare((PetscObject)snes,SNESTR,&iseqtr);CHKERRQ(ierr);
  if (snes->ksp_ewconv && !iseqtr) {
    KSP ksp;
    ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
    ierr = KSPSetConvergenceTest(ksp,SNES_KSP_EW_Converged_Private,snes);CHKERRQ(ierr);
  }

  if (snes->setup) {ierr = (*snes->setup)(snes);CHKERRQ(ierr);}
  snes->setupcalled = 1;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESDestroy"
/*@C
   SNESDestroy - Destroys the nonlinear solver context that was created
   with SNESCreate().

   Collective on SNES

   Input Parameter:
.  snes - the SNES context

   Level: beginner

.keywords: SNES, nonlinear, destroy

.seealso: SNESCreate(), SNESSolve()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESDestroy(SNES snes)
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  if (--snes->refct > 0) PetscFunctionReturn(0);

  /* if memory was published with AMS then destroy it */
  ierr = PetscObjectDepublish(snes);CHKERRQ(ierr);

  if (snes->destroy) {ierr = (*(snes)->destroy)(snes);CHKERRQ(ierr);}
  if (snes->kspconvctx) {ierr = PetscFree(snes->kspconvctx);CHKERRQ(ierr);}
  if (snes->jacobian) {ierr = MatDestroy(snes->jacobian);CHKERRQ(ierr);}
  if (snes->jacobian_pre) {ierr = MatDestroy(snes->jacobian_pre);CHKERRQ(ierr);}
  if (snes->afine) {ierr = VecDestroy(snes->afine);CHKERRQ(ierr);}
  ierr = KSPDestroy(snes->ksp);CHKERRQ(ierr);
  if (snes->vwork) {ierr = VecDestroyVecs(snes->vwork,snes->nvwork);CHKERRQ(ierr);}
  for (i=0; i<snes->numbermonitors; i++) {
    if (snes->monitordestroy[i]) {
      ierr = (*snes->monitordestroy[i])(snes->monitorcontext[i]);CHKERRQ(ierr);
    }
  }
  ierr = PetscHeaderDestroy(snes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ----------- Routines to set solver parameters ---------- */

#undef __FUNCT__  
#define __FUNCT__ "SNESSetTolerances"
/*@
   SNESSetTolerances - Sets various parameters used in convergence tests.

   Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  abstol - absolute convergence tolerance
.  rtol - relative convergence tolerance
.  stol -  convergence tolerance in terms of the norm
           of the change in the solution between steps
.  maxit - maximum number of iterations
-  maxf - maximum number of function evaluations

   Options Database Keys: 
+    -snes_atol <abstol> - Sets abstol
.    -snes_rtol <rtol> - Sets rtol
.    -snes_stol <stol> - Sets stol
.    -snes_max_it <maxit> - Sets maxit
-    -snes_max_funcs <maxf> - Sets maxf

   Notes:
   The default maximum number of iterations is 50.
   The default maximum number of function evaluations is 1000.

   Level: intermediate

.keywords: SNES, nonlinear, set, convergence, tolerances

.seealso: SNESSetTrustRegionTolerance()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESSetTolerances(SNES snes,PetscReal abstol,PetscReal rtol,PetscReal stol,PetscInt maxit,PetscInt maxf)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  if (abstol != PETSC_DEFAULT)  snes->abstol      = abstol;
  if (rtol != PETSC_DEFAULT)  snes->rtol      = rtol;
  if (stol != PETSC_DEFAULT)  snes->xtol      = stol;
  if (maxit != PETSC_DEFAULT) snes->max_its   = maxit;
  if (maxf != PETSC_DEFAULT)  snes->max_funcs = maxf;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESGetTolerances"
/*@
   SNESGetTolerances - Gets various parameters used in convergence tests.

   Not Collective

   Input Parameters:
+  snes - the SNES context
.  abstol - absolute convergence tolerance
.  rtol - relative convergence tolerance
.  stol -  convergence tolerance in terms of the norm
           of the change in the solution between steps
.  maxit - maximum number of iterations
-  maxf - maximum number of function evaluations

   Notes:
   The user can specify PETSC_NULL for any parameter that is not needed.

   Level: intermediate

.keywords: SNES, nonlinear, get, convergence, tolerances

.seealso: SNESSetTolerances()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESGetTolerances(SNES snes,PetscReal *abstol,PetscReal *rtol,PetscReal *stol,PetscInt *maxit,PetscInt *maxf)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  if (abstol)  *abstol  = snes->abstol;
  if (rtol)  *rtol  = snes->rtol;
  if (stol)  *stol  = snes->xtol;
  if (maxit) *maxit = snes->max_its;
  if (maxf)  *maxf  = snes->max_funcs;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESSetTrustRegionTolerance"
/*@
   SNESSetTrustRegionTolerance - Sets the trust region parameter tolerance.  

   Collective on SNES

   Input Parameters:
+  snes - the SNES context
-  tol - tolerance
   
   Options Database Key: 
.  -snes_trtol <tol> - Sets tol

   Level: intermediate

.keywords: SNES, nonlinear, set, trust region, tolerance

.seealso: SNESSetTolerances()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESSetTrustRegionTolerance(SNES snes,PetscReal tol)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  snes->deltatol = tol;
  PetscFunctionReturn(0);
}

/* 
   Duplicate the lg monitors for SNES from KSP; for some reason with 
   dynamic libraries things don't work under Sun4 if we just use 
   macros instead of functions
*/
#undef __FUNCT__  
#define __FUNCT__ "SNESLGMonitor"
PetscErrorCode PETSCSNES_DLLEXPORT SNESLGMonitor(SNES snes,PetscInt it,PetscReal norm,void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  ierr = KSPLGMonitor((KSP)snes,it,norm,ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESLGMonitorCreate"
PetscErrorCode PETSCSNES_DLLEXPORT SNESLGMonitorCreate(const char host[],const char label[],int x,int y,int m,int n,PetscDrawLG *draw)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPLGMonitorCreate(host,label,x,y,m,n,draw);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESLGMonitorDestroy"
PetscErrorCode PETSCSNES_DLLEXPORT SNESLGMonitorDestroy(PetscDrawLG draw)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPLGMonitorDestroy(draw);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------ Routines to set performance monitoring options ----------- */

#undef __FUNCT__  
#define __FUNCT__ "SNESSetMonitor"
/*@C
   SNESSetMonitor - Sets an ADDITIONAL function that is to be used at every
   iteration of the nonlinear solver to display the iteration's 
   progress.   

   Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  func - monitoring routine
.  mctx - [optional] user-defined context for private data for the 
          monitor routine (use PETSC_NULL if no context is desitre)
-  monitordestroy - [optional] routine that frees monitor context
          (may be PETSC_NULL)

   Calling sequence of func:
$     int func(SNES snes,PetscInt its, PetscReal norm,void *mctx)

+    snes - the SNES context
.    its - iteration number
.    norm - 2-norm function value (may be estimated)
-    mctx - [optional] monitoring context

   Options Database Keys:
+    -snes_monitor        - sets SNESDefaultMonitor()
.    -snes_xmonitor       - sets line graph monitor,
                            uses SNESLGMonitorCreate()
_    -snes_cancelmonitors - cancels all monitors that have
                            been hardwired into a code by 
                            calls to SNESSetMonitor(), but
                            does not cancel those set via
                            the options database.

   Notes: 
   Several different monitoring routines may be set by calling
   SNESSetMonitor() multiple times; all will be called in the 
   order in which they were set.

   Level: intermediate

.keywords: SNES, nonlinear, set, monitor

.seealso: SNESDefaultMonitor(), SNESClearMonitor()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESSetMonitor(SNES snes,PetscErrorCode (*func)(SNES,PetscInt,PetscReal,void*),void *mctx,PetscErrorCode (*monitordestroy)(void*))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  if (snes->numbermonitors >= MAXSNESMONITORS) {
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Too many monitors set");
  }
  snes->monitor[snes->numbermonitors]           = func;
  snes->monitordestroy[snes->numbermonitors]    = monitordestroy;
  snes->monitorcontext[snes->numbermonitors++]  = (void*)mctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESClearMonitor"
/*@C
   SNESClearMonitor - Clears all the monitor functions for a SNES object.

   Collective on SNES

   Input Parameters:
.  snes - the SNES context

   Options Database Key:
.  -snes_cancelmonitors - cancels all monitors that have been hardwired
    into a code by calls to SNESSetMonitor(), but does not cancel those 
    set via the options database

   Notes: 
   There is no way to clear one specific monitor from a SNES object.

   Level: intermediate

.keywords: SNES, nonlinear, set, monitor

.seealso: SNESDefaultMonitor(), SNESSetMonitor()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESClearMonitor(SNES snes)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  snes->numbermonitors = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESSetConvergenceTest"
/*@C
   SNESSetConvergenceTest - Sets the function that is to be used 
   to test for convergence of the nonlinear iterative solution.   

   Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  func - routine to test for convergence
-  cctx - [optional] context for private data for the convergence routine 
          (may be PETSC_NULL)

   Calling sequence of func:
$     PetscErrorCode func (SNES snes,PetscReal xnorm,PetscReal gnorm,PetscReal f,SNESConvergedReason *reason,void *cctx)

+    snes - the SNES context
.    cctx - [optional] convergence context
.    reason - reason for convergence/divergence
.    xnorm - 2-norm of current iterate
.    gnorm - 2-norm of current step
-    f - 2-norm of function

   Level: advanced

.keywords: SNES, nonlinear, set, convergence, test

.seealso: SNESConverged_LS(), SNESConverged_TR()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESSetConvergenceTest(SNES snes,PetscErrorCode (*func)(SNES,PetscReal,PetscReal,PetscReal,SNESConvergedReason*,void*),void *cctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  (snes)->converged = func;
  (snes)->cnvP      = cctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESGetConvergedReason"
/*@C
   SNESGetConvergedReason - Gets the reason the SNES iteration was stopped.

   Not Collective

   Input Parameter:
.  snes - the SNES context

   Output Parameter:
.  reason - negative value indicates diverged, positive value converged, see petscsnes.h or the 
            manual pages for the individual convergence tests for complete lists

   Level: intermediate

   Notes: Can only be called after the call the SNESSolve() is complete.

.keywords: SNES, nonlinear, set, convergence, test

.seealso: SNESSetConvergenceTest(), SNESConverged_LS(), SNESConverged_TR(), SNESConvergedReason
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESGetConvergedReason(SNES snes,SNESConvergedReason *reason)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  PetscValidPointer(reason,2);
  *reason = snes->reason;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESSetConvergenceHistory"
/*@
   SNESSetConvergenceHistory - Sets the array used to hold the convergence history.

   Collective on SNES

   Input Parameters:
+  snes - iterative context obtained from SNESCreate()
.  a   - array to hold history
.  its - integer array holds the number of linear iterations for each solve.
.  na  - size of a and its
-  reset - PETSC_TRUE indicates each new nonlinear solve resets the history counter to zero,
           else it continues storing new values for new nonlinear solves after the old ones

   Notes:
   If set, this array will contain the function norms computed
   at each step.

   This routine is useful, e.g., when running a code for purposes
   of accurate performance monitoring, when no I/O should be done
   during the section of code that is being timed.

   Level: intermediate

.keywords: SNES, set, convergence, history

.seealso: SNESGetConvergenceHistory()

@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESSetConvergenceHistory(SNES snes,PetscReal a[],PetscInt *its,PetscInt na,PetscTruth reset)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  if (na) PetscValidScalarPointer(a,2);
  snes->conv_hist       = a;
  snes->conv_hist_its   = its;
  snes->conv_hist_max   = na;
  snes->conv_hist_reset = reset;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESGetConvergenceHistory"
/*@C
   SNESGetConvergenceHistory - Gets the array used to hold the convergence history.

   Collective on SNES

   Input Parameter:
.  snes - iterative context obtained from SNESCreate()

   Output Parameters:
.  a   - array to hold history
.  its - integer array holds the number of linear iterations (or
         negative if not converged) for each solve.
-  na  - size of a and its

   Notes:
    The calling sequence for this routine in Fortran is
$   call SNESGetConvergenceHistory(SNES snes, integer na, integer ierr)

   This routine is useful, e.g., when running a code for purposes
   of accurate performance monitoring, when no I/O should be done
   during the section of code that is being timed.

   Level: intermediate

.keywords: SNES, get, convergence, history

.seealso: SNESSetConvergencHistory()

@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESGetConvergenceHistory(SNES snes,PetscReal *a[],PetscInt *its[],PetscInt *na)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  if (a)   *a   = snes->conv_hist;
  if (its) *its = snes->conv_hist_its;
  if (na) *na   = snes->conv_hist_len;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESSetUpdate"
/*@C
  SNESSetUpdate - Sets the general-purpose update function called
  at the beginning of every step of the iteration.

  Collective on SNES

  Input Parameters:
. snes - The nonlinear solver context
. func - The function

  Calling sequence of func:
. func (SNES snes, PetscInt step);

. step - The current step of the iteration

  Level: intermediate

.keywords: SNES, update

.seealso SNESDefaultUpdate(), SNESSetRhsBC(), SNESSetSolutionBC()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESSetUpdate(SNES snes, PetscErrorCode (*func)(SNES, PetscInt))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_COOKIE,1);
  snes->update = func;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESDefaultUpdate"
/*@
  SNESDefaultUpdate - The default update function which does nothing.

  Not collective

  Input Parameters:
. snes - The nonlinear solver context
. step - The current step of the iteration

  Level: intermediate

.keywords: SNES, update
.seealso SNESSetUpdate(), SNESDefaultRhsBC(), SNESDefaultSolutionBC()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESDefaultUpdate(SNES snes, PetscInt step)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESScaleStep_Private"
/*
   SNESScaleStep_Private - Scales a step so that its length is less than the
   positive parameter delta.

    Input Parameters:
+   snes - the SNES context
.   y - approximate solution of linear system
.   fnorm - 2-norm of current function
-   delta - trust region size

    Output Parameters:
+   gpnorm - predicted function norm at the new point, assuming local 
    linearization.  The value is zero if the step lies within the trust 
    region, and exceeds zero otherwise.
-   ynorm - 2-norm of the step

    Note:
    For non-trust region methods such as SNESLS, the parameter delta 
    is set to be the maximum allowable step size.  

.keywords: SNES, nonlinear, scale, step
*/
PetscErrorCode SNESScaleStep_Private(SNES snes,Vec y,PetscReal *fnorm,PetscReal *delta,PetscReal *gpnorm,PetscReal *ynorm)
{
  PetscReal      nrm;
  PetscScalar    cnorm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  PetscValidHeaderSpecific(y,VEC_COOKIE,2);
  PetscCheckSameComm(snes,1,y,2);

  ierr = VecNorm(y,NORM_2,&nrm);CHKERRQ(ierr);
  if (nrm > *delta) {
     nrm = *delta/nrm;
     *gpnorm = (1.0 - nrm)*(*fnorm);
     cnorm = nrm;
     ierr = VecScale(y,cnorm);CHKERRQ(ierr);
     *ynorm = *delta;
  } else {
     *gpnorm = 0.0;
     *ynorm = nrm;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESSolve"
/*@
   SNESSolve - Solves a nonlinear system F(x) = b.
   Call SNESSolve() after calling SNESCreate() and optional routines of the form SNESSetXXX().

   Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  b - the constant part of the equation, or PETSC_NULL to use zero.
-  x - the solution vector, or PETSC_NULL if it was set with SNESSetSolution()

   Notes:
   The user should initialize the vector,x, with the initial guess
   for the nonlinear solve prior to calling SNESSolve.  In particular,
   to employ an initial guess of zero, the user should explicitly set
   this vector to zero by calling VecSet().

   Level: beginner

.keywords: SNES, nonlinear, solve

.seealso: SNESCreate(), SNESDestroy(), SNESSetFunction(), SNESSetJacobian(), SNESSetRhs(), SNESSetSolution()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESSolve(SNES snes,Vec b,Vec x)
{
  PetscErrorCode ierr;
  PetscTruth     flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  if (!snes->solve) SETERRQ(PETSC_ERR_ORDER,"SNESSetType() or SNESSetFromOptions() must be called before SNESSolve()");

  if (b) {
    ierr = SNESSetRhs(snes, b); CHKERRQ(ierr);
  }
  if (x) {
    PetscValidHeaderSpecific(x,VEC_COOKIE,3);
    PetscCheckSameComm(snes,1,x,3);
  } else {
    ierr = SNESGetSolution(snes, &x); CHKERRQ(ierr);
    if (!x) {
      ierr = VecDuplicate(snes->vec_func_always, &x); CHKERRQ(ierr);
    }
  }
  snes->vec_sol = snes->vec_sol_always = x;
  if (!snes->setupcalled) {
    ierr = SNESSetUp(snes);CHKERRQ(ierr);
  }
  if (snes->conv_hist_reset) snes->conv_hist_len = 0;
  ierr = PetscLogEventBegin(SNES_Solve,snes,0,0,0);CHKERRQ(ierr);
  snes->nfuncs = 0; snes->linear_its = 0; snes->numFailures = 0;

  ierr = PetscExceptionTry1((*(snes)->solve)(snes),PETSC_ERR_ARG_DOMAIN);
  if (PetscExceptionValue(ierr)) {
    /* this means that a caller above me has also tryed this exception so I don't handle it here, pass it up */
    PetscErrorCode pierr = PetscLogEventEnd(SNES_Solve,snes,0,0,0);CHKERRQ(pierr);
  } else if (PetscExceptionCaught(ierr,PETSC_ERR_ARG_DOMAIN)) {
    /* translate exception into SNES not converged reason */
    snes->reason = SNES_DIVERGED_FUNCTION_DOMAIN;
    ierr = 0;
  } 
  CHKERRQ(ierr);

  ierr = PetscLogEventEnd(SNES_Solve,snes,0,0,0);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(snes->prefix,"-snes_view",&flg);CHKERRQ(ierr);
  if (flg && !PetscPreLoadingOn) { ierr = SNESView(snes,PETSC_VIEWER_STDOUT_(snes->comm));CHKERRQ(ierr); }
  ierr = PetscOptionsHasName(snes->prefix,"-snes_test_local_min",&flg);CHKERRQ(ierr);
  if (flg && !PetscPreLoadingOn) { ierr = SNESTestLocalMin(snes);CHKERRQ(ierr); }
  if (snes->printreason) {
    if (snes->reason > 0) {
      ierr = PetscPrintf(snes->comm,"Nonlinear solve converged due to %s\n",SNESConvergedReasons[snes->reason]);CHKERRQ(ierr);
    } else {
      ierr = PetscPrintf(snes->comm,"Nonlinear solve did not converge due to %s\n",SNESConvergedReasons[snes->reason]);CHKERRQ(ierr);
    }
  }

  PetscFunctionReturn(0);
}

/* --------- Internal routines for SNES Package --------- */

#undef __FUNCT__  
#define __FUNCT__ "SNESSetType"
/*@C
   SNESSetType - Sets the method for the nonlinear solver.  

   Collective on SNES

   Input Parameters:
+  snes - the SNES context
-  type - a known method

   Options Database Key:
.  -snes_type <type> - Sets the method; use -help for a list
   of available methods (for instance, ls or tr)

   Notes:
   See "petsc/include/petscsnes.h" for available methods (for instance)
+    SNESLS - Newton's method with line search
     (systems of nonlinear equations)
.    SNESTR - Newton's method with trust region
     (systems of nonlinear equations)

  Normally, it is best to use the SNESSetFromOptions() command and then
  set the SNES solver type from the options database rather than by using
  this routine.  Using the options database provides the user with
  maximum flexibility in evaluating the many nonlinear solvers.
  The SNESSetType() routine is provided for those situations where it
  is necessary to set the nonlinear solver independently of the command
  line or options database.  This might be the case, for example, when
  the choice of solver changes during the execution of the program,
  and the user's application is taking responsibility for choosing the
  appropriate method.

  Level: intermediate

.keywords: SNES, set, type

.seealso: SNESType, SNESCreate()

@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESSetType(SNES snes,SNESType type)
{
  PetscErrorCode ierr,(*r)(SNES);
  PetscTruth     match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  PetscValidCharPointer(type,2);

  ierr = PetscTypeCompare((PetscObject)snes,type,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  if (snes->setupcalled) {
    ierr       = (*(snes)->destroy)(snes);CHKERRQ(ierr);
    snes->data = 0;
  }

  /* Get the function pointers for the iterative method requested */
  if (!SNESRegisterAllCalled) {ierr = SNESRegisterAll(PETSC_NULL);CHKERRQ(ierr);}
  ierr =  PetscFListFind(snes->comm,SNESList,type,(void (**)(void)) &r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PETSC_ERR_ARG_UNKNOWN_TYPE,"Unable to find requested SNES type %s",type);
  if (snes->data) {ierr = PetscFree(snes->data);CHKERRQ(ierr);}
  snes->data = 0;
  ierr = (*r)(snes);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)snes,type);CHKERRQ(ierr);
  PetscFunctionReturn(0); 
}


/* --------------------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "SNESRegisterDestroy"
/*@C
   SNESRegisterDestroy - Frees the list of nonlinear solvers that were
   registered by SNESRegisterDynamic().

   Not Collective

   Level: advanced

.keywords: SNES, nonlinear, register, destroy

.seealso: SNESRegisterAll(), SNESRegisterAll()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESRegisterDestroy(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (SNESList) {
    ierr = PetscFListDestroy(&SNESList);CHKERRQ(ierr);
    SNESList = 0;
  }
  SNESRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESGetType"
/*@C
   SNESGetType - Gets the SNES method type and name (as a string).

   Not Collective

   Input Parameter:
.  snes - nonlinear solver context

   Output Parameter:
.  type - SNES method (a character string)

   Level: intermediate

.keywords: SNES, nonlinear, get, type, name
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESGetType(SNES snes,SNESType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  PetscValidPointer(type,2);
  *type = snes->type_name;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESGetSolution"
/*@C
   SNESGetSolution - Returns the vector where the approximate solution is
   stored.

   Not Collective, but Vec is parallel if SNES is parallel

   Input Parameter:
.  snes - the SNES context

   Output Parameter:
.  x - the solution

   Level: intermediate

.keywords: SNES, nonlinear, get, solution

.seealso: SNESSetSolution(), SNESGetFunction(), SNESGetSolutionUpdate()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESGetSolution(SNES snes,Vec *x)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  PetscValidPointer(x,2);
  *x = snes->vec_sol_always;
  PetscFunctionReturn(0);
}  

#undef __FUNCT__  
#define __FUNCT__ "SNESSetSolution"
/*@C
   SNESSetSolution - Sets the vector where the approximate solution is stored.

   Not Collective, but Vec is parallel if SNES is parallel

   Input Parameters:
+  snes - the SNES context
-  x - the solution

   Output Parameter:

   Level: intermediate

.keywords: SNES, nonlinear, set, solution

.seealso: SNESGetSolution(), SNESGetFunction(), SNESGetSolutionUpdate()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESSetSolution(SNES snes,Vec x)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  PetscValidHeaderSpecific(x,VEC_COOKIE,2);
  PetscCheckSameComm(snes,1,x,2);
  snes->vec_sol_always = x;
  PetscFunctionReturn(0);
}  

#undef __FUNCT__  
#define __FUNCT__ "SNESGetSolutionUpdate"
/*@C
   SNESGetSolutionUpdate - Returns the vector where the solution update is
   stored. 

   Not Collective, but Vec is parallel if SNES is parallel

   Input Parameter:
.  snes - the SNES context

   Output Parameter:
.  x - the solution update

   Level: advanced

.keywords: SNES, nonlinear, get, solution, update

.seealso: SNESGetSolution(), SNESGetFunction
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESGetSolutionUpdate(SNES snes,Vec *x)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  PetscValidPointer(x,2);
  *x = snes->vec_sol_update_always;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESGetFunction"
/*@C
   SNESGetFunction - Returns the vector where the function is stored.

   Not Collective, but Vec is parallel if SNES is parallel

   Input Parameter:
.  snes - the SNES context

   Output Parameter:
+  r - the function (or PETSC_NULL)
.  func - the function (or PETSC_NULL)
-  ctx - the function context (or PETSC_NULL)

   Level: advanced

.keywords: SNES, nonlinear, get, function

.seealso: SNESSetFunction(), SNESGetSolution()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESGetFunction(SNES snes,Vec *r,PetscErrorCode (**func)(SNES,Vec,Vec,void*),void **ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  if (r)    *r    = snes->vec_func_always;
  if (func) *func = snes->computefunction;
  if (ctx)  *ctx  = snes->funP;
  PetscFunctionReturn(0);
}  

#undef __FUNCT__  
#define __FUNCT__ "SNESSetOptionsPrefix"
/*@C
   SNESSetOptionsPrefix - Sets the prefix used for searching for all 
   SNES options in the database.

   Collective on SNES

   Input Parameter:
+  snes - the SNES context
-  prefix - the prefix to prepend to all option names

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the hyphen.

   Level: advanced

.keywords: SNES, set, options, prefix, database

.seealso: SNESSetFromOptions()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESSetOptionsPrefix(SNES snes,const char prefix[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)snes,prefix);CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(snes->ksp,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESAppendOptionsPrefix"
/*@C
   SNESAppendOptionsPrefix - Appends to the prefix used for searching for all 
   SNES options in the database.

   Collective on SNES

   Input Parameters:
+  snes - the SNES context
-  prefix - the prefix to prepend to all option names

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the hyphen.

   Level: advanced

.keywords: SNES, append, options, prefix, database

.seealso: SNESGetOptionsPrefix()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESAppendOptionsPrefix(SNES snes,const char prefix[])
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  ierr = PetscObjectAppendOptionsPrefix((PetscObject)snes,prefix);CHKERRQ(ierr);
  ierr = KSPAppendOptionsPrefix(snes->ksp,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESGetOptionsPrefix"
/*@C
   SNESGetOptionsPrefix - Sets the prefix used for searching for all 
   SNES options in the database.

   Not Collective

   Input Parameter:
.  snes - the SNES context

   Output Parameter:
.  prefix - pointer to the prefix string used

   Notes: On the fortran side, the user should pass in a string 'prifix' of
   sufficient length to hold the prefix.

   Level: advanced

.keywords: SNES, get, options, prefix, database

.seealso: SNESAppendOptionsPrefix()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESGetOptionsPrefix(SNES snes,const char *prefix[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  ierr = PetscObjectGetOptionsPrefix((PetscObject)snes,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "SNESRegister"
/*@C
  SNESRegister - See SNESRegisterDynamic()

  Level: advanced
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESRegister(const char sname[],const char path[],const char name[],PetscErrorCode (*function)(SNES))
{
  char           fullname[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFListConcat(path,name,fullname);CHKERRQ(ierr);
  ierr = PetscFListAdd(&SNESList,sname,fullname,(void (*)(void))function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESTestLocalMin"
PetscErrorCode PETSCSNES_DLLEXPORT SNESTestLocalMin(SNES snes)
{
  PetscErrorCode ierr;
  PetscInt       N,i,j;
  Vec            u,uh,fh;
  PetscScalar    value;
  PetscReal      norm;

  PetscFunctionBegin;
  ierr = SNESGetSolution(snes,&u);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&uh);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&fh);CHKERRQ(ierr);

  /* currently only works for sequential */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Testing FormFunction() for local min\n");
  ierr = VecGetSize(u,&N);CHKERRQ(ierr);
  for (i=0; i<N; i++) {
    ierr = VecCopy(u,uh);CHKERRQ(ierr);
    ierr  = PetscPrintf(PETSC_COMM_WORLD,"i = %D\n",i);CHKERRQ(ierr);
    for (j=-10; j<11; j++) {
      value = PetscSign(j)*exp(PetscAbs(j)-10.0);
      ierr  = VecSetValue(uh,i,value,ADD_VALUES);CHKERRQ(ierr);
      ierr  = SNESComputeFunction(snes,uh,fh);CHKERRQ(ierr);
      ierr  = VecNorm(fh,NORM_2,&norm);CHKERRQ(ierr);
      ierr  = PetscPrintf(PETSC_COMM_WORLD,"       j norm %D %18.16e\n",j,norm);CHKERRQ(ierr);
      value = -value;
      ierr  = VecSetValue(uh,i,value,ADD_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = VecDestroy(uh);CHKERRQ(ierr);
  ierr = VecDestroy(fh);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
