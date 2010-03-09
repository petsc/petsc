#define PETSCSNES_DLL

#include "private/snesimpl.h"      /*I "petscsnes.h"  I*/

PetscTruth SNESRegisterAllCalled = PETSC_FALSE;
PetscFList SNESList              = PETSC_NULL;

/* Logging support */
PetscCookie PETSCSNES_DLLEXPORT SNES_COOKIE;
PetscLogEvent  SNES_Solve, SNES_LineSearch, SNES_FunctionEval, SNES_JacobianEval;

#undef __FUNCT__  
#define __FUNCT__ "SNESSetFunctionDomainError"
/*@
   SNESSetFunctionDomainError - tells SNES that the input vector to your FormFunction is not
     in the functions domain. For example, negative pressure.

   Collective on SNES

   Input Parameters:
.  SNES - the SNES context

   Level: advanced

.keywords: SNES, view

.seealso: SNESCreate(), SNESSetFunction()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESSetFunctionDomainError(SNES snes)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  snes->domainerror = PETSC_TRUE;
  PetscFunctionReturn(0);
}

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
  SNESKSPEW           *kctx;
  PetscErrorCode      ierr;
  KSP                 ksp;
  const SNESType      type;
  PetscTruth          iascii,isstring;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  if (!viewer) {
    ierr = PetscViewerASCIIGetStdout(((PetscObject)snes)->comm,&viewer);CHKERRQ(ierr);
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_COOKIE,2);
  PetscCheckSameComm(snes,1,viewer,2);

  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_STRING,&isstring);CHKERRQ(ierr);
  if (iascii) {
    if (((PetscObject)snes)->prefix) {
      ierr = PetscViewerASCIIPrintf(viewer,"SNES Object:(%s)\n",((PetscObject)snes)->prefix);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"SNES Object:\n");CHKERRQ(ierr);
    }
    ierr = SNESGetType(snes,&type);CHKERRQ(ierr);
    if (type) {
      ierr = PetscViewerASCIIPrintf(viewer,"  type: %s\n",type);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  type: not set yet\n");CHKERRQ(ierr);
    }
    if (snes->ops->view) {
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = (*snes->ops->view)(snes,viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"  maximum iterations=%D, maximum function evaluations=%D\n",snes->max_its,snes->max_funcs);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  tolerances: relative=%G, absolute=%G, solution=%G\n",
                 snes->rtol,snes->abstol,snes->xtol);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  total number of linear solver iterations=%D\n",snes->linear_its);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  total number of function evaluations=%D\n",snes->nfuncs);CHKERRQ(ierr);
    if (snes->ksp_ewconv) {
      kctx = (SNESKSPEW *)snes->kspconvctx;
      if (kctx) {
        ierr = PetscViewerASCIIPrintf(viewer,"  Eisenstat-Walker computation of KSP relative tolerance (version %D)\n",kctx->version);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"    rtol_0=%G, rtol_max=%G, threshold=%G\n",kctx->rtol_0,kctx->rtol_max,kctx->threshold);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"    gamma=%G, alpha=%G, alpha2=%G\n",kctx->gamma,kctx->alpha,kctx->alpha2);CHKERRQ(ierr);
      }
    }
    if (snes->lagpreconditioner == -1) {
      ierr = PetscViewerASCIIPrintf(viewer,"  Preconditioned is never rebuilt\n");CHKERRQ(ierr);
    } else if (snes->lagpreconditioner > 1) {
      ierr = PetscViewerASCIIPrintf(viewer,"  Preconditioned is rebuilt every %D new Jacobians\n",snes->lagpreconditioner);CHKERRQ(ierr);
    }
    if (snes->lagjacobian == -1) {
      ierr = PetscViewerASCIIPrintf(viewer,"  Jacobian is never rebuilt\n");CHKERRQ(ierr);
    } else if (snes->lagjacobian > 1) {
      ierr = PetscViewerASCIIPrintf(viewer,"  Jacobian is rebuilt every %D Newton iterations\n",snes->lagjacobian);CHKERRQ(ierr);
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

EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESDefaultMatrixFreeCreate2(SNES,Vec,Mat*);

#undef __FUNCT__  
#define __FUNCT__ "SNESSetUpMatrixFree_Private"
static PetscErrorCode SNESSetUpMatrixFree_Private(SNES snes, PetscTruth hasOperator, PetscInt version)
{
  Mat            J;
  KSP            ksp;
  PC             pc;
  PetscTruth     match;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);

  if(!snes->vec_func && (snes->jacobian || snes->jacobian_pre)) {
    Mat A = snes->jacobian, B = snes->jacobian_pre;
    ierr = MatGetVecs(A ? A : B, PETSC_NULL,&snes->vec_func);CHKERRQ(ierr);
  }

  if (version == 1) {
    ierr = MatCreateSNESMF(snes,&J);CHKERRQ(ierr);
    ierr = MatMFFDSetOptionsPrefix(J,((PetscObject)snes)->prefix);CHKERRQ(ierr);
    ierr = MatMFFDSetFromOptions(J);CHKERRQ(ierr);
  } else if (version == 2) {
    if (!snes->vec_func) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"SNESSetFunction() must be called first");
#if !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_SCALAR_SINGLE) && !defined(PETSC_USE_SCALAR_MAT_SINGLE) && !defined(PETSC_USE_SCALAR_LONG_DOUBLE) && !defined(PETSC_USE_SCALAR_INT)
    ierr = SNESDefaultMatrixFreeCreate2(snes,snes->vec_func,&J);CHKERRQ(ierr);
#else
    SETERRQ(PETSC_ERR_SUP, "matrix-free operator rutines (version 2)");
#endif
  } else {
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE, "matrix-free operator rutines, only version 1 and 2");
  }
  
  ierr = PetscInfo1(snes,"Setting default matrix-free operator routines (version %D)\n", version);CHKERRQ(ierr);
  if (hasOperator) {
    /* This version replaces the user provided Jacobian matrix with a
       matrix-free version but still employs the user-provided preconditioner matrix. */
    ierr = SNESSetJacobian(snes,J,0,0,0);CHKERRQ(ierr);
  } else {
    /* This version replaces both the user-provided Jacobian and the user-
       provided preconditioner matrix with the default matrix free version. */
    ierr = SNESSetJacobian(snes,J,J,MatMFFDComputeJacobian,snes->funP);CHKERRQ(ierr);
    /* Force no preconditioner */
    ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    ierr = PetscTypeCompare((PetscObject)pc,PCSHELL,&match);CHKERRQ(ierr);
    if (!match) {
      ierr = PetscInfo(snes,"Setting default matrix-free preconditioner routines\nThat is no preconditioner is being used\n");CHKERRQ(ierr);
      ierr = PCSetType(pc,PCNONE);CHKERRQ(ierr);
    }
  }
  ierr = MatDestroy(J);CHKERRQ(ierr);

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
.  -snes_max_linear_solve_fail - number of linear solver failures before SNESSolve() stops
.  -snes_lag_preconditioner <lag> - how often preconditioner is rebuilt (use -1 to never rebuild)
.  -snes_lag_jacobian <lag> - how often Jacobian is rebuilt (use -1 to never rebuild)
.  -snes_trtol <trtol> - trust region tolerance
.  -snes_no_convergence_test - skip convergence test in nonlinear 
                               solver; hence iterations will continue until max_it
                               or some other criterion is reached. Saves expense
                               of convergence test
.  -snes_monitor <optional filename> - prints residual norm at each iteration. if no
                                       filename given prints to stdout
.  -snes_monitor_solution - plots solution at each iteration
.  -snes_monitor_residual - plots residual (not its norm) at each iteration
.  -snes_monitor_solution_update - plots update to solution at each iteration 
.  -snes_monitor_draw - plots residual norm at each iteration 
.  -snes_fd - use finite differences to compute Jacobian; very slow, only for testing
.  -snes_mf_ksp_monitor - if using matrix-free multiply then print h at each KSP iteration
-  -snes_converged_reason - print the reason for convergence/divergence after each solve

    Options Database for Eisenstat-Walker method:
+  -snes_ksp_ew - use Eisenstat-Walker method for determining linear system convergence
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
  PetscTruth              flg,mf,mf_operator;
  PetscInt                i,indx,lag,mf_version;
  MatStructure            matflag;
  const char              *deft = SNESLS;
  const char              *convtests[] = {"default","skip"};
  SNESKSPEW               *kctx = NULL;
  char                    type[256], monfilename[PETSC_MAX_PATH_LEN];
  PetscViewerASCIIMonitor monviewer;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);

  if (!SNESRegisterAllCalled) {ierr = SNESRegisterAll(PETSC_NULL);CHKERRQ(ierr);}
  ierr = PetscOptionsBegin(((PetscObject)snes)->comm,((PetscObject)snes)->prefix,"Nonlinear solver (SNES) options","SNES");CHKERRQ(ierr); 
    if (((PetscObject)snes)->type_name) { deft = ((PetscObject)snes)->type_name; } 
    ierr = PetscOptionsList("-snes_type","Nonlinear solver method","SNESSetType",SNESList,deft,type,256,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = SNESSetType(snes,type);CHKERRQ(ierr);
    } else if (!((PetscObject)snes)->type_name) {
      ierr = SNESSetType(snes,deft);CHKERRQ(ierr);
    }
    /* not used here, but called so will go into help messaage */
    ierr = PetscOptionsName("-snes_view","Print detailed information on solver used","SNESView",0);CHKERRQ(ierr);

    ierr = PetscOptionsReal("-snes_stol","Stop if step length less than","SNESSetTolerances",snes->xtol,&snes->xtol,0);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-snes_atol","Stop if function norm less than","SNESSetTolerances",snes->abstol,&snes->abstol,0);CHKERRQ(ierr);

    ierr = PetscOptionsReal("-snes_rtol","Stop if decrease in function norm less than","SNESSetTolerances",snes->rtol,&snes->rtol,0);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-snes_max_it","Maximum iterations","SNESSetTolerances",snes->max_its,&snes->max_its,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-snes_max_funcs","Maximum function evaluations","SNESSetTolerances",snes->max_funcs,&snes->max_funcs,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-snes_max_fail","Maximum failures","SNESSetTolerances",snes->maxFailures,&snes->maxFailures,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-snes_max_linear_solve_fail","Maximum failures in linear solves allowed","SNESSetMaxLinearSolveFailures",snes->maxLinearSolveFailures,&snes->maxLinearSolveFailures,PETSC_NULL);CHKERRQ(ierr);

    ierr = PetscOptionsInt("-snes_lag_preconditioner","How often to rebuild preconditioner","SNESSetLagPreconditioner",snes->lagpreconditioner,&lag,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = SNESSetLagPreconditioner(snes,lag);CHKERRQ(ierr);
    }
    ierr = PetscOptionsInt("-snes_lag_jacobian","How often to rebuild Jacobian","SNESSetLagJacobian",snes->lagjacobian,&lag,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = SNESSetLagJacobian(snes,lag);CHKERRQ(ierr);
    }

    ierr = PetscOptionsEList("-snes_convergence_test","Convergence test","SNESSetConvergenceTest",convtests,2,"default",&indx,&flg);CHKERRQ(ierr);
    if (flg) {
      switch (indx) {
      case 0: ierr = SNESSetConvergenceTest(snes,SNESDefaultConverged,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr); break;
      case 1: ierr = SNESSetConvergenceTest(snes,SNESSkipConverged,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);    break;
      }
    }

    ierr = PetscOptionsTruth("-snes_converged_reason","Print reason for converged or diverged","SNESSolve",snes->printreason,&snes->printreason,PETSC_NULL);CHKERRQ(ierr);

    kctx = (SNESKSPEW *)snes->kspconvctx;

    ierr = PetscOptionsTruth("-snes_ksp_ew","Use Eisentat-Walker linear system convergence test","SNESKSPSetUseEW",snes->ksp_ewconv,&snes->ksp_ewconv,PETSC_NULL);CHKERRQ(ierr);

    ierr = PetscOptionsInt("-snes_ksp_ew_version","Version 1, 2 or 3","SNESKSPSetParametersEW",kctx->version,&kctx->version,0);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-snes_ksp_ew_rtol0","0 <= rtol0 < 1","SNESKSPSetParametersEW",kctx->rtol_0,&kctx->rtol_0,0);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-snes_ksp_ew_rtolmax","0 <= rtolmax < 1","SNESKSPSetParametersEW",kctx->rtol_max,&kctx->rtol_max,0);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-snes_ksp_ew_gamma","0 <= gamma <= 1","SNESKSPSetParametersEW",kctx->gamma,&kctx->gamma,0);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-snes_ksp_ew_alpha","1 < alpha <= 2","SNESKSPSetParametersEW",kctx->alpha,&kctx->alpha,0);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-snes_ksp_ew_alpha2","alpha2","SNESKSPSetParametersEW",kctx->alpha2,&kctx->alpha2,0);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-snes_ksp_ew_threshold","0 < threshold < 1","SNESKSPSetParametersEW",kctx->threshold,&kctx->threshold,0);CHKERRQ(ierr);

    flg  = PETSC_FALSE;
    ierr = PetscOptionsTruth("-snes_monitor_cancel","Remove all monitors","SNESMonitorCancel",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) {ierr = SNESMonitorCancel(snes);CHKERRQ(ierr);}

    ierr = PetscOptionsString("-snes_monitor","Monitor norm of function","SNESMonitorSet","stdout",monfilename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PetscViewerASCIIMonitorCreate(((PetscObject)snes)->comm,monfilename,((PetscObject)snes)->tablevel,&monviewer);CHKERRQ(ierr);
      ierr = SNESMonitorSet(snes,SNESMonitorDefault,monviewer,(PetscErrorCode (*)(void*))PetscViewerASCIIMonitorDestroy);CHKERRQ(ierr);
    }

    ierr = PetscOptionsString("-snes_monitor_range","Monitor range of elements of function","SNESMonitorSet","stdout",monfilename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = SNESMonitorSet(snes,SNESMonitorRange,0,0);CHKERRQ(ierr);
    }

    ierr = PetscOptionsString("-snes_ratiomonitor","Monitor ratios of norms of function","SNESMonitorSetRatio","stdout",monfilename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PetscViewerASCIIMonitorCreate(((PetscObject)snes)->comm,monfilename,((PetscObject)snes)->tablevel,&monviewer);CHKERRQ(ierr);
      ierr = SNESMonitorSetRatio(snes,monviewer);CHKERRQ(ierr);
    }

    ierr = PetscOptionsString("-snes_monitor_short","Monitor norm of function (fewer digits)","SNESMonitorSet","stdout",monfilename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PetscViewerASCIIMonitorCreate(((PetscObject)snes)->comm,monfilename,((PetscObject)snes)->tablevel,&monviewer);CHKERRQ(ierr);
      ierr = SNESMonitorSet(snes,SNESMonitorDefaultShort,monviewer,(PetscErrorCode (*)(void*))PetscViewerASCIIMonitorDestroy);CHKERRQ(ierr);
    }

    flg  = PETSC_FALSE;
    ierr = PetscOptionsTruth("-snes_monitor_solution","Plot solution at each iteration","SNESMonitorSolution",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) {ierr = SNESMonitorSet(snes,SNESMonitorSolution,0,0);CHKERRQ(ierr);}
    flg  = PETSC_FALSE;
    ierr = PetscOptionsTruth("-snes_monitor_solution_update","Plot correction at each iteration","SNESMonitorSolutionUpdate",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) {ierr = SNESMonitorSet(snes,SNESMonitorSolutionUpdate,0,0);CHKERRQ(ierr);}
    flg  = PETSC_FALSE;
    ierr = PetscOptionsTruth("-snes_monitor_residual","Plot residual at each iteration","SNESMonitorResidual",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) {ierr = SNESMonitorSet(snes,SNESMonitorResidual,0,0);CHKERRQ(ierr);}
    flg  = PETSC_FALSE;
    ierr = PetscOptionsTruth("-snes_monitor_draw","Plot function norm at each iteration","SNESMonitorLG",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) {ierr = SNESMonitorSet(snes,SNESMonitorLG,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);}
    flg  = PETSC_FALSE;
    ierr = PetscOptionsTruth("-snes_monitor_range_draw","Plot function range at each iteration","SNESMonitorLG",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) {ierr = SNESMonitorSet(snes,SNESMonitorLGRange,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);}

    flg  = PETSC_FALSE;
    ierr = PetscOptionsTruth("-snes_fd","Use finite differences (slow) to compute Jacobian","SNESDefaultComputeJacobian",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) {
      ierr = SNESSetJacobian(snes,snes->jacobian,snes->jacobian_pre,SNESDefaultComputeJacobian,snes->funP);CHKERRQ(ierr);
      ierr = PetscInfo(snes,"Setting default finite difference Jacobian matrix\n");CHKERRQ(ierr);
    }

    mf = mf_operator = PETSC_FALSE;
    flg = PETSC_FALSE;
    ierr = PetscOptionsTruth("-snes_mf_operator","Use a Matrix-Free Jacobian with user-provided preconditioner matrix","MatCreateSNESMF",PETSC_FALSE,&mf_operator,&flg);CHKERRQ(ierr);
    if (flg && mf_operator) mf = PETSC_TRUE;
    flg = PETSC_FALSE;
    ierr = PetscOptionsTruth("-snes_mf","Use a Matrix-Free Jacobian with no preconditioner matrix","MatCreateSNESMF",PETSC_FALSE,&mf,&flg);CHKERRQ(ierr);
    if (!flg && mf_operator) mf = PETSC_TRUE;
    mf_version = 1;
    ierr = PetscOptionsInt("-snes_mf_version","Matrix-Free routines version 1 or 2","None",mf_version,&mf_version,0);CHKERRQ(ierr);

    for(i = 0; i < numberofsetfromoptions; i++) {
      ierr = (*othersetfromoptions[i])(snes);CHKERRQ(ierr);
    }

    if (snes->ops->setfromoptions) {
      ierr = (*snes->ops->setfromoptions)(snes);CHKERRQ(ierr);
    }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  if (mf) { ierr = SNESSetUpMatrixFree_Private(snes, mf_operator, mf_version);CHKERRQ(ierr); }
  
  if (!snes->ksp) {ierr = SNESGetKSP(snes,&snes->ksp);CHKERRQ(ierr);}
  ierr = KSPGetOperators(snes->ksp,PETSC_NULL,PETSC_NULL,&matflag);
  ierr = KSPSetOperators(snes->ksp,snes->jacobian,snes->jacobian_pre,matflag);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(snes->ksp);CHKERRQ(ierr);

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

.keywords: SNES, nonlinear, get, iteration, number, 

.seealso:   SNESGetFunctionNorm(), SNESGetLinearSolveIterations()
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

.seealso: SNESGetFunction(), SNESGetIterationNumber(), SNESGetLinearSolveIterations()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESGetFunctionNorm(SNES snes,PetscReal *fnorm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  PetscValidScalarPointer(fnorm,2);
  *fnorm = snes->norm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESGetNonlinearStepFailures"
/*@
   SNESGetNonlinearStepFailures - Gets the number of unsuccessful steps
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

.seealso: SNESGetMaxLinearSolveFailures(), SNESGetLinearSolveIterations(), SNESSetMaxLinearSolveFailures(), SNESGetLinearSolveFailures(),
          SNESSetMaxNonlinearStepFailures(), SNESGetMaxNonlinearStepFailures()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESGetNonlinearStepFailures(SNES snes,PetscInt* nfails)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  PetscValidIntPointer(nfails,2);
  *nfails = snes->numFailures;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESSetMaxNonlinearStepFailures"
/*@
   SNESSetMaxNonlinearStepFailures - Sets the maximum number of unsuccessful steps
   attempted by the nonlinear solver before it gives up.

   Not Collective

   Input Parameters:
+  snes     - SNES context
-  maxFails - maximum of unsuccessful steps

   Level: intermediate

.keywords: SNES, nonlinear, set, maximum, unsuccessful, steps

.seealso: SNESGetMaxLinearSolveFailures(), SNESGetLinearSolveIterations(), SNESSetMaxLinearSolveFailures(), SNESGetLinearSolveFailures(),
          SNESGetMaxNonlinearStepFailures(), SNESGetNonlinearStepFailures()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESSetMaxNonlinearStepFailures(SNES snes, PetscInt maxFails)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  snes->maxFailures = maxFails;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESGetMaxNonlinearStepFailures"
/*@
   SNESGetMaxNonlinearStepFailures - Gets the maximum number of unsuccessful steps
   attempted by the nonlinear solver before it gives up.

   Not Collective

   Input Parameter:
.  snes     - SNES context

   Output Parameter:
.  maxFails - maximum of unsuccessful steps

   Level: intermediate

.keywords: SNES, nonlinear, get, maximum, unsuccessful, steps

.seealso: SNESGetMaxLinearSolveFailures(), SNESGetLinearSolveIterations(), SNESSetMaxLinearSolveFailures(), SNESGetLinearSolveFailures(),
          SNESSetMaxNonlinearStepFailures(), SNESGetNonlinearStepFailures()
 
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESGetMaxNonlinearStepFailures(SNES snes, PetscInt *maxFails)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  PetscValidIntPointer(maxFails,2);
  *maxFails = snes->maxFailures;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESGetNumberFunctionEvals"
/*@
   SNESGetNumberFunctionEvals - Gets the number of user provided function evaluations
     done by SNES.

   Not Collective

   Input Parameter:
.  snes     - SNES context

   Output Parameter:
.  nfuncs - number of evaluations

   Level: intermediate

.keywords: SNES, nonlinear, get, maximum, unsuccessful, steps

.seealso: SNESGetMaxLinearSolveFailures(), SNESGetLinearSolveIterations(), SNESSetMaxLinearSolveFailures(), SNESGetLinearSolveFailures()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESGetNumberFunctionEvals(SNES snes, PetscInt *nfuncs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  PetscValidIntPointer(nfuncs,2);
  *nfuncs = snes->nfuncs;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESGetLinearSolveFailures"
/*@
   SNESGetLinearSolveFailures - Gets the number of failed (non-converged)
   linear solvers.

   Not Collective

   Input Parameter:
.  snes - SNES context

   Output Parameter:
.  nfails - number of failed solves

   Notes:
   This counter is reset to zero for each successive call to SNESSolve().

   Level: intermediate

.keywords: SNES, nonlinear, get, number, unsuccessful, steps

.seealso: SNESGetMaxLinearSolveFailures(), SNESGetLinearSolveIterations(), SNESSetMaxLinearSolveFailures()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESGetLinearSolveFailures(SNES snes,PetscInt* nfails)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  PetscValidIntPointer(nfails,2);
  *nfails = snes->numLinearSolveFailures;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESSetMaxLinearSolveFailures"
/*@
   SNESSetMaxLinearSolveFailures - the number of failed linear solve attempts
   allowed before SNES returns with a diverged reason of SNES_DIVERGED_LINEAR_SOLVE

   Collective on SNES

   Input Parameters:
+  snes     - SNES context
-  maxFails - maximum allowed linear solve failures

   Level: intermediate

   Notes: By default this is 0; that is SNES returns on the first failed linear solve

.keywords: SNES, nonlinear, set, maximum, unsuccessful, steps

.seealso: SNESGetLinearSolveFailures(), SNESGetMaxLinearSolveFailures(), SNESGetLinearSolveIterations()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESSetMaxLinearSolveFailures(SNES snes, PetscInt maxFails)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  snes->maxLinearSolveFailures = maxFails;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESGetMaxLinearSolveFailures"
/*@
   SNESGetMaxLinearSolveFailures - gets the maximum number of linear solve failures that
     are allowed before SNES terminates

   Not Collective

   Input Parameter:
.  snes     - SNES context

   Output Parameter:
.  maxFails - maximum of unsuccessful solves allowed

   Level: intermediate

   Notes: By default this is 1; that is SNES returns on the first failed linear solve

.keywords: SNES, nonlinear, get, maximum, unsuccessful, steps

.seealso: SNESGetLinearSolveFailures(), SNESGetLinearSolveIterations(), SNESSetMaxLinearSolveFailures(),
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESGetMaxLinearSolveFailures(SNES snes, PetscInt *maxFails)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  PetscValidIntPointer(maxFails,2);
  *maxFails = snes->maxLinearSolveFailures;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESGetLinearSolveIterations"
/*@
   SNESGetLinearSolveIterations - Gets the total number of linear iterations
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

.seealso:  SNESGetIterationNumber(), SNESGetFunctionNorm(), SNESGetLinearSolveFailures(), SNESGetMaxLinearSolveFailures()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESGetLinearSolveIterations(SNES snes,PetscInt* lits)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  PetscValidIntPointer(lits,2);
  *lits = snes->linear_its;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESGetKSP"
/*@
   SNESGetKSP - Returns the KSP context for a SNES solver.

   Not Collective, but if SNES object is parallel, then KSP object is parallel

   Input Parameter:
.  snes - the SNES context

   Output Parameter:
.  ksp - the KSP context

   Notes:
   The user can then directly manipulate the KSP context to set various
   options, etc.  Likewise, the user can then extract and manipulate the 
   PC contexts as well.

   Level: beginner

.keywords: SNES, nonlinear, get, KSP, context

.seealso: KSPGetPC(), SNESCreate(), KSPCreate(), SNESSetKSP()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESGetKSP(SNES snes,KSP *ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  PetscValidPointer(ksp,2);

  if (!snes->ksp) {
    ierr = KSPCreate(((PetscObject)snes)->comm,&snes->ksp);CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)snes->ksp,(PetscObject)snes,1);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(snes,snes->ksp);CHKERRQ(ierr);
  }
  *ksp = snes->ksp;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESSetKSP"
/*@
   SNESSetKSP - Sets a KSP context for the SNES object to use

   Not Collective, but the SNES and KSP objects must live on the same MPI_Comm

   Input Parameters:
+  snes - the SNES context
-  ksp - the KSP context

   Notes:
   The SNES object already has its KSP object, you can obtain with SNESGetKSP()
   so this routine is rarely needed.

   The KSP object that is already in the SNES object has its reference count
   decreased by one.

   Level: developer

.keywords: SNES, nonlinear, get, KSP, context

.seealso: KSPGetPC(), SNESCreate(), KSPCreate(), SNESSetKSP()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESSetKSP(SNES snes,KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  PetscValidHeaderSpecific(ksp,KSP_COOKIE,2);
  PetscCheckSameComm(snes,1,ksp,2);
  ierr = PetscObjectReference((PetscObject)ksp);CHKERRQ(ierr);
  if (snes->ksp) {ierr = PetscObjectDereference((PetscObject)snes->ksp);CHKERRQ(ierr);}
  snes->ksp = ksp;
  PetscFunctionReturn(0);
}

#if 0
#undef __FUNCT__  
#define __FUNCT__ "SNESPublish_Petsc"
static PetscErrorCode SNESPublish_Petsc(PetscObject obj)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
#endif

/* -----------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "SNESCreate"
/*@
   SNESCreate - Creates a nonlinear solver context.

   Collective on MPI_Comm

   Input Parameters:
.  comm - MPI communicator

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

.seealso: SNESSolve(), SNESDestroy(), SNES, SNESSetLagPreconditioner()

@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESCreate(MPI_Comm comm,SNES *outsnes)
{
  PetscErrorCode      ierr;
  SNES                snes;
  SNESKSPEW           *kctx;

  PetscFunctionBegin;
  PetscValidPointer(outsnes,2);
  *outsnes = PETSC_NULL;
#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = SNESInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif

  ierr = PetscHeaderCreate(snes,_p_SNES,struct _SNESOps,SNES_COOKIE,0,"SNES",comm,SNESDestroy,SNESView);CHKERRQ(ierr);

  snes->ops->converged    = SNESDefaultConverged;
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
  snes->lagjacobian       = 1;
  snes->lagpreconditioner = 1;
  snes->numbermonitors    = 0;
  snes->data              = 0;
  snes->setupcalled       = PETSC_FALSE;
  snes->ksp_ewconv        = PETSC_FALSE;
  snes->vwork             = 0;
  snes->nwork             = 0;
  snes->conv_hist_len     = 0;
  snes->conv_hist_max     = 0;
  snes->conv_hist         = PETSC_NULL;
  snes->conv_hist_its     = PETSC_NULL;
  snes->conv_hist_reset   = PETSC_TRUE;
  snes->reason            = SNES_CONVERGED_ITERATING;

  snes->numLinearSolveFailures = 0;
  snes->maxLinearSolveFailures = 1;

  /* Create context to compute Eisenstat-Walker relative tolerance for KSP */
  ierr = PetscNewLog(snes,SNESKSPEW,&kctx);CHKERRQ(ierr);
  snes->kspconvctx  = (void*)kctx;
  kctx->version     = 2;
  kctx->rtol_0      = .3; /* Eisenstat and Walker suggest rtol_0=.5, but 
                             this was too large for some test cases */
  kctx->rtol_last   = 0.0;
  kctx->rtol_max    = .9;
  kctx->gamma       = 1.0;
  kctx->alpha       = .5*(1.0 + sqrt(5.0));
  kctx->alpha2      = kctx->alpha;
  kctx->threshold   = .1;
  kctx->lresid_last = 0.0;
  kctx->norm_last   = 0.0;

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
.  r - vector to store function value
.  func - function evaluation routine
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
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  PetscValidHeaderSpecific(r,VEC_COOKIE,2);
  PetscCheckSameComm(snes,1,r,2);
  ierr = PetscObjectReference((PetscObject)r);CHKERRQ(ierr); 
  if (snes->vec_func) { ierr = VecDestroy(snes->vec_func);CHKERRQ(ierr); }
  snes->ops->computefunction = func; 
  snes->vec_func             = r;
  snes->funP                 = ctx;
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "SNESGetRhs"
/*@C
   SNESGetRhs - Gets the vector for solving F(x) = rhs. If rhs is not set
   it assumes a zero right hand side.

   Collective on SNES

   Input Parameter:
.  snes - the SNES context

   Output Parameter:
.  rhs - the right hand side vector or PETSC_NULL if the right hand side vector is null

   Level: intermediate

.keywords: SNES, nonlinear, get, function, right hand side

.seealso: SNESGetSolution(), SNESGetFunction(), SNESComputeFunction(), SNESSetJacobian(), SNESSetFunction()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESGetRhs(SNES snes,Vec *rhs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  PetscValidPointer(rhs,2);
  *rhs = snes->vec_rhs;
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
  if (snes->ops->computefunction) {
    PetscStackPush("SNES user function");
    CHKMEMQ;
    ierr = (*snes->ops->computefunction)(snes,x,y,snes->funP);
    CHKMEMQ;
    PetscStackPop;
    if (PetscExceptionValue(ierr)) {
      PetscErrorCode pierr = PetscLogEventEnd(SNES_FunctionEval,snes,x,y,0);CHKERRQ(pierr);
    }
    CHKERRQ(ierr);
  } else if (snes->vec_rhs) {
    ierr = MatMult(snes->jacobian, x, y);CHKERRQ(ierr);
  } else {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "Must call SNESSetFunction() before SNESComputeFunction(), likely called from SNESSolve().");
  }
  if (snes->vec_rhs) {
    ierr = VecAXPY(y,-1.0,snes->vec_rhs);CHKERRQ(ierr);
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
-  flag - flag indicating matrix structure (one of, SAME_NONZERO_PATTERN,DIFFERENT_NONZERO_PATTERN,SAME_PRECONDITIONER)

  Options Database Keys: 
+    -snes_lag_preconditioner <lag>
-    -snes_lag_jacobian <lag>

   Notes: 
   Most users should not need to explicitly call this routine, as it 
   is used internally within the nonlinear solvers. 

   See KSPSetOperators() for important information about setting the
   flag parameter.

   Level: developer

.keywords: SNES, compute, Jacobian, matrix

.seealso:  SNESSetJacobian(), KSPSetOperators(), MatStructure, SNESSetLagPreconditioner(), SNESSetLagJacobian()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESComputeJacobian(SNES snes,Vec X,Mat *A,Mat *B,MatStructure *flg)
{
  PetscErrorCode ierr;
  PetscTruth     flag;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  PetscValidHeaderSpecific(X,VEC_COOKIE,2);
  PetscValidPointer(flg,5);
  PetscCheckSameComm(snes,1,X,2);
  if (!snes->ops->computejacobian) PetscFunctionReturn(0);

  /* make sure that MatAssemblyBegin/End() is called on A matrix if it is matrix free */

  if (snes->lagjacobian == -2) {
    snes->lagjacobian = -1;
    ierr = PetscInfo(snes,"Recomputing Jacobian/preconditioner because lag is -2 (means compute Jacobian, but then never again) \n");CHKERRQ(ierr);
  } else if (snes->lagjacobian == -1) {
    *flg = SAME_PRECONDITIONER;
    ierr = PetscInfo(snes,"Reusing Jacobian/preconditioner because lag is -1\n");CHKERRQ(ierr);
    ierr = PetscTypeCompare((PetscObject)*A,MATMFFD,&flag);CHKERRQ(ierr);
    if (flag) {
      ierr = MatAssemblyBegin(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
  } else if (snes->lagjacobian > 1 && snes->iter % snes->lagjacobian) {
    *flg = SAME_PRECONDITIONER;
    ierr = PetscInfo2(snes,"Reusing Jacobian/preconditioner because lag is %D and SNES iteration is %D\n",snes->lagjacobian,snes->iter);CHKERRQ(ierr);
    ierr = PetscTypeCompare((PetscObject)*A,MATMFFD,&flag);CHKERRQ(ierr);
    if (flag) {
      ierr = MatAssemblyBegin(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
  }

  *flg = DIFFERENT_NONZERO_PATTERN;
  ierr = PetscLogEventBegin(SNES_JacobianEval,snes,X,*A,*B);CHKERRQ(ierr);
  PetscStackPush("SNES user Jacobian function");
  CHKMEMQ;
  ierr = (*snes->ops->computejacobian)(snes,X,A,B,flg,snes->jacP);CHKERRQ(ierr);
  CHKMEMQ;
  PetscStackPop;
  ierr = PetscLogEventEnd(SNES_JacobianEval,snes,X,*A,*B);CHKERRQ(ierr);

  if (snes->lagpreconditioner == -2) {
    ierr = PetscInfo(snes,"Rebuilding preconditioner exactly once since lag is -2\n");CHKERRQ(ierr);
    snes->lagpreconditioner = -1;
  } else if (snes->lagpreconditioner == -1) {
    *flg = SAME_PRECONDITIONER;
    ierr = PetscInfo(snes,"Reusing preconditioner because lag is -1\n");CHKERRQ(ierr);
  } else if (snes->lagpreconditioner > 1 && snes->iter % snes->lagpreconditioner) {
    *flg = SAME_PRECONDITIONER;
    ierr = PetscInfo2(snes,"Reusing preconditioner because lag is %D and SNES iteration is %D\n",snes->lagpreconditioner,snes->iter);CHKERRQ(ierr);
  }

  /* make sure user returned a correct Jacobian and preconditioner */
  /* PetscValidHeaderSpecific(*A,MAT_COOKIE,3);
    PetscValidHeaderSpecific(*B,MAT_COOKIE,4);   */
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
   structure (same as flag in KSPSetOperators()), one of SAME_NONZERO_PATTERN,DIFFERENT_NONZERO_PATTERN,SAME_PRECONDITIONER
-  ctx - [optional] user-defined Jacobian context

   Notes: 
   See KSPSetOperators() for important information about setting the flag
   output parameter in the routine func().  Be sure to read this information!

   The routine func() takes Mat * as the matrix arguments rather than Mat.  
   This allows the Jacobian evaluation routine to replace A and/or B with a 
   completely new new matrix structure (not just different matrix elements)
   when appropriate, for instance, if the nonzero structure is changing
   throughout the global iterations.

   If the A matrix and B matrix are different you must call MatAssemblyBegin/End() on
   each matrix.

   Level: beginner

.keywords: SNES, nonlinear, set, Jacobian, matrix

.seealso: KSPSetOperators(), SNESSetFunction(), MatMFFDComputeJacobian(), SNESDefaultComputeJacobianColor(), MatStructure
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESSetJacobian(SNES snes,Mat A,Mat B,PetscErrorCode (*func)(SNES,Vec,Mat*,Mat*,MatStructure*,void*),void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  if (A) PetscValidHeaderSpecific(A,MAT_COOKIE,2);
  if (B) PetscValidHeaderSpecific(B,MAT_COOKIE,3);
  if (A) PetscCheckSameComm(snes,1,A,2);
  if (B) PetscCheckSameComm(snes,1,B,3);
  if (func) snes->ops->computejacobian = func;
  if (ctx)  snes->jacP                 = ctx;
  if (A) {
    ierr = PetscObjectReference((PetscObject)A);CHKERRQ(ierr);
    if (snes->jacobian) {ierr = MatDestroy(snes->jacobian);CHKERRQ(ierr);}
    snes->jacobian = A;
  }
  if (B) {
    ierr = PetscObjectReference((PetscObject)B);CHKERRQ(ierr);
    if (snes->jacobian_pre) {ierr = MatDestroy(snes->jacobian_pre);CHKERRQ(ierr);}
    snes->jacobian_pre = B;
  }
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
  if (func) *func = snes->ops->computejacobian;
  if (ctx)  *ctx  = snes->jacP;
  PetscFunctionReturn(0);
}

/* ----- Routines to initialize and destroy a nonlinear solver ---- */

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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  if (snes->setupcalled) PetscFunctionReturn(0);

  if (!((PetscObject)snes)->type_name) {
    ierr = SNESSetType(snes,SNESLS);CHKERRQ(ierr);
  }

  if (!snes->vec_func && !snes->vec_rhs) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Must call SNESSetFunction() first");
  }
  if (!snes->ops->computefunction && !snes->vec_rhs) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Must call SNESSetFunction() first");
  }
  if (snes->vec_func == snes->vec_sol) {
    SETERRQ(PETSC_ERR_ARG_IDN,"Solution vector cannot be function vector");
  }
  
  if (!snes->ksp) {ierr = SNESGetKSP(snes, &snes->ksp);CHKERRQ(ierr);}
  
  if (snes->ops->setup) {
    ierr = (*snes->ops->setup)(snes);CHKERRQ(ierr);
  }
  snes->setupcalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESDestroy"
/*@
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  if (--((PetscObject)snes)->refct > 0) PetscFunctionReturn(0);

  /* if memory was published with AMS then destroy it */
  ierr = PetscObjectDepublish(snes);CHKERRQ(ierr);
  if (snes->ops->destroy) {ierr = (*(snes)->ops->destroy)(snes);CHKERRQ(ierr);}
  
  if (snes->vec_rhs) {ierr = VecDestroy(snes->vec_rhs);CHKERRQ(ierr);}
  if (snes->vec_sol) {ierr = VecDestroy(snes->vec_sol);CHKERRQ(ierr);}
  if (snes->vec_func) {ierr = VecDestroy(snes->vec_func);CHKERRQ(ierr);}
  if (snes->jacobian) {ierr = MatDestroy(snes->jacobian);CHKERRQ(ierr);}
  if (snes->jacobian_pre) {ierr = MatDestroy(snes->jacobian_pre);CHKERRQ(ierr);}
  if (snes->ksp) {ierr = KSPDestroy(snes->ksp);CHKERRQ(ierr);}
  ierr = PetscFree(snes->kspconvctx);CHKERRQ(ierr);
  if (snes->vwork) {ierr = VecDestroyVecs(snes->vwork,snes->nvwork);CHKERRQ(ierr);}
  ierr = SNESMonitorCancel(snes);CHKERRQ(ierr);
  if (snes->ops->convergeddestroy) {ierr = (*snes->ops->convergeddestroy)(snes->cnvP);CHKERRQ(ierr);}
  ierr = PetscHeaderDestroy(snes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ----------- Routines to set solver parameters ---------- */

#undef __FUNCT__  
#define __FUNCT__ "SNESSetLagPreconditioner"
/*@
   SNESSetLagPreconditioner - Determines when the preconditioner is rebuilt in the nonlinear solve.

   Collective on SNES

   Input Parameters:
+  snes - the SNES context
-  lag - -1 indicates NEVER rebuild, 1 means rebuild every time the Jacobian is computed within a single nonlinear solve, 2 means every second time
         the Jacobian is built etc. -2 indicates rebuild preconditioner at next chance but then never rebuild after that

   Options Database Keys: 
.    -snes_lag_preconditioner <lag>

   Notes:
   The default is 1
   The preconditioner is ALWAYS built in the first iteration of a nonlinear solve unless lag is -1
   If  -1 is used before the very first nonlinear solve the preconditioner is still built because there is no previous preconditioner to use

   Level: intermediate

.keywords: SNES, nonlinear, set, convergence, tolerances

.seealso: SNESSetTrustRegionTolerance(), SNESGetLagPreconditioner(), SNESSetLagJacobian(), SNESGetLagJacobian()

@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESSetLagPreconditioner(SNES snes,PetscInt lag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  if (lag < -2) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Lag must be -2, -1, 1 or greater");
  if (!lag) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Lag cannot be 0");
  snes->lagpreconditioner = lag;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESGetLagPreconditioner"
/*@
   SNESGetLagPreconditioner - Indicates how often the preconditioner is rebuilt

   Collective on SNES

   Input Parameter:
.  snes - the SNES context
 
   Output Parameter:
.   lag - -1 indicates NEVER rebuild, 1 means rebuild every time the Jacobian is computed within a single nonlinear solve, 2 means every second time
         the Jacobian is built etc. -2 indicates rebuild preconditioner at next chance but then never rebuild after that

   Options Database Keys: 
.    -snes_lag_preconditioner <lag>

   Notes:
   The default is 1
   The preconditioner is ALWAYS built in the first iteration of a nonlinear solve unless lag is -1

   Level: intermediate

.keywords: SNES, nonlinear, set, convergence, tolerances

.seealso: SNESSetTrustRegionTolerance(), SNESSetLagPreconditioner()

@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESGetLagPreconditioner(SNES snes,PetscInt *lag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  *lag = snes->lagpreconditioner;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESSetLagJacobian"
/*@
   SNESSetLagJacobian - Determines when the Jacobian is rebuilt in the nonlinear solve. See SNESSetLagPreconditioner() for determining how
     often the preconditioner is rebuilt.

   Collective on SNES

   Input Parameters:
+  snes - the SNES context
-  lag - -1 indicates NEVER rebuild, 1 means rebuild every time the Jacobian is computed within a single nonlinear solve, 2 means every second time
         the Jacobian is built etc. -2 means rebuild at next chance but then never again

   Options Database Keys: 
.    -snes_lag_jacobian <lag>

   Notes:
   The default is 1
   The Jacobian is ALWAYS built in the first iteration of a nonlinear solve unless lag is -1
   If  -1 is used before the very first nonlinear solve the CODE WILL FAIL! because no Jacobian is used, use -2 to indicate you want it recomputed
   at the next Newton step but never again (unless it is reset to another value)

   Level: intermediate

.keywords: SNES, nonlinear, set, convergence, tolerances

.seealso: SNESSetTrustRegionTolerance(), SNESGetLagPreconditioner(), SNESSetLagPreconditioner(), SNESGetLagJacobian()

@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESSetLagJacobian(SNES snes,PetscInt lag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  if (lag < -2) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Lag must be -2, -1, 1 or greater");
  if (!lag) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Lag cannot be 0");
  snes->lagjacobian = lag;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESGetLagJacobian"
/*@
   SNESGetLagJacobian - Indicates how often the Jacobian is rebuilt. See SNESGetLagPreconditioner() to determine when the preconditioner is rebuilt

   Collective on SNES

   Input Parameter:
.  snes - the SNES context
 
   Output Parameter:
.   lag - -1 indicates NEVER rebuild, 1 means rebuild every time the Jacobian is computed within a single nonlinear solve, 2 means every second time
         the Jacobian is built etc.

   Options Database Keys: 
.    -snes_lag_jacobian <lag>

   Notes:
   The default is 1
   The jacobian is ALWAYS built in the first iteration of a nonlinear solve unless lag is -1

   Level: intermediate

.keywords: SNES, nonlinear, set, convergence, tolerances

.seealso: SNESSetTrustRegionTolerance(), SNESSetLagJacobian(), SNESSetLagPreconditioner(), SNESGetLagPreconditioner()

@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESGetLagJacobian(SNES snes,PetscInt *lag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  *lag = snes->lagjacobian;
  PetscFunctionReturn(0);
}

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
.  atol - absolute convergence tolerance
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
PetscErrorCode PETSCSNES_DLLEXPORT SNESGetTolerances(SNES snes,PetscReal *atol,PetscReal *rtol,PetscReal *stol,PetscInt *maxit,PetscInt *maxf)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  if (atol)  *atol  = snes->abstol;
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
#define __FUNCT__ "SNESMonitorLG"
PetscErrorCode PETSCSNES_DLLEXPORT SNESMonitorLG(SNES snes,PetscInt it,PetscReal norm,void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  ierr = KSPMonitorLG((KSP)snes,it,norm,ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESMonitorLGCreate"
PetscErrorCode PETSCSNES_DLLEXPORT SNESMonitorLGCreate(const char host[],const char label[],int x,int y,int m,int n,PetscDrawLG *draw)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPMonitorLGCreate(host,label,x,y,m,n,draw);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESMonitorLGDestroy"
PetscErrorCode PETSCSNES_DLLEXPORT SNESMonitorLGDestroy(PetscDrawLG draw)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPMonitorLGDestroy(draw);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

extern PetscErrorCode PETSCSNES_DLLEXPORT SNESMonitorRange_Private(SNES,PetscInt,PetscReal*);
#undef __FUNCT__  
#define __FUNCT__ "SNESMonitorLGRange"
PetscErrorCode PETSCSNES_DLLEXPORT SNESMonitorLGRange(SNES snes,PetscInt n,PetscReal rnorm,void *monctx)
{
  PetscDrawLG      lg;
  PetscErrorCode   ierr;
  PetscReal        x,y,per;
  PetscViewer      v = (PetscViewer)monctx;
  static PetscReal prev; /* should be in the context */
  PetscDraw        draw;
  PetscFunctionBegin;
  if (!monctx) {
    MPI_Comm    comm;

    ierr   = PetscObjectGetComm((PetscObject)snes,&comm);CHKERRQ(ierr);
    v      = PETSC_VIEWER_DRAW_(comm);
  }
  ierr   = PetscViewerDrawGetDrawLG(v,0,&lg);CHKERRQ(ierr);
  if (!n) {ierr = PetscDrawLGReset(lg);CHKERRQ(ierr);}
  ierr   = PetscDrawLGGetDraw(lg,&draw);CHKERRQ(ierr);
  ierr   = PetscDrawSetTitle(draw,"Residual norm");CHKERRQ(ierr);
  x = (PetscReal) n;
  if (rnorm > 0.0) y = log10(rnorm); else y = -15.0;
  ierr = PetscDrawLGAddPoint(lg,&x,&y);CHKERRQ(ierr);
  if (n < 20 || !(n % 5)) {
    ierr = PetscDrawLGDraw(lg);CHKERRQ(ierr);
  }

  ierr = PetscViewerDrawGetDrawLG(v,1,&lg);CHKERRQ(ierr);
  if (!n) {ierr = PetscDrawLGReset(lg);CHKERRQ(ierr);}
  ierr = PetscDrawLGGetDraw(lg,&draw);CHKERRQ(ierr);
  ierr = PetscDrawSetTitle(draw,"% elemts > .2*max elemt");CHKERRQ(ierr);
  ierr =  SNESMonitorRange_Private(snes,n,&per);CHKERRQ(ierr);
  x = (PetscReal) n;
  y = 100.0*per;
  ierr = PetscDrawLGAddPoint(lg,&x,&y);CHKERRQ(ierr);
  if (n < 20 || !(n % 5)) {
    ierr = PetscDrawLGDraw(lg);CHKERRQ(ierr);
  }

  ierr = PetscViewerDrawGetDrawLG(v,2,&lg);CHKERRQ(ierr);
  if (!n) {prev = rnorm;ierr = PetscDrawLGReset(lg);CHKERRQ(ierr);}
  ierr = PetscDrawLGGetDraw(lg,&draw);CHKERRQ(ierr);
  ierr = PetscDrawSetTitle(draw,"(norm -oldnorm)/oldnorm");CHKERRQ(ierr);
  x = (PetscReal) n;
  y = (prev - rnorm)/prev;
  ierr = PetscDrawLGAddPoint(lg,&x,&y);CHKERRQ(ierr);
  if (n < 20 || !(n % 5)) {
    ierr = PetscDrawLGDraw(lg);CHKERRQ(ierr);
  }

  ierr = PetscViewerDrawGetDrawLG(v,3,&lg);CHKERRQ(ierr);
  if (!n) {ierr = PetscDrawLGReset(lg);CHKERRQ(ierr);}
  ierr = PetscDrawLGGetDraw(lg,&draw);CHKERRQ(ierr);
  ierr = PetscDrawSetTitle(draw,"(norm -oldnorm)/oldnorm*(% > .2 max)");CHKERRQ(ierr);
  x = (PetscReal) n;
  y = (prev - rnorm)/(prev*per);
  if (n > 2) { /*skip initial crazy value */
    ierr = PetscDrawLGAddPoint(lg,&x,&y);CHKERRQ(ierr);
  }
  if (n < 20 || !(n % 5)) {
    ierr = PetscDrawLGDraw(lg);CHKERRQ(ierr);
  }
  prev = rnorm;
  PetscFunctionReturn(0);
} 

#undef __FUNCT__  
#define __FUNCT__ "SNESMonitorLGRangeCreate"
PetscErrorCode PETSCSNES_DLLEXPORT SNESMonitorLGRangeCreate(const char host[],const char label[],int x,int y,int m,int n,PetscDrawLG *draw)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPMonitorLGCreate(host,label,x,y,m,n,draw);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESMonitorLGRangeDestroy"
PetscErrorCode PETSCSNES_DLLEXPORT SNESMonitorLGRangeDestroy(PetscDrawLG draw)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPMonitorLGDestroy(draw);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------ Routines to set performance monitoring options ----------- */

#undef __FUNCT__  
#define __FUNCT__ "SNESMonitorSet"
/*@C
   SNESMonitorSet - Sets an ADDITIONAL function that is to be used at every
   iteration of the nonlinear solver to display the iteration's 
   progress.   

   Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  func - monitoring routine
.  mctx - [optional] user-defined context for private data for the 
          monitor routine (use PETSC_NULL if no context is desired)
-  monitordestroy - [optional] routine that frees monitor context
          (may be PETSC_NULL)

   Calling sequence of func:
$     int func(SNES snes,PetscInt its, PetscReal norm,void *mctx)

+    snes - the SNES context
.    its - iteration number
.    norm - 2-norm function value (may be estimated)
-    mctx - [optional] monitoring context

   Options Database Keys:
+    -snes_monitor        - sets SNESMonitorDefault()
.    -snes_monitor_draw    - sets line graph monitor,
                            uses SNESMonitorLGCreate()
_    -snes_monitor_cancel - cancels all monitors that have
                            been hardwired into a code by 
                            calls to SNESMonitorSet(), but
                            does not cancel those set via
                            the options database.

   Notes: 
   Several different monitoring routines may be set by calling
   SNESMonitorSet() multiple times; all will be called in the 
   order in which they were set.

   Fortran notes: Only a single monitor function can be set for each SNES object

   Level: intermediate

.keywords: SNES, nonlinear, set, monitor

.seealso: SNESMonitorDefault(), SNESMonitorCancel()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESMonitorSet(SNES snes,PetscErrorCode (*monitor)(SNES,PetscInt,PetscReal,void*),void *mctx,PetscErrorCode (*monitordestroy)(void*))
{
  PetscInt i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  if (snes->numbermonitors >= MAXSNESMONITORS) {
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Too many monitors set");
  }
  for (i=0; i<snes->numbermonitors;i++) {
    if (monitor == snes->monitor[i] && monitordestroy == snes->monitordestroy[i] && mctx == snes->monitorcontext[i]) PetscFunctionReturn(0);

    /* check if both default monitors that share common ASCII viewer */
    if (monitor == snes->monitor[i] && monitor == SNESMonitorDefault) {
      if (mctx && snes->monitorcontext[i]) {
        PetscErrorCode          ierr;
        PetscViewerASCIIMonitor viewer1 = (PetscViewerASCIIMonitor) mctx;
        PetscViewerASCIIMonitor viewer2 = (PetscViewerASCIIMonitor) snes->monitorcontext[i];
        if (viewer1->viewer == viewer2->viewer) {
          ierr = (*monitordestroy)(mctx);CHKERRQ(ierr);
          PetscFunctionReturn(0);
        }
      }
    }
  }
  snes->monitor[snes->numbermonitors]           = monitor;
  snes->monitordestroy[snes->numbermonitors]    = monitordestroy;
  snes->monitorcontext[snes->numbermonitors++]  = (void*)mctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESMonitorCancel"
/*@C
   SNESMonitorCancel - Clears all the monitor functions for a SNES object.

   Collective on SNES

   Input Parameters:
.  snes - the SNES context

   Options Database Key:
.  -snes_monitor_cancel - cancels all monitors that have been hardwired
    into a code by calls to SNESMonitorSet(), but does not cancel those 
    set via the options database

   Notes: 
   There is no way to clear one specific monitor from a SNES object.

   Level: intermediate

.keywords: SNES, nonlinear, set, monitor

.seealso: SNESMonitorDefault(), SNESMonitorSet()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESMonitorCancel(SNES snes)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  for (i=0; i<snes->numbermonitors; i++) {
    if (snes->monitordestroy[i]) {
      ierr = (*snes->monitordestroy[i])(snes->monitorcontext[i]);CHKERRQ(ierr);
    }
  }
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
.  cctx - [optional] context for private data for the convergence routine  (may be PETSC_NULL)
-  destroy - [optional] destructor for the context (may be PETSC_NULL; PETSC_NULL_FUNCTION in Fortran)

   Calling sequence of func:
$     PetscErrorCode func (SNES snes,PetscInt it,PetscReal xnorm,PetscReal gnorm,PetscReal f,SNESConvergedReason *reason,void *cctx)

+    snes - the SNES context
.    it - current iteration (0 is the first and is before any Newton step)
.    cctx - [optional] convergence context
.    reason - reason for convergence/divergence
.    xnorm - 2-norm of current iterate
.    gnorm - 2-norm of current step
-    f - 2-norm of function

   Level: advanced

.keywords: SNES, nonlinear, set, convergence, test

.seealso: SNESDefaultConverged(), SNESSkipConverged()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESSetConvergenceTest(SNES snes,PetscErrorCode (*func)(SNES,PetscInt,PetscReal,PetscReal,PetscReal,SNESConvergedReason*,void*),void *cctx,PetscErrorCode (*destroy)(void*))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  if (!func) func = SNESSkipConverged;
  if (snes->ops->convergeddestroy) {
    ierr = (*snes->ops->convergeddestroy)(snes->cnvP);CHKERRQ(ierr);
  }
  snes->ops->converged        = func;
  snes->ops->convergeddestroy = destroy;
  snes->cnvP                  = cctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESGetConvergedReason"
/*@
   SNESGetConvergedReason - Gets the reason the SNES iteration was stopped.

   Not Collective

   Input Parameter:
.  snes - the SNES context

   Output Parameter:
.  reason - negative value indicates diverged, positive value converged, see SNESConvergedReason or the 
            manual pages for the individual convergence tests for complete lists

   Level: intermediate

   Notes: Can only be called after the call the SNESSolve() is complete.

.keywords: SNES, nonlinear, set, convergence, test

.seealso: SNESSetConvergenceTest(), SNESConvergedReason
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
.  a   - array to hold history, this array will contain the function norms computed at each step
.  its - integer array holds the number of linear iterations for each solve.
.  na  - size of a and its
-  reset - PETSC_TRUE indicates each new nonlinear solve resets the history counter to zero,
           else it continues storing new values for new nonlinear solves after the old ones

   This routine is useful, e.g., when running a code for purposes
   of accurate performance monitoring, when no I/O should be done
   during the section of code that is being timed.

   Level: intermediate

.keywords: SNES, set, convergence, history

.seealso: SNESGetConvergenceHistory()

@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESSetConvergenceHistory(SNES snes,PetscReal a[],PetscInt its[],PetscInt na,PetscTruth reset)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  if (na)  PetscValidScalarPointer(a,2);
  if (its) PetscValidIntPointer(its,3);
  snes->conv_hist       = a;
  snes->conv_hist_its   = its;
  snes->conv_hist_max   = na;
  snes->conv_hist_len   = 0;
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
  if (na)  *na  = snes->conv_hist_len;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESSetUpdate"
/*@C
  SNESSetUpdate - Sets the general-purpose update function called
  at the beginning o every iteration of the nonlinear solve. Specifically
  it is called just before the Jacobian is "evaluated".

  Collective on SNES

  Input Parameters:
. snes - The nonlinear solver context
. func - The function

  Calling sequence of func:
. func (SNES snes, PetscInt step);

. step - The current step of the iteration

  Level: advanced

  Note: This is NOT what one uses to update the ghost points before a function evaluation, that should be done at the beginning of your FormFunction()
        This is not used by most users.

.keywords: SNES, update

.seealso SNESDefaultUpdate(), SNESSetJacobian(), SNESSolve()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESSetUpdate(SNES snes, PetscErrorCode (*func)(SNES, PetscInt))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_COOKIE,1);
  snes->ops->update = func;
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
.seealso SNESSetUpdate(), SNESDefaultRhsBC(), SNESDefaultShortolutionBC()
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
/*@C
   SNESSolve - Solves a nonlinear system F(x) = b.
   Call SNESSolve() after calling SNESCreate() and optional routines of the form SNESSetXXX().

   Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  b - the constant part of the equation, or PETSC_NULL to use zero.
-  x - the solution vector.

   Notes:
   The user should initialize the vector,x, with the initial guess
   for the nonlinear solve prior to calling SNESSolve.  In particular,
   to employ an initial guess of zero, the user should explicitly set
   this vector to zero by calling VecSet().

   Level: beginner

.keywords: SNES, nonlinear, solve

.seealso: SNESCreate(), SNESDestroy(), SNESSetFunction(), SNESSetJacobian()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESSolve(SNES snes,Vec b,Vec x)
{
  PetscErrorCode ierr;
  PetscTruth     flg;
  char           filename[PETSC_MAX_PATH_LEN];
  PetscViewer    viewer;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  PetscValidHeaderSpecific(x,VEC_COOKIE,3);
  PetscCheckSameComm(snes,1,x,3);
  if (b) PetscValidHeaderSpecific(b,VEC_COOKIE,2);
  if (b) PetscCheckSameComm(snes,1,b,2);

  /* set solution vector */
  ierr = PetscObjectReference((PetscObject)x);CHKERRQ(ierr);
  if (snes->vec_sol) { ierr = VecDestroy(snes->vec_sol);CHKERRQ(ierr); }
  snes->vec_sol = x;
  /* set afine vector if provided */
  if (b) { ierr = PetscObjectReference((PetscObject)b);CHKERRQ(ierr); }
  if (snes->vec_rhs) { ierr = VecDestroy(snes->vec_rhs);CHKERRQ(ierr); }
  snes->vec_rhs = b;
  
  if (!snes->vec_func && snes->vec_rhs) {
    ierr = VecDuplicate(b, &snes->vec_func);CHKERRQ(ierr);
  }

  ierr = SNESSetUp(snes);CHKERRQ(ierr);

  if (snes->conv_hist_reset) snes->conv_hist_len = 0;
  snes->nfuncs = 0; snes->linear_its = 0; snes->numFailures = 0;

  ierr = PetscLogEventBegin(SNES_Solve,snes,0,0,0);CHKERRQ(ierr);
  ierr = (*snes->ops->solve)(snes);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(SNES_Solve,snes,0,0,0);CHKERRQ(ierr);
  if (snes->domainerror){
    snes->reason      = SNES_DIVERGED_FUNCTION_DOMAIN;
    snes->domainerror = PETSC_FALSE;
  } 

  if (!snes->reason) {
    SETERRQ(PETSC_ERR_PLIB,"Internal error, solver returned without setting converged reason");
  }
  
  ierr = PetscOptionsGetString(((PetscObject)snes)->prefix,"-snes_view",filename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if (flg && !PetscPreLoadingOn) {
    ierr = PetscViewerASCIIOpen(((PetscObject)snes)->comm,filename,&viewer);CHKERRQ(ierr);
    ierr = SNESView(snes,viewer);CHKERRQ(ierr); 
    ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
  }

  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetTruth(((PetscObject)snes)->prefix,"-snes_test_local_min",&flg,PETSC_NULL);CHKERRQ(ierr);
  if (flg && !PetscPreLoadingOn) { ierr = SNESTestLocalMin(snes);CHKERRQ(ierr); }
  if (snes->printreason) {
    if (snes->reason > 0) {
      ierr = PetscPrintf(((PetscObject)snes)->comm,"Nonlinear solve converged due to %s\n",SNESConvergedReasons[snes->reason]);CHKERRQ(ierr);
    } else {
      ierr = PetscPrintf(((PetscObject)snes)->comm,"Nonlinear solve did not converge due to %s\n",SNESConvergedReasons[snes->reason]);CHKERRQ(ierr);
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
PetscErrorCode PETSCSNES_DLLEXPORT SNESSetType(SNES snes,const SNESType type)
{
  PetscErrorCode ierr,(*r)(SNES);
  PetscTruth     match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  PetscValidCharPointer(type,2);

  ierr = PetscTypeCompare((PetscObject)snes,type,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  ierr =  PetscFListFind(SNESList,((PetscObject)snes)->comm,type,(void (**)(void)) &r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PETSC_ERR_ARG_UNKNOWN_TYPE,"Unable to find requested SNES type %s",type);
  /* Destroy the previous private SNES context */
  if (snes->ops->destroy) { ierr = (*(snes)->ops->destroy)(snes);CHKERRQ(ierr); }
  /* Reinitialize function pointers in SNESOps structure */
  snes->ops->setup          = 0;
  snes->ops->solve          = 0;
  snes->ops->view           = 0;
  snes->ops->setfromoptions = 0;
  snes->ops->destroy        = 0;
  /* Call the SNESCreate_XXX routine for this particular Nonlinear solver */
  snes->setupcalled = PETSC_FALSE;
  ierr = (*r)(snes);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)snes,type);CHKERRQ(ierr);
  PetscFunctionReturn(0); 
}


/* --------------------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "SNESRegisterDestroy"
/*@
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
  ierr = PetscFListDestroy(&SNESList);CHKERRQ(ierr);
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
PetscErrorCode PETSCSNES_DLLEXPORT SNESGetType(SNES snes,const SNESType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  PetscValidPointer(type,2);
  *type = ((PetscObject)snes)->type_name;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESGetSolution"
/*@
   SNESGetSolution - Returns the vector where the approximate solution is
   stored.

   Not Collective, but Vec is parallel if SNES is parallel

   Input Parameter:
.  snes - the SNES context

   Output Parameter:
.  x - the solution

   Level: intermediate

.keywords: SNES, nonlinear, get, solution

.seealso:  SNESGetSolutionUpdate(), SNESGetFunction()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESGetSolution(SNES snes,Vec *x)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  PetscValidPointer(x,2);
  *x = snes->vec_sol;
  PetscFunctionReturn(0);
}  

#undef __FUNCT__  
#define __FUNCT__ "SNESGetSolutionUpdate"
/*@
   SNESGetSolutionUpdate - Returns the vector where the solution update is
   stored. 

   Not Collective, but Vec is parallel if SNES is parallel

   Input Parameter:
.  snes - the SNES context

   Output Parameter:
.  x - the solution update

   Level: advanced

.keywords: SNES, nonlinear, get, solution, update

.seealso: SNESGetSolution(), SNESGetFunction()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESGetSolutionUpdate(SNES snes,Vec *x)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  PetscValidPointer(x,2);
  *x = snes->vec_sol_update;
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
  if (r)    *r    = snes->vec_func;
  if (func) *func = snes->ops->computefunction;
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
  if (!snes->ksp) {ierr = SNESGetKSP(snes,&snes->ksp);CHKERRQ(ierr);}
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
  if (!snes->ksp) {ierr = SNESGetKSP(snes,&snes->ksp);CHKERRQ(ierr);}
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

   Notes: On the fortran side, the user should pass in a string 'prefix' of
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

#undef __FUNCT__  
#define __FUNCT__ "SNESKSPSetUseEW"
/*@
   SNESKSPSetUseEW - Sets SNES use Eisenstat-Walker method for
   computing relative tolerance for linear solvers within an inexact
   Newton method.

   Collective on SNES

   Input Parameters:
+  snes - SNES context
-  flag - PETSC_TRUE or PETSC_FALSE

    Options Database:
+  -snes_ksp_ew - use Eisenstat-Walker method for determining linear system convergence
.  -snes_ksp_ew_version ver - version of  Eisenstat-Walker method
.  -snes_ksp_ew_rtol0 <rtol0> - Sets rtol0
.  -snes_ksp_ew_rtolmax <rtolmax> - Sets rtolmax
.  -snes_ksp_ew_gamma <gamma> - Sets gamma
.  -snes_ksp_ew_alpha <alpha> - Sets alpha
.  -snes_ksp_ew_alpha2 <alpha2> - Sets alpha2 
-  -snes_ksp_ew_threshold <threshold> - Sets threshold

   Notes:
   Currently, the default is to use a constant relative tolerance for 
   the inner linear solvers.  Alternatively, one can use the 
   Eisenstat-Walker method, where the relative convergence tolerance 
   is reset at each Newton iteration according progress of the nonlinear 
   solver. 

   Level: advanced

   Reference:
   S. C. Eisenstat and H. F. Walker, "Choosing the forcing terms in an 
   inexact Newton method", SISC 17 (1), pp.16-32, 1996.

.keywords: SNES, KSP, Eisenstat, Walker, convergence, test, inexact, Newton

.seealso: SNESKSPGetUseEW(), SNESKSPGetParametersEW(), SNESKSPSetParametersEW()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESKSPSetUseEW(SNES snes,PetscTruth flag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  snes->ksp_ewconv = flag;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESKSPGetUseEW"
/*@
   SNESKSPGetUseEW - Gets if SNES is using Eisenstat-Walker method
   for computing relative tolerance for linear solvers within an
   inexact Newton method.

   Not Collective

   Input Parameter:
.  snes - SNES context

   Output Parameter:
.  flag - PETSC_TRUE or PETSC_FALSE

   Notes:
   Currently, the default is to use a constant relative tolerance for 
   the inner linear solvers.  Alternatively, one can use the 
   Eisenstat-Walker method, where the relative convergence tolerance 
   is reset at each Newton iteration according progress of the nonlinear 
   solver. 

   Level: advanced

   Reference:
   S. C. Eisenstat and H. F. Walker, "Choosing the forcing terms in an 
   inexact Newton method", SISC 17 (1), pp.16-32, 1996.

.keywords: SNES, KSP, Eisenstat, Walker, convergence, test, inexact, Newton

.seealso: SNESKSPSetUseEW(), SNESKSPGetParametersEW(), SNESKSPSetParametersEW()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESKSPGetUseEW(SNES snes, PetscTruth *flag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  PetscValidPointer(flag,2);
  *flag = snes->ksp_ewconv;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESKSPSetParametersEW"
/*@
   SNESKSPSetParametersEW - Sets parameters for Eisenstat-Walker
   convergence criteria for the linear solvers within an inexact
   Newton method.

   Collective on SNES
 
   Input Parameters:
+    snes - SNES context
.    version - version 1, 2 (default is 2) or 3
.    rtol_0 - initial relative tolerance (0 <= rtol_0 < 1)
.    rtol_max - maximum relative tolerance (0 <= rtol_max < 1)
.    gamma - multiplicative factor for version 2 rtol computation
             (0 <= gamma2 <= 1)
.    alpha - power for version 2 rtol computation (1 < alpha <= 2)
.    alpha2 - power for safeguard
-    threshold - threshold for imposing safeguard (0 < threshold < 1)

   Note:
   Version 3 was contributed by Luis Chacon, June 2006.

   Use PETSC_DEFAULT to retain the default for any of the parameters.

   Level: advanced

   Reference:
   S. C. Eisenstat and H. F. Walker, "Choosing the forcing terms in an 
   inexact Newton method", Utah State University Math. Stat. Dept. Res. 
   Report 6/94/75, June, 1994, to appear in SIAM J. Sci. Comput. 

.keywords: SNES, KSP, Eisenstat, Walker, set, parameters

.seealso: SNESKSPSetUseEW(), SNESKSPGetUseEW(), SNESKSPGetParametersEW()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESKSPSetParametersEW(SNES snes,PetscInt version,PetscReal rtol_0,PetscReal rtol_max,
							    PetscReal gamma,PetscReal alpha,PetscReal alpha2,PetscReal threshold)
{
  SNESKSPEW *kctx;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  kctx = (SNESKSPEW*)snes->kspconvctx;
  if (!kctx) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"No Eisenstat-Walker context existing");

  if (version != PETSC_DEFAULT)   kctx->version   = version;
  if (rtol_0 != PETSC_DEFAULT)    kctx->rtol_0    = rtol_0;
  if (rtol_max != PETSC_DEFAULT)  kctx->rtol_max  = rtol_max;
  if (gamma != PETSC_DEFAULT)     kctx->gamma     = gamma;
  if (alpha != PETSC_DEFAULT)     kctx->alpha     = alpha;
  if (alpha2 != PETSC_DEFAULT)    kctx->alpha2    = alpha2;
  if (threshold != PETSC_DEFAULT) kctx->threshold = threshold;
  
  if (kctx->version < 1 || kctx->version > 3) {
    SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Only versions 1, 2 and 3 are supported: %D",kctx->version);
  }
  if (kctx->rtol_0 < 0.0 || kctx->rtol_0 >= 1.0) {
    SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"0.0 <= rtol_0 < 1.0: %G",kctx->rtol_0);
  }
  if (kctx->rtol_max < 0.0 || kctx->rtol_max >= 1.0) {
    SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"0.0 <= rtol_max (%G) < 1.0\n",kctx->rtol_max);
  }
  if (kctx->gamma < 0.0 || kctx->gamma > 1.0) {
    SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"0.0 <= gamma (%G) <= 1.0\n",kctx->gamma);
  }
  if (kctx->alpha <= 1.0 || kctx->alpha > 2.0) {
    SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"1.0 < alpha (%G) <= 2.0\n",kctx->alpha);
  }
  if (kctx->threshold <= 0.0 || kctx->threshold >= 1.0) {
    SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"0.0 < threshold (%G) < 1.0\n",kctx->threshold);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESKSPGetParametersEW"
/*@
   SNESKSPGetParametersEW - Gets parameters for Eisenstat-Walker
   convergence criteria for the linear solvers within an inexact
   Newton method.

   Not Collective
 
   Input Parameters:
     snes - SNES context

   Output Parameters:
+    version - version 1, 2 (default is 2) or 3
.    rtol_0 - initial relative tolerance (0 <= rtol_0 < 1)
.    rtol_max - maximum relative tolerance (0 <= rtol_max < 1)
.    gamma - multiplicative factor for version 2 rtol computation
             (0 <= gamma2 <= 1)
.    alpha - power for version 2 rtol computation (1 < alpha <= 2)
.    alpha2 - power for safeguard
-    threshold - threshold for imposing safeguard (0 < threshold < 1)

   Level: advanced

.keywords: SNES, KSP, Eisenstat, Walker, get, parameters

.seealso: SNESKSPSetUseEW(), SNESKSPGetUseEW(), SNESKSPSetParametersEW()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESKSPGetParametersEW(SNES snes,PetscInt *version,PetscReal *rtol_0,PetscReal *rtol_max,
							    PetscReal *gamma,PetscReal *alpha,PetscReal *alpha2,PetscReal *threshold)
{
  SNESKSPEW *kctx;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_COOKIE,1);
  kctx = (SNESKSPEW*)snes->kspconvctx;
  if (!kctx) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"No Eisenstat-Walker context existing");
  if(version)   *version   = kctx->version;
  if(rtol_0)    *rtol_0    = kctx->rtol_0;
  if(rtol_max)  *rtol_max  = kctx->rtol_max;
  if(gamma)     *gamma     = kctx->gamma;
  if(alpha)     *alpha     = kctx->alpha;
  if(alpha2)    *alpha2    = kctx->alpha2;
  if(threshold) *threshold = kctx->threshold;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESKSPEW_PreSolve"
static PetscErrorCode SNESKSPEW_PreSolve(SNES snes, KSP ksp, Vec b, Vec x)
{
  PetscErrorCode ierr;
  SNESKSPEW      *kctx = (SNESKSPEW*)snes->kspconvctx;
  PetscReal      rtol=PETSC_DEFAULT,stol;

  PetscFunctionBegin;
  if (!kctx) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"No Eisenstat-Walker context exists");
  if (!snes->iter) { /* first time in, so use the original user rtol */
    rtol = kctx->rtol_0;
  } else {
    if (kctx->version == 1) {
      rtol = (snes->norm - kctx->lresid_last)/kctx->norm_last;
      if (rtol < 0.0) rtol = -rtol;
      stol = pow(kctx->rtol_last,kctx->alpha2);
      if (stol > kctx->threshold) rtol = PetscMax(rtol,stol);
    } else if (kctx->version == 2) {
      rtol = kctx->gamma * pow(snes->norm/kctx->norm_last,kctx->alpha);
      stol = kctx->gamma * pow(kctx->rtol_last,kctx->alpha);
      if (stol > kctx->threshold) rtol = PetscMax(rtol,stol);
    } else if (kctx->version == 3) {/* contributed by Luis Chacon, June 2006. */
      rtol = kctx->gamma * pow(snes->norm/kctx->norm_last,kctx->alpha);
      /* safeguard: avoid sharp decrease of rtol */
      stol = kctx->gamma*pow(kctx->rtol_last,kctx->alpha);
      stol = PetscMax(rtol,stol);
      rtol = PetscMin(kctx->rtol_0,stol);
      /* safeguard: avoid oversolving */
      stol = kctx->gamma*(snes->ttol)/snes->norm;
      stol = PetscMax(rtol,stol);
      rtol = PetscMin(kctx->rtol_0,stol);
    } else SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Only versions 1, 2 or 3 are supported: %D",kctx->version);
  }
  /* safeguard: avoid rtol greater than one */
  rtol = PetscMin(rtol,kctx->rtol_max);
  ierr = KSPSetTolerances(ksp,rtol,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
  ierr = PetscInfo3(snes,"iter %D, Eisenstat-Walker (version %D) KSP rtol=%G\n",snes->iter,kctx->version,rtol);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESKSPEW_PostSolve"
static PetscErrorCode SNESKSPEW_PostSolve(SNES snes, KSP ksp, Vec b, Vec x)
{
  PetscErrorCode ierr;
  SNESKSPEW      *kctx = (SNESKSPEW*)snes->kspconvctx;
  PCSide         pcside;
  Vec            lres;

  PetscFunctionBegin;
  if (!kctx) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"No Eisenstat-Walker context exists");
  ierr = KSPGetTolerances(ksp,&kctx->rtol_last,0,0,0);CHKERRQ(ierr);
  ierr = SNESGetFunctionNorm(snes,&kctx->norm_last);CHKERRQ(ierr);
  if (kctx->version == 1) {
    ierr = KSPGetPreconditionerSide(ksp,&pcside);CHKERRQ(ierr);
    if (pcside == PC_RIGHT) { /* XXX Should we also test KSP_UNPRECONDITIONED_NORM ? */
      /* KSP residual is true linear residual */
      ierr = KSPGetResidualNorm(ksp,&kctx->lresid_last);CHKERRQ(ierr);
    } else {
      /* KSP residual is preconditioned residual */
      /* compute true linear residual norm */
      ierr = VecDuplicate(b,&lres);CHKERRQ(ierr);
      ierr = MatMult(snes->jacobian,x,lres);CHKERRQ(ierr);
      ierr = VecAYPX(lres,-1.0,b);CHKERRQ(ierr);
      ierr = VecNorm(lres,NORM_2,&kctx->lresid_last);CHKERRQ(ierr);
      ierr = VecDestroy(lres);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNES_KSPSolve"
PetscErrorCode SNES_KSPSolve(SNES snes, KSP ksp, Vec b, Vec x)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (snes->ksp_ewconv) { ierr = SNESKSPEW_PreSolve(snes,ksp,b,x);CHKERRQ(ierr);  }
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
  if (snes->ksp_ewconv) { ierr = SNESKSPEW_PostSolve(snes,ksp,b,x);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}
