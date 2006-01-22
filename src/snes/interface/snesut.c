#define PETSCSNES_DLL

#include "src/snes/snesimpl.h"       /*I   "petscsnes.h"   I*/

#undef __FUNCT__  
#define __FUNCT__ "SNESVecViewMonitor"
/*@C
   SNESVecViewMonitor - Monitors progress of the SNES solvers by calling 
   VecView() for the approximate solution at each iteration.

   Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  its - iteration number
.  fgnorm - 2-norm of residual
-  dummy - either a viewer or PETSC_NULL

   Level: intermediate

.keywords: SNES, nonlinear, vector, monitor, view

.seealso: SNESSetMonitor(), SNESDefaultMonitor(), VecView()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESVecViewMonitor(SNES snes,PetscInt its,PetscReal fgnorm,void *dummy)
{
  PetscErrorCode ierr;
  Vec            x;
  PetscViewer    viewer = (PetscViewer) dummy;

  PetscFunctionBegin;
  ierr = SNESGetSolution(snes,&x);CHKERRQ(ierr);
  if (!viewer) {
    MPI_Comm comm;
    ierr   = PetscObjectGetComm((PetscObject)snes,&comm);CHKERRQ(ierr);
    viewer = PETSC_VIEWER_DRAW_(comm);
  }
  ierr = VecView(x,viewer);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESVecViewResidualMonitor"
/*@C
   SNESVecViewResidualMonitor - Monitors progress of the SNES solvers by calling 
   VecView() for the residual at each iteration.

   Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  its - iteration number
.  fgnorm - 2-norm of residual
-  dummy - either a viewer or PETSC_NULL

   Level: intermediate

.keywords: SNES, nonlinear, vector, monitor, view

.seealso: SNESSetMonitor(), SNESDefaultMonitor(), VecView()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESVecViewResidualMonitor(SNES snes,PetscInt its,PetscReal fgnorm,void *dummy)
{
  PetscErrorCode ierr;
  Vec            x;
  PetscViewer    viewer = (PetscViewer) dummy;

  PetscFunctionBegin;
  ierr = SNESGetFunction(snes,&x,0,0);CHKERRQ(ierr);
  if (!viewer) {
    MPI_Comm comm;
    ierr   = PetscObjectGetComm((PetscObject)snes,&comm);CHKERRQ(ierr);
    viewer = PETSC_VIEWER_DRAW_(comm);
  }
  ierr = VecView(x,viewer);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESVecViewUpdateMonitor"
/*@C
   SNESVecViewUpdateMonitor - Monitors progress of the SNES solvers by calling 
   VecView() for the UPDATE to the solution at each iteration.

   Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  its - iteration number
.  fgnorm - 2-norm of residual
-  dummy - either a viewer or PETSC_NULL

   Level: intermediate

.keywords: SNES, nonlinear, vector, monitor, view

.seealso: SNESSetMonitor(), SNESDefaultMonitor(), VecView()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESVecViewUpdateMonitor(SNES snes,PetscInt its,PetscReal fgnorm,void *dummy)
{
  PetscErrorCode ierr;
  Vec            x;
  PetscViewer    viewer = (PetscViewer) dummy;

  PetscFunctionBegin;
  ierr = SNESGetSolutionUpdate(snes,&x);CHKERRQ(ierr);
  if (!viewer) {
    MPI_Comm comm;
    ierr   = PetscObjectGetComm((PetscObject)snes,&comm);CHKERRQ(ierr);
    viewer = PETSC_VIEWER_DRAW_(comm);
  }
  ierr = VecView(x,viewer);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESDefaultMonitor"
/*@C
   SNESDefaultMonitor - Monitors progress of the SNES solvers (default).

   Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  its - iteration number
.  fgnorm - 2-norm of residual
-  dummy - unused context

   Notes:
   This routine prints the residual norm at each iteration.

   Level: intermediate

.keywords: SNES, nonlinear, default, monitor, norm

.seealso: SNESSetMonitor(), SNESVecViewMonitor()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESDefaultMonitor(SNES snes,PetscInt its,PetscReal fgnorm,void *dummy)
{
  PetscErrorCode ierr;
  PetscViewer    viewer = (PetscViewer) dummy;

  PetscFunctionBegin;
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_(snes->comm);
  ierr = PetscViewerASCIIPrintf(viewer,"%3D SNES Function norm %14.12e \n",its,fgnorm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

typedef struct {
  PetscViewer viewer;
  PetscReal   *history;
} SNESRatioMonitorContext;

#undef __FUNCT__  
#define __FUNCT__ "SNESRatioMonitor"
/*@C
   SNESRatioMonitor - Monitors progress of the SNES solvers by printing the ratio
   of residual norm at each iteration to the previous.

   Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  its - iteration number
.  fgnorm - 2-norm of residual (or gradient)
-  dummy -  context of monitor

   Level: intermediate

.keywords: SNES, nonlinear, monitor, norm

.seealso: SNESSetMonitor(), SNESVecViewMonitor()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESRatioMonitor(SNES snes,PetscInt its,PetscReal fgnorm,void *dummy)
{
  PetscErrorCode          ierr;
  PetscInt                len;
  PetscReal               *history;
  SNESRatioMonitorContext *ctx = (SNESRatioMonitorContext*)dummy;

  PetscFunctionBegin;
  ierr = SNESGetConvergenceHistory(snes,&history,PETSC_NULL,&len);CHKERRQ(ierr);
  if (!its || !history || its > len) {
    ierr = PetscViewerASCIIPrintf(ctx->viewer,"%3D SNES Function norm %14.12e \n",its,fgnorm);CHKERRQ(ierr);
  } else {
    PetscReal ratio = fgnorm/history[its-1];
    ierr = PetscViewerASCIIPrintf(ctx->viewer,"%3D SNES Function norm %14.12e %G \n",its,fgnorm,ratio);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
   If the we set the history monitor space then we must destroy it
*/
#undef __FUNCT__  
#define __FUNCT__ "SNESRatioMonitorDestroy"
PetscErrorCode SNESRatioMonitorDestroy(void *ct)
{
  PetscErrorCode          ierr;
  SNESRatioMonitorContext *ctx = (SNESRatioMonitorContext*)ct;

  PetscFunctionBegin;
  ierr = PetscFree(ctx->history);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(ctx->viewer);CHKERRQ(ierr);
  ierr = PetscFree(ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESSetRatioMonitor"
/*@C
   SNESSetRatioMonitor - Sets SNES to use a monitor that prints the 
   ratio of the function norm at each iteration.

   Collective on SNES

   Input Parameters:
+   snes - the SNES context
-   viewer - ASCII viewer to print output

   Level: intermediate

.keywords: SNES, nonlinear, monitor, norm

.seealso: SNESSetMonitor(), SNESVecViewMonitor(), SNESDefaultMonitor()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESSetRatioMonitor(SNES snes,PetscViewer viewer)
{
  PetscErrorCode          ierr;
  SNESRatioMonitorContext *ctx;
  PetscReal               *history;

  PetscFunctionBegin;
  if (!viewer) {
    viewer = PETSC_VIEWER_STDOUT_(snes->comm);
    ierr   = PetscObjectReference((PetscObject)viewer);CHKERRQ(ierr);
  } 
  ierr = PetscNew(SNESRatioMonitorContext,&ctx);
  ierr = SNESGetConvergenceHistory(snes,&history,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  if (!history) {
    ierr = PetscMalloc(100*sizeof(PetscReal),&ctx->history);CHKERRQ(ierr);
    ierr = SNESSetConvergenceHistory(snes,ctx->history,0,100,PETSC_TRUE);CHKERRQ(ierr);
  }
  ctx->viewer = viewer;
  ierr = SNESSetMonitor(snes,SNESRatioMonitor,ctx,SNESRatioMonitorDestroy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "SNESDefaultSMonitor"
/*
     Default (short) SNES Monitor, same as SNESDefaultMonitor() except
  it prints fewer digits of the residual as the residual gets smaller.
  This is because the later digits are meaningless and are often 
  different on different machines; by using this routine different 
  machines will usually generate the same output.
*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESDefaultSMonitor(SNES snes,PetscInt its,PetscReal fgnorm,void *dummy)
{
  PetscErrorCode ierr;
  PetscViewer    viewer = (PetscViewer) dummy;

  PetscFunctionBegin;
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_(snes->comm);
  if (fgnorm > 1.e-9) {
    ierr = PetscViewerASCIIPrintf(viewer,"%3D SNES Function norm %G \n",its,fgnorm);CHKERRQ(ierr);
  } else if (fgnorm > 1.e-11){
    ierr = PetscViewerASCIIPrintf(viewer,"%3D SNES Function norm %5.3e \n",its,fgnorm);CHKERRQ(ierr);
  } else {
    ierr = PetscViewerASCIIPrintf(viewer,"%3D SNES Function norm < 1.e-11\n",its);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
/* ---------------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "SNESConverged_LS"
/*@C 
   SNESConverged_LS - Monitors the convergence of the solvers for
   systems of nonlinear equations (default).

   Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  xnorm - 2-norm of current iterate
.  pnorm - 2-norm of current step 
.  fnorm - 2-norm of function at current iterate
-  dummy - unused context

   Output Parameter:
.   reason  - one of
$  SNES_CONVERGED_FNORM_ABS       - (fnorm < abstol),
$  SNES_CONVERGED_PNORM_RELATIVE  - (pnorm < xtol*xnorm),
$  SNES_CONVERGED_FNORM_RELATIVE  - (fnorm < rtol*fnorm0),
$  SNES_DIVERGED_FUNCTION_COUNT   - (nfct > maxf),
$  SNES_DIVERGED_FNORM_NAN        - (fnorm == NaN),
$  SNES_CONVERGED_ITERATING       - (otherwise),

   where
+    maxf - maximum number of function evaluations,
            set with SNESSetTolerances()
.    nfct - number of function evaluations,
.    abstol - absolute function norm tolerance,
            set with SNESSetTolerances()
-    rtol - relative function norm tolerance, set with SNESSetTolerances()

   Level: intermediate

.keywords: SNES, nonlinear, default, converged, convergence

.seealso: SNESSetConvergenceTest(), SNESEisenstatWalkerConverged()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESConverged_LS(SNES snes,PetscReal xnorm,PetscReal pnorm,PetscReal fnorm,SNESConvergedReason *reason,void *dummy)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (fnorm != fnorm) {
    ierr = PetscInfo(snes,"Failed to converged, function norm is NaN\n");CHKERRQ(ierr);
    *reason = SNES_DIVERGED_FNORM_NAN;
  } else if (fnorm <= snes->ttol) {
    ierr = PetscInfo2(snes,"Converged due to function norm %G < %G (relative tolerance)\n",fnorm,snes->ttol);CHKERRQ(ierr);
    *reason = SNES_CONVERGED_FNORM_RELATIVE;
  } else if (fnorm < snes->abstol) {
    ierr = PetscInfo2(snes,"Converged due to function norm %G < %G\n",fnorm,snes->abstol);CHKERRQ(ierr);
    *reason = SNES_CONVERGED_FNORM_ABS;
  } else if (pnorm < snes->xtol*xnorm) {
    ierr = PetscInfo3(snes,"Converged due to small update length: %G < %G * %G\n",pnorm,snes->xtol,xnorm);CHKERRQ(ierr);
    *reason = SNES_CONVERGED_PNORM_RELATIVE;
  } else if (snes->nfuncs >= snes->max_funcs) {
    ierr = PetscInfo2(snes,"Exceeded maximum number of function evaluations: %D > %D\n",snes->nfuncs,snes->max_funcs);CHKERRQ(ierr);
    *reason = SNES_DIVERGED_FUNCTION_COUNT ;
  } else {
    *reason = SNES_CONVERGED_ITERATING;
  }
  PetscFunctionReturn(0);
}
/* ------------------------------------------------------------ */
#undef __FUNCT__  
#define __FUNCT__ "SNES_KSP_SetConvergenceTestEW"
/*@
   SNES_KSP_SetConvergenceTestEW - Sets alternative convergence test
   for the linear solvers within an inexact Newton method.  

   Collective on SNES

   Input Parameter:
.  snes - SNES context

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
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNES_KSP_SetConvergenceTestEW(SNES snes)
{
  PetscFunctionBegin;
  snes->ksp_ewconv = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNES_KSP_SetParametersEW"
/*@
   SNES_KSP_SetParametersEW - Sets parameters for Eisenstat-Walker
   convergence criteria for the linear solvers within an inexact
   Newton method.

   Collective on SNES
 
   Input Parameters:
+    snes - SNES context
.    version - version 1 or 2 (default is 2)
.    rtol_0 - initial relative tolerance (0 <= rtol_0 < 1)
.    rtol_max - maximum relative tolerance (0 <= rtol_max < 1)
.    alpha - power for version 2 rtol computation (1 < alpha <= 2)
.    alpha2 - power for safeguard
.    gamma2 - multiplicative factor for version 2 rtol computation
              (0 <= gamma2 <= 1)
-    threshold - threshold for imposing safeguard (0 < threshold < 1)

   Note:
   Use PETSC_DEFAULT to retain the default for any of the parameters.

   Level: advanced

   Reference:
   S. C. Eisenstat and H. F. Walker, "Choosing the forcing terms in an 
   inexact Newton method", Utah State University Math. Stat. Dept. Res. 
   Report 6/94/75, June, 1994, to appear in SIAM J. Sci. Comput. 

.keywords: SNES, KSP, Eisenstat, Walker, set, parameters

.seealso: SNES_KSP_SetConvergenceTestEW()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNES_KSP_SetParametersEW(SNES snes,PetscInt version,PetscReal rtol_0,PetscReal rtol_max,PetscReal gamma2,PetscReal alpha,
                                        PetscReal alpha2,PetscReal threshold)
{
  SNES_KSP_EW_ConvCtx *kctx = (SNES_KSP_EW_ConvCtx*)snes->kspconvctx;

  PetscFunctionBegin;
  if (!kctx) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"No Eisenstat-Walker context existing");
  if (version != PETSC_DEFAULT)   kctx->version   = version;
  if (rtol_0 != PETSC_DEFAULT)    kctx->rtol_0    = rtol_0;
  if (rtol_max != PETSC_DEFAULT)  kctx->rtol_max  = rtol_max;
  if (gamma2 != PETSC_DEFAULT)    kctx->gamma     = gamma2;
  if (alpha != PETSC_DEFAULT)     kctx->alpha     = alpha;
  if (alpha2 != PETSC_DEFAULT)    kctx->alpha2    = alpha2;
  if (threshold != PETSC_DEFAULT) kctx->threshold = threshold;
  if (kctx->rtol_0 < 0.0 || kctx->rtol_0 >= 1.0) {
    SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"0.0 <= rtol_0 < 1.0: %G",kctx->rtol_0);
  }
  if (kctx->rtol_max < 0.0 || kctx->rtol_max >= 1.0) {
    SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"0.0 <= rtol_max (%G) < 1.0\n",kctx->rtol_max);
  }
  if (kctx->threshold <= 0.0 || kctx->threshold >= 1.0) {
    SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"0.0 < threshold (%G) < 1.0\n",kctx->threshold);
  }
  if (kctx->gamma < 0.0 || kctx->gamma > 1.0) {
    SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"0.0 <= gamma (%G) <= 1.0\n",kctx->gamma);
  }
  if (kctx->alpha <= 1.0 || kctx->alpha > 2.0) {
    SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"1.0 < alpha (%G) <= 2.0\n",kctx->alpha);
  }
  if (kctx->version != 1 && kctx->version !=2) {
    SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Only versions 1 and 2 are supported: %D",kctx->version);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNES_KSP_EW_ComputeRelativeTolerance_Private"
PetscErrorCode SNES_KSP_EW_ComputeRelativeTolerance_Private(SNES snes,KSP ksp)
{
  SNES_KSP_EW_ConvCtx *kctx = (SNES_KSP_EW_ConvCtx*)snes->kspconvctx;
  PetscReal           rtol = 0.0,stol;
  PetscErrorCode      ierr;

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
    } else SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Only versions 1 or 2 are supported: %D",kctx->version);
  }
  rtol = PetscMin(rtol,kctx->rtol_max);
  kctx->rtol_last = rtol;
  ierr = PetscInfo3(snes,"iter %D, Eisenstat-Walker (version %D) KSP rtol = %G\n",snes->iter,kctx->version,rtol);CHKERRQ(ierr);
  ierr = KSPSetTolerances(ksp,rtol,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
  kctx->norm_last = snes->norm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNES_KSP_EW_Converged_Private"
PetscErrorCode SNES_KSP_EW_Converged_Private(KSP ksp,PetscInt n,PetscReal rnorm,KSPConvergedReason *reason,void *ctx)
{
  SNES                snes = (SNES)ctx;
  SNES_KSP_EW_ConvCtx *kctx = (SNES_KSP_EW_ConvCtx*)snes->kspconvctx;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  if (!kctx) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"No Eisenstat-Walker context set");
  if (!n) {ierr = SNES_KSP_EW_ComputeRelativeTolerance_Private(snes,ksp);CHKERRQ(ierr);}
  ierr = KSPDefaultConverged(ksp,n,rnorm,reason,ctx);CHKERRQ(ierr);
  kctx->lresid_last = rnorm;
  if (*reason) {
    ierr = PetscInfo2(snes,"KSP iterations=%D, rnorm=%G\n",n,rnorm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


