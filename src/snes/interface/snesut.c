#define PETSCSNES_DLL

#include "include/private/snesimpl.h"       /*I   "petscsnes.h"   I*/

#undef __FUNCT__  
#define __FUNCT__ "SNESMonitorSolution"
/*@C
   SNESMonitorSolution - Monitors progress of the SNES solvers by calling 
   VecView() for the approximate solution at each iteration.

   Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  its - iteration number
.  fgnorm - 2-norm of residual
-  dummy - either a viewer or PETSC_NULL

   Level: intermediate

.keywords: SNES, nonlinear, vector, monitor, view

.seealso: SNESMonitorSet(), SNESMonitorDefault(), VecView()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESMonitorSolution(SNES snes,PetscInt its,PetscReal fgnorm,void *dummy)
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
#define __FUNCT__ "SNESMonitorResidual"
/*@C
   SNESMonitorResidual - Monitors progress of the SNES solvers by calling 
   VecView() for the residual at each iteration.

   Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  its - iteration number
.  fgnorm - 2-norm of residual
-  dummy - either a viewer or PETSC_NULL

   Level: intermediate

.keywords: SNES, nonlinear, vector, monitor, view

.seealso: SNESMonitorSet(), SNESMonitorDefault(), VecView()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESMonitorResidual(SNES snes,PetscInt its,PetscReal fgnorm,void *dummy)
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
#define __FUNCT__ "SNESMonitorSolutionUpdate"
/*@C
   SNESMonitorSolutionUpdate - Monitors progress of the SNES solvers by calling 
   VecView() for the UPDATE to the solution at each iteration.

   Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  its - iteration number
.  fgnorm - 2-norm of residual
-  dummy - either a viewer or PETSC_NULL

   Level: intermediate

.keywords: SNES, nonlinear, vector, monitor, view

.seealso: SNESMonitorSet(), SNESMonitorDefault(), VecView()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESMonitorSolutionUpdate(SNES snes,PetscInt its,PetscReal fgnorm,void *dummy)
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
#define __FUNCT__ "SNESMonitorDefault"
/*@C
   SNESMonitorDefault - Monitors progress of the SNES solvers (default).

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

.seealso: SNESMonitorSet(), SNESMonitorSolution()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESMonitorDefault(SNES snes,PetscInt its,PetscReal fgnorm,void *dummy)
{
  PetscErrorCode          ierr;
  PetscViewerASCIIMonitor viewer = (PetscViewerASCIIMonitor) dummy;

  PetscFunctionBegin;
  if (!dummy) {
    ierr = PetscViewerASCIIMonitorCreate(snes->comm,"stdout",0,&viewer);CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIMonitorPrintf(viewer,"%3D SNES Function norm %14.12e \n",its,fgnorm);CHKERRQ(ierr);
  if (!dummy) {
    ierr = PetscViewerASCIIMonitorDestroy(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

typedef struct {
  PetscViewerASCIIMonitor viewer;
  PetscReal               *history;
} SNESMonitorRatioContext;

#undef __FUNCT__  
#define __FUNCT__ "SNESMonitorRatio"
/*@C
   SNESMonitorRatio - Monitors progress of the SNES solvers by printing the ratio
   of residual norm at each iteration to the previous.

   Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  its - iteration number
.  fgnorm - 2-norm of residual (or gradient)
-  dummy -  context of monitor

   Level: intermediate

.keywords: SNES, nonlinear, monitor, norm

.seealso: SNESMonitorSet(), SNESMonitorSolution()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESMonitorRatio(SNES snes,PetscInt its,PetscReal fgnorm,void *dummy)
{
  PetscErrorCode          ierr;
  PetscInt                len;
  PetscReal               *history;
  SNESMonitorRatioContext *ctx = (SNESMonitorRatioContext*)dummy;

  PetscFunctionBegin;
  ierr = SNESGetConvergenceHistory(snes,&history,PETSC_NULL,&len);CHKERRQ(ierr);
  if (!its || !history || its > len) {
    ierr = PetscViewerASCIIMonitorPrintf(ctx->viewer,"%3D SNES Function norm %14.12e \n",its,fgnorm);CHKERRQ(ierr);
  } else {
    PetscReal ratio = fgnorm/history[its-1];
    ierr = PetscViewerASCIIMonitorPrintf(ctx->viewer,"%3D SNES Function norm %14.12e %G \n",its,fgnorm,ratio);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
   If the we set the history monitor space then we must destroy it
*/
#undef __FUNCT__  
#define __FUNCT__ "SNESMonitorRatioDestroy"
PetscErrorCode SNESMonitorRatioDestroy(void *ct)
{
  PetscErrorCode          ierr;
  SNESMonitorRatioContext *ctx = (SNESMonitorRatioContext*)ct;

  PetscFunctionBegin;
  ierr = PetscFree(ctx->history);CHKERRQ(ierr);
  ierr = PetscViewerASCIIMonitorDestroy(ctx->viewer);CHKERRQ(ierr);
  ierr = PetscFree(ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESMonitorSetRatio"
/*@C
   SNESMonitorSetRatio - Sets SNES to use a monitor that prints the 
   ratio of the function norm at each iteration.

   Collective on SNES

   Input Parameters:
+   snes - the SNES context
-   viewer - ASCII viewer to print output

   Level: intermediate

.keywords: SNES, nonlinear, monitor, norm

.seealso: SNESMonitorSet(), SNESMonitorSolution(), SNESMonitorDefault()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESMonitorSetRatio(SNES snes,PetscViewerASCIIMonitor viewer)
{
  PetscErrorCode          ierr;
  SNESMonitorRatioContext *ctx;
  PetscReal               *history;

  PetscFunctionBegin;
  if (!viewer) {
    ierr = PetscViewerASCIIMonitorCreate(snes->comm,"stdout",0,&viewer);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)viewer);CHKERRQ(ierr);
  } 
  ierr = PetscNew(SNESMonitorRatioContext,&ctx);
  ierr = SNESGetConvergenceHistory(snes,&history,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  if (!history) {
    ierr = PetscMalloc(100*sizeof(PetscReal),&ctx->history);CHKERRQ(ierr);
    ierr = SNESSetConvergenceHistory(snes,ctx->history,0,100,PETSC_TRUE);CHKERRQ(ierr);
  }
  ctx->viewer = viewer;
  ierr = SNESMonitorSet(snes,SNESMonitorRatio,ctx,SNESMonitorRatioDestroy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "SNESMonitorDefaultShort"
/*
     Default (short) SNES Monitor, same as SNESMonitorDefault() except
  it prints fewer digits of the residual as the residual gets smaller.
  This is because the later digits are meaningless and are often 
  different on different machines; by using this routine different 
  machines will usually generate the same output.
*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESMonitorDefaultShort(SNES snes,PetscInt its,PetscReal fgnorm,void *dummy)
{
  PetscErrorCode          ierr;
  PetscViewerASCIIMonitor viewer = (PetscViewerASCIIMonitor) dummy;

  PetscFunctionBegin;
  if (!dummy) {
    ierr = PetscViewerASCIIMonitorCreate(snes->comm,"stdout",0,&viewer);CHKERRQ(ierr);
  }
  if (fgnorm > 1.e-9) {
    ierr = PetscViewerASCIIMonitorPrintf(viewer,"%3D SNES Function norm %G \n",its,fgnorm);CHKERRQ(ierr);
  } else if (fgnorm > 1.e-11){
    ierr = PetscViewerASCIIMonitorPrintf(viewer,"%3D SNES Function norm %5.3e \n",its,fgnorm);CHKERRQ(ierr);
  } else {
    ierr = PetscViewerASCIIMonitorPrintf(viewer,"%3D SNES Function norm < 1.e-11\n",its);CHKERRQ(ierr);
  }
  if (!dummy) {
    ierr = PetscViewerASCIIMonitorDestroy(viewer);CHKERRQ(ierr);
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
.  it - the iteration (0 indicates before any Newton steps)
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

.seealso: SNESSetConvergenceTest()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESConverged_LS(SNES snes,PetscInt it,PetscReal xnorm,PetscReal pnorm,PetscReal fnorm,SNESConvergedReason *reason,void *dummy)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *reason = SNES_CONVERGED_ITERATING;

  if (!it) {
    /* set parameter for default relative tolerance convergence test */
    snes->ttol = fnorm*snes->rtol;
  }
  if (fnorm != fnorm) {
    ierr = PetscInfo(snes,"Failed to converged, function norm is NaN\n");CHKERRQ(ierr);
    *reason = SNES_DIVERGED_FNORM_NAN;
  } else if (fnorm < snes->abstol) {
    ierr = PetscInfo2(snes,"Converged due to function norm %G < %G\n",fnorm,snes->abstol);CHKERRQ(ierr);
    *reason = SNES_CONVERGED_FNORM_ABS;
  } else if (snes->nfuncs >= snes->max_funcs) {
    ierr = PetscInfo2(snes,"Exceeded maximum number of function evaluations: %D > %D\n",snes->nfuncs,snes->max_funcs);CHKERRQ(ierr);
    *reason = SNES_DIVERGED_FUNCTION_COUNT;
  }

  if (it && !*reason) {
    if (fnorm <= snes->ttol) {
      ierr = PetscInfo2(snes,"Converged due to function norm %G < %G (relative tolerance)\n",fnorm,snes->ttol);CHKERRQ(ierr);
      *reason = SNES_CONVERGED_FNORM_RELATIVE;
    } else if (pnorm < snes->xtol*xnorm) {
      ierr = PetscInfo3(snes,"Converged due to small update length: %G < %G * %G\n",pnorm,snes->xtol,xnorm);CHKERRQ(ierr);
      *reason = SNES_CONVERGED_PNORM_RELATIVE;
    }
  }
  PetscFunctionReturn(0);
}
