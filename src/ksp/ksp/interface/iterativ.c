/*
   This file contains some simple default routines.
   These routines should be SHORT, since they will be included in every
   executable image that uses the iterative routines (note that, through
   the registry system, we provide a way to load only the truly necessary
   files)
 */
#include <petsc/private/kspimpl.h>   /*I "petscksp.h" I*/
#include <petscdmshell.h>
#include <petscdraw.h>

/*@
   KSPGetResidualNorm - Gets the last (approximate preconditioned)
   residual norm that has been computed.

   Not Collective

   Input Parameters:
.  ksp - the iterative context

   Output Parameters:
.  rnorm - residual norm

   Level: intermediate

.seealso: `KSPBuildResidual()`
@*/
PetscErrorCode  KSPGetResidualNorm(KSP ksp,PetscReal *rnorm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidRealPointer(rnorm,2);
  *rnorm = ksp->rnorm;
  PetscFunctionReturn(0);
}

/*@
   KSPGetIterationNumber - Gets the current iteration number; if the
         KSPSolve() is complete, returns the number of iterations
         used.

   Not Collective

   Input Parameters:
.  ksp - the iterative context

   Output Parameters:
.  its - number of iterations

   Level: intermediate

   Notes:
      During the ith iteration this returns i-1
.seealso: `KSPBuildResidual()`, `KSPGetResidualNorm()`, `KSPGetTotalIterations()`
@*/
PetscErrorCode  KSPGetIterationNumber(KSP ksp,PetscInt *its)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidIntPointer(its,2);
  *its = ksp->its;
  PetscFunctionReturn(0);
}

/*@
   KSPGetTotalIterations - Gets the total number of iterations this KSP object has performed since was created, counted over all linear solves

   Not Collective

   Input Parameters:
.  ksp - the iterative context

   Output Parameters:
.  its - total number of iterations

   Level: intermediate

   Notes:
    Use KSPGetIterationNumber() to get the count for the most recent solve only
   If this is called within a linear solve (such as in a KSPMonitor routine) then it does not include iterations within that current solve

.seealso: `KSPBuildResidual()`, `KSPGetResidualNorm()`, `KSPGetIterationNumber()`
@*/
PetscErrorCode  KSPGetTotalIterations(KSP ksp,PetscInt *its)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidIntPointer(its,2);
  *its = ksp->totalits;
  PetscFunctionReturn(0);
}

/*@C
  KSPMonitorResidual - Print the preconditioned residual norm at each iteration of an iterative solver.

  Collective on ksp

  Input Parameters:
+ ksp   - iterative context
. n     - iteration number
. rnorm - 2-norm (preconditioned) residual value (may be estimated).
- vf    - The viewer context

  Options Database Key:
. -ksp_monitor - Activates KSPMonitorResidual()

  Notes:
  This is not called directly by users, rather one calls `KSPMonitorSet()`, with this function as an argument, to cause the monitor
  to be used during the KSP solve.

  Level: intermediate

.seealso: `KSPMonitorSet()`, `KSPMonitorTrueResidual()`,  `KSPMonitorResidualDraw()`, `KSPMonitorResidualDrawLG()`,
          `KSPMonitorResidualRange()`, `KSPMonitorTrueResidualDraw()`, `KSPMonitorTrueResidualDrawLG()`, `KSPMonitorTrueResidualMax()`,
          `KSPMonitorSingularValue()`, `KSPMonitorSolutionDrawLG()`, `KSPMonitorSolutionDraw()`, `KSPMonitorSolution()`,
          `KSPMonitorErrorDrawLG()`, `KSPMonitorErrorDraw()`, KSPMonitorError()`
@*/
PetscErrorCode KSPMonitorResidual(KSP ksp, PetscInt n, PetscReal rnorm, PetscViewerAndFormat *vf)
{
  PetscViewer       viewer = vf->viewer;
  PetscViewerFormat format = vf->format;
  PetscInt          tablevel;
  const char       *prefix;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 4);
  PetscCall(PetscObjectGetTabLevel((PetscObject) ksp, &tablevel));
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject) ksp, &prefix));
  PetscCall(PetscViewerPushFormat(viewer, format));
  PetscCall(PetscViewerASCIIAddTab(viewer, tablevel));
  if (n == 0 && prefix) PetscCall(PetscViewerASCIIPrintf(viewer, "  Residual norms for %s solve.\n", prefix));
  PetscCall(PetscViewerASCIIPrintf(viewer, "%3" PetscInt_FMT " KSP Residual norm %14.12e \n", n, (double) rnorm));
  PetscCall(PetscViewerASCIISubtractTab(viewer, tablevel));
  PetscCall(PetscViewerPopFormat(viewer));
  PetscFunctionReturn(0);
}

/*@C
  KSPMonitorResidualDraw - Plots the preconditioned residual at each iteration of an iterative solver.

  Collective on ksp

  Input Parameters:
+ ksp   - iterative context
. n     - iteration number
. rnorm - 2-norm (preconditioned) residual value (may be estimated).
- vf    - The viewer context

  Options Database Key:
. -ksp_monitor draw - Activates KSPMonitorResidualDraw()

  Notes:
  This is not called directly by users, rather one calls `KSPMonitorSet()`, with this function as an argument, to cause the monitor
  to be used during the KSP solve.

  Level: intermediate

.seealso: `KSPMonitorSet()`, `KSPMonitorTrueResidual()`, , `KSPMonitorResidual()`, `KSPMonitorResidualDrawLG()`
@*/
PetscErrorCode KSPMonitorResidualDraw(KSP ksp, PetscInt n, PetscReal rnorm, PetscViewerAndFormat *vf)
{
  PetscViewer       viewer = vf->viewer;
  PetscViewerFormat format = vf->format;
  Vec               r;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 4);
  PetscCall(PetscViewerPushFormat(viewer, format));
  PetscCall(KSPBuildResidual(ksp, NULL, NULL, &r));
  PetscCall(PetscObjectSetName((PetscObject) r, "Residual"));
  PetscCall(PetscObjectCompose((PetscObject) r, "__Vec_bc_zero__", (PetscObject) ksp));
  PetscCall(VecView(r, viewer));
  PetscCall(PetscObjectCompose((PetscObject) r, "__Vec_bc_zero__", NULL));
  PetscCall(VecDestroy(&r));
  PetscCall(PetscViewerPopFormat(viewer));
  PetscFunctionReturn(0);
}

/*@C
  KSPMonitorResidualDrawLG - Plots the preconditioned residual norm at each iteration of an iterative solver.

  Collective on ksp

  Input Parameters:
+ ksp   - iterative context
. n     - iteration number
. rnorm - 2-norm (preconditioned) residual value (may be estimated).
- vf    - The viewer context

  Options Database Key:
. -ksp_monitor draw::draw_lg - Activates KSPMonitorResidualDrawLG()

  Notes:
  This is not called directly by users, rather one calls `KSPMonitorSet()`, with this function as an argument, to cause the monitor
  to be used during the KSP solve.

  Level: intermediate

.seealso: `KSPMonitorSet()`, `KSPMonitorTrueResidual()`, `KSPMonitorResidualDraw()`, `KSPMonitorResidual()`
@*/
PetscErrorCode KSPMonitorResidualDrawLG(KSP ksp, PetscInt n, PetscReal rnorm, PetscViewerAndFormat *vf)
{
  PetscViewer        viewer = vf->viewer;
  PetscViewerFormat  format = vf->format;
  PetscDrawLG        lg     = vf->lg;
  KSPConvergedReason reason;
  PetscReal          x, y;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 4);
  PetscValidHeaderSpecific(lg, PETSC_DRAWLG_CLASSID, 4);
  PetscCall(PetscViewerPushFormat(viewer, format));
  if (!n) PetscCall(PetscDrawLGReset(lg));
  x = (PetscReal) n;
  if (rnorm > 0.0) y = PetscLog10Real(rnorm);
  else y = -15.0;
  PetscCall(PetscDrawLGAddPoint(lg, &x, &y));
  PetscCall(KSPGetConvergedReason(ksp, &reason));
  if (n <= 20 || !(n % 5) || reason) {
    PetscCall(PetscDrawLGDraw(lg));
    PetscCall(PetscDrawLGSave(lg));
  }
  PetscCall(PetscViewerPopFormat(viewer));
  PetscFunctionReturn(0);
}

/*@C
  KSPMonitorResidualDrawLGCreate - Creates the plotter for the preconditioned residual.

  Collective on ksp

  Input Parameters:
+ viewer - The PetscViewer
. format - The viewer format
- ctx    - An optional user context

  Output Parameter:
. vf    - The viewer context

  Level: intermediate

.seealso: `KSPMonitorSet()`, `KSPMonitorTrueResidual()`
@*/
PetscErrorCode KSPMonitorResidualDrawLGCreate(PetscViewer viewer, PetscViewerFormat format, void *ctx, PetscViewerAndFormat **vf)
{
  PetscFunctionBegin;
  PetscCall(PetscViewerAndFormatCreate(viewer, format, vf));
  (*vf)->data = ctx;
  PetscCall(KSPMonitorLGCreate(PetscObjectComm((PetscObject) viewer), NULL, NULL, "Log Residual Norm", 1, NULL, PETSC_DECIDE, PETSC_DECIDE, 400, 300, &(*vf)->lg));
  PetscFunctionReturn(0);
}

/*
  This is the same as KSPMonitorResidual() except it prints fewer digits of the residual as the residual gets smaller.
  This is because the later digits are meaningless and are often different on different machines; by using this routine different
  machines will usually generate the same output.

  Deprecated: Intentionally has no manual page
*/
PetscErrorCode KSPMonitorResidualShort(KSP ksp, PetscInt its, PetscReal fnorm, PetscViewerAndFormat *vf)
{
  PetscViewer       viewer = vf->viewer;
  PetscViewerFormat format = vf->format;
  PetscInt          tablevel;
  const char       *prefix;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 4);
  PetscCall(PetscObjectGetTabLevel((PetscObject) ksp, &tablevel));
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject) ksp, &prefix));
  PetscCall(PetscViewerPushFormat(viewer, format));
  PetscCall(PetscViewerASCIIAddTab(viewer, tablevel));
  if (its == 0 && prefix)  PetscCall(PetscViewerASCIIPrintf(viewer, "  Residual norms for %s solve.\n", prefix));
  if (fnorm > 1.e-9)       PetscCall(PetscViewerASCIIPrintf(viewer, "%3" PetscInt_FMT " KSP Residual norm %g \n", its, (double) fnorm));
  else if (fnorm > 1.e-11) PetscCall(PetscViewerASCIIPrintf(viewer, "%3" PetscInt_FMT " KSP Residual norm %5.3e \n", its, (double) fnorm));
  else                     PetscCall(PetscViewerASCIIPrintf(viewer, "%3" PetscInt_FMT " KSP Residual norm < 1.e-11\n", its));
  PetscCall(PetscViewerASCIISubtractTab(viewer, tablevel));
  PetscCall(PetscViewerPopFormat(viewer));
  PetscFunctionReturn(0);
}

PetscErrorCode KSPMonitorRange_Private(KSP ksp, PetscInt it, PetscReal *per)
{
  Vec                resid;
  const PetscScalar *r;
  PetscReal          rmax, pwork;
  PetscInt           i, n, N;

  PetscFunctionBegin;
  PetscCall(KSPBuildResidual(ksp, NULL, NULL, &resid));
  PetscCall(VecNorm(resid, NORM_INFINITY, &rmax));
  PetscCall(VecGetLocalSize(resid, &n));
  PetscCall(VecGetSize(resid, &N));
  PetscCall(VecGetArrayRead(resid, &r));
  pwork = 0.0;
  for (i = 0; i < n; ++i) pwork += (PetscAbsScalar(r[i]) > .20*rmax);
  PetscCall(VecRestoreArrayRead(resid, &r));
  PetscCall(VecDestroy(&resid));
  PetscCall(MPIU_Allreduce(&pwork, per, 1, MPIU_REAL, MPIU_SUM, PetscObjectComm((PetscObject) ksp)));
  *per = *per/N;
  PetscFunctionReturn(0);
}

/*@C
  KSPMonitorResidualRange - Prints the percentage of residual elements that are more then 10 percent of the maximum value.

  Collective on ksp

  Input Parameters:
+ ksp   - iterative context
. it    - iteration number
. rnorm - 2-norm (preconditioned) residual value (may be estimated).
- vf    - The viewer context

  Options Database Key:
. -ksp_monitor_range - Activates KSPMonitorResidualRange()

  Notes:
  This is not called directly by users, rather one calls `KSPMonitorSet()`, with this function as an argument, to cause the monitor
  to be used during the KSP solve.

  Level: intermediate

.seealso: `KSPMonitorSet()`, `KSPMonitorResidual()`
@*/
PetscErrorCode KSPMonitorResidualRange(KSP ksp, PetscInt it, PetscReal rnorm, PetscViewerAndFormat *vf)
{
  static PetscReal  prev;
  PetscViewer       viewer = vf->viewer;
  PetscViewerFormat format = vf->format;
  PetscInt          tablevel;
  const char       *prefix;
  PetscReal         perc, rel;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 4);
  PetscCall(PetscObjectGetTabLevel((PetscObject) ksp, &tablevel));
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject) ksp, &prefix));
  PetscCall(PetscViewerPushFormat(viewer, format));
  PetscCall(PetscViewerASCIIAddTab(viewer, tablevel));
  if (!it) prev = rnorm;
  if (it == 0 && prefix) PetscCall(PetscViewerASCIIPrintf(viewer, "  Residual norms for %s solve.\n", prefix));
  PetscCall(KSPMonitorRange_Private(ksp, it, &perc));
  rel  = (prev - rnorm)/prev;
  prev = rnorm;
  PetscCall(PetscViewerASCIIPrintf(viewer, "%3" PetscInt_FMT " KSP preconditioned resid norm %14.12e Percent values above 20 percent of maximum %5.2f relative decrease %5.2e ratio %5.2e \n", it, (double) rnorm, (double) (100.0*perc), (double) rel, (double) (rel/perc)));
  PetscCall(PetscViewerASCIISubtractTab(viewer, tablevel));
  PetscCall(PetscViewerPopFormat(viewer));
  PetscFunctionReturn(0);
}

/*@C
  KSPMonitorTrueResidual - Prints the true residual norm, as well as the preconditioned residual norm, at each iteration of an iterative solver.

  Collective on ksp

  Input Parameters:
+ ksp   - iterative context
. n     - iteration number
. rnorm - 2-norm (preconditioned) residual value (may be estimated).
- vf    - The viewer context

  Options Database Key:
. -ksp_monitor_true_residual - Activates KSPMonitorTrueResidual()

  Notes:
  When using right preconditioning, these values are equivalent.

  This is not called directly by users, rather one calls `KSPMonitorSet()`, with this function as an argument, to cause the monitor
  to be used during the KSP solve.

  Level: intermediate

.seealso: `KSPMonitorSet()`, `KSPMonitorResidual()`, `KSPMonitorTrueResidualMaxNorm()`
@*/
PetscErrorCode KSPMonitorTrueResidual(KSP ksp, PetscInt n, PetscReal rnorm, PetscViewerAndFormat *vf)
{
  PetscViewer       viewer = vf->viewer;
  PetscViewerFormat format = vf->format;
  Vec               r;
  PetscReal         truenorm, bnorm;
  char              normtype[256];
  PetscInt          tablevel;
  const char       *prefix;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 4);
  PetscCall(PetscObjectGetTabLevel((PetscObject) ksp, &tablevel));
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject) ksp, &prefix));
  PetscCall(PetscStrncpy(normtype, KSPNormTypes[ksp->normtype], sizeof(normtype)));
  PetscCall(PetscStrtolower(normtype));
  PetscCall(KSPBuildResidual(ksp, NULL, NULL, &r));
  PetscCall(VecNorm(r, NORM_2, &truenorm));
  PetscCall(VecNorm(ksp->vec_rhs, NORM_2, &bnorm));
  PetscCall(VecDestroy(&r));

  PetscCall(PetscViewerPushFormat(viewer, format));
  PetscCall(PetscViewerASCIIAddTab(viewer, tablevel));
  if (n == 0 && prefix) PetscCall(PetscViewerASCIIPrintf(viewer, "  Residual norms for %s solve.\n", prefix));
  PetscCall(PetscViewerASCIIPrintf(viewer, "%3" PetscInt_FMT " KSP %s resid norm %14.12e true resid norm %14.12e ||r(i)||/||b|| %14.12e\n", n, normtype, (double) rnorm, (double) truenorm, (double) (truenorm/bnorm)));
  PetscCall(PetscViewerASCIISubtractTab(viewer, tablevel));
  PetscCall(PetscViewerPopFormat(viewer));
  PetscFunctionReturn(0);
}

/*@C
  KSPMonitorTrueResidualDraw - Plots the true residual at each iteration of an iterative solver.

  Collective on ksp

  Input Parameters:
+ ksp   - iterative context
. n     - iteration number
. rnorm - 2-norm (preconditioned) residual value (may be estimated).
- vf    - The viewer context

  Options Database Key:
. -ksp_monitor_true_residual draw - Activates KSPMonitorResidualDraw()

  Notes:
  This is not called directly by users, rather one calls `KSPMonitorSet()`, with this function as an argument, to cause the monitor
  to be used during the KSP solve.

  Level: intermediate

.seealso: `KSPMonitorSet()`, `KSPMonitorTrueResidual()`
@*/
PetscErrorCode KSPMonitorTrueResidualDraw(KSP ksp, PetscInt n, PetscReal rnorm, PetscViewerAndFormat *vf)
{
  PetscViewer       viewer = vf->viewer;
  PetscViewerFormat format = vf->format;
  Vec               r;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 4);
  PetscCall(PetscViewerPushFormat(viewer, format));
  PetscCall(KSPBuildResidual(ksp, NULL, NULL, &r));
  PetscCall(PetscObjectSetName((PetscObject) r, "Residual"));
  PetscCall(PetscObjectCompose((PetscObject) r, "__Vec_bc_zero__", (PetscObject) ksp));
  PetscCall(VecView(r, viewer));
  PetscCall(PetscObjectCompose((PetscObject) r, "__Vec_bc_zero__", NULL));
  PetscCall(VecDestroy(&r));
  PetscCall(PetscViewerPopFormat(viewer));
  PetscFunctionReturn(0);
}

/*@C
  KSPMonitorTrueResidualDrawLG - Plots the true residual norm at each iteration of an iterative solver.

  Collective on ksp

  Input Parameters:
+ ksp   - iterative context
. n     - iteration number
. rnorm - 2-norm (preconditioned) residual value (may be estimated).
- vf    - The viewer context

  Options Database Key:
. -ksp_monitor_true_residual draw::draw_lg - Activates KSPMonitorTrueResidualDrawLG()

  Notes:
  This is not called directly by users, rather one calls `KSPMonitorSet()`, with this function as an argument, to cause the monitor
  to be used during the KSP solve.

  Level: intermediate

.seealso: `KSPMonitorSet()`, `KSPMonitorTrueResidual()`
@*/
PetscErrorCode KSPMonitorTrueResidualDrawLG(KSP ksp, PetscInt n, PetscReal rnorm, PetscViewerAndFormat *vf)
{
  PetscViewer        viewer = vf->viewer;
  PetscViewerFormat  format = vf->format;
  PetscDrawLG        lg     = vf->lg;
  Vec                r;
  KSPConvergedReason reason;
  PetscReal          truenorm, x[2], y[2];

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 4);
  PetscValidHeaderSpecific(lg, PETSC_DRAWLG_CLASSID, 4);
  PetscCall(KSPBuildResidual(ksp, NULL, NULL, &r));
  PetscCall(VecNorm(r, NORM_2, &truenorm));
  PetscCall(VecDestroy(&r));
  PetscCall(PetscViewerPushFormat(viewer, format));
  if (!n) PetscCall(PetscDrawLGReset(lg));
  x[0] = (PetscReal) n;
  if (rnorm > 0.0) y[0] = PetscLog10Real(rnorm);
  else y[0] = -15.0;
  x[1] = (PetscReal) n;
  if (truenorm > 0.0) y[1] = PetscLog10Real(truenorm);
  else y[1] = -15.0;
  PetscCall(PetscDrawLGAddPoint(lg, x, y));
  PetscCall(KSPGetConvergedReason(ksp, &reason));
  if (n <= 20 || !(n % 5) || reason) {
    PetscCall(PetscDrawLGDraw(lg));
    PetscCall(PetscDrawLGSave(lg));
  }
  PetscCall(PetscViewerPopFormat(viewer));
  PetscFunctionReturn(0);
}

/*@C
  KSPMonitorTrueResidualDrawLGCreate - Creates the plotter for the preconditioned residual.

  Collective on ksp

  Input Parameters:
+ viewer - The PetscViewer
. format - The viewer format
- ctx    - An optional user context

  Output Parameter:
. vf    - The viewer context

  Notes:
  This is not called directly by users, rather one calls `KSPMonitorSet()`, with this function as an argument, to cause the monitor
  to be used during the KSP solve.

  Level: intermediate

.seealso: `KSPMonitorSet()`, `KSPMonitorTrueResidual()`
@*/
PetscErrorCode KSPMonitorTrueResidualDrawLGCreate(PetscViewer viewer, PetscViewerFormat format, void *ctx, PetscViewerAndFormat **vf)
{
  const char    *names[] = {"preconditioned", "true"};

  PetscFunctionBegin;
  PetscCall(PetscViewerAndFormatCreate(viewer, format, vf));
  (*vf)->data = ctx;
  PetscCall(KSPMonitorLGCreate(PetscObjectComm((PetscObject) viewer), NULL, NULL, "Log Residual Norm", 2, names, PETSC_DECIDE, PETSC_DECIDE, 400, 300, &(*vf)->lg));
  PetscFunctionReturn(0);
}

/*@C
  KSPMonitorTrueResidualMax - Prints the true residual max norm at each iteration of an iterative solver.

  Collective on ksp

  Input Parameters:
+ ksp   - iterative context
. n     - iteration number
. rnorm - 2-norm (preconditioned) residual value (may be estimated).
- vf    - The viewer context

  Options Database Key:
. -ksp_monitor_true_residual_max - Activates KSPMonitorTrueResidualMax()

  Notes:
  When using right preconditioning, these values are equivalent.

  This is not called directly by users, rather one calls `KSPMonitorSet()`, with this function as an argument, to cause the monitor
  to be used during the KSP solve.

  Level: intermediate

.seealso: `KSPMonitorSet()`, `KSPMonitorResidual()`, `KSPMonitorTrueResidualMaxNorm()`
@*/
PetscErrorCode KSPMonitorTrueResidualMax(KSP ksp, PetscInt n, PetscReal rnorm, PetscViewerAndFormat *vf)
{
  PetscViewer       viewer = vf->viewer;
  PetscViewerFormat format = vf->format;
  Vec               r;
  PetscReal         truenorm, bnorm;
  char              normtype[256];
  PetscInt          tablevel;
  const char       *prefix;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 4);
  PetscCall(PetscObjectGetTabLevel((PetscObject) ksp, &tablevel));
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject) ksp, &prefix));
  PetscCall(PetscStrncpy(normtype, KSPNormTypes[ksp->normtype], sizeof(normtype)));
  PetscCall(PetscStrtolower(normtype));
  PetscCall(KSPBuildResidual(ksp, NULL, NULL, &r));
  PetscCall(VecNorm(r, NORM_INFINITY, &truenorm));
  PetscCall(VecNorm(ksp->vec_rhs, NORM_INFINITY, &bnorm));
  PetscCall(VecDestroy(&r));

  PetscCall(PetscViewerPushFormat(viewer, format));
  PetscCall(PetscViewerASCIIAddTab(viewer, tablevel));
  if (n == 0 && prefix) PetscCall(PetscViewerASCIIPrintf(viewer, "  Residual norms for %s solve.\n", prefix));
  PetscCall(PetscViewerASCIIPrintf(viewer, "%3" PetscInt_FMT " KSP %s true resid max norm %14.12e ||r(i)||/||b|| %14.12e\n", n, normtype, (double) truenorm, (double) (truenorm/bnorm)));
  PetscCall(PetscViewerASCIISubtractTab(viewer, tablevel));
  PetscCall(PetscViewerPopFormat(viewer));
  PetscFunctionReturn(0);
}

/*@C
  KSPMonitorError - Prints the error norm, as well as the preconditioned residual norm, at each iteration of an iterative solver.

  Collective on ksp

  Input Parameters:
+ ksp   - iterative context
. n     - iteration number
. rnorm - 2-norm (preconditioned) residual value (may be estimated).
- vf    - The viewer context

  Options Database Key:
. -ksp_monitor_error - Activates KSPMonitorError()

  Notes:
  This is not called directly by users, rather one calls `KSPMonitorSet()`, with this function as an argument, to cause the monitor
  to be used during the KSP solve.

  Level: intermediate

.seealso: `KSPMonitorSet()`, `KSPMonitorResidual()`, `KSPMonitorTrueResidualMaxNorm()`
@*/
PetscErrorCode KSPMonitorError(KSP ksp, PetscInt n, PetscReal rnorm, PetscViewerAndFormat *vf)
{
  PetscViewer       viewer = vf->viewer;
  PetscViewerFormat format = vf->format;
  DM                dm;
  Vec               sol;
  PetscReal        *errors;
  PetscInt          Nf, f;
  PetscInt          tablevel;
  const char       *prefix;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 4);
  PetscCall(PetscObjectGetTabLevel((PetscObject) ksp, &tablevel));
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject) ksp, &prefix));
  PetscCall(KSPGetDM(ksp, &dm));
  PetscCall(DMGetNumFields(dm, &Nf));
  PetscCall(DMGetGlobalVector(dm, &sol));
  PetscCall(KSPBuildSolution(ksp, sol, NULL));
  /* TODO: Make a different monitor that flips sign for SNES, Newton system is A dx = -b, so we need to negate the solution */
  PetscCall(VecScale(sol, -1.0));
  PetscCall(PetscCalloc1(Nf, &errors));
  PetscCall(DMComputeError(dm, sol, errors, NULL));

  PetscCall(PetscViewerPushFormat(viewer, format));
  PetscCall(PetscViewerASCIIAddTab(viewer, tablevel));
  if (n == 0 && prefix) PetscCall(PetscViewerASCIIPrintf(viewer, "  Error norms for %s solve.\n", prefix));
  PetscCall(PetscViewerASCIIPrintf(viewer, "%3" PetscInt_FMT " KSP Error norm %s", n, Nf > 1 ? "[" : ""));
  PetscCall(PetscViewerASCIIUseTabs(viewer, PETSC_FALSE));
  for (f = 0; f < Nf; ++f) {
    if (f > 0) PetscCall(PetscViewerASCIIPrintf(viewer, ", "));
    PetscCall(PetscViewerASCIIPrintf(viewer, "%14.12e", (double) errors[f]));
  }
  PetscCall(PetscViewerASCIIPrintf(viewer, "%s resid norm %14.12e\n", Nf > 1 ? "]" : "", (double) rnorm));
  PetscCall(PetscViewerASCIIUseTabs(viewer, PETSC_TRUE));
  PetscCall(PetscViewerASCIISubtractTab(viewer, tablevel));
  PetscCall(PetscViewerPopFormat(viewer));
  PetscCall(DMRestoreGlobalVector(dm, &sol));
  PetscFunctionReturn(0);
}

/*@C
  KSPMonitorErrorDraw - Plots the error at each iteration of an iterative solver.

  Collective on ksp

  Input Parameters:
+ ksp   - iterative context
. n     - iteration number
. rnorm - 2-norm (preconditioned) residual value (may be estimated).
- vf    - The viewer context

  Options Database Key:
. -ksp_monitor_error draw - Activates KSPMonitorErrorDraw()

  Notes:
  This is not called directly by users, rather one calls `KSPMonitorSet()`, with this function as an argument, to cause the monitor
  to be used during the KSP solve.

  Level: intermediate

.seealso: `KSPMonitorSet()`, `KSPMonitorTrueResidual()`
@*/
PetscErrorCode KSPMonitorErrorDraw(KSP ksp, PetscInt n, PetscReal rnorm, PetscViewerAndFormat *vf)
{
  PetscViewer       viewer = vf->viewer;
  PetscViewerFormat format = vf->format;
  DM                dm;
  Vec               sol, e;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 4);
  PetscCall(PetscViewerPushFormat(viewer, format));
  PetscCall(KSPGetDM(ksp, &dm));
  PetscCall(DMGetGlobalVector(dm, &sol));
  PetscCall(KSPBuildSolution(ksp, sol, NULL));
  PetscCall(DMComputeError(dm, sol, NULL, &e));
  PetscCall(PetscObjectSetName((PetscObject) e, "Error"));
  PetscCall(PetscObjectCompose((PetscObject) e, "__Vec_bc_zero__", (PetscObject) ksp));
  PetscCall(VecView(e, viewer));
  PetscCall(PetscObjectCompose((PetscObject) e, "__Vec_bc_zero__", NULL));
  PetscCall(VecDestroy(&e));
  PetscCall(DMRestoreGlobalVector(dm, &sol));
  PetscCall(PetscViewerPopFormat(viewer));
  PetscFunctionReturn(0);
}

/*@C
  KSPMonitorErrorDrawLG - Plots the error and residual norm at each iteration of an iterative solver.

  Collective on ksp

  Input Parameters:
+ ksp   - iterative context
. n     - iteration number
. rnorm - 2-norm (preconditioned) residual value (may be estimated).
- vf    - The viewer context

  Options Database Key:
. -ksp_monitor_error draw::draw_lg - Activates KSPMonitorTrueResidualDrawLG()

  Notes:
  This is not called directly by users, rather one calls `KSPMonitorSet()`, with this function as an argument, to cause the monitor
  to be used during the KSP solve.

  Level: intermediate

.seealso: `KSPMonitorSet()`, `KSPMonitorTrueResidual()`
@*/
PetscErrorCode KSPMonitorErrorDrawLG(KSP ksp, PetscInt n, PetscReal rnorm, PetscViewerAndFormat *vf)
{
  PetscViewer        viewer = vf->viewer;
  PetscViewerFormat  format = vf->format;
  PetscDrawLG        lg     = vf->lg;
  DM                 dm;
  Vec                sol;
  KSPConvergedReason reason;
  PetscReal         *x, *errors;
  PetscInt           Nf, f;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 4);
  PetscValidHeaderSpecific(lg, PETSC_DRAWLG_CLASSID, 4);
  PetscCall(KSPGetDM(ksp, &dm));
  PetscCall(DMGetNumFields(dm, &Nf));
  PetscCall(DMGetGlobalVector(dm, &sol));
  PetscCall(KSPBuildSolution(ksp, sol, NULL));
  /* TODO: Make a different monitor that flips sign for SNES, Newton system is A dx = -b, so we need to negate the solution */
  PetscCall(VecScale(sol, -1.0));
  PetscCall(PetscCalloc2(Nf+1, &x, Nf+1, &errors));
  PetscCall(DMComputeError(dm, sol, errors, NULL));

  PetscCall(PetscViewerPushFormat(viewer, format));
  if (!n) PetscCall(PetscDrawLGReset(lg));
  for (f = 0; f < Nf; ++f) {
    x[f]      = (PetscReal) n;
    errors[f] = errors[f] > 0.0 ? PetscLog10Real(errors[f]) : -15.;
  }
  x[Nf]      = (PetscReal) n;
  errors[Nf] = rnorm > 0.0 ? PetscLog10Real(rnorm) : -15.;
  PetscCall(PetscDrawLGAddPoint(lg, x, errors));
  PetscCall(KSPGetConvergedReason(ksp, &reason));
  if (n <= 20 || !(n % 5) || reason) {
    PetscCall(PetscDrawLGDraw(lg));
    PetscCall(PetscDrawLGSave(lg));
  }
  PetscCall(PetscViewerPopFormat(viewer));
  PetscFunctionReturn(0);
}

/*@C
  KSPMonitorErrorDrawLGCreate - Creates the plotter for the error and preconditioned residual.

  Collective on ksp

  Input Parameters:
+ viewer - The PetscViewer
. format - The viewer format
- ctx    - An optional user context

  Output Parameter:
. vf    - The viewer context

  Notes:
  This is not called directly by users, rather one calls `KSPMonitorSet()`, with this function as an argument, to cause the monitor
  to be used during the KSP solve.

  Level: intermediate

.seealso: `KSPMonitorSet()`, `KSPMonitorTrueResidual()`
@*/
PetscErrorCode KSPMonitorErrorDrawLGCreate(PetscViewer viewer, PetscViewerFormat format, void *ctx, PetscViewerAndFormat **vf)
{
  KSP            ksp = (KSP) ctx;
  DM             dm;
  char         **names;
  PetscInt       Nf, f;

  PetscFunctionBegin;
  PetscCall(KSPGetDM(ksp, &dm));
  PetscCall(DMGetNumFields(dm, &Nf));
  PetscCall(PetscMalloc1(Nf+1, &names));
  for (f = 0; f < Nf; ++f) {
    PetscObject disc;
    const char *fname;
    char        lname[PETSC_MAX_PATH_LEN];

    PetscCall(DMGetField(dm, f, NULL, &disc));
    PetscCall(PetscObjectGetName(disc, &fname));
    PetscCall(PetscStrncpy(lname, fname, PETSC_MAX_PATH_LEN));
    PetscCall(PetscStrlcat(lname, " Error", PETSC_MAX_PATH_LEN));
    PetscCall(PetscStrallocpy(lname, &names[f]));
  }
  PetscCall(PetscStrallocpy("residual", &names[Nf]));
  PetscCall(PetscViewerAndFormatCreate(viewer, format, vf));
  (*vf)->data = ctx;
  PetscCall(KSPMonitorLGCreate(PetscObjectComm((PetscObject) viewer), NULL, NULL, "Log Error Norm", Nf+1, (const char **) names, PETSC_DECIDE, PETSC_DECIDE, 400, 300, &(*vf)->lg));
  for (f = 0; f <= Nf; ++f) PetscCall(PetscFree(names[f]));
  PetscCall(PetscFree(names));
  PetscFunctionReturn(0);
}

/*@C
  KSPMonitorSolution - Print the solution norm at each iteration of an iterative solver.

  Collective on ksp

  Input Parameters:
+ ksp   - iterative context
. n     - iteration number
. rnorm - 2-norm (preconditioned) residual value (may be estimated).
- vf    - The viewer context

  Options Database Key:
. -ksp_monitor_solution - Activates KSPMonitorSolution()

  Notes:
  This is not called directly by users, rather one calls `KSPMonitorSet()`, with this function as an argument, to cause the monitor
  to be used during the KSP solve.

  Level: intermediate

.seealso: `KSPMonitorSet()`, `KSPMonitorTrueResidual()`
@*/
PetscErrorCode KSPMonitorSolution(KSP ksp, PetscInt n, PetscReal rnorm, PetscViewerAndFormat *vf)
{
  PetscViewer       viewer = vf->viewer;
  PetscViewerFormat format = vf->format;
  Vec               x;
  PetscReal         snorm;
  PetscInt          tablevel;
  const char       *prefix;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 4);
  PetscCall(KSPBuildSolution(ksp, NULL, &x));
  PetscCall(VecNorm(x, NORM_2, &snorm));
  PetscCall(PetscObjectGetTabLevel((PetscObject) ksp, &tablevel));
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject) ksp, &prefix));
  PetscCall(PetscViewerPushFormat(viewer, format));
  PetscCall(PetscViewerASCIIAddTab(viewer, tablevel));
  if (n == 0 && prefix) PetscCall(PetscViewerASCIIPrintf(viewer, "  Solution norms for %s solve.\n", prefix));
  PetscCall(PetscViewerASCIIPrintf(viewer, "%3" PetscInt_FMT " KSP Solution norm %14.12e \n", n, (double) snorm));
  PetscCall(PetscViewerASCIISubtractTab(viewer, tablevel));
  PetscCall(PetscViewerPopFormat(viewer));
  PetscFunctionReturn(0);
}

/*@C
  KSPMonitorSolutionDraw - Plots the solution at each iteration of an iterative solver.

  Collective on ksp

  Input Parameters:
+ ksp   - iterative context
. n     - iteration number
. rnorm - 2-norm (preconditioned) residual value (may be estimated).
- vf    - The viewer context

  Options Database Key:
. -ksp_monitor_solution draw - Activates KSPMonitorSolutionDraw()

  Notes:
  This is not called directly by users, rather one calls `KSPMonitorSet()`, with this function as an argument, to cause the monitor
  to be used during the KSP solve.

  Level: intermediate

.seealso: `KSPMonitorSet()`, `KSPMonitorTrueResidual()`
@*/
PetscErrorCode KSPMonitorSolutionDraw(KSP ksp, PetscInt n, PetscReal rnorm, PetscViewerAndFormat *vf)
{
  PetscViewer       viewer = vf->viewer;
  PetscViewerFormat format = vf->format;
  Vec               x;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 4);
  PetscCall(KSPBuildSolution(ksp, NULL, &x));
  PetscCall(PetscViewerPushFormat(viewer, format));
  PetscCall(PetscObjectSetName((PetscObject) x, "Solution"));
  PetscCall(PetscObjectCompose((PetscObject) x, "__Vec_bc_zero__", (PetscObject) ksp));
  PetscCall(VecView(x, viewer));
  PetscCall(PetscObjectCompose((PetscObject) x, "__Vec_bc_zero__", NULL));
  PetscCall(PetscViewerPopFormat(viewer));
  PetscFunctionReturn(0);
}

/*@C
  KSPMonitorSolutionDrawLG - Plots the solution norm at each iteration of an iterative solver.

  Collective on ksp

  Input Parameters:
+ ksp   - iterative context
. n     - iteration number
. rnorm - 2-norm (preconditioned) residual value (may be estimated).
- vf    - The viewer context

  Options Database Key:
. -ksp_monitor_solution draw::draw_lg - Activates KSPMonitorSolutionDrawLG()

  Notes:
  This is not called directly by users, rather one calls `KSPMonitorSet()`, with this function as an argument, to cause the monitor
  to be used during the KSP solve.

  Level: intermediate

.seealso: `KSPMonitorSet()`, `KSPMonitorTrueResidual()`
@*/
PetscErrorCode KSPMonitorSolutionDrawLG(KSP ksp, PetscInt n, PetscReal rnorm, PetscViewerAndFormat *vf)
{
  PetscViewer        viewer = vf->viewer;
  PetscViewerFormat  format = vf->format;
  PetscDrawLG        lg     = vf->lg;
  Vec                u;
  KSPConvergedReason reason;
  PetscReal          snorm, x, y;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 4);
  PetscValidHeaderSpecific(lg, PETSC_DRAWLG_CLASSID, 4);
  PetscCall(KSPBuildSolution(ksp, NULL, &u));
  PetscCall(VecNorm(u, NORM_2, &snorm));
  PetscCall(PetscViewerPushFormat(viewer, format));
  if (!n) PetscCall(PetscDrawLGReset(lg));
  x = (PetscReal) n;
  if (snorm > 0.0) y = PetscLog10Real(snorm);
  else y = -15.0;
  PetscCall(PetscDrawLGAddPoint(lg, &x, &y));
  PetscCall(KSPGetConvergedReason(ksp, &reason));
  if (n <= 20 || !(n % 5) || reason) {
    PetscCall(PetscDrawLGDraw(lg));
    PetscCall(PetscDrawLGSave(lg));
  }
  PetscCall(PetscViewerPopFormat(viewer));
  PetscFunctionReturn(0);
}

/*@C
  KSPMonitorSolutionDrawLGCreate - Creates the plotter for the solution.

  Collective on ksp

  Input Parameters:
+ viewer - The PetscViewer
. format - The viewer format
- ctx    - An optional user context

  Output Parameter:
. vf    - The viewer context

  Notes:
  This is not called directly by users, rather one calls `KSPMonitorSet()`, with this function as an argument, to cause the monitor
  to be used during the KSP solve.

  Level: intermediate

.seealso: `KSPMonitorSet()`, `KSPMonitorTrueResidual()`
@*/
PetscErrorCode KSPMonitorSolutionDrawLGCreate(PetscViewer viewer, PetscViewerFormat format, void *ctx, PetscViewerAndFormat **vf)
{
  PetscFunctionBegin;
  PetscCall(PetscViewerAndFormatCreate(viewer, format, vf));
  (*vf)->data = ctx;
  PetscCall(KSPMonitorLGCreate(PetscObjectComm((PetscObject) viewer), NULL, NULL, "Log Solution Norm", 1, NULL, PETSC_DECIDE, PETSC_DECIDE, 400, 300, &(*vf)->lg));
  PetscFunctionReturn(0);
}

/*@C
  KSPMonitorSingularValue - Prints the two norm of the true residual and estimation of the extreme singular values of the preconditioned problem at each iteration.

  Logically Collective on ksp

  Input Parameters:
+ ksp   - the iterative context
. n     - the iteration
. rnorm - the two norm of the residual
- vf    - The viewer context

  Options Database Key:
. -ksp_monitor_singular_value - Activates KSPMonitorSingularValue()

  Notes:
  The CG solver uses the Lanczos technique for eigenvalue computation,
  while GMRES uses the Arnoldi technique; other iterative methods do
  not currently compute singular values.

  This is not called directly by users, rather one calls `KSPMonitorSet()`, with this function as an argument, to cause the monitor
  to be used during the KSP solve.

  Level: intermediate

.seealso: `KSPComputeExtremeSingularValues()`, `KSPMonitorSingularValueCreate()`
@*/
PetscErrorCode KSPMonitorSingularValue(KSP ksp, PetscInt n, PetscReal rnorm, PetscViewerAndFormat *vf)
{
  PetscViewer       viewer = vf->viewer;
  PetscViewerFormat format = vf->format;
  PetscReal         emin, emax;
  PetscInt          tablevel;
  const char       *prefix;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 4);
  PetscCall(PetscObjectGetTabLevel((PetscObject) ksp, &tablevel));
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject) ksp, &prefix));
  PetscCall(PetscViewerPushFormat(viewer, format));
  PetscCall(PetscViewerASCIIAddTab(viewer, tablevel));
  if (n == 0 && prefix) PetscCall(PetscViewerASCIIPrintf(viewer, "  Residual norms for %s solve.\n", prefix));
  if (!ksp->calc_sings) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "%3" PetscInt_FMT " KSP Residual norm %14.12e \n", n, (double) rnorm));
  } else {
    PetscCall(KSPComputeExtremeSingularValues(ksp, &emax, &emin));
    PetscCall(PetscViewerASCIIPrintf(viewer, "%3" PetscInt_FMT " KSP Residual norm %14.12e %% max %14.12e min %14.12e max/min %14.12e\n", n, (double) rnorm, (double) emax, (double) emin, (double) (emax/emin)));
  }
  PetscCall(PetscViewerASCIISubtractTab(viewer, tablevel));
  PetscCall(PetscViewerPopFormat(viewer));
  PetscFunctionReturn(0);
}

/*@C
  KSPMonitorSingularValueCreate - Creates the singular value monitor.

  Collective on ksp

  Input Parameters:
+ viewer - The PetscViewer
. format - The viewer format
- ctx    - An optional user context

  Output Parameter:
. vf    - The viewer context

  Level: intermediate

.seealso: `KSPMonitorSet()`, `KSPMonitorSingularValue()`
@*/
PetscErrorCode KSPMonitorSingularValueCreate(PetscViewer viewer, PetscViewerFormat format, void *ctx, PetscViewerAndFormat **vf)
{
  KSP            ksp = (KSP) ctx;

  PetscFunctionBegin;
  PetscCall(PetscViewerAndFormatCreate(viewer, format, vf));
  (*vf)->data = ctx;
  PetscCall(KSPSetComputeSingularValues(ksp, PETSC_TRUE));
  PetscFunctionReturn(0);
}

/*@C
   KSPMonitorDynamicTolerance - Recompute the inner tolerance in every outer iteration in an adaptive way.

   Collective on ksp

   Input Parameters:
+  ksp   - iterative context
.  n     - iteration number (not used)
.  fnorm - the current residual norm
-  dummy - some context as a C struct. fields:
             coef: a scaling coefficient. default 1.0. can be passed through
                   -sub_ksp_dynamic_tolerance_param
             bnrm: norm of the right-hand side. store it to avoid repeated calculation

   Notes:
   This may be useful for a flexibly preconditioner Krylov method to
   control the accuracy of the inner solves needed to guarantee the
   convergence of the outer iterations.

   This is not called directly by users, rather one calls `KSPMonitorSet()`, with this function as an argument, to cause the monitor
   to be used during the KSP solve.

   Level: advanced

.seealso: `KSPMonitorDynamicToleranceDestroy()`
@*/
PetscErrorCode KSPMonitorDynamicTolerance(KSP ksp,PetscInt its,PetscReal fnorm,void *dummy)
{
  PC             pc;
  PetscReal      outer_rtol, outer_abstol, outer_dtol, inner_rtol;
  PetscInt       outer_maxits,nksp,first,i;
  KSPDynTolCtx   *scale   = (KSPDynTolCtx*)dummy;
  KSP            *subksp = NULL;
  KSP            kspinner;
  PetscBool      flg;

  PetscFunctionBegin;
  PetscCall(KSPGetPC(ksp, &pc));

  /* compute inner_rtol */
  if (scale->bnrm < 0.0) {
    Vec b;
    PetscCall(KSPGetRhs(ksp, &b));
    PetscCall(VecNorm(b, NORM_2, &(scale->bnrm)));
  }
  PetscCall(KSPGetTolerances(ksp, &outer_rtol, &outer_abstol, &outer_dtol, &outer_maxits));
  inner_rtol = PetscMin(scale->coef * scale->bnrm * outer_rtol / fnorm, 0.999);
  /* PetscCall(PetscPrintf(PETSC_COMM_WORLD, "        Inner rtol = %g\n",
     (double)inner_rtol)); */

  /* if pc is ksp */
  PetscCall(PetscObjectTypeCompare((PetscObject)pc,PCKSP,&flg));
  if (flg) {
    PetscCall(PCKSPGetKSP(pc, &kspinner));
    PetscCall(KSPSetTolerances(kspinner, inner_rtol, outer_abstol, outer_dtol, outer_maxits));
    PetscFunctionReturn(0);
  }

  /* if pc is bjacobi */
  PetscCall(PetscObjectTypeCompare((PetscObject)pc,PCBJACOBI,&flg));
  if (flg) {
    PetscCall(PCBJacobiGetSubKSP(pc, &nksp, &first, &subksp));
    if (subksp) {
      for (i=0; i<nksp; i++) {
        PetscCall(KSPSetTolerances(subksp[i], inner_rtol, outer_abstol, outer_dtol, outer_maxits));
      }
      PetscFunctionReturn(0);
    }
  }

  /* if pc is deflation*/
  PetscCall(PetscObjectTypeCompare((PetscObject)pc,PCDEFLATION,&flg));
  if (flg) {
    PetscCall(PCDeflationGetCoarseKSP(pc,&kspinner));
    PetscCall(KSPSetTolerances(kspinner,inner_rtol,outer_abstol,outer_dtol,PETSC_DEFAULT));
    PetscFunctionReturn(0);
  }

  /* todo: dynamic tolerance may apply to other types of pc too */
  PetscFunctionReturn(0);
}

/*
  Destroy the dummy context used in KSPMonitorDynamicTolerance()
*/
PetscErrorCode KSPMonitorDynamicToleranceDestroy(void **dummy)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(*dummy));
  PetscFunctionReturn(0);
}

/*@C
   KSPConvergedSkip - Convergence test that do not return as converged
   until the maximum number of iterations is reached.

   Collective on ksp

   Input Parameters:
+  ksp   - iterative context
.  n     - iteration number
.  rnorm - 2-norm residual value (may be estimated)
-  dummy - unused convergence context

   Returns:
.  reason - KSP_CONVERGED_ITERATING, KSP_CONVERGED_ITS

   Notes:
   This should be used as the convergence test with the option
   KSPSetNormType(ksp,KSP_NORM_NONE), since norms of the residual are
   not computed. Convergence is then declared after the maximum number
   of iterations have been reached. Useful when one is using CG or
   BiCGStab as a smoother.

   Level: advanced

.seealso: `KSPSetConvergenceTest()`, `KSPSetTolerances()`, `KSPSetNormType()`
@*/
PetscErrorCode  KSPConvergedSkip(KSP ksp,PetscInt n,PetscReal rnorm,KSPConvergedReason *reason,void *dummy)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidPointer(reason,4);
  *reason = KSP_CONVERGED_ITERATING;
  if (n >= ksp->max_it) *reason = KSP_CONVERGED_ITS;
  PetscFunctionReturn(0);
}

/*@C
   KSPConvergedDefaultCreate - Creates and initializes the space used by the KSPConvergedDefault() function context

   Note Collective

   Output Parameter:
.  ctx - convergence context

   Level: intermediate

.seealso: `KSPConvergedDefault()`, `KSPConvergedDefaultDestroy()`, `KSPSetConvergenceTest()`, `KSPSetTolerances()`,
          `KSPConvergedSkip()`, `KSPConvergedReason`, `KSPGetConvergedReason()`, `KSPConvergedDefaultSetUIRNorm()`, `KSPConvergedDefaultSetUMIRNorm()`, `KSPConvergedDefaultSetConvergedMaxits()`
@*/
PetscErrorCode  KSPConvergedDefaultCreate(void **ctx)
{
  KSPConvergedDefaultCtx *cctx;

  PetscFunctionBegin;
  PetscCall(PetscNew(&cctx));
  *ctx = cctx;
  PetscFunctionReturn(0);
}

/*@
   KSPConvergedDefaultSetUIRNorm - makes the default convergence test use || B*(b - A*(initial guess))||
      instead of || B*b ||. In the case of right preconditioner or if KSPSetNormType(ksp,KSP_NORM_UNPRECONDITIONED)
      is used there is no B in the above formula. UIRNorm is short for Use Initial Residual Norm.

   Collective on ksp

   Input Parameters:
.  ksp   - iterative context

   Options Database:
.   -ksp_converged_use_initial_residual_norm <bool> - Use initial residual norm for computing relative convergence

   Notes:
   Use KSPSetTolerances() to alter the defaults for rtol, abstol, dtol.

   The precise values of reason are macros such as KSP_CONVERGED_RTOL, which
   are defined in petscksp.h.

   If the convergence test is not KSPConvergedDefault() then this is ignored.

   If right preconditioning is being used then B does not appear in the above formula.

   Level: intermediate

.seealso: `KSPSetConvergenceTest()`, `KSPSetTolerances()`, `KSPConvergedSkip()`, `KSPConvergedReason`, `KSPGetConvergedReason()`, `KSPConvergedDefaultSetUMIRNorm()`, `KSPConvergedDefaultSetConvergedMaxits()`
@*/
PetscErrorCode  KSPConvergedDefaultSetUIRNorm(KSP ksp)
{
  KSPConvergedDefaultCtx *ctx = (KSPConvergedDefaultCtx*) ksp->cnvP;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  if (ksp->converged != KSPConvergedDefault) PetscFunctionReturn(0);
  PetscCheck(!ctx->mininitialrtol,PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_WRONGSTATE,"Cannot use KSPConvergedDefaultSetUIRNorm() and KSPConvergedDefaultSetUMIRNorm() together");
  ctx->initialrtol = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@
   KSPConvergedDefaultSetUMIRNorm - makes the default convergence test use min(|| B*(b - A*(initial guess))||,|| B*b ||)
      In the case of right preconditioner or if KSPSetNormType(ksp,KSP_NORM_UNPRECONDITIONED)
      is used there is no B in the above formula. UMIRNorm is short for Use Minimum Initial Residual Norm.

   Collective on ksp

   Input Parameters:
.  ksp   - iterative context

   Options Database:
.   -ksp_converged_use_min_initial_residual_norm <bool> - Use minimum of initial residual norm and b for computing relative convergence

   Use KSPSetTolerances() to alter the defaults for rtol, abstol, dtol.

   The precise values of reason are macros such as KSP_CONVERGED_RTOL, which
   are defined in petscksp.h.

   Level: intermediate

.seealso: `KSPSetConvergenceTest()`, `KSPSetTolerances()`, `KSPConvergedSkip()`, `KSPConvergedReason`, `KSPGetConvergedReason()`, `KSPConvergedDefaultSetUIRNorm()`, `KSPConvergedDefaultSetConvergedMaxits()`
@*/
PetscErrorCode  KSPConvergedDefaultSetUMIRNorm(KSP ksp)
{
  KSPConvergedDefaultCtx *ctx = (KSPConvergedDefaultCtx*) ksp->cnvP;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  if (ksp->converged != KSPConvergedDefault) PetscFunctionReturn(0);
  PetscCheck(!ctx->initialrtol,PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_WRONGSTATE,"Cannot use KSPConvergedDefaultSetUIRNorm() and KSPConvergedDefaultSetUMIRNorm() together");
  ctx->mininitialrtol = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@
   KSPConvergedDefaultSetConvergedMaxits - allows the default convergence test to declare convergence and return KSP_CONVERGED_ITS if the maximum number of iterations is reached

   Collective on ksp

   Input Parameters:
+  ksp - iterative context
-  flg - boolean flag

   Options Database:
.   -ksp_converged_maxits <bool> - Declare convergence if the maximum number of iterations is reached

   Use KSPSetTolerances() to alter the defaults for rtol, abstol, dtol.

   The precise values of reason are macros such as KSP_CONVERGED_RTOL, which
   are defined in petscksp.h.

   Level: intermediate

.seealso: `KSPSetConvergenceTest()`, `KSPSetTolerances()`, `KSPConvergedSkip()`, `KSPConvergedReason`, `KSPGetConvergedReason()`, `KSPConvergedDefaultSetUMIRNorm()`, `KSPConvergedDefaultSetUIRNorm()`
@*/
PetscErrorCode  KSPConvergedDefaultSetConvergedMaxits(KSP ksp, PetscBool flg)
{
  KSPConvergedDefaultCtx *ctx = (KSPConvergedDefaultCtx*) ksp->cnvP;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidLogicalCollectiveBool(ksp,flg,2);
  if (ksp->converged != KSPConvergedDefault) PetscFunctionReturn(0);
  ctx->convmaxits = flg;
  PetscFunctionReturn(0);
}

/*@C
   KSPConvergedDefault - Determines convergence of the linear iterative solvers by default

   Collective on ksp

   Input Parameters:
+  ksp   - iterative context
.  n     - iteration number
.  rnorm - residual norm (may be estimated, depending on the method may be the preconditioned residual norm)
-  ctx - convergence context which must be created by KSPConvergedDefaultCreate()

   Output Parameter:
.  reason - the convergence reason; it is positive if the iteration has converged,
            negative if the iteration has diverged, and KSP_CONVERGED_ITERATING otherwise

   Notes:
   KSPConvergedDefault() reaches convergence when   rnorm < MAX (rtol * rnorm_0, abstol);
   Divergence is detected if rnorm > dtol * rnorm_0, or when failures are detected throughout the iteration.
   By default, reaching the maximum number of iterations is considered divergence (i.e. KSP_DIVERGED_ITS).
   In order to have PETSc declaring convergence in such a case (i.e. KSP_CONVERGED_ITS), users can use KSPConvergedDefaultSetConvergedMaxits()

   where:
+     rtol - relative tolerance,
.     abstol - absolute tolerance.
.     dtol - divergence tolerance,
-     rnorm_0 - the two norm of the right hand side (or the preconditioned norm, depending on what was set with
          KSPSetNormType(). When initial guess is non-zero you
          can call KSPConvergedDefaultSetUIRNorm() to use the norm of (b - A*(initial guess))
          as the starting point for relative norm convergence testing, that is as rnorm_0

   Use KSPSetTolerances() to alter the defaults for rtol, abstol, dtol.

   Use KSPSetNormType() (or -ksp_norm_type <none,preconditioned,unpreconditioned,natural>) to change the norm used for computing rnorm

   The precise values of reason are macros such as KSP_CONVERGED_RTOL, which are defined in petscksp.h.

   This routine is used by KSP by default so the user generally never needs call it directly.

   Use KSPSetConvergenceTest() to provide your own test instead of using this one.

   Level: intermediate

.seealso: `KSPSetConvergenceTest()`, `KSPSetTolerances()`, `KSPConvergedSkip()`, `KSPConvergedReason`, `KSPGetConvergedReason()`,
          `KSPConvergedDefaultSetUIRNorm()`, `KSPConvergedDefaultSetUMIRNorm()`, `KSPConvergedDefaultSetConvergedMaxits()`, `KSPConvergedDefaultCreate()`, `KSPConvergedDefaultDestroy()`
@*/
PetscErrorCode  KSPConvergedDefault(KSP ksp,PetscInt n,PetscReal rnorm,KSPConvergedReason *reason,void *ctx)
{
  KSPConvergedDefaultCtx *cctx = (KSPConvergedDefaultCtx*) ctx;
  KSPNormType            normtype;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidLogicalCollectiveInt(ksp,n,2);
  PetscValidPointer(reason,4);
  PetscCheck(cctx,PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_NULL,"Convergence context must have been created with KSPConvergedDefaultCreate()");
  *reason = KSP_CONVERGED_ITERATING;

  if (cctx->convmaxits && n >= ksp->max_it) {
    *reason = KSP_CONVERGED_ITS;
    PetscCall(PetscInfo(ksp,"Linear solver has converged. Maximum number of iterations reached %" PetscInt_FMT "\n",n));
    PetscFunctionReturn(0);
  }
  PetscCall(KSPGetNormType(ksp,&normtype));
  if (normtype == KSP_NORM_NONE) PetscFunctionReturn(0);

  if (!n) {
    /* if user gives initial guess need to compute norm of b */
    if (!ksp->guess_zero && !cctx->initialrtol) {
      PetscReal snorm = 0.0;
      if (ksp->normtype == KSP_NORM_UNPRECONDITIONED || ksp->pc_side == PC_RIGHT) {
        PetscCall(PetscInfo(ksp,"user has provided nonzero initial guess, computing 2-norm of RHS\n"));
        PetscCall(VecNorm(ksp->vec_rhs,NORM_2,&snorm));        /*     <- b'*b */
      } else {
        Vec z;
        /* Should avoid allocating the z vector each time but cannot stash it in cctx because if KSPReset() is called the vector size might change */
        PetscCall(VecDuplicate(ksp->vec_rhs,&z));
        PetscCall(KSP_PCApply(ksp,ksp->vec_rhs,z));
        if (ksp->normtype == KSP_NORM_PRECONDITIONED) {
          PetscCall(PetscInfo(ksp,"user has provided nonzero initial guess, computing 2-norm of preconditioned RHS\n"));
          PetscCall(VecNorm(z,NORM_2,&snorm));                 /*    dp <- b'*B'*B*b */
        } else if (ksp->normtype == KSP_NORM_NATURAL) {
          PetscScalar norm;
          PetscCall(PetscInfo(ksp,"user has provided nonzero initial guess, computing natural norm of RHS\n"));
          PetscCall(VecDot(ksp->vec_rhs,z,&norm));
          snorm = PetscSqrtReal(PetscAbsScalar(norm));                            /*    dp <- b'*B*b */
        }
        PetscCall(VecDestroy(&z));
      }
      /* handle special case of zero RHS and nonzero guess */
      if (!snorm) {
        PetscCall(PetscInfo(ksp,"Special case, user has provided nonzero initial guess and zero RHS\n"));
        snorm = rnorm;
      }
      if (cctx->mininitialrtol) ksp->rnorm0 = PetscMin(snorm,rnorm);
      else ksp->rnorm0 = snorm;
    } else {
      ksp->rnorm0 = rnorm;
    }
    ksp->ttol = PetscMax(ksp->rtol*ksp->rnorm0,ksp->abstol);
  }

  if (n <= ksp->chknorm) PetscFunctionReturn(0);

  if (PetscIsInfOrNanReal(rnorm)) {
    PCFailedReason pcreason;
    PetscInt       sendbuf,recvbuf;
    PetscCall(PCGetFailedReasonRank(ksp->pc,&pcreason));
    sendbuf = (PetscInt)pcreason;
    PetscCallMPI(MPI_Allreduce(&sendbuf,&recvbuf,1,MPIU_INT,MPIU_MAX,PetscObjectComm((PetscObject)ksp)));
    if (recvbuf) {
      *reason = KSP_DIVERGED_PC_FAILED;
      PetscCall(PCSetFailedReason(ksp->pc,(PCFailedReason)recvbuf));
      PetscCall(PetscInfo(ksp,"Linear solver pcsetup fails, declaring divergence \n"));
    } else {
      *reason = KSP_DIVERGED_NANORINF;
      PetscCall(PetscInfo(ksp,"Linear solver has created a not a number (NaN) as the residual norm, declaring divergence \n"));
    }
  } else if (rnorm <= ksp->ttol) {
    if (rnorm < ksp->abstol) {
      PetscCall(PetscInfo(ksp,"Linear solver has converged. Residual norm %14.12e is less than absolute tolerance %14.12e at iteration %" PetscInt_FMT "\n",(double)rnorm,(double)ksp->abstol,n));
      *reason = KSP_CONVERGED_ATOL;
    } else {
      if (cctx->initialrtol) {
        PetscCall(PetscInfo(ksp,"Linear solver has converged. Residual norm %14.12e is less than relative tolerance %14.12e times initial residual norm %14.12e at iteration %" PetscInt_FMT "\n",(double)rnorm,(double)ksp->rtol,(double)ksp->rnorm0,n));
      } else {
        PetscCall(PetscInfo(ksp,"Linear solver has converged. Residual norm %14.12e is less than relative tolerance %14.12e times initial right hand side norm %14.12e at iteration %" PetscInt_FMT "\n",(double)rnorm,(double)ksp->rtol,(double)ksp->rnorm0,n));
      }
      *reason = KSP_CONVERGED_RTOL;
    }
  } else if (rnorm >= ksp->divtol*ksp->rnorm0) {
    PetscCall(PetscInfo(ksp,"Linear solver is diverging. Initial right hand size norm %14.12e, current residual norm %14.12e at iteration %" PetscInt_FMT "\n",(double)ksp->rnorm0,(double)rnorm,n));
    *reason = KSP_DIVERGED_DTOL;
  }
  PetscFunctionReturn(0);
}

/*@C
   KSPConvergedDefaultDestroy - Frees the space used by the KSPConvergedDefault() function context

   Not Collective

   Input Parameters:
.  ctx - convergence context

   Level: intermediate

.seealso: `KSPConvergedDefault()`, `KSPConvergedDefaultCreate()`, `KSPSetConvergenceTest()`, `KSPSetTolerances()`, `KSPConvergedSkip()`,
          `KSPConvergedReason`, `KSPGetConvergedReason()`, `KSPConvergedDefaultSetUIRNorm()`, `KSPConvergedDefaultSetUMIRNorm()`
@*/
PetscErrorCode  KSPConvergedDefaultDestroy(void *ctx)
{
  KSPConvergedDefaultCtx *cctx = (KSPConvergedDefaultCtx*) ctx;

  PetscFunctionBegin;
  PetscCall(VecDestroy(&cctx->work));
  PetscCall(PetscFree(ctx));
  PetscFunctionReturn(0);
}

/*
   KSPBuildSolutionDefault - Default code to create/move the solution.

   Collective on ksp

   Input Parameters:
+  ksp - iterative context
-  v   - pointer to the user's vector

   Output Parameter:
.  V - pointer to a vector containing the solution

   Level: advanced

   Developers Note: This is PETSC_EXTERN because it may be used by user written plugin KSP implementations

.seealso: `KSPGetSolution()`, `KSPBuildResidualDefault()`
*/
PetscErrorCode KSPBuildSolutionDefault(KSP ksp,Vec v,Vec *V)
{
  PetscFunctionBegin;
  if (ksp->pc_side == PC_RIGHT) {
    if (ksp->pc) {
      if (v) {
        PetscCall(KSP_PCApply(ksp,ksp->vec_sol,v)); *V = v;
      } else SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"Not working with right preconditioner");
    } else {
      if (v) {
        PetscCall(VecCopy(ksp->vec_sol,v)); *V = v;
      } else *V = ksp->vec_sol;
    }
  } else if (ksp->pc_side == PC_SYMMETRIC) {
    if (ksp->pc) {
      PetscCheck(!ksp->transpose_solve,PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"Not working with symmetric preconditioner and transpose solve");
      if (v) {
        PetscCall(PCApplySymmetricRight(ksp->pc,ksp->vec_sol,v));
        *V = v;
      } else SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"Not working with symmetric preconditioner");
    } else {
      if (v) {
        PetscCall(VecCopy(ksp->vec_sol,v)); *V = v;
      } else *V = ksp->vec_sol;
    }
  } else {
    if (v) {
      PetscCall(VecCopy(ksp->vec_sol,v)); *V = v;
    } else *V = ksp->vec_sol;
  }
  PetscFunctionReturn(0);
}

/*
   KSPBuildResidualDefault - Default code to compute the residual.

   Collecive on ksp

   Input Parameters:
.  ksp - iterative context
.  t   - pointer to temporary vector
.  v   - pointer to user vector

   Output Parameter:
.  V - pointer to a vector containing the residual

   Level: advanced

   Developers Note: This is PETSC_EXTERN because it may be used by user written plugin KSP implementations

.seealso: `KSPBuildSolutionDefault()`
*/
PetscErrorCode KSPBuildResidualDefault(KSP ksp,Vec t,Vec v,Vec *V)
{
  Mat            Amat,Pmat;

  PetscFunctionBegin;
  if (!ksp->pc) PetscCall(KSPGetPC(ksp,&ksp->pc));
  PetscCall(PCGetOperators(ksp->pc,&Amat,&Pmat));
  PetscCall(KSPBuildSolution(ksp,t,NULL));
  PetscCall(KSP_MatMult(ksp,Amat,t,v));
  PetscCall(VecAYPX(v,-1.0,ksp->vec_rhs));
  *V   = v;
  PetscFunctionReturn(0);
}

/*@C
  KSPCreateVecs - Gets a number of work vectors.

  Collective on ksp

  Input Parameters:
+ ksp  - iterative context
. rightn  - number of right work vectors
- leftn   - number of left work vectors to allocate

  Output Parameters:
+  right - the array of vectors created
-  left - the array of left vectors

   Note: The right vector has as many elements as the matrix has columns. The left
     vector has as many elements as the matrix has rows.

   The vectors are new vectors that are not owned by the KSP, they should be destroyed with calls to VecDestroyVecs() when no longer needed.

   Developers Note: First tries to duplicate the rhs and solution vectors of the KSP, if they do not exist tries to get them from the matrix, if
                    that does not exist tries to get them from the DM (if it is provided).

   Level: advanced

.seealso: `MatCreateVecs()`, `VecDestroyVecs()`

@*/
PetscErrorCode KSPCreateVecs(KSP ksp,PetscInt rightn, Vec **right,PetscInt leftn,Vec **left)
{
  Vec            vecr = NULL,vecl = NULL;
  PetscBool      matset,pmatset,isshell,preferdm = PETSC_FALSE;
  Mat            mat = NULL;

  PetscFunctionBegin;
  if (ksp->dm) {
    PetscCall(PetscObjectTypeCompare((PetscObject) ksp->dm, DMSHELL, &isshell));
    preferdm = isshell ? PETSC_FALSE : PETSC_TRUE;
  }
  if (rightn) {
    PetscCheck(right,PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_INCOMP,"You asked for right vectors but did not pass a pointer to hold them");
    if (ksp->vec_sol) vecr = ksp->vec_sol;
    else {
      if (preferdm) {
        PetscCall(DMGetGlobalVector(ksp->dm,&vecr));
      } else if (ksp->pc) {
        PetscCall(PCGetOperatorsSet(ksp->pc,&matset,&pmatset));
        /* check for mat before pmat because for KSPLSQR pmat may be a different size than mat since pmat maybe mat'*mat */
        if (matset) {
          PetscCall(PCGetOperators(ksp->pc,&mat,NULL));
          PetscCall(MatCreateVecs(mat,&vecr,NULL));
        } else if (pmatset) {
          PetscCall(PCGetOperators(ksp->pc,NULL,&mat));
          PetscCall(MatCreateVecs(mat,&vecr,NULL));
        }
      }
      if (!vecr && ksp->dm) {
        PetscCall(DMGetGlobalVector(ksp->dm,&vecr));
      }
      PetscCheck(vecr,PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_WRONGSTATE,"You requested a vector from a KSP that cannot provide one");
    }
    PetscCall(VecDuplicateVecs(vecr,rightn,right));
    if (!ksp->vec_sol) {
      if (preferdm) {
        PetscCall(DMRestoreGlobalVector(ksp->dm,&vecr));
      } else if (mat) {
        PetscCall(VecDestroy(&vecr));
      } else if (ksp->dm) {
        PetscCall(DMRestoreGlobalVector(ksp->dm,&vecr));
      }
    }
  }
  if (leftn) {
    PetscCheck(left,PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_INCOMP,"You asked for left vectors but did not pass a pointer to hold them");
    if (ksp->vec_rhs) vecl = ksp->vec_rhs;
    else {
      if (preferdm) {
        PetscCall(DMGetGlobalVector(ksp->dm,&vecl));
      } else if (ksp->pc) {
        PetscCall(PCGetOperatorsSet(ksp->pc,&matset,&pmatset));
        /* check for mat before pmat because for KSPLSQR pmat may be a different size than mat since pmat maybe mat'*mat */
        if (matset) {
          PetscCall(PCGetOperators(ksp->pc,&mat,NULL));
          PetscCall(MatCreateVecs(mat,NULL,&vecl));
        } else if (pmatset) {
          PetscCall(PCGetOperators(ksp->pc,NULL,&mat));
          PetscCall(MatCreateVecs(mat,NULL,&vecl));
        }
      }
      if (!vecl && ksp->dm) PetscCall(DMGetGlobalVector(ksp->dm,&vecl));
      PetscCheck(vecl,PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_WRONGSTATE,"You requested a vector from a KSP that cannot provide one");
    }
    PetscCall(VecDuplicateVecs(vecl,leftn,left));
    if (!ksp->vec_rhs) {
      if (preferdm) {
        PetscCall(DMRestoreGlobalVector(ksp->dm,&vecl));
      } else if (mat) {
        PetscCall(VecDestroy(&vecl));
      } else if (ksp->dm) {
        PetscCall(DMRestoreGlobalVector(ksp->dm,&vecl));
      }
    }
  }
  PetscFunctionReturn(0);
}

/*@C
  KSPSetWorkVecs - Sets a number of work vectors into a KSP object

  Collective on ksp

  Input Parameters:
+ ksp  - iterative context
- nw   - number of work vectors to allocate

  Level: developer

  Developers Note: This is PETSC_EXTERN because it may be used by user written plugin KSP implementations

@*/
PetscErrorCode KSPSetWorkVecs(KSP ksp,PetscInt nw)
{
  PetscFunctionBegin;
  PetscCall(VecDestroyVecs(ksp->nwork,&ksp->work));
  ksp->nwork = nw;
  PetscCall(KSPCreateVecs(ksp,nw,&ksp->work,0,NULL));
  PetscCall(PetscLogObjectParents(ksp,nw,ksp->work));
  PetscFunctionReturn(0);
}

/*
  KSPDestroyDefault - Destroys a iterative context variable for methods with
  no separate context.  Preferred calling sequence KSPDestroy().

  Input Parameter:
. ksp - the iterative context

   Developers Note: This is PETSC_EXTERN because it may be used by user written plugin KSP implementations

*/
PetscErrorCode KSPDestroyDefault(KSP ksp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscCall(PetscFree(ksp->data));
  PetscFunctionReturn(0);
}

/*@
   KSPGetConvergedReason - Gets the reason the KSP iteration was stopped.

   Not Collective

   Input Parameter:
.  ksp - the KSP context

   Output Parameter:
.  reason - negative value indicates diverged, positive value converged, see KSPConvergedReason

   Possible values for reason: See also manual page for each reason
$  KSP_CONVERGED_RTOL (residual 2-norm decreased by a factor of rtol, from 2-norm of right hand side)
$  KSP_CONVERGED_ATOL (residual 2-norm less than abstol)
$  KSP_CONVERGED_ITS (used by the preonly preconditioner that always uses ONE iteration, or when the KSPConvergedSkip() convergence test routine is set.
$  KSP_CONVERGED_CG_NEG_CURVE (see note below)
$  KSP_CONVERGED_CG_CONSTRAINED (see note below)
$  KSP_CONVERGED_STEP_LENGTH (see note below)
$  KSP_CONVERGED_ITERATING (returned if the solver is not yet finished)
$  KSP_DIVERGED_ITS  (required more than its to reach convergence)
$  KSP_DIVERGED_DTOL (residual norm increased by a factor of divtol)
$  KSP_DIVERGED_NANORINF (residual norm became Not-a-number or Inf likely due to 0/0)
$  KSP_DIVERGED_BREAKDOWN (generic breakdown in method)
$  KSP_DIVERGED_BREAKDOWN_BICG (Initial residual is orthogonal to preconditioned initial residual. Try a different preconditioner, or a different initial Level.)

   Options Database:
.   -ksp_converged_reason - prints the reason to standard out

   Notes:
    If this routine is called before or doing the KSPSolve() the value of KSP_CONVERGED_ITERATING is returned

   The values  KSP_CONVERGED_CG_NEG_CURVE, KSP_CONVERGED_CG_CONSTRAINED, and KSP_CONVERGED_STEP_LENGTH are returned only by the special KSPNASH, KSPSTCG, and KSPGLTR
   solvers which are used by the SNESNEWTONTR (trust region) solver.

   Level: intermediate

.seealso: `KSPSetConvergenceTest()`, `KSPConvergedDefault()`, `KSPSetTolerances()`, `KSPConvergedReason`,
          `KSPConvergedReasonView()`
@*/
PetscErrorCode  KSPGetConvergedReason(KSP ksp,KSPConvergedReason *reason)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidPointer(reason,2);
  *reason = ksp->reason;
  PetscFunctionReturn(0);
}

/*@C
   KSPGetConvergedReasonString - Return a human readable string for ksp converged reason

   Not Collective

   Input Parameter:
.  ksp - the KSP context

   Output Parameter:
.  strreason - a human readable string that describes ksp converged reason

   Level: beginner

.seealso: `KSPGetConvergedReason()`
@*/
PetscErrorCode KSPGetConvergedReasonString(KSP ksp,const char** strreason)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidPointer(strreason,2);
  *strreason = KSPConvergedReasons[ksp->reason];
  PetscFunctionReturn(0);
}

#include <petsc/private/dmimpl.h>
/*@
   KSPSetDM - Sets the DM that may be used by some preconditioners

   Logically Collective on ksp

   Input Parameters:
+  ksp - the preconditioner context
-  dm - the dm, cannot be NULL

   Notes:
   If this is used then the KSP will attempt to use the DM to create the matrix and use the routine set with
   DMKSPSetComputeOperators(). Use KSPSetDMActive(ksp,PETSC_FALSE) to instead use the matrix you've provided with
   KSPSetOperators().

   A DM can only be used for solving one problem at a time because information about the problem is stored on the DM,
   even when not using interfaces like DMKSPSetComputeOperators().  Use DMClone() to get a distinct DM when solving
   different problems using the same function space.

   Level: intermediate

.seealso: `KSPGetDM()`, `KSPSetDMActive()`, `KSPSetComputeOperators()`, `KSPSetComputeRHS()`, `KSPSetComputeInitialGuess()`, `DMKSPSetComputeOperators()`, `DMKSPSetComputeRHS()`, `DMKSPSetComputeInitialGuess()`
@*/
PetscErrorCode  KSPSetDM(KSP ksp,DM dm)
{
  PC             pc;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidHeaderSpecific(dm,DM_CLASSID,2);
  PetscCall(PetscObjectReference((PetscObject)dm));
  if (ksp->dm) {                /* Move the DMSNES context over to the new DM unless the new DM already has one */
    if (ksp->dm->dmksp && !dm->dmksp) {
      DMKSP kdm;
      PetscCall(DMCopyDMKSP(ksp->dm,dm));
      PetscCall(DMGetDMKSP(ksp->dm,&kdm));
      if (kdm->originaldm == ksp->dm) kdm->originaldm = dm; /* Grant write privileges to the replacement DM */
    }
    PetscCall(DMDestroy(&ksp->dm));
  }
  ksp->dm       = dm;
  ksp->dmAuto   = PETSC_FALSE;
  PetscCall(KSPGetPC(ksp,&pc));
  PetscCall(PCSetDM(pc,dm));
  ksp->dmActive = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@
   KSPSetDMActive - Indicates the DM should be used to generate the linear system matrix and right hand side

   Logically Collective on ksp

   Input Parameters:
+  ksp - the preconditioner context
-  flg - use the DM

   Notes:
   By default KSPSetDM() sets the DM as active, call KSPSetDMActive(ksp,PETSC_FALSE); after KSPSetDM(ksp,dm) to not have the KSP object use the DM to generate the matrices.

   Level: intermediate

.seealso: `KSPGetDM()`, `KSPSetDM()`, `SNESSetDM()`, `KSPSetComputeOperators()`, `KSPSetComputeRHS()`, `KSPSetComputeInitialGuess()`
@*/
PetscErrorCode  KSPSetDMActive(KSP ksp,PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidLogicalCollectiveBool(ksp,flg,2);
  ksp->dmActive = flg;
  PetscFunctionReturn(0);
}

/*@
   KSPGetDM - Gets the DM that may be used by some preconditioners

   Not Collective

   Input Parameter:
. ksp - the preconditioner context

   Output Parameter:
.  dm - the dm

   Level: intermediate

.seealso: `KSPSetDM()`, `KSPSetDMActive()`
@*/
PetscErrorCode  KSPGetDM(KSP ksp,DM *dm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  if (!ksp->dm) {
    PetscCall(DMShellCreate(PetscObjectComm((PetscObject)ksp),&ksp->dm));
    ksp->dmAuto = PETSC_TRUE;
  }
  *dm = ksp->dm;
  PetscFunctionReturn(0);
}

/*@
   KSPSetApplicationContext - Sets the optional user-defined context for the linear solver.

   Logically Collective on ksp

   Input Parameters:
+  ksp - the KSP context
-  usrP - optional user context

   Fortran Notes:
    To use this from Fortran you must write a Fortran interface definition for this
    function that tells Fortran the Fortran derived data type that you are passing in as the ctx argument.

   Level: intermediate

.seealso: `KSPGetApplicationContext()`
@*/
PetscErrorCode  KSPSetApplicationContext(KSP ksp,void *usrP)
{
  PC             pc;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  ksp->user = usrP;
  PetscCall(KSPGetPC(ksp,&pc));
  PetscCall(PCSetApplicationContext(pc,usrP));
  PetscFunctionReturn(0);
}

/*@
   KSPGetApplicationContext - Gets the user-defined context for the linear solver.

   Not Collective

   Input Parameter:
.  ksp - KSP context

   Output Parameter:
.  usrP - user context

   Fortran Notes:
    To use this from Fortran you must write a Fortran interface definition for this
    function that tells Fortran the Fortran derived data type that you are passing in as the ctx argument.

   Level: intermediate

.seealso: `KSPSetApplicationContext()`
@*/
PetscErrorCode  KSPGetApplicationContext(KSP ksp,void *usrP)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  *(void**)usrP = ksp->user;
  PetscFunctionReturn(0);
}

#include <petsc/private/pcimpl.h>

/*@
   KSPCheckSolve - Checks if the PCSetUp() or KSPSolve() failed and set the error flag for the outer PC. A KSP_DIVERGED_ITS is
         not considered a failure in this context

   Collective on ksp

   Input Parameters:
+  ksp - the linear solver (KSP) context.
.  pc - the preconditioner context
-  vec - a vector that will be initialized with Inf to indicate lack of convergence

   Notes: this may be called by a subset of the processes in the PC

   Level: developer

   Developer Note: this is used to manage returning from preconditioners whose inner KSP solvers have failed in some way

.seealso: `KSPCreate()`, `KSPSetType()`, `KSP`, `KSPCheckNorm()`, `KSPCheckDot()`
@*/
PetscErrorCode KSPCheckSolve(KSP ksp,PC pc,Vec vec)
{
  PCFailedReason pcreason;
  PC             subpc;

  PetscFunctionBegin;
  PetscCall(KSPGetPC(ksp,&subpc));
  PetscCall(PCGetFailedReason(subpc,&pcreason));
  if (pcreason || (ksp->reason < 0 && ksp->reason != KSP_DIVERGED_ITS)) {
    PetscCheck(!pc->erroriffailure,PETSC_COMM_SELF,PETSC_ERR_NOT_CONVERGED,"Detected not converged in KSP inner solve: KSP reason %s PC reason %s",KSPConvergedReasons[ksp->reason],PCFailedReasons[pcreason]);
    else {
      PetscCall(PetscInfo(ksp,"Detected not converged in KSP inner solve: KSP reason %s PC reason %s\n",KSPConvergedReasons[ksp->reason],PCFailedReasons[pcreason]));
      pc->failedreason = PC_SUBPC_ERROR;
      if (vec) PetscCall(VecSetInf(vec));
    }
  }
  PetscFunctionReturn(0);
}
