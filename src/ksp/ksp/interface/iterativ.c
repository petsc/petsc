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

.seealso: KSPBuildResidual()
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
.seealso: KSPBuildResidual(), KSPGetResidualNorm(), KSPGetTotalIterations()
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

.seealso: KSPBuildResidual(), KSPGetResidualNorm(), KSPGetIterationNumber()
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

  Level: intermediate

.seealso: KSPMonitorSet(), KSPMonitorTrueResidual()
@*/
PetscErrorCode KSPMonitorResidual(KSP ksp, PetscInt n, PetscReal rnorm, PetscViewerAndFormat *vf)
{
  PetscViewer       viewer = vf->viewer;
  PetscViewerFormat format = vf->format;
  PetscInt          tablevel;
  const char       *prefix;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 4);
  ierr = PetscObjectGetTabLevel((PetscObject) ksp, &tablevel);CHKERRQ(ierr);
  ierr = PetscObjectGetOptionsPrefix((PetscObject) ksp, &prefix);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewer, format);CHKERRQ(ierr);
  ierr = PetscViewerASCIIAddTab(viewer, tablevel);CHKERRQ(ierr);
  if (n == 0 && prefix) {ierr = PetscViewerASCIIPrintf(viewer, "  Residual norms for %s solve.\n", prefix);CHKERRQ(ierr);}
  ierr = PetscViewerASCIIPrintf(viewer, "%3D KSP Residual norm %14.12e \n", n, (double) rnorm);CHKERRQ(ierr);
  ierr = PetscViewerASCIISubtractTab(viewer, tablevel);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
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

  Level: intermediate

.seealso: KSPMonitorSet(), KSPMonitorTrueResidual()
@*/
PetscErrorCode KSPMonitorResidualDraw(KSP ksp, PetscInt n, PetscReal rnorm, PetscViewerAndFormat *vf)
{
  PetscViewer       viewer = vf->viewer;
  PetscViewerFormat format = vf->format;
  Vec               r;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 4);
  ierr = PetscViewerPushFormat(viewer, format);CHKERRQ(ierr);
  ierr = KSPBuildResidual(ksp, NULL, NULL, &r);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) r, "Residual");CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject) r, "__Vec_bc_zero__", (PetscObject) ksp);CHKERRQ(ierr);
  ierr = VecView(r, viewer);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject) r, "__Vec_bc_zero__", NULL);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
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

  Level: intermediate

.seealso: KSPMonitorSet(), KSPMonitorTrueResidual()
@*/
PetscErrorCode KSPMonitorResidualDrawLG(KSP ksp, PetscInt n, PetscReal rnorm, PetscViewerAndFormat *vf)
{
  PetscViewer        viewer = vf->viewer;
  PetscViewerFormat  format = vf->format;
  PetscDrawLG        lg     = vf->lg;
  KSPConvergedReason reason;
  PetscReal          x, y;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 4);
  PetscValidHeaderSpecific(lg, PETSC_DRAWLG_CLASSID, 5);
  ierr = PetscViewerPushFormat(viewer, format);CHKERRQ(ierr);
  if (!n) {ierr = PetscDrawLGReset(lg);CHKERRQ(ierr);}
  x = (PetscReal) n;
  if (rnorm > 0.0) y = PetscLog10Real(rnorm);
  else y = -15.0;
  ierr = PetscDrawLGAddPoint(lg, &x, &y);CHKERRQ(ierr);
  ierr = KSPGetConvergedReason(ksp, &reason);CHKERRQ(ierr);
  if (n <= 20 || !(n % 5) || reason) {
    ierr = PetscDrawLGDraw(lg);CHKERRQ(ierr);
    ierr = PetscDrawLGSave(lg);CHKERRQ(ierr);
  }
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
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

.seealso: KSPMonitorSet(), KSPMonitorTrueResidual()
@*/
PetscErrorCode KSPMonitorResidualDrawLGCreate(PetscViewer viewer, PetscViewerFormat format, void *ctx, PetscViewerAndFormat **vf)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerAndFormatCreate(viewer, format, vf);CHKERRQ(ierr);
  (*vf)->data = ctx;
  ierr = KSPMonitorLGCreate(PetscObjectComm((PetscObject) viewer), NULL, NULL, "Log Residual Norm", 1, NULL, PETSC_DECIDE, PETSC_DECIDE, 400, 300, &(*vf)->lg);CHKERRQ(ierr);
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
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 4);
  ierr = PetscObjectGetTabLevel((PetscObject) ksp, &tablevel);CHKERRQ(ierr);
  ierr = PetscObjectGetOptionsPrefix((PetscObject) ksp, &prefix);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewer, format);CHKERRQ(ierr);
  ierr = PetscViewerASCIIAddTab(viewer, tablevel);CHKERRQ(ierr);
  if (its == 0 && prefix)  {ierr = PetscViewerASCIIPrintf(viewer, "  Residual norms for %s solve.\n", prefix);CHKERRQ(ierr);}
  if (fnorm > 1.e-9)       {ierr = PetscViewerASCIIPrintf(viewer, "%3D KSP Residual norm %g \n", its, (double) fnorm);CHKERRQ(ierr);}
  else if (fnorm > 1.e-11) {ierr = PetscViewerASCIIPrintf(viewer, "%3D KSP Residual norm %5.3e \n", its, (double) fnorm);CHKERRQ(ierr);}
  else                     {ierr = PetscViewerASCIIPrintf(viewer, "%3D KSP Residual norm < 1.e-11\n", its);CHKERRQ(ierr);}
  ierr = PetscViewerASCIISubtractTab(viewer, tablevel);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode KSPMonitorRange_Private(KSP ksp, PetscInt it, PetscReal *per)
{
  Vec                resid;
  const PetscScalar *r;
  PetscReal          rmax, pwork;
  PetscInt           i, n, N;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = KSPBuildResidual(ksp, NULL, NULL, &resid);CHKERRQ(ierr);
  ierr = VecNorm(resid, NORM_INFINITY, &rmax);CHKERRQ(ierr);
  ierr = VecGetLocalSize(resid, &n);CHKERRQ(ierr);
  ierr = VecGetSize(resid, &N);CHKERRQ(ierr);
  ierr = VecGetArrayRead(resid, &r);CHKERRQ(ierr);
  pwork = 0.0;
  for (i = 0; i < n; ++i) pwork += (PetscAbsScalar(r[i]) > .20*rmax);
  ierr = VecRestoreArrayRead(resid, &r);CHKERRQ(ierr);
  ierr = VecDestroy(&resid);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&pwork, per, 1, MPIU_REAL, MPIU_SUM, PetscObjectComm((PetscObject) ksp));CHKERRQ(ierr);
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

  Level: intermediate

.seealso: KSPMonitorSet(), KSPMonitorResidual()
@*/
PetscErrorCode KSPMonitorResidualRange(KSP ksp, PetscInt it, PetscReal rnorm, PetscViewerAndFormat *vf)
{
  static PetscReal  prev;
  PetscViewer       viewer = vf->viewer;
  PetscViewerFormat format = vf->format;
  PetscInt          tablevel;
  const char       *prefix;
  PetscReal         perc, rel;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 4);
  ierr = PetscObjectGetTabLevel((PetscObject) ksp, &tablevel);CHKERRQ(ierr);
  ierr = PetscObjectGetOptionsPrefix((PetscObject) ksp, &prefix);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewer, format);CHKERRQ(ierr);
  ierr = PetscViewerASCIIAddTab(viewer, tablevel);CHKERRQ(ierr);
  if (!it) prev = rnorm;
  if (it == 0 && prefix) {ierr = PetscViewerASCIIPrintf(viewer, "  Residual norms for %s solve.\n", prefix);CHKERRQ(ierr);}
  ierr = KSPMonitorRange_Private(ksp, it, &perc);CHKERRQ(ierr);
  rel  = (prev - rnorm)/prev;
  prev = rnorm;
  ierr = PetscViewerASCIIPrintf(viewer, "%3D KSP preconditioned resid norm %14.12e Percent values above 20 percent of maximum %5.2f relative decrease %5.2e ratio %5.2e \n", it, (double) rnorm, (double) (100.0*perc), (double) rel, (double) (rel/perc));CHKERRQ(ierr);
  ierr = PetscViewerASCIISubtractTab(viewer, tablevel);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
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

  Level: intermediate

.seealso: KSPMonitorSet(), KSPMonitorResidual(),KSPMonitorTrueResidualMaxNorm()
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
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 4);
  ierr = PetscObjectGetTabLevel((PetscObject) ksp, &tablevel);CHKERRQ(ierr);
  ierr = PetscObjectGetOptionsPrefix((PetscObject) ksp, &prefix);CHKERRQ(ierr);
  ierr = PetscStrncpy(normtype, KSPNormTypes[ksp->normtype], sizeof(normtype));CHKERRQ(ierr);
  ierr = PetscStrtolower(normtype);CHKERRQ(ierr);
  ierr = KSPBuildResidual(ksp, NULL, NULL, &r);CHKERRQ(ierr);
  ierr = VecNorm(r, NORM_2, &truenorm);CHKERRQ(ierr);
  ierr = VecNorm(ksp->vec_rhs, NORM_2, &bnorm);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);

  ierr = PetscViewerPushFormat(viewer, format);CHKERRQ(ierr);
  ierr = PetscViewerASCIIAddTab(viewer, tablevel);CHKERRQ(ierr);
  if (n == 0 && prefix) {ierr = PetscViewerASCIIPrintf(viewer, "  Residual norms for %s solve.\n", prefix);CHKERRQ(ierr);}
  ierr = PetscViewerASCIIPrintf(viewer, "%3D KSP %s resid norm %14.12e true resid norm %14.12e ||r(i)||/||b|| %14.12e\n", n, normtype, (double) rnorm, (double) truenorm, (double) (truenorm/bnorm));CHKERRQ(ierr);
  ierr = PetscViewerASCIISubtractTab(viewer, tablevel);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
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

  Level: intermediate

.seealso: KSPMonitorSet(), KSPMonitorTrueResidual()
@*/
PetscErrorCode KSPMonitorTrueResidualDraw(KSP ksp, PetscInt n, PetscReal rnorm, PetscViewerAndFormat *vf)
{
  PetscViewer       viewer = vf->viewer;
  PetscViewerFormat format = vf->format;
  Vec               r;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 4);
  ierr = PetscViewerPushFormat(viewer, format);CHKERRQ(ierr);
  ierr = KSPBuildResidual(ksp, NULL, NULL, &r);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) r, "Residual");CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject) r, "__Vec_bc_zero__", (PetscObject) ksp);CHKERRQ(ierr);
  ierr = VecView(r, viewer);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject) r, "__Vec_bc_zero__", NULL);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
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

  Level: intermediate

.seealso: KSPMonitorSet(), KSPMonitorTrueResidual()
@*/
PetscErrorCode KSPMonitorTrueResidualDrawLG(KSP ksp, PetscInt n, PetscReal rnorm, PetscViewerAndFormat *vf)
{
  PetscViewer        viewer = vf->viewer;
  PetscViewerFormat  format = vf->format;
  PetscDrawLG        lg     = vf->lg;
  Vec                r;
  KSPConvergedReason reason;
  PetscReal          truenorm, x[2], y[2];
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 4);
  PetscValidHeaderSpecific(lg, PETSC_DRAWLG_CLASSID, 5);
  ierr = KSPBuildResidual(ksp, NULL, NULL, &r);CHKERRQ(ierr);
  ierr = VecNorm(r, NORM_2, &truenorm);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewer, format);CHKERRQ(ierr);
  if (!n) {ierr = PetscDrawLGReset(lg);CHKERRQ(ierr);}
  x[0] = (PetscReal) n;
  if (rnorm > 0.0) y[0] = PetscLog10Real(rnorm);
  else y[0] = -15.0;
  x[1] = (PetscReal) n;
  if (truenorm > 0.0) y[1] = PetscLog10Real(truenorm);
  else y[1] = -15.0;
  ierr = PetscDrawLGAddPoint(lg, x, y);CHKERRQ(ierr);
  ierr = KSPGetConvergedReason(ksp, &reason);CHKERRQ(ierr);
  if (n <= 20 || !(n % 5) || reason) {
    ierr = PetscDrawLGDraw(lg);CHKERRQ(ierr);
    ierr = PetscDrawLGSave(lg);CHKERRQ(ierr);
  }
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
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

  Level: intermediate

.seealso: KSPMonitorSet(), KSPMonitorTrueResidual()
@*/
PetscErrorCode KSPMonitorTrueResidualDrawLGCreate(PetscViewer viewer, PetscViewerFormat format, void *ctx, PetscViewerAndFormat **vf)
{
  const char    *names[] = {"preconditioned", "true"};
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerAndFormatCreate(viewer, format, vf);CHKERRQ(ierr);
  (*vf)->data = ctx;
  ierr = KSPMonitorLGCreate(PetscObjectComm((PetscObject) viewer), NULL, NULL, "Log Residual Norm", 2, names, PETSC_DECIDE, PETSC_DECIDE, 400, 300, &(*vf)->lg);CHKERRQ(ierr);
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

  Level: intermediate

.seealso: KSPMonitorSet(), KSPMonitorResidual(),KSPMonitorTrueResidualMaxNorm()
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
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 4);
  ierr = PetscObjectGetTabLevel((PetscObject) ksp, &tablevel);CHKERRQ(ierr);
  ierr = PetscObjectGetOptionsPrefix((PetscObject) ksp, &prefix);CHKERRQ(ierr);
  ierr = PetscStrncpy(normtype, KSPNormTypes[ksp->normtype], sizeof(normtype));CHKERRQ(ierr);
  ierr = PetscStrtolower(normtype);CHKERRQ(ierr);
  ierr = KSPBuildResidual(ksp, NULL, NULL, &r);CHKERRQ(ierr);
  ierr = VecNorm(r, NORM_INFINITY, &truenorm);CHKERRQ(ierr);
  ierr = VecNorm(ksp->vec_rhs, NORM_INFINITY, &bnorm);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);

  ierr = PetscViewerPushFormat(viewer, format);CHKERRQ(ierr);
  ierr = PetscViewerASCIIAddTab(viewer, tablevel);CHKERRQ(ierr);
  if (n == 0 && prefix) {ierr = PetscViewerASCIIPrintf(viewer, "  Residual norms for %s solve.\n", prefix);CHKERRQ(ierr);}
  ierr = PetscViewerASCIIPrintf(viewer, "%3D KSP %s true resid max norm %14.12e ||r(i)||/||b|| %14.12e\n", n, normtype, (double) truenorm, (double) (truenorm/bnorm));CHKERRQ(ierr);
  ierr = PetscViewerASCIISubtractTab(viewer, tablevel);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
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

  Level: intermediate

.seealso: KSPMonitorSet(), KSPMonitorResidual(),KSPMonitorTrueResidualMaxNorm()
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
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 4);
  ierr = PetscObjectGetTabLevel((PetscObject) ksp, &tablevel);CHKERRQ(ierr);
  ierr = PetscObjectGetOptionsPrefix((PetscObject) ksp, &prefix);CHKERRQ(ierr);
  ierr = KSPGetDM(ksp, &dm);CHKERRQ(ierr);
  ierr = DMGetNumFields(dm, &Nf);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dm, &sol);CHKERRQ(ierr);
  ierr = KSPBuildSolution(ksp, sol, NULL);CHKERRQ(ierr);
  /* TODO: Make a different monitor that flips sign for SNES, Newton system is A dx = -b, so we need to negate the solution */
  ierr = VecScale(sol, -1.0);CHKERRQ(ierr);
  ierr = PetscCalloc1(Nf, &errors);CHKERRQ(ierr);
  ierr = DMComputeError(dm, sol, errors, NULL);CHKERRQ(ierr);

  ierr = PetscViewerPushFormat(viewer, format);CHKERRQ(ierr);
  ierr = PetscViewerASCIIAddTab(viewer, tablevel);CHKERRQ(ierr);
  if (n == 0 && prefix) {ierr = PetscViewerASCIIPrintf(viewer, "  Error norms for %s solve.\n", prefix);CHKERRQ(ierr);}
  ierr = PetscViewerASCIIPrintf(viewer, "%3D KSP Error norm %s", n, Nf > 1 ? "[" : "");CHKERRQ(ierr);
  ierr = PetscViewerASCIIUseTabs(viewer, PETSC_FALSE);CHKERRQ(ierr);
  for (f = 0; f < Nf; ++f) {
    if (f > 0) {ierr = PetscViewerASCIIPrintf(viewer, ", ");CHKERRQ(ierr);}
    ierr = PetscViewerASCIIPrintf(viewer, "%14.12e", (double) errors[f]);CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPrintf(viewer, "%s resid norm %14.12e\n", Nf > 1 ? "]" : "", (double) rnorm);CHKERRQ(ierr);
  ierr = PetscViewerASCIIUseTabs(viewer, PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscViewerASCIISubtractTab(viewer, tablevel);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm, &sol);CHKERRQ(ierr);
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

  Level: intermediate

.seealso: KSPMonitorSet(), KSPMonitorTrueResidual()
@*/
PetscErrorCode KSPMonitorErrorDraw(KSP ksp, PetscInt n, PetscReal rnorm, PetscViewerAndFormat *vf)
{
  PetscViewer       viewer = vf->viewer;
  PetscViewerFormat format = vf->format;
  DM                dm;
  Vec               sol, e;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 4);
  ierr = PetscViewerPushFormat(viewer, format);CHKERRQ(ierr);
  ierr = KSPGetDM(ksp, &dm);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dm, &sol);CHKERRQ(ierr);
  ierr = KSPBuildSolution(ksp, sol, NULL);CHKERRQ(ierr);
  ierr = DMComputeError(dm, sol, NULL, &e);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) e, "Error");CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject) e, "__Vec_bc_zero__", (PetscObject) ksp);CHKERRQ(ierr);
  ierr = VecView(e, viewer);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject) e, "__Vec_bc_zero__", NULL);CHKERRQ(ierr);
  ierr = VecDestroy(&e);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm, &sol);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
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

  Level: intermediate

.seealso: KSPMonitorSet(), KSPMonitorTrueResidual()
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
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 4);
  PetscValidHeaderSpecific(lg, PETSC_DRAWLG_CLASSID, 5);
  ierr = KSPGetDM(ksp, &dm);CHKERRQ(ierr);
  ierr = DMGetNumFields(dm, &Nf);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dm, &sol);CHKERRQ(ierr);
  ierr = KSPBuildSolution(ksp, sol, NULL);CHKERRQ(ierr);
  /* TODO: Make a different monitor that flips sign for SNES, Newton system is A dx = -b, so we need to negate the solution */
  ierr = VecScale(sol, -1.0);CHKERRQ(ierr);
  ierr = PetscCalloc2(Nf+1, &x, Nf+1, &errors);CHKERRQ(ierr);
  ierr = DMComputeError(dm, sol, errors, NULL);CHKERRQ(ierr);

  ierr = PetscViewerPushFormat(viewer, format);CHKERRQ(ierr);
  if (!n) {ierr = PetscDrawLGReset(lg);CHKERRQ(ierr);}
  for (f = 0; f < Nf; ++f) {
    x[f]      = (PetscReal) n;
    errors[f] = errors[f] > 0.0 ? PetscLog10Real(errors[f]) : -15.;
  }
  x[Nf]      = (PetscReal) n;
  errors[Nf] = rnorm > 0.0 ? PetscLog10Real(rnorm) : -15.;
  ierr = PetscDrawLGAddPoint(lg, x, errors);CHKERRQ(ierr);
  ierr = KSPGetConvergedReason(ksp, &reason);CHKERRQ(ierr);
  if (n <= 20 || !(n % 5) || reason) {
    ierr = PetscDrawLGDraw(lg);CHKERRQ(ierr);
    ierr = PetscDrawLGSave(lg);CHKERRQ(ierr);
  }
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
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

  Level: intermediate

.seealso: KSPMonitorSet(), KSPMonitorTrueResidual()
@*/
PetscErrorCode KSPMonitorErrorDrawLGCreate(PetscViewer viewer, PetscViewerFormat format, void *ctx, PetscViewerAndFormat **vf)
{
  KSP            ksp = (KSP) ctx;
  DM             dm;
  char         **names;
  PetscInt       Nf, f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPGetDM(ksp, &dm);CHKERRQ(ierr);
  ierr = DMGetNumFields(dm, &Nf);CHKERRQ(ierr);
  ierr = PetscMalloc1(Nf+1, &names);CHKERRQ(ierr);
  for (f = 0; f < Nf; ++f) {
    PetscObject disc;
    const char *fname;
    char        lname[PETSC_MAX_PATH_LEN];

    ierr = DMGetField(dm, f, NULL, &disc);CHKERRQ(ierr);
    ierr = PetscObjectGetName(disc, &fname);CHKERRQ(ierr);
    ierr = PetscStrncpy(lname, fname, PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
    ierr = PetscStrlcat(lname, " Error", PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
    ierr = PetscStrallocpy(lname, &names[f]);CHKERRQ(ierr);
  }
  ierr = PetscStrallocpy("residual", &names[Nf]);CHKERRQ(ierr);
  ierr = PetscViewerAndFormatCreate(viewer, format, vf);CHKERRQ(ierr);
  (*vf)->data = ctx;
  ierr = KSPMonitorLGCreate(PetscObjectComm((PetscObject) viewer), NULL, NULL, "Log Error Norm", Nf+1, (const char **) names, PETSC_DECIDE, PETSC_DECIDE, 400, 300, &(*vf)->lg);CHKERRQ(ierr);
  for (f = 0; f <= Nf; ++f) {ierr = PetscFree(names[f]);CHKERRQ(ierr);}
  ierr = PetscFree(names);CHKERRQ(ierr);
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

  Level: intermediate

.seealso: KSPMonitorSet(), KSPMonitorTrueResidual()
@*/
PetscErrorCode KSPMonitorSolution(KSP ksp, PetscInt n, PetscReal rnorm, PetscViewerAndFormat *vf)
{
  PetscViewer       viewer = vf->viewer;
  PetscViewerFormat format = vf->format;
  Vec               x;
  PetscReal         snorm;
  PetscInt          tablevel;
  const char       *prefix;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 4);
  ierr = KSPBuildSolution(ksp, NULL, &x);CHKERRQ(ierr);
  ierr = VecNorm(x, NORM_2, &snorm);CHKERRQ(ierr);
  ierr = PetscObjectGetTabLevel((PetscObject) ksp, &tablevel);CHKERRQ(ierr);
  ierr = PetscObjectGetOptionsPrefix((PetscObject) ksp, &prefix);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewer, format);CHKERRQ(ierr);
  ierr = PetscViewerASCIIAddTab(viewer, tablevel);CHKERRQ(ierr);
  if (n == 0 && prefix) {ierr = PetscViewerASCIIPrintf(viewer, "  Solution norms for %s solve.\n", prefix);CHKERRQ(ierr);}
  ierr = PetscViewerASCIIPrintf(viewer, "%3D KSP Solution norm %14.12e \n", n, (double) snorm);CHKERRQ(ierr);
  ierr = PetscViewerASCIISubtractTab(viewer, tablevel);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
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

  Level: intermediate

.seealso: KSPMonitorSet(), KSPMonitorTrueResidual()
@*/
PetscErrorCode KSPMonitorSolutionDraw(KSP ksp, PetscInt n, PetscReal rnorm, PetscViewerAndFormat *vf)
{
  PetscViewer       viewer = vf->viewer;
  PetscViewerFormat format = vf->format;
  Vec               x;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 4);
  ierr = KSPBuildSolution(ksp, NULL, &x);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewer, format);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) x, "Solution");CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject) x, "__Vec_bc_zero__", (PetscObject) ksp);CHKERRQ(ierr);
  ierr = VecView(x, viewer);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject) x, "__Vec_bc_zero__", NULL);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
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

  Level: intermediate

.seealso: KSPMonitorSet(), KSPMonitorTrueResidual()
@*/
PetscErrorCode KSPMonitorSolutionDrawLG(KSP ksp, PetscInt n, PetscReal rnorm, PetscViewerAndFormat *vf)
{
  PetscViewer        viewer = vf->viewer;
  PetscViewerFormat  format = vf->format;
  PetscDrawLG        lg     = vf->lg;
  Vec                u;
  KSPConvergedReason reason;
  PetscReal          snorm, x, y;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 4);
  PetscValidHeaderSpecific(lg, PETSC_DRAWLG_CLASSID, 5);
  ierr = KSPBuildSolution(ksp, NULL, &u);CHKERRQ(ierr);
  ierr = VecNorm(u, NORM_2, &snorm);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewer, format);CHKERRQ(ierr);
  if (!n) {ierr = PetscDrawLGReset(lg);CHKERRQ(ierr);}
  x = (PetscReal) n;
  if (snorm > 0.0) y = PetscLog10Real(snorm);
  else y = -15.0;
  ierr = PetscDrawLGAddPoint(lg, &x, &y);CHKERRQ(ierr);
  ierr = KSPGetConvergedReason(ksp, &reason);CHKERRQ(ierr);
  if (n <= 20 || !(n % 5) || reason) {
    ierr = PetscDrawLGDraw(lg);CHKERRQ(ierr);
    ierr = PetscDrawLGSave(lg);CHKERRQ(ierr);
  }
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
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

  Level: intermediate

.seealso: KSPMonitorSet(), KSPMonitorTrueResidual()
@*/
PetscErrorCode KSPMonitorSolutionDrawLGCreate(PetscViewer viewer, PetscViewerFormat format, void *ctx, PetscViewerAndFormat **vf)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerAndFormatCreate(viewer, format, vf);CHKERRQ(ierr);
  (*vf)->data = ctx;
  ierr = KSPMonitorLGCreate(PetscObjectComm((PetscObject) viewer), NULL, NULL, "Log Solution Norm", 1, NULL, PETSC_DECIDE, PETSC_DECIDE, 400, 300, &(*vf)->lg);CHKERRQ(ierr);
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

  Level: intermediate

.seealso: KSPComputeExtremeSingularValues()
@*/
PetscErrorCode KSPMonitorSingularValue(KSP ksp, PetscInt n, PetscReal rnorm, PetscViewerAndFormat *vf)
{
  PetscViewer       viewer = vf->viewer;
  PetscViewerFormat format = vf->format;
  PetscReal         emin, emax;
  PetscInt          tablevel;
  const char       *prefix;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 4);
  ierr = PetscObjectGetTabLevel((PetscObject) ksp, &tablevel);CHKERRQ(ierr);
  ierr = PetscObjectGetOptionsPrefix((PetscObject) ksp, &prefix);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewer, format);CHKERRQ(ierr);
  ierr = PetscViewerASCIIAddTab(viewer, tablevel);CHKERRQ(ierr);
  if (n == 0 && prefix) {ierr = PetscViewerASCIIPrintf(viewer, "  Residual norms for %s solve.\n", prefix);CHKERRQ(ierr);}
  if (!ksp->calc_sings) {
    ierr = PetscViewerASCIIPrintf(viewer, "%3D KSP Residual norm %14.12e \n", n, (double) rnorm);CHKERRQ(ierr);
  } else {
    ierr = KSPComputeExtremeSingularValues(ksp, &emax, &emin);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "%3D KSP Residual norm %14.12e %% max %14.12e min %14.12e max/min %14.12e\n", n, (double) rnorm, (double) emax, (double) emin, (double) (emax/emin));CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIISubtractTab(viewer, tablevel);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
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

.seealso: KSPMonitorSet(), KSPMonitorSingularValue()
@*/
PetscErrorCode KSPMonitorSingularValueCreate(PetscViewer viewer, PetscViewerFormat format, void *ctx, PetscViewerAndFormat **vf)
{
  KSP            ksp = (KSP) ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerAndFormatCreate(viewer, format, vf);CHKERRQ(ierr);
  (*vf)->data = ctx;
  ierr = KSPSetComputeSingularValues(ksp, PETSC_TRUE);CHKERRQ(ierr);
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
   control the accuracy of the inner solves needed to gaurantee the
   convergence of the outer iterations.

   Level: advanced

.seealso: KSPMonitorDynamicToleranceDestroy()
@*/
PetscErrorCode KSPMonitorDynamicTolerance(KSP ksp,PetscInt its,PetscReal fnorm,void *dummy)
{
  PetscErrorCode ierr;
  PC             pc;
  PetscReal      outer_rtol, outer_abstol, outer_dtol, inner_rtol;
  PetscInt       outer_maxits,nksp,first,i;
  KSPDynTolCtx   *scale   = (KSPDynTolCtx*)dummy;
  KSP            *subksp = NULL;
  KSP            kspinner;
  PetscBool      flg;

  PetscFunctionBegin;
  ierr = KSPGetPC(ksp, &pc);CHKERRQ(ierr);

  /* compute inner_rtol */
  if (scale->bnrm < 0.0) {
    Vec b;
    ierr = KSPGetRhs(ksp, &b);CHKERRQ(ierr);
    ierr = VecNorm(b, NORM_2, &(scale->bnrm));CHKERRQ(ierr);
  }
  ierr       = KSPGetTolerances(ksp, &outer_rtol, &outer_abstol, &outer_dtol, &outer_maxits);CHKERRQ(ierr);
  inner_rtol = PetscMin(scale->coef * scale->bnrm * outer_rtol / fnorm, 0.999);
  /*ierr = PetscPrintf(PETSC_COMM_WORLD, "        Inner rtol = %g\n", (double)inner_rtol);CHKERRQ(ierr);*/

  /* if pc is ksp */
  ierr = PetscObjectTypeCompare((PetscObject)pc,PCKSP,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PCKSPGetKSP(pc, &kspinner);CHKERRQ(ierr);
    ierr = KSPSetTolerances(kspinner, inner_rtol, outer_abstol, outer_dtol, outer_maxits);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  /* if pc is bjacobi */
  ierr = PetscObjectTypeCompare((PetscObject)pc,PCBJACOBI,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PCBJacobiGetSubKSP(pc, &nksp, &first, &subksp);CHKERRQ(ierr);
    if (subksp) {
      for (i=0; i<nksp; i++) {
        ierr = KSPSetTolerances(subksp[i], inner_rtol, outer_abstol, outer_dtol, outer_maxits);CHKERRQ(ierr);
      }
      PetscFunctionReturn(0);
    }
  }

  /* if pc is deflation*/
  ierr = PetscObjectTypeCompare((PetscObject)pc,PCDEFLATION,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PCDeflationGetCoarseKSP(pc,&kspinner);CHKERRQ(ierr);
    ierr = KSPSetTolerances(kspinner,inner_rtol,outer_abstol,outer_dtol,PETSC_DEFAULT);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(*dummy);CHKERRQ(ierr);
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

.seealso: KSPSetConvergenceTest(), KSPSetTolerances(), KSPSetNormType()
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

.seealso: KSPConvergedDefault(), KSPConvergedDefaultDestroy(), KSPSetConvergenceTest(), KSPSetTolerances(),
          KSPConvergedSkip(), KSPConvergedReason, KSPGetConvergedReason(), KSPConvergedDefaultSetUIRNorm(), KSPConvergedDefaultSetUMIRNorm(), KSPConvergedDefaultSetConvergedMaxits()
@*/
PetscErrorCode  KSPConvergedDefaultCreate(void **ctx)
{
  PetscErrorCode         ierr;
  KSPConvergedDefaultCtx *cctx;

  PetscFunctionBegin;
  ierr = PetscNew(&cctx);CHKERRQ(ierr);
  *ctx = cctx;
  PetscFunctionReturn(0);
}

/*@
   KSPConvergedDefaultSetUIRNorm - makes the default convergence test use || B*(b - A*(initial guess))||
      instead of || B*b ||. In the case of right preconditioner or if KSPSetNormType(ksp,KSP_NORM_UNPRECONDIITONED)
      is used there is no B in the above formula. UIRNorm is short for Use Initial Residual Norm.

   Collective on ksp

   Input Parameters:
.  ksp   - iterative context

   Options Database:
.   -ksp_converged_use_initial_residual_norm

   Notes:
   Use KSPSetTolerances() to alter the defaults for rtol, abstol, dtol.

   The precise values of reason are macros such as KSP_CONVERGED_RTOL, which
   are defined in petscksp.h.

   If the convergence test is not KSPConvergedDefault() then this is ignored.

   If right preconditioning is being used then B does not appear in the above formula.


   Level: intermediate

.seealso: KSPSetConvergenceTest(), KSPSetTolerances(), KSPConvergedSkip(), KSPConvergedReason, KSPGetConvergedReason(), KSPConvergedDefaultSetUMIRNorm(), KSPConvergedDefaultSetConvergedMaxits()
@*/
PetscErrorCode  KSPConvergedDefaultSetUIRNorm(KSP ksp)
{
  KSPConvergedDefaultCtx *ctx = (KSPConvergedDefaultCtx*) ksp->cnvP;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  if (ksp->converged != KSPConvergedDefault) PetscFunctionReturn(0);
  if (ctx->mininitialrtol) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_WRONGSTATE,"Cannot use KSPConvergedDefaultSetUIRNorm() and KSPConvergedDefaultSetUMIRNorm() together");
  ctx->initialrtol = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@
   KSPConvergedDefaultSetUMIRNorm - makes the default convergence test use min(|| B*(b - A*(initial guess))||,|| B*b ||)
      In the case of right preconditioner or if KSPSetNormType(ksp,KSP_NORM_UNPRECONDIITONED)
      is used there is no B in the above formula. UMIRNorm is short for Use Minimum Initial Residual Norm.

   Collective on ksp

   Input Parameters:
.  ksp   - iterative context

   Options Database:
.   -ksp_converged_use_min_initial_residual_norm

   Use KSPSetTolerances() to alter the defaults for rtol, abstol, dtol.

   The precise values of reason are macros such as KSP_CONVERGED_RTOL, which
   are defined in petscksp.h.

   Level: intermediate

.seealso: KSPSetConvergenceTest(), KSPSetTolerances(), KSPConvergedSkip(), KSPConvergedReason, KSPGetConvergedReason(), KSPConvergedDefaultSetUIRNorm(), KSPConvergedDefaultSetConvergedMaxits()
@*/
PetscErrorCode  KSPConvergedDefaultSetUMIRNorm(KSP ksp)
{
  KSPConvergedDefaultCtx *ctx = (KSPConvergedDefaultCtx*) ksp->cnvP;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  if (ksp->converged != KSPConvergedDefault) PetscFunctionReturn(0);
  if (ctx->initialrtol) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_WRONGSTATE,"Cannot use KSPConvergedDefaultSetUIRNorm() and KSPConvergedDefaultSetUMIRNorm() together");
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
.   -ksp_converged_maxits

   Use KSPSetTolerances() to alter the defaults for rtol, abstol, dtol.

   The precise values of reason are macros such as KSP_CONVERGED_RTOL, which
   are defined in petscksp.h.

   Level: intermediate

.seealso: KSPSetConvergenceTest(), KSPSetTolerances(), KSPConvergedSkip(), KSPConvergedReason, KSPGetConvergedReason(), KSPConvergedDefaultSetUMIRNorm(), KSPConvergedDefaultSetUIRNorm()
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
+   positive - if the iteration has converged
.   negative - if the iteration has diverged
-   KSP_CONVERGED_ITERATING - otherwise.

   Notes:
   KSPConvergedDefault() reaches convergence when   rnorm < MAX (rtol * rnorm_0, abstol);
   Divergence is detected if rnorm > dtol * rnorm_0, or when failures are detected throughout the iteration.
   By default, reaching the maximum number of iterations is considered divergence (i.e. KSP_DIVERGED_ITS).
   In order to have PETSc declaring convergence in such a case (i.e. KSP_CONVERGED_ITS), users can use KSPConvergedDefaultSetConvergedMaxits()

   where:
+     rtol = relative tolerance,
.     abstol = absolute tolerance.
.     dtol = divergence tolerance,
-     rnorm_0 is the two norm of the right hand side (or the preconditioned norm, depending on what was set with
          KSPSetNormType(). When initial guess is non-zero you
          can call KSPConvergedDefaultSetUIRNorm() to use the norm of (b - A*(initial guess))
          as the starting point for relative norm convergence testing, that is as rnorm_0

   Use KSPSetTolerances() to alter the defaults for rtol, abstol, dtol.

   Use KSPSetNormType() (or -ksp_norm_type <none,preconditioned,unpreconditioned,natural>) to change the norm used for computing rnorm

   The precise values of reason are macros such as KSP_CONVERGED_RTOL, which are defined in petscksp.h.

   This routine is used by KSP by default so the user generally never needs call it directly.

   Use KSPSetConvergenceTest() to provide your own test instead of using this one.

   Level: intermediate

.seealso: KSPSetConvergenceTest(), KSPSetTolerances(), KSPConvergedSkip(), KSPConvergedReason, KSPGetConvergedReason(),
          KSPConvergedDefaultSetUIRNorm(), KSPConvergedDefaultSetUMIRNorm(), KSPConvergedDefaultSetConvergedMaxits(), KSPConvergedDefaultCreate(), KSPConvergedDefaultDestroy()
@*/
PetscErrorCode  KSPConvergedDefault(KSP ksp,PetscInt n,PetscReal rnorm,KSPConvergedReason *reason,void *ctx)
{
  PetscErrorCode         ierr;
  KSPConvergedDefaultCtx *cctx = (KSPConvergedDefaultCtx*) ctx;
  KSPNormType            normtype;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidLogicalCollectiveInt(ksp,n,2);
  PetscValidPointer(reason,4);
  if (!cctx) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_NULL,"Convergence context must have been created with KSPConvergedDefaultCreate()");
  *reason = KSP_CONVERGED_ITERATING;

  if (cctx->convmaxits && n >= ksp->max_it) {
    *reason = KSP_CONVERGED_ITS;
    ierr    = PetscInfo1(ksp,"Linear solver has converged. Maximum number of iterations reached %D\n",n);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = KSPGetNormType(ksp,&normtype);CHKERRQ(ierr);
  if (normtype == KSP_NORM_NONE) PetscFunctionReturn(0);

  if (!n) {
    /* if user gives initial guess need to compute norm of b */
    if (!ksp->guess_zero && !cctx->initialrtol) {
      PetscReal snorm = 0.0;
      if (ksp->normtype == KSP_NORM_UNPRECONDITIONED || ksp->pc_side == PC_RIGHT) {
        ierr = PetscInfo(ksp,"user has provided nonzero initial guess, computing 2-norm of RHS\n");CHKERRQ(ierr);
        ierr = VecNorm(ksp->vec_rhs,NORM_2,&snorm);CHKERRQ(ierr);        /*     <- b'*b */
      } else {
        Vec z;
        /* Should avoid allocating the z vector each time but cannot stash it in cctx because if KSPReset() is called the vector size might change */
        ierr = VecDuplicate(ksp->vec_rhs,&z);CHKERRQ(ierr);
        ierr = KSP_PCApply(ksp,ksp->vec_rhs,z);CHKERRQ(ierr);
        if (ksp->normtype == KSP_NORM_PRECONDITIONED) {
          ierr = PetscInfo(ksp,"user has provided nonzero initial guess, computing 2-norm of preconditioned RHS\n");CHKERRQ(ierr);
          ierr = VecNorm(z,NORM_2,&snorm);CHKERRQ(ierr);                 /*    dp <- b'*B'*B*b */
        } else if (ksp->normtype == KSP_NORM_NATURAL) {
          PetscScalar norm;
          ierr  = PetscInfo(ksp,"user has provided nonzero initial guess, computing natural norm of RHS\n");CHKERRQ(ierr);
          ierr  = VecDot(ksp->vec_rhs,z,&norm);CHKERRQ(ierr);
          snorm = PetscSqrtReal(PetscAbsScalar(norm));                            /*    dp <- b'*B*b */
        }
        ierr = VecDestroy(&z);CHKERRQ(ierr);
      }
      /* handle special case of zero RHS and nonzero guess */
      if (!snorm) {
        ierr  = PetscInfo(ksp,"Special case, user has provided nonzero initial guess and zero RHS\n");CHKERRQ(ierr);
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
    ierr = PCGetFailedReasonRank(ksp->pc,&pcreason);CHKERRQ(ierr);
    sendbuf = (PetscInt)pcreason;
    ierr = MPI_Allreduce(&sendbuf,&recvbuf,1,MPIU_INT,MPIU_MAX,PetscObjectComm((PetscObject)ksp));CHKERRMPI(ierr);
    if (recvbuf) {
      *reason = KSP_DIVERGED_PC_FAILED;
      ierr = PCSetFailedReason(ksp->pc,(PCFailedReason)recvbuf);CHKERRQ(ierr);
      ierr    = PetscInfo(ksp,"Linear solver pcsetup fails, declaring divergence \n");CHKERRQ(ierr);
    } else {
      *reason = KSP_DIVERGED_NANORINF;
      ierr    = PetscInfo(ksp,"Linear solver has created a not a number (NaN) as the residual norm, declaring divergence \n");CHKERRQ(ierr);
    }
  } else if (rnorm <= ksp->ttol) {
    if (rnorm < ksp->abstol) {
      ierr    = PetscInfo3(ksp,"Linear solver has converged. Residual norm %14.12e is less than absolute tolerance %14.12e at iteration %D\n",(double)rnorm,(double)ksp->abstol,n);CHKERRQ(ierr);
      *reason = KSP_CONVERGED_ATOL;
    } else {
      if (cctx->initialrtol) {
        ierr = PetscInfo4(ksp,"Linear solver has converged. Residual norm %14.12e is less than relative tolerance %14.12e times initial residual norm %14.12e at iteration %D\n",(double)rnorm,(double)ksp->rtol,(double)ksp->rnorm0,n);CHKERRQ(ierr);
      } else {
        ierr = PetscInfo4(ksp,"Linear solver has converged. Residual norm %14.12e is less than relative tolerance %14.12e times initial right hand side norm %14.12e at iteration %D\n",(double)rnorm,(double)ksp->rtol,(double)ksp->rnorm0,n);CHKERRQ(ierr);
      }
      *reason = KSP_CONVERGED_RTOL;
    }
  } else if (rnorm >= ksp->divtol*ksp->rnorm0) {
    ierr    = PetscInfo3(ksp,"Linear solver is diverging. Initial right hand size norm %14.12e, current residual norm %14.12e at iteration %D\n",(double)ksp->rnorm0,(double)rnorm,n);CHKERRQ(ierr);
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

.seealso: KSPConvergedDefault(), KSPConvergedDefaultCreate(), KSPSetConvergenceTest(), KSPSetTolerances(), KSPConvergedSkip(),
          KSPConvergedReason, KSPGetConvergedReason(), KSPConvergedDefaultSetUIRNorm(), KSPConvergedDefaultSetUMIRNorm()
@*/
PetscErrorCode  KSPConvergedDefaultDestroy(void *ctx)
{
  PetscErrorCode         ierr;
  KSPConvergedDefaultCtx *cctx = (KSPConvergedDefaultCtx*) ctx;

  PetscFunctionBegin;
  ierr = VecDestroy(&cctx->work);CHKERRQ(ierr);
  ierr = PetscFree(ctx);CHKERRQ(ierr);
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

.seealso: KSPGetSolution(), KSPBuildResidualDefault()
*/
PetscErrorCode KSPBuildSolutionDefault(KSP ksp,Vec v,Vec *V)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ksp->pc_side == PC_RIGHT) {
    if (ksp->pc) {
      if (v) {
        ierr = KSP_PCApply(ksp,ksp->vec_sol,v);CHKERRQ(ierr); *V = v;
      } else SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"Not working with right preconditioner");
    } else {
      if (v) {
        ierr = VecCopy(ksp->vec_sol,v);CHKERRQ(ierr); *V = v;
      } else *V = ksp->vec_sol;
    }
  } else if (ksp->pc_side == PC_SYMMETRIC) {
    if (ksp->pc) {
      if (ksp->transpose_solve) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"Not working with symmetric preconditioner and transpose solve");
      if (v) {
        ierr = PCApplySymmetricRight(ksp->pc,ksp->vec_sol,v);CHKERRQ(ierr);
        *V = v;
      } else SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"Not working with symmetric preconditioner");
    } else {
      if (v) {
        ierr = VecCopy(ksp->vec_sol,v);CHKERRQ(ierr); *V = v;
      } else *V = ksp->vec_sol;
    }
  } else {
    if (v) {
      ierr = VecCopy(ksp->vec_sol,v);CHKERRQ(ierr); *V = v;
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

.seealso: KSPBuildSolutionDefault()
*/
PetscErrorCode KSPBuildResidualDefault(KSP ksp,Vec t,Vec v,Vec *V)
{
  PetscErrorCode ierr;
  Mat            Amat,Pmat;

  PetscFunctionBegin;
  if (!ksp->pc) {ierr = KSPGetPC(ksp,&ksp->pc);CHKERRQ(ierr);}
  ierr = PCGetOperators(ksp->pc,&Amat,&Pmat);CHKERRQ(ierr);
  ierr = KSPBuildSolution(ksp,t,NULL);CHKERRQ(ierr);
  ierr = KSP_MatMult(ksp,Amat,t,v);CHKERRQ(ierr);
  ierr = VecAYPX(v,-1.0,ksp->vec_rhs);CHKERRQ(ierr);
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

  Output Parameter:
+  right - the array of vectors created
-  left - the array of left vectors

   Note: The right vector has as many elements as the matrix has columns. The left
     vector has as many elements as the matrix has rows.

   The vectors are new vectors that are not owned by the KSP, they should be destroyed with calls to VecDestroyVecs() when no longer needed.

   Developers Note: First tries to duplicate the rhs and solution vectors of the KSP, if they do not exist tries to get them from the matrix, if
                    that does not exist tries to get them from the DM (if it is provided).

   Level: advanced

.seealso:   MatCreateVecs(), VecDestroyVecs()

@*/
PetscErrorCode KSPCreateVecs(KSP ksp,PetscInt rightn, Vec **right,PetscInt leftn,Vec **left)
{
  PetscErrorCode ierr;
  Vec            vecr = NULL,vecl = NULL;
  PetscBool      matset,pmatset;
  Mat            mat = NULL;

  PetscFunctionBegin;
  if (rightn) {
    if (!right) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_INCOMP,"You asked for right vectors but did not pass a pointer to hold them");
    if (ksp->vec_sol) vecr = ksp->vec_sol;
    else {
      if (ksp->pc) {
        ierr = PCGetOperatorsSet(ksp->pc,&matset,&pmatset);CHKERRQ(ierr);
        /* check for mat before pmat because for KSPLSQR pmat may be a different size than mat since pmat maybe mat'*mat */
        if (matset) {
          ierr = PCGetOperators(ksp->pc,&mat,NULL);CHKERRQ(ierr);
          ierr = MatCreateVecs(mat,&vecr,NULL);CHKERRQ(ierr);
        } else if (pmatset) {
          ierr = PCGetOperators(ksp->pc,NULL,&mat);CHKERRQ(ierr);
          ierr = MatCreateVecs(mat,&vecr,NULL);CHKERRQ(ierr);
        }
      }
      if (!vecr) {
        if (ksp->dm) {
          ierr = DMGetGlobalVector(ksp->dm,&vecr);CHKERRQ(ierr);
        } else SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_WRONGSTATE,"You requested a vector from a KSP that cannot provide one");
      }
    }
    ierr = VecDuplicateVecs(vecr,rightn,right);CHKERRQ(ierr);
    if (!ksp->vec_sol) {
      if (mat) {
        ierr = VecDestroy(&vecr);CHKERRQ(ierr);
      } else if (ksp->dm) {
        ierr = DMRestoreGlobalVector(ksp->dm,&vecr);CHKERRQ(ierr);
      }
    }
  }
  if (leftn) {
    if (!left) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_INCOMP,"You asked for left vectors but did not pass a pointer to hold them");
    if (ksp->vec_rhs) vecl = ksp->vec_rhs;
    else {
      if (ksp->pc) {
        ierr = PCGetOperatorsSet(ksp->pc,&matset,&pmatset);CHKERRQ(ierr);
        /* check for mat before pmat because for KSPLSQR pmat may be a different size than mat since pmat maybe mat'*mat */
        if (matset) {
          ierr = PCGetOperators(ksp->pc,&mat,NULL);CHKERRQ(ierr);
          ierr = MatCreateVecs(mat,NULL,&vecl);CHKERRQ(ierr);
        } else if (pmatset) {
          ierr = PCGetOperators(ksp->pc,NULL,&mat);CHKERRQ(ierr);
          ierr = MatCreateVecs(mat,NULL,&vecl);CHKERRQ(ierr);
        }
      }
      if (!vecl) {
        if (ksp->dm) {
          ierr = DMGetGlobalVector(ksp->dm,&vecl);CHKERRQ(ierr);
        } else SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_WRONGSTATE,"You requested a vector from a KSP that cannot provide one");
      }
    }
    ierr = VecDuplicateVecs(vecl,leftn,left);CHKERRQ(ierr);
    if (!ksp->vec_rhs) {
      if (mat) {
        ierr = VecDestroy(&vecl);CHKERRQ(ierr);
      } else if (ksp->dm) {
        ierr = DMRestoreGlobalVector(ksp->dm,&vecl);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr       = VecDestroyVecs(ksp->nwork,&ksp->work);CHKERRQ(ierr);
  ksp->nwork = nw;
  ierr       = KSPCreateVecs(ksp,nw,&ksp->work,0,NULL);CHKERRQ(ierr);
  ierr       = PetscLogObjectParents(ksp,nw,ksp->work);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  ierr = PetscFree(ksp->data);CHKERRQ(ierr);
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

.seealso: KSPSetConvergenceTest(), KSPConvergedDefault(), KSPSetTolerances(), KSPConvergedReason,
          KSPConvergedReasonView()
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

   Level: basic

.seealso: KSPGetConvergedReason()
@*/
PetscErrorCode KSPGetConvergedReasonString(KSP ksp,const char** strreason)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidCharPointer(strreason,2);
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

.seealso: KSPGetDM(), KSPSetDMActive(), KSPSetComputeOperators(), KSPSetComputeRHS(), KSPSetComputeInitialGuess(), DMKSPSetComputeOperators(), DMKSPSetComputeRHS(), DMKSPSetComputeInitialGuess()
@*/
PetscErrorCode  KSPSetDM(KSP ksp,DM dm)
{
  PetscErrorCode ierr;
  PC             pc;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidHeaderSpecific(dm,DM_CLASSID,2);
  ierr = PetscObjectReference((PetscObject)dm);CHKERRQ(ierr);
  if (ksp->dm) {                /* Move the DMSNES context over to the new DM unless the new DM already has one */
    if (ksp->dm->dmksp && !dm->dmksp) {
      DMKSP kdm;
      ierr = DMCopyDMKSP(ksp->dm,dm);CHKERRQ(ierr);
      ierr = DMGetDMKSP(ksp->dm,&kdm);CHKERRQ(ierr);
      if (kdm->originaldm == ksp->dm) kdm->originaldm = dm; /* Grant write privileges to the replacement DM */
    }
    ierr = DMDestroy(&ksp->dm);CHKERRQ(ierr);
  }
  ksp->dm       = dm;
  ksp->dmAuto   = PETSC_FALSE;
  ierr          = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr          = PCSetDM(pc,dm);CHKERRQ(ierr);
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

.seealso: KSPGetDM(), KSPSetDM(), SNESSetDM(), KSPSetComputeOperators(), KSPSetComputeRHS(), KSPSetComputeInitialGuess()
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


.seealso: KSPSetDM(), KSPSetDMActive()
@*/
PetscErrorCode  KSPGetDM(KSP ksp,DM *dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  if (!ksp->dm) {
    ierr        = DMShellCreate(PetscObjectComm((PetscObject)ksp),&ksp->dm);CHKERRQ(ierr);
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

.seealso: KSPGetApplicationContext()
@*/
PetscErrorCode  KSPSetApplicationContext(KSP ksp,void *usrP)
{
  PetscErrorCode ierr;
  PC             pc;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  ksp->user = usrP;
  ierr      = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr      = PCSetApplicationContext(pc,usrP);CHKERRQ(ierr);
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

.seealso: KSPSetApplicationContext()
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

   Input Parameter:
+  ksp - the linear solver (KSP) context.
.  pc - the preconditioner context
-  vec - a vector that will be initialized with Inf to indicate lack of convergence

   Notes: this may be called by a subset of the processes in the PC

   Level: developer

   Developer Note: this is used to manage returning from preconditioners whose inner KSP solvers have failed in some way

.seealso: KSPCreate(), KSPSetType(), KSP, KSPCheckNorm(), KSPCheckDot()
@*/
PetscErrorCode KSPCheckSolve(KSP ksp,PC pc,Vec vec)
{
  PetscErrorCode     ierr;
  PCFailedReason     pcreason;
  PC                 subpc;

  PetscFunctionBegin;
  ierr = KSPGetPC(ksp,&subpc);CHKERRQ(ierr);
  ierr = PCGetFailedReason(subpc,&pcreason);CHKERRQ(ierr);
  if (pcreason || (ksp->reason < 0 && ksp->reason != KSP_DIVERGED_ITS)) {
    if (pc->erroriffailure) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_NOT_CONVERGED,"Detected not converged in KSP inner solve: KSP reason %s PC reason %s",KSPConvergedReasons[ksp->reason],PCFailedReasons[pcreason]);
    else {
      ierr = PetscInfo2(ksp,"Detected not converged in KSP inner solve: KSP reason %s PC reason %s\n",KSPConvergedReasons[ksp->reason],PCFailedReasons[pcreason]);CHKERRQ(ierr);
      pc->failedreason = PC_SUBPC_ERROR;
      if (vec) {
        ierr = VecSetInf(vec);CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}
