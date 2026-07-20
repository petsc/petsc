#include <petsc/private/kspimpl.h> /*I  "petscksp.h"   I*/
#include <petscdraw.h>

/*@C
  KSPMonitorLGRange - Prints line graphs summarizing the residual norm, the fraction of elements that dominate the residual, and the convergence factor at each iteration of the `KSP` solver

  Collective

  Input Parameters:
+ ksp    - iterative context
. n      - iteration number
. rnorm  - the 2-norm of the residual (or an approximation)
- monctx - a `PetscViewer` (typically of type `PETSCVIEWERDRAW`) containing the line graphs to update

  Level: intermediate

  Note:
  Suitable for use as a `KSP` monitor via `KSPMonitorSet()`. Draws four line graphs: the log residual norm, the
  percentage of entries whose magnitude exceeds $0.2$ times the maximum, the relative change in residual norm per
  iteration, and their product.

.seealso: [](ch_ksp), `KSP`, `KSPMonitorSet()`, `KSPMonitorResidual()`, `PETSCVIEWERDRAW`, `PetscDrawLG`
@*/
PetscErrorCode KSPMonitorLGRange(KSP ksp, PetscInt n, PetscReal rnorm, void *monctx)
{
  PetscDrawLG      lg;
  PetscReal        x, y, per;
  PetscViewer      v = (PetscViewer)monctx;
  static PetscReal prev; /* should be in the context */
  PetscDraw        draw;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v, PETSC_VIEWER_CLASSID, 4);

  PetscCall(KSPMonitorRange_Private(ksp, n, &per));
  if (!n) prev = rnorm;

  PetscCall(PetscViewerDrawGetDrawLG(v, 0, &lg));
  if (!n) PetscCall(PetscDrawLGReset(lg));
  PetscCall(PetscDrawLGGetDraw(lg, &draw));
  PetscCall(PetscDrawSetTitle(draw, "Residual norm"));
  x = (PetscReal)n;
  if (rnorm > 0.0) y = PetscLog10Real(rnorm);
  else y = -15.0;
  PetscCall(PetscDrawLGAddPoint(lg, &x, &y));
  if (n < 20 || !(n % 5) || ksp->reason) {
    PetscCall(PetscDrawLGDraw(lg));
    PetscCall(PetscDrawLGSave(lg));
  }

  PetscCall(PetscViewerDrawGetDrawLG(v, 1, &lg));
  if (!n) PetscCall(PetscDrawLGReset(lg));
  PetscCall(PetscDrawLGGetDraw(lg, &draw));
  PetscCall(PetscDrawSetTitle(draw, "% elements > .2*max element"));
  x = (PetscReal)n;
  y = 100.0 * per;
  PetscCall(PetscDrawLGAddPoint(lg, &x, &y));
  if (n < 20 || !(n % 5) || ksp->reason) {
    PetscCall(PetscDrawLGDraw(lg));
    PetscCall(PetscDrawLGSave(lg));
  }

  PetscCall(PetscViewerDrawGetDrawLG(v, 2, &lg));
  if (!n) PetscCall(PetscDrawLGReset(lg));
  PetscCall(PetscDrawLGGetDraw(lg, &draw));
  PetscCall(PetscDrawSetTitle(draw, "(norm - oldnorm)/oldnorm"));
  x = (PetscReal)n;
  y = (prev - rnorm) / prev;
  PetscCall(PetscDrawLGAddPoint(lg, &x, &y));
  if (n < 20 || !(n % 5) || ksp->reason) {
    PetscCall(PetscDrawLGDraw(lg));
    PetscCall(PetscDrawLGSave(lg));
  }

  PetscCall(PetscViewerDrawGetDrawLG(v, 3, &lg));
  if (!n) PetscCall(PetscDrawLGReset(lg));
  PetscCall(PetscDrawLGGetDraw(lg, &draw));
  PetscCall(PetscDrawSetTitle(draw, "(norm - oldnorm)/oldnorm*(% > .2 max)"));
  x = (PetscReal)n;
  y = (prev - rnorm) / (prev * per);
  if (n > 5) PetscCall(PetscDrawLGAddPoint(lg, &x, &y));
  if (n < 20 || !(n % 5) || ksp->reason) {
    PetscCall(PetscDrawLGDraw(lg));
    PetscCall(PetscDrawLGSave(lg));
  }
  prev = rnorm;
  PetscFunctionReturn(PETSC_SUCCESS);
}
