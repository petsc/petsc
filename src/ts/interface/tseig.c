
#include <petsc/private/tsimpl.h> /*I "petscts.h"  I*/
#include <petscdraw.h>

/* ------------------------------------------------------------------------*/
struct _n_TSMonitorSPEigCtx {
  PetscDrawSP drawsp;
  KSP         ksp;
  PetscInt    howoften; /* when > 0 uses step % howoften, when negative only final solution plotted */
  PetscBool   computeexplicitly;
  MPI_Comm    comm;
  PetscRandom rand;
  PetscReal   xmin, xmax, ymin, ymax;
};

/*@C
   TSMonitorSPEigCtxCreate - Creates a context for use with `TS` to monitor the eigenvalues of the linearized operator

   Collective

   Input Parameters:
+  host - the X display to open, or null for the local machine
.  label - the title to put in the title bar
.  x, y - the screen coordinates of the upper left coordinate of the window
.  m, n - the screen width and height in pixels
-  howoften - if positive then determines the frequency of the plotting, if -1 then only at the final time

   Output Parameter:
.  ctx - the context

   Options Database Key:
.  -ts_monitor_sp_eig - plot egienvalues of linearized right hand side

   Level: intermediate

   Notes:
   Use `TSMonitorSPEigCtxDestroy()` to destroy the context

   Currently only works if the Jacobian is provided explicitly.

   Currently only works for ODEs u_t - F(t,u) = 0; that is with no mass matrix.

.seealso: [](chapter_ts), `TSMonitorSPEigTimeStep()`, `TSMonitorSet()`, `TSMonitorLGSolution()`, `TSMonitorLGError()`
@*/
PetscErrorCode TSMonitorSPEigCtxCreate(MPI_Comm comm, const char host[], const char label[], int x, int y, int m, int n, PetscInt howoften, TSMonitorSPEigCtx *ctx)
{
  PetscDraw win;
  PC        pc;

  PetscFunctionBegin;
  PetscCall(PetscNew(ctx));
  PetscCall(PetscRandomCreate(comm, &(*ctx)->rand));
  PetscCall(PetscRandomSetFromOptions((*ctx)->rand));
  PetscCall(PetscDrawCreate(comm, host, label, x, y, m, n, &win));
  PetscCall(PetscDrawSetFromOptions(win));
  PetscCall(PetscDrawSPCreate(win, 1, &(*ctx)->drawsp));
  PetscCall(KSPCreate(comm, &(*ctx)->ksp));
  PetscCall(KSPSetOptionsPrefix((*ctx)->ksp, "ts_monitor_sp_eig_")); /* this is wrong, used use also prefix from the TS */
  PetscCall(KSPSetType((*ctx)->ksp, KSPGMRES));
  PetscCall(KSPGMRESSetRestart((*ctx)->ksp, 200));
  PetscCall(KSPSetTolerances((*ctx)->ksp, 1.e-10, PETSC_DEFAULT, PETSC_DEFAULT, 200));
  PetscCall(KSPSetComputeSingularValues((*ctx)->ksp, PETSC_TRUE));
  PetscCall(KSPSetFromOptions((*ctx)->ksp));
  PetscCall(KSPGetPC((*ctx)->ksp, &pc));
  PetscCall(PCSetType(pc, PCNONE));

  (*ctx)->howoften          = howoften;
  (*ctx)->computeexplicitly = PETSC_FALSE;

  PetscCall(PetscOptionsGetBool(NULL, NULL, "-ts_monitor_sp_eig_explicitly", &(*ctx)->computeexplicitly, NULL));

  (*ctx)->comm = comm;
  (*ctx)->xmin = -2.1;
  (*ctx)->xmax = 1.1;
  (*ctx)->ymin = -1.1;
  (*ctx)->ymax = 1.1;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSLinearStabilityIndicator(TS ts, PetscReal xr, PetscReal xi, PetscBool *flg)
{
  PetscReal yr, yi;

  PetscFunctionBegin;
  PetscCall(TSComputeLinearStability(ts, xr, xi, &yr, &yi));
  if ((yr * yr + yi * yi) <= 1.0) *flg = PETSC_TRUE;
  else *flg = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode TSMonitorSPEig(TS ts, PetscInt step, PetscReal ptime, Vec v, void *monctx)
{
  TSMonitorSPEigCtx ctx = (TSMonitorSPEigCtx)monctx;
  KSP               ksp = ctx->ksp;
  PetscInt          n, N, nits, neig, i, its = 200;
  PetscReal        *r, *c, time_step_save;
  PetscDrawSP       drawsp = ctx->drawsp;
  Mat               A, B;
  Vec               xdot;
  SNES              snes;

  PetscFunctionBegin;
  if (step < 0) PetscFunctionReturn(0); /* -1 indicates interpolated solution */
  if (!step) PetscFunctionReturn(0);
  if (((ctx->howoften > 0) && (!(step % ctx->howoften))) || ((ctx->howoften == -1) && ts->reason)) {
    PetscCall(VecDuplicate(v, &xdot));
    PetscCall(TSGetSNES(ts, &snes));
    PetscCall(SNESGetJacobian(snes, &A, &B, NULL, NULL));
    PetscCall(MatDuplicate(A, MAT_DO_NOT_COPY_VALUES, &B));
    /*
       This doesn't work because methods keep and use internal information about the shift so it
       seems we would need code for each method to trick the correct Jacobian in being computed.
     */
    time_step_save = ts->time_step;
    ts->time_step  = PETSC_MAX_REAL;

    PetscCall(SNESComputeJacobian(snes, v, A, B));

    ts->time_step = time_step_save;

    PetscCall(KSPSetOperators(ksp, B, B));
    PetscCall(VecGetSize(v, &n));
    if (n < 200) its = n;
    PetscCall(KSPSetTolerances(ksp, 1.e-10, PETSC_DEFAULT, PETSC_DEFAULT, its));
    PetscCall(VecSetRandom(xdot, ctx->rand));
    PetscCall(KSPSolve(ksp, xdot, xdot));
    PetscCall(VecDestroy(&xdot));
    PetscCall(KSPGetIterationNumber(ksp, &nits));
    N = nits + 2;

    if (nits) {
      PetscDraw     draw;
      PetscReal     pause;
      PetscDrawAxis axis;
      PetscReal     xmin, xmax, ymin, ymax;

      PetscCall(PetscDrawSPReset(drawsp));
      PetscCall(PetscDrawSPSetLimits(drawsp, ctx->xmin, ctx->xmax, ctx->ymin, ctx->ymax));
      PetscCall(PetscMalloc2(PetscMax(n, N), &r, PetscMax(n, N), &c));
      if (ctx->computeexplicitly) {
        PetscCall(KSPComputeEigenvaluesExplicitly(ksp, n, r, c));
        neig = n;
      } else {
        PetscCall(KSPComputeEigenvalues(ksp, N, r, c, &neig));
      }
      /* We used the positive operator to be able to reuse KSPs that require positive definiteness, now flip the spectrum as is conventional for ODEs */
      for (i = 0; i < neig; i++) r[i] = -r[i];
      for (i = 0; i < neig; i++) {
        if (ts->ops->linearstability) {
          PetscReal fr, fi;
          PetscCall(TSComputeLinearStability(ts, r[i], c[i], &fr, &fi));
          if ((fr * fr + fi * fi) > 1.0) PetscCall(PetscPrintf(ctx->comm, "Linearized Eigenvalue %g + %g i linear stability function %g norm indicates unstable scheme \n", (double)r[i], (double)c[i], (double)(fr * fr + fi * fi)));
        }
        PetscCall(PetscDrawSPAddPoint(drawsp, r + i, c + i));
      }
      PetscCall(PetscFree2(r, c));
      PetscCall(PetscDrawSPGetDraw(drawsp, &draw));
      PetscCall(PetscDrawGetPause(draw, &pause));
      PetscCall(PetscDrawSetPause(draw, 0.0));
      PetscCall(PetscDrawSPDraw(drawsp, PETSC_TRUE));
      PetscCall(PetscDrawSetPause(draw, pause));
      if (ts->ops->linearstability) {
        PetscCall(PetscDrawSPGetAxis(drawsp, &axis));
        PetscCall(PetscDrawAxisGetLimits(axis, &xmin, &xmax, &ymin, &ymax));
        PetscCall(PetscDrawIndicatorFunction(draw, xmin, xmax, ymin, ymax, PETSC_DRAW_CYAN, (PetscErrorCode(*)(void *, PetscReal, PetscReal, PetscBool *))TSLinearStabilityIndicator, ts));
        PetscCall(PetscDrawSPDraw(drawsp, PETSC_FALSE));
      }
      PetscCall(PetscDrawSPSave(drawsp));
    }
    PetscCall(MatDestroy(&B));
  }
  PetscFunctionReturn(0);
}

/*@C
   TSMonitorSPEigCtxDestroy - Destroys a scatter plot context that was created with `TSMonitorSPEigCtxCreate()`.

   Collective on ctx

   Input Parameter:
.  ctx - the monitor context

   Level: intermediate

   Note:
   Should be passed to `TSMonitorSet()` along with `TSMonitorSPEig()` an the context created with `TSMonitorSPEigCtxCreate()`

.seealso: [](chapter_ts), `TSMonitorSPEigCtxCreate()`, `TSMonitorSet()`, `TSMonitorSPEig();`
@*/
PetscErrorCode TSMonitorSPEigCtxDestroy(TSMonitorSPEigCtx *ctx)
{
  PetscDraw draw;

  PetscFunctionBegin;
  PetscCall(PetscDrawSPGetDraw((*ctx)->drawsp, &draw));
  PetscCall(PetscDrawDestroy(&draw));
  PetscCall(PetscDrawSPDestroy(&(*ctx)->drawsp));
  PetscCall(KSPDestroy(&(*ctx)->ksp));
  PetscCall(PetscRandomDestroy(&(*ctx)->rand));
  PetscCall(PetscFree(*ctx));
  PetscFunctionReturn(0);
}
