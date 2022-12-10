#include <petsc/private/dmnetworkimpl.h> /*I "petscdmplex.h" I*/
#include <petscts.h>
#include <petscdraw.h>

/*
   TSMonitorLGCtxDestroy - Destroys  line graph contexts that where created with TSMonitorLGCtxNetworkCreate().

   Collective

   Input Parameter:
.  ctx - the monitor context

*/
PetscErrorCode TSMonitorLGCtxNetworkDestroy(TSMonitorLGCtxNetwork *ctx)
{
  PetscInt i;

  PetscFunctionBegin;
  for (i = 0; i < (*ctx)->nlg; i++) PetscCall(PetscDrawLGDestroy(&(*ctx)->lg[i]));
  PetscCall(PetscFree((*ctx)->lg));
  PetscCall(PetscFree(*ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TSMonitorLGCtxNetworkCreate(TS ts, const char host[], const char label[], int x, int y, int m, int n, PetscInt howoften, TSMonitorLGCtxNetwork *ctx)
{
  PetscDraw draw;
  MPI_Comm  comm;
  DM        dm;
  PetscInt  i, Start, End, e, nvar;

  PetscFunctionBegin;
  PetscCall(TSGetDM(ts, &dm));
  PetscCall(PetscObjectGetComm((PetscObject)ts, &comm));
  PetscCall(PetscNew(ctx));
  i = 0;
  /* loop over edges counting number of line graphs needed */
  PetscCall(DMNetworkGetEdgeRange(dm, &Start, &End));
  for (e = Start; e < End; e++) {
    PetscCall(DMNetworkGetComponent(dm, e, ALL_COMPONENTS, NULL, NULL, &nvar));
    if (!nvar) continue;
    i++;
  }
  /* loop over vertices */
  PetscCall(DMNetworkGetVertexRange(dm, &Start, &End));
  for (e = Start; e < End; e++) {
    PetscCall(DMNetworkGetComponent(dm, e, ALL_COMPONENTS, NULL, NULL, &nvar));
    if (!nvar) continue;
    i++;
  }
  (*ctx)->nlg = i;
  PetscCall(PetscMalloc1(i, &(*ctx)->lg));

  i = 0;
  /* loop over edges creating all needed line graphs*/
  PetscCall(DMNetworkGetEdgeRange(dm, &Start, &End));
  for (e = Start; e < End; e++) {
    PetscCall(DMNetworkGetComponent(dm, e, ALL_COMPONENTS, NULL, NULL, &nvar));
    if (!nvar) continue;
    PetscCall(PetscDrawCreate(comm, host, label, x, y, m, n, &draw));
    PetscCall(PetscDrawSetFromOptions(draw));
    PetscCall(PetscDrawLGCreate(draw, nvar, &(*ctx)->lg[i]));
    PetscCall(PetscDrawLGSetFromOptions((*ctx)->lg[i]));
    PetscCall(PetscDrawDestroy(&draw));
    i++;
  }
  /* loop over vertices */
  PetscCall(DMNetworkGetVertexRange(dm, &Start, &End));
  for (e = Start; e < End; e++) {
    PetscCall(DMNetworkGetComponent(dm, e, ALL_COMPONENTS, NULL, NULL, &nvar));
    if (!nvar) continue;
    PetscCall(PetscDrawCreate(comm, host, label, x, y, m, n, &draw));
    PetscCall(PetscDrawSetFromOptions(draw));
    PetscCall(PetscDrawLGCreate(draw, nvar, &(*ctx)->lg[i]));
    PetscCall(PetscDrawLGSetFromOptions((*ctx)->lg[i]));
    PetscCall(PetscDrawDestroy(&draw));
    i++;
  }
  PetscCall(PetscDrawDestroy(&draw));
  (*ctx)->howoften = howoften;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   TSMonitorLGCtxNetworkSolution - Monitors progress of the `TS` solvers for a `DMNETWORK` solution with one window for each vertex and each edge

   Collective

   Input Parameters:
+  ts - the `TS` context
.  step - current time-step
.  ptime - current time
.  u - current solution
-  dctx - the `TSMonitorLGCtxNetwork` object that contains all the options for the monitoring, this is created with `TSMonitorLGCtxCreateNetwork()`

   Options Database Key:
.   -ts_monitor_lg_solution_variables

   Level: intermediate

   Note:
    Each process in a parallel run displays its component solutions in a separate window

*/
PetscErrorCode TSMonitorLGCtxNetworkSolution(TS ts, PetscInt step, PetscReal ptime, Vec u, void *dctx)
{
  TSMonitorLGCtxNetwork ctx = (TSMonitorLGCtxNetwork)dctx;
  const PetscScalar    *xv;
  PetscScalar          *yv;
  PetscInt              i, v, Start, End, offset, nvar, e;
  TSConvergedReason     reason;
  DM                    dm;
  Vec                   uv;

  PetscFunctionBegin;
  if (step < 0) PetscFunctionReturn(PETSC_SUCCESS); /* -1 indicates interpolated solution */
  if (!step) {
    PetscDrawAxis axis;

    for (i = 0; i < ctx->nlg; i++) {
      PetscCall(PetscDrawLGGetAxis(ctx->lg[i], &axis));
      PetscCall(PetscDrawAxisSetLabels(axis, "Solution as function of time", "Time", "Solution"));
      PetscCall(PetscDrawLGReset(ctx->lg[i]));
    }
  }

  if (ctx->semilogy) {
    PetscInt n, j;

    PetscCall(VecDuplicate(u, &uv));
    PetscCall(VecCopy(u, uv));
    PetscCall(VecGetArray(uv, &yv));
    PetscCall(VecGetLocalSize(uv, &n));
    for (j = 0; j < n; j++) {
      if (PetscRealPart(yv[j]) <= 0.0) yv[j] = -12;
      else yv[j] = PetscLog10Real(PetscRealPart(yv[j]));
    }
    xv = yv;
  } else {
    PetscCall(VecGetArrayRead(u, &xv));
  }
  /* iterate over edges */
  PetscCall(TSGetDM(ts, &dm));
  i = 0;
  PetscCall(DMNetworkGetEdgeRange(dm, &Start, &End));
  for (e = Start; e < End; e++) {
    PetscCall(DMNetworkGetComponent(dm, e, ALL_COMPONENTS, NULL, NULL, &nvar));
    if (!nvar) continue;

    PetscCall(DMNetworkGetLocalVecOffset(dm, e, ALL_COMPONENTS, &offset));
    PetscCall(PetscDrawLGAddCommonPoint(ctx->lg[i], ptime, (const PetscReal *)(xv + offset)));
    i++;
  }

  /* iterate over vertices */
  PetscCall(DMNetworkGetVertexRange(dm, &Start, &End));
  for (v = Start; v < End; v++) {
    PetscCall(DMNetworkGetComponent(dm, v, ALL_COMPONENTS, NULL, NULL, &nvar));
    if (!nvar) continue;

    PetscCall(DMNetworkGetLocalVecOffset(dm, v, ALL_COMPONENTS, &offset));
    PetscCall(PetscDrawLGAddCommonPoint(ctx->lg[i], ptime, (const PetscReal *)(xv + offset)));
    i++;
  }
  if (ctx->semilogy) {
    PetscCall(VecRestoreArray(uv, &yv));
    PetscCall(VecDestroy(&uv));
  } else {
    PetscCall(VecRestoreArrayRead(u, &xv));
  }

  PetscCall(TSGetConvergedReason(ts, &reason));
  if (((ctx->howoften > 0) && (!(step % ctx->howoften))) || ((ctx->howoften == -1) && reason)) {
    for (i = 0; i < ctx->nlg; i++) {
      PetscCall(PetscDrawLGDraw(ctx->lg[i]));
      PetscCall(PetscDrawLGSave(ctx->lg[i]));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
