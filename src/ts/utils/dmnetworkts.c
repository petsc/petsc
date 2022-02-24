#include <petsc/private/dmnetworkimpl.h> /*I "petscdmplex.h" I*/
#include <petscts.h>
#include <petscdraw.h>

/*
   TSMonitorLGCtxDestroy - Destroys  line graph contexts that where created with TSMonitorLGCtxNetworkCreate().

   Collective on TSMonitorLGCtx_Network

   Input Parameter:
.  ctx - the monitor context

*/
PetscErrorCode  TSMonitorLGCtxNetworkDestroy(TSMonitorLGCtxNetwork *ctx)
{
  PetscInt       i;

  PetscFunctionBegin;
  for (i=0; i<(*ctx)->nlg; i++) {
    CHKERRQ(PetscDrawLGDestroy(&(*ctx)->lg[i]));
  }
  CHKERRQ(PetscFree((*ctx)->lg));
  CHKERRQ(PetscFree(*ctx));
  PetscFunctionReturn(0);
}

PetscErrorCode  TSMonitorLGCtxNetworkCreate(TS ts,const char host[],const char label[],int x,int y,int m,int n,PetscInt howoften,TSMonitorLGCtxNetwork *ctx)
{
  PetscDraw      draw;
  MPI_Comm       comm;
  DM             dm;
  PetscInt       i,Start,End,e,nvar;

  PetscFunctionBegin;
  CHKERRQ(TSGetDM(ts,&dm));
  CHKERRQ(PetscObjectGetComm((PetscObject)ts,&comm));
  CHKERRQ(PetscNew(ctx));
  i = 0;
  /* loop over edges counting number of line graphs needed */
  CHKERRQ(DMNetworkGetEdgeRange(dm,&Start,&End));
  for (e=Start; e<End; e++) {
    CHKERRQ(DMNetworkGetComponent(dm,e,ALL_COMPONENTS,NULL,NULL,&nvar));
    if (!nvar) continue;
    i++;
  }
  /* loop over vertices */
  CHKERRQ(DMNetworkGetVertexRange(dm,&Start,&End));
  for (e=Start; e<End; e++) {
    CHKERRQ(DMNetworkGetComponent(dm,e,ALL_COMPONENTS,NULL,NULL,&nvar));
    if (!nvar) continue;
    i++;
  }
  (*ctx)->nlg = i;
  CHKERRQ(PetscMalloc1(i,&(*ctx)->lg));

  i = 0;
  /* loop over edges creating all needed line graphs*/
  CHKERRQ(DMNetworkGetEdgeRange(dm,&Start,&End));
  for (e=Start; e<End; e++) {
    CHKERRQ(DMNetworkGetComponent(dm,e,ALL_COMPONENTS,NULL,NULL,&nvar));
    if (!nvar) continue;
    CHKERRQ(PetscDrawCreate(comm,host,label,x,y,m,n,&draw));
    CHKERRQ(PetscDrawSetFromOptions(draw));
    CHKERRQ(PetscDrawLGCreate(draw,nvar,&(*ctx)->lg[i]));
    CHKERRQ(PetscDrawLGSetFromOptions((*ctx)->lg[i]));
    CHKERRQ(PetscDrawDestroy(&draw));
    i++;
  }
  /* loop over vertices */
  CHKERRQ(DMNetworkGetVertexRange(dm,&Start,&End));
  for (e=Start; e<End; e++) {
    CHKERRQ(DMNetworkGetComponent(dm,e,ALL_COMPONENTS,NULL,NULL,&nvar));
    if (!nvar) continue;
    CHKERRQ(PetscDrawCreate(comm,host,label,x,y,m,n,&draw));
    CHKERRQ(PetscDrawSetFromOptions(draw));
    CHKERRQ(PetscDrawLGCreate(draw,nvar,&(*ctx)->lg[i]));
    CHKERRQ(PetscDrawLGSetFromOptions((*ctx)->lg[i]));
    CHKERRQ(PetscDrawDestroy(&draw));
    i++;
  }
  CHKERRQ(PetscDrawDestroy(&draw));
  (*ctx)->howoften = howoften;
  PetscFunctionReturn(0);
}

/*
   TSMonitorLGCtxNetworkSolution - Monitors progress of the TS solvers for a DMNetwork solution with one window for each vertex and each edge

   Collective on TS

   Input Parameters:
+  ts - the TS context
.  step - current time-step
.  ptime - current time
.  u - current solution
-  dctx - the TSMonitorLGCtxNetwork object that contains all the options for the monitoring, this is created with TSMonitorLGCtxCreateNetwork()

   Options Database:
.   -ts_monitor_lg_solution_variables

   Level: intermediate

   Notes:
    Each process in a parallel run displays its component solutions in a separate window

*/
PetscErrorCode  TSMonitorLGCtxNetworkSolution(TS ts,PetscInt step,PetscReal ptime,Vec u,void *dctx)
{
  TSMonitorLGCtxNetwork ctx = (TSMonitorLGCtxNetwork)dctx;
  const PetscScalar     *xv;
  PetscScalar           *yv;
  PetscInt              i,v,Start,End,offset,nvar,e;
  TSConvergedReason     reason;
  DM                    dm;
  Vec                   uv;

  PetscFunctionBegin;
  if (step < 0) PetscFunctionReturn(0); /* -1 indicates interpolated solution */
  if (!step) {
    PetscDrawAxis axis;

    for (i=0; i<ctx->nlg; i++) {
      CHKERRQ(PetscDrawLGGetAxis(ctx->lg[i],&axis));
      CHKERRQ(PetscDrawAxisSetLabels(axis,"Solution as function of time","Time","Solution"));
      CHKERRQ(PetscDrawLGReset(ctx->lg[i]));
    }
  }

  if (ctx->semilogy) {
    PetscInt n,j;

    CHKERRQ(VecDuplicate(u,&uv));
    CHKERRQ(VecCopy(u,uv));
    CHKERRQ(VecGetArray(uv,&yv));
    CHKERRQ(VecGetLocalSize(uv,&n));
    for (j=0; j<n; j++) {
      if (PetscRealPart(yv[j]) <= 0.0) yv[j] = -12;
      else yv[j] = PetscLog10Real(PetscRealPart(yv[j]));
    }
    xv = yv;
  } else {
    CHKERRQ(VecGetArrayRead(u,&xv));
  }
  /* iterate over edges */
  CHKERRQ(TSGetDM(ts,&dm));
  i = 0;
  CHKERRQ(DMNetworkGetEdgeRange(dm,&Start,&End));
  for (e=Start; e<End; e++) {
    CHKERRQ(DMNetworkGetComponent(dm,e,ALL_COMPONENTS,NULL,NULL,&nvar));
    if (!nvar) continue;

    CHKERRQ(DMNetworkGetLocalVecOffset(dm,e,ALL_COMPONENTS,&offset));
    CHKERRQ(PetscDrawLGAddCommonPoint(ctx->lg[i],ptime,(const PetscReal*)(xv+offset)));
    i++;
  }

  /* iterate over vertices */
  CHKERRQ(DMNetworkGetVertexRange(dm,&Start,&End));
  for (v=Start; v<End; v++) {
    CHKERRQ(DMNetworkGetComponent(dm,v,ALL_COMPONENTS,NULL,NULL,&nvar));
    if (!nvar) continue;

    CHKERRQ(DMNetworkGetLocalVecOffset(dm,v,ALL_COMPONENTS,&offset));
    CHKERRQ(PetscDrawLGAddCommonPoint(ctx->lg[i],ptime,(const PetscReal*)(xv+offset)));
    i++;
  }
  if (ctx->semilogy) {
    CHKERRQ(VecRestoreArray(uv,&yv));
    CHKERRQ(VecDestroy(&uv));
  } else {
    CHKERRQ(VecRestoreArrayRead(u,&xv));
  }

  CHKERRQ(TSGetConvergedReason(ts,&reason));
  if (((ctx->howoften > 0) && (!(step % ctx->howoften))) || ((ctx->howoften == -1) && reason)) {
    for (i=0; i<ctx->nlg; i++) {
      CHKERRQ(PetscDrawLGDraw(ctx->lg[i]));
      CHKERRQ(PetscDrawLGSave(ctx->lg[i]));
    }
  }
  PetscFunctionReturn(0);
}
