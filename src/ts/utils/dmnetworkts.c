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
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  for (i=0; i<(*ctx)->nlg; i++) {
    ierr = PetscDrawLGDestroy(&(*ctx)->lg[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree((*ctx)->lg);CHKERRQ(ierr);
  ierr = PetscFree(*ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode  TSMonitorLGCtxNetworkCreate(TS ts,const char host[],const char label[],int x,int y,int m,int n,PetscInt howoften,TSMonitorLGCtxNetwork *ctx)
{
  PetscDraw      draw;
  PetscErrorCode ierr;
  MPI_Comm       comm;
  DM             dm;
  PetscInt       i,Start,End,e,nvar;

  PetscFunctionBegin;
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)ts,&comm);CHKERRQ(ierr);
  ierr = PetscNew(ctx);CHKERRQ(ierr);
  i = 0;
  /* loop over edges counting number of line graphs needed */
  ierr = DMNetworkGetEdgeRange(dm,&Start,&End);CHKERRQ(ierr);
  for (e=Start; e<End; e++) {
    ierr = DMNetworkGetNumVariables(dm,e,&nvar);CHKERRQ(ierr);
    if (!nvar) continue;
    i++;
  }
  /* loop over vertices */
  ierr = DMNetworkGetVertexRange(dm,&Start,&End);CHKERRQ(ierr);
  for (e=Start; e<End; e++) {
    ierr = DMNetworkGetNumVariables(dm,e,&nvar);CHKERRQ(ierr);
    if (!nvar) continue;
    i++;
  }
  (*ctx)->nlg = i;
  ierr = PetscMalloc1(i,&(*ctx)->lg);CHKERRQ(ierr);

  i = 0;
  /* loop over edges creating all needed line graphs*/
  ierr = DMNetworkGetEdgeRange(dm,&Start,&End);CHKERRQ(ierr);
  for (e=Start; e<End; e++) {
    ierr = DMNetworkGetNumVariables(dm,e,&nvar);CHKERRQ(ierr);
    if (!nvar) continue;
    ierr = PetscDrawCreate(comm,host,label,x,y,m,n,&draw);CHKERRQ(ierr);
    ierr = PetscDrawSetFromOptions(draw);CHKERRQ(ierr);
    ierr = PetscDrawLGCreate(draw,nvar,&(*ctx)->lg[i]);CHKERRQ(ierr);
    ierr = PetscDrawLGSetFromOptions((*ctx)->lg[i]);CHKERRQ(ierr);
    ierr = PetscDrawDestroy(&draw);CHKERRQ(ierr);
    i++;
  }
  /* loop over vertices */
  ierr = DMNetworkGetVertexRange(dm,&Start,&End);CHKERRQ(ierr);
  for (e=Start; e<End; e++) {
    ierr = DMNetworkGetNumVariables(dm,e,&nvar);CHKERRQ(ierr);
    if (!nvar) continue;
    ierr = PetscDrawCreate(comm,host,label,x,y,m,n,&draw);CHKERRQ(ierr);
    ierr = PetscDrawSetFromOptions(draw);CHKERRQ(ierr);
    ierr = PetscDrawLGCreate(draw,nvar,&(*ctx)->lg[i]);CHKERRQ(ierr);
    ierr = PetscDrawLGSetFromOptions((*ctx)->lg[i]);CHKERRQ(ierr);
    ierr = PetscDrawDestroy(&draw);CHKERRQ(ierr);
    i++;
  }
  ierr = PetscDrawDestroy(&draw);CHKERRQ(ierr);
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
  PetscErrorCode        ierr;
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
      ierr = PetscDrawLGGetAxis(ctx->lg[i],&axis);CHKERRQ(ierr);
      ierr = PetscDrawAxisSetLabels(axis,"Solution as function of time","Time","Solution");CHKERRQ(ierr);
      ierr = PetscDrawLGReset(ctx->lg[i]);CHKERRQ(ierr);
    }
  }

  if (ctx->semilogy) {
    PetscInt n,j;

    ierr = VecDuplicate(u,&uv);CHKERRQ(ierr);
    ierr = VecCopy(u,uv);CHKERRQ(ierr);
    ierr = VecGetArray(uv,&yv);CHKERRQ(ierr);
    ierr = VecGetLocalSize(uv,&n);CHKERRQ(ierr);
    for (j=0; j<n; j++) {
      if (PetscRealPart(yv[j]) <= 0.0) yv[j] = -12;
      else yv[j] = PetscLog10Real(PetscRealPart(yv[j]));
    }
    xv = yv;
  } else {
    ierr = VecGetArrayRead(u,&xv);CHKERRQ(ierr);
  }
  /* iterate over edges */
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  i = 0;
  ierr = DMNetworkGetEdgeRange(dm,&Start,&End);CHKERRQ(ierr);
  for (e=Start; e<End; e++) {
    ierr = DMNetworkGetNumVariables(dm,e,&nvar);CHKERRQ(ierr);
    if (!nvar) continue;

    ierr = DMNetworkGetVariableOffset(dm,e,&offset);CHKERRQ(ierr);
    ierr = PetscDrawLGAddCommonPoint(ctx->lg[i],ptime,(const PetscReal*)(xv+offset));CHKERRQ(ierr);
    i++;
  }

  /* iterate over vertices */
  ierr = DMNetworkGetVertexRange(dm,&Start,&End);CHKERRQ(ierr);
  for (v=Start; v<End; v++) {
    ierr = DMNetworkGetNumVariables(dm,v,&nvar);CHKERRQ(ierr);
    if (!nvar) continue;

    ierr = DMNetworkGetVariableOffset(dm,v,&offset);CHKERRQ(ierr);
    ierr = PetscDrawLGAddCommonPoint(ctx->lg[i],ptime,(const PetscReal*)(xv+offset));CHKERRQ(ierr);
    i++;
  }
  if (ctx->semilogy) {
    ierr = VecRestoreArray(uv,&yv);CHKERRQ(ierr);
    ierr = VecDestroy(&uv);CHKERRQ(ierr);
  } else {
    ierr = VecRestoreArrayRead(u,&xv);CHKERRQ(ierr);
  }

  ierr = TSGetConvergedReason(ts,&reason);CHKERRQ(ierr);
  if (((ctx->howoften > 0) && (!(step % ctx->howoften))) || ((ctx->howoften == -1) && reason)) {
    for (i=0; i<ctx->nlg; i++) {
      ierr = PetscDrawLGDraw(ctx->lg[i]);CHKERRQ(ierr);
      ierr = PetscDrawLGSave(ctx->lg[i]);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}
