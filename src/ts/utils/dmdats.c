#include <petscdmda.h>          /*I "petscdmda.h" I*/
#include <petsc/private/dmimpl.h>
#include <petsc/private/tsimpl.h>   /*I "petscts.h" I*/
#include <petscdraw.h>

/* This structure holds the user-provided DMDA callbacks */
typedef struct {
  PetscErrorCode (*ifunctionlocal)(DMDALocalInfo*,PetscReal,void*,void*,void*,void*);
  PetscErrorCode (*rhsfunctionlocal)(DMDALocalInfo*,PetscReal,void*,void*,void*);
  PetscErrorCode (*ijacobianlocal)(DMDALocalInfo*,PetscReal,void*,void*,PetscReal,Mat,Mat,void*);
  PetscErrorCode (*rhsjacobianlocal)(DMDALocalInfo*,PetscReal,void*,Mat,Mat,void*);
  void       *ifunctionlocalctx;
  void       *ijacobianlocalctx;
  void       *rhsfunctionlocalctx;
  void       *rhsjacobianlocalctx;
  InsertMode ifunctionlocalimode;
  InsertMode rhsfunctionlocalimode;
} DMTS_DA;

static PetscErrorCode DMTSDestroy_DMDA(DMTS sdm)
{
  PetscFunctionBegin;
  CHKERRQ(PetscFree(sdm->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMTSDuplicate_DMDA(DMTS oldsdm,DMTS sdm)
{
  PetscFunctionBegin;
  CHKERRQ(PetscNewLog(sdm,(DMTS_DA**)&sdm->data));
  if (oldsdm->data) CHKERRQ(PetscMemcpy(sdm->data,oldsdm->data,sizeof(DMTS_DA)));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMDATSGetContext(DM dm,DMTS sdm,DMTS_DA **dmdats)
{
  PetscFunctionBegin;
  *dmdats = NULL;
  if (!sdm->data) {
    CHKERRQ(PetscNewLog(dm,(DMTS_DA**)&sdm->data));
    sdm->ops->destroy   = DMTSDestroy_DMDA;
    sdm->ops->duplicate = DMTSDuplicate_DMDA;
  }
  *dmdats = (DMTS_DA*)sdm->data;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSComputeIFunction_DMDA(TS ts,PetscReal ptime,Vec X,Vec Xdot,Vec F,void *ctx)
{
  DM             dm;
  DMTS_DA        *dmdats = (DMTS_DA*)ctx;
  DMDALocalInfo  info;
  Vec            Xloc,Xdotloc;
  void           *x,*f,*xdot;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,3);
  PetscValidHeaderSpecific(F,VEC_CLASSID,5);
  PetscCheck(dmdats->ifunctionlocal,PetscObjectComm((PetscObject)ts),PETSC_ERR_PLIB,"Corrupt context");
  CHKERRQ(TSGetDM(ts,&dm));
  CHKERRQ(DMGetLocalVector(dm,&Xdotloc));
  CHKERRQ(DMGlobalToLocalBegin(dm,Xdot,INSERT_VALUES,Xdotloc));
  CHKERRQ(DMGlobalToLocalEnd(dm,Xdot,INSERT_VALUES,Xdotloc));
  CHKERRQ(DMGetLocalVector(dm,&Xloc));
  CHKERRQ(DMGlobalToLocalBegin(dm,X,INSERT_VALUES,Xloc));
  CHKERRQ(DMGlobalToLocalEnd(dm,X,INSERT_VALUES,Xloc));
  CHKERRQ(DMDAGetLocalInfo(dm,&info));
  CHKERRQ(DMDAVecGetArray(dm,Xloc,&x));
  CHKERRQ(DMDAVecGetArray(dm,Xdotloc,&xdot));
  switch (dmdats->ifunctionlocalimode) {
  case INSERT_VALUES: {
    CHKERRQ(DMDAVecGetArray(dm,F,&f));
    CHKMEMQ;
    CHKERRQ((*dmdats->ifunctionlocal)(&info,ptime,x,xdot,f,dmdats->ifunctionlocalctx));
    CHKMEMQ;
    CHKERRQ(DMDAVecRestoreArray(dm,F,&f));
  } break;
  case ADD_VALUES: {
    Vec Floc;
    CHKERRQ(DMGetLocalVector(dm,&Floc));
    CHKERRQ(VecZeroEntries(Floc));
    CHKERRQ(DMDAVecGetArray(dm,Floc,&f));
    CHKMEMQ;
    CHKERRQ((*dmdats->ifunctionlocal)(&info,ptime,x,xdot,f,dmdats->ifunctionlocalctx));
    CHKMEMQ;
    CHKERRQ(DMDAVecRestoreArray(dm,Floc,&f));
    CHKERRQ(VecZeroEntries(F));
    CHKERRQ(DMLocalToGlobalBegin(dm,Floc,ADD_VALUES,F));
    CHKERRQ(DMLocalToGlobalEnd(dm,Floc,ADD_VALUES,F));
    CHKERRQ(DMRestoreLocalVector(dm,&Floc));
  } break;
  default: SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_ARG_INCOMP,"Cannot use imode=%d",(int)dmdats->ifunctionlocalimode);
  }
  CHKERRQ(DMDAVecRestoreArray(dm,Xloc,&x));
  CHKERRQ(DMRestoreLocalVector(dm,&Xloc));
  CHKERRQ(DMDAVecRestoreArray(dm,Xdotloc,&xdot));
  CHKERRQ(DMRestoreLocalVector(dm,&Xdotloc));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSComputeIJacobian_DMDA(TS ts,PetscReal ptime,Vec X,Vec Xdot,PetscReal shift,Mat A,Mat B,void *ctx)
{
  DM             dm;
  DMTS_DA        *dmdats = (DMTS_DA*)ctx;
  DMDALocalInfo  info;
  Vec            Xloc;
  void           *x,*xdot;

  PetscFunctionBegin;
  PetscCheck(dmdats->ifunctionlocal,PetscObjectComm((PetscObject)ts),PETSC_ERR_PLIB,"Corrupt context");
  CHKERRQ(TSGetDM(ts,&dm));

  if (dmdats->ijacobianlocal) {
    CHKERRQ(DMGetLocalVector(dm,&Xloc));
    CHKERRQ(DMGlobalToLocalBegin(dm,X,INSERT_VALUES,Xloc));
    CHKERRQ(DMGlobalToLocalEnd(dm,X,INSERT_VALUES,Xloc));
    CHKERRQ(DMDAGetLocalInfo(dm,&info));
    CHKERRQ(DMDAVecGetArray(dm,Xloc,&x));
    CHKERRQ(DMDAVecGetArray(dm,Xdot,&xdot));
    CHKMEMQ;
    CHKERRQ((*dmdats->ijacobianlocal)(&info,ptime,x,xdot,shift,A,B,dmdats->ijacobianlocalctx));
    CHKMEMQ;
    CHKERRQ(DMDAVecRestoreArray(dm,Xloc,&x));
    CHKERRQ(DMDAVecRestoreArray(dm,Xdot,&xdot));
    CHKERRQ(DMRestoreLocalVector(dm,&Xloc));
  } else SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_PLIB,"TSComputeIJacobian_DMDA() called without calling DMDATSSetIJacobian()");
  /* This will be redundant if the user called both, but it's too common to forget. */
  if (A != B) {
    CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSComputeRHSFunction_DMDA(TS ts,PetscReal ptime,Vec X,Vec F,void *ctx)
{
  DM             dm;
  DMTS_DA        *dmdats = (DMTS_DA*)ctx;
  DMDALocalInfo  info;
  Vec            Xloc;
  void           *x,*f;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,3);
  PetscValidHeaderSpecific(F,VEC_CLASSID,4);
  PetscCheck(dmdats->rhsfunctionlocal,PetscObjectComm((PetscObject)ts),PETSC_ERR_PLIB,"Corrupt context");
  CHKERRQ(TSGetDM(ts,&dm));
  CHKERRQ(DMGetLocalVector(dm,&Xloc));
  CHKERRQ(DMGlobalToLocalBegin(dm,X,INSERT_VALUES,Xloc));
  CHKERRQ(DMGlobalToLocalEnd(dm,X,INSERT_VALUES,Xloc));
  CHKERRQ(DMDAGetLocalInfo(dm,&info));
  CHKERRQ(DMDAVecGetArray(dm,Xloc,&x));
  switch (dmdats->rhsfunctionlocalimode) {
  case INSERT_VALUES: {
    CHKERRQ(DMDAVecGetArray(dm,F,&f));
    CHKMEMQ;
    CHKERRQ((*dmdats->rhsfunctionlocal)(&info,ptime,x,f,dmdats->rhsfunctionlocalctx));
    CHKMEMQ;
    CHKERRQ(DMDAVecRestoreArray(dm,F,&f));
  } break;
  case ADD_VALUES: {
    Vec Floc;
    CHKERRQ(DMGetLocalVector(dm,&Floc));
    CHKERRQ(VecZeroEntries(Floc));
    CHKERRQ(DMDAVecGetArray(dm,Floc,&f));
    CHKMEMQ;
    CHKERRQ((*dmdats->rhsfunctionlocal)(&info,ptime,x,f,dmdats->rhsfunctionlocalctx));
    CHKMEMQ;
    CHKERRQ(DMDAVecRestoreArray(dm,Floc,&f));
    CHKERRQ(VecZeroEntries(F));
    CHKERRQ(DMLocalToGlobalBegin(dm,Floc,ADD_VALUES,F));
    CHKERRQ(DMLocalToGlobalEnd(dm,Floc,ADD_VALUES,F));
    CHKERRQ(DMRestoreLocalVector(dm,&Floc));
  } break;
  default: SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_ARG_INCOMP,"Cannot use imode=%d",(int)dmdats->rhsfunctionlocalimode);
  }
  CHKERRQ(DMDAVecRestoreArray(dm,Xloc,&x));
  CHKERRQ(DMRestoreLocalVector(dm,&Xloc));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSComputeRHSJacobian_DMDA(TS ts,PetscReal ptime,Vec X,Mat A,Mat B,void *ctx)
{
  DM             dm;
  DMTS_DA        *dmdats = (DMTS_DA*)ctx;
  DMDALocalInfo  info;
  Vec            Xloc;
  void           *x;

  PetscFunctionBegin;
  PetscCheck(dmdats->rhsfunctionlocal,PetscObjectComm((PetscObject)ts),PETSC_ERR_PLIB,"Corrupt context");
  CHKERRQ(TSGetDM(ts,&dm));

  if (dmdats->rhsjacobianlocal) {
    CHKERRQ(DMGetLocalVector(dm,&Xloc));
    CHKERRQ(DMGlobalToLocalBegin(dm,X,INSERT_VALUES,Xloc));
    CHKERRQ(DMGlobalToLocalEnd(dm,X,INSERT_VALUES,Xloc));
    CHKERRQ(DMDAGetLocalInfo(dm,&info));
    CHKERRQ(DMDAVecGetArray(dm,Xloc,&x));
    CHKMEMQ;
    CHKERRQ((*dmdats->rhsjacobianlocal)(&info,ptime,x,A,B,dmdats->rhsjacobianlocalctx));
    CHKMEMQ;
    CHKERRQ(DMDAVecRestoreArray(dm,Xloc,&x));
    CHKERRQ(DMRestoreLocalVector(dm,&Xloc));
  } else SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_PLIB,"TSComputeRHSJacobian_DMDA() called without calling DMDATSSetRHSJacobian()");
  /* This will be redundant if the user called both, but it's too common to forget. */
  if (A != B) {
    CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(0);
}

/*@C
   DMDATSSetRHSFunctionLocal - set a local residual evaluation function

   Logically Collective

   Input Parameters:
+  dm - DM to associate callback with
.  imode - insert mode for the residual
.  func - local residual evaluation
-  ctx - optional context for local residual evaluation

   Calling sequence for func:

$ func(DMDALocalInfo info,PetscReal t,void *x,void *f,void *ctx)

+  info - DMDALocalInfo defining the subdomain to evaluate the residual on
.  t - time at which to evaluate residual
.  x - array of local state information
.  f - output array of local residual information
-  ctx - optional user context

   Level: beginner

.seealso: DMTSSetRHSFunction(), DMDATSSetRHSJacobianLocal(), DMDASNESSetFunctionLocal()
@*/
PetscErrorCode DMDATSSetRHSFunctionLocal(DM dm,InsertMode imode,DMDATSRHSFunctionLocal func,void *ctx)
{
  DMTS           sdm;
  DMTS_DA        *dmdats;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  CHKERRQ(DMGetDMTSWrite(dm,&sdm));
  CHKERRQ(DMDATSGetContext(dm,sdm,&dmdats));
  dmdats->rhsfunctionlocalimode = imode;
  dmdats->rhsfunctionlocal      = func;
  dmdats->rhsfunctionlocalctx   = ctx;
  CHKERRQ(DMTSSetRHSFunction(dm,TSComputeRHSFunction_DMDA,dmdats));
  PetscFunctionReturn(0);
}

/*@C
   DMDATSSetRHSJacobianLocal - set a local residual evaluation function

   Logically Collective

   Input Parameters:
+  dm    - DM to associate callback with
.  func  - local RHS Jacobian evaluation routine
-  ctx   - optional context for local jacobian evaluation

   Calling sequence for func:

$ func(DMDALocalInfo* info,PetscReal t,void* x,Mat J,Mat B,void *ctx);

+  info - DMDALocalInfo defining the subdomain to evaluate the residual on
.  t    - time at which to evaluate residual
.  x    - array of local state information
.  J    - Jacobian matrix
.  B    - preconditioner matrix; often same as J
-  ctx  - optional context passed above

   Level: beginner

.seealso: DMTSSetRHSJacobian(), DMDATSSetRHSFunctionLocal(), DMDASNESSetJacobianLocal()
@*/
PetscErrorCode DMDATSSetRHSJacobianLocal(DM dm,DMDATSRHSJacobianLocal func,void *ctx)
{
  DMTS           sdm;
  DMTS_DA        *dmdats;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  CHKERRQ(DMGetDMTSWrite(dm,&sdm));
  CHKERRQ(DMDATSGetContext(dm,sdm,&dmdats));
  dmdats->rhsjacobianlocal    = func;
  dmdats->rhsjacobianlocalctx = ctx;
  CHKERRQ(DMTSSetRHSJacobian(dm,TSComputeRHSJacobian_DMDA,dmdats));
  PetscFunctionReturn(0);
}

/*@C
   DMDATSSetIFunctionLocal - set a local residual evaluation function

   Logically Collective

   Input Parameters:
+  dm   - DM to associate callback with
.  func - local residual evaluation
-  ctx  - optional context for local residual evaluation

   Calling sequence for func:
+  info - DMDALocalInfo defining the subdomain to evaluate the residual on
.  t    - time at which to evaluate residual
.  x    - array of local state information
.  xdot - array of local time derivative information
.  f    - output array of local function evaluation information
-  ctx - optional context passed above

   Level: beginner

.seealso: DMTSSetIFunction(), DMDATSSetIJacobianLocal(), DMDASNESSetFunctionLocal()
@*/
PetscErrorCode DMDATSSetIFunctionLocal(DM dm,InsertMode imode,DMDATSIFunctionLocal func,void *ctx)
{
  DMTS           sdm;
  DMTS_DA        *dmdats;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  CHKERRQ(DMGetDMTSWrite(dm,&sdm));
  CHKERRQ(DMDATSGetContext(dm,sdm,&dmdats));
  dmdats->ifunctionlocalimode = imode;
  dmdats->ifunctionlocal      = func;
  dmdats->ifunctionlocalctx   = ctx;
  CHKERRQ(DMTSSetIFunction(dm,TSComputeIFunction_DMDA,dmdats));
  PetscFunctionReturn(0);
}

/*@C
   DMDATSSetIJacobianLocal - set a local residual evaluation function

   Logically Collective

   Input Parameters:
+  dm   - DM to associate callback with
.  func - local residual evaluation
-  ctx   - optional context for local residual evaluation

   Calling sequence for func:

$ func(DMDALocalInfo* info,PetscReal t,void* x,void *xdot,PetscScalar shift,Mat J,Mat B,void *ctx);

+  info - DMDALocalInfo defining the subdomain to evaluate the residual on
.  t    - time at which to evaluate the jacobian
.  x    - array of local state information
.  xdot - time derivative at this state
.  shift - see TSSetIJacobian() for the meaning of this parameter
.  J    - Jacobian matrix
.  B    - preconditioner matrix; often same as J
-  ctx  - optional context passed above

   Level: beginner

.seealso: DMTSSetJacobian(), DMDATSSetIFunctionLocal(), DMDASNESSetJacobianLocal()
@*/
PetscErrorCode DMDATSSetIJacobianLocal(DM dm,DMDATSIJacobianLocal func,void *ctx)
{
  DMTS           sdm;
  DMTS_DA        *dmdats;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  CHKERRQ(DMGetDMTSWrite(dm,&sdm));
  CHKERRQ(DMDATSGetContext(dm,sdm,&dmdats));
  dmdats->ijacobianlocal    = func;
  dmdats->ijacobianlocalctx = ctx;
  CHKERRQ(DMTSSetIJacobian(dm,TSComputeIJacobian_DMDA,dmdats));
  PetscFunctionReturn(0);
}

PetscErrorCode TSMonitorDMDARayDestroy(void **mctx)
{
  TSMonitorDMDARayCtx *rayctx = (TSMonitorDMDARayCtx *) *mctx;

  PetscFunctionBegin;
  if (rayctx->lgctx) CHKERRQ(TSMonitorLGCtxDestroy(&rayctx->lgctx));
  CHKERRQ(VecDestroy(&rayctx->ray));
  CHKERRQ(VecScatterDestroy(&rayctx->scatter));
  CHKERRQ(PetscViewerDestroy(&rayctx->viewer));
  CHKERRQ(PetscFree(rayctx));
  PetscFunctionReturn(0);
}

PetscErrorCode TSMonitorDMDARay(TS ts,PetscInt steps,PetscReal time,Vec u,void *mctx)
{
  TSMonitorDMDARayCtx *rayctx = (TSMonitorDMDARayCtx*)mctx;
  Vec                 solution;

  PetscFunctionBegin;
  CHKERRQ(TSGetSolution(ts,&solution));
  CHKERRQ(VecScatterBegin(rayctx->scatter,solution,rayctx->ray,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(rayctx->scatter,solution,rayctx->ray,INSERT_VALUES,SCATTER_FORWARD));
  if (rayctx->viewer) {
    CHKERRQ(VecView(rayctx->ray,rayctx->viewer));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode  TSMonitorLGDMDARay(TS ts, PetscInt step, PetscReal ptime, Vec u, void *ctx)
{
  TSMonitorDMDARayCtx *rayctx = (TSMonitorDMDARayCtx *) ctx;
  TSMonitorLGCtx       lgctx  = (TSMonitorLGCtx) rayctx->lgctx;
  Vec                  v      = rayctx->ray;
  const PetscScalar   *a;
  PetscInt             dim;

  PetscFunctionBegin;
  CHKERRQ(VecScatterBegin(rayctx->scatter, u, v, INSERT_VALUES, SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(rayctx->scatter, u, v, INSERT_VALUES, SCATTER_FORWARD));
  if (!step) {
    PetscDrawAxis axis;

    CHKERRQ(PetscDrawLGGetAxis(lgctx->lg, &axis));
    CHKERRQ(PetscDrawAxisSetLabels(axis, "Solution Ray as function of time", "Time", "Solution"));
    CHKERRQ(VecGetLocalSize(rayctx->ray, &dim));
    CHKERRQ(PetscDrawLGSetDimension(lgctx->lg, dim));
    CHKERRQ(PetscDrawLGReset(lgctx->lg));
  }
  CHKERRQ(VecGetArrayRead(v, &a));
#if defined(PETSC_USE_COMPLEX)
  {
    PetscReal *areal;
    PetscInt   i,n;
    CHKERRQ(VecGetLocalSize(v, &n));
    CHKERRQ(PetscMalloc1(n, &areal));
    for (i = 0; i < n; ++i) areal[i] = PetscRealPart(a[i]);
    CHKERRQ(PetscDrawLGAddCommonPoint(lgctx->lg, ptime, areal));
    CHKERRQ(PetscFree(areal));
  }
#else
  CHKERRQ(PetscDrawLGAddCommonPoint(lgctx->lg, ptime, a));
#endif
  CHKERRQ(VecRestoreArrayRead(v, &a));
  if (((lgctx->howoften > 0) && (!(step % lgctx->howoften))) || ((lgctx->howoften == -1) && ts->reason)) {
    CHKERRQ(PetscDrawLGDraw(lgctx->lg));
    CHKERRQ(PetscDrawLGSave(lgctx->lg));
  }
  PetscFunctionReturn(0);
}
