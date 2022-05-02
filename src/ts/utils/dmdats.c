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
  PetscCall(PetscFree(sdm->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMTSDuplicate_DMDA(DMTS oldsdm,DMTS sdm)
{
  PetscFunctionBegin;
  PetscCall(PetscNewLog(sdm,(DMTS_DA**)&sdm->data));
  if (oldsdm->data) PetscCall(PetscMemcpy(sdm->data,oldsdm->data,sizeof(DMTS_DA)));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMDATSGetContext(DM dm,DMTS sdm,DMTS_DA **dmdats)
{
  PetscFunctionBegin;
  *dmdats = NULL;
  if (!sdm->data) {
    PetscCall(PetscNewLog(dm,(DMTS_DA**)&sdm->data));
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
  PetscCall(TSGetDM(ts,&dm));
  PetscCall(DMGetLocalVector(dm,&Xdotloc));
  PetscCall(DMGlobalToLocalBegin(dm,Xdot,INSERT_VALUES,Xdotloc));
  PetscCall(DMGlobalToLocalEnd(dm,Xdot,INSERT_VALUES,Xdotloc));
  PetscCall(DMGetLocalVector(dm,&Xloc));
  PetscCall(DMGlobalToLocalBegin(dm,X,INSERT_VALUES,Xloc));
  PetscCall(DMGlobalToLocalEnd(dm,X,INSERT_VALUES,Xloc));
  PetscCall(DMDAGetLocalInfo(dm,&info));
  PetscCall(DMDAVecGetArray(dm,Xloc,&x));
  PetscCall(DMDAVecGetArray(dm,Xdotloc,&xdot));
  switch (dmdats->ifunctionlocalimode) {
  case INSERT_VALUES: {
    PetscCall(DMDAVecGetArray(dm,F,&f));
    CHKMEMQ;
    PetscCall((*dmdats->ifunctionlocal)(&info,ptime,x,xdot,f,dmdats->ifunctionlocalctx));
    CHKMEMQ;
    PetscCall(DMDAVecRestoreArray(dm,F,&f));
  } break;
  case ADD_VALUES: {
    Vec Floc;
    PetscCall(DMGetLocalVector(dm,&Floc));
    PetscCall(VecZeroEntries(Floc));
    PetscCall(DMDAVecGetArray(dm,Floc,&f));
    CHKMEMQ;
    PetscCall((*dmdats->ifunctionlocal)(&info,ptime,x,xdot,f,dmdats->ifunctionlocalctx));
    CHKMEMQ;
    PetscCall(DMDAVecRestoreArray(dm,Floc,&f));
    PetscCall(VecZeroEntries(F));
    PetscCall(DMLocalToGlobalBegin(dm,Floc,ADD_VALUES,F));
    PetscCall(DMLocalToGlobalEnd(dm,Floc,ADD_VALUES,F));
    PetscCall(DMRestoreLocalVector(dm,&Floc));
  } break;
  default: SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_ARG_INCOMP,"Cannot use imode=%d",(int)dmdats->ifunctionlocalimode);
  }
  PetscCall(DMDAVecRestoreArray(dm,Xloc,&x));
  PetscCall(DMRestoreLocalVector(dm,&Xloc));
  PetscCall(DMDAVecRestoreArray(dm,Xdotloc,&xdot));
  PetscCall(DMRestoreLocalVector(dm,&Xdotloc));
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
  PetscCall(TSGetDM(ts,&dm));

  if (dmdats->ijacobianlocal) {
    PetscCall(DMGetLocalVector(dm,&Xloc));
    PetscCall(DMGlobalToLocalBegin(dm,X,INSERT_VALUES,Xloc));
    PetscCall(DMGlobalToLocalEnd(dm,X,INSERT_VALUES,Xloc));
    PetscCall(DMDAGetLocalInfo(dm,&info));
    PetscCall(DMDAVecGetArray(dm,Xloc,&x));
    PetscCall(DMDAVecGetArray(dm,Xdot,&xdot));
    CHKMEMQ;
    PetscCall((*dmdats->ijacobianlocal)(&info,ptime,x,xdot,shift,A,B,dmdats->ijacobianlocalctx));
    CHKMEMQ;
    PetscCall(DMDAVecRestoreArray(dm,Xloc,&x));
    PetscCall(DMDAVecRestoreArray(dm,Xdot,&xdot));
    PetscCall(DMRestoreLocalVector(dm,&Xloc));
  } else SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_PLIB,"TSComputeIJacobian_DMDA() called without calling DMDATSSetIJacobian()");
  /* This will be redundant if the user called both, but it's too common to forget. */
  if (A != B) {
    PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
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
  PetscCall(TSGetDM(ts,&dm));
  PetscCall(DMGetLocalVector(dm,&Xloc));
  PetscCall(DMGlobalToLocalBegin(dm,X,INSERT_VALUES,Xloc));
  PetscCall(DMGlobalToLocalEnd(dm,X,INSERT_VALUES,Xloc));
  PetscCall(DMDAGetLocalInfo(dm,&info));
  PetscCall(DMDAVecGetArray(dm,Xloc,&x));
  switch (dmdats->rhsfunctionlocalimode) {
  case INSERT_VALUES: {
    PetscCall(DMDAVecGetArray(dm,F,&f));
    CHKMEMQ;
    PetscCall((*dmdats->rhsfunctionlocal)(&info,ptime,x,f,dmdats->rhsfunctionlocalctx));
    CHKMEMQ;
    PetscCall(DMDAVecRestoreArray(dm,F,&f));
  } break;
  case ADD_VALUES: {
    Vec Floc;
    PetscCall(DMGetLocalVector(dm,&Floc));
    PetscCall(VecZeroEntries(Floc));
    PetscCall(DMDAVecGetArray(dm,Floc,&f));
    CHKMEMQ;
    PetscCall((*dmdats->rhsfunctionlocal)(&info,ptime,x,f,dmdats->rhsfunctionlocalctx));
    CHKMEMQ;
    PetscCall(DMDAVecRestoreArray(dm,Floc,&f));
    PetscCall(VecZeroEntries(F));
    PetscCall(DMLocalToGlobalBegin(dm,Floc,ADD_VALUES,F));
    PetscCall(DMLocalToGlobalEnd(dm,Floc,ADD_VALUES,F));
    PetscCall(DMRestoreLocalVector(dm,&Floc));
  } break;
  default: SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_ARG_INCOMP,"Cannot use imode=%d",(int)dmdats->rhsfunctionlocalimode);
  }
  PetscCall(DMDAVecRestoreArray(dm,Xloc,&x));
  PetscCall(DMRestoreLocalVector(dm,&Xloc));
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
  PetscCall(TSGetDM(ts,&dm));

  if (dmdats->rhsjacobianlocal) {
    PetscCall(DMGetLocalVector(dm,&Xloc));
    PetscCall(DMGlobalToLocalBegin(dm,X,INSERT_VALUES,Xloc));
    PetscCall(DMGlobalToLocalEnd(dm,X,INSERT_VALUES,Xloc));
    PetscCall(DMDAGetLocalInfo(dm,&info));
    PetscCall(DMDAVecGetArray(dm,Xloc,&x));
    CHKMEMQ;
    PetscCall((*dmdats->rhsjacobianlocal)(&info,ptime,x,A,B,dmdats->rhsjacobianlocalctx));
    CHKMEMQ;
    PetscCall(DMDAVecRestoreArray(dm,Xloc,&x));
    PetscCall(DMRestoreLocalVector(dm,&Xloc));
  } else SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_PLIB,"TSComputeRHSJacobian_DMDA() called without calling DMDATSSetRHSJacobian()");
  /* This will be redundant if the user called both, but it's too common to forget. */
  if (A != B) {
    PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
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

.seealso: `DMTSSetRHSFunction()`, `DMDATSSetRHSJacobianLocal()`, `DMDASNESSetFunctionLocal()`
@*/
PetscErrorCode DMDATSSetRHSFunctionLocal(DM dm,InsertMode imode,DMDATSRHSFunctionLocal func,void *ctx)
{
  DMTS           sdm;
  DMTS_DA        *dmdats;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscCall(DMGetDMTSWrite(dm,&sdm));
  PetscCall(DMDATSGetContext(dm,sdm,&dmdats));
  dmdats->rhsfunctionlocalimode = imode;
  dmdats->rhsfunctionlocal      = func;
  dmdats->rhsfunctionlocalctx   = ctx;
  PetscCall(DMTSSetRHSFunction(dm,TSComputeRHSFunction_DMDA,dmdats));
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

.seealso: `DMTSSetRHSJacobian()`, `DMDATSSetRHSFunctionLocal()`, `DMDASNESSetJacobianLocal()`
@*/
PetscErrorCode DMDATSSetRHSJacobianLocal(DM dm,DMDATSRHSJacobianLocal func,void *ctx)
{
  DMTS           sdm;
  DMTS_DA        *dmdats;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscCall(DMGetDMTSWrite(dm,&sdm));
  PetscCall(DMDATSGetContext(dm,sdm,&dmdats));
  dmdats->rhsjacobianlocal    = func;
  dmdats->rhsjacobianlocalctx = ctx;
  PetscCall(DMTSSetRHSJacobian(dm,TSComputeRHSJacobian_DMDA,dmdats));
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

.seealso: `DMTSSetIFunction()`, `DMDATSSetIJacobianLocal()`, `DMDASNESSetFunctionLocal()`
@*/
PetscErrorCode DMDATSSetIFunctionLocal(DM dm,InsertMode imode,DMDATSIFunctionLocal func,void *ctx)
{
  DMTS           sdm;
  DMTS_DA        *dmdats;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscCall(DMGetDMTSWrite(dm,&sdm));
  PetscCall(DMDATSGetContext(dm,sdm,&dmdats));
  dmdats->ifunctionlocalimode = imode;
  dmdats->ifunctionlocal      = func;
  dmdats->ifunctionlocalctx   = ctx;
  PetscCall(DMTSSetIFunction(dm,TSComputeIFunction_DMDA,dmdats));
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

.seealso: `DMTSSetJacobian()`, `DMDATSSetIFunctionLocal()`, `DMDASNESSetJacobianLocal()`
@*/
PetscErrorCode DMDATSSetIJacobianLocal(DM dm,DMDATSIJacobianLocal func,void *ctx)
{
  DMTS           sdm;
  DMTS_DA        *dmdats;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscCall(DMGetDMTSWrite(dm,&sdm));
  PetscCall(DMDATSGetContext(dm,sdm,&dmdats));
  dmdats->ijacobianlocal    = func;
  dmdats->ijacobianlocalctx = ctx;
  PetscCall(DMTSSetIJacobian(dm,TSComputeIJacobian_DMDA,dmdats));
  PetscFunctionReturn(0);
}

PetscErrorCode TSMonitorDMDARayDestroy(void **mctx)
{
  TSMonitorDMDARayCtx *rayctx = (TSMonitorDMDARayCtx *) *mctx;

  PetscFunctionBegin;
  if (rayctx->lgctx) PetscCall(TSMonitorLGCtxDestroy(&rayctx->lgctx));
  PetscCall(VecDestroy(&rayctx->ray));
  PetscCall(VecScatterDestroy(&rayctx->scatter));
  PetscCall(PetscViewerDestroy(&rayctx->viewer));
  PetscCall(PetscFree(rayctx));
  PetscFunctionReturn(0);
}

PetscErrorCode TSMonitorDMDARay(TS ts,PetscInt steps,PetscReal time,Vec u,void *mctx)
{
  TSMonitorDMDARayCtx *rayctx = (TSMonitorDMDARayCtx*)mctx;
  Vec                 solution;

  PetscFunctionBegin;
  PetscCall(TSGetSolution(ts,&solution));
  PetscCall(VecScatterBegin(rayctx->scatter,solution,rayctx->ray,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(rayctx->scatter,solution,rayctx->ray,INSERT_VALUES,SCATTER_FORWARD));
  if (rayctx->viewer) {
    PetscCall(VecView(rayctx->ray,rayctx->viewer));
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
  PetscCall(VecScatterBegin(rayctx->scatter, u, v, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(rayctx->scatter, u, v, INSERT_VALUES, SCATTER_FORWARD));
  if (!step) {
    PetscDrawAxis axis;

    PetscCall(PetscDrawLGGetAxis(lgctx->lg, &axis));
    PetscCall(PetscDrawAxisSetLabels(axis, "Solution Ray as function of time", "Time", "Solution"));
    PetscCall(VecGetLocalSize(rayctx->ray, &dim));
    PetscCall(PetscDrawLGSetDimension(lgctx->lg, dim));
    PetscCall(PetscDrawLGReset(lgctx->lg));
  }
  PetscCall(VecGetArrayRead(v, &a));
#if defined(PETSC_USE_COMPLEX)
  {
    PetscReal *areal;
    PetscInt   i,n;
    PetscCall(VecGetLocalSize(v, &n));
    PetscCall(PetscMalloc1(n, &areal));
    for (i = 0; i < n; ++i) areal[i] = PetscRealPart(a[i]);
    PetscCall(PetscDrawLGAddCommonPoint(lgctx->lg, ptime, areal));
    PetscCall(PetscFree(areal));
  }
#else
  PetscCall(PetscDrawLGAddCommonPoint(lgctx->lg, ptime, a));
#endif
  PetscCall(VecRestoreArrayRead(v, &a));
  if (((lgctx->howoften > 0) && (!(step % lgctx->howoften))) || ((lgctx->howoften == -1) && ts->reason)) {
    PetscCall(PetscDrawLGDraw(lgctx->lg));
    PetscCall(PetscDrawLGSave(lgctx->lg));
  }
  PetscFunctionReturn(0);
}
