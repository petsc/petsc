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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(sdm->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMTSDuplicate_DMDA(DMTS oldsdm,DMTS sdm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(sdm,(DMTS_DA**)&sdm->data);CHKERRQ(ierr);
  if (oldsdm->data) {ierr = PetscMemcpy(sdm->data,oldsdm->data,sizeof(DMTS_DA));CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode DMDATSGetContext(DM dm,DMTS sdm,DMTS_DA **dmdats)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *dmdats = NULL;
  if (!sdm->data) {
    ierr = PetscNewLog(dm,(DMTS_DA**)&sdm->data);CHKERRQ(ierr);
    sdm->ops->destroy   = DMTSDestroy_DMDA;
    sdm->ops->duplicate = DMTSDuplicate_DMDA;
  }
  *dmdats = (DMTS_DA*)sdm->data;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSComputeIFunction_DMDA(TS ts,PetscReal ptime,Vec X,Vec Xdot,Vec F,void *ctx)
{
  PetscErrorCode ierr;
  DM             dm;
  DMTS_DA        *dmdats = (DMTS_DA*)ctx;
  DMDALocalInfo  info;
  Vec            Xloc,Xdotloc;
  void           *x,*f,*xdot;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,3);
  PetscValidHeaderSpecific(F,VEC_CLASSID,5);
  if (!dmdats->ifunctionlocal) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_PLIB,"Corrupt context");
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm,&Xdotloc);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm,Xdot,INSERT_VALUES,Xdotloc);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm,Xdot,INSERT_VALUES,Xdotloc);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm,&Xloc);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(dm,&info);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(dm,Xloc,&x);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(dm,Xdotloc,&xdot);CHKERRQ(ierr);
  switch (dmdats->ifunctionlocalimode) {
  case INSERT_VALUES: {
    ierr = DMDAVecGetArray(dm,F,&f);CHKERRQ(ierr);
    CHKMEMQ;
    ierr = (*dmdats->ifunctionlocal)(&info,ptime,x,xdot,f,dmdats->ifunctionlocalctx);CHKERRQ(ierr);
    CHKMEMQ;
    ierr = DMDAVecRestoreArray(dm,F,&f);CHKERRQ(ierr);
  } break;
  case ADD_VALUES: {
    Vec Floc;
    ierr = DMGetLocalVector(dm,&Floc);CHKERRQ(ierr);
    ierr = VecZeroEntries(Floc);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(dm,Floc,&f);CHKERRQ(ierr);
    CHKMEMQ;
    ierr = (*dmdats->ifunctionlocal)(&info,ptime,x,xdot,f,dmdats->ifunctionlocalctx);CHKERRQ(ierr);
    CHKMEMQ;
    ierr = DMDAVecRestoreArray(dm,Floc,&f);CHKERRQ(ierr);
    ierr = VecZeroEntries(F);CHKERRQ(ierr);
    ierr = DMLocalToGlobalBegin(dm,Floc,ADD_VALUES,F);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(dm,Floc,ADD_VALUES,F);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm,&Floc);CHKERRQ(ierr);
  } break;
  default: SETERRQ1(PetscObjectComm((PetscObject)ts),PETSC_ERR_ARG_INCOMP,"Cannot use imode=%d",(int)dmdats->ifunctionlocalimode);
  }
  ierr = DMDAVecRestoreArray(dm,Xloc,&x);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&Xloc);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(dm,Xdotloc,&xdot);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&Xdotloc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSComputeIJacobian_DMDA(TS ts,PetscReal ptime,Vec X,Vec Xdot,PetscReal shift,Mat A,Mat B,void *ctx)
{
  PetscErrorCode ierr;
  DM             dm;
  DMTS_DA        *dmdats = (DMTS_DA*)ctx;
  DMDALocalInfo  info;
  Vec            Xloc;
  void           *x,*xdot;

  PetscFunctionBegin;
  if (!dmdats->ifunctionlocal) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_PLIB,"Corrupt context");
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);

  if (dmdats->ijacobianlocal) {
    ierr = DMGetLocalVector(dm,&Xloc);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(dm,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dm,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
    ierr = DMDAGetLocalInfo(dm,&info);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(dm,Xloc,&x);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(dm,Xdot,&xdot);CHKERRQ(ierr);
    CHKMEMQ;
    ierr = (*dmdats->ijacobianlocal)(&info,ptime,x,xdot,shift,A,B,dmdats->ijacobianlocalctx);CHKERRQ(ierr);
    CHKMEMQ;
    ierr = DMDAVecRestoreArray(dm,Xloc,&x);CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(dm,Xdot,&xdot);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm,&Xloc);CHKERRQ(ierr);
  } else SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_PLIB,"TSComputeIJacobian_DMDA() called without calling DMDATSSetIJacobian()");
  /* This will be redundant if the user called both, but it's too common to forget. */
  if (A != B) {
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSComputeRHSFunction_DMDA(TS ts,PetscReal ptime,Vec X,Vec F,void *ctx)
{
  PetscErrorCode ierr;
  DM             dm;
  DMTS_DA        *dmdats = (DMTS_DA*)ctx;
  DMDALocalInfo  info;
  Vec            Xloc;
  void           *x,*f;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,3);
  PetscValidHeaderSpecific(F,VEC_CLASSID,4);
  if (!dmdats->rhsfunctionlocal) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_PLIB,"Corrupt context");
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm,&Xloc);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(dm,&info);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(dm,Xloc,&x);CHKERRQ(ierr);
  switch (dmdats->rhsfunctionlocalimode) {
  case INSERT_VALUES: {
    ierr = DMDAVecGetArray(dm,F,&f);CHKERRQ(ierr);
    CHKMEMQ;
    ierr = (*dmdats->rhsfunctionlocal)(&info,ptime,x,f,dmdats->rhsfunctionlocalctx);CHKERRQ(ierr);
    CHKMEMQ;
    ierr = DMDAVecRestoreArray(dm,F,&f);CHKERRQ(ierr);
  } break;
  case ADD_VALUES: {
    Vec Floc;
    ierr = DMGetLocalVector(dm,&Floc);CHKERRQ(ierr);
    ierr = VecZeroEntries(Floc);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(dm,Floc,&f);CHKERRQ(ierr);
    CHKMEMQ;
    ierr = (*dmdats->rhsfunctionlocal)(&info,ptime,x,f,dmdats->rhsfunctionlocalctx);CHKERRQ(ierr);
    CHKMEMQ;
    ierr = DMDAVecRestoreArray(dm,Floc,&f);CHKERRQ(ierr);
    ierr = VecZeroEntries(F);CHKERRQ(ierr);
    ierr = DMLocalToGlobalBegin(dm,Floc,ADD_VALUES,F);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(dm,Floc,ADD_VALUES,F);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm,&Floc);CHKERRQ(ierr);
  } break;
  default: SETERRQ1(PetscObjectComm((PetscObject)ts),PETSC_ERR_ARG_INCOMP,"Cannot use imode=%d",(int)dmdats->rhsfunctionlocalimode);
  }
  ierr = DMDAVecRestoreArray(dm,Xloc,&x);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&Xloc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSComputeRHSJacobian_DMDA(TS ts,PetscReal ptime,Vec X,Mat A,Mat B,void *ctx)
{
  PetscErrorCode ierr;
  DM             dm;
  DMTS_DA        *dmdats = (DMTS_DA*)ctx;
  DMDALocalInfo  info;
  Vec            Xloc;
  void           *x;

  PetscFunctionBegin;
  if (!dmdats->rhsfunctionlocal) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_PLIB,"Corrupt context");
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);

  if (dmdats->rhsjacobianlocal) {
    ierr = DMGetLocalVector(dm,&Xloc);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(dm,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dm,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
    ierr = DMDAGetLocalInfo(dm,&info);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(dm,Xloc,&x);CHKERRQ(ierr);
    CHKMEMQ;
    ierr = (*dmdats->rhsjacobianlocal)(&info,ptime,x,A,B,dmdats->rhsjacobianlocalctx);CHKERRQ(ierr);
    CHKMEMQ;
    ierr = DMDAVecRestoreArray(dm,Xloc,&x);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm,&Xloc);CHKERRQ(ierr);
  } else SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_PLIB,"TSComputeRHSJacobian_DMDA() called without calling DMDATSSetRHSJacobian()");
  /* This will be redundant if the user called both, but it's too common to forget. */
  if (A != B) {
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  DMTS           sdm;
  DMTS_DA        *dmdats;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMTSWrite(dm,&sdm);CHKERRQ(ierr);
  ierr = DMDATSGetContext(dm,sdm,&dmdats);CHKERRQ(ierr);
  dmdats->rhsfunctionlocalimode = imode;
  dmdats->rhsfunctionlocal      = func;
  dmdats->rhsfunctionlocalctx   = ctx;
  ierr = DMTSSetRHSFunction(dm,TSComputeRHSFunction_DMDA,dmdats);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  DMTS           sdm;
  DMTS_DA        *dmdats;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMTSWrite(dm,&sdm);CHKERRQ(ierr);
  ierr = DMDATSGetContext(dm,sdm,&dmdats);CHKERRQ(ierr);
  dmdats->rhsjacobianlocal    = func;
  dmdats->rhsjacobianlocalctx = ctx;
  ierr = DMTSSetRHSJacobian(dm,TSComputeRHSJacobian_DMDA,dmdats);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  DMTS           sdm;
  DMTS_DA        *dmdats;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMTSWrite(dm,&sdm);CHKERRQ(ierr);
  ierr = DMDATSGetContext(dm,sdm,&dmdats);CHKERRQ(ierr);
  dmdats->ifunctionlocalimode = imode;
  dmdats->ifunctionlocal      = func;
  dmdats->ifunctionlocalctx   = ctx;
  ierr = DMTSSetIFunction(dm,TSComputeIFunction_DMDA,dmdats);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  DMTS           sdm;
  DMTS_DA        *dmdats;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMTSWrite(dm,&sdm);CHKERRQ(ierr);
  ierr = DMDATSGetContext(dm,sdm,&dmdats);CHKERRQ(ierr);
  dmdats->ijacobianlocal    = func;
  dmdats->ijacobianlocalctx = ctx;
  ierr = DMTSSetIJacobian(dm,TSComputeIJacobian_DMDA,dmdats);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TSMonitorDMDARayDestroy(void **mctx)
{
  TSMonitorDMDARayCtx *rayctx = (TSMonitorDMDARayCtx *) *mctx;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  if (rayctx->lgctx) {ierr = TSMonitorLGCtxDestroy(&rayctx->lgctx);CHKERRQ(ierr);}
  ierr = VecDestroy(&rayctx->ray);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&rayctx->scatter);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&rayctx->viewer);CHKERRQ(ierr);
  ierr = PetscFree(rayctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TSMonitorDMDARay(TS ts,PetscInt steps,PetscReal time,Vec u,void *mctx)
{
  TSMonitorDMDARayCtx *rayctx = (TSMonitorDMDARayCtx*)mctx;
  Vec                 solution;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = TSGetSolution(ts,&solution);CHKERRQ(ierr);
  ierr = VecScatterBegin(rayctx->scatter,solution,rayctx->ray,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(rayctx->scatter,solution,rayctx->ray,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  if (rayctx->viewer) {
    ierr = VecView(rayctx->ray,rayctx->viewer);CHKERRQ(ierr);
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
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  ierr = VecScatterBegin(rayctx->scatter, u, v, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(rayctx->scatter, u, v, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  if (!step) {
    PetscDrawAxis axis;

    ierr = PetscDrawLGGetAxis(lgctx->lg, &axis);CHKERRQ(ierr);
    ierr = PetscDrawAxisSetLabels(axis, "Solution Ray as function of time", "Time", "Solution");CHKERRQ(ierr);
    ierr = VecGetLocalSize(rayctx->ray, &dim);CHKERRQ(ierr);
    ierr = PetscDrawLGSetDimension(lgctx->lg, dim);CHKERRQ(ierr);
    ierr = PetscDrawLGReset(lgctx->lg);CHKERRQ(ierr);
  }
  ierr = VecGetArrayRead(v, &a);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  {
    PetscReal *areal;
    PetscInt   i,n;
    ierr = VecGetLocalSize(v, &n);CHKERRQ(ierr);
    ierr = PetscMalloc1(n, &areal);CHKERRQ(ierr);
    for (i = 0; i < n; ++i) areal[i] = PetscRealPart(a[i]);
    ierr = PetscDrawLGAddCommonPoint(lgctx->lg, ptime, areal);CHKERRQ(ierr);
    ierr = PetscFree(areal);CHKERRQ(ierr);
  }
#else
  ierr = PetscDrawLGAddCommonPoint(lgctx->lg, ptime, a);CHKERRQ(ierr);
#endif
  ierr = VecRestoreArrayRead(v, &a);CHKERRQ(ierr);
  if (((lgctx->howoften > 0) && (!(step % lgctx->howoften))) || ((lgctx->howoften == -1) && ts->reason)) {
    ierr = PetscDrawLGDraw(lgctx->lg);CHKERRQ(ierr);
    ierr = PetscDrawLGSave(lgctx->lg);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
