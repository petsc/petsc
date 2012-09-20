#include <petscdmda.h>          /*I "petscdmda.h" I*/
#include <petsc-private/dmimpl.h>
#include <petsc-private/tsimpl.h>   /*I "petscts.h" I*/

/* This structure holds the user-provided DMDA callbacks */
typedef struct {
  PetscErrorCode (*ifunctionlocal)(DMDALocalInfo*,PetscReal,void*,void*,void*,void*);
  PetscErrorCode (*rhsfunctionlocal)(DMDALocalInfo*,PetscReal,void*,void*,void*);
  PetscErrorCode (*ijacobianlocal)(DMDALocalInfo*,PetscReal,void*,void*,PetscReal,Mat,Mat,MatStructure*,void*);
  PetscErrorCode (*rhsjacobianlocal)(DMDALocalInfo*,PetscReal,void*,Mat,Mat,MatStructure*,void*);
  void *ifunctionlocalctx;
  void *ijacobianlocalctx;
  void *rhsfunctionlocalctx;
  void *rhsjacobianlocalctx;
  InsertMode ifunctionlocalimode;
  InsertMode rhsfunctionlocalimode;
} DM_DA_TS;

#undef __FUNCT__
#define __FUNCT__ "TSDMDestroy_DMDA"
static PetscErrorCode TSDMDestroy_DMDA(TSDM sdm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(sdm->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDATSGetContext"
static PetscErrorCode DMDATSGetContext(DM dm,TSDM sdm,DM_DA_TS **dmdats)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *dmdats = PETSC_NULL;
  if (!sdm->data) {
    ierr = PetscNewLog(dm,DM_DA_TS,&sdm->data);CHKERRQ(ierr);
    sdm->destroy = TSDMDestroy_DMDA;
  }
  *dmdats = (DM_DA_TS*)sdm->data;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSComputeIFunction_DMDA"
/*
  This function should eventually replace:
    DMDAComputeFunction() and DMDAComputeFunction1()
 */
static PetscErrorCode TSComputeIFunction_DMDA(TS ts,PetscReal ptime,Vec X,Vec Xdot,Vec F,void *ctx)
{
  PetscErrorCode ierr;
  DM             dm;
  DM_DA_TS     *dmdats = (DM_DA_TS*)ctx;
  DMDALocalInfo  info;
  Vec            Xloc;
  void           *x,*f,*xdot;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,2);
  PetscValidHeaderSpecific(F,VEC_CLASSID,3);
  if (!dmdats->ifunctionlocal) SETERRQ(((PetscObject)ts)->comm,PETSC_ERR_PLIB,"Corrupt context");
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm,&Xloc);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(dm,&info);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(dm,Xloc,&x);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(dm,Xdot,&xdot);CHKERRQ(ierr);
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
  default: SETERRQ1(((PetscObject)ts)->comm,PETSC_ERR_ARG_INCOMP,"Cannot use imode=%d",(int)dmdats->ifunctionlocalimode);
  }
  ierr = DMDAVecRestoreArray(dm,Xloc,&x);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&Xloc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSComputeIJacobian_DMDA"
/*
  This function should eventually replace:
    DMComputeJacobian() and DMDAComputeJacobian1()
 */
static PetscErrorCode TSComputeIJacobian_DMDA(TS ts,PetscReal ptime,Vec X,Vec Xdot,PetscReal shift,Mat *A,Mat *B,MatStructure *mstr,void *ctx)
{
  PetscErrorCode ierr;
  DM             dm;
  DM_DA_TS     *dmdats = (DM_DA_TS*)ctx;
  DMDALocalInfo  info;
  Vec            Xloc;
  void           *x,*xdot;

  PetscFunctionBegin;
  if (!dmdats->ifunctionlocal) SETERRQ(((PetscObject)ts)->comm,PETSC_ERR_PLIB,"Corrupt context");
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);

  if (dmdats->ijacobianlocal) {
    ierr = DMGetLocalVector(dm,&Xloc);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(dm,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dm,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
    ierr = DMDAGetLocalInfo(dm,&info);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(dm,Xloc,&x);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(dm,Xdot,&xdot);CHKERRQ(ierr);
    CHKMEMQ;
    ierr = (*dmdats->ijacobianlocal)(&info,ptime,x,xdot,shift,*A,*B,mstr,dmdats->ijacobianlocalctx);CHKERRQ(ierr);
    CHKMEMQ;
    ierr = DMDAVecRestoreArray(dm,Xloc,&x);CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(dm,Xdot,&xdot);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm,&Xloc);CHKERRQ(ierr);
  } else {
    SETERRQ(((PetscObject)ts)->comm,PETSC_ERR_PLIB,"TSComputeIJacobian_DMDA() called without calling DMDATSSetIJacobian()");
  }
  /* This will be redundant if the user called both, but it's too common to forget. */
  if (*A != *B) {
    ierr = MatAssemblyBegin(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSComputeRHSFunction_DMDA"
/*
  This function should eventually replace:
    DMDAComputeFunction() and DMDAComputeFunction1()
 */
static PetscErrorCode TSComputeRHSFunction_DMDA(TS ts,PetscReal ptime,Vec X,Vec F,void *ctx)
{
  PetscErrorCode ierr;
  DM             dm;
  DM_DA_TS     *dmdats = (DM_DA_TS*)ctx;
  DMDALocalInfo  info;
  Vec            Xloc;
  void           *x,*f;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,2);
  PetscValidHeaderSpecific(F,VEC_CLASSID,3);
  if (!dmdats->rhsfunctionlocal) SETERRQ(((PetscObject)ts)->comm,PETSC_ERR_PLIB,"Corrupt context");
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
  default: SETERRQ1(((PetscObject)ts)->comm,PETSC_ERR_ARG_INCOMP,"Cannot use imode=%d",(int)dmdats->rhsfunctionlocalimode);
  }
  ierr = DMDAVecRestoreArray(dm,Xloc,&x);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&Xloc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSComputeRHSJacobian_DMDA"
/*
  This function should eventually replace:
    DMComputeJacobian() and DMDAComputeJacobian1()
 */
static PetscErrorCode TSComputeRHSJacobian_DMDA(TS ts,PetscReal ptime,Vec X,Mat *A,Mat *B,MatStructure *mstr,void *ctx)
{
  PetscErrorCode ierr;
  DM             dm;
  DM_DA_TS     *dmdats = (DM_DA_TS*)ctx;
  DMDALocalInfo  info;
  Vec            Xloc;
  void           *x;

  PetscFunctionBegin;
  if (!dmdats->rhsfunctionlocal) SETERRQ(((PetscObject)ts)->comm,PETSC_ERR_PLIB,"Corrupt context");
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);

  if (dmdats->rhsjacobianlocal) {
    ierr = DMGetLocalVector(dm,&Xloc);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(dm,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dm,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
    ierr = DMDAGetLocalInfo(dm,&info);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(dm,Xloc,&x);CHKERRQ(ierr);
    CHKMEMQ;
    ierr = (*dmdats->rhsjacobianlocal)(&info,ptime,x,*A,*B,mstr,dmdats->rhsjacobianlocalctx);CHKERRQ(ierr);
    CHKMEMQ;
    ierr = DMDAVecRestoreArray(dm,Xloc,&x);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm,&Xloc);CHKERRQ(ierr);
  } else {
    SETERRQ(((PetscObject)ts)->comm,PETSC_ERR_PLIB,"TSComputeRHSJacobian_DMDA() called without calling DMDATSSetRHSJacobian()");
  }
  /* This will be redundant if the user called both, but it's too common to forget. */
  if (*A != *B) {
    ierr = MatAssemblyBegin(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMDATSSetRHSFunctionLocal"
/*@C
   DMDATSSetRHSFunctionLocal - set a local residual evaluation function

   Logically Collective

   Input Arguments:
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
  TSDM         sdm;
  DM_DA_TS     *dmdats;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMTSGetContext(dm,&sdm);CHKERRQ(ierr);
  ierr = DMDATSGetContext(dm,sdm,&dmdats);CHKERRQ(ierr);
  dmdats->rhsfunctionlocalimode = imode;
  dmdats->rhsfunctionlocal = func;
  dmdats->rhsfunctionlocalctx = ctx;
  ierr = DMTSSetRHSFunction(dm,TSComputeRHSFunction_DMDA,dmdats);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDATSSetRHSJacobianLocal"
/*@C
   DMDATSSetRHSJacobianLocal - set a local residual evaluation function

   Logically Collective

   Input Arguments:
+  dm    - DM to associate callback with
.  func  - local RHS Jacobian evaluation routine
-  ctx   - optional context for local jacobian evaluation

   Calling sequence for func:

$ func(DMDALocalInfo* info,PetscReal t,void* x,Mat J,Mat B,MatStructure *flg,void *ctx);

+  info - DMDALocalInfo defining the subdomain to evaluate the residual on
.  t    - time at which to evaluate residual
.  x    - array of local state information
.  J    - Jacobian matrix
.  B    - preconditioner matrix; often same as J
.  flg  - flag indicating information about the preconditioner matrix structure (same as flag in KSPSetOperators())
-  ctx  - optional context passed above

   Level: beginner

.seealso: DMTSSetRHSJacobian(), DMDATSSetRHSFunctionLocal(), DMDASNESSetJacobianLocal()
@*/
PetscErrorCode DMDATSSetRHSJacobianLocal(DM dm,DMDATSRHSJacobianLocal func,void *ctx)
{
  PetscErrorCode ierr;
  TSDM         sdm;
  DM_DA_TS     *dmdats;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMTSGetContext(dm,&sdm);CHKERRQ(ierr);
  ierr = DMDATSGetContext(dm,sdm,&dmdats);CHKERRQ(ierr);
  dmdats->rhsjacobianlocal = func;
  dmdats->rhsjacobianlocalctx = ctx;
  ierr = DMTSSetRHSJacobian(dm,TSComputeRHSJacobian_DMDA,dmdats);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMDATSSetIFunctionLocal"
/*@C
   DMDATSSetIFunctionLocal - set a local residual evaluation function

   Logically Collective

   Input Arguments:
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
  TSDM         sdm;
  DM_DA_TS     *dmdats;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMTSGetContext(dm,&sdm);CHKERRQ(ierr);
  ierr = DMDATSGetContext(dm,sdm,&dmdats);CHKERRQ(ierr);
  dmdats->ifunctionlocalimode = imode;
  dmdats->ifunctionlocal = func;
  dmdats->ifunctionlocalctx = ctx;
  ierr = DMTSSetIFunction(dm,TSComputeIFunction_DMDA,dmdats);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDATSSetIJacobianLocal"
/*@C
   DMDATSSetIJacobianLocal - set a local residual evaluation function

   Logically Collective

   Input Arguments:
+  dm   - DM to associate callback with
.  func - local residual evaluation
-  ctx   - optional context for local residual evaluation

   Calling sequence for func:

$ func(DMDALocalInfo* info,PetscReal t,void* x,void *xdot,Mat J,Mat B,MatStructure *flg,void *ctx);

+  info - DMDALocalInfo defining the subdomain to evaluate the residual on
.  t    - time at which to evaluate the jacobian
.  x    - array of local state information
.  xdot - time derivative at this state
.  J    - Jacobian matrix
.  B    - preconditioner matrix; often same as J
.  flg  - flag indicating information about the preconditioner matrix structure (same as flag in KSPSetOperators())
-  ctx  - optional context passed above

   Level: beginner

.seealso: DMTSSetJacobian(), DMDATSSetIFunctionLocal(), DMDASNESSetJacobianLocal()
@*/
PetscErrorCode DMDATSSetIJacobianLocal(DM dm,DMDATSIJacobianLocal func,void *ctx)
{
  PetscErrorCode ierr;
  TSDM         sdm;
  DM_DA_TS     *dmdats;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMTSGetContext(dm,&sdm);CHKERRQ(ierr);
  ierr = DMDATSGetContext(dm,sdm,&dmdats);CHKERRQ(ierr);
  dmdats->ijacobianlocal = func;
  dmdats->ijacobianlocalctx = ctx;
  ierr = DMTSSetIJacobian(dm,TSComputeIJacobian_DMDA,dmdats);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
