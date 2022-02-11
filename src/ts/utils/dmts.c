#include <petsc/private/tsimpl.h>     /*I "petscts.h" I*/
#include <petsc/private/dmimpl.h>

static PetscErrorCode DMTSDestroy(DMTS *kdm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*kdm) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*kdm),DMTS_CLASSID,1);
  if (--((PetscObject)(*kdm))->refct > 0) {*kdm = NULL; PetscFunctionReturn(0);}
  if ((*kdm)->ops->destroy) {ierr = ((*kdm)->ops->destroy)(*kdm);CHKERRQ(ierr);}
  ierr = PetscHeaderDestroy(kdm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMTSLoad(DMTS kdm,PetscViewer viewer)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerBinaryRead(viewer,&kdm->ops->ifunction,1,NULL,PETSC_FUNCTION);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,&kdm->ops->ifunctionview,1,NULL,PETSC_FUNCTION);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,&kdm->ops->ifunctionload,1,NULL,PETSC_FUNCTION);CHKERRQ(ierr);
  if (kdm->ops->ifunctionload) {
    ierr = (*kdm->ops->ifunctionload)(&kdm->ifunctionctx,viewer);CHKERRQ(ierr);
  }
  ierr = PetscViewerBinaryRead(viewer,&kdm->ops->ijacobian,1,NULL,PETSC_FUNCTION);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,&kdm->ops->ijacobianview,1,NULL,PETSC_FUNCTION);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,&kdm->ops->ijacobianload,1,NULL,PETSC_FUNCTION);CHKERRQ(ierr);
  if (kdm->ops->ijacobianload) {
    ierr = (*kdm->ops->ijacobianload)(&kdm->ijacobianctx,viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMTSView(DMTS kdm,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      isascii,isbinary;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
  if (isascii) {
#if defined(PETSC_SERIALIZE_FUNCTIONS)
    const char *fname;

    ierr = PetscFPTFind(kdm->ops->ifunction,&fname);CHKERRQ(ierr);
    if (fname) {
      ierr = PetscViewerASCIIPrintf(viewer,"  IFunction used by TS: %s\n",fname);CHKERRQ(ierr);
    }
    ierr = PetscFPTFind(kdm->ops->ijacobian,&fname);CHKERRQ(ierr);
    if (fname) {
      ierr = PetscViewerASCIIPrintf(viewer,"  IJacobian function used by TS: %s\n",fname);CHKERRQ(ierr);
    }
#endif
  } else if (isbinary) {
    struct {
      TSIFunction ifunction;
    } funcstruct;
    struct {
      PetscErrorCode (*ifunctionview)(void*,PetscViewer);
    } funcviewstruct;
    struct {
      PetscErrorCode (*ifunctionload)(void**,PetscViewer);
    } funcloadstruct;
    struct {
      TSIJacobian ijacobian;
    } jacstruct;
    struct {
      PetscErrorCode (*ijacobianview)(void*,PetscViewer);
    } jacviewstruct;
    struct {
      PetscErrorCode (*ijacobianload)(void**,PetscViewer);
    } jacloadstruct;

    funcstruct.ifunction         = kdm->ops->ifunction;
    funcviewstruct.ifunctionview = kdm->ops->ifunctionview;
    funcloadstruct.ifunctionload = kdm->ops->ifunctionload;
    ierr = PetscViewerBinaryWrite(viewer,&funcstruct,1,PETSC_FUNCTION);CHKERRQ(ierr);
    ierr = PetscViewerBinaryWrite(viewer,&funcviewstruct,1,PETSC_FUNCTION);CHKERRQ(ierr);
    ierr = PetscViewerBinaryWrite(viewer,&funcloadstruct,1,PETSC_FUNCTION);CHKERRQ(ierr);
    if (kdm->ops->ifunctionview) {
      ierr = (*kdm->ops->ifunctionview)(kdm->ifunctionctx,viewer);CHKERRQ(ierr);
    }
    jacstruct.ijacobian = kdm->ops->ijacobian;
    jacviewstruct.ijacobianview = kdm->ops->ijacobianview;
    jacloadstruct.ijacobianload = kdm->ops->ijacobianload;
    ierr = PetscViewerBinaryWrite(viewer,&jacstruct,1,PETSC_FUNCTION);CHKERRQ(ierr);
    ierr = PetscViewerBinaryWrite(viewer,&jacviewstruct,1,PETSC_FUNCTION);CHKERRQ(ierr);
    ierr = PetscViewerBinaryWrite(viewer,&jacloadstruct,1,PETSC_FUNCTION);CHKERRQ(ierr);
    if (kdm->ops->ijacobianview) {
      ierr = (*kdm->ops->ijacobianview)(kdm->ijacobianctx,viewer);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMTSCreate(MPI_Comm comm,DMTS *kdm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSInitializePackage();CHKERRQ(ierr);
  ierr = PetscHeaderCreate(*kdm, DMTS_CLASSID, "DMTS", "DMTS", "DMTS", comm, DMTSDestroy, DMTSView);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Attaches the DMTS to the coarse level.
 * Under what conditions should we copy versus duplicate?
 */
static PetscErrorCode DMCoarsenHook_DMTS(DM dm,DM dmc,void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMCopyDMTS(dm,dmc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* This could restrict auxiliary information to the coarse level.
 */
static PetscErrorCode DMRestrictHook_DMTS(DM dm,Mat Restrict,Vec rscale,Mat Inject,DM dmc,void *ctx)
{

  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMSubDomainHook_DMTS(DM dm,DM subdm,void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMCopyDMTS(dm,subdm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* This could restrict auxiliary information to the coarse level.
 */
static PetscErrorCode DMSubDomainRestrictHook_DMTS(DM dm,VecScatter gscat,VecScatter lscat,DM subdm,void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

/*@C
   DMTSCopy - copies the information in a DMTS to another DMTS

   Not Collective

   Input Parameters:
+  kdm - Original DMTS
-  nkdm - DMTS to receive the data, should have been created with DMTSCreate()

   Level: developer

.seealso: DMTSCreate(), DMTSDestroy()
@*/
PetscErrorCode DMTSCopy(DMTS kdm,DMTS nkdm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(kdm,DMTS_CLASSID,1);
  PetscValidHeaderSpecific(nkdm,DMTS_CLASSID,2);
  nkdm->ops->rhsfunction = kdm->ops->rhsfunction;
  nkdm->ops->rhsjacobian = kdm->ops->rhsjacobian;
  nkdm->ops->ifunction   = kdm->ops->ifunction;
  nkdm->ops->ijacobian   = kdm->ops->ijacobian;
  nkdm->ops->i2function  = kdm->ops->i2function;
  nkdm->ops->i2jacobian  = kdm->ops->i2jacobian;
  nkdm->ops->solution    = kdm->ops->solution;
  nkdm->ops->destroy     = kdm->ops->destroy;
  nkdm->ops->duplicate   = kdm->ops->duplicate;

  nkdm->rhsfunctionctx = kdm->rhsfunctionctx;
  nkdm->rhsjacobianctx = kdm->rhsjacobianctx;
  nkdm->ifunctionctx   = kdm->ifunctionctx;
  nkdm->ijacobianctx   = kdm->ijacobianctx;
  nkdm->i2functionctx  = kdm->i2functionctx;
  nkdm->i2jacobianctx  = kdm->i2jacobianctx;
  nkdm->solutionctx    = kdm->solutionctx;

  nkdm->data = kdm->data;

  /*
  nkdm->fortran_func_pointers[0] = kdm->fortran_func_pointers[0];
  nkdm->fortran_func_pointers[1] = kdm->fortran_func_pointers[1];
  nkdm->fortran_func_pointers[2] = kdm->fortran_func_pointers[2];
  */

  /* implementation specific copy hooks */
  if (kdm->ops->duplicate) {ierr = (*kdm->ops->duplicate)(kdm,nkdm);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/*@C
   DMGetDMTS - get read-only private DMTS context from a DM

   Not Collective

   Input Parameter:
.  dm - DM to be used with TS

   Output Parameter:
.  tsdm - private DMTS context

   Level: developer

   Notes:
   Use DMGetDMTSWrite() if write access is needed. The DMTSSetXXX API should be used wherever possible.

.seealso: DMGetDMTSWrite()
@*/
PetscErrorCode DMGetDMTS(DM dm,DMTS *tsdm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  *tsdm = (DMTS) dm->dmts;
  if (!*tsdm) {
    ierr = PetscInfo(dm,"Creating new DMTS\n");CHKERRQ(ierr);
    ierr = DMTSCreate(PetscObjectComm((PetscObject)dm),tsdm);CHKERRQ(ierr);
    dm->dmts = (PetscObject) *tsdm;
    (*tsdm)->originaldm = dm;
    ierr = DMCoarsenHookAdd(dm,DMCoarsenHook_DMTS,DMRestrictHook_DMTS,NULL);CHKERRQ(ierr);
    ierr = DMSubDomainHookAdd(dm,DMSubDomainHook_DMTS,DMSubDomainRestrictHook_DMTS,NULL);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
   DMGetDMTSWrite - get write access to private DMTS context from a DM

   Not Collective

   Input Parameter:
.  dm - DM to be used with TS

   Output Parameter:
.  tsdm - private DMTS context

   Level: developer

.seealso: DMGetDMTS()
@*/
PetscErrorCode DMGetDMTSWrite(DM dm,DMTS *tsdm)
{
  PetscErrorCode ierr;
  DMTS           sdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMTS(dm,&sdm);CHKERRQ(ierr);
  PetscCheckFalse(!sdm->originaldm,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"DMTS has a NULL originaldm");
  if (sdm->originaldm != dm) {  /* Copy on write */
    DMTS oldsdm = sdm;
    ierr     = PetscInfo(dm,"Copying DMTS due to write\n");CHKERRQ(ierr);
    ierr     = DMTSCreate(PetscObjectComm((PetscObject)dm),&sdm);CHKERRQ(ierr);
    ierr     = DMTSCopy(oldsdm,sdm);CHKERRQ(ierr);
    ierr     = DMTSDestroy((DMTS*)&dm->dmts);CHKERRQ(ierr);
    dm->dmts = (PetscObject) sdm;
    sdm->originaldm = dm;
  }
  *tsdm = sdm;
  PetscFunctionReturn(0);
}

/*@C
   DMCopyDMTS - copies a DM context to a new DM

   Logically Collective

   Input Parameters:
+  dmsrc - DM to obtain context from
-  dmdest - DM to add context to

   Level: developer

   Note:
   The context is copied by reference. This function does not ensure that a context exists.

.seealso: DMGetDMTS(), TSSetDM()
@*/
PetscErrorCode DMCopyDMTS(DM dmsrc,DM dmdest)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dmsrc,DM_CLASSID,1);
  PetscValidHeaderSpecific(dmdest,DM_CLASSID,2);
  ierr         = DMTSDestroy((DMTS*)&dmdest->dmts);CHKERRQ(ierr);
  dmdest->dmts = dmsrc->dmts;
  ierr         = PetscObjectReference(dmdest->dmts);CHKERRQ(ierr);
  ierr         = DMCoarsenHookAdd(dmdest,DMCoarsenHook_DMTS,DMRestrictHook_DMTS,NULL);CHKERRQ(ierr);
  ierr         = DMSubDomainHookAdd(dmdest,DMSubDomainHook_DMTS,DMSubDomainRestrictHook_DMTS,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   DMTSSetIFunction - set TS implicit function evaluation function

   Not Collective

   Input Parameters:
+  dm - DM to be used with TS
.  func - function evaluating f(t,u,u_t)
-  ctx - context for residual evaluation

   Calling sequence of func:
$     PetscErrorCode func(TS ts,PetscReal t,Vec u,Vec u_t,Vec F,ctx);

+  t   - time at step/stage being solved
.  u   - state vector
.  u_t - time derivative of state vector
.  F   - function vector
-  ctx - [optional] user-defined context for matrix evaluation routine

   Level: advanced

   Note:
   TSSetFunction() is normally used, but it calls this function internally because the user context is actually
   associated with the DM.  This makes the interface consistent regardless of whether the user interacts with a DM or
   not. If DM took a more central role at some later date, this could become the primary method of setting the residual.

.seealso: DMTSSetContext(), TSSetIFunction(), DMTSSetJacobian()
@*/
PetscErrorCode DMTSSetIFunction(DM dm,TSIFunction func,void *ctx)
{
  PetscErrorCode ierr;
  DMTS           tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMTSWrite(dm,&tsdm);CHKERRQ(ierr);
  if (func) tsdm->ops->ifunction = func;
  if (ctx)  tsdm->ifunctionctx = ctx;
  PetscFunctionReturn(0);
}

/*@C
   DMTSGetIFunction - get TS implicit residual evaluation function

   Not Collective

   Input Parameter:
.  dm - DM to be used with TS

   Output Parameters:
+  func - function evaluation function, see TSSetIFunction() for calling sequence
-  ctx - context for residual evaluation

   Level: advanced

   Note:
   TSGetFunction() is normally used, but it calls this function internally because the user context is actually
   associated with the DM.

.seealso: DMTSSetContext(), DMTSSetFunction(), TSSetFunction()
@*/
PetscErrorCode DMTSGetIFunction(DM dm,TSIFunction *func,void **ctx)
{
  PetscErrorCode ierr;
  DMTS           tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMTS(dm,&tsdm);CHKERRQ(ierr);
  if (func) *func = tsdm->ops->ifunction;
  if (ctx)  *ctx = tsdm->ifunctionctx;
  PetscFunctionReturn(0);
}

/*@C
   DMTSSetI2Function - set TS implicit function evaluation function for 2nd order systems

   Not Collective

   Input Parameters:
+  dm - DM to be used with TS
.  fun - function evaluation routine
-  ctx - context for residual evaluation

   Calling sequence of fun:
$     PetscErrorCode fun(TS ts,PetscReal t,Vec U,Vec U_t,Vec U_tt,Vec F,ctx);

+  t    - time at step/stage being solved
.  U    - state vector
.  U_t  - time derivative of state vector
.  U_tt - second time derivative of state vector
.  F    - function vector
-  ctx  - [optional] user-defined context for matrix evaluation routine (may be NULL)

   Level: advanced

   Note:
   TSSetI2Function() is normally used, but it calls this function internally because the user context is actually
   associated with the DM.

.seealso: TSSetI2Function()
@*/
PetscErrorCode DMTSSetI2Function(DM dm,TSI2Function fun,void *ctx)
{
  DMTS           tsdm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMTSWrite(dm,&tsdm);CHKERRQ(ierr);
  if (fun) tsdm->ops->i2function = fun;
  if (ctx) tsdm->i2functionctx   = ctx;
  PetscFunctionReturn(0);
}

/*@C
   DMTSGetI2Function - get TS implicit residual evaluation function for 2nd order systems

   Not Collective

   Input Parameter:
.  dm - DM to be used with TS

   Output Parameters:
+  fun - function evaluation function, see TSSetI2Function() for calling sequence
-  ctx - context for residual evaluation

   Level: advanced

   Note:
   TSGetI2Function() is normally used, but it calls this function internally because the user context is actually
   associated with the DM.

.seealso: DMTSSetI2Function(),TSGetI2Function()
@*/
PetscErrorCode DMTSGetI2Function(DM dm,TSI2Function *fun,void **ctx)
{
  DMTS           tsdm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMTS(dm,&tsdm);CHKERRQ(ierr);
  if (fun) *fun = tsdm->ops->i2function;
  if (ctx) *ctx = tsdm->i2functionctx;
  PetscFunctionReturn(0);
}

/*@C
   DMTSSetI2Jacobian - set TS implicit Jacobian evaluation function for 2nd order systems

   Not Collective

   Input Parameters:
+  dm - DM to be used with TS
.  fun - Jacobian evaluation routine
-  ctx - context for Jacobian evaluation

   Calling sequence of jac:
$    PetscErrorCode jac(TS ts,PetscReal t,Vec U,Vec U_t,Vec U_tt,PetscReal v,PetscReal a,Mat J,Mat P,void *ctx);

+  t    - time at step/stage being solved
.  U    - state vector
.  U_t  - time derivative of state vector
.  U_tt - second time derivative of state vector
.  v    - shift for U_t
.  a    - shift for U_tt
.  J    - Jacobian of G(U) = F(t,U,W+v*U,W'+a*U), equivalent to dF/dU + v*dF/dU_t  + a*dF/dU_tt
.  P    - preconditioning matrix for J, may be same as J
-  ctx  - [optional] user-defined context for matrix evaluation routine

   Level: advanced

   Note:
   TSSetI2Jacobian() is normally used, but it calls this function internally because the user context is actually
   associated with the DM.

.seealso: TSSetI2Jacobian()
@*/
PetscErrorCode DMTSSetI2Jacobian(DM dm,TSI2Jacobian jac,void *ctx)
{
  DMTS           tsdm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMTSWrite(dm,&tsdm);CHKERRQ(ierr);
  if (jac) tsdm->ops->i2jacobian = jac;
  if (ctx) tsdm->i2jacobianctx   = ctx;
  PetscFunctionReturn(0);
}

/*@C
   DMTSGetI2Jacobian - get TS implicit Jacobian evaluation function for 2nd order systems

   Not Collective

   Input Parameter:
.  dm - DM to be used with TS

   Output Parameters:
+  jac - Jacobian evaluation function, see TSSetI2Jacobian() for calling sequence
-  ctx - context for Jacobian evaluation

   Level: advanced

   Note:
   TSGetI2Jacobian() is normally used, but it calls this function internally because the user context is actually
   associated with the DM.

.seealso: DMTSSetI2Jacobian(),TSGetI2Jacobian()
@*/
PetscErrorCode DMTSGetI2Jacobian(DM dm,TSI2Jacobian *jac,void **ctx)
{
  DMTS           tsdm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMTS(dm,&tsdm);CHKERRQ(ierr);
  if (jac) *jac = tsdm->ops->i2jacobian;
  if (ctx) *ctx = tsdm->i2jacobianctx;
  PetscFunctionReturn(0);
}

/*@C
   DMTSSetRHSFunction - set TS explicit residual evaluation function

   Not Collective

   Input Parameters:
+  dm - DM to be used with TS
.  func - RHS function evaluation routine
-  ctx - context for residual evaluation

    Calling sequence of func:
$     PetscErrorCode func(TS ts,PetscReal t,Vec u,Vec F,void *ctx);

+   ts - timestep context
.   t - current timestep
.   u - input vector
.   F - function vector
-   ctx - [optional] user-defined function context

   Level: advanced

   Note:
   TSSetRSHFunction() is normally used, but it calls this function internally because the user context is actually
   associated with the DM.  This makes the interface consistent regardless of whether the user interacts with a DM or
   not. If DM took a more central role at some later date, this could become the primary method of setting the residual.

.seealso: DMTSSetContext(), TSSetRHSFunction(), DMTSSetJacobian()
@*/
PetscErrorCode DMTSSetRHSFunction(DM dm,TSRHSFunction func,void *ctx)
{
  PetscErrorCode ierr;
  DMTS           tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMTSWrite(dm,&tsdm);CHKERRQ(ierr);
  if (func) tsdm->ops->rhsfunction = func;
  if (ctx)  tsdm->rhsfunctionctx = ctx;
  PetscFunctionReturn(0);
}

/*@C
   DMTSSetTransientVariable - sets function to transform from state to transient variables

   Logically Collective

   Input Parameters:
+  dm - DM to be used with TS
.  tvar - a function that transforms to transient variables
-  ctx - a context for tvar

    Calling sequence of tvar:
$     PetscErrorCode tvar(TS ts,Vec p,Vec c,void *ctx);

+   ts - timestep context
.   p - input vector (primative form)
.   c - output vector, transient variables (conservative form)
-   ctx - [optional] user-defined function context

   Level: advanced

   Notes:
   This is typically used to transform from primitive to conservative variables so that a time integrator (e.g., TSBDF)
   can be conservative.  In this context, primitive variables P are used to model the state (e.g., because they lead to
   well-conditioned formulations even in limiting cases such as low-Mach or zero porosity).  The transient variable is
   C(P), specified by calling this function.  An IFunction thus receives arguments (P, Cdot) and the IJacobian must be
   evaluated via the chain rule, as in

     dF/dP + shift * dF/dCdot dC/dP.

.seealso: TSSetTransientVariable(), DMTSGetTransientVariable(), DMTSSetIFunction(), DMTSSetIJacobian()
@*/
PetscErrorCode DMTSSetTransientVariable(DM dm,TSTransientVariable tvar,void *ctx)
{
  PetscErrorCode ierr;
  DMTS           dmts;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMTSWrite(dm,&dmts);CHKERRQ(ierr);
  dmts->ops->transientvar = tvar;
  dmts->transientvarctx = ctx;
  PetscFunctionReturn(0);
}

/*@C
   DMTSGetTransientVariable - gets function to transform from state to transient variables

   Logically Collective

   Input Parameter:
.  dm - DM to be used with TS

   Output Parameters:
+  tvar - a function that transforms to transient variables
-  ctx - a context for tvar

   Level: advanced

.seealso: DMTSSetTransientVariable(), DMTSGetIFunction(), DMTSGetIJacobian()
@*/
PetscErrorCode DMTSGetTransientVariable(DM dm,TSTransientVariable *tvar,void *ctx)
{
  PetscErrorCode ierr;
  DMTS           dmts;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMTS(dm,&dmts);CHKERRQ(ierr);
  if (tvar) *tvar = dmts->ops->transientvar;
  if (ctx)  *(void**)ctx = dmts->transientvarctx;
  PetscFunctionReturn(0);
}

/*@C
   DMTSGetSolutionFunction - gets the TS solution evaluation function

   Not Collective

   Input Parameter:
.  dm - DM to be used with TS

   Output Parameters:
+  func - solution function evaluation function, see TSSetSolution() for calling sequence
-  ctx - context for solution evaluation

   Level: advanced

.seealso: DMTSSetContext(), TSSetFunction(), DMTSSetJacobian(), DMTSSetSolutionFunction()
@*/
PetscErrorCode DMTSGetSolutionFunction(DM dm,TSSolutionFunction *func,void **ctx)
{
  PetscErrorCode ierr;
  DMTS           tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMTS(dm,&tsdm);CHKERRQ(ierr);
  if (func) *func = tsdm->ops->solution;
  if (ctx)  *ctx  = tsdm->solutionctx;
  PetscFunctionReturn(0);
}

/*@C
   DMTSSetSolutionFunction - set TS solution evaluation function

   Not Collective

   Input Parameters:
+  dm - DM to be used with TS
.  func - solution function evaluation routine
-  ctx - context for solution evaluation

    Calling sequence of f:
$     PetscErrorCode f(TS ts,PetscReal t,Vec u,void *ctx);

+   ts - timestep context
.   t - current timestep
.   u - output vector
-   ctx - [optional] user-defined function context

   Level: advanced

   Note:
   TSSetSolutionFunction() is normally used, but it calls this function internally because the user context is actually
   associated with the DM.  This makes the interface consistent regardless of whether the user interacts with a DM or
   not. If DM took a more central role at some later date, this could become the primary method of setting the residual.

.seealso: DMTSSetContext(), TSSetFunction(), DMTSSetJacobian(), DMTSGetSolutionFunction()
@*/
PetscErrorCode DMTSSetSolutionFunction(DM dm,TSSolutionFunction func,void *ctx)
{
  PetscErrorCode ierr;
  DMTS           tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMTSWrite(dm,&tsdm);CHKERRQ(ierr);
  if (func) tsdm->ops->solution = func;
  if (ctx)  tsdm->solutionctx   = ctx;
  PetscFunctionReturn(0);
}

/*@C
   DMTSSetForcingFunction - set TS forcing function evaluation function

   Not Collective

   Input Parameters:
+  dm - DM to be used with TS
.  f - forcing function evaluation routine
-  ctx - context for solution evaluation

    Calling sequence of func:
$     PetscErrorCode func (TS ts,PetscReal t,Vec f,void *ctx);

+   ts - timestep context
.   t - current timestep
.   f - output vector
-   ctx - [optional] user-defined function context

   Level: advanced

   Note:
   TSSetForcingFunction() is normally used, but it calls this function internally because the user context is actually
   associated with the DM.  This makes the interface consistent regardless of whether the user interacts with a DM or
   not. If DM took a more central role at some later date, this could become the primary method of setting the residual.

.seealso: DMTSSetContext(), TSSetFunction(), DMTSSetJacobian(), TSSetForcingFunction(), DMTSGetForcingFunction()
@*/
PetscErrorCode DMTSSetForcingFunction(DM dm,TSForcingFunction f,void *ctx)
{
  PetscErrorCode ierr;
  DMTS           tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMTSWrite(dm,&tsdm);CHKERRQ(ierr);
  if (f)    tsdm->ops->forcing = f;
  if (ctx)  tsdm->forcingctx   = ctx;
  PetscFunctionReturn(0);
}

/*@C
   DMTSGetForcingFunction - get TS forcing function evaluation function

   Not Collective

   Input Parameter:
.   dm - DM to be used with TS

   Output Parameters:
+  f - forcing function evaluation function; see TSForcingFunction for details
-  ctx - context for solution evaluation

   Level: advanced

   Note:
   TSSetForcingFunction() is normally used, but it calls this function internally because the user context is actually
   associated with the DM.  This makes the interface consistent regardless of whether the user interacts with a DM or
   not. If DM took a more central role at some later date, this could become the primary method of setting the residual.

.seealso: DMTSSetContext(), TSSetFunction(), DMTSSetJacobian(), TSSetForcingFunction(), DMTSGetForcingFunction()
@*/
PetscErrorCode DMTSGetForcingFunction(DM dm,TSForcingFunction *f,void **ctx)
{
  PetscErrorCode ierr;
  DMTS           tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMTSWrite(dm,&tsdm);CHKERRQ(ierr);
  if (f)   *f   = tsdm->ops->forcing;
  if (ctx) *ctx = tsdm->forcingctx;
  PetscFunctionReturn(0);
}

/*@C
   DMTSGetRHSFunction - get TS explicit residual evaluation function

   Not Collective

   Input Parameter:
.  dm - DM to be used with TS

   Output Parameters:
+  func - residual evaluation function, see TSSetRHSFunction() for calling sequence
-  ctx - context for residual evaluation

   Level: advanced

   Note:
   TSGetFunction() is normally used, but it calls this function internally because the user context is actually
   associated with the DM.

.seealso: DMTSSetContext(), DMTSSetRHSFunction(), TSSetRHSFunction()
@*/
PetscErrorCode DMTSGetRHSFunction(DM dm,TSRHSFunction *func,void **ctx)
{
  PetscErrorCode ierr;
  DMTS           tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMTS(dm,&tsdm);CHKERRQ(ierr);
  if (func) *func = tsdm->ops->rhsfunction;
  if (ctx)  *ctx = tsdm->rhsfunctionctx;
  PetscFunctionReturn(0);
}

/*@C
   DMTSSetIJacobian - set TS Jacobian evaluation function

   Not Collective

   Input Parameters:
+  dm - DM to be used with TS
.  func - Jacobian evaluation routine
-  ctx - context for residual evaluation

   Calling sequence of f:
$    PetscErrorCode f(TS ts,PetscReal t,Vec U,Vec U_t,PetscReal a,Mat Amat,Mat Pmat,void *ctx);

+  t    - time at step/stage being solved
.  U    - state vector
.  U_t  - time derivative of state vector
.  a    - shift
.  Amat - (approximate) Jacobian of F(t,U,W+a*U), equivalent to dF/dU + a*dF/dU_t
.  Pmat - matrix used for constructing preconditioner, usually the same as Amat
-  ctx  - [optional] user-defined context for matrix evaluation routine

   Level: advanced

   Note:
   TSSetJacobian() is normally used, but it calls this function internally because the user context is actually
   associated with the DM.  This makes the interface consistent regardless of whether the user interacts with a DM or
   not. If DM took a more central role at some later date, this could become the primary method of setting the Jacobian.

.seealso: DMTSSetContext(), TSSetRHSFunction(), DMTSGetJacobian(), TSSetIJacobian(), TSSetIFunction()
@*/
PetscErrorCode DMTSSetIJacobian(DM dm,TSIJacobian func,void *ctx)
{
  PetscErrorCode ierr;
  DMTS           sdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMTSWrite(dm,&sdm);CHKERRQ(ierr);
  if (func) sdm->ops->ijacobian = func;
  if (ctx)  sdm->ijacobianctx   = ctx;
  PetscFunctionReturn(0);
}

/*@C
   DMTSGetIJacobian - get TS Jacobian evaluation function

   Not Collective

   Input Parameter:
.  dm - DM to be used with TS

   Output Parameters:
+  func - Jacobian evaluation function, see TSSetIJacobian() for calling sequence
-  ctx - context for residual evaluation

   Level: advanced

   Note:
   TSGetJacobian() is normally used, but it calls this function internally because the user context is actually
   associated with the DM.  This makes the interface consistent regardless of whether the user interacts with a DM or
   not. If DM took a more central role at some later date, this could become the primary method of setting the Jacobian.

.seealso: DMTSSetContext(), TSSetFunction(), DMTSSetJacobian()
@*/
PetscErrorCode DMTSGetIJacobian(DM dm,TSIJacobian *func,void **ctx)
{
  PetscErrorCode ierr;
  DMTS           tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMTS(dm,&tsdm);CHKERRQ(ierr);
  if (func) *func = tsdm->ops->ijacobian;
  if (ctx)  *ctx = tsdm->ijacobianctx;
  PetscFunctionReturn(0);
}

/*@C
   DMTSSetRHSJacobian - set TS Jacobian evaluation function

   Not Collective

   Input Parameters:
+  dm - DM to be used with TS
.  func - Jacobian evaluation routine
-  ctx - context for residual evaluation

   Calling sequence of func:
$     PetscErrorCode func(TS ts,PetscReal t,Vec u,Mat A,Mat B,void *ctx);

+  t - current timestep
.  u - input vector
.  Amat - (approximate) Jacobian matrix
.  Pmat - matrix from which preconditioner is to be constructed (usually the same as Amat)
-  ctx - [optional] user-defined context for matrix evaluation routine

   Level: advanced

   Note:
   TSSetJacobian() is normally used, but it calls this function internally because the user context is actually
   associated with the DM.  This makes the interface consistent regardless of whether the user interacts with a DM or
   not. If DM took a more central role at some later date, this could become the primary method of setting the Jacobian.

.seealso: DMTSSetContext(), TSSetFunction(), DMTSGetJacobian(), TSSetRHSJacobian()
@*/
PetscErrorCode DMTSSetRHSJacobian(DM dm,TSRHSJacobian func,void *ctx)
{
  PetscErrorCode ierr;
  DMTS           tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMTSWrite(dm,&tsdm);CHKERRQ(ierr);
  if (func) tsdm->ops->rhsjacobian = func;
  if (ctx)  tsdm->rhsjacobianctx = ctx;
  PetscFunctionReturn(0);
}

/*@C
   DMTSGetRHSJacobian - get TS Jacobian evaluation function

   Not Collective

   Input Parameter:
.  dm - DM to be used with TS

   Output Parameters:
+  func - Jacobian evaluation function, see TSSetRHSJacobian() for calling sequence
-  ctx - context for residual evaluation

   Level: advanced

   Note:
   TSGetJacobian() is normally used, but it calls this function internally because the user context is actually
   associated with the DM.  This makes the interface consistent regardless of whether the user interacts with a DM or
   not. If DM took a more central role at some later date, this could become the primary method of setting the Jacobian.

.seealso: DMTSSetContext(), TSSetRHSFunction(), DMTSSetRHSJacobian(), TSSetRHSJacobian()
@*/
PetscErrorCode DMTSGetRHSJacobian(DM dm,TSRHSJacobian *func,void **ctx)
{
  PetscErrorCode ierr;
  DMTS           tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMTS(dm,&tsdm);CHKERRQ(ierr);
  if (func) *func = tsdm->ops->rhsjacobian;
  if (ctx)  *ctx = tsdm->rhsjacobianctx;
  PetscFunctionReturn(0);
}

/*@C
   DMTSSetIFunctionSerialize - sets functions used to view and load a IFunction context

   Not Collective

   Input Parameters:
+  dm - DM to be used with TS
.  view - viewer function
-  load - loading function

   Level: advanced

.seealso: DMTSSetContext(), TSSetFunction(), DMTSSetJacobian()
@*/
PetscErrorCode DMTSSetIFunctionSerialize(DM dm,PetscErrorCode (*view)(void*,PetscViewer),PetscErrorCode (*load)(void**,PetscViewer))
{
  PetscErrorCode ierr;
  DMTS           tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMTSWrite(dm,&tsdm);CHKERRQ(ierr);
  tsdm->ops->ifunctionview = view;
  tsdm->ops->ifunctionload = load;
  PetscFunctionReturn(0);
}

/*@C
   DMTSSetIJacobianSerialize - sets functions used to view and load a IJacobian context

   Not Collective

   Input Parameters:
+  dm - DM to be used with TS
.  view - viewer function
-  load - loading function

   Level: advanced

.seealso: DMTSSetContext(), TSSetFunction(), DMTSSetJacobian()
@*/
PetscErrorCode DMTSSetIJacobianSerialize(DM dm,PetscErrorCode (*view)(void*,PetscViewer),PetscErrorCode (*load)(void**,PetscViewer))
{
  PetscErrorCode ierr;
  DMTS           tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMTSWrite(dm,&tsdm);CHKERRQ(ierr);
  tsdm->ops->ijacobianview = view;
  tsdm->ops->ijacobianload = load;
  PetscFunctionReturn(0);
}
