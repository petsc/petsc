#include <petsc/private/tsimpl.h> /*I "petscts.h" I*/
#include <petsc/private/dmimpl.h>

static PetscErrorCode DMTSUnsetRHSFunctionContext_DMTS(DMTS tsdm)
{
  PetscFunctionBegin;
  PetscCall(PetscObjectCompose((PetscObject)tsdm, "rhs function ctx", NULL));
  tsdm->rhsfunctionctxcontainer = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMTSUnsetRHSJacobianContext_DMTS(DMTS tsdm)
{
  PetscFunctionBegin;
  PetscCall(PetscObjectCompose((PetscObject)tsdm, "rhs jacobian ctx", NULL));
  tsdm->rhsjacobianctxcontainer = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMTSUnsetIFunctionContext_DMTS(DMTS tsdm)
{
  PetscFunctionBegin;
  PetscCall(PetscObjectCompose((PetscObject)tsdm, "ifunction ctx", NULL));
  tsdm->ifunctionctxcontainer = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMTSUnsetIJacobianContext_DMTS(DMTS tsdm)
{
  PetscFunctionBegin;
  PetscCall(PetscObjectCompose((PetscObject)tsdm, "ijacobian ctx", NULL));
  tsdm->ijacobianctxcontainer = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMTSUnsetI2FunctionContext_DMTS(DMTS tsdm)
{
  PetscFunctionBegin;
  PetscCall(PetscObjectCompose((PetscObject)tsdm, "i2function ctx", NULL));
  tsdm->i2functionctxcontainer = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMTSUnsetI2JacobianContext_DMTS(DMTS tsdm)
{
  PetscFunctionBegin;
  PetscCall(PetscObjectCompose((PetscObject)tsdm, "i2jacobian ctx", NULL));
  tsdm->i2jacobianctxcontainer = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMTSDestroy(DMTS *kdm)
{
  PetscFunctionBegin;
  if (!*kdm) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*kdm), DMTS_CLASSID, 1);
  if (--((PetscObject)(*kdm))->refct > 0) {
    *kdm = NULL;
    PetscFunctionReturn(0);
  }
  PetscCall(DMTSUnsetRHSFunctionContext_DMTS(*kdm));
  PetscCall(DMTSUnsetRHSJacobianContext_DMTS(*kdm));
  PetscCall(DMTSUnsetIFunctionContext_DMTS(*kdm));
  PetscCall(DMTSUnsetIJacobianContext_DMTS(*kdm));
  PetscCall(DMTSUnsetI2FunctionContext_DMTS(*kdm));
  PetscCall(DMTSUnsetI2JacobianContext_DMTS(*kdm));
  PetscTryTypeMethod(*kdm, destroy);
  PetscCall(PetscHeaderDestroy(kdm));
  PetscFunctionReturn(0);
}

PetscErrorCode DMTSLoad(DMTS kdm, PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscCall(PetscViewerBinaryRead(viewer, &kdm->ops->ifunction, 1, NULL, PETSC_FUNCTION));
  PetscCall(PetscViewerBinaryRead(viewer, &kdm->ops->ifunctionview, 1, NULL, PETSC_FUNCTION));
  PetscCall(PetscViewerBinaryRead(viewer, &kdm->ops->ifunctionload, 1, NULL, PETSC_FUNCTION));
  if (kdm->ops->ifunctionload) {
    void *ctx;

    PetscCall(PetscContainerGetPointer(kdm->ifunctionctxcontainer, &ctx));
    PetscCall((*kdm->ops->ifunctionload)(&ctx, viewer));
  }
  PetscCall(PetscViewerBinaryRead(viewer, &kdm->ops->ijacobian, 1, NULL, PETSC_FUNCTION));
  PetscCall(PetscViewerBinaryRead(viewer, &kdm->ops->ijacobianview, 1, NULL, PETSC_FUNCTION));
  PetscCall(PetscViewerBinaryRead(viewer, &kdm->ops->ijacobianload, 1, NULL, PETSC_FUNCTION));
  if (kdm->ops->ijacobianload) {
    void *ctx;

    PetscCall(PetscContainerGetPointer(kdm->ijacobianctxcontainer, &ctx));
    PetscCall((*kdm->ops->ijacobianload)(&ctx, viewer));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMTSView(DMTS kdm, PetscViewer viewer)
{
  PetscBool isascii, isbinary;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERBINARY, &isbinary));
  if (isascii) {
#if defined(PETSC_SERIALIZE_FUNCTIONS)
    const char *fname;

    PetscCall(PetscFPTFind(kdm->ops->ifunction, &fname));
    if (fname) PetscCall(PetscViewerASCIIPrintf(viewer, "  IFunction used by TS: %s\n", fname));
    PetscCall(PetscFPTFind(kdm->ops->ijacobian, &fname));
    if (fname) PetscCall(PetscViewerASCIIPrintf(viewer, "  IJacobian function used by TS: %s\n", fname));
#endif
  } else if (isbinary) {
    struct {
      TSIFunction ifunction;
    } funcstruct;
    struct {
      PetscErrorCode (*ifunctionview)(void *, PetscViewer);
    } funcviewstruct;
    struct {
      PetscErrorCode (*ifunctionload)(void **, PetscViewer);
    } funcloadstruct;
    struct {
      TSIJacobian ijacobian;
    } jacstruct;
    struct {
      PetscErrorCode (*ijacobianview)(void *, PetscViewer);
    } jacviewstruct;
    struct {
      PetscErrorCode (*ijacobianload)(void **, PetscViewer);
    } jacloadstruct;

    funcstruct.ifunction         = kdm->ops->ifunction;
    funcviewstruct.ifunctionview = kdm->ops->ifunctionview;
    funcloadstruct.ifunctionload = kdm->ops->ifunctionload;
    PetscCall(PetscViewerBinaryWrite(viewer, &funcstruct, 1, PETSC_FUNCTION));
    PetscCall(PetscViewerBinaryWrite(viewer, &funcviewstruct, 1, PETSC_FUNCTION));
    PetscCall(PetscViewerBinaryWrite(viewer, &funcloadstruct, 1, PETSC_FUNCTION));
    if (kdm->ops->ifunctionview) {
      void *ctx;

      PetscCall(PetscContainerGetPointer(kdm->ifunctionctxcontainer, &ctx));
      PetscCall((*kdm->ops->ifunctionview)(ctx, viewer));
    }
    jacstruct.ijacobian         = kdm->ops->ijacobian;
    jacviewstruct.ijacobianview = kdm->ops->ijacobianview;
    jacloadstruct.ijacobianload = kdm->ops->ijacobianload;
    PetscCall(PetscViewerBinaryWrite(viewer, &jacstruct, 1, PETSC_FUNCTION));
    PetscCall(PetscViewerBinaryWrite(viewer, &jacviewstruct, 1, PETSC_FUNCTION));
    PetscCall(PetscViewerBinaryWrite(viewer, &jacloadstruct, 1, PETSC_FUNCTION));
    if (kdm->ops->ijacobianview) {
      void *ctx;

      PetscCall(PetscContainerGetPointer(kdm->ijacobianctxcontainer, &ctx));
      PetscCall((*kdm->ops->ijacobianview)(ctx, viewer));
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMTSCreate(MPI_Comm comm, DMTS *kdm)
{
  PetscFunctionBegin;
  PetscCall(TSInitializePackage());
  PetscCall(PetscHeaderCreate(*kdm, DMTS_CLASSID, "DMTS", "DMTS", "DMTS", comm, DMTSDestroy, DMTSView));
  PetscFunctionReturn(0);
}

/* Attaches the DMTS to the coarse level.
 * Under what conditions should we copy versus duplicate?
 */
static PetscErrorCode DMCoarsenHook_DMTS(DM dm, DM dmc, void *ctx)
{
  PetscFunctionBegin;
  PetscCall(DMCopyDMTS(dm, dmc));
  PetscFunctionReturn(0);
}

/* This could restrict auxiliary information to the coarse level.
 */
static PetscErrorCode DMRestrictHook_DMTS(DM dm, Mat Restrict, Vec rscale, Mat Inject, DM dmc, void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMSubDomainHook_DMTS(DM dm, DM subdm, void *ctx)
{
  PetscFunctionBegin;
  PetscCall(DMCopyDMTS(dm, subdm));
  PetscFunctionReturn(0);
}

/* This could restrict auxiliary information to the coarse level.
 */
static PetscErrorCode DMSubDomainRestrictHook_DMTS(DM dm, VecScatter gscat, VecScatter lscat, DM subdm, void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

/*@C
   DMTSCopy - copies the information in a `DMTS` to another `DMTS`

   Not Collective

   Input Parameters:
+  kdm - Original `DMTS`
-  nkdm - `DMTS` to receive the data, should have been created with `DMTSCreate()`

   Level: developer

.seealso: [](chapter_ts), `DMTSCreate()`, `DMTSDestroy()`
@*/
PetscErrorCode DMTSCopy(DMTS kdm, DMTS nkdm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(kdm, DMTS_CLASSID, 1);
  PetscValidHeaderSpecific(nkdm, DMTS_CLASSID, 2);
  nkdm->ops->rhsfunction = kdm->ops->rhsfunction;
  nkdm->ops->rhsjacobian = kdm->ops->rhsjacobian;
  nkdm->ops->ifunction   = kdm->ops->ifunction;
  nkdm->ops->ijacobian   = kdm->ops->ijacobian;
  nkdm->ops->i2function  = kdm->ops->i2function;
  nkdm->ops->i2jacobian  = kdm->ops->i2jacobian;
  nkdm->ops->solution    = kdm->ops->solution;
  nkdm->ops->destroy     = kdm->ops->destroy;
  nkdm->ops->duplicate   = kdm->ops->duplicate;

  nkdm->solutionctx             = kdm->solutionctx;
  nkdm->rhsfunctionctxcontainer = kdm->rhsfunctionctxcontainer;
  nkdm->rhsjacobianctxcontainer = kdm->rhsjacobianctxcontainer;
  nkdm->ifunctionctxcontainer   = kdm->ifunctionctxcontainer;
  nkdm->ijacobianctxcontainer   = kdm->ijacobianctxcontainer;
  nkdm->i2functionctxcontainer  = kdm->i2functionctxcontainer;
  nkdm->i2jacobianctxcontainer  = kdm->i2jacobianctxcontainer;
  if (nkdm->rhsfunctionctxcontainer) PetscCall(PetscObjectCompose((PetscObject)nkdm, "rhs function ctx", (PetscObject)nkdm->rhsfunctionctxcontainer));
  if (nkdm->rhsjacobianctxcontainer) PetscCall(PetscObjectCompose((PetscObject)nkdm, "rhs jacobian ctx", (PetscObject)nkdm->rhsjacobianctxcontainer));
  if (nkdm->ifunctionctxcontainer) PetscCall(PetscObjectCompose((PetscObject)nkdm, "ifunction ctx", (PetscObject)nkdm->ifunctionctxcontainer));
  if (nkdm->ijacobianctxcontainer) PetscCall(PetscObjectCompose((PetscObject)nkdm, "ijacobian ctx", (PetscObject)nkdm->ijacobianctxcontainer));
  if (nkdm->i2functionctxcontainer) PetscCall(PetscObjectCompose((PetscObject)nkdm, "i2function ctx", (PetscObject)nkdm->i2functionctxcontainer));
  if (nkdm->i2jacobianctxcontainer) PetscCall(PetscObjectCompose((PetscObject)nkdm, "i2jacobian ctx", (PetscObject)nkdm->i2jacobianctxcontainer));

  nkdm->data = kdm->data;

  /*
  nkdm->fortran_func_pointers[0] = kdm->fortran_func_pointers[0];
  nkdm->fortran_func_pointers[1] = kdm->fortran_func_pointers[1];
  nkdm->fortran_func_pointers[2] = kdm->fortran_func_pointers[2];
  */

  /* implementation specific copy hooks */
  PetscTryTypeMethod(kdm, duplicate, nkdm);
  PetscFunctionReturn(0);
}

/*@C
   DMGetDMTS - get read-only private `DMTS` context from a `DM`

   Not Collective

   Input Parameter:
.  dm - `DM` to be used with `TS`

   Output Parameter:
.  tsdm - private `DMTS` context

   Level: developer

   Notes:
   Use `DMGetDMTSWrite()` if write access is needed. The `DMTSSetXXX()` API should be used wherever possible.

.seealso: [](chapter_ts), `DMTS`, `DMGetDMTSWrite()`
@*/
PetscErrorCode DMGetDMTS(DM dm, DMTS *tsdm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  *tsdm = (DMTS)dm->dmts;
  if (!*tsdm) {
    PetscCall(PetscInfo(dm, "Creating new DMTS\n"));
    PetscCall(DMTSCreate(PetscObjectComm((PetscObject)dm), tsdm));
    dm->dmts            = (PetscObject)*tsdm;
    (*tsdm)->originaldm = dm;
    PetscCall(DMCoarsenHookAdd(dm, DMCoarsenHook_DMTS, DMRestrictHook_DMTS, NULL));
    PetscCall(DMSubDomainHookAdd(dm, DMSubDomainHook_DMTS, DMSubDomainRestrictHook_DMTS, NULL));
  }
  PetscFunctionReturn(0);
}

/*@C
   DMGetDMTSWrite - get write access to private `DMTS` context from a `DM`

   Not Collective

   Input Parameter:
.  dm - `DM` to be used with `TS`

   Output Parameter:
.  tsdm - private `DMTS` context

   Level: developer

.seealso: [](chapter_ts), `DMTS`, `DMGetDMTS()`
@*/
PetscErrorCode DMGetDMTSWrite(DM dm, DMTS *tsdm)
{
  DMTS sdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTS(dm, &sdm));
  PetscCheck(sdm->originaldm, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "DMTS has a NULL originaldm");
  if (sdm->originaldm != dm) { /* Copy on write */
    DMTS oldsdm = sdm;
    PetscCall(PetscInfo(dm, "Copying DMTS due to write\n"));
    PetscCall(DMTSCreate(PetscObjectComm((PetscObject)dm), &sdm));
    PetscCall(DMTSCopy(oldsdm, sdm));
    PetscCall(DMTSDestroy((DMTS *)&dm->dmts));
    dm->dmts        = (PetscObject)sdm;
    sdm->originaldm = dm;
  }
  *tsdm = sdm;
  PetscFunctionReturn(0);
}

/*@C
   DMCopyDMTS - copies a `DM` context to a new `DM`

   Logically Collective

   Input Parameters:
+  dmsrc - `DM` to obtain context from
-  dmdest - `DM` to add context to

   Level: developer

   Note:
   The context is copied by reference. This function does not ensure that a context exists.

.seealso: [](chapter_ts), `DMTS`, `DMGetDMTS()`, `TSSetDM()`
@*/
PetscErrorCode DMCopyDMTS(DM dmsrc, DM dmdest)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dmsrc, DM_CLASSID, 1);
  PetscValidHeaderSpecific(dmdest, DM_CLASSID, 2);
  PetscCall(DMTSDestroy((DMTS *)&dmdest->dmts));
  dmdest->dmts = dmsrc->dmts;
  PetscCall(PetscObjectReference(dmdest->dmts));
  PetscCall(DMCoarsenHookAdd(dmdest, DMCoarsenHook_DMTS, DMRestrictHook_DMTS, NULL));
  PetscCall(DMSubDomainHookAdd(dmdest, DMSubDomainHook_DMTS, DMSubDomainRestrictHook_DMTS, NULL));
  PetscFunctionReturn(0);
}

/*@C
   DMTSSetIFunction - set `TS` implicit function evaluation function

   Not Collective

   Input Parameters:
+  dm - `DM` to be used with `TS`
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
   `TSSetFunction()` is normally used, but it calls this function internally because the user context is actually
   associated with the `DM`.  This makes the interface consistent regardless of whether the user interacts with a `DM` or
   not. If DM took a more central role at some later date, this could become the primary method of setting the residual.

.seealso: [](chapter_ts), `TS`, `DM`, `DMTSSetContext()`, `TSSetIFunction()`, `DMTSSetJacobian()`
@*/
PetscErrorCode DMTSSetIFunction(DM dm, TSIFunction func, void *ctx)
{
  DMTS tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTSWrite(dm, &tsdm));
  if (func) tsdm->ops->ifunction = func;
  if (ctx) {
    PetscContainer ctxcontainer;
    PetscCall(PetscContainerCreate(PetscObjectComm((PetscObject)tsdm), &ctxcontainer));
    PetscCall(PetscContainerSetPointer(ctxcontainer, ctx));
    PetscCall(PetscObjectCompose((PetscObject)tsdm, "ifunction ctx", (PetscObject)ctxcontainer));
    tsdm->ifunctionctxcontainer = ctxcontainer;
    PetscCall(PetscContainerDestroy(&ctxcontainer));
  }
  PetscFunctionReturn(0);
}

/*@C
   DMTSSetIFunctionContextDestroy - set `TS` implicit evaluation context destroy function

   Not Collective

   Input Parameters:
+  dm - `DM` to be used with `TS`
-  f - implicit evaluation context destroy function

   Level: advanced

.seealso: [](chapter_ts), `DM`, `TS`, `DMTSSetIFunction()`, `TSSetIFunction()`
@*/
PetscErrorCode DMTSSetIFunctionContextDestroy(DM dm, PetscErrorCode (*f)(void *))
{
  DMTS tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTSWrite(dm, &tsdm));
  if (tsdm->ifunctionctxcontainer) PetscCall(PetscContainerSetUserDestroy(tsdm->ifunctionctxcontainer, f));
  PetscFunctionReturn(0);
}

PetscErrorCode DMTSUnsetIFunctionContext_Internal(DM dm)
{
  DMTS tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTSWrite(dm, &tsdm));
  PetscCall(DMTSUnsetIFunctionContext_DMTS(tsdm));
  PetscFunctionReturn(0);
}

/*@C
   DMTSGetIFunction - get `TS` implicit residual evaluation function

   Not Collective

   Input Parameter:
.  dm - `DM` to be used with `TS`

   Output Parameters:
+  func - function evaluation function, see `TSSetIFunction()` for calling sequence
-  ctx - context for residual evaluation

   Level: advanced

   Note:
   `TSGetFunction()` is normally used, but it calls this function internally because the user context is actually
   associated with the `DM`.

.seealso: [](chapter_ts), `TS`, `DM`, `DMTSSetContext()`, `DMTSSetFunction()`, `TSSetFunction()`
@*/
PetscErrorCode DMTSGetIFunction(DM dm, TSIFunction *func, void **ctx)
{
  DMTS tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTS(dm, &tsdm));
  if (func) *func = tsdm->ops->ifunction;
  if (ctx) {
    if (tsdm->ifunctionctxcontainer) PetscCall(PetscContainerGetPointer(tsdm->ifunctionctxcontainer, ctx));
    else *ctx = NULL;
  }
  PetscFunctionReturn(0);
}

/*@C
   DMTSSetI2Function - set `TS` implicit function evaluation function for 2nd order systems

   Not Collective

   Input Parameters:
+  dm - `DM` to be used with `TS`
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
   `TSSetI2Function()` is normally used, but it calls this function internally because the user context is actually
   associated with the `DM`.

.seealso: [](chapter_ts), `DM`, `TS`, `TSSetI2Function()`
@*/
PetscErrorCode DMTSSetI2Function(DM dm, TSI2Function fun, void *ctx)
{
  DMTS tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTSWrite(dm, &tsdm));
  if (fun) tsdm->ops->i2function = fun;
  if (ctx) {
    PetscContainer ctxcontainer;
    PetscCall(PetscContainerCreate(PetscObjectComm((PetscObject)tsdm), &ctxcontainer));
    PetscCall(PetscContainerSetPointer(ctxcontainer, ctx));
    PetscCall(PetscObjectCompose((PetscObject)tsdm, "i2function ctx", (PetscObject)ctxcontainer));
    tsdm->i2functionctxcontainer = ctxcontainer;
    PetscCall(PetscContainerDestroy(&ctxcontainer));
  }
  PetscFunctionReturn(0);
}

/*@C
   DMTSSetI2FunctionContextDestroy - set `TS` implicit evaluation for 2nd order systems context destroy

   Not Collective

   Input Parameters:
+  dm - `DM` to be used with `TS`
-  f - implicit evaluation context destroy function

   Level: advanced

   Note:
   `TSSetI2FunctionContextDestroy()` is normally used, but it calls this function internally because the user context is actually
   associated with the `DM`.

.seealso: [](chapter_ts), `TSSetI2FunctionContextDestroy()`, `DMTSSetI2Function()`, `TSSetI2Function()`
@*/
PetscErrorCode DMTSSetI2FunctionContextDestroy(DM dm, PetscErrorCode (*f)(void *))
{
  DMTS tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTSWrite(dm, &tsdm));
  if (tsdm->i2functionctxcontainer) PetscCall(PetscContainerSetUserDestroy(tsdm->i2functionctxcontainer, f));
  PetscFunctionReturn(0);
}

PetscErrorCode DMTSUnsetI2FunctionContext_Internal(DM dm)
{
  DMTS tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTSWrite(dm, &tsdm));
  PetscCall(DMTSUnsetI2FunctionContext_DMTS(tsdm));
  PetscFunctionReturn(0);
}

/*@C
   DMTSGetI2Function - get `TS` implicit residual evaluation function for 2nd order systems

   Not Collective

   Input Parameter:
.  dm - `DM` to be used with `TS`

   Output Parameters:
+  fun - function evaluation function, see `TSSetI2Function()` for calling sequence
-  ctx - context for residual evaluation

   Level: advanced

   Note:
   `TSGetI2Function()` is normally used, but it calls this function internally because the user context is actually
   associated with the `DM`.

.seealso: [](chapter_ts), `DM`, `TS`, `DMTSSetI2Function()`, `TSGetI2Function()`
@*/
PetscErrorCode DMTSGetI2Function(DM dm, TSI2Function *fun, void **ctx)
{
  DMTS tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTS(dm, &tsdm));
  if (fun) *fun = tsdm->ops->i2function;
  if (ctx) {
    if (tsdm->i2functionctxcontainer) PetscCall(PetscContainerGetPointer(tsdm->i2functionctxcontainer, ctx));
    else *ctx = NULL;
  }
  PetscFunctionReturn(0);
}

/*@C
   DMTSSetI2Jacobian - set `TS` implicit Jacobian evaluation function for 2nd order systems

   Not Collective

   Input Parameters:
+  dm - `DM` to be used with `TS`
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
   `TSSetI2Jacobian()` is normally used, but it calls this function internally because the user context is actually
   associated with the `DM`.

.seealso: [](chapter_ts), `DM`, `TS`, `TSSetI2Jacobian()`
@*/
PetscErrorCode DMTSSetI2Jacobian(DM dm, TSI2Jacobian jac, void *ctx)
{
  DMTS tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTSWrite(dm, &tsdm));
  if (jac) tsdm->ops->i2jacobian = jac;
  if (ctx) {
    PetscContainer ctxcontainer;
    PetscCall(PetscContainerCreate(PetscObjectComm((PetscObject)tsdm), &ctxcontainer));
    PetscCall(PetscContainerSetPointer(ctxcontainer, ctx));
    PetscCall(PetscObjectCompose((PetscObject)tsdm, "i2jacobian ctx", (PetscObject)ctxcontainer));
    tsdm->i2jacobianctxcontainer = ctxcontainer;
    PetscCall(PetscContainerDestroy(&ctxcontainer));
  }
  PetscFunctionReturn(0);
}

/*@C
   DMTSSetI2JacobianContextDestroy - set `TS` implicit Jacobian evaluation for 2nd order systems context destroy function

   Not Collective

   Input Parameters:
+  dm - `DM` to be used with `TS`
-  f - implicit Jacobian evaluation context destroy function

   Level: advanced

.seealso: [](chapter_ts), `DM`, `TS`, `TSSetI2JacobianContextDestroy()`, `DMTSSetI2Jacobian()`, `TSSetI2Jacobian()`
@*/
PetscErrorCode DMTSSetI2JacobianContextDestroy(DM dm, PetscErrorCode (*f)(void *))
{
  DMTS tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTSWrite(dm, &tsdm));
  if (tsdm->i2jacobianctxcontainer) PetscCall(PetscContainerSetUserDestroy(tsdm->i2jacobianctxcontainer, f));
  PetscFunctionReturn(0);
}

PetscErrorCode DMTSUnsetI2JacobianContext_Internal(DM dm)
{
  DMTS tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTSWrite(dm, &tsdm));
  PetscCall(DMTSUnsetI2JacobianContext_DMTS(tsdm));
  PetscFunctionReturn(0);
}

/*@C
   DMTSGetI2Jacobian - get `TS` implicit Jacobian evaluation function for 2nd order systems

   Not Collective

   Input Parameter:
.  dm - `DM` to be used with `TS`

   Output Parameters:
+  jac - Jacobian evaluation function, see `TSSetI2Jacobian()` for calling sequence
-  ctx - context for Jacobian evaluation

   Level: advanced

   Note:
   `TSGetI2Jacobian()` is normally used, but it calls this function internally because the user context is actually
   associated with the `DM`.

.seealso: [](chapter_ts), `DM`, `TS`, `DMTSSetI2Jacobian()`, `TSGetI2Jacobian()`
@*/
PetscErrorCode DMTSGetI2Jacobian(DM dm, TSI2Jacobian *jac, void **ctx)
{
  DMTS tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTS(dm, &tsdm));
  if (jac) *jac = tsdm->ops->i2jacobian;
  if (ctx) {
    if (tsdm->i2jacobianctxcontainer) PetscCall(PetscContainerGetPointer(tsdm->i2jacobianctxcontainer, ctx));
    else *ctx = NULL;
  }
  PetscFunctionReturn(0);
}

/*@C
   DMTSSetRHSFunction - set `TS` explicit residual evaluation function

   Not Collective

   Input Parameters:
+  dm - `DM` to be used with `TS`
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
   `TSSetRHSFunction()` is normally used, but it calls this function internally because the user context is actually
   associated with the `DM`.  This makes the interface consistent regardless of whether the user interacts with a `DM` or
   not. If `DM` took a more central role at some later date, this could become the primary method of setting the residual.

.seealso: [](chapter_ts), `DM`, `TS`, `DMTSSetContext()`, `TSSetRHSFunction()`, `DMTSSetJacobian()`
@*/
PetscErrorCode DMTSSetRHSFunction(DM dm, TSRHSFunction func, void *ctx)
{
  DMTS tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTSWrite(dm, &tsdm));
  if (func) tsdm->ops->rhsfunction = func;
  if (ctx) {
    PetscContainer ctxcontainer;
    PetscCall(PetscContainerCreate(PetscObjectComm((PetscObject)tsdm), &ctxcontainer));
    PetscCall(PetscContainerSetPointer(ctxcontainer, ctx));
    PetscCall(PetscObjectCompose((PetscObject)tsdm, "rhs function ctx", (PetscObject)ctxcontainer));
    tsdm->rhsfunctionctxcontainer = ctxcontainer;
    PetscCall(PetscContainerDestroy(&ctxcontainer));
  }
  PetscFunctionReturn(0);
}

/*@C
   DMTSSetRHSFunctionContextDestroy - set `TS` explicit residual evaluation context destroy function

   Not Collective

   Input Parameters:
+  dm - `DM` to be used with `TS`
-  f - explicit evaluation context destroy function

   Level: advanced

   Note:
   `TSSetRHSFunctionContextDestroy()` is normally used, but it calls this function internally because the user context is actually
   associated with the `DM`.  This makes the interface consistent regardless of whether the user interacts with a `DM` or
   not.

   Developer Note:
   If `DM` took a more central role at some later date, this could become the primary method of setting the residual.

.seealso: [](chapter_ts), `TSSetRHSFunctionContextDestroy()`, `DMTSSetRHSFunction()`, `TSSetRHSFunction()`
@*/
PetscErrorCode DMTSSetRHSFunctionContextDestroy(DM dm, PetscErrorCode (*f)(void *))
{
  DMTS tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTSWrite(dm, &tsdm));
  if (tsdm->rhsfunctionctxcontainer) PetscCall(PetscContainerSetUserDestroy(tsdm->rhsfunctionctxcontainer, f));
  PetscFunctionReturn(0);
}

PetscErrorCode DMTSUnsetRHSFunctionContext_Internal(DM dm)
{
  DMTS tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTSWrite(dm, &tsdm));
  PetscCall(DMTSUnsetRHSFunctionContext_DMTS(tsdm));
  tsdm->rhsfunctionctxcontainer = NULL;
  PetscFunctionReturn(0);
}

/*@C
   DMTSSetTransientVariable - sets function to transform from state to transient variables

   Logically Collective

   Input Parameters:
+  dm - `DM` to be used with `TS`
.  tvar - a function that transforms to transient variables
-  ctx - a context for tvar

    Calling sequence of tvar:
$     PetscErrorCode tvar(TS ts,Vec p,Vec c,void *ctx);

+   ts - timestep context
.   p - input vector (primitive form)
.   c - output vector, transient variables (conservative form)
-   ctx - [optional] user-defined function context

   Level: advanced

   Notes:
   This is typically used to transform from primitive to conservative variables so that a time integrator (e.g., `TSBDF`)
   can be conservative.  In this context, primitive variables P are used to model the state (e.g., because they lead to
   well-conditioned formulations even in limiting cases such as low-Mach or zero porosity).  The transient variable is
   C(P), specified by calling this function.  An IFunction thus receives arguments (P, Cdot) and the IJacobian must be
   evaluated via the chain rule, as in

     dF/dP + shift * dF/dCdot dC/dP.

.seealso: [](chapter_ts), `TS`, `TSBDF`, `TSSetTransientVariable()`, `DMTSGetTransientVariable()`, `DMTSSetIFunction()`, `DMTSSetIJacobian()`
@*/
PetscErrorCode DMTSSetTransientVariable(DM dm, TSTransientVariable tvar, void *ctx)
{
  DMTS dmts;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTSWrite(dm, &dmts));
  dmts->ops->transientvar = tvar;
  dmts->transientvarctx   = ctx;
  PetscFunctionReturn(0);
}

/*@C
   DMTSGetTransientVariable - gets function to transform from state to transient variables set with `DMTSSetTransientVariable()`

   Logically Collective

   Input Parameter:
.  dm - `DM` to be used with `TS`

   Output Parameters:
+  tvar - a function that transforms to transient variables
-  ctx - a context for tvar

   Level: advanced

.seealso: [](chapter_ts), `DM`, `DMTSSetTransientVariable()`, `DMTSGetIFunction()`, `DMTSGetIJacobian()`
@*/
PetscErrorCode DMTSGetTransientVariable(DM dm, TSTransientVariable *tvar, void *ctx)
{
  DMTS dmts;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTS(dm, &dmts));
  if (tvar) *tvar = dmts->ops->transientvar;
  if (ctx) *(void **)ctx = dmts->transientvarctx;
  PetscFunctionReturn(0);
}

/*@C
   DMTSGetSolutionFunction - gets the `TS` solution evaluation function

   Not Collective

   Input Parameter:
.  dm - `DM` to be used with `TS`

   Output Parameters:
+  func - solution function evaluation function, see `TSSetSolution()` for calling sequence
-  ctx - context for solution evaluation

   Level: advanced

.seealso: [](chapter_ts), `TS`, `DM`, `DMTSSetContext()`, `TSSetFunction()`, `DMTSSetJacobian()`, `DMTSSetSolutionFunction()`
@*/
PetscErrorCode DMTSGetSolutionFunction(DM dm, TSSolutionFunction *func, void **ctx)
{
  DMTS tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTS(dm, &tsdm));
  if (func) *func = tsdm->ops->solution;
  if (ctx) *ctx = tsdm->solutionctx;
  PetscFunctionReturn(0);
}

/*@C
   DMTSSetSolutionFunction - set `TS` solution evaluation function

   Not Collective

   Input Parameters:
+  dm - `DM` to be used with `TS`
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
   `TSSetSolutionFunction()` is normally used, but it calls this function internally because the user context is actually
   associated with the `DM`.  This makes the interface consistent regardless of whether the user interacts with a `DM` or
   not. If `DM` took a more central role at some later date, this could become the primary method of setting the residual.

.seealso: [](chapter_ts), `DM`, `TS`, `DMTSSetContext()`, `TSSetFunction()`, `DMTSSetJacobian()`, `DMTSGetSolutionFunction()`
@*/
PetscErrorCode DMTSSetSolutionFunction(DM dm, TSSolutionFunction func, void *ctx)
{
  DMTS tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTSWrite(dm, &tsdm));
  if (func) tsdm->ops->solution = func;
  if (ctx) tsdm->solutionctx = ctx;
  PetscFunctionReturn(0);
}

/*@C
   DMTSSetForcingFunction - set `TS` forcing function evaluation function

   Not Collective

   Input Parameters:
+  dm - `DM` to be used with `TS`
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
   `TSSetForcingFunction()` is normally used, but it calls this function internally because the user context is actually
   associated with the `DM`.  This makes the interface consistent regardless of whether the user interacts with a `DM` or
   not. If `DM` took a more central role at some later date, this could become the primary method of setting the residual.

.seealso: [](chapter_ts), `DM`, `TS`, `DMTSSetContext()`, `TSSetFunction()`, `DMTSSetJacobian()`, `TSSetForcingFunction()`, `DMTSGetForcingFunction()`
@*/
PetscErrorCode DMTSSetForcingFunction(DM dm, TSForcingFunction f, void *ctx)
{
  DMTS tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTSWrite(dm, &tsdm));
  if (f) tsdm->ops->forcing = f;
  if (ctx) tsdm->forcingctx = ctx;
  PetscFunctionReturn(0);
}

/*@C
   DMTSGetForcingFunction - get `TS` forcing function evaluation function

   Not Collective

   Input Parameter:
.   dm - `DM` to be used with `TS`

   Output Parameters:
+  f - forcing function evaluation function; see `TSForcingFunction` for details
-  ctx - context for solution evaluation

   Level: advanced

   Note:
   `TSSetForcingFunction()` is normally used, but it calls this function internally because the user context is actually
   associated with the `DM`.  This makes the interface consistent regardless of whether the user interacts with a `DM` or
   not. If `DM` took a more central role at some later date, this could become the primary method of setting the residual.

.seealso: [](chapter_ts), `TS`, `DM`, `DMTSSetContext()`, `TSSetFunction()`, `DMTSSetJacobian()`, `TSSetForcingFunction()`, `DMTSGetForcingFunction()`
@*/
PetscErrorCode DMTSGetForcingFunction(DM dm, TSForcingFunction *f, void **ctx)
{
  DMTS tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTSWrite(dm, &tsdm));
  if (f) *f = tsdm->ops->forcing;
  if (ctx) *ctx = tsdm->forcingctx;
  PetscFunctionReturn(0);
}

/*@C
   DMTSGetRHSFunction - get `TS` explicit residual evaluation function

   Not Collective

   Input Parameter:
.  dm - `DM` to be used with `TS`

   Output Parameters:
+  func - residual evaluation function, see `TSSetRHSFunction()` for calling sequence
-  ctx - context for residual evaluation

   Level: advanced

   Note:
   `TSGetFunction()` is normally used, but it calls this function internally because the user context is actually
   associated with the DM.

.seealso: [](chapter_ts), `DM`, `TS`, `DMTSSetContext()`, `DMTSSetRHSFunction()`, `TSSetRHSFunction()`
@*/
PetscErrorCode DMTSGetRHSFunction(DM dm, TSRHSFunction *func, void **ctx)
{
  DMTS tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTS(dm, &tsdm));
  if (func) *func = tsdm->ops->rhsfunction;
  if (ctx) {
    if (tsdm->rhsfunctionctxcontainer) PetscCall(PetscContainerGetPointer(tsdm->rhsfunctionctxcontainer, ctx));
    else *ctx = NULL;
  }
  PetscFunctionReturn(0);
}

/*@C
   DMTSSetIJacobian - set `TS` Jacobian evaluation function

   Not Collective

   Input Parameters:
+  dm - `DM` to be used with `TS`
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
   `TSSetJacobian()` is normally used, but it calls this function internally because the user context is actually
   associated with the `DM`.  This makes the interface consistent regardless of whether the user interacts with a `DM` or
   not. If `DM` took a more central role at some later date, this could become the primary method of setting the Jacobian.

.seealso: [](chapter_ts), `TS`, `DM`, `DMTSSetContext()`, `TSSetRHSFunction()`, `DMTSGetJacobian()`, `TSSetIJacobian()`, `TSSetIFunction()`
@*/
PetscErrorCode DMTSSetIJacobian(DM dm, TSIJacobian func, void *ctx)
{
  DMTS tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTSWrite(dm, &tsdm));
  if (func) tsdm->ops->ijacobian = func;
  if (ctx) {
    PetscContainer ctxcontainer;
    PetscCall(PetscContainerCreate(PetscObjectComm((PetscObject)tsdm), &ctxcontainer));
    PetscCall(PetscContainerSetPointer(ctxcontainer, ctx));
    PetscCall(PetscObjectCompose((PetscObject)tsdm, "ijacobian ctx", (PetscObject)ctxcontainer));
    tsdm->ijacobianctxcontainer = ctxcontainer;
    PetscCall(PetscContainerDestroy(&ctxcontainer));
  }
  PetscFunctionReturn(0);
}

/*@C
   DMTSSetIJacobianContextDestroy - set `TS` Jacobian evaluation context destroy function

   Not Collective

   Input Parameters:
+  dm - `DM` to be used with `TS`
-  f - Jacobian evaluation context destroy function

   Level: advanced

   Note:
   `TSSetIJacobianContextDestroy()` is normally used, but it calls this function internally because the user context is actually
   associated with the `DM`.  This makes the interface consistent regardless of whether the user interacts with a `DM` or
   not.

   Developer Note:
   If `DM` took a more central role at some later date, this could become the primary method of setting the Jacobian.

.seealso: [](chapter_ts), `TSSetIJacobianContextDestroy()`, `TSSetI2JacobianContextDestroy()`, `DMTSSetIJacobian()`, `TSSetIJacobian()`
@*/
PetscErrorCode DMTSSetIJacobianContextDestroy(DM dm, PetscErrorCode (*f)(void *))
{
  DMTS tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTSWrite(dm, &tsdm));
  if (tsdm->ijacobianctxcontainer) PetscCall(PetscContainerSetUserDestroy(tsdm->ijacobianctxcontainer, f));
  PetscFunctionReturn(0);
}

PetscErrorCode DMTSUnsetIJacobianContext_Internal(DM dm)
{
  DMTS tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTSWrite(dm, &tsdm));
  PetscCall(DMTSUnsetIJacobianContext_DMTS(tsdm));
  PetscFunctionReturn(0);
}

/*@C
   DMTSGetIJacobian - get `TS` Jacobian evaluation function

   Not Collective

   Input Parameter:
.  dm - `DM` to be used with `TS`

   Output Parameters:
+  func - Jacobian evaluation function, see `TSSetIJacobian()` for calling sequence
-  ctx - context for residual evaluation

   Level: advanced

   Note:
   `TSGetJacobian()` is normally used, but it calls this function internally because the user context is actually
   associated with the `DM`.  This makes the interface consistent regardless of whether the user interacts with a `DM` or
   not. If `DM` took a more central role at some later date, this could become the primary method of setting the Jacobian.

.seealso: [](chapter_ts), `DM`, `TS`, `DMTSSetContext()`, `TSSetFunction()`, `DMTSSetJacobian()`
@*/
PetscErrorCode DMTSGetIJacobian(DM dm, TSIJacobian *func, void **ctx)
{
  DMTS tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTS(dm, &tsdm));
  if (func) *func = tsdm->ops->ijacobian;
  if (ctx) {
    if (tsdm->ijacobianctxcontainer) PetscCall(PetscContainerGetPointer(tsdm->ijacobianctxcontainer, ctx));
    else *ctx = NULL;
  }
  PetscFunctionReturn(0);
}

/*@C
   DMTSSetRHSJacobian - set `TS` Jacobian evaluation function

   Not Collective

   Input Parameters:
+  dm - `DM` to be used with `TS`
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
   `TSSetJacobian()` is normally used, but it calls this function internally because the user context is actually
   associated with the `DM`.  This makes the interface consistent regardless of whether the user interacts with a `DM` or
   not.

   Developer Note:
   If `DM` took a more central role at some later date, this could become the primary method of setting the Jacobian.

.seealso: [](chapter_ts), `DMTSSetContext()`, `TSSetFunction()`, `DMTSGetJacobian()`, `TSSetRHSJacobian()`
@*/
PetscErrorCode DMTSSetRHSJacobian(DM dm, TSRHSJacobian func, void *ctx)
{
  DMTS tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTSWrite(dm, &tsdm));
  if (func) tsdm->ops->rhsjacobian = func;
  if (ctx) {
    PetscContainer ctxcontainer;
    PetscCall(PetscContainerCreate(PetscObjectComm((PetscObject)tsdm), &ctxcontainer));
    PetscCall(PetscContainerSetPointer(ctxcontainer, ctx));
    PetscCall(PetscObjectCompose((PetscObject)tsdm, "rhs jacobian ctx", (PetscObject)ctxcontainer));
    tsdm->rhsjacobianctxcontainer = ctxcontainer;
    PetscCall(PetscContainerDestroy(&ctxcontainer));
  }
  PetscFunctionReturn(0);
}

/*@C
   DMTSSetRHSJacobianContextDestroy - set `TS` Jacobian evaluation context destroy function

   Not Collective

   Input Parameters:
+  dm - `DM` to be used with `TS`
-  f - Jacobian evaluation context destroy function

   Level: advanced

   Note:
   The user usually calls `TSSetRHSJacobianContextDestroy()` which calls this routine

.seealso: [](chapter_ts), `TS`, `TSSetRHSJacobianContextDestroy()`, `DMTSSetRHSJacobian()`, `TSSetRHSJacobian()`
@*/
PetscErrorCode DMTSSetRHSJacobianContextDestroy(DM dm, PetscErrorCode (*f)(void *))
{
  DMTS tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTSWrite(dm, &tsdm));
  if (tsdm->rhsjacobianctxcontainer) PetscCall(PetscContainerSetUserDestroy(tsdm->rhsjacobianctxcontainer, f));
  PetscFunctionReturn(0);
}

PetscErrorCode DMTSUnsetRHSJacobianContext_Internal(DM dm)
{
  DMTS tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTSWrite(dm, &tsdm));
  PetscCall(DMTSUnsetRHSJacobianContext_DMTS(tsdm));
  PetscFunctionReturn(0);
}

/*@C
   DMTSGetRHSJacobian - get `TS` Jacobian evaluation function

   Not Collective

   Input Parameter:
.  dm - `DM` to be used with `TS`

   Output Parameters:
+  func - Jacobian evaluation function, see `TSSetRHSJacobian()` for calling sequence
-  ctx - context for residual evaluation

   Level: advanced

   Note:
   `TSGetJacobian()` is normally used, but it calls this function internally because the user context is actually
   associated with the `DM`.  This makes the interface consistent regardless of whether the user interacts with a `DM` or
   not. If `DM` took a more central role at some later date, this could become the primary method of setting the Jacobian.

.seealso: [](chapter_ts), `DM`, `TS`, `DMTSSetContext()`, `TSSetRHSFunction()`, `DMTSSetRHSJacobian()`, `TSSetRHSJacobian()`
@*/
PetscErrorCode DMTSGetRHSJacobian(DM dm, TSRHSJacobian *func, void **ctx)
{
  DMTS tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTS(dm, &tsdm));
  if (func) *func = tsdm->ops->rhsjacobian;
  if (ctx) {
    if (tsdm->rhsjacobianctxcontainer) PetscCall(PetscContainerGetPointer(tsdm->rhsjacobianctxcontainer, ctx));
    else *ctx = NULL;
  }
  PetscFunctionReturn(0);
}

/*@C
   DMTSSetIFunctionSerialize - sets functions used to view and load a IFunction context

   Not Collective

   Input Parameters:
+  dm - `DM` to be used with `TS`
.  view - viewer function
-  load - loading function

   Level: advanced

.seealso: [](chapter_ts), `DM`, `TS`, `DMTSSetContext()`, `TSSetFunction()`, `DMTSSetJacobian()`
@*/
PetscErrorCode DMTSSetIFunctionSerialize(DM dm, PetscErrorCode (*view)(void *, PetscViewer), PetscErrorCode (*load)(void **, PetscViewer))
{
  DMTS tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTSWrite(dm, &tsdm));
  tsdm->ops->ifunctionview = view;
  tsdm->ops->ifunctionload = load;
  PetscFunctionReturn(0);
}

/*@C
   DMTSSetIJacobianSerialize - sets functions used to view and load a IJacobian context

   Not Collective

   Input Parameters:
+  dm - `DM` to be used with `TS`
.  view - viewer function
-  load - loading function

   Level: advanced

.seealso: [](chapter_ts), `DM`, `TS`, `DMTSSetContext()`, `TSSetFunction()`, `DMTSSetJacobian()`
@*/
PetscErrorCode DMTSSetIJacobianSerialize(DM dm, PetscErrorCode (*view)(void *, PetscViewer), PetscErrorCode (*load)(void **, PetscViewer))
{
  DMTS tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTSWrite(dm, &tsdm));
  tsdm->ops->ijacobianview = view;
  tsdm->ops->ijacobianload = load;
  PetscFunctionReturn(0);
}
