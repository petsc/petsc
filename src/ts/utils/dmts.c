#include <petsc/private/tsimpl.h> /*I "petscts.h" I*/
#include <petsc/private/dmimpl.h>

static PetscErrorCode DMTSUnsetRHSFunctionContext_DMTS(DMTS tsdm)
{
  PetscFunctionBegin;
  PetscCall(PetscObjectCompose((PetscObject)tsdm, "rhs function ctx", NULL));
  tsdm->rhsfunctionctxcontainer = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMTSUnsetRHSJacobianContext_DMTS(DMTS tsdm)
{
  PetscFunctionBegin;
  PetscCall(PetscObjectCompose((PetscObject)tsdm, "rhs jacobian ctx", NULL));
  tsdm->rhsjacobianctxcontainer = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMTSUnsetIFunctionContext_DMTS(DMTS tsdm)
{
  PetscFunctionBegin;
  PetscCall(PetscObjectCompose((PetscObject)tsdm, "ifunction ctx", NULL));
  tsdm->ifunctionctxcontainer = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMTSUnsetIJacobianContext_DMTS(DMTS tsdm)
{
  PetscFunctionBegin;
  PetscCall(PetscObjectCompose((PetscObject)tsdm, "ijacobian ctx", NULL));
  tsdm->ijacobianctxcontainer = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMTSUnsetI2FunctionContext_DMTS(DMTS tsdm)
{
  PetscFunctionBegin;
  PetscCall(PetscObjectCompose((PetscObject)tsdm, "i2function ctx", NULL));
  tsdm->i2functionctxcontainer = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMTSUnsetI2JacobianContext_DMTS(DMTS tsdm)
{
  PetscFunctionBegin;
  PetscCall(PetscObjectCompose((PetscObject)tsdm, "i2jacobian ctx", NULL));
  tsdm->i2jacobianctxcontainer = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMTSDestroy(DMTS *kdm)
{
  PetscFunctionBegin;
  if (!*kdm) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific(*kdm, DMTS_CLASSID, 1);
  if (--((PetscObject)*kdm)->refct > 0) {
    *kdm = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(DMTSUnsetRHSFunctionContext_DMTS(*kdm));
  PetscCall(DMTSUnsetRHSJacobianContext_DMTS(*kdm));
  PetscCall(DMTSUnsetIFunctionContext_DMTS(*kdm));
  PetscCall(DMTSUnsetIJacobianContext_DMTS(*kdm));
  PetscCall(DMTSUnsetI2FunctionContext_DMTS(*kdm));
  PetscCall(DMTSUnsetI2JacobianContext_DMTS(*kdm));
  PetscTryTypeMethod(*kdm, destroy);
  PetscCall(PetscHeaderDestroy(kdm));
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscFunctionReturn(PETSC_SUCCESS);
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
      TSIFunctionFn *ifunction;
    } funcstruct;
    struct {
      PetscErrorCode (*ifunctionview)(void *, PetscViewer);
    } funcviewstruct;
    struct {
      PetscErrorCode (*ifunctionload)(void **, PetscViewer);
    } funcloadstruct;
    struct {
      TSIJacobianFn *ijacobian;
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
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMTSCreate(MPI_Comm comm, DMTS *kdm)
{
  PetscFunctionBegin;
  PetscCall(TSInitializePackage());
  PetscCall(PetscHeaderCreate(*kdm, DMTS_CLASSID, "DMTS", "DMTS", "DMTS", comm, DMTSDestroy, DMTSView));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Attaches the DMTS to the coarse level.
 * Under what conditions should we copy versus duplicate?
 */
static PetscErrorCode DMCoarsenHook_DMTS(DM dm, DM dmc, void *ctx)
{
  PetscFunctionBegin;
  PetscCall(DMCopyDMTS(dm, dmc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* This could restrict auxiliary information to the coarse level.
 */
static PetscErrorCode DMRestrictHook_DMTS(DM dm, Mat Restrict, Vec rscale, Mat Inject, DM dmc, void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMSubDomainHook_DMTS(DM dm, DM subdm, void *ctx)
{
  PetscFunctionBegin;
  PetscCall(DMCopyDMTS(dm, subdm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* This could restrict auxiliary information to the coarse level.
 */
static PetscErrorCode DMSubDomainRestrictHook_DMTS(DM dm, VecScatter gscat, VecScatter lscat, DM subdm, void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMTSCopy - copies the information in a `DMTS` to another `DMTS`

  Not Collective

  Input Parameters:
+ kdm  - Original `DMTS`
- nkdm - `DMTS` to receive the data, should have been created with `DMTSCreate()`

  Level: developer

.seealso: [](ch_ts), `DMTSCreate()`, `DMTSDestroy()`
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
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMGetDMTS - get read-only private `DMTS` context from a `DM`

  Not Collective

  Input Parameter:
. dm - `DM` to be used with `TS`

  Output Parameter:
. tsdm - private `DMTS` context

  Level: developer

  Notes:
  Use `DMGetDMTSWrite()` if write access is needed. The `DMTSSetXXX()` API should be used wherever possible.

.seealso: [](ch_ts), `DMTS`, `DMGetDMTSWrite()`
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
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMGetDMTSWrite - get write access to private `DMTS` context from a `DM`

  Not Collective

  Input Parameter:
. dm - `DM` to be used with `TS`

  Output Parameter:
. tsdm - private `DMTS` context

  Level: developer

.seealso: [](ch_ts), `DMTS`, `DMGetDMTS()`
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
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMCopyDMTS - copies a `DMTS` context to a new `DM`

  Logically Collective

  Input Parameters:
+ dmsrc  - `DM` to obtain context from
- dmdest - `DM` to add context to

  Level: developer

  Note:
  The context is copied by reference. This function does not ensure that a context exists.

.seealso: [](ch_ts), `DMTS`, `DMGetDMTS()`, `TSSetDM()`
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
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMTSSetIFunction - set `TS` implicit function evaluation function into a `DMTS`

  Not Collective

  Input Parameters:
+ dm   - `DM` to be used with `TS`
. func - function evaluating f(t,u,u_t)
- ctx  - context for residual evaluation

  Level: developer

  Note:
  `TSSetIFunction()` is normally used, but it calls this function internally because the user context is actually
  associated with the `DM`.  This makes the interface consistent regardless of whether the user interacts with a `DM` or
  not. If `DM` took a more central role at some later date, this could become the primary method of setting the residual.

.seealso: [](ch_ts), `DMTS`, `TS`, `DM`, `TSIFunctionFn`
@*/
PetscErrorCode DMTSSetIFunction(DM dm, TSIFunctionFn *func, void *ctx)
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
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMTSSetIFunctionContextDestroy - set `TS` implicit evaluation context destroy function into a `DMTS`

  Not Collective

  Input Parameters:
+ dm - `DM` to be used with `TS`
- f  - implicit evaluation context destroy function, see `PetscCtxDestroyFn` for its calling sequence

  Level: developer

.seealso: [](ch_ts), `DMTS`, `DM`, `TS`, `DMTSSetIFunction()`, `TSSetIFunction()`, `PetscCtxDestroyFn`
@*/
PetscErrorCode DMTSSetIFunctionContextDestroy(DM dm, PetscCtxDestroyFn *f)
{
  DMTS tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTSWrite(dm, &tsdm));
  if (tsdm->ifunctionctxcontainer) PetscCall(PetscContainerSetCtxDestroy(tsdm->ifunctionctxcontainer, f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMTSUnsetIFunctionContext_Internal(DM dm)
{
  DMTS tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTSWrite(dm, &tsdm));
  PetscCall(DMTSUnsetIFunctionContext_DMTS(tsdm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMTSGetIFunction - get `TS` implicit residual evaluation function from a `DMTS`

  Not Collective

  Input Parameter:
. dm - `DM` to be used with `TS`

  Output Parameters:
+ func - function evaluation function, for calling sequence see `TSIFunctionFn`
- ctx  - context for residual evaluation

  Level: developer

  Note:
  `TSGetIFunction()` is normally used, but it calls this function internally because the user context is actually
  associated with the `DM`.

.seealso: [](ch_ts), `DMTS`, `TS`, `DM`, `DMTSSetIFunction()`, `TSIFunctionFn`
@*/
PetscErrorCode DMTSGetIFunction(DM dm, TSIFunctionFn **func, void **ctx)
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
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMTSSetI2Function - set `TS` implicit function evaluation function for 2nd order systems into a `TSDM`

  Not Collective

  Input Parameters:
+ dm  - `DM` to be used with `TS`
. fun - function evaluation routine
- ctx - context for residual evaluation

  Level: developer

  Note:
  `TSSetI2Function()` is normally used, but it calls this function internally because the user context is actually
  associated with the `DM`.

.seealso: [](ch_ts), `DMTS`, `DM`, `TS`, `TSSetI2Function()`
@*/
PetscErrorCode DMTSSetI2Function(DM dm, TSI2FunctionFn *fun, void *ctx)
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
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMTSSetI2FunctionContextDestroy - set `TS` implicit evaluation for 2nd order systems context destroy into a `DMTS`

  Not Collective

  Input Parameters:
+ dm - `DM` to be used with `TS`
- f  - implicit evaluation context destroy function, see `PetscCtxDestroyFn` for its calling sequence

  Level: developer

  Note:
  `TSSetI2FunctionContextDestroy()` is normally used, but it calls this function internally because the user context is actually
  associated with the `DM`.

.seealso: [](ch_ts), `DMTS`, `TSSetI2FunctionContextDestroy()`, `DMTSSetI2Function()`, `TSSetI2Function()`
@*/
PetscErrorCode DMTSSetI2FunctionContextDestroy(DM dm, PetscCtxDestroyFn *f)
{
  DMTS tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTSWrite(dm, &tsdm));
  if (tsdm->i2functionctxcontainer) PetscCall(PetscContainerSetCtxDestroy(tsdm->i2functionctxcontainer, f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMTSUnsetI2FunctionContext_Internal(DM dm)
{
  DMTS tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTSWrite(dm, &tsdm));
  PetscCall(DMTSUnsetI2FunctionContext_DMTS(tsdm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMTSGetI2Function - get `TS` implicit residual evaluation function for 2nd order systems from a `DMTS`

  Not Collective

  Input Parameter:
. dm - `DM` to be used with `TS`

  Output Parameters:
+ fun - function evaluation function, for calling sequence see `TSSetI2Function()`
- ctx - context for residual evaluation

  Level: developer

  Note:
  `TSGetI2Function()` is normally used, but it calls this function internally because the user context is actually
  associated with the `DM`.

.seealso: [](ch_ts), `DMTS`, `DM`, `TS`, `DMTSSetI2Function()`, `TSGetI2Function()`
@*/
PetscErrorCode DMTSGetI2Function(DM dm, TSI2FunctionFn **fun, void **ctx)
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
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMTSSetI2Jacobian - set `TS` implicit Jacobian evaluation function for 2nd order systems from a `DMTS`

  Not Collective

  Input Parameters:
+ dm  - `DM` to be used with `TS`
. jac - Jacobian evaluation routine
- ctx - context for Jacobian evaluation

  Level: developer

  Note:
  `TSSetI2Jacobian()` is normally used, but it calls this function internally because the user context is actually
  associated with the `DM`.

.seealso: [](ch_ts), `DMTS`, `DM`, `TS`, `TSI2JacobianFn`, `TSSetI2Jacobian()`
@*/
PetscErrorCode DMTSSetI2Jacobian(DM dm, TSI2JacobianFn *jac, void *ctx)
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
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMTSSetI2JacobianContextDestroy - set `TS` implicit Jacobian evaluation for 2nd order systems context destroy function into a `DMTS`

  Not Collective

  Input Parameters:
+ dm - `DM` to be used with `TS`
- f  - implicit Jacobian evaluation context destroy function, see `PetscCtxDestroyFn` for its calling sequence

  Level: developer

  Note:
  Normally `TSSetI2JacobianContextDestroy()` is used

.seealso: [](ch_ts), `DMTS`, `DM`, `TS`, `TSSetI2JacobianContextDestroy()`, `DMTSSetI2Jacobian()`, `TSSetI2Jacobian()`
@*/
PetscErrorCode DMTSSetI2JacobianContextDestroy(DM dm, PetscCtxDestroyFn *f)
{
  DMTS tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTSWrite(dm, &tsdm));
  if (tsdm->i2jacobianctxcontainer) PetscCall(PetscContainerSetCtxDestroy(tsdm->i2jacobianctxcontainer, f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMTSUnsetI2JacobianContext_Internal(DM dm)
{
  DMTS tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTSWrite(dm, &tsdm));
  PetscCall(DMTSUnsetI2JacobianContext_DMTS(tsdm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMTSGetI2Jacobian - get `TS` implicit Jacobian evaluation function for 2nd order systems from a `DMTS`

  Not Collective

  Input Parameter:
. dm - `DM` to be used with `TS`

  Output Parameters:
+ jac - Jacobian evaluation function,  for calling sequence see `TSI2JacobianFn`
- ctx - context for Jacobian evaluation

  Level: developer

  Note:
  `TSGetI2Jacobian()` is normally used, but it calls this function internally because the user context is actually
  associated with the `DM`.

.seealso: [](ch_ts), `DMTS`, `DM`, `TS`, `DMTSSetI2Jacobian()`, `TSGetI2Jacobian()`, `TSI2JacobianFn`
@*/
PetscErrorCode DMTSGetI2Jacobian(DM dm, TSI2JacobianFn **jac, void **ctx)
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
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMTSSetRHSFunction - set `TS` explicit residual evaluation function into a `DMTS`

  Not Collective

  Input Parameters:
+ dm   - `DM` to be used with `TS`
. func - RHS function evaluation routine, see `TSRHSFunctionFn` for the calling sequence
- ctx  - context for residual evaluation

  Level: developer

  Note:
  `TSSetRHSFunction()` is normally used, but it calls this function internally because the user context is actually
  associated with the `DM`.  This makes the interface consistent regardless of whether the user interacts with a `DM` or
  not. If `DM` took a more central role at some later date, this could become the primary method of setting the residual.

.seealso: [](ch_ts), `DMTS`, `DM`, `TS`, `TSRHSFunctionFn`
@*/
PetscErrorCode DMTSSetRHSFunction(DM dm, TSRHSFunctionFn *func, void *ctx)
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
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMTSSetRHSFunctionContextDestroy - set `TS` explicit residual evaluation context destroy function into a `DMTS`

  Not Collective

  Input Parameters:
+ dm - `DM` to be used with `TS`
- f  - explicit evaluation context destroy function, see `PetscCtxDestroyFn` for its calling sequence

  Level: developer

  Note:
  `TSSetRHSFunctionContextDestroy()` is normally used, but it calls this function internally because the user context is actually
  associated with the `DM`.  This makes the interface consistent regardless of whether the user interacts with a `DM` or
  not.

  Developer Notes:
  If `DM` took a more central role at some later date, this could become the primary method of setting the residual.

.seealso: [](ch_ts), `DMTS`, `TSSetRHSFunctionContextDestroy()`, `DMTSSetRHSFunction()`, `TSSetRHSFunction()`
@*/
PetscErrorCode DMTSSetRHSFunctionContextDestroy(DM dm, PetscCtxDestroyFn *f)
{
  DMTS tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTSWrite(dm, &tsdm));
  if (tsdm->rhsfunctionctxcontainer) PetscCall(PetscContainerSetCtxDestroy(tsdm->rhsfunctionctxcontainer, f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMTSUnsetRHSFunctionContext_Internal(DM dm)
{
  DMTS tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTSWrite(dm, &tsdm));
  PetscCall(DMTSUnsetRHSFunctionContext_DMTS(tsdm));
  tsdm->rhsfunctionctxcontainer = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMTSSetTransientVariable - sets function to transform from state to transient variables into a `DMTS`

  Logically Collective

  Input Parameters:
+ dm   - `DM` to be used with `TS`
. tvar - a function that transforms to transient variables, see `TSTransientVariableFn` for the calling sequence
- ctx  - a context for tvar

  Level: developer

  Notes:
  Normally `TSSetTransientVariable()` is used

  This is typically used to transform from primitive to conservative variables so that a time integrator (e.g., `TSBDF`)
  can be conservative.  In this context, primitive variables P are used to model the state (e.g., because they lead to
  well-conditioned formulations even in limiting cases such as low-Mach or zero porosity).  The transient variable is
  C(P), specified by calling this function.  An IFunction thus receives arguments (P, Cdot) and the IJacobian must be
  evaluated via the chain rule, as in

  $$
  dF/dP + shift * dF/dCdot dC/dP.
  $$

.seealso: [](ch_ts), `DMTS`, `TS`, `TSBDF`, `TSSetTransientVariable()`, `DMTSGetTransientVariable()`, `DMTSSetIFunction()`, `DMTSSetIJacobian()`, `TSTransientVariableFn`
@*/
PetscErrorCode DMTSSetTransientVariable(DM dm, TSTransientVariableFn *tvar, void *ctx)
{
  DMTS dmts;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTSWrite(dm, &dmts));
  dmts->ops->transientvar = tvar;
  dmts->transientvarctx   = ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMTSGetTransientVariable - gets function to transform from state to transient variables set with `DMTSSetTransientVariable()` from a `TSDM`

  Logically Collective

  Input Parameter:
. dm - `DM` to be used with `TS`

  Output Parameters:
+ tvar - a function that transforms to transient variables, see `TSTransientVariableFn` for the calling sequence
- ctx  - a context for tvar

  Level: developer

  Note:
  Normally `TSSetTransientVariable()` is used

.seealso: [](ch_ts), `DMTS`, `DM`, `DMTSSetTransientVariable()`, `DMTSGetIFunction()`, `DMTSGetIJacobian()`, `TSTransientVariableFn`
@*/
PetscErrorCode DMTSGetTransientVariable(DM dm, TSTransientVariableFn **tvar, void *ctx)
{
  DMTS dmts;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTS(dm, &dmts));
  if (tvar) *tvar = dmts->ops->transientvar;
  if (ctx) *(void **)ctx = dmts->transientvarctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMTSGetSolutionFunction - gets the `TS` solution evaluation function from a `DMTS`

  Not Collective

  Input Parameter:
. dm - `DM` to be used with `TS`

  Output Parameters:
+ func - solution function evaluation function, for calling sequence see `TSSolutionFn`
- ctx  - context for solution evaluation

  Level: developer

.seealso: [](ch_ts), `DMTS`, `TS`, `DM`, `DMTSSetSolutionFunction()`, `TSSolutionFn`
@*/
PetscErrorCode DMTSGetSolutionFunction(DM dm, TSSolutionFn **func, void **ctx)
{
  DMTS tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTS(dm, &tsdm));
  if (func) *func = tsdm->ops->solution;
  if (ctx) *ctx = tsdm->solutionctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMTSSetSolutionFunction - set `TS` solution evaluation function into a `DMTS`

  Not Collective

  Input Parameters:
+ dm   - `DM` to be used with `TS`
. func - solution function evaluation routine, for calling sequence see `TSSolutionFn`
- ctx  - context for solution evaluation

  Level: developer

  Note:
  `TSSetSolutionFunction()` is normally used, but it calls this function internally because the user context is actually
  associated with the `DM`.  This makes the interface consistent regardless of whether the user interacts with a `DM` or
  not. If `DM` took a more central role at some later date, this could become the primary method of setting the residual.

.seealso: [](ch_ts), `DMTS`, `DM`, `TS`, `DMTSGetSolutionFunction()`, `TSSolutionFn`
@*/
PetscErrorCode DMTSSetSolutionFunction(DM dm, TSSolutionFn *func, void *ctx)
{
  DMTS tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTSWrite(dm, &tsdm));
  if (func) tsdm->ops->solution = func;
  if (ctx) tsdm->solutionctx = ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMTSSetForcingFunction - set `TS` forcing function evaluation function into a `DMTS`

  Not Collective

  Input Parameters:
+ dm   - `DM` to be used with `TS`
. func - forcing function evaluation routine, for calling sequence see `TSForcingFn`
- ctx  - context for solution evaluation

  Level: developer

  Note:
  `TSSetForcingFunction()` is normally used, but it calls this function internally because the user context is actually
  associated with the `DM`.  This makes the interface consistent regardless of whether the user interacts with a `DM` or
  not. If `DM` took a more central role at some later date, this could become the primary method of setting the residual.

.seealso: [](ch_ts), `DMTS`, `DM`, `TS`, `TSForcingFn`, `TSSetForcingFunction()`, `DMTSGetForcingFunction()`
@*/
PetscErrorCode DMTSSetForcingFunction(DM dm, TSForcingFn *func, void *ctx)
{
  DMTS tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTSWrite(dm, &tsdm));
  if (func) tsdm->ops->forcing = func;
  if (ctx) tsdm->forcingctx = ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMTSGetForcingFunction - get `TS` forcing function evaluation function from a `DMTS`

  Not Collective

  Input Parameter:
. dm - `DM` to be used with `TS`

  Output Parameters:
+ f   - forcing function evaluation function; see `TSForcingFn` for the calling sequence
- ctx - context for solution evaluation

  Level: developer

  Note:
  `TSSetForcingFunction()` is normally used, but it calls this function internally because the user context is actually
  associated with the `DM`.  This makes the interface consistent regardless of whether the user interacts with a `DM` or
  not. If `DM` took a more central role at some later date, this could become the primary method of setting the residual.

.seealso: [](ch_ts), `DMTS`, `TS`, `DM`, `TSSetForcingFunction()`, `TSForcingFn`
@*/
PetscErrorCode DMTSGetForcingFunction(DM dm, TSForcingFn **f, void **ctx)
{
  DMTS tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTSWrite(dm, &tsdm));
  if (f) *f = tsdm->ops->forcing;
  if (ctx) *ctx = tsdm->forcingctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMTSGetRHSFunction - get `TS` explicit residual evaluation function from a `DMTS`

  Not Collective

  Input Parameter:
. dm - `DM` to be used with `TS`

  Output Parameters:
+ func - residual evaluation function, for calling sequence see `TSRHSFunctionFn`
- ctx  - context for residual evaluation

  Level: developer

  Note:
  `TSGetRHSFunction()` is normally used, but it calls this function internally because the user context is actually
  associated with the DM.

.seealso: [](ch_ts), `DMTS`, `DM`, `TS`, `TSRHSFunctionFn`, `TSGetRHSFunction()`
@*/
PetscErrorCode DMTSGetRHSFunction(DM dm, TSRHSFunctionFn **func, void **ctx)
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
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMTSSetIJacobian - set `TS` Jacobian evaluation function into a `DMTS`

  Not Collective

  Input Parameters:
+ dm   - `DM` to be used with `TS`
. func - Jacobian evaluation routine, see `TSIJacobianFn` for the calling sequence
- ctx  - context for residual evaluation

  Level: developer

  Note:
  `TSSetIJacobian()` is normally used, but it calls this function internally because the user context is actually
  associated with the `DM`.  This makes the interface consistent regardless of whether the user interacts with a `DM` or
  not. If `DM` took a more central role at some later date, this could become the primary method of setting the Jacobian.

.seealso: [](ch_ts), `DMTS`, `TS`, `DM`, `TSIJacobianFn`, `DMTSGetIJacobian()`, `TSSetIJacobian()`
@*/
PetscErrorCode DMTSSetIJacobian(DM dm, TSIJacobianFn *func, void *ctx)
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
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMTSSetIJacobianContextDestroy - set `TS` Jacobian evaluation context destroy function into a `DMTS`

  Not Collective

  Input Parameters:
+ dm - `DM` to be used with `TS`
- f  - Jacobian evaluation context destroy function, see `PetscCtxDestroyFn` for its calling sequence

  Level: developer

  Note:
  `TSSetIJacobianContextDestroy()` is normally used, but it calls this function internally because the user context is actually
  associated with the `DM`.  This makes the interface consistent regardless of whether the user interacts with a `DM` or
  not.

  Developer Notes:
  If `DM` took a more central role at some later date, this could become the primary method of setting the Jacobian.

.seealso: [](ch_ts), `DMTS`, `TSSetIJacobianContextDestroy()`, `TSSetI2JacobianContextDestroy()`, `DMTSSetIJacobian()`, `TSSetIJacobian()`
@*/
PetscErrorCode DMTSSetIJacobianContextDestroy(DM dm, PetscCtxDestroyFn *f)
{
  DMTS tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTSWrite(dm, &tsdm));
  if (tsdm->ijacobianctxcontainer) PetscCall(PetscContainerSetCtxDestroy(tsdm->ijacobianctxcontainer, f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMTSUnsetIJacobianContext_Internal(DM dm)
{
  DMTS tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTSWrite(dm, &tsdm));
  PetscCall(DMTSUnsetIJacobianContext_DMTS(tsdm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMTSGetIJacobian - get `TS` Jacobian evaluation function from a `DMTS`

  Not Collective

  Input Parameter:
. dm - `DM` to be used with `TS`

  Output Parameters:
+ func - Jacobian evaluation function, for calling sequence see `TSIJacobianFn`
- ctx  - context for residual evaluation

  Level: developer

  Note:
  `TSGetIJacobian()` is normally used, but it calls this function internally because the user context is actually
  associated with the `DM`.  This makes the interface consistent regardless of whether the user interacts with a `DM` or
  not. If `DM` took a more central role at some later date, this could become the primary method of setting the Jacobian.

.seealso: [](ch_ts), `DMTS`, `DM`, `TS`, `DMTSSetIJacobian()`, `TSIJacobianFn`
@*/
PetscErrorCode DMTSGetIJacobian(DM dm, TSIJacobianFn **func, void **ctx)
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
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMTSSetRHSJacobian - set `TS` Jacobian evaluation function into a `DMTS`

  Not Collective

  Input Parameters:
+ dm   - `DM` to be used with `TS`
. func - Jacobian evaluation routine, for calling sequence see `TSIJacobianFn`
- ctx  - context for residual evaluation

  Level: developer

  Note:
  `TSSetRHSJacobian()` is normally used, but it calls this function internally because the user context is actually
  associated with the `DM`.  This makes the interface consistent regardless of whether the user interacts with a `DM` or
  not.

  Developer Notes:
  If `DM` took a more central role at some later date, this could become the primary method of setting the Jacobian.

.seealso: [](ch_ts), `DMTS`, `TSRHSJacobianFn`, `DMTSGetRHSJacobian()`, `TSSetRHSJacobian()`
@*/
PetscErrorCode DMTSSetRHSJacobian(DM dm, TSRHSJacobianFn *func, void *ctx)
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
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMTSSetRHSJacobianContextDestroy - set `TS` Jacobian evaluation context destroy function from a `DMTS`

  Not Collective

  Input Parameters:
+ dm - `DM` to be used with `TS`
- f  - Jacobian evaluation context destroy function, see `PetscCtxDestroyFn` for its calling sequence

  Level: developer

  Note:
  The user usually calls `TSSetRHSJacobianContextDestroy()` which calls this routine

.seealso: [](ch_ts), `DMTS`, `TS`, `TSSetRHSJacobianContextDestroy()`, `DMTSSetRHSJacobian()`, `TSSetRHSJacobian()`
@*/
PetscErrorCode DMTSSetRHSJacobianContextDestroy(DM dm, PetscCtxDestroyFn *f)
{
  DMTS tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTSWrite(dm, &tsdm));
  if (tsdm->rhsjacobianctxcontainer) PetscCall(PetscContainerSetCtxDestroy(tsdm->rhsjacobianctxcontainer, f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMTSUnsetRHSJacobianContext_Internal(DM dm)
{
  DMTS tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTSWrite(dm, &tsdm));
  PetscCall(DMTSUnsetRHSJacobianContext_DMTS(tsdm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMTSGetRHSJacobian - get `TS` Jacobian evaluation function from a `DMTS`

  Not Collective

  Input Parameter:
. dm - `DM` to be used with `TS`

  Output Parameters:
+ func - Jacobian evaluation function, for calling sequence see `TSRHSJacobianFn`
- ctx  - context for residual evaluation

  Level: developer

  Note:
  `TSGetRHSJacobian()` is normally used, but it calls this function internally because the user context is actually
  associated with the `DM`.  This makes the interface consistent regardless of whether the user interacts with a `DM` or
  not. If `DM` took a more central role at some later date, this could become the primary method of setting the Jacobian.

.seealso: [](ch_ts), `DMTS`, `DM`, `TS`, `DMTSSetRHSJacobian()`, `TSRHSJacobianFn`
@*/
PetscErrorCode DMTSGetRHSJacobian(DM dm, TSRHSJacobianFn **func, void **ctx)
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
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMTSSetIFunctionSerialize - sets functions used to view and load a `TSIFunctionFn` context

  Not Collective

  Input Parameters:
+ dm   - `DM` to be used with `TS`
. view - viewer function
- load - loading function

  Level: developer

.seealso: [](ch_ts), `DMTS`, `DM`, `TS`
@*/
PetscErrorCode DMTSSetIFunctionSerialize(DM dm, PetscErrorCode (*view)(void *, PetscViewer), PetscErrorCode (*load)(void **, PetscViewer))
{
  DMTS tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTSWrite(dm, &tsdm));
  tsdm->ops->ifunctionview = view;
  tsdm->ops->ifunctionload = load;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMTSSetIJacobianSerialize - sets functions used to view and load a `TSIJacobianFn` context

  Not Collective

  Input Parameters:
+ dm   - `DM` to be used with `TS`
. view - viewer function
- load - loading function

  Level: developer

.seealso: [](ch_ts), `DMTS`, `DM`, `TS`
@*/
PetscErrorCode DMTSSetIJacobianSerialize(DM dm, PetscErrorCode (*view)(void *, PetscViewer), PetscErrorCode (*load)(void **, PetscViewer))
{
  DMTS tsdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTSWrite(dm, &tsdm));
  tsdm->ops->ijacobianview = view;
  tsdm->ops->ijacobianload = load;
  PetscFunctionReturn(PETSC_SUCCESS);
}
