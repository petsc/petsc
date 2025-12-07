#include <petsc/private/tsimpl.h> /*I  "petscts.h" I*/

PetscClassId TSADAPT_CLASSID;

static PetscFunctionList TSAdaptList;
static PetscBool         TSAdaptPackageInitialized;
static PetscBool         TSAdaptRegisterAllCalled;

PETSC_EXTERN PetscErrorCode TSAdaptCreate_None(TSAdapt);
PETSC_EXTERN PetscErrorCode TSAdaptCreate_Basic(TSAdapt);
PETSC_EXTERN PetscErrorCode TSAdaptCreate_DSP(TSAdapt);
PETSC_EXTERN PetscErrorCode TSAdaptCreate_CFL(TSAdapt);
PETSC_EXTERN PetscErrorCode TSAdaptCreate_GLEE(TSAdapt);
PETSC_EXTERN PetscErrorCode TSAdaptCreate_History(TSAdapt);

/*@C
  TSAdaptRegister -  adds a TSAdapt implementation

  Not Collective, No Fortran Support

  Input Parameters:
+ sname    - name of user-defined adaptivity scheme
- function - routine to create method context

  Level: advanced

  Notes:
  `TSAdaptRegister()` may be called multiple times to add several user-defined families.

  Example Usage:
.vb
   TSAdaptRegister("my_scheme", MySchemeCreate);
.ve

  Then, your scheme can be chosen with the procedural interface via
.vb
  TSAdaptSetType(ts, "my_scheme")
.ve
  or at runtime via the option
.vb
  -ts_adapt_type my_scheme
.ve

.seealso: [](ch_ts), [](sec_ts_error_control), `TSAdaptRegisterAll()`
@*/
PetscErrorCode TSAdaptRegister(const char sname[], PetscErrorCode (*function)(TSAdapt))
{
  PetscFunctionBegin;
  PetscCall(TSAdaptInitializePackage());
  PetscCall(PetscFunctionListAdd(&TSAdaptList, sname, function));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TSAdaptRegisterAll - Registers all of the adaptivity schemes in `TSAdapt`

  Not Collective

  Level: advanced

.seealso: [](ch_ts), `TSAdaptRegisterDestroy()`
@*/
PetscErrorCode TSAdaptRegisterAll(void)
{
  PetscFunctionBegin;
  if (TSAdaptRegisterAllCalled) PetscFunctionReturn(PETSC_SUCCESS);
  TSAdaptRegisterAllCalled = PETSC_TRUE;
  PetscCall(TSAdaptRegister(TSADAPTNONE, TSAdaptCreate_None));
  PetscCall(TSAdaptRegister(TSADAPTBASIC, TSAdaptCreate_Basic));
  PetscCall(TSAdaptRegister(TSADAPTDSP, TSAdaptCreate_DSP));
  PetscCall(TSAdaptRegister(TSADAPTCFL, TSAdaptCreate_CFL));
  PetscCall(TSAdaptRegister(TSADAPTGLEE, TSAdaptCreate_GLEE));
  PetscCall(TSAdaptRegister(TSADAPTHISTORY, TSAdaptCreate_History));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TSAdaptFinalizePackage - This function destroys everything in the `TS` package. It is
  called from `PetscFinalize()`.

  Level: developer

.seealso: [](ch_ts), `PetscFinalize()`
@*/
PetscErrorCode TSAdaptFinalizePackage(void)
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListDestroy(&TSAdaptList));
  TSAdaptPackageInitialized = PETSC_FALSE;
  TSAdaptRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TSAdaptInitializePackage - This function initializes everything in the `TSAdapt` package. It is
  called from `TSInitializePackage()`.

  Level: developer

.seealso: [](ch_ts), `PetscInitialize()`
@*/
PetscErrorCode TSAdaptInitializePackage(void)
{
  PetscFunctionBegin;
  if (TSAdaptPackageInitialized) PetscFunctionReturn(PETSC_SUCCESS);
  TSAdaptPackageInitialized = PETSC_TRUE;
  PetscCall(PetscClassIdRegister("TSAdapt", &TSADAPT_CLASSID));
  PetscCall(TSAdaptRegisterAll());
  PetscCall(PetscRegisterFinalize(TSAdaptFinalizePackage));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSAdaptSetType - sets the approach used for the error adapter

  Logicially Collective

  Input Parameters:
+ adapt - the `TS` adapter, most likely obtained with `TSGetAdapt()`
- type  - one of the `TSAdaptType`

  Options Database Key:
. -ts_adapt_type <basic or dsp or none> - to set the adapter type

  Level: intermediate

.seealso: [](ch_ts), [](sec_ts_error_control), `TSGetAdapt()`, `TSAdaptDestroy()`, `TSAdaptType`, `TSAdaptGetType()`
@*/
PetscErrorCode TSAdaptSetType(TSAdapt adapt, TSAdaptType type)
{
  PetscBool match;
  PetscErrorCode (*r)(TSAdapt);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt, TSADAPT_CLASSID, 1);
  PetscAssertPointer(type, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)adapt, type, &match));
  if (match) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscFunctionListFind(TSAdaptList, type, &r));
  PetscCheck(r, PetscObjectComm((PetscObject)adapt), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown TSAdapt type \"%s\" given", type);
  PetscTryTypeMethod(adapt, destroy);
  PetscCall(PetscMemzero(adapt->ops, sizeof(struct _TSAdaptOps)));
  PetscCall(PetscObjectChangeTypeName((PetscObject)adapt, type));
  PetscCall((*r)(adapt));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSAdaptGetType - gets the `TS` adapter method type (as a string).

  Not Collective

  Input Parameter:
. adapt - The `TS` adapter, most likely obtained with `TSGetAdapt()`

  Output Parameter:
. type - The name of `TS` adapter method

  Level: intermediate

.seealso: [](ch_ts), [](sec_ts_error_control), `TSAdapt`, `TSAdaptType`, `TSAdaptSetType()`
@*/
PetscErrorCode TSAdaptGetType(TSAdapt adapt, TSAdaptType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt, TSADAPT_CLASSID, 1);
  PetscAssertPointer(type, 2);
  *type = ((PetscObject)adapt)->type_name;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TSAdaptSetOptionsPrefix(TSAdapt adapt, const char prefix[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt, TSADAPT_CLASSID, 1);
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)adapt, prefix));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSAdaptLoad - Loads a TSAdapt that has been stored in binary with `TSAdaptView()`.

  Collective

  Input Parameters:
+ adapt  - the newly loaded `TSAdapt`, this needs to have been created with `TSAdaptCreate()` or
           some related function before a call to `TSAdaptLoad()`.
- viewer - binary file viewer, obtained from `PetscViewerBinaryOpen()` or
           HDF5 file viewer, obtained from `PetscViewerHDF5Open()`

  Level: intermediate

  Note:
  The type is determined by the data in the file, any type set into the `TSAdapt` before this call is ignored.

.seealso: [](ch_ts), `PetscViewerBinaryOpen()`, `TSAdaptView()`, `MatLoad()`, `VecLoad()`, `TSAdapt`
@*/
PetscErrorCode TSAdaptLoad(TSAdapt adapt, PetscViewer viewer)
{
  PetscBool isbinary;
  char      type[256];

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt, TSADAPT_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERBINARY, &isbinary));
  PetscCheck(isbinary, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid viewer; open viewer with PetscViewerBinaryOpen()");

  PetscCall(PetscViewerBinaryRead(viewer, type, 256, NULL, PETSC_CHAR));
  PetscCall(TSAdaptSetType(adapt, type));
  PetscTryTypeMethod(adapt, load, viewer);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSAdaptView - Prints the `TSAdapt` data structure.

  Collective

  Input Parameters:
+ adapt  - the `TSAdapt` context obtained from `TSGetAdapt()`
- viewer - visualization context

  Options Database Key:
. -ts_view - calls `TSView()` at end of `TSStep()`

  Level: advanced

  Notes:
  This is called by `TSView()` so rarely called directly.

  The available visualization contexts include
+     `PETSC_VIEWER_STDOUT_SELF` - standard output (default)
-     `PETSC_VIEWER_STDOUT_WORLD` - synchronized standard
  output where only the first processor opens
  the file. All other processes send their
  data to the first process to print.

  The user can open an alternative visualization context with
  `PetscViewerASCIIOpen()` - output to a specified file.

  In the debugger you can do call `TSAdaptView`(adapt,0) to display the `TSAdapt`. (The same holds for any PETSc object viewer).

.seealso: [](ch_ts), [](sec_ts_error_control), `TSAdapt`, `TSView()`, `PetscViewer`, `PetscViewerASCIIOpen()`
@*/
PetscErrorCode TSAdaptView(TSAdapt adapt, PetscViewer viewer)
{
  PetscBool isascii, isbinary, isnone, isglee;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt, TSADAPT_CLASSID, 1);
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)adapt), &viewer));
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCheckSameComm(adapt, 1, viewer, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERBINARY, &isbinary));
  if (isascii) {
    PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)adapt, viewer));
    PetscCall(PetscObjectTypeCompare((PetscObject)adapt, TSADAPTNONE, &isnone));
    PetscCall(PetscObjectTypeCompare((PetscObject)adapt, TSADAPTGLEE, &isglee));
    if (!isnone) {
      if (adapt->always_accept) PetscCall(PetscViewerASCIIPrintf(viewer, "  always accepting steps\n"));
      PetscCall(PetscViewerASCIIPrintf(viewer, "  safety factor %g\n", (double)adapt->safety));
      PetscCall(PetscViewerASCIIPrintf(viewer, "  extra safety factor after step rejection %g\n", (double)adapt->reject_safety));
      PetscCall(PetscViewerASCIIPrintf(viewer, "  clip fastest increase %g\n", (double)adapt->clip[1]));
      PetscCall(PetscViewerASCIIPrintf(viewer, "  clip fastest decrease %g\n", (double)adapt->clip[0]));
      PetscCall(PetscViewerASCIIPrintf(viewer, "  maximum allowed timestep %g\n", (double)adapt->dt_max));
      PetscCall(PetscViewerASCIIPrintf(viewer, "  minimum allowed timestep %g\n", (double)adapt->dt_min));
      PetscCall(PetscViewerASCIIPrintf(viewer, "  maximum solution absolute value to be ignored %g\n", (double)adapt->ignore_max));
    }
    if (isglee) {
      if (adapt->glee_use_local) {
        PetscCall(PetscViewerASCIIPrintf(viewer, "  GLEE uses local error control\n"));
      } else {
        PetscCall(PetscViewerASCIIPrintf(viewer, "  GLEE uses global error control\n"));
      }
    }
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscTryTypeMethod(adapt, view, viewer);
    PetscCall(PetscViewerASCIIPopTab(viewer));
  } else if (isbinary) {
    char type[256];

    /* need to save FILE_CLASS_ID for adapt class */
    PetscCall(PetscStrncpy(type, ((PetscObject)adapt)->type_name, 256));
    PetscCall(PetscViewerBinaryWrite(viewer, type, 256, PETSC_CHAR));
  } else PetscTryTypeMethod(adapt, view, viewer);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSAdaptReset - Resets a `TSAdapt` context to its defaults

  Collective

  Input Parameter:
. adapt - the `TSAdapt` context obtained from `TSGetAdapt()` or `TSAdaptCreate()`

  Level: developer

.seealso: [](ch_ts), [](sec_ts_error_control), `TSGetAdapt()`, `TSAdapt`, `TSAdaptCreate()`, `TSAdaptDestroy()`
@*/
PetscErrorCode TSAdaptReset(TSAdapt adapt)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt, TSADAPT_CLASSID, 1);
  PetscTryTypeMethod(adapt, reset);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TSAdaptDestroy(TSAdapt *adapt)
{
  PetscFunctionBegin;
  if (!*adapt) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific(*adapt, TSADAPT_CLASSID, 1);
  if (--((PetscObject)*adapt)->refct > 0) {
    *adapt = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(TSAdaptReset(*adapt));

  PetscTryTypeMethod(*adapt, destroy);
  PetscCall(PetscViewerDestroy(&(*adapt)->monitor));
  PetscCall(PetscHeaderDestroy(adapt));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSAdaptSetMonitor - Monitor the choices made by the adaptive controller

  Collective

  Input Parameters:
+ adapt - adaptive controller context
- flg   - `PETSC_TRUE` to active a monitor, `PETSC_FALSE` to disable

  Options Database Key:
. -ts_adapt_monitor - to turn on monitoring

  Level: intermediate

.seealso: [](ch_ts), [](sec_ts_error_control), `TSAdapt`, `TSGetAdapt()`, `TSAdaptChoose()`
@*/
PetscErrorCode TSAdaptSetMonitor(TSAdapt adapt, PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt, TSADAPT_CLASSID, 1);
  PetscValidLogicalCollectiveBool(adapt, flg, 2);
  if (flg) {
    if (!adapt->monitor) PetscCall(PetscViewerASCIIOpen(PetscObjectComm((PetscObject)adapt), "stdout", &adapt->monitor));
  } else {
    PetscCall(PetscViewerDestroy(&adapt->monitor));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TSAdaptSetCheckStage - Set a callback to check convergence for a stage

  Logically Collective

  Input Parameters:
+ adapt - adaptive controller context
- func  - stage check function

  Calling sequence:
+ adapt  - adaptive controller context
. ts     - time stepping context
. t      - current time
. Y      - current solution vector
- accept - pending choice of whether to accept, can be modified by this routine

  Level: advanced

.seealso: [](ch_ts), [](sec_ts_error_control), `TSAdapt`, `TSGetAdapt()`, `TSAdaptChoose()`
@*/
PetscErrorCode TSAdaptSetCheckStage(TSAdapt adapt, PetscErrorCode (*func)(TSAdapt adapt, TS ts, PetscReal t, Vec Y, PetscBool *accept))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt, TSADAPT_CLASSID, 1);
  adapt->checkstage = func;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSAdaptSetAlwaysAccept - Set whether to always accept steps regardless of
  any error or stability condition not meeting the prescribed goal.

  Logically Collective

  Input Parameters:
+ adapt - time step adaptivity context, usually gotten with `TSGetAdapt()`
- flag  - whether to always accept steps

  Options Database Key:
. -ts_adapt_always_accept - to always accept steps

  Level: intermediate

.seealso: [](ch_ts), [](sec_ts_error_control), `TSAdapt`, `TSGetAdapt()`, `TSAdaptChoose()`
@*/
PetscErrorCode TSAdaptSetAlwaysAccept(TSAdapt adapt, PetscBool flag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt, TSADAPT_CLASSID, 1);
  PetscValidLogicalCollectiveBool(adapt, flag, 2);
  adapt->always_accept = flag;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSAdaptSetSafety - Set safety factors for time step adaptor

  Logically Collective

  Input Parameters:
+ adapt         - adaptive controller context
. safety        - safety factor relative to target error/stability goal
- reject_safety - extra safety factor to apply if the last step was rejected

  Options Database Keys:
+ -ts_adapt_safety <safety>               - to set safety factor
- -ts_adapt_reject_safety <reject_safety> - to set reject safety factor

  Level: intermediate

  Note:
  Use `PETSC_CURRENT` to keep the current value for either parameter

  Fortran Note:
  Use `PETSC_CURRENT_REAL`

.seealso: [](ch_ts), [](sec_ts_error_control), `TSAdapt`, `TSAdaptGetSafety()`, `TSAdaptChoose()`
@*/
PetscErrorCode TSAdaptSetSafety(TSAdapt adapt, PetscReal safety, PetscReal reject_safety)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt, TSADAPT_CLASSID, 1);
  PetscValidLogicalCollectiveReal(adapt, safety, 2);
  PetscValidLogicalCollectiveReal(adapt, reject_safety, 3);
  PetscCheck(safety == (PetscReal)PETSC_CURRENT || safety >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Safety factor %g must be non negative", (double)safety);
  PetscCheck(safety == (PetscReal)PETSC_CURRENT || safety <= 1, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Safety factor %g must be less than one", (double)safety);
  PetscCheck(reject_safety == (PetscReal)PETSC_CURRENT || reject_safety >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Reject safety factor %g must be non negative", (double)reject_safety);
  PetscCheck(reject_safety == (PetscReal)PETSC_CURRENT || reject_safety <= 1, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Reject safety factor %g must be less than one", (double)reject_safety);
  if (safety != (PetscReal)PETSC_CURRENT) adapt->safety = safety;
  if (reject_safety != (PetscReal)PETSC_CURRENT) adapt->reject_safety = reject_safety;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSAdaptGetSafety - Get safety factors for time step adapter

  Not Collective

  Input Parameter:
. adapt - adaptive controller context

  Output Parameters:
+ safety        - safety factor relative to target error/stability goal
- reject_safety - extra safety factor to apply if the last step was rejected

  Level: intermediate

.seealso: [](ch_ts), [](sec_ts_error_control), `TSAdapt`, `TSAdaptSetSafety()`, `TSAdaptChoose()`
@*/
PetscErrorCode TSAdaptGetSafety(TSAdapt adapt, PetscReal *safety, PetscReal *reject_safety)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt, TSADAPT_CLASSID, 1);
  if (safety) PetscAssertPointer(safety, 2);
  if (reject_safety) PetscAssertPointer(reject_safety, 3);
  if (safety) *safety = adapt->safety;
  if (reject_safety) *reject_safety = adapt->reject_safety;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSAdaptSetMaxIgnore - Set error estimation threshold. Solution components below this threshold value will not be considered when computing error norms
  for time step adaptivity (in absolute value). A negative value (default) of the threshold leads to considering all solution components.

  Logically Collective

  Input Parameters:
+ adapt      - adaptive controller context
- max_ignore - threshold for solution components that are ignored during error estimation

  Options Database Key:
. -ts_adapt_max_ignore <max_ignore> - to set the threshold

  Level: intermediate

.seealso: [](ch_ts), [](sec_ts_error_control), `TSAdapt`, `TSAdaptGetMaxIgnore()`, `TSAdaptChoose()`
@*/
PetscErrorCode TSAdaptSetMaxIgnore(TSAdapt adapt, PetscReal max_ignore)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt, TSADAPT_CLASSID, 1);
  PetscValidLogicalCollectiveReal(adapt, max_ignore, 2);
  adapt->ignore_max = max_ignore;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSAdaptGetMaxIgnore - Get error estimation threshold. Solution components below this threshold value will not be considered when computing error norms
  for time step adaptivity (in absolute value).

  Not Collective

  Input Parameter:
. adapt - adaptive controller context

  Output Parameter:
. max_ignore - threshold for solution components that are ignored during error estimation

  Level: intermediate

.seealso: [](ch_ts), [](sec_ts_error_control), `TSAdapt`, `TSAdaptSetMaxIgnore()`, `TSAdaptChoose()`
@*/
PetscErrorCode TSAdaptGetMaxIgnore(TSAdapt adapt, PetscReal *max_ignore)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt, TSADAPT_CLASSID, 1);
  PetscAssertPointer(max_ignore, 2);
  *max_ignore = adapt->ignore_max;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSAdaptSetClip - Sets the admissible decrease/increase factor in step size in the time step adapter

  Logically collective

  Input Parameters:
+ adapt - adaptive controller context
. low   - admissible decrease factor
- high  - admissible increase factor

  Options Database Key:
. -ts_adapt_clip <low>,<high> - to set admissible time step decrease and increase factors

  Level: intermediate

  Note:
  Use `PETSC_CURRENT` to keep the current value for either parameter

  Fortran Note:
  Use `PETSC_CURRENT_REAL`

.seealso: [](ch_ts), [](sec_ts_error_control), `TSAdapt`, `TSAdaptChoose()`, `TSAdaptGetClip()`, `TSAdaptSetScaleSolveFailed()`
@*/
PetscErrorCode TSAdaptSetClip(TSAdapt adapt, PetscReal low, PetscReal high)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt, TSADAPT_CLASSID, 1);
  PetscValidLogicalCollectiveReal(adapt, low, 2);
  PetscValidLogicalCollectiveReal(adapt, high, 3);
  PetscCheck(low == (PetscReal)PETSC_CURRENT || low >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Decrease factor %g must be non negative", (double)low);
  PetscCheck(low == (PetscReal)PETSC_CURRENT || low <= 1, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Decrease factor %g must be less than one", (double)low);
  PetscCheck(high == (PetscReal)PETSC_CURRENT || high >= 1, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Increase factor %g must be greater than one", (double)high);
  if (low != (PetscReal)PETSC_CURRENT) adapt->clip[0] = low;
  if (high != (PetscReal)PETSC_CURRENT) adapt->clip[1] = high;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSAdaptGetClip - Gets the admissible decrease/increase factor in step size in the time step adapter

  Not Collective

  Input Parameter:
. adapt - adaptive controller context

  Output Parameters:
+ low  - optional, admissible decrease factor
- high - optional, admissible increase factor

  Level: intermediate

.seealso: [](ch_ts), [](sec_ts_error_control), `TSAdapt`, `TSAdaptChoose()`, `TSAdaptSetClip()`, `TSAdaptSetScaleSolveFailed()`
@*/
PetscErrorCode TSAdaptGetClip(TSAdapt adapt, PetscReal *low, PetscReal *high)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt, TSADAPT_CLASSID, 1);
  if (low) PetscAssertPointer(low, 2);
  if (high) PetscAssertPointer(high, 3);
  if (low) *low = adapt->clip[0];
  if (high) *high = adapt->clip[1];
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSAdaptSetScaleSolveFailed - Scale step size by this factor if solve fails

  Logically Collective

  Input Parameters:
+ adapt - adaptive controller context
- scale - scale

  Options Database Key:
. -ts_adapt_scale_solve_failed <scale> - to set scale step by this factor if solve fails

  Level: intermediate

.seealso: [](ch_ts), [](sec_ts_error_control), `TSAdapt`, `TSAdaptChoose()`, `TSAdaptGetScaleSolveFailed()`, `TSAdaptGetClip()`
@*/
PetscErrorCode TSAdaptSetScaleSolveFailed(TSAdapt adapt, PetscReal scale)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt, TSADAPT_CLASSID, 1);
  PetscValidLogicalCollectiveReal(adapt, scale, 2);
  PetscCheck(scale > 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Scale factor %g must be positive", (double)scale);
  PetscCheck(scale <= 1, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Scale factor %g must be less than one", (double)scale);
  adapt->scale_solve_failed = scale;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSAdaptGetScaleSolveFailed - Gets the admissible decrease/increase factor in step size

  Not Collective

  Input Parameter:
. adapt - adaptive controller context

  Output Parameter:
. scale - scale factor

  Level: intermediate

.seealso: [](ch_ts), [](sec_ts_error_control), `TSAdapt`, `TSAdaptChoose()`, `TSAdaptSetScaleSolveFailed()`, `TSAdaptSetClip()`
@*/
PetscErrorCode TSAdaptGetScaleSolveFailed(TSAdapt adapt, PetscReal *scale)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt, TSADAPT_CLASSID, 1);
  if (scale) PetscAssertPointer(scale, 2);
  if (scale) *scale = adapt->scale_solve_failed;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSAdaptSetStepLimits - Set the minimum and maximum step sizes to be considered by the time step controller

  Logically Collective

  Input Parameters:
+ adapt - time step adaptivity context, usually gotten with `TSGetAdapt()`
. hmin  - minimum time step
- hmax  - maximum time step

  Options Database Keys:
+ -ts_adapt_dt_min <min> - to set minimum time step
- -ts_adapt_dt_max <max> - to set maximum time step

  Level: intermediate

  Note:
  Use `PETSC_CURRENT` to keep the current value for either parameter

  Fortran Note:
  Use `PETSC_CURRENT_REAL`

.seealso: [](ch_ts), [](sec_ts_error_control), `TSAdapt`, `TSAdaptGetStepLimits()`, `TSAdaptChoose()`
@*/
PetscErrorCode TSAdaptSetStepLimits(TSAdapt adapt, PetscReal hmin, PetscReal hmax)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt, TSADAPT_CLASSID, 1);
  PetscValidLogicalCollectiveReal(adapt, hmin, 2);
  PetscValidLogicalCollectiveReal(adapt, hmax, 3);
  PetscCheck(hmin == (PetscReal)PETSC_CURRENT || hmin >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Minimum time step %g must be non negative", (double)hmin);
  PetscCheck(hmax == (PetscReal)PETSC_CURRENT || hmax >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Minimum time step %g must be non negative", (double)hmax);
  if (hmin != (PetscReal)PETSC_CURRENT) adapt->dt_min = hmin;
  if (hmax != (PetscReal)PETSC_CURRENT) adapt->dt_max = hmax;
  hmin = adapt->dt_min;
  hmax = adapt->dt_max;
  PetscCheck(hmax > hmin, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Maximum time step %g must greater than minimum time step %g", (double)hmax, (double)hmin);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSAdaptGetStepLimits - Get the minimum and maximum step sizes to be considered by the time step controller

  Not Collective

  Input Parameter:
. adapt - time step adaptivity context, usually gotten with `TSGetAdapt()`

  Output Parameters:
+ hmin - minimum time step
- hmax - maximum time step

  Level: intermediate

.seealso: [](ch_ts), [](sec_ts_error_control), `TSAdapt`, `TSAdaptSetStepLimits()`, `TSAdaptChoose()`
@*/
PetscErrorCode TSAdaptGetStepLimits(TSAdapt adapt, PetscReal *hmin, PetscReal *hmax)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt, TSADAPT_CLASSID, 1);
  if (hmin) PetscAssertPointer(hmin, 2);
  if (hmax) PetscAssertPointer(hmax, 3);
  if (hmin) *hmin = adapt->dt_min;
  if (hmax) *hmax = adapt->dt_max;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TSAdaptSetFromOptions - Sets various `TSAdapt` parameters from user options.

  Collective

  Input Parameters:
+ adapt              - the `TSAdapt` context
- PetscOptionsObject - object created by `PetscOptionsBegin()`

  Options Database Keys:
+ -ts_adapt_type <type>                - algorithm to use for adaptivity
. -ts_adapt_always_accept              - always accept steps regardless of error/stability goals
. -ts_adapt_safety <safety>            - safety factor relative to target error/stability goal
. -ts_adapt_reject_safety <safety>     - extra safety factor to apply if the last step was rejected
. -ts_adapt_clip <low,high>            - admissible time step decrease and increase factors
. -ts_adapt_dt_min <min>               - minimum timestep to use
. -ts_adapt_dt_max <max>               - maximum timestep to use
. -ts_adapt_scale_solve_failed <scale> - scale timestep by this factor if a solve fails
. -ts_adapt_wnormtype <2 or infinity>  - type of norm for computing error estimates
- -ts_adapt_time_step_increase_delay   - number of timesteps to delay increasing the time step after it has been decreased due to failed solver

  Level: advanced

  Note:
  This function is automatically called by `TSSetFromOptions()`

.seealso: [](ch_ts), [](sec_ts_error_control), `TSAdapt`, `TSGetAdapt()`, `TSAdaptSetType()`, `TSAdaptSetAlwaysAccept()`, `TSAdaptSetSafety()`,
          `TSAdaptSetClip()`, `TSAdaptSetScaleSolveFailed()`, `TSAdaptSetStepLimits()`, `TSAdaptSetMonitor()`
@*/
PetscErrorCode TSAdaptSetFromOptions(TSAdapt adapt, PetscOptionItems PetscOptionsObject)
{
  char      type[256] = TSADAPTBASIC;
  PetscReal safety, reject_safety, clip[2], scale, hmin, hmax;
  PetscBool set, flg;
  PetscInt  two;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt, TSADAPT_CLASSID, 1);
  /* This should use PetscOptionsBegin() if/when this becomes an object used outside of TS, but currently this
   * function can only be called from inside TSSetFromOptions()  */
  PetscOptionsHeadBegin(PetscOptionsObject, "TS Adaptivity options");
  PetscCall(PetscOptionsFList("-ts_adapt_type", "Algorithm to use for adaptivity", "TSAdaptSetType", TSAdaptList, ((PetscObject)adapt)->type_name ? ((PetscObject)adapt)->type_name : type, type, sizeof(type), &flg));
  if (flg || !((PetscObject)adapt)->type_name) PetscCall(TSAdaptSetType(adapt, type));

  PetscCall(PetscOptionsBool("-ts_adapt_always_accept", "Always accept the step", "TSAdaptSetAlwaysAccept", adapt->always_accept, &flg, &set));
  if (set) PetscCall(TSAdaptSetAlwaysAccept(adapt, flg));

  safety        = adapt->safety;
  reject_safety = adapt->reject_safety;
  PetscCall(PetscOptionsReal("-ts_adapt_safety", "Safety factor relative to target error/stability goal", "TSAdaptSetSafety", safety, &safety, &set));
  PetscCall(PetscOptionsReal("-ts_adapt_reject_safety", "Extra safety factor to apply if the last step was rejected", "TSAdaptSetSafety", reject_safety, &reject_safety, &flg));
  if (set || flg) PetscCall(TSAdaptSetSafety(adapt, safety, reject_safety));

  two     = 2;
  clip[0] = adapt->clip[0];
  clip[1] = adapt->clip[1];
  PetscCall(PetscOptionsRealArray("-ts_adapt_clip", "Admissible decrease/increase factor in step size", "TSAdaptSetClip", clip, &two, &set));
  PetscCheck(!set || (two == 2), PetscObjectComm((PetscObject)adapt), PETSC_ERR_ARG_OUTOFRANGE, "Must give exactly two values to -ts_adapt_clip");
  if (set) PetscCall(TSAdaptSetClip(adapt, clip[0], clip[1]));

  hmin = adapt->dt_min;
  hmax = adapt->dt_max;
  PetscCall(PetscOptionsReal("-ts_adapt_dt_min", "Minimum time step considered", "TSAdaptSetStepLimits", hmin, &hmin, &set));
  PetscCall(PetscOptionsReal("-ts_adapt_dt_max", "Maximum time step considered", "TSAdaptSetStepLimits", hmax, &hmax, &flg));
  if (set || flg) PetscCall(TSAdaptSetStepLimits(adapt, hmin, hmax));

  PetscCall(PetscOptionsReal("-ts_adapt_max_ignore", "Adaptor ignores (absolute) solution values smaller than this value", "", adapt->ignore_max, &adapt->ignore_max, &set));
  PetscCall(PetscOptionsBool("-ts_adapt_glee_use_local", "GLEE adaptor uses local error estimation for step control", "", adapt->glee_use_local, &adapt->glee_use_local, &set));

  PetscCall(PetscOptionsReal("-ts_adapt_scale_solve_failed", "Scale step by this factor if solve fails", "TSAdaptSetScaleSolveFailed", adapt->scale_solve_failed, &scale, &set));
  if (set) PetscCall(TSAdaptSetScaleSolveFailed(adapt, scale));

  PetscCall(PetscOptionsEnum("-ts_adapt_wnormtype", "Type of norm computed for error estimation", "", NormTypes, (PetscEnum)adapt->wnormtype, (PetscEnum *)&adapt->wnormtype, NULL));
  PetscCheck(adapt->wnormtype == NORM_2 || adapt->wnormtype == NORM_INFINITY, PetscObjectComm((PetscObject)adapt), PETSC_ERR_SUP, "Only 2-norm and infinite norm supported");

  PetscCall(PetscOptionsInt("-ts_adapt_time_step_increase_delay", "Number of timesteps to delay increasing the time step after it has been decreased due to failed solver", "TSAdaptSetTimeStepIncreaseDelay", adapt->timestepjustdecreased_delay, &adapt->timestepjustdecreased_delay, NULL));

  PetscCall(PetscOptionsBool("-ts_adapt_monitor", "Print choices made by adaptive controller", "TSAdaptSetMonitor", adapt->monitor ? PETSC_TRUE : PETSC_FALSE, &flg, &set));
  if (set) PetscCall(TSAdaptSetMonitor(adapt, flg));

  PetscTryTypeMethod(adapt, setfromoptions, PetscOptionsObject);
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSAdaptCandidatesClear - clear any previously set candidate schemes

  Logically Collective

  Input Parameter:
. adapt - adaptive controller

  Level: developer

.seealso: [](ch_ts), [](sec_ts_error_control), `TSAdapt`, `TSAdaptCreate()`, `TSAdaptCandidateAdd()`, `TSAdaptChoose()`
@*/
PetscErrorCode TSAdaptCandidatesClear(TSAdapt adapt)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt, TSADAPT_CLASSID, 1);
  PetscCall(PetscMemzero(&adapt->candidates, sizeof(adapt->candidates)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TSAdaptCandidateAdd - add a candidate scheme for the adaptive controller to select from

  Logically Collective; No Fortran Support

  Input Parameters:
+ adapt      - time step adaptivity context, obtained with `TSGetAdapt()` or `TSAdaptCreate()`
. name       - name of the candidate scheme to add
. order      - order of the candidate scheme
. stageorder - stage order of the candidate scheme
. ccfl       - stability coefficient relative to explicit Euler, used for CFL constraints
. cost       - relative measure of the amount of work required for the candidate scheme
- inuse      - indicates that this scheme is the one currently in use, this flag can only be set for one scheme

  Level: developer

.seealso: [](ch_ts), [](sec_ts_error_control), `TSAdapt`, `TSAdaptCandidatesClear()`, `TSAdaptChoose()`
@*/
PetscErrorCode TSAdaptCandidateAdd(TSAdapt adapt, const char name[], PetscInt order, PetscInt stageorder, PetscReal ccfl, PetscReal cost, PetscBool inuse)
{
  PetscInt c;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt, TSADAPT_CLASSID, 1);
  PetscCheck(order >= 1, PetscObjectComm((PetscObject)adapt), PETSC_ERR_ARG_OUTOFRANGE, "Classical order %" PetscInt_FMT " must be a positive integer", order);
  if (inuse) {
    PetscCheck(!adapt->candidates.inuse_set, PetscObjectComm((PetscObject)adapt), PETSC_ERR_ARG_WRONGSTATE, "Cannot set the inuse method twice, maybe forgot to call TSAdaptCandidatesClear()");
    adapt->candidates.inuse_set = PETSC_TRUE;
  }
  /* first slot if this is the current scheme, otherwise the next available slot */
  c = inuse ? 0 : !adapt->candidates.inuse_set + adapt->candidates.n;

  adapt->candidates.name[c]       = name;
  adapt->candidates.order[c]      = order;
  adapt->candidates.stageorder[c] = stageorder;
  adapt->candidates.ccfl[c]       = ccfl;
  adapt->candidates.cost[c]       = cost;
  adapt->candidates.n++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TSAdaptCandidatesGet - Get the list of candidate orders of accuracy and cost

  Not Collective

  Input Parameter:
. adapt - time step adaptivity context

  Output Parameters:
+ n          - number of candidate schemes, always at least 1
. order      - the order of each candidate scheme
. stageorder - the stage order of each candidate scheme
. ccfl       - the CFL coefficient of each scheme
- cost       - the relative cost of each scheme

  Level: developer

  Note:
  The current scheme is always returned in the first slot

.seealso: [](ch_ts), [](sec_ts_error_control), `TSAdapt`, `TSAdaptCandidatesClear()`, `TSAdaptCandidateAdd()`, `TSAdaptChoose()`
@*/
PetscErrorCode TSAdaptCandidatesGet(TSAdapt adapt, PetscInt *n, const PetscInt **order, const PetscInt **stageorder, const PetscReal **ccfl, const PetscReal **cost)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt, TSADAPT_CLASSID, 1);
  if (n) *n = adapt->candidates.n;
  if (order) *order = adapt->candidates.order;
  if (stageorder) *stageorder = adapt->candidates.stageorder;
  if (ccfl) *ccfl = adapt->candidates.ccfl;
  if (cost) *cost = adapt->candidates.cost;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TSAdaptChoose - choose which method and step size to use for the next step

  Collective

  Input Parameters:
+ adapt - adaptive controller
. ts    - time stepper
- h     - current step size

  Output Parameters:
+ next_sc - optional, scheme to use for the next step
. next_h  - step size to use for the next step
- accept  - `PETSC_TRUE` to accept the current step, `PETSC_FALSE` to repeat the current step with the new step size

  Level: developer

  Note:
  The input value of parameter accept is retained from the last time step, so it will be `PETSC_FALSE` if the step is
  being retried after an initial rejection.

.seealso: [](ch_ts), [](sec_ts_error_control), `TSAdapt`, `TSAdaptCandidatesClear()`, `TSAdaptCandidateAdd()`
@*/
PetscErrorCode TSAdaptChoose(TSAdapt adapt, TS ts, PetscReal h, PetscInt *next_sc, PetscReal *next_h, PetscBool *accept)
{
  PetscInt  ncandidates = adapt->candidates.n;
  PetscInt  scheme      = 0;
  PetscReal wlte        = -1.0;
  PetscReal wltea       = -1.0;
  PetscReal wlter       = -1.0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt, TSADAPT_CLASSID, 1);
  PetscValidHeaderSpecific(ts, TS_CLASSID, 2);
  if (next_sc) PetscAssertPointer(next_sc, 4);
  PetscAssertPointer(next_h, 5);
  PetscAssertPointer(accept, 6);
  if (next_sc) *next_sc = 0;

  /* Do not mess with adaptivity while handling events */
  if (ts->event && ts->event->processing) {
    *next_h = h;
    *accept = PETSC_TRUE;
    if (adapt->monitor) {
      PetscCall(PetscViewerASCIIAddTab(adapt->monitor, ((PetscObject)adapt)->tablevel));

      if (ts->event->iterctr == 0) {
        /*
          An event has been found, now finalising the event processing: performing the 1st and 2nd post-event steps.
          Entering this if-branch means both these steps (set to either PETSC_DECIDE or numerical value) are managed
          by the event handler. In this case the 1st post-event step is always accepted, without interference of TSAdapt.
          Note: if the 2nd post-event step is not managed by the event handler (e.g. given 1st = numerical, 2nd = PETSC_DECIDE),
          this if-branch is not entered, and TSAdapt may reject/adjust the proposed 1st post-event step.
        */
        PetscCall(PetscViewerASCIIPrintf(adapt->monitor, "    TSAdapt does not interfere, step %3" PetscInt_FMT " accepted. Processing post-event steps: 1-st accepted just now, 2-nd yet to come\n", ts->steps));
      } else PetscCall(PetscViewerASCIIPrintf(adapt->monitor, "    TSAdapt does not interfere, step %3" PetscInt_FMT " accepted. Event handling in progress\n", ts->steps));

      PetscCall(PetscViewerASCIISubtractTab(adapt->monitor, ((PetscObject)adapt)->tablevel));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscUseTypeMethod(adapt, choose, ts, h, &scheme, next_h, accept, &wlte, &wltea, &wlter);
  PetscCheck(scheme >= 0 && (ncandidates <= 0 || scheme < ncandidates), PetscObjectComm((PetscObject)adapt), PETSC_ERR_ARG_OUTOFRANGE, "Chosen scheme %" PetscInt_FMT " not in valid range 0..%" PetscInt_FMT, scheme, ncandidates - 1);
  PetscCheck(*next_h >= 0, PetscObjectComm((PetscObject)adapt), PETSC_ERR_ARG_OUTOFRANGE, "Computed step size %g must be positive", (double)*next_h);
  if (next_sc) *next_sc = scheme;

  if (*accept && ts->exact_final_time == TS_EXACTFINALTIME_MATCHSTEP) {
    /* Increase/reduce step size if end time of next step is close to or overshoots max time */
    PetscReal t = ts->ptime + ts->time_step, tend, tmax, h1, hmax;
    PetscReal a = (PetscReal)(1.0 + adapt->matchstepfac[0]);
    PetscReal b = adapt->matchstepfac[1];

    /*
      Logic in using 'dt_span_cached':
      1. It always overrides *next_h, except (any of):
         a) the current step was rejected,
         b) the adaptor proposed to decrease the next step,
         c) the adaptor proposed *next_h > dt_span_cached.
      2. If *next_h was adjusted by eval_times points (or the final point):
           -- when dt_span_cached is filled (>0), it keeps its value,
           -- when dt_span_cached is clear (==0), it gets the unadjusted version of *next_h.
      3. If *next_h was not adjusted as in (2), dt_span_cached is cleared.
      Note, if a combination (1.b || 1.c) && (3) takes place, this means that
      dt_span_cached remains unused at the moment of clearing.
      If (1.a) takes place, dt_span_cached keeps its value.
      Also, dt_span_cached can be updated by the event handler, see tsevent.c.
    */
    if (h <= *next_h && *next_h <= adapt->dt_eval_times_cached) *next_h = adapt->dt_eval_times_cached; /* try employing the cache */
    h1   = *next_h;
    tend = t + h1;

    if (ts->eval_times && ts->eval_times->time_point_idx < ts->eval_times->num_time_points) {
      PetscCheck(ts->eval_times->worktol == 0, PetscObjectComm((PetscObject)adapt), PETSC_ERR_PLIB, "Unexpected state (tspan->worktol != 0) in TSAdaptChoose()");
      ts->eval_times->worktol = ts->eval_times->reltol * h1 + ts->eval_times->abstol;
      if (PetscIsCloseAtTol(t, ts->eval_times->time_points[ts->eval_times->time_point_idx], ts->eval_times->worktol, 0)) /* hit a span time point */
        if (ts->eval_times->time_point_idx + 1 < ts->eval_times->num_time_points) tmax = ts->eval_times->time_points[ts->eval_times->time_point_idx + 1];
        else tmax = ts->max_time; /* hit the last span time point */
      else tmax = ts->eval_times->time_points[ts->eval_times->time_point_idx];
    } else tmax = ts->max_time;
    tmax = PetscMin(tmax, ts->max_time);
    hmax = tmax - t;

    if (t < tmax && tend > tmax) *next_h = hmax;
    if (t < tmax && tend < tmax && h1 * b > hmax) *next_h = hmax / 2;
    if (t < tmax && tend < tmax && h1 * a > hmax) *next_h = hmax;
    if (ts->eval_times && h1 != *next_h && !adapt->dt_eval_times_cached) adapt->dt_eval_times_cached = h1; /* cache the step size if it is to be changed    */
    if (ts->eval_times && h1 == *next_h && adapt->dt_eval_times_cached) adapt->dt_eval_times_cached = 0;   /* clear the cache if the step size is unchanged */
  }
  if (adapt->monitor) {
    const char *sc_name = (scheme < ncandidates) ? adapt->candidates.name[scheme] : "";
    PetscCall(PetscViewerASCIIAddTab(adapt->monitor, ((PetscObject)adapt)->tablevel));
    if (wlte < 0) {
      PetscCall(PetscViewerASCIIPrintf(adapt->monitor, "    TSAdapt %s %s %" PetscInt_FMT ":%s step %3" PetscInt_FMT " %s t=%-11g+%10.3e dt=%-10.3e\n", ((PetscObject)adapt)->type_name, ((PetscObject)ts)->type_name, scheme, sc_name, ts->steps, *accept ? "accepted" : "rejected",
                                       (double)ts->ptime, (double)h, (double)*next_h));
    } else {
      PetscCall(PetscViewerASCIIPrintf(adapt->monitor, "    TSAdapt %s %s %" PetscInt_FMT ":%s step %3" PetscInt_FMT " %s t=%-11g+%10.3e dt=%-10.3e wlte=%5.3g  wltea=%5.3g wlter=%5.3g\n", ((PetscObject)adapt)->type_name, ((PetscObject)ts)->type_name, scheme, sc_name, ts->steps, *accept ? "accepted" : "rejected",
                                       (double)ts->ptime, (double)h, (double)*next_h, (double)wlte, (double)wltea, (double)wlter));
    }
    PetscCall(PetscViewerASCIISubtractTab(adapt->monitor, ((PetscObject)adapt)->tablevel));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSAdaptSetTimeStepIncreaseDelay - The number of timesteps to wait after a decrease in the timestep due to failed solver
  before increasing the time step.

  Logicially Collective

  Input Parameters:
+ adapt - adaptive controller context
- cnt   - the number of timesteps

  Options Database Key:
. -ts_adapt_time_step_increase_delay cnt - number of steps to delay the increase

  Level: advanced

  Notes:
  This is to prevent an adaptor from bouncing back and forth between two nearby timesteps. The default is 0.

  The successful use of this option is problem dependent

  Developer Notes:
  There is no theory to support this option

.seealso: [](ch_ts), [](sec_ts_error_control), `TSAdapt`
@*/
PetscErrorCode TSAdaptSetTimeStepIncreaseDelay(TSAdapt adapt, PetscInt cnt)
{
  PetscFunctionBegin;
  adapt->timestepjustdecreased_delay = cnt;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSAdaptCheckStage - checks whether to accept a stage, (e.g. reject and change time step size if nonlinear solve fails or solution vector is infeasible)

  Collective

  Input Parameters:
+ adapt - adaptive controller context
. ts    - time stepper
. t     - Current simulation time
- Y     - Current solution vector

  Output Parameter:
. accept - `PETSC_TRUE` to accept the stage, `PETSC_FALSE` to reject

  Level: developer

.seealso: [](ch_ts), [](sec_ts_error_control), `TSAdapt`
@*/
PetscErrorCode TSAdaptCheckStage(TSAdapt adapt, TS ts, PetscReal t, Vec Y, PetscBool *accept)
{
  SNESConvergedReason snesreason = SNES_CONVERGED_ITERATING;
  PetscBool           func_accept, snes_div_func;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt, TSADAPT_CLASSID, 1);
  PetscValidHeaderSpecific(ts, TS_CLASSID, 2);
  PetscAssertPointer(accept, 5);

  PetscCall(TSFunctionDomainError(ts, t, Y, &func_accept));
  if (ts->snes) PetscCall(SNESGetConvergedReason(ts->snes, &snesreason));
  snes_div_func = (PetscBool)(snesreason == SNES_DIVERGED_FUNCTION_DOMAIN);
  if (func_accept && snesreason < 0 && !snes_div_func) {
    *accept = PETSC_FALSE;
    PetscCall(PetscInfo(ts, "Step=%" PetscInt_FMT ", nonlinear solve failure: %s\n", ts->steps, SNESConvergedReasons[snesreason]));
    if (++ts->num_snes_failures >= ts->max_snes_failures && ts->max_snes_failures != PETSC_UNLIMITED) {
      ts->reason = TS_DIVERGED_NONLINEAR_SOLVE;
      PetscCall(PetscInfo(ts, "Step=%" PetscInt_FMT ", nonlinear solve failures %" PetscInt_FMT " greater than current TS allowed, stopping solve\n", ts->steps, ts->num_snes_failures));
      if (adapt->monitor) {
        PetscCall(PetscViewerASCIIAddTab(adapt->monitor, ((PetscObject)adapt)->tablevel));
        PetscCall(PetscViewerASCIIPrintf(adapt->monitor, "    TSAdapt %s step %3" PetscInt_FMT " stage rejected t=%-11g+%10.3e, nonlinear solve failures %" PetscInt_FMT " greater than current TS allowed\n", ((PetscObject)adapt)->type_name, ts->steps,
                                         (double)ts->ptime, (double)ts->time_step, ts->num_snes_failures));
        PetscCall(PetscViewerASCIISubtractTab(adapt->monitor, ((PetscObject)adapt)->tablevel));
      }
    }
  } else {
    *accept = (PetscBool)(func_accept && !snes_div_func);
    if (*accept && adapt->checkstage) PetscCall((*adapt->checkstage)(adapt, ts, t, Y, accept));
    if (!*accept) {
      const char *user_func = !func_accept ? "TSSetFunctionDomainError()" : "TSAdaptSetCheckStage";
      const char *snes_err  = "SNES invalid function domain";
      const char *err_msg   = snes_div_func && func_accept ? snes_err : user_func;
      PetscCall(PetscInfo(ts, "Step=%" PetscInt_FMT ", solution rejected by %s\n", ts->steps, err_msg));
      if (adapt->monitor) {
        PetscCall(PetscViewerASCIIAddTab(adapt->monitor, ((PetscObject)adapt)->tablevel));
        PetscCall(PetscViewerASCIIPrintf(adapt->monitor, "    TSAdapt %s step %3" PetscInt_FMT " stage rejected by %s\n", ((PetscObject)adapt)->type_name, ts->steps, err_msg));
        PetscCall(PetscViewerASCIISubtractTab(adapt->monitor, ((PetscObject)adapt)->tablevel));
      }
    }
  }

  if (!*accept && !ts->reason) {
    PetscReal dt, new_dt;
    PetscCall(TSGetTimeStep(ts, &dt));
    new_dt = dt * adapt->scale_solve_failed;
    PetscCall(TSSetTimeStep(ts, new_dt));
    adapt->timestepjustdecreased += adapt->timestepjustdecreased_delay;
    if (adapt->monitor) {
      PetscCall(PetscViewerASCIIAddTab(adapt->monitor, ((PetscObject)adapt)->tablevel));
      PetscCall(PetscViewerASCIIPrintf(adapt->monitor, "    TSAdapt %s step %3" PetscInt_FMT " stage rejected (SNES reason %s) t=%-11g+%10.3e retrying with dt=%-10.3e\n", ((PetscObject)adapt)->type_name, ts->steps, SNESConvergedReasons[snesreason],
                                       (double)ts->ptime, (double)dt, (double)new_dt));
      PetscCall(PetscViewerASCIISubtractTab(adapt->monitor, ((PetscObject)adapt)->tablevel));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSAdaptCreate - create an adaptive controller context for time stepping

  Collective

  Input Parameter:
. comm - The communicator

  Output Parameter:
. inadapt - new `TSAdapt` object

  Level: developer

  Note:
  `TSAdapt` creation is handled by `TS`, so users should not need to call this function.

.seealso: [](ch_ts), [](sec_ts_error_control), `TSAdapt`, `TSGetAdapt()`, `TSAdaptSetType()`, `TSAdaptDestroy()`
@*/
PetscErrorCode TSAdaptCreate(MPI_Comm comm, TSAdapt *inadapt)
{
  TSAdapt adapt;

  PetscFunctionBegin;
  PetscAssertPointer(inadapt, 2);
  PetscCall(TSAdaptInitializePackage());

  PetscCall(PetscHeaderCreate(adapt, TSADAPT_CLASSID, "TSAdapt", "Time stepping adaptivity", "TS", comm, TSAdaptDestroy, TSAdaptView));
  adapt->always_accept      = PETSC_FALSE;
  adapt->safety             = 0.9;
  adapt->reject_safety      = 0.5;
  adapt->clip[0]            = 0.1;
  adapt->clip[1]            = 10.;
  adapt->dt_min             = 1e-20;
  adapt->dt_max             = 1e+20;
  adapt->ignore_max         = -1.0;
  adapt->glee_use_local     = PETSC_TRUE;
  adapt->scale_solve_failed = 0.25;
  /* these two safety factors are not public, and they are used only in the TS_EXACTFINALTIME_MATCHSTEP case
     to prevent from situations were unreasonably small time steps are taken in order to match the final time */
  adapt->matchstepfac[0]             = 0.01; /* allow 1% step size increase in the last step */
  adapt->matchstepfac[1]             = 2.0;  /* halve last step if it is greater than what remains divided this factor */
  adapt->wnormtype                   = NORM_2;
  adapt->timestepjustdecreased_delay = 0;
  *inadapt                           = adapt;
  PetscFunctionReturn(PETSC_SUCCESS);
}
