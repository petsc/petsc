#include <petsc/private/tsimpl.h> /*I "petscts.h"  I*/
#include <petsc/private/tshistoryimpl.h>
#include <petscdm.h>

PetscFunctionList TSTrajectoryList              = NULL;
PetscBool         TSTrajectoryRegisterAllCalled = PETSC_FALSE;
PetscClassId      TSTRAJECTORY_CLASSID;
PetscLogEvent     TSTrajectory_Set, TSTrajectory_Get, TSTrajectory_GetVecs, TSTrajectory_SetUp;

/*@C
  TSTrajectoryRegister - Adds a way of storing trajectories to the `TS` package

  Not Collective

  Input Parameters:
+ sname        - the name of a new user-defined creation routine
- function - the creation routine itself

  Level: developer

  Note:
  `TSTrajectoryRegister()` may be called multiple times to add several user-defined tses.

.seealso: [](chapter_ts), `TSTrajectoryRegisterAll()`
@*/
PetscErrorCode TSTrajectoryRegister(const char sname[], PetscErrorCode (*function)(TSTrajectory, TS))
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListAdd(&TSTrajectoryList, sname, function));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSTrajectorySet - Sets a vector of state in the trajectory object

  Collective

  Input Parameters:
+ tj      - the trajectory object
. ts      - the time stepper object (optional)
. stepnum - the step number
. time    - the current time
- X       - the current solution

  Level: developer

  Note:
  Usually one does not call this routine, it is called automatically during `TSSolve()`

.seealso: [](chapter_ts), `TSTrajectorySetUp()`, `TSTrajectoryDestroy()`, `TSTrajectorySetType()`, `TSTrajectorySetVariableNames()`, `TSGetTrajectory()`, `TSTrajectoryGet()`, `TSTrajectoryGetVecs()`
@*/
PetscErrorCode TSTrajectorySet(TSTrajectory tj, TS ts, PetscInt stepnum, PetscReal time, Vec X)
{
  PetscFunctionBegin;
  if (!tj) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific(tj, TSTRAJECTORY_CLASSID, 1);
  if (ts) PetscValidHeaderSpecific(ts, TS_CLASSID, 2);
  PetscValidLogicalCollectiveInt(tj, stepnum, 3);
  PetscValidLogicalCollectiveReal(tj, time, 4);
  PetscValidHeaderSpecific(X, VEC_CLASSID, 5);
  PetscCheck(tj->setupcalled, PetscObjectComm((PetscObject)tj), PETSC_ERR_ORDER, "TSTrajectorySetUp should be called first");
  if (tj->monitor) PetscCall(PetscViewerASCIIPrintf(tj->monitor, "TSTrajectorySet: stepnum %" PetscInt_FMT ", time %g (stages %" PetscInt_FMT ")\n", stepnum, (double)time, (PetscInt)!tj->solution_only));
  PetscCall(PetscLogEventBegin(TSTrajectory_Set, tj, ts, 0, 0));
  PetscUseTypeMethod(tj, set, ts, stepnum, time, X);
  PetscCall(PetscLogEventEnd(TSTrajectory_Set, tj, ts, 0, 0));
  if (tj->usehistory) PetscCall(TSHistoryUpdate(tj->tsh, stepnum, time));
  if (tj->lag.caching) tj->lag.Udotcached.time = PETSC_MIN_REAL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSTrajectoryGetNumSteps - Return the number of steps registered in the `TSTrajectory` via `TSTrajectorySet()`.

  Not Collective.

  Input Parameter:
. tj - the trajectory object

  Output Parameter:
. steps - the number of steps

  Level: developer

.seealso: [](chapter_ts), `TS`, `TSTrajectorySet()`
@*/
PetscErrorCode TSTrajectoryGetNumSteps(TSTrajectory tj, PetscInt *steps)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tj, TSTRAJECTORY_CLASSID, 1);
  PetscValidIntPointer(steps, 2);
  PetscCall(TSHistoryGetNumSteps(tj->tsh, steps));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSTrajectoryGet - Updates the solution vector of a time stepper object by querying the `TSTrajectory`

  Collective

  Input Parameters:
+ tj      - the trajectory object
. ts      - the time stepper object
- stepnum - the step number

  Output Parameter:
. time    - the time associated with the step number

  Level: developer

  Note:
  Usually one does not call this routine, it is called automatically during `TSSolve()`

.seealso: [](chapter_ts), `TS`, `TSSolve()`, `TSTrajectorySetUp()`, `TSTrajectoryDestroy()`, `TSTrajectorySetType()`, `TSTrajectorySetVariableNames()`, `TSGetTrajectory()`, `TSTrajectorySet()`, `TSTrajectoryGetVecs()`, `TSGetSolution()`
@*/
PetscErrorCode TSTrajectoryGet(TSTrajectory tj, TS ts, PetscInt stepnum, PetscReal *time)
{
  PetscFunctionBegin;
  PetscCheck(tj, PetscObjectComm((PetscObject)ts), PETSC_ERR_ARG_WRONGSTATE, "TS solver did not save trajectory");
  PetscValidHeaderSpecific(tj, TSTRAJECTORY_CLASSID, 1);
  PetscValidHeaderSpecific(ts, TS_CLASSID, 2);
  PetscValidLogicalCollectiveInt(tj, stepnum, 3);
  PetscValidRealPointer(time, 4);
  PetscCheck(tj->setupcalled, PetscObjectComm((PetscObject)tj), PETSC_ERR_ORDER, "TSTrajectorySetUp should be called first");
  PetscCheck(stepnum >= 0, PetscObjectComm((PetscObject)tj), PETSC_ERR_PLIB, "Requesting negative step number");
  if (tj->monitor) {
    PetscCall(PetscViewerASCIIPrintf(tj->monitor, "TSTrajectoryGet: stepnum %" PetscInt_FMT ", stages %" PetscInt_FMT "\n", stepnum, (PetscInt)!tj->solution_only));
    PetscCall(PetscViewerFlush(tj->monitor));
  }
  PetscCall(PetscLogEventBegin(TSTrajectory_Get, tj, ts, 0, 0));
  PetscUseTypeMethod(tj, get, ts, stepnum, time);
  PetscCall(PetscLogEventEnd(TSTrajectory_Get, tj, ts, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSTrajectoryGetVecs - Reconstructs the vector of state and its time derivative using information from the `TSTrajectory` and, possibly, from the `TS`

  Collective

  Input Parameters:
+ tj      - the trajectory object
. ts      - the time stepper object (optional)
- stepnum - the requested step number

  Output Parameters:
+ time - On input time for the step if step number is `PETSC_DECIDE`, on output the time associated with the step number
. U    - state vector (can be `NULL`)
- Udot - time derivative of state vector (can be `NULL`)

  Level: developer

  Notes:
  If the step number is `PETSC_DECIDE`, the time argument is used to inquire the trajectory.
  If the requested time does not match any in the trajectory, Lagrangian interpolations are returned.

.seealso: [](chapter_ts), `TS`, `TSTrajectory`, `TSTrajectorySetUp()`, `TSTrajectoryDestroy()`, `TSTrajectorySetType()`, `TSTrajectorySetVariableNames()`, `TSGetTrajectory()`, `TSTrajectorySet()`, `TSTrajectoryGet()`
@*/
PetscErrorCode TSTrajectoryGetVecs(TSTrajectory tj, TS ts, PetscInt stepnum, PetscReal *time, Vec U, Vec Udot)
{
  PetscFunctionBegin;
  PetscCheck(tj, PetscObjectComm((PetscObject)ts), PETSC_ERR_ARG_WRONGSTATE, "TS solver did not save trajectory");
  PetscValidHeaderSpecific(tj, TSTRAJECTORY_CLASSID, 1);
  if (ts) PetscValidHeaderSpecific(ts, TS_CLASSID, 2);
  PetscValidLogicalCollectiveInt(tj, stepnum, 3);
  PetscValidRealPointer(time, 4);
  if (U) PetscValidHeaderSpecific(U, VEC_CLASSID, 5);
  if (Udot) PetscValidHeaderSpecific(Udot, VEC_CLASSID, 6);
  if (!U && !Udot) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCheck(tj->setupcalled, PetscObjectComm((PetscObject)tj), PETSC_ERR_ORDER, "TSTrajectorySetUp should be called first");
  PetscCall(PetscLogEventBegin(TSTrajectory_GetVecs, tj, ts, 0, 0));
  if (tj->monitor) {
    PetscInt pU, pUdot;
    pU    = U ? 1 : 0;
    pUdot = Udot ? 1 : 0;
    PetscCall(PetscViewerASCIIPrintf(tj->monitor, "Requested by GetVecs %" PetscInt_FMT " %" PetscInt_FMT ": stepnum %" PetscInt_FMT ", time %g\n", pU, pUdot, stepnum, (double)*time));
    PetscCall(PetscViewerFlush(tj->monitor));
  }
  if (U && tj->lag.caching) {
    PetscObjectId    id;
    PetscObjectState state;

    PetscCall(PetscObjectStateGet((PetscObject)U, &state));
    PetscCall(PetscObjectGetId((PetscObject)U, &id));
    if (stepnum == PETSC_DECIDE) {
      if (id == tj->lag.Ucached.id && *time == tj->lag.Ucached.time && state == tj->lag.Ucached.state) U = NULL;
    } else {
      if (id == tj->lag.Ucached.id && stepnum == tj->lag.Ucached.step && state == tj->lag.Ucached.state) U = NULL;
    }
    if (tj->monitor && !U) {
      PetscCall(PetscViewerASCIIPushTab(tj->monitor));
      PetscCall(PetscViewerASCIIPrintf(tj->monitor, "State vector cached\n"));
      PetscCall(PetscViewerASCIIPopTab(tj->monitor));
      PetscCall(PetscViewerFlush(tj->monitor));
    }
  }
  if (Udot && tj->lag.caching) {
    PetscObjectId    id;
    PetscObjectState state;

    PetscCall(PetscObjectStateGet((PetscObject)Udot, &state));
    PetscCall(PetscObjectGetId((PetscObject)Udot, &id));
    if (stepnum == PETSC_DECIDE) {
      if (id == tj->lag.Udotcached.id && *time == tj->lag.Udotcached.time && state == tj->lag.Udotcached.state) Udot = NULL;
    } else {
      if (id == tj->lag.Udotcached.id && stepnum == tj->lag.Udotcached.step && state == tj->lag.Udotcached.state) Udot = NULL;
    }
    if (tj->monitor && !Udot) {
      PetscCall(PetscViewerASCIIPushTab(tj->monitor));
      PetscCall(PetscViewerASCIIPrintf(tj->monitor, "Derivative vector cached\n"));
      PetscCall(PetscViewerASCIIPopTab(tj->monitor));
      PetscCall(PetscViewerFlush(tj->monitor));
    }
  }
  if (!U && !Udot) {
    PetscCall(PetscLogEventEnd(TSTrajectory_GetVecs, tj, ts, 0, 0));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  if (stepnum == PETSC_DECIDE || Udot) { /* reverse search for requested time in TSHistory */
    if (tj->monitor) PetscCall(PetscViewerASCIIPushTab(tj->monitor));
    /* cached states will be updated in the function */
    PetscCall(TSTrajectoryReconstruct_Private(tj, ts, *time, U, Udot));
    if (tj->monitor) {
      PetscCall(PetscViewerASCIIPopTab(tj->monitor));
      PetscCall(PetscViewerFlush(tj->monitor));
    }
  } else if (U) { /* we were asked to load from stepnum, use TSTrajectoryGet */
    TS  fakets = ts;
    Vec U2;

    /* use a fake TS if ts is missing */
    if (!ts) {
      PetscCall(PetscObjectQuery((PetscObject)tj, "__fake_ts", (PetscObject *)&fakets));
      if (!fakets) {
        PetscCall(TSCreate(PetscObjectComm((PetscObject)tj), &fakets));
        PetscCall(PetscObjectCompose((PetscObject)tj, "__fake_ts", (PetscObject)fakets));
        PetscCall(PetscObjectDereference((PetscObject)fakets));
        PetscCall(VecDuplicate(U, &U2));
        PetscCall(TSSetSolution(fakets, U2));
        PetscCall(PetscObjectDereference((PetscObject)U2));
      }
    }
    PetscCall(TSTrajectoryGet(tj, fakets, stepnum, time));
    PetscCall(TSGetSolution(fakets, &U2));
    PetscCall(VecCopy(U2, U));
    PetscCall(PetscObjectStateGet((PetscObject)U, &tj->lag.Ucached.state));
    PetscCall(PetscObjectGetId((PetscObject)U, &tj->lag.Ucached.id));
    tj->lag.Ucached.time = *time;
    tj->lag.Ucached.step = stepnum;
  }
  PetscCall(PetscLogEventEnd(TSTrajectory_GetVecs, tj, ts, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   TSTrajectoryViewFromOptions - View a `TSTrajectory` based on values in the options database

   Collective

   Input Parameters:
+  A - the `TSTrajectory` context
.  obj - Optional object that provides prefix used for option name
-  name - command line option

   Level: intermediate

.seealso: [](chapter_ts), `TSTrajectory`, `TSTrajectoryView`, `PetscObjectViewFromOptions()`, `TSTrajectoryCreate()`
@*/
PetscErrorCode TSTrajectoryViewFromOptions(TSTrajectory A, PetscObject obj, const char name[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, TSTRAJECTORY_CLASSID, 1);
  PetscCall(PetscObjectViewFromOptions((PetscObject)A, obj, name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
    TSTrajectoryView - Prints information about the trajectory object

    Collective

    Input Parameters:
+   tj - the `TSTrajectory` context obtained from `TSTrajectoryCreate()`
-   viewer - visualization context

    Options Database Key:
.   -ts_trajectory_view - calls `TSTrajectoryView()` at end of `TSAdjointStep()`

    Level: developer

    Notes:
    The available visualization contexts include
+     `PETSC_VIEWER_STDOUT_SELF` - standard output (default)
-     `PETSC_VIEWER_STDOUT_WORLD` - synchronized standard
         output where only the first processor opens
         the file.  All other processors send their
         data to the first processor to print.

    The user can open an alternative visualization context with
    `PetscViewerASCIIOpen()` - output to a specified file.

.seealso: [](chapter_ts), `TS`, `TSTrajectory`, `PetscViewer`, `PetscViewerASCIIOpen()`
@*/
PetscErrorCode TSTrajectoryView(TSTrajectory tj, PetscViewer viewer)
{
  PetscBool iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tj, TSTRAJECTORY_CLASSID, 1);
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)tj), &viewer));
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCheckSameComm(tj, 1, viewer, 2);

  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)tj, viewer));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  total number of recomputations for adjoint calculation = %" PetscInt_FMT "\n", tj->recomps));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  disk checkpoint reads = %" PetscInt_FMT "\n", tj->diskreads));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  disk checkpoint writes = %" PetscInt_FMT "\n", tj->diskwrites));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscTryTypeMethod(tj, view, viewer);
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   TSTrajectorySetVariableNames - Sets the name of each component in the solution vector so that it may be saved with the trajectory

   Collective

   Input Parameters:
+  tr - the trajectory context
-  names - the names of the components, final string must be NULL

   Level: intermediate

   Fortran Note:
   Fortran interface is not possible because of the string array argument

.seealso: [](chapter_ts), `TSTrajectory`, `TSGetTrajectory()`
@*/
PetscErrorCode TSTrajectorySetVariableNames(TSTrajectory ctx, const char *const *names)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ctx, TSTRAJECTORY_CLASSID, 1);
  PetscValidPointer(names, 2);
  PetscCall(PetscStrArrayDestroy(&ctx->names));
  PetscCall(PetscStrArrayallocpy(names, &ctx->names));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   TSTrajectorySetTransform - Solution vector will be transformed by provided function before being saved to disk

   Collective

   Input Parameters:
+  tj - the `TSTrajectory` context
.  transform - the transform function
.  destroy - function to destroy the optional context
-  ctx - optional context used by transform function

   Level: intermediate

.seealso: [](chapter_ts), `TSTrajectorySetVariableNames()`, `TSTrajectory`, `TSMonitorLGSetTransform()`
@*/
PetscErrorCode TSTrajectorySetTransform(TSTrajectory tj, PetscErrorCode (*transform)(void *, Vec, Vec *), PetscErrorCode (*destroy)(void *), void *tctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tj, TSTRAJECTORY_CLASSID, 1);
  tj->transform        = transform;
  tj->transformdestroy = destroy;
  tj->transformctx     = tctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSTrajectoryCreate - This function creates an empty trajectory object used to store the time dependent solution of an ODE/DAE

  Collective

  Input Parameter:
. comm - the communicator

  Output Parameter:
. tj   - the trajectory object

  Level: developer

  Notes:
    Usually one does not call this routine, it is called automatically when one calls `TSSetSaveTrajectory()`.

.seealso: [](chapter_ts), `TS`, `TSTrajectory`, `TSTrajectorySetUp()`, `TSTrajectoryDestroy()`, `TSTrajectorySetType()`, `TSTrajectorySetVariableNames()`, `TSGetTrajectory()`, `TSTrajectorySetKeepFiles()`
@*/
PetscErrorCode TSTrajectoryCreate(MPI_Comm comm, TSTrajectory *tj)
{
  TSTrajectory t;

  PetscFunctionBegin;
  PetscValidPointer(tj, 2);
  *tj = NULL;
  PetscCall(TSInitializePackage());

  PetscCall(PetscHeaderCreate(t, TSTRAJECTORY_CLASSID, "TSTrajectory", "Time stepping", "TS", comm, TSTrajectoryDestroy, TSTrajectoryView));
  t->setupcalled = PETSC_FALSE;
  PetscCall(TSHistoryCreate(comm, &t->tsh));

  t->lag.order            = 1;
  t->lag.L                = NULL;
  t->lag.T                = NULL;
  t->lag.W                = NULL;
  t->lag.WW               = NULL;
  t->lag.TW               = NULL;
  t->lag.TT               = NULL;
  t->lag.caching          = PETSC_TRUE;
  t->lag.Ucached.id       = 0;
  t->lag.Ucached.state    = -1;
  t->lag.Ucached.time     = PETSC_MIN_REAL;
  t->lag.Ucached.step     = PETSC_MAX_INT;
  t->lag.Udotcached.id    = 0;
  t->lag.Udotcached.state = -1;
  t->lag.Udotcached.time  = PETSC_MIN_REAL;
  t->lag.Udotcached.step  = PETSC_MAX_INT;
  t->adjoint_solve_mode   = PETSC_TRUE;
  t->solution_only        = PETSC_FALSE;
  t->keepfiles            = PETSC_FALSE;
  t->usehistory           = PETSC_TRUE;
  *tj                     = t;
  PetscCall(TSTrajectorySetFiletemplate(t, "TS-%06" PetscInt_FMT ".bin"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TSTrajectorySetType - Sets the storage method to be used as in a trajectory

  Collective

  Input Parameters:
+ tj   - the `TSTrajectory` context
. ts   - the `TS` context
- type - a known method

  Options Database Key:
. -ts_trajectory_type <type> - Sets the method; use -help for a list of available methods (for instance, basic)

   Level: developer

  Developer Note:
  Why does this option require access to the `TS`

.seealso: [](chapter_ts), `TSTrajectory`, `TS`, `TSTrajectoryCreate()`, `TSTrajectorySetFromOptions()`, `TSTrajectoryDestroy()`, `TSTrajectoryGetType()`
@*/
PetscErrorCode TSTrajectorySetType(TSTrajectory tj, TS ts, TSTrajectoryType type)
{
  PetscErrorCode (*r)(TSTrajectory, TS);
  PetscBool match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tj, TSTRAJECTORY_CLASSID, 1);
  PetscCall(PetscObjectTypeCompare((PetscObject)tj, type, &match));
  if (match) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscFunctionListFind(TSTrajectoryList, type, &r));
  PetscCheck(r, PETSC_COMM_SELF, PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown TSTrajectory type: %s", type);
  if (tj->ops->destroy) {
    PetscCall((*(tj)->ops->destroy)(tj));

    tj->ops->destroy = NULL;
  }
  PetscCall(PetscMemzero(tj->ops, sizeof(*tj->ops)));

  PetscCall(PetscObjectChangeTypeName((PetscObject)tj, type));
  PetscCall((*r)(tj, ts));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TSTrajectoryGetType - Gets the trajectory type

  Collective

  Input Parameters:
+ tj   - the `TSTrajectory` context
- ts   - the `TS` context

  Output Parameter:
. type - a known method

  Level: developer

.seealso: [](chapter_ts), `TS`, `TSTrajectory`, `TSTrajectoryCreate()`, `TSTrajectorySetFromOptions()`, `TSTrajectoryDestroy()`, `TSTrajectorySetType()`
@*/
PetscErrorCode TSTrajectoryGetType(TSTrajectory tj, TS ts, TSTrajectoryType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tj, TSTRAJECTORY_CLASSID, 1);
  if (type) *type = ((PetscObject)tj)->type_name;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode TSTrajectoryCreate_Basic(TSTrajectory, TS);
PETSC_EXTERN PetscErrorCode TSTrajectoryCreate_Singlefile(TSTrajectory, TS);
PETSC_EXTERN PetscErrorCode TSTrajectoryCreate_Memory(TSTrajectory, TS);
PETSC_EXTERN PetscErrorCode TSTrajectoryCreate_Visualization(TSTrajectory, TS);

/*@C
  TSTrajectoryRegisterAll - Registers all of the `TSTrajectory` storage schecmes in the `TS` package.

  Not Collective

  Level: developer

.seealso: [](chapter_ts), `TSTrajectory`, `TSTrajectoryRegister()`
@*/
PetscErrorCode TSTrajectoryRegisterAll(void)
{
  PetscFunctionBegin;
  if (TSTrajectoryRegisterAllCalled) PetscFunctionReturn(PETSC_SUCCESS);
  TSTrajectoryRegisterAllCalled = PETSC_TRUE;

  PetscCall(TSTrajectoryRegister(TSTRAJECTORYBASIC, TSTrajectoryCreate_Basic));
  PetscCall(TSTrajectoryRegister(TSTRAJECTORYSINGLEFILE, TSTrajectoryCreate_Singlefile));
  PetscCall(TSTrajectoryRegister(TSTRAJECTORYMEMORY, TSTrajectoryCreate_Memory));
  PetscCall(TSTrajectoryRegister(TSTRAJECTORYVISUALIZATION, TSTrajectoryCreate_Visualization));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   TSTrajectoryReset - Resets a trajectory context

   Collective

   Input Parameter:
.  tj - the `TSTrajectory` context obtained from `TSGetTrajectory()`

   Level: developer

.seealso: [](chapter_ts), `TS`, `TSTrajectory`, `TSTrajectoryCreate()`, `TSTrajectorySetUp()`
@*/
PetscErrorCode TSTrajectoryReset(TSTrajectory tj)
{
  PetscFunctionBegin;
  if (!tj) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific(tj, TSTRAJECTORY_CLASSID, 1);
  PetscTryTypeMethod(tj, reset);
  PetscCall(PetscFree(tj->dirfiletemplate));
  PetscCall(TSHistoryDestroy(&tj->tsh));
  PetscCall(TSHistoryCreate(PetscObjectComm((PetscObject)tj), &tj->tsh));
  tj->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   TSTrajectoryDestroy - Destroys a trajectory context

   Collective

   Input Parameter:
.  tj - the `TSTrajectory` context obtained from `TSTrajectoryCreate()`

   Level: developer

.seealso: [](chapter_ts), `TSTrajectory`, `TSTrajectoryCreate()`, `TSTrajectorySetUp()`
@*/
PetscErrorCode TSTrajectoryDestroy(TSTrajectory *tj)
{
  PetscFunctionBegin;
  if (!*tj) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific((*tj), TSTRAJECTORY_CLASSID, 1);
  if (--((PetscObject)(*tj))->refct > 0) {
    *tj = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(TSTrajectoryReset(*tj));
  PetscCall(TSHistoryDestroy(&(*tj)->tsh));
  PetscCall(VecDestroyVecs((*tj)->lag.order + 1, &(*tj)->lag.W));
  PetscCall(PetscFree5((*tj)->lag.L, (*tj)->lag.T, (*tj)->lag.WW, (*tj)->lag.TT, (*tj)->lag.TW));
  PetscCall(VecDestroy(&(*tj)->U));
  PetscCall(VecDestroy(&(*tj)->Udot));

  if ((*tj)->transformdestroy) PetscCall((*(*tj)->transformdestroy)((*tj)->transformctx));
  PetscTryTypeMethod((*tj), destroy);
  if (!((*tj)->keepfiles)) {
    PetscMPIInt rank;
    MPI_Comm    comm;

    PetscCall(PetscObjectGetComm((PetscObject)(*tj), &comm));
    PetscCallMPI(MPI_Comm_rank(comm, &rank));
    if (rank == 0 && (*tj)->dirname) { /* we own the directory, so we run PetscRMTree on it */
      PetscCall(PetscRMTree((*tj)->dirname));
    }
  }
  PetscCall(PetscStrArrayDestroy(&(*tj)->names));
  PetscCall(PetscFree((*tj)->dirname));
  PetscCall(PetscFree((*tj)->filetemplate));
  PetscCall(PetscHeaderDestroy(tj));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  TSTrajectorySetTypeFromOptions_Private - Sets the type of `TSTrajectory` from user options.

  Collective

  Input Parameter:
+ tj - the `TSTrajectory` context
- ts - the TS context

  Options Database Key:
. -ts_trajectory_type <type> - TSTRAJECTORYBASIC, TSTRAJECTORYMEMORY, TSTRAJECTORYSINGLEFILE, TSTRAJECTORYVISUALIZATION

  Level: developer

.seealso: [](chapter_ts), `TSTrajectory`, `TSTrajectoryType`, `TSTrajectorySetFromOptions()`, `TSTrajectorySetType()`
*/
static PetscErrorCode TSTrajectorySetTypeFromOptions_Private(PetscOptionItems *PetscOptionsObject, TSTrajectory tj, TS ts)
{
  PetscBool   opt;
  const char *defaultType;
  char        typeName[256];

  PetscFunctionBegin;
  if (((PetscObject)tj)->type_name) defaultType = ((PetscObject)tj)->type_name;
  else defaultType = TSTRAJECTORYBASIC;

  PetscCall(TSTrajectoryRegisterAll());
  PetscCall(PetscOptionsFList("-ts_trajectory_type", "TSTrajectory method", "TSTrajectorySetType", TSTrajectoryList, defaultType, typeName, 256, &opt));
  if (opt) {
    PetscCall(TSTrajectorySetType(tj, ts, typeName));
  } else {
    PetscCall(TSTrajectorySetType(tj, ts, defaultType));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   TSTrajectorySetUseHistory - Use `TSHistory` in `TSTrajectory`

   Collective

   Input Parameters:
+  tj - the `TSTrajectory` context
-  flg - `PETSC_TRUE` to save, `PETSC_FALSE` to disable

   Options Database Key:
.  -ts_trajectory_use_history - have it use `TSHistory`

   Level: advanced

.seealso: [](chapter_ts), `TSTrajectory`, `TSTrajectoryCreate()`, `TSTrajectoryDestroy()`, `TSTrajectorySetUp()`
@*/
PetscErrorCode TSTrajectorySetUseHistory(TSTrajectory tj, PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tj, TSTRAJECTORY_CLASSID, 1);
  PetscValidLogicalCollectiveBool(tj, flg, 2);
  tj->usehistory = flg;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   TSTrajectorySetMonitor - Monitor the schedules generated by the `TSTrajectory` checkpointing controller

   Collective

   Input Parameters:
+  tj - the `TSTrajectory` context
-  flg - `PETSC_TRUE` to active a monitor, `PETSC_FALSE` to disable

   Options Database Key:
.  -ts_trajectory_monitor - print `TSTrajectory` information

   Level: developer

.seealso: [](chapter_ts), `TSTrajectory`, `TSTrajectoryCreate()`, `TSTrajectoryDestroy()`, `TSTrajectorySetUp()`
@*/
PetscErrorCode TSTrajectorySetMonitor(TSTrajectory tj, PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tj, TSTRAJECTORY_CLASSID, 1);
  PetscValidLogicalCollectiveBool(tj, flg, 2);
  if (flg) tj->monitor = PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)tj));
  else tj->monitor = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   TSTrajectorySetKeepFiles - Keep the files generated by the `TSTrajectory` once the program is done

   Collective

   Input Parameters:
+  tj - the `TSTrajectory` context
-  flg - `PETSC_TRUE` to save, `PETSC_FALSE` to disable

   Options Database Key:
.  -ts_trajectory_keep_files - have it keep the files

   Level: advanced

   Note:
    By default the `TSTrajectory` used for adjoint computations, `TSTRAJECTORYBASIC`, removes the files it generates at the end of the run. This causes the files to be kept.

.seealso: [](chapter_ts), `TSTrajectoryCreate()`, `TSTrajectoryDestroy()`, `TSTrajectorySetUp()`, `TSTrajectorySetMonitor()`
@*/
PetscErrorCode TSTrajectorySetKeepFiles(TSTrajectory tj, PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tj, TSTRAJECTORY_CLASSID, 1);
  PetscValidLogicalCollectiveBool(tj, flg, 2);
  tj->keepfiles = flg;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   TSTrajectorySetDirname - Specify the name of the directory where `TSTrajectory` disk checkpoints are stored.

   Collective

   Input Parameters:
+  tj      - the `TSTrajectory` context
-  dirname - the directory name

   Options Database Key:
.  -ts_trajectory_dirname - set the directory name

   Level: developer

   Notes:
    The final location of the files is determined by dirname/filetemplate where filetemplate was provided by `TSTrajectorySetFiletemplate()`

   If this is not called `TSTrajectory` selects a unique new name for the directory

.seealso: [](chapter_ts), `TSTrajectory`, `TSTrajectorySetFiletemplate()`, `TSTrajectorySetUp()`
@*/
PetscErrorCode TSTrajectorySetDirname(TSTrajectory tj, const char dirname[])
{
  PetscBool flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tj, TSTRAJECTORY_CLASSID, 1);
  PetscCall(PetscStrcmp(tj->dirname, dirname, &flg));
  PetscCheck(flg || !tj->dirfiletemplate, PetscObjectComm((PetscObject)tj), PETSC_ERR_ARG_WRONGSTATE, "Cannot set directoryname after TSTrajectory has been setup");
  PetscCall(PetscFree(tj->dirname));
  PetscCall(PetscStrallocpy(dirname, &tj->dirname));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   TSTrajectorySetFiletemplate - Specify the name template for the files storing `TSTrajectory` checkpoints.

   Collective

   Input Parameters:
+  tj      - the `TSTrajectory` context
-  filetemplate - the template

   Options Database Key:
.  -ts_trajectory_file_template - set the file name template

   Level: developer

   Notes:
    The name template should be of the form, for example filename-%06" PetscInt_FMT ".bin It should not begin with a leading /

   The final location of the files is determined by dirname/filetemplate where dirname was provided by `TSTrajectorySetDirname()`. The %06" PetscInt_FMT " is replaced by the
   timestep counter

.seealso: [](chapter_ts), `TSTrajectory`, `TSTrajectorySetDirname()`, `TSTrajectorySetUp()`
@*/
PetscErrorCode TSTrajectorySetFiletemplate(TSTrajectory tj, const char filetemplate[])
{
  const char *ptr = NULL, *ptr2 = NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tj, TSTRAJECTORY_CLASSID, 1);
  PetscValidCharPointer(filetemplate, 2);
  PetscCheck(!tj->dirfiletemplate, PetscObjectComm((PetscObject)tj), PETSC_ERR_ARG_WRONGSTATE, "Cannot set filetemplate after TSTrajectory has been setup");

  PetscCheck(filetemplate[0], PetscObjectComm((PetscObject)tj), PETSC_ERR_USER, "-ts_trajectory_file_template requires a file name template, e.g. filename-%%06" PetscInt_FMT ".bin");
  /* Do some cursory validation of the input. */
  PetscCall(PetscStrstr(filetemplate, "%", (char **)&ptr));
  PetscCheck(ptr, PetscObjectComm((PetscObject)tj), PETSC_ERR_USER, "-ts_trajectory_file_template requires a file name template, e.g. filename-%%06" PetscInt_FMT ".bin");
  for (ptr++; ptr && *ptr; ptr++) {
    PetscCall(PetscStrchr(PetscInt_FMT "DiouxX", *ptr, (char **)&ptr2));
    PetscCheck(ptr2 || (*ptr >= '0' && *ptr <= '9'), PetscObjectComm((PetscObject)tj), PETSC_ERR_USER, "Invalid file template argument to -ts_trajectory_file_template, should look like filename-%%06" PetscInt_FMT ".bin");
    if (ptr2) break;
  }
  PetscCall(PetscFree(tj->filetemplate));
  PetscCall(PetscStrallocpy(filetemplate, &tj->filetemplate));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   TSTrajectorySetFromOptions - Sets various `TSTrajectory` parameters from user options.

   Collective

   Input Parameters:
+  tj - the `TSTrajectory` context obtained from `TSGetTrajectory()`
-  ts - the `TS` context

   Options Database Keys:
+  -ts_trajectory_type <type> - basic, memory, singlefile, visualization
.  -ts_trajectory_keep_files <true,false> - keep the files generated by the code after the program ends. This is true by default for singlefile and visualization
-  -ts_trajectory_monitor - print `TSTrajectory` information

   Level: developer

   Note:
    This is not normally called directly by users

.seealso: [](chapter_ts), `TSTrajectory`, `TSSetSaveTrajectory()`, `TSTrajectorySetUp()`
@*/
PetscErrorCode TSTrajectorySetFromOptions(TSTrajectory tj, TS ts)
{
  PetscBool set, flg;
  char      dirname[PETSC_MAX_PATH_LEN], filetemplate[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tj, TSTRAJECTORY_CLASSID, 1);
  if (ts) PetscValidHeaderSpecific(ts, TS_CLASSID, 2);
  PetscObjectOptionsBegin((PetscObject)tj);
  PetscCall(TSTrajectorySetTypeFromOptions_Private(PetscOptionsObject, tj, ts));
  PetscCall(PetscOptionsBool("-ts_trajectory_use_history", "Turn on/off usage of TSHistory", NULL, tj->usehistory, &tj->usehistory, NULL));
  PetscCall(PetscOptionsBool("-ts_trajectory_monitor", "Print checkpointing schedules", "TSTrajectorySetMonitor", tj->monitor ? PETSC_TRUE : PETSC_FALSE, &flg, &set));
  if (set) PetscCall(TSTrajectorySetMonitor(tj, flg));
  PetscCall(PetscOptionsInt("-ts_trajectory_reconstruction_order", "Interpolation order for reconstruction", NULL, tj->lag.order, &tj->lag.order, NULL));
  PetscCall(PetscOptionsBool("-ts_trajectory_reconstruction_caching", "Turn on/off caching of TSTrajectoryGetVecs input", NULL, tj->lag.caching, &tj->lag.caching, NULL));
  PetscCall(PetscOptionsBool("-ts_trajectory_adjointmode", "Instruct the trajectory that will be used in a TSAdjointSolve()", NULL, tj->adjoint_solve_mode, &tj->adjoint_solve_mode, NULL));
  PetscCall(PetscOptionsBool("-ts_trajectory_solution_only", "Checkpoint solution only", "TSTrajectorySetSolutionOnly", tj->solution_only, &tj->solution_only, NULL));
  PetscCall(PetscOptionsBool("-ts_trajectory_keep_files", "Keep any trajectory files generated during the run", "TSTrajectorySetKeepFiles", tj->keepfiles, &flg, &set));
  if (set) PetscCall(TSTrajectorySetKeepFiles(tj, flg));

  PetscCall(PetscOptionsString("-ts_trajectory_dirname", "Directory name for TSTrajectory file", "TSTrajectorySetDirname", NULL, dirname, sizeof(dirname) - 14, &set));
  if (set) PetscCall(TSTrajectorySetDirname(tj, dirname));

  PetscCall(PetscOptionsString("-ts_trajectory_file_template", "Template for TSTrajectory file name, use filename-%06" PetscInt_FMT ".bin", "TSTrajectorySetFiletemplate", NULL, filetemplate, sizeof(filetemplate), &set));
  if (set) PetscCall(TSTrajectorySetFiletemplate(tj, filetemplate));

  /* Handle specific TSTrajectory options */
  PetscTryTypeMethod(tj, setfromoptions, PetscOptionsObject);
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   TSTrajectorySetUp - Sets up the internal data structures, e.g. stacks, for the later use
   of a `TS` `TSTrajectory`.

   Collective

   Input Parameters:
+  tj - the `TSTrajectory` context
-  ts - the TS context obtained from `TSCreate()`

   Level: developer

.seealso: [](chapter_ts), `TSTrajectory`, `TSSetSaveTrajectory()`, `TSTrajectoryCreate()`, `TSTrajectoryDestroy()`
@*/
PetscErrorCode TSTrajectorySetUp(TSTrajectory tj, TS ts)
{
  size_t s1, s2;

  PetscFunctionBegin;
  if (!tj) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific(tj, TSTRAJECTORY_CLASSID, 1);
  if (ts) PetscValidHeaderSpecific(ts, TS_CLASSID, 2);
  if (tj->setupcalled) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscLogEventBegin(TSTrajectory_SetUp, tj, ts, 0, 0));
  if (!((PetscObject)tj)->type_name) PetscCall(TSTrajectorySetType(tj, ts, TSTRAJECTORYBASIC));
  PetscTryTypeMethod(tj, setup, ts);

  tj->setupcalled = PETSC_TRUE;

  /* Set the counters to zero */
  tj->recomps    = 0;
  tj->diskreads  = 0;
  tj->diskwrites = 0;
  PetscCall(PetscStrlen(tj->dirname, &s1));
  PetscCall(PetscStrlen(tj->filetemplate, &s2));
  PetscCall(PetscFree(tj->dirfiletemplate));
  PetscCall(PetscMalloc((s1 + s2 + 10) * sizeof(char), &tj->dirfiletemplate));
  PetscCall(PetscSNPrintf(tj->dirfiletemplate, s1 + s2 + 10, "%s/%s", tj->dirname, tj->filetemplate));
  PetscCall(PetscLogEventEnd(TSTrajectory_SetUp, tj, ts, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   TSTrajectorySetSolutionOnly - Tells the trajectory to store just the solution, and not any intermediate stage information

   Collective

   Input Parameters:
+  tj  - the `TSTrajectory` context obtained with `TSGetTrajectory()`
-  flg - the boolean flag

   Level: developer

.seealso: [](chapter_ts), `TSTrajectory`, `TSSetSaveTrajectory()`, `TSTrajectoryCreate()`, `TSTrajectoryDestroy()`, `TSTrajectoryGetSolutionOnly()`
@*/
PetscErrorCode TSTrajectorySetSolutionOnly(TSTrajectory tj, PetscBool solution_only)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tj, TSTRAJECTORY_CLASSID, 1);
  PetscValidLogicalCollectiveBool(tj, solution_only, 2);
  tj->solution_only = solution_only;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   TSTrajectoryGetSolutionOnly - Gets the value set with `TSTrajectorySetSolutionOnly()`.

   Logically Collective

   Input Parameter:
.  tj  - the `TSTrajectory` context

   Output Parameter:
.  flg - the boolean flag

   Level: developer

.seealso: [](chapter_ts), `TSTrajectory`, `TSSetSaveTrajectory()`, `TSTrajectoryCreate()`, `TSTrajectoryDestroy()`, `TSTrajectorySetSolutionOnly()`
@*/
PetscErrorCode TSTrajectoryGetSolutionOnly(TSTrajectory tj, PetscBool *solution_only)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tj, TSTRAJECTORY_CLASSID, 1);
  PetscValidBoolPointer(solution_only, 2);
  *solution_only = tj->solution_only;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   TSTrajectoryGetUpdatedHistoryVecs - Get updated state and time-derivative history vectors.

   Collective

   Input Parameters:
+  tj   - the `TSTrajectory` context
.  ts   - the `TS` solver context
-  time - the requested time

   Output Parameters:
+  U    - state vector at given time (can be interpolated)
-  Udot - time-derivative vector at given time (can be interpolated)

   Level: developer

   Notes:
   The vectors are interpolated if time does not match any time step stored in the `TSTrajectory()`. Pass NULL to not request a vector.

   This function differs from `TSTrajectoryGetVecs()` since the vectors obtained cannot be modified, and they need to be returned by
   calling `TSTrajectoryRestoreUpdatedHistoryVecs()`.

.seealso: [](chapter_ts), `TSTrajectory`, `TSSetSaveTrajectory()`, `TSTrajectoryCreate()`, `TSTrajectoryDestroy()`, `TSTrajectoryRestoreUpdatedHistoryVecs()`, `TSTrajectoryGetVecs()`
@*/
PetscErrorCode TSTrajectoryGetUpdatedHistoryVecs(TSTrajectory tj, TS ts, PetscReal time, Vec *U, Vec *Udot)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tj, TSTRAJECTORY_CLASSID, 1);
  PetscValidHeaderSpecific(ts, TS_CLASSID, 2);
  PetscValidLogicalCollectiveReal(tj, time, 3);
  if (U) PetscValidPointer(U, 4);
  if (Udot) PetscValidPointer(Udot, 5);
  if (U && !tj->U) {
    DM dm;

    PetscCall(TSGetDM(ts, &dm));
    PetscCall(DMCreateGlobalVector(dm, &tj->U));
  }
  if (Udot && !tj->Udot) {
    DM dm;

    PetscCall(TSGetDM(ts, &dm));
    PetscCall(DMCreateGlobalVector(dm, &tj->Udot));
  }
  PetscCall(TSTrajectoryGetVecs(tj, ts, PETSC_DECIDE, &time, U ? tj->U : NULL, Udot ? tj->Udot : NULL));
  if (U) {
    PetscCall(VecLockReadPush(tj->U));
    *U = tj->U;
  }
  if (Udot) {
    PetscCall(VecLockReadPush(tj->Udot));
    *Udot = tj->Udot;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   TSTrajectoryRestoreUpdatedHistoryVecs - Restores updated state and time-derivative history vectors obtained with `TSTrajectoryGetUpdatedHistoryVecs()`.

   Collective

   Input Parameters:
+  tj   - the `TSTrajectory` context
.  U    - state vector at given time (can be interpolated)
-  Udot - time-derivative vector at given time (can be interpolated)

   Level: developer

.seealso: [](chapter_ts), `TSTrajectory`, `TSTrajectoryGetUpdatedHistoryVecs()`
@*/
PetscErrorCode TSTrajectoryRestoreUpdatedHistoryVecs(TSTrajectory tj, Vec *U, Vec *Udot)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tj, TSTRAJECTORY_CLASSID, 1);
  if (U) PetscValidHeaderSpecific(*U, VEC_CLASSID, 2);
  if (Udot) PetscValidHeaderSpecific(*Udot, VEC_CLASSID, 3);
  PetscCheck(!U || *U == tj->U, PetscObjectComm((PetscObject)*U), PETSC_ERR_USER, "U was not obtained from TSTrajectoryGetUpdatedHistoryVecs()");
  PetscCheck(!Udot || *Udot == tj->Udot, PetscObjectComm((PetscObject)*Udot), PETSC_ERR_USER, "Udot was not obtained from TSTrajectoryGetUpdatedHistoryVecs()");
  if (U) {
    PetscCall(VecLockReadPop(tj->U));
    *U = NULL;
  }
  if (Udot) {
    PetscCall(VecLockReadPop(tj->Udot));
    *Udot = NULL;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
