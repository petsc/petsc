#include <petscviewer.h>
#include <petsc/private/logimpl.h> /*I "petscsys.h" I*/
#include <petsc/private/loghandlerimpl.h>
#include <petsc/private/petscimpl.h>

/*@
  PetscLogHandlerCreate - Create a log handler for profiling events and stages.  PETSc
  provides several implementations of `PetscLogHandler` that interface to different ways to
  summarize or visualize profiling data: see `PetscLogHandlerType` for a list.

  Collective

  Input Parameter:
. comm - the communicator for synchronizing and viewing events with this handler

  Output Parameter:
. handler - the `PetscLogHandler`

  Level: developer

  Notes:
  This does not put the handler in use in PETSc's global logging system: use `PetscLogHandlerStart()` after creation.

  See `PetscLogHandler` for example usage.

.seealso: [](ch_profiling), `PetscLogHandler`, `PetscLogHandlerSetType()`, `PetscLogHandlerStart()`, `PetscLogHandlerStop()`
@*/
PetscErrorCode PetscLogHandlerCreate(MPI_Comm comm, PetscLogHandler *handler)
{
  PetscLogHandler h;

  PetscFunctionBegin;
  *handler = NULL;
  PetscCall(PetscLogHandlerPackageInitialize());
  // We do not use PetscHeaderCreate() here because having PetscLogObjectCreate() run for PetscLogHandler would be very fragile
  PetscCall(PetscNew(&h));
  PetscCall(PetscHeaderCreate_Private((PetscObject)(h), PETSCLOGHANDLER_CLASSID, "PetscLogHandler", "Profile events, stages, and objects", "Profiling", comm, (PetscObjectDestroyFn *)PetscLogHandlerDestroy, (PetscObjectViewFn *)PetscLogHandlerView));
  *handler = h;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogHandlerDestroy - Destroy a `PetscLogHandler`

  Logically collective

  Input Parameter:
. handler - handler to be destroyed

  Level: developer

.seealso: [](ch_profiling), `PetscLogHandler`, `PetscLogHandlerCreate()`
@*/
PetscErrorCode PetscLogHandlerDestroy(PetscLogHandler *handler)
{
  PetscLogHandler h;

  PetscFunctionBegin;
  if (!*handler) PetscFunctionReturn(PETSC_SUCCESS);
  h        = *handler;
  *handler = NULL;
  PetscValidHeaderSpecific(h, PETSCLOGHANDLER_CLASSID, 1);
  if (--((PetscObject)h)->refct > 0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscTryTypeMethod(h, destroy);
  PetscCall(PetscLogStateDestroy(&h->state));
  // We do not use PetscHeaderDestroy() because having PetscLogObjectDestroy() run for PetscLgoHandler would be very fragile
  PetscCall(PetscHeaderDestroy_Private((PetscObject)(h), PETSC_FALSE));
  PetscCall(PetscFree(h));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogHandlerSetState - Set the logging state that provides the stream of events and stages for a log handler.

  Logically collective

  Input Parameters:
+ h     - the `PetscLogHandler`
- state - the `PetscLogState`

  Level: developer

  Note:
  Most users well not need to set a state explicitly: the global logging state (`PetscLogGetState()`) is set when calling `PetscLogHandlerStart()`

.seealso: [](ch_profiling), `PetscLogHandler`, `PetscLogState`, `PetscLogEventBegin()`, `PetscLogHandlerStart()`
@*/
PetscErrorCode PetscLogHandlerSetState(PetscLogHandler h, PetscLogState state)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(h, PETSCLOGHANDLER_CLASSID, 1);
  if (state) {
    PetscAssertPointer(state, 2);
    state->refct++;
  }
  PetscCall(PetscLogStateDestroy(&h->state));
  h->state = state;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogHandlerGetState - Get the logging state that provides the stream of events and stages for a log handler.

  Logically collective

  Input Parameter:
. h - the `PetscLogHandler`

  Output Parameter:
. state - the `PetscLogState`

  Level: developer

  Note:
  For a log handler started with `PetscLogHandlerStart()`, this will be the PETSc global logging state (`PetscLogGetState()`)

.seealso: [](ch_profiling), `PetscLogHandler`, `PetscLogState`, `PetscLogEventBegin()`, `PetscLogHandlerStart()`
@*/
PetscErrorCode PetscLogHandlerGetState(PetscLogHandler h, PetscLogState *state)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(h, PETSCLOGHANDLER_CLASSID, 1);
  PetscAssertPointer(state, 2);
  *state = h->state;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogHandlerEventBegin - Record the beginning of an event in a log handler

  Not collective

  Input Parameters:
+ h  - the `PetscLogHandler`
. e  - a registered `PetscLogEvent`
. o1 - `PetscObject` associated with the event (may be `NULL`)
. o2 - `PetscObject` associated with the event (may be `NULL`)
. o3 - `PetscObject` associated with the event (may be `NULL`)
- o4 - `PetscObject` associated with the event (may be `NULL`)

  Level: developer

  Note:
  Most users will use `PetscLogEventBegin()`, which will call this function for all handlers registered with `PetscLogHandlerStart()`

.seealso: [](ch_profiling), `PetscLogHandler`, `PetscLogEventBegin()`, `PetscLogEventEnd()`, `PetscLogEventSync()`, `PetscLogHandlerEventEnd()`, `PetscLogHandlerEventSync()`
@*/
PetscErrorCode PetscLogHandlerEventBegin(PetscLogHandler h, PetscLogEvent e, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(h, PETSCLOGHANDLER_CLASSID, 1);
  PetscTryTypeMethod(h, eventbegin, e, o1, o2, o3, o4);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogHandlerEventEnd - Record the end of an event in a log handler

  Not collective

  Input Parameters:
+ h  - the `PetscLogHandler`
. e  - a registered `PetscLogEvent`
. o1 - `PetscObject` associated with the event (may be `NULL`)
. o2 - `PetscObject` associated with the event (may be `NULL`)
. o3 - `PetscObject` associated with the event (may be `NULL`)
- o4 - `PetscObject` associated with the event (may be `NULL`)

  Level: developer

  Note:
  Most users will use `PetscLogEventEnd()`, which will call this function for all handlers registered with `PetscLogHandlerStart()`

.seealso: [](ch_profiling), `PetscLogHandler`, `PetscLogEventBegin()`, `PetscLogEventEnd()`, `PetscLogEventSync()`, `PetscLogHandlerEventBegin()`, `PetscLogHandlerEventSync()`
@*/
PetscErrorCode PetscLogHandlerEventEnd(PetscLogHandler h, PetscLogEvent e, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(h, PETSCLOGHANDLER_CLASSID, 1);
  PetscTryTypeMethod(h, eventend, e, o1, o2, o3, o4);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogHandlerEventSync - Synchronize a logging event

  Collective

  Input Parameters:
+ h    - the `PetscLogHandler`
. e    - a registered `PetscLogEvent`
- comm - the communicator over which to synchronize `e`

  Level: developer

  Note:
  Most users will use `PetscLogEventSync()`, which will call this function for all handlers registered with `PetscLogHandlerStart()`

.seealso: [](ch_profiling), `PetscLogHandler`, `PetscLogEventBegin()`, `PetscLogEventEnd()`, `PetscLogEventSync()`, `PetscLogHandlerEventBegin()`, `PetscLogHandlerEventEnd()`
@*/
PetscErrorCode PetscLogHandlerEventSync(PetscLogHandler h, PetscLogEvent e, MPI_Comm comm)
{
  MPI_Comm    h_comm;
  PetscMPIInt size;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(h, PETSCLOGHANDLER_CLASSID, 1);
  PetscCall(PetscObjectGetComm((PetscObject)h, &h_comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  if (comm == MPI_COMM_NULL || size == 1) PetscFunctionReturn(PETSC_SUCCESS); // nothing to sync
  if (PetscDefined(USE_DEBUG)) {
    PetscMPIInt h_comm_world, compare;
    PetscCallMPI(MPI_Comm_compare(h_comm, PETSC_COMM_WORLD, &h_comm_world));
    PetscCallMPI(MPI_Comm_compare(h_comm, comm, &compare));
    // only synchronze if h->comm and comm have the same processes or h->comm is PETSC_COMM_WORLD
    PetscCheck(h_comm_world != MPI_UNEQUAL || compare != MPI_UNEQUAL, comm, PETSC_ERR_SUP, "PetscLogHandlerSync does not support arbitrary mismatched communicators");
  }
  PetscTryTypeMethod(h, eventsync, e, comm);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogHandlerObjectCreate - Record the creation of an object in a log handler.

  Not collective

  Input Parameters:
+ h   - the `PetscLogHandler`
- obj - a newly created `PetscObject`

  Level: developer

  Notes:
  Most users will use `PetscLogObjectCreate()`, which will call this function for all handlers registered with `PetscLogHandlerStart()`.

.seealso: [](ch_profiling), `PetscLogHandler`, `PetscLogObjectCreate()`, `PetscLogObjectDestroy()`, `PetscLogHandlerObjectDestroy()`
@*/
PetscErrorCode PetscLogHandlerObjectCreate(PetscLogHandler h, PetscObject obj)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(h, PETSCLOGHANDLER_CLASSID, 1);
  PetscTryTypeMethod(h, objectcreate, obj);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogHandlerObjectDestroy - Record the destruction of an object in a log handler.

  Not collective

  Input Parameters:
+ h   - the `PetscLogHandler`
- obj - a newly created `PetscObject`

  Level: developer

  Notes:
  Most users will use `PetscLogObjectDestroy()`, which will call this function for all handlers registered with `PetscLogHandlerStart()`.

.seealso: [](ch_profiling), `PetscLogHandler`, `PetscLogObjectCreate()`, `PetscLogObjectDestroy()`, `PetscLogHandlerObjectCreate()`
@*/
PetscErrorCode PetscLogHandlerObjectDestroy(PetscLogHandler h, PetscObject obj)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(h, PETSCLOGHANDLER_CLASSID, 1);
  PetscTryTypeMethod(h, objectdestroy, obj);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogHandlerStagePush - Begin a new logging stage in a log handler.

  Not collective

  Input Parameters:
+ h     - the `PetscLogHandler`
- stage - a registered `PetscLogStage`

  Level: developer

  Notes:
  Most users will use `PetscLogStagePush()`, which will call this function for all handlers registered with `PetscLogHandlerStart()`.

  This function is called right before the stage is pushed for the handler's `PetscLogState`, so `PetscLogStateGetCurrentStage()`
  can be used to see what the previous stage was.

.seealso: [](ch_profiling), `PetscLogHandler`, `PetscLogStagePush()`, `PetscLogStagePop()`, `PetscLogHandlerStagePop()`
@*/
PetscErrorCode PetscLogHandlerStagePush(PetscLogHandler h, PetscLogStage stage)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(h, PETSCLOGHANDLER_CLASSID, 1);
  PetscTryTypeMethod(h, stagepush, stage);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogHandlerStagePop - End the current logging stage in a log handler.

  Not collective

  Input Parameters:
+ h     - the `PetscLogHandler`
- stage - a registered `PetscLogStage`

  Level: developer

  Notes:
  Most users will use `PetscLogStagePop()`, which will call this function for all handlers registered with `PetscLogHandlerStart()`.

  This function is called right after the stage is popped for the handler's `PetscLogState`, so `PetscLogStateGetCurrentStage()`
  can be used to see what the next stage will be.

.seealso: [](ch_profiling), `PetscLogHandler`, `PetscLogStagePush()`, `PetscLogStagePop()`, `PetscLogHandlerStagePush()`
@*/
PetscErrorCode PetscLogHandlerStagePop(PetscLogHandler h, PetscLogStage stage)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(h, PETSCLOGHANDLER_CLASSID, 1);
  PetscTryTypeMethod(h, stagepop, stage);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogHandlerView - View the data recorded in a log handler.

  Collective

  Input Parameters:
+ h      - the `PetscLogHandler`
- viewer - the `PetscViewer`

  Level: developer

.seealso: [](ch_profiling), `PetscLogHandler`, `PetscLogView()`
@*/
PetscErrorCode PetscLogHandlerView(PetscLogHandler h, PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(h, PETSCLOGHANDLER_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscTryTypeMethod(h, view, viewer);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscLogHandlerGetEventPerfInfo - Get a direct reference to the `PetscEventPerfInfo` of a stage and event

  Not collective, No Fortran Support

  Input Parameters:
+ handler - a `PetscLogHandler`
. stage   - a `PetscLogStage` (or `PETSC_DEFAULT` for the current stage)
- event   - a `PetscLogEvent`

  Output Parameter:
. event_info - a pointer to a performance log for `event` during `stage` (or `NULL` if this handler does not use
              `PetscEventPerfInfo` to record performance data); writing to `event_info` will change the record in
              `handler`

  Level: developer

.seealso: [](ch_profiling), `PetscLogEventGetPerfInfo()`, `PETSCLOGHANDLERDEFAULT`
@*/
PetscErrorCode PetscLogHandlerGetEventPerfInfo(PetscLogHandler handler, PetscLogStage stage, PetscLogEvent event, PetscEventPerfInfo **event_info)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(handler, PETSCLOGHANDLER_CLASSID, 1);
  PetscAssertPointer(event_info, 4);
  *event_info = NULL;
  PetscTryMethod(handler, "PetscLogHandlerGetEventPerfInfo_C", (PetscLogHandler, PetscLogStage, PetscLogEvent, PetscEventPerfInfo **), (handler, stage, event, event_info));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscLogHandlerGetStagePerfInfo - Get a direct reference to the `PetscEventPerfInfo` of a stage

  Not collective, No Fortran Support

  Input Parameters:
+ handler - a `PetscLogHandler`
- stage   - a `PetscLogStage` (or `PETSC_DEFAULT` for the current stage)

  Output Parameter:
. stage_info - a pointer to a performance log for `stage` (or `NULL` if this handler does not use `PetscEventPerfInfo`
               to record performance data); writing to `stage_info` will change the record in `handler`

  Level: developer

.seealso: [](ch_profiling), `PetscLogEventGetPerfInfo()`, `PETSCLOGHANDLERDEFAULT`
@*/
PetscErrorCode PetscLogHandlerGetStagePerfInfo(PetscLogHandler handler, PetscLogStage stage, PetscEventPerfInfo **stage_info)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(handler, PETSCLOGHANDLER_CLASSID, 1);
  PetscAssertPointer(stage_info, 3);
  *stage_info = NULL;
  PetscTryMethod(handler, "PetscLogHandlerGetStagePerfInfo_C", (PetscLogHandler, PetscLogStage, PetscEventPerfInfo **), (handler, stage, stage_info));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogHandlerSetLogActions - Determines whether actions are logged for a log handler.

  Not Collective

  Input Parameters:
+ handler - a `PetscLogHandler`
- flag    - `PETSC_TRUE` if actions are to be logged (ignored if `handler` does not log actions)

  Level: developer

  Notes:
  The default log handler `PETSCLOGHANDLERDEFAULT` implements this function, but others generally do not.  You can use
  `PetscLogSetLogActions()` to call this function for the default log handler that is connected to the global
  logging state (`PetscLogGetState()`).

  Logging of actions continues to consume more memory as the program runs. Long running programs should consider
  turning this feature off.

.seealso: [](ch_profiling), `PetscLogSetLogActions()`, `PetscLogStagePush()`, `PetscLogStagePop()`, `PetscLogGetDefaultHandler()`
@*/
PetscErrorCode PetscLogHandlerSetLogActions(PetscLogHandler handler, PetscBool flag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(handler, PETSCLOGHANDLER_CLASSID, 1);
  PetscTryMethod(handler, "PetscLogHandlerSetLogActions_C", (PetscLogHandler, PetscBool), (handler, flag));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogHandlerSetLogObjects - Determines whether objects are logged for a log handler.

  Not Collective

  Input Parameters:
+ handler - a `PetscLogHandler`
- flag    - `PETSC_TRUE` if objects are to be logged (ignored if `handler` does not log objects)

  Level: developer

  Notes:
  The default log handler `PETSCLOGHANDLERDEFAULT` implements this function, but others generally do not.  You can use
  `PetscLogSetLogObjects()` to call this function for the default log handler that is connected to the global
  logging state (`PetscLogGetState()`).

  Logging of objects continues to consume more memory as the program runs. Long running programs should consider
  turning this feature off.

.seealso: [](ch_profiling), `PetscLogSetLogObjects()`, `PetscLogStagePush()`, `PetscLogStagePop()`, `PetscLogGetDefaultHandler()`
@*/
PetscErrorCode PetscLogHandlerSetLogObjects(PetscLogHandler handler, PetscBool flag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(handler, PETSCLOGHANDLER_CLASSID, 1);
  PetscTryMethod(handler, "PetscLogHandlerSetLogObjects_C", (PetscLogHandler, PetscBool), (handler, flag));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscLogHandlerLogObjectState_Internal(PetscLogHandler handler, PetscObject obj, const char format[], va_list argp)
{
  PetscFunctionBegin;
  PetscTryMethod(handler, "PetscLogHandlerLogObjectState_C", (PetscLogHandler, PetscObject, const char *, va_list), (handler, obj, format, argp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscLogHandlerLogObjectState - Record information about an object with the default log handler

  Not Collective, No Fortran Support

  Input Parameters:
+ handler - a `PetscLogHandler`
. obj     - the `PetscObject`
. format  - a printf-style format string
- ...     - printf arguments to format

  Level: developer

  Note:
  The default log handler `PETSCLOGHANDLERDEFAULT` implements this function, but others generally do not.  You can use
  `PetscLogObjectState()` to call this function for the default log handler that is connected to the global
  logging state (`PetscLogGetState()`).

.seealso: [](ch_profiling), `PetscLogObjectState`, `PetscLogObjectCreate()`, `PetscLogObjectDestroy()`, `PetscLogGetDefaultHandler()`
@*/
PetscErrorCode PetscLogHandlerLogObjectState(PetscLogHandler handler, PetscObject obj, const char format[], ...)
{
  va_list argp;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(handler, PETSCLOGHANDLER_CLASSID, 1);
  PetscValidHeader(obj, 2);
  PetscAssertPointer(format, 3);
  va_start(argp, format);
  PetscCall(PetscLogHandlerLogObjectState_Internal(handler, obj, format, argp));
  va_end(argp);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogHandlerGetNumObjects - Get the number of objects that were logged with a log handler

  Not Collective

  Input Parameter:
. handler - a `PetscLogHandler`

  Output Parameter:
. num_objects - the number of objects whose creations and destructions were logged with `handler`
                (`PetscLogHandlerObjectCreate()` / `PetscLogHandlerObjectDestroy()`), or -1
                if the handler does not keep track of this number.

  Level: developer

  Note:
  The default log handler `PETSCLOGHANDLERDEFAULT` implements this function, but others generally do not.

.seealso: [](ch_profiling)
@*/
PetscErrorCode PetscLogHandlerGetNumObjects(PetscLogHandler handler, PetscInt *num_objects)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(handler, PETSCLOGHANDLER_CLASSID, 1);
  PetscAssertPointer(num_objects, 2);
  PetscTryMethod(handler, "PetscLogHandlerGetNumObjects_C", (PetscLogHandler, PetscInt *), (handler, num_objects));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogHandlerEventDeactivatePush - Temporarily deactivate a logging event for a log handler

  Not collective

  Input Parameters:
+ handler - a `PetscLogHandler`
. stage   - a `PetscLogStage` (or `PETSC_DEFAULT` for the current stage)
- event   - a `PetscLogEvent`

  Level: developer

  Note:
  The default log handler `PETSCLOGHANDLERDEFAULT` implements this function, but others generally do not.  You can use
  `PetscLogEventDeactivatePush()` to call this function for the default log handler that is connected to the global
  logging state (`PetscLogGetState()`).

.seealso: [](ch_profiling), `PetscLogHandlerEventDeactivatePop()`
@*/
PetscErrorCode PetscLogHandlerEventDeactivatePush(PetscLogHandler handler, PetscLogStage stage, PetscLogEvent event)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(handler, PETSCLOGHANDLER_CLASSID, 1);
  PetscTryMethod(handler, "PetscLogHandlerEventDeactivatePush_C", (PetscLogHandler, PetscLogStage, PetscLogEvent), (handler, stage, event));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogHandlerEventDeactivatePop - Undo temporary deactivation a logging event for a log handler

  Not collective

  Input Parameters:
+ handler - a `PetscLogHandler`
. stage   - a `PetscLogStage` (or `PETSC_DEFAULT` for the current stage)
- event   - a `PetscLogEvent`

  Level: developer

  Note:
  The default log handler `PETSCLOGHANDLERDEFAULT` implements this function, but others generally do not.  You can use
  `PetscLogEventDeactivatePop()` to call this function for the default log handler that is connected to the global
  logging state (`PetscLogGetState()`).

.seealso: [](ch_profiling), `PetscLogHandlerEventDeactivatePush()`
@*/
PetscErrorCode PetscLogHandlerEventDeactivatePop(PetscLogHandler handler, PetscLogStage stage, PetscLogEvent event)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(handler, PETSCLOGHANDLER_CLASSID, 1);
  PetscTryMethod(handler, "PetscLogHandlerEventDeactivatePop_C", (PetscLogHandler, PetscLogStage, PetscLogEvent), (handler, stage, event));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogHandlerEventsPause - Put event logging into "paused" mode (see `PetscLogEventsPause()` for details.) for a log handler

  Not collective

  Input Parameter:
. handler - a `PetscLogHandler`

  Level: developer

  Note:
  The default log handler `PETSCLOGHANDLERDEFAULT` implements this function, but others generally do not.  You can use
  `PetscLogEventsPause()` to call this function for the default log handler that is connected to the global
  logging state (`PetscLogGetState()`).

.seealso: [](ch_profiling), `PetscLogHandlerEventsResume()`
@*/
PetscErrorCode PetscLogHandlerEventsPause(PetscLogHandler handler)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(handler, PETSCLOGHANDLER_CLASSID, 1);
  PetscTryMethod(handler, "PetscLogHandlerEventsPause_C", (PetscLogHandler), (handler));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogHandlerEventsResume - Resume event logging that had been put into "paused" mode (see `PetscLogEventsPause()` for details.) for a log handler

  Not collective

  Input Parameter:
. handler - a `PetscLogHandler`

  Level: developer

  Note:
  The default log handler `PETSCLOGHANDLERDEFAULT` implements this function, but others generally do not.  You can use
  `PetscLogEventsResume()` to call this function for the default log handler that is connected to the global
  logging state (`PetscLogGetState()`).

.seealso: [](ch_profiling), `PetscLogHandlerEventsPause()`
@*/
PetscErrorCode PetscLogHandlerEventsResume(PetscLogHandler handler)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(handler, PETSCLOGHANDLER_CLASSID, 1);
  PetscTryMethod(handler, "PetscLogHandlerEventsResume_C", (PetscLogHandler), (handler));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogHandlerDump - Dump the records of a log handler to file

  Not collective

  Input Parameters:
+ handler - a `PetscLogHandler`
- sname   - the name of the file to dump log data to

  Level: developer

  Note:
  The default log handler `PETSCLOGHANDLERDEFAULT` implements this function, but others generally do not.  You can use
  `PetscLogDump()` to call this function for the default log handler that is connected to the global
  logging state (`PetscLogGetState()`).

.seealso: [](ch_profiling)
@*/
PetscErrorCode PetscLogHandlerDump(PetscLogHandler handler, const char sname[])
{
  PetscFunctionBegin;
  PetscTryMethod(handler, "PetscLogHandlerDump_C", (PetscLogHandler, const char *), (handler, sname));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogHandlerStageSetVisible - Set the visibility of logging stage in `PetscLogHandlerView()` for a log handler

  Not collective

  Input Parameters:
+ handler   - a `PetscLogHandler`
. stage     - a `PetscLogStage`
- isVisible - the visibility flag, `PETSC_TRUE` to print, else `PETSC_FALSE` (defaults to `PETSC_TRUE`)

  Level: developer

  Note:
  The default log handler `PETSCLOGHANDLERDEFAULT` implements this function, but others generally do not.  You can use
  `PetscLogStageSetVisible()` to call this function for the default log handler that is connected to the global
  logging state (`PetscLogGetState()`).

.seealso: [](ch_profiling), `PetscLogHandlerStageGetVisible()`
@*/
PetscErrorCode PetscLogHandlerStageSetVisible(PetscLogHandler handler, PetscLogStage stage, PetscBool isVisible)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(handler, PETSCLOGHANDLER_CLASSID, 1);
  PetscTryMethod(handler, "PetscLogHandlerStageSetVisible_C", (PetscLogHandler, PetscLogStage, PetscBool), (handler, stage, isVisible));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogHandlerStageGetVisible - Get the visibility of logging stage in `PetscLogHandlerView()` for a log handler

  Not collective

  Input Parameters:
+ handler - a `PetscLogHandler`
- stage   - a `PetscLogStage`

  Output Parameter:
. isVisible - the visibility flag, `PETSC_TRUE` to print, else `PETSC_FALSE` (defaults to `PETSC_TRUE`)

  Level: developer

  Note:
  The default log handler `PETSCLOGHANDLERDEFAULT` implements this function, but others generally do not.  You can use
  `PetscLogStageGetVisible()` to call this function for the default log handler that is connected to the global
  logging state (`PetscLogGetState()`).

.seealso: [](ch_profiling), `PetscLogHandlerStageSetVisible()`
@*/
PetscErrorCode PetscLogHandlerStageGetVisible(PetscLogHandler handler, PetscLogStage stage, PetscBool *isVisible)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(handler, PETSCLOGHANDLER_CLASSID, 1);
  PetscTryMethod(handler, "PetscLogHandlerStageGetVisible_C", (PetscLogHandler, PetscLogStage, PetscBool *), (handler, stage, isVisible));
  PetscFunctionReturn(PETSC_SUCCESS);
}
