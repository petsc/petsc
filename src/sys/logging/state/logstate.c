#include <petsc/private/logimpl.h> /*I "petsclog.h" I*/

/*@
  PetscLogStateCreate - Create a logging state.

  Not collective

  Output Parameters:
. state - a `PetscLogState`

  Level: developer

  Note:
  Most users will not need to create a `PetscLogState`.  The global state `PetscLogState()`
  is created in `PetscInitialize()`.

.seealso: [](ch_profiling), `PetscLogState`, `PetscLogStateDestroy()`
@*/
PetscErrorCode PetscLogStateCreate(PetscLogState *state)
{
  PetscInt      num_entries, max_events, max_stages;
  PetscLogState s;

  PetscFunctionBegin;
  PetscCall(PetscNew(state));
  s = *state;
  PetscCall(PetscLogRegistryCreate(&s->registry));
  PetscCall(PetscIntStackCreate(&s->stage_stack));
  PetscCall(PetscLogRegistryGetNumEvents(s->registry, NULL, &max_events));
  PetscCall(PetscLogRegistryGetNumStages(s->registry, NULL, &max_stages));

  s->bt_num_events = max_events + 1; // one extra column for default stage activity
  s->bt_num_stages = max_stages;
  num_entries      = s->bt_num_events * s->bt_num_stages;
  PetscCall(PetscBTCreate(num_entries, &s->active));
  s->current_stage = -1;
  s->refct         = 1;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogStateDestroy - Destroy a logging state.

  Not collective

  Input Parameters:
. state - a `PetscLogState`

  Level: developer

  Note:
  Most users will not need to destroy a `PetscLogState`.  The global state `PetscLogState()`
  is destroyed in `PetscFinalize()`.

.seealso: [](ch_profiling), `PetscLogState`, `PetscLogStateCreate()`
@*/
PetscErrorCode PetscLogStateDestroy(PetscLogState *state)
{
  PetscLogState s;

  PetscFunctionBegin;
  s      = *state;
  *state = NULL;
  if (s == NULL || --(s->refct) > 0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscLogRegistryDestroy(s->registry));
  PetscCall(PetscIntStackDestroy(s->stage_stack));
  PetscCall(PetscBTDestroy(&s->active));
  PetscCall(PetscFree(s));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogStateStagePush - Start a new logging stage.

  Not collective

  Input Parameters:
+ state - a `PetscLogState`
- stage - a registered `PetscLogStage`

  Level: developer

  Note:
  This is called for the global state (`PetscLogGetState()`) in `PetscLogStagePush()`.

.seealso: [](ch_profiling), `PetscLogState`, `PetscLogStateStageRegister()`, `PetscLogStateStagePop()`, `PetscLogStateGetCurrentStage()`
@*/
PetscErrorCode PetscLogStateStagePush(PetscLogState state, PetscLogStage stage)
{
  PetscFunctionBegin;
  if (PetscDefined(USE_DEBUG)) {
    PetscInt num_stages;
    PetscCall(PetscLogRegistryGetNumStages(state->registry, &num_stages, NULL));
    PetscCheck(stage >= 0 && stage < num_stages, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid stage %d not in [0,%" PetscInt_FMT ")", stage, num_stages);
  }
  PetscCall(PetscIntStackPush(state->stage_stack, stage));
  state->current_stage = stage;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogStateStagePop - End a running logging stage.

  Not collective

  Input Parameter:
. state - a `PetscLogState`

  Level: developer

  Note:
  This is called for the global state (`PetscLogGetState()`) in `PetscLogStagePush()`.

.seealso: [](ch_profiling), `PetscLogState`, `PetscLogStateStageRegister()`, `PetscLogStateStagePush()`, `PetscLogStateGetCurrentStage()`
@*/
PetscErrorCode PetscLogStateStagePop(PetscLogState state)
{
  int       curStage;
  PetscBool empty;

  PetscFunctionBegin;
  PetscCall(PetscIntStackPop(state->stage_stack, &curStage));
  PetscCall(PetscIntStackEmpty(state->stage_stack, &empty));
  if (!empty) PetscCall(PetscIntStackTop(state->stage_stack, &state->current_stage));
  else state->current_stage = -1;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogStateGetCurrentStage - Get the last stage that was started

  Not collective

  Input Parameter:
. state - a `PetscLogState`

  Output Parameter:
. current - the last `PetscLogStage` started with `PetscLogStateStagePop()`

  Level: developer

  Note:
  This is called for the global state (`PetscLogGetState()`) in `PetscLogGetCurrentStage()`.

.seealso: [](ch_profiling), `PetscLogState`, `PetscLogStateStageRegister()`, `PetscLogStateStagePush()`, `PetscLogStateStagePop()`
@*/
PetscErrorCode PetscLogStateGetCurrentStage(PetscLogState state, PetscLogStage *current)
{
  PetscFunctionBegin;
  *current = state->current_stage;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogStateResize(PetscLogState state)
{
  PetscBT  active_new;
  PetscInt new_num_events;
  PetscInt new_num_stages;

  PetscFunctionBegin;
  PetscCall(PetscLogRegistryGetNumEvents(state->registry, NULL, &new_num_events));
  new_num_events++;
  PetscCall(PetscLogRegistryGetNumStages(state->registry, NULL, &new_num_stages));

  if (state->bt_num_events == new_num_events && state->bt_num_stages == new_num_stages) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCheck((new_num_stages % PETSC_BITS_PER_BYTE) == 0, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "new number of stages must be multiple of %d", PETSC_BITS_PER_BYTE);
  PetscCall(PetscBTCreate(new_num_events * new_num_stages, &active_new));
  if (new_num_stages == state->bt_num_stages) {
    // single memcpy
    size_t num_chars = (state->bt_num_stages * state->bt_num_events) / PETSC_BITS_PER_BYTE;

    PetscCall(PetscMemcpy(active_new, state->active, num_chars));
  } else {
    size_t num_chars_old = state->bt_num_stages / PETSC_BITS_PER_BYTE;
    size_t num_chars_new = new_num_stages / PETSC_BITS_PER_BYTE;

    for (PetscInt i = 0; i < state->bt_num_events; i++) PetscCall(PetscMemcpy(&active_new[i * num_chars_new], &state->active[i * num_chars_old], num_chars_old));
  }
  PetscCall(PetscBTDestroy(&state->active));
  state->active        = active_new;
  state->bt_num_events = new_num_events;
  state->bt_num_stages = new_num_stages;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogStateStageRegister - Register a new stage with a logging state

  Not collective

  Input Parameters:
+ state - a `PetscLogState`
- sname - a unique name

  Output Parameter:
. stage - the identifier for the registered stage

  Level: developer

  Note:
  This is called for the global state (`PetscLogGetState()`) in `PetscLogStageRegister()`.

.seealso: [](ch_profiling), `PetscLogState`, `PetscLogStateStagePush()`, `PetscLogStateStagePop()`
@*/
PetscErrorCode PetscLogStateStageRegister(PetscLogState state, const char sname[], PetscLogStage *stage)
{
  PetscInt s;

  PetscFunctionBegin;
  PetscCall(PetscLogRegistryStageRegister(state->registry, sname, stage));
  PetscCall(PetscLogStateResize(state));
  s = *stage;
  PetscCall(PetscBTSet(state->active, s)); // stages are by default active
  for (PetscInt e = 1; e < state->bt_num_events; e++) {
    // copy "Main Stage" activities
    if (PetscBTLookup(state->active, 0 + e * state->bt_num_stages)) {
      PetscCall(PetscBTSet(state->active, s + e * state->bt_num_stages));
    } else {
      PetscCall(PetscBTClear(state->active, s + e * state->bt_num_stages));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogStateEventRegister - Register a new event with a logging state

  Not collective

  Input Parameters:
+ state - a `PetscLogState`
. sname - a unique name
- id    - the `PetscClassId` for the type of object most closely associated with this event

  Output Parameter:
. event - the identifier for the registered event

  Level: developer

  Note:
  This is called for the global state (`PetscLogGetState()`) in `PetscLogEventRegister()`.

.seealso: [](ch_profiling), `PetscLogState`, `PetscLogStageRegister()`
@*/
PetscErrorCode PetscLogStateEventRegister(PetscLogState state, const char sname[], PetscClassId id, PetscLogEvent *event)
{
  PetscInt e;

  PetscFunctionBegin;
  *event = PETSC_DECIDE;
  PetscCall(PetscLogRegistryGetEventFromName(state->registry, sname, event));
  if (*event > 0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscLogRegistryEventRegister(state->registry, sname, id, event));
  PetscCall(PetscLogStateResize(state));
  e = *event;
  for (PetscInt s = 0; s < state->bt_num_stages; s++) PetscCall(PetscBTSet(state->active, s + (e + 1) * state->bt_num_stages)); // events are by default active
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogStateEventSetCollective - Set the collective nature of a logging event

  Logically collective

  Input Parameters:
+ state      - a `PetscLogState`
. event      - a registered `PetscLogEvent`
- collective - if `PETSC_TRUE`, MPI processes synchronize during this event, and `PetscLogHandlerEventSync()` can be used to help measure the delays between when the processes begin the event

  Level: developer

  Note:
  This is called for the global state (`PetscLogGetState()`) in `PetscLogEventSetCollective()`.

.seealso: [](ch_profiling), `PetscLogState`, `PetscLogEventRegister()`
@*/
PetscErrorCode PetscLogStateEventSetCollective(PetscLogState state, PetscLogEvent event, PetscBool collective)
{
  PetscFunctionBegin;
  PetscCall(PetscLogRegistryEventSetCollective(state->registry, event, collective));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogStateStageSetActive - Mark a stage as active or inactive.

  Not collective

  Input Parameters:
+ state    - a `PetscLogState`
. stage    - a registered `PetscLogStage`
- isActive - if `PETSC_FALSE`, `PetscLogStateEventGetActive()` will return `PETSC_FALSE` for all events during this stage

  Level: developer

  Note:
  This is called for the global state (`PetscLogGetState()`) in `PetscLogStageSetActive()`

.seealso: [](ch_profiling), `PetscLogState`, `PetscLogStateEventSetActive()`
@*/
PetscErrorCode PetscLogStateStageSetActive(PetscLogState state, PetscLogStage stage, PetscBool isActive)
{
  PetscFunctionBegin;
  PetscCheck(stage >= 0 && stage < state->bt_num_stages, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid stage %d should be in [0,%d)", stage, state->bt_num_stages);
  if (isActive) {
    for (PetscInt e = 0; e < state->bt_num_events; e++) PetscCall(PetscBTSet(state->active, stage + e * state->bt_num_stages));
  } else {
    for (PetscInt e = 0; e < state->bt_num_events; e++) PetscCall(PetscBTClear(state->active, stage + e * state->bt_num_stages));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogStateStageGetActive - Check if a logging stage is active or inactive.

  Not collective

  Input Parameters:
+ state - a `PetscLogState`
- stage - a registered `PetscLogStage`

  Output Parameter:
. isActive - if `PETSC_FALSE`, the state should not send logging events to log handlers during this stage.

  Level: developer

  Note:
  This is called for the global state (`PetscLogGetState()`) in `PetscLogStageGetActive()`.

.seealso: [](ch_profiling), `PetscLogState`, `PetscLogStageSetActive()`, `PetscLogHandler`, `PetscLogHandlerStart()`, `PetscLogHandlerEventBegin()`, `PetscLogHandlerEventEnd()`
@*/
PetscErrorCode PetscLogStateStageGetActive(PetscLogState state, PetscLogStage stage, PetscBool *isActive)
{
  PetscFunctionBegin;
  *isActive = PetscBTLookup(state->active, stage) ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogStateEventSetActive - Set a logging event as active or inactive during a logging stage.

  Not collective

  Input Parameters:
+ state    - a `PetscLogState`
. stage    - a registered `PetscLogStage`, or `PETSC_DEFAULT` for the current stage
. event    - a registered `PetscLogEvent`
- isActive - if `PETSC_FALSE`, `PetscLogStateEventGetActive()` will return `PETSC_FALSE` for this stage and this event

  Level: developer

  Note:
  This is called for the global state (`PetscLogGetState()`) in `PetscLogEventActivate()` and `PetscLogEventDeactivate()`.

.seealso: [](ch_profiling), `PetscLogState`, `PetscLogEventGetActive()`, `PetscLogStateGetCurrentStage()`, `PetscLogEventSetActiveAll()`
@*/
PetscErrorCode PetscLogStateEventSetActive(PetscLogState state, PetscLogStage stage, PetscLogEvent event, PetscBool isActive)
{
  PetscFunctionBegin;
  stage = (stage < 0) ? state->current_stage : stage;
  PetscCheck(event >= 0 && event < state->bt_num_events, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid event %d should be in [0,%d)", event, state->bt_num_events);
  PetscCheck(stage >= 0 && stage < state->bt_num_stages, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid stage %d should be in [0,%d)", stage, state->bt_num_stages);
  PetscCall((isActive ? PetscBTSet : PetscBTClear)(state->active, state->current_stage + (event + 1) * state->bt_num_stages));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogStateEventSetActiveAll - Set logging event as active or inactive for all logging stages

  Not collective

  Input Parameters:
+ state    - a `PetscLogState`
. event    - a registered `PetscLogEvent`
- isActive - if `PETSC_FALSE`, `PetscLogStateEventGetActive()` will return `PETSC_FALSE` for all stages and all events

  Level: developer

  Note:
  This is called for the global state (`PetscLogGetState()`) in `PetscLogEventSetActiveAll()`.

.seealso: [](ch_profiling), `PetscLogState`, `PetscLogEventGetActive()`
@*/
PetscErrorCode PetscLogStateEventSetActiveAll(PetscLogState state, PetscLogEvent event, PetscBool isActive)
{
  PetscFunctionBegin;
  PetscCheck(event >= 0 && event < state->bt_num_events, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid event %d should be in [0,%d)", event, state->bt_num_events);
  for (int stage = 0; stage < state->bt_num_stages; stage++) PetscCall((isActive ? PetscBTSet : PetscBTClear)(state->active, state->current_stage + (event + 1) * state->bt_num_stages));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogStateClassSetActive - Set logging events associated with an event as active or inactive during a logging stage.

  Not collective

  Input Parameters:
+ state    - a `PetscLogState`
. stage    - a registered `PetscLogStage`, or `PETSC_DEFAULT` for the current stage
. classid  - a `PetscClassId`
- isActive - if `PETSC_FALSE`, `PetscLogStateEventGetActive()` will return
             `PETSC_FALSE` for this stage and all events that were associated
             with this class when they were registered (see
             `PetscLogStateEventRegister()`).

  Level: developer

  Note:
  This is called for the global state (`PetscLogGetState()`) in `PetscLogEventActivateClass()` and `PetscLogEventDeactivateClass()`.

.seealso: [](ch_profiling), `PetscLogState`, `PetscLogEventGetActive()`, `PetscLogStateEventSetActive()`
@*/
PetscErrorCode PetscLogStateClassSetActive(PetscLogState state, PetscLogStage stage, PetscClassId classid, PetscBool isActive)
{
  PetscInt num_events;

  PetscFunctionBegin;
  stage = stage < 0 ? state->current_stage : stage;
  PetscCheck(stage >= 0 && stage < state->bt_num_stages, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid stage %d should be in [0,%d)", stage, state->bt_num_stages);
  PetscCall(PetscLogRegistryGetNumEvents(state->registry, &num_events, NULL));
  for (PetscLogEvent e = 0; e < num_events; e++) {
    PetscLogEventInfo event_info;
    PetscCall(PetscLogRegistryEventGetInfo(state->registry, e, &event_info));
    if (event_info.classid == classid) PetscCall((isActive ? PetscBTSet : PetscBTClear)(state->active, stage + (e + 1) * state->bt_num_stages));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogStateClassSetActiveAll - Set logging events associated with an event as active or inactive for all logging stages

  Not collective

  Input Parameters:
+ state    - a `PetscLogState`
. classid  - a `PetscClassId`
- isActive - if `PETSC_FALSE`, `PetscLogStateEventGetActive()` will return
             `PETSC_FALSE` for all events that were associated with this class when they
             were registered (see `PetscLogStateEventRegister()`).

  Level: developer

  Note:
  This is called for the global state (`PetscLogGetState()`) in `PetscLogEventIncludeClass()` and `PetscLogEventExcludeClass()`.

.seealso: [](ch_profiling), `PetscLogState`, `PetscLogEventGetActive()`, `PetscLogStateClassSetActive()`
@*/
PetscErrorCode PetscLogStateClassSetActiveAll(PetscLogState state, PetscClassId classid, PetscBool isActive)
{
  PetscInt num_events, num_stages;

  PetscFunctionBegin;
  PetscCall(PetscLogRegistryGetNumEvents(state->registry, &num_events, NULL));
  PetscCall(PetscLogRegistryGetNumStages(state->registry, &num_stages, NULL));
  for (PetscLogEvent e = 0; e < num_events; e++) {
    PetscLogEventInfo event_info;
    PetscCall(PetscLogRegistryEventGetInfo(state->registry, e, &event_info));
    if (event_info.classid == classid) {
      for (PetscLogStage s = 0; s < num_stages; s++) PetscCall((isActive ? PetscBTSet : PetscBTClear)(state->active, s + (e + 1) * state->bt_num_stages));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogStateEventGetActive - Check if a logging event is active or inactive during a logging stage.

  Not collective

  Input Parameters:
+ state - a `PetscLogState`
. stage - a registered `PetscLogStage`, or `PETSC_DEFAULT` for the current stage
- event - a registered `PetscLogEvent`

  Output Parameter:
. isActive - If `PETSC_FALSE`, log handlers should not be notified of the event's beginning or end.

  Level: developer

  Note:
  This is called for the global state (`PetscLogGetState()`) in `PetscLogEventGetActive()`, where it has significance
  for what information is sent to log handlers.

.seealso: [](ch_profiling), `PetscLogState`, `PetscLogEventGetActive()`, `PetscLogStateGetCurrentStage()`, `PetscLogHandler()`
@*/
PetscErrorCode PetscLogStateEventGetActive(PetscLogState state, PetscLogStage stage, PetscLogEvent event, PetscBool *isActive)
{
  PetscFunctionBegin;
  stage = (stage < 0) ? state->current_stage : stage;
  PetscCheck(event >= 0 && event < state->bt_num_events, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid event %d should be in [0,%d)", event, state->bt_num_events);
  PetscCheck(stage >= 0 && stage < state->bt_num_stages, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid stage %d should be in [0,%d)", event, state->bt_num_stages);
  *isActive = PetscLogStateStageEventIsActive(state, stage, event) ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogStateGetEventFromName - Get a `PetscLogEvent` from the name it was registered with.

  Not collective

  Input Parameters:
+ state - a `PetscLogState`
- name  - an event's name

  Output Parameter:
. event - the event's id

  Level: developer

.seealso: [](ch_profiling), `PetscLogState`, `PetscLogStateEventRegister()`, `PetscLogStateEventGetInfo()`
@*/
PetscErrorCode PetscLogStateGetEventFromName(PetscLogState state, const char name[], PetscLogEvent *event)
{
  PetscFunctionBegin;
  PetscCall(PetscLogRegistryGetEventFromName(state->registry, name, event));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogStateGetStageFromName - Get a `PetscLogStage` from the name it was registered with.

  Not collective

  Input Parameters:
+ state - a `PetscLogState`
- name  - a stage's name

  Output Parameter:
. stage - the stage's id

  Level: developer

.seealso: [](ch_profiling), `PetscLogState`, `PetscLogStateStageRegister()`, `PetscLogStateStageGetInfo()`
@*/
PetscErrorCode PetscLogStateGetStageFromName(PetscLogState state, const char name[], PetscLogStage *stage)
{
  PetscFunctionBegin;
  PetscCall(PetscLogRegistryGetStageFromName(state->registry, name, stage));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogStateGetClassFromName - Get a `PetscLogClass` from the name of the class it was registered with.

  Not collective

  Input Parameters:
+ state - a `PetscLogState`
- name  - the name string of the class

  Output Parameter:
. clss - the classes's logging id

  Level: developer

.seealso: [](ch_profiling), `PetscLogState`, `PetscLogStateClassRegister()`, `PetscLogStateClassGetInfo()`
@*/
PetscErrorCode PetscLogStateGetClassFromName(PetscLogState state, const char name[], PetscLogClass *clss)
{
  PetscFunctionBegin;
  PetscCall(PetscLogRegistryGetClassFromName(state->registry, name, clss));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogStateGetClassFromClassId - Get a `PetscLogClass` from the `PetscClassId` it was registered with.

  Not collective

  Input Parameters:
+ state   - a `PetscLogState`
- classid - a `PetscClassId`

  Output Parameter:
. clss - the classes's logging id

  Level: developer

.seealso: [](ch_profiling), `PetscLogState`, `PetscLogStateClassRegister()`, `PetscLogStateClassGetInfo()`
@*/
PetscErrorCode PetscLogStateGetClassFromClassId(PetscLogState state, PetscClassId classid, PetscLogClass *clss)
{
  PetscFunctionBegin;
  PetscCall(PetscLogRegistryGetClassFromClassId(state->registry, classid, clss));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogStateGetNumEvents - Get the number of registered events in a logging state.

  Not collective

  Input Parameter:
. state - a `PetscLogState`

  Output Parameter:
. numEvents - the number of registered `PetscLogEvent`s

  Level: developer

.seealso: [](ch_profiling), `PetscLogState`, `PetscLogStateEventRegister()`
@*/
PetscErrorCode PetscLogStateGetNumEvents(PetscLogState state, PetscInt *numEvents)
{
  PetscFunctionBegin;
  PetscCall(PetscLogRegistryGetNumEvents(state->registry, numEvents, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogStateGetNumStages - Get the number of registered stages in a logging state.

  Not collective

  Input Parameter:
. state - a `PetscLogState`

  Output Parameter:
. numStages - the number of registered `PetscLogStage`s

  Level: developer

.seealso: [](ch_profiling), `PetscLogState`, `PetscLogStateStageRegister()`
@*/
PetscErrorCode PetscLogStateGetNumStages(PetscLogState state, PetscInt *numStages)
{
  PetscFunctionBegin;
  PetscCall(PetscLogRegistryGetNumStages(state->registry, numStages, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogStateGetNumClasses - Get the number of registered classes in a logging state.

  Not collective

  Input Parameter:
. state - a `PetscLogState`

  Output Parameter:
. numClasses - the number of registered `PetscLogClass`s

  Level: developer

.seealso: [](ch_profiling), `PetscLogState`, `PetscLogStateClassRegister()`
@*/
PetscErrorCode PetscLogStateGetNumClasses(PetscLogState state, PetscInt *numClasses)
{
  PetscFunctionBegin;
  PetscCall(PetscLogRegistryGetNumClasses(state->registry, numClasses, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogStateEventGetInfo - Get the registration information of an event

  Not collective

  Input Parameters:
+ state - a `PetscLogState`
- event - a registered `PetscLogEvent`

  Output Parameter:
. info - the `PetscLogEventInfo` of the event will be copied into info

  Level: developer

.seealso: [](ch_profiling), `PetscLogState`, `PetscLogStateEventRegister()`, `PetscLogStateGetEventFromName()`
@*/
PetscErrorCode PetscLogStateEventGetInfo(PetscLogState state, PetscLogEvent event, PetscLogEventInfo *info)
{
  PetscFunctionBegin;
  PetscCall(PetscLogRegistryEventGetInfo(state->registry, event, info));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogStateStageGetInfo - Get the registration information of an stage

  Not collective

  Input Parameters:
+ state - a `PetscLogState`
- stage - a registered `PetscLogStage`

  Output Parameter:
. info - the `PetscLogStageInfo` of the stage will be copied into info

  Level: developer

.seealso: [](ch_profiling), `PetscLogState`, `PetscLogStateStageRegister()`, `PetscLogStateGetStageFromName()`
@*/
PetscErrorCode PetscLogStateStageGetInfo(PetscLogState state, PetscLogStage stage, PetscLogStageInfo *info)
{
  PetscFunctionBegin;
  PetscCall(PetscLogRegistryStageGetInfo(state->registry, stage, info));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogStateClassRegister - Register a class to with a `PetscLogState` used by `PetscLogHandler`s.

  Logically collective on `PETSC_COMM_WORLD`

  Input Parameters:
+ state - a `PetscLogState`
. name  - the name of a class registered with `PetscClassIdRegister()`
- id    - the `PetscClassId` obtained from `PetscClassIdRegister()`

  Output Parameter:
. logclass - a `PetscLogClass` for this class with this state

  Level: developer

  Note:
  Classes are automatically registered with PETSc's global logging state (`PetscLogGetState()`), so this
  is only needed for non-global states.

.seealso: [](ch_profiling), `PetscLogStateClassGetInfo()` `PetscLogStateGetClassFromName()`, `PetscLogStateGetClassFromClassId()`
@*/
PetscErrorCode PetscLogStateClassRegister(PetscLogState state, const char name[], PetscClassId id, PetscLogClass *logclass)
{
  PetscFunctionBegin;
  PetscCall(PetscLogRegistryClassRegister(state->registry, name, id, logclass));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogStateClassGetInfo - Get the registration information of an class

  Not collective

  Input Parameters:
+ state - a `PetscLogState`
- clss  - a registered `PetscLogClass`

  Output Parameter:
. info - the `PetscLogClassInfo` of the class will be copied into info

  Level: developer

.seealso: [](ch_profiling), `PetscLogState`, `PetscLogStateClassRegister()`, `PetscLogStateGetClassFromName()`
@*/
PetscErrorCode PetscLogStateClassGetInfo(PetscLogState state, PetscLogClass clss, PetscLogClassInfo *info)
{
  PetscFunctionBegin;
  PetscCall(PetscLogRegistryClassGetInfo(state->registry, clss, info));
  PetscFunctionReturn(PETSC_SUCCESS);
}
