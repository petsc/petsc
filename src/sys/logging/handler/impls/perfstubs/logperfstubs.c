#include <petsc/private/logimpl.h> /*I "petscsys.h" I*/
#include <petsc/private/loghandlerimpl.h>
#include <../src/sys/perfstubs/timer.h>

typedef struct _n_PetscEventPS {
  void *timer;
  int   depth;
} PetscEventPS;

PETSC_LOG_RESIZABLE_ARRAY(PSArray, PetscEventPS, void *, NULL, NULL, NULL)

typedef struct _n_PetscLogHandler_Perfstubs *PetscLogHandler_Perfstubs;

struct _n_PetscLogHandler_Perfstubs {
  PetscLogPSArray events;
  PetscLogPSArray stages;
  PetscBool       started_perfstubs;
};

static PetscErrorCode PetscLogHandlerContextCreate_Perfstubs(PetscLogHandler_Perfstubs *ps_p)
{
  PetscLogHandler_Perfstubs ps;

  PetscFunctionBegin;
  PetscCall(PetscNew(ps_p));
  ps = *ps_p;
  PetscCall(PetscLogPSArrayCreate(128, &ps->events));
  PetscCall(PetscLogPSArrayCreate(8, &ps->stages));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerDestroy_Perfstubs(PetscLogHandler h)
{
  PetscInt                  num_events, num_stages;
  PetscLogHandler_Perfstubs ps = (PetscLogHandler_Perfstubs)h->data;

  PetscFunctionBegin;
  PetscCall(PetscLogPSArrayGetSize(ps->events, &num_events, NULL));
  for (PetscInt i = 0; i < num_events; i++) {
    PetscEventPS event = {NULL, 0};

    PetscCall(PetscLogPSArrayGet(ps->events, i, &event));
    PetscStackCallExternalVoid("ps_timer_destroy_", ps_timer_destroy_(event.timer));
  }
  PetscCall(PetscLogPSArrayDestroy(&ps->events));

  PetscCall(PetscLogPSArrayGetSize(ps->stages, &num_stages, NULL));
  for (PetscInt i = 0; i < num_stages; i++) {
    PetscEventPS stage = {NULL, 0};

    PetscCall(PetscLogPSArrayGet(ps->stages, i, &stage));
    PetscStackCallExternalVoid("ps_timer_destroy_", ps_timer_destroy_(stage.timer));
  }
  PetscCall(PetscLogPSArrayDestroy(&ps->stages));

  if (ps->started_perfstubs) PetscStackCallExternalVoid("ps_finalize_", ps_finalize_());
  PetscCall(PetscFree(ps));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerPSUpdateEvents(PetscLogHandler h)
{
  PetscLogHandler_Perfstubs ps = (PetscLogHandler_Perfstubs)h->data;
  PetscLogState             state;
  PetscInt                  num_events, num_events_old;

  PetscFunctionBegin;
  PetscCall(PetscLogHandlerGetState(h, &state));
  PetscCall(PetscLogStateGetNumEvents(state, &num_events));
  PetscCall(PetscLogPSArrayGetSize(ps->events, &num_events_old, NULL));
  for (PetscInt i = num_events_old; i < num_events; i++) {
    PetscLogEventInfo event_info = {NULL, -1, PETSC_FALSE};
    PetscEventPS      ps_event   = {NULL, 0};
    PetscLogEvent     ei;

    PetscCall(PetscMPIIntCast(i, &ei));
    PetscCall(PetscLogStateEventGetInfo(state, ei, &event_info));
    PetscStackCallExternalVoid("ps_timer_create_", ps_event.timer = ps_timer_create_(event_info.name));
    ps_event.depth = 0;
    PetscCall(PetscLogPSArrayPush(ps->events, ps_event));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerPSUpdateStages(PetscLogHandler h)
{
  PetscLogHandler_Perfstubs ps = (PetscLogHandler_Perfstubs)h->data;
  PetscLogState             state;
  PetscInt                  num_stages, num_stages_old;

  PetscFunctionBegin;
  PetscCall(PetscLogHandlerGetState(h, &state));
  PetscCall(PetscLogStateGetNumStages(state, &num_stages));
  PetscCall(PetscLogPSArrayGetSize(ps->stages, &num_stages_old, NULL));
  for (PetscInt i = num_stages_old; i < num_stages; i++) {
    PetscLogStageInfo stage_info = {NULL};
    PetscEventPS      ps_stage   = {NULL, 0};
    PetscLogEvent     si;

    PetscCall(PetscMPIIntCast(i, &si));
    PetscCall(PetscLogStateStageGetInfo(state, si, &stage_info));
    PetscStackCallExternalVoid("ps_timer_create_", ps_stage.timer = ps_timer_create_(stage_info.name));
    ps_stage.depth = 0;
    PetscCall(PetscLogPSArrayPush(ps->stages, ps_stage));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerEventBegin_Perfstubs(PetscLogHandler handler, PetscLogEvent event, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  PetscLogHandler_Perfstubs ps       = (PetscLogHandler_Perfstubs)handler->data;
  PetscEventPS              ps_event = {NULL, 0};

  PetscFunctionBegin;
  if (event >= ps->events->num_entries) PetscCall(PetscLogHandlerPSUpdateEvents(handler));
  PetscCall(PetscLogPSArrayGet(ps->events, event, &ps_event));
  ps_event.depth++;
  PetscCall(PetscLogPSArraySet(ps->events, event, ps_event));
  if (ps_event.depth == 1 && ps_event.timer != NULL) PetscStackCallExternalVoid("ps_timer_start_", ps_timer_start_(ps_event.timer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerEventEnd_Perfstubs(PetscLogHandler handler, PetscLogEvent event, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  PetscLogHandler_Perfstubs ps       = (PetscLogHandler_Perfstubs)handler->data;
  PetscEventPS              ps_event = {NULL, 0};

  PetscFunctionBegin;
  if (event >= ps->events->num_entries) PetscCall(PetscLogHandlerPSUpdateEvents(handler));
  PetscCall(PetscLogPSArrayGet(ps->events, event, &ps_event));
  ps_event.depth--;
  PetscCall(PetscLogPSArraySet(ps->events, event, ps_event));
  if (ps_event.depth == 0 && ps_event.timer != NULL) PetscStackCallExternalVoid("ps_timer_stop_", ps_timer_stop_(ps_event.timer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerStagePush_Perfstubs(PetscLogHandler handler, PetscLogStage stage)
{
  PetscLogHandler_Perfstubs ps       = (PetscLogHandler_Perfstubs)handler->data;
  PetscEventPS              ps_event = {NULL, 0};

  PetscFunctionBegin;
  if (stage >= ps->stages->num_entries) PetscCall(PetscLogHandlerPSUpdateStages(handler));
  PetscCall(PetscLogPSArrayGet(ps->stages, stage, &ps_event));
  if (ps_event.timer != NULL) PetscStackCallExternalVoid("ps_timer_start_", ps_timer_start_(ps_event.timer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerStagePop_Perfstubs(PetscLogHandler handler, PetscLogStage stage)
{
  PetscLogHandler_Perfstubs ps       = (PetscLogHandler_Perfstubs)handler->data;
  PetscEventPS              ps_event = {NULL, 0};

  PetscFunctionBegin;
  if (stage >= ps->stages->num_entries) PetscCall(PetscLogHandlerPSUpdateStages(handler));
  PetscCall(PetscLogPSArrayGet(ps->stages, stage, &ps_event));
  if (ps_event.timer != NULL) PetscStackCallExternalVoid("ps_timer_stop_", ps_timer_stop_(ps_event.timer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  PETSCLOGHANDLERPERFSTUBS - PETSCLOGHANDLERPERFSTUBS = "perfstubs" -  A
  `PetscLogHandler` that collects data for the PerfStubs/TAU instrumentation
  library.  A log handler of this type is created and started by
  `PetscLogPerfstubsBegin()`.

  Level: developer

.seealso: [](ch_profiling), `PetscLogHandler`
M*/

PETSC_INTERN PetscErrorCode PetscLogHandlerCreate_Perfstubs(PetscLogHandler handler)
{
  PetscBool                 started_perfstubs;
  PetscLogHandler_Perfstubs lps;

  PetscFunctionBegin;
  if (perfstubs_initialized == PERFSTUBS_UNKNOWN) {
    PetscStackCallExternalVoid("ps_initialize_", ps_initialize_());
    started_perfstubs = PETSC_TRUE;
  } else {
    started_perfstubs = PETSC_FALSE;
  }
  PetscCheck(perfstubs_initialized == PERFSTUBS_SUCCESS, PetscObjectComm((PetscObject)handler), PETSC_ERR_LIB, "perfstubs could not be initialized");
  PetscCall(PetscLogHandlerContextCreate_Perfstubs(&lps));
  lps->started_perfstubs   = started_perfstubs;
  handler->data            = (void *)lps;
  handler->ops->destroy    = PetscLogHandlerDestroy_Perfstubs;
  handler->ops->eventbegin = PetscLogHandlerEventBegin_Perfstubs;
  handler->ops->eventend   = PetscLogHandlerEventEnd_Perfstubs;
  handler->ops->stagepush  = PetscLogHandlerStagePush_Perfstubs;
  handler->ops->stagepop   = PetscLogHandlerStagePop_Perfstubs;
  PetscFunctionReturn(PETSC_SUCCESS);
}
