const char help[] = "How to create a log handler using the PetscLogHandler interface";

#include <petscsys.h>
#include <petsc/private/hashmapi.h> // use PetscHMapI: a PetscInt -> PetscInt hashmap
#include <petsctime.h>              // use PetscTimeSubtract() and PetscTimeAdd()
#include <petscviewer.h>
#include <petsc/private/loghandlerimpl.h> // use the struct _p_PetscLogHandler behind PetscLogHandler

/* Log handlers that use the PetscLogHandler interface get their information
   from the PetscLogState available to each handler and the user-defined
   context pointer.  Compare this example to src/sys/tutorials/ex6.c.

   A logging event can be started multiple times before it stops: for example,
   a linear solve may involve a subsolver, so PetscLogEventBegin() can be
   called for the event KSP_Solve multiple times before a call to
   PetscLogEventEnd().  The user defined handler in this example shows how many
   times an event is running. */

#define PETSCLOGHANDLEREX7 "ex7"

typedef struct _HandlerCtx *HandlerCtx;

struct _HandlerCtx {
  PetscHMapI running;
  PetscInt   num_objects_created;
  PetscInt   num_objects_destroyed;
};

static PetscErrorCode HandlerCtxCreate(HandlerCtx *ctx_p)
{
  HandlerCtx ctx;

  PetscFunctionBegin;
  PetscCall(PetscNew(ctx_p));
  ctx = *ctx_p;
  PetscCall(PetscHMapICreate(&ctx->running));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode HandlerCtxDestroy(HandlerCtx *ctx_p)
{
  HandlerCtx ctx;

  PetscFunctionBegin;
  ctx    = *ctx_p;
  *ctx_p = NULL;
  PetscCall(PetscHMapIDestroy(&ctx->running));
  PetscCall(PetscFree(ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#define PrintData(format_string, ...) \
  do { \
    PetscMPIInt    _rank; \
    PetscLogDouble _time; \
    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &_rank)); \
    PetscCall(PetscTime(&_time)); \
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "[%d:%-7g:%-33s] " format_string, _rank, _time, PETSC_FUNCTION_NAME, __VA_ARGS__)); \
  } while (0)

static PetscErrorCode PetscLogHandlerEventBegin_Ex7(PetscLogHandler h, PetscLogEvent e, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  HandlerCtx        ctx;
  PetscInt          count;
  PetscLogState     state;
  PetscLogEventInfo event_info;
  PetscBool         is_active;

  PetscFunctionBegin;
  // This callback will only be invoked if the event is active
  PetscCall(PetscLogHandlerGetState(h, &state));
  PetscCall(PetscLogStateEventGetActive(state, PETSC_DEFAULT, e, &is_active));
  PetscAssert(is_active, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Event handler called for inactive event");

  ctx = (HandlerCtx)h->data;
  PetscCall(PetscHMapIGetWithDefault(ctx->running, (PetscInt)e, 0, &count));
  count += 1;
  PetscCall(PetscLogStateEventGetInfo(state, e, &event_info));
  PrintData("Event \"%s\" started: now running %" PetscInt_FMT " times\n", event_info.name, count);
  PetscCall(PetscHMapISet(ctx->running, (PetscInt)e, count));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerEventEnd_Ex7(PetscLogHandler h, PetscLogEvent e, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  HandlerCtx        ctx;
  PetscInt          count;
  PetscLogState     state;
  PetscLogEventInfo event_info;

  PetscFunctionBegin;
  ctx = (HandlerCtx)h->data;
  PetscCall(PetscLogHandlerGetState(h, &state));
  PetscCall(PetscHMapIGetWithDefault(ctx->running, (PetscInt)e, 0, &count));
  count -= 1;
  PetscCall(PetscLogStateEventGetInfo(state, e, &event_info));
  PrintData("Event \"%s\" stopped: now running %" PetscInt_FMT " times\n", event_info.name, count);
  PetscCall(PetscHMapISet(ctx->running, (PetscInt)e, count));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerEventSync_Ex7(PetscLogHandler h, PetscLogEvent e, MPI_Comm comm)
{
  PetscLogState     state;
  PetscLogEventInfo event_info;
  PetscLogDouble    time = 0.0;

  PetscFunctionBegin;
  PetscCall(PetscTimeSubtract(&time));
  PetscCallMPI(MPI_Barrier(comm));
  PetscCall(PetscTimeAdd(&time));
  PetscCall(PetscLogHandlerGetState(h, &state));
  PetscCall(PetscLogStateEventGetInfo(state, e, &event_info));
  PrintData("Event \"%s\" synced: took %g seconds\n", event_info.name, (double)time);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerObjectCreate_Ex7(PetscLogHandler h, PetscObject obj)
{
  HandlerCtx ctx;

  PetscFunctionBegin;
  ctx = (HandlerCtx)h->data;
  ctx->num_objects_created++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerObjectDestroy_Ex7(PetscLogHandler h, PetscObject obj)
{
  HandlerCtx ctx;

  PetscFunctionBegin;
  ctx = (HandlerCtx)h->data;
  ctx->num_objects_destroyed++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerStagePush_Ex7(PetscLogHandler h, PetscLogStage new_stage)
{
  PetscLogStage     old_stage;
  PetscLogStageInfo new_info;
  PetscLogState     state;

  PetscFunctionBegin;
  PetscCall(PetscLogHandlerGetState(h, &state));
  PetscCall(PetscLogStateStageGetInfo(state, new_stage, &new_info));
  PetscCall(PetscLogStateGetCurrentStage(state, &old_stage));
  if (old_stage >= 0) {
    PetscLogStageInfo old_info;
    PetscCall(PetscLogStateStageGetInfo(state, old_stage, &old_info));
    PrintData("Pushing stage stage \"%s\" (replacing \"%s\")\n", new_info.name, old_info.name);
  } else {
    PrintData("Pushing initial stage \"%s\"\n", new_info.name);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerStagePop_Ex7(PetscLogHandler h, PetscLogStage old_stage)
{
  PetscLogStage     new_stage;
  PetscLogStageInfo old_info;
  PetscLogState     state;

  PetscFunctionBegin;
  PetscCall(PetscLogHandlerGetState(h, &state));
  PetscCall(PetscLogStateStageGetInfo(state, old_stage, &old_info));
  PetscCall(PetscLogStateGetCurrentStage(state, &new_stage));
  if (new_stage >= 0) {
    PetscLogStageInfo new_info;

    PetscCall(PetscLogStateStageGetInfo(state, new_stage, &new_info));
    PrintData("Popping stage \"%s\" (back to \"%s\")\n", old_info.name, new_info.name);
  } else {
    PrintData("Popping initial stage \"%s\"\n", old_info.name);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerView_Ex7(PetscLogHandler h, PetscViewer viewer)
{
  PetscBool is_ascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &is_ascii));
  if (is_ascii) {
    HandlerCtx ctx;
    PetscInt   num_entries;

    ctx = (HandlerCtx)h->data;
    PetscCall(PetscHMapIGetSize(ctx->running, &num_entries));
    PetscCall(PetscViewerASCIIPrintf(viewer, "%" PetscInt_FMT " events were seen by the handler\n", num_entries));
    PetscCall(PetscViewerASCIIPrintf(viewer, "%" PetscInt_FMT " object(s) were created and %" PetscInt_FMT " object(s) were destroyed\n", ctx->num_objects_created, ctx->num_objects_created));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// An example of overloading one of the methods defined using PetscObjectComposeFunction()
static PetscErrorCode PetscLogHandlerLogObjectState_Ex7(PetscLogHandler h, PetscObject obj, const char format[], va_list argp)
{
  const char *name;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetName(obj, &name));
  PrintData("Logged state for \"%s\": ", name);
  PetscCall(PetscVFPrintf(PETSC_STDOUT, format, argp));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerDestroy_Ex7(PetscLogHandler h)
{
  HandlerCtx ctx;

  PetscFunctionBegin;
  ctx = (HandlerCtx)h->data;
  PetscCall(HandlerCtxDestroy(&ctx));
  PetscCall(PetscObjectComposeFunction((PetscObject)h, "PetscLogHandlerLogObjectState_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerCreate_Ex7(PetscLogHandler handler)
{
  HandlerCtx ctx;

  PetscFunctionBegin;
  PetscCall(HandlerCtxCreate(&ctx));
  handler->data               = (void *)ctx;
  handler->ops->destroy       = PetscLogHandlerDestroy_Ex7;
  handler->ops->view          = PetscLogHandlerView_Ex7;
  handler->ops->eventbegin    = PetscLogHandlerEventBegin_Ex7;
  handler->ops->eventend      = PetscLogHandlerEventEnd_Ex7;
  handler->ops->eventsync     = PetscLogHandlerEventSync_Ex7;
  handler->ops->objectcreate  = PetscLogHandlerObjectCreate_Ex7;
  handler->ops->objectdestroy = PetscLogHandlerObjectDestroy_Ex7;
  handler->ops->stagepush     = PetscLogHandlerStagePush_Ex7;
  handler->ops->stagepop      = PetscLogHandlerStagePop_Ex7;
  PetscCall(PetscObjectComposeFunction((PetscObject)handler, "PetscLogHandlerLogObjectState_C", PetscLogHandlerLogObjectState_Ex7));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscClassId        user_classid;
  PetscLogEvent       event_1, event_2;
  PetscLogStage       stage_1;
  PetscContainer      user_object;
  PetscLogHandler     h;
  PetscLogDouble      time;
  PetscLogHandlerType type;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscLogHandlerRegister(PETSCLOGHANDLEREX7, PetscLogHandlerCreate_Ex7));

  PetscCall(PetscLogHandlerCreate(PETSC_COMM_WORLD, &h));
  PetscCall(PetscLogHandlerSetType(h, PETSCLOGHANDLEREX7));
  PetscCall(PetscLogHandlerGetType(h, &type));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Log handler type is: %s\n", type));
  PetscCall(PetscLogHandlerStart(h));

  PetscCall(PetscClassIdRegister("User class", &user_classid));
  PetscCall(PetscLogEventRegister("Event 1", user_classid, &event_1));
  PetscCall(PetscLogEventRegister("Event 2", user_classid, &event_2));
  PetscCall(PetscLogStageRegister("Stage 1", &stage_1));

  PetscCall(PetscLogEventBegin(event_1, NULL, NULL, NULL, NULL));
  PetscCall(PetscLogStagePush(stage_1));
  PetscCall(PetscLogEventBegin(event_2, NULL, NULL, NULL, NULL));
  PetscCall(PetscLogEventSync(event_1, PETSC_COMM_WORLD));
  PetscCall(PetscLogEventBegin(event_1, NULL, NULL, NULL, NULL));
  PetscCall(PetscTime(&time));
  PetscCall(PetscContainerCreate(PETSC_COMM_SELF, &user_object));
  PetscCall(PetscObjectSetName((PetscObject)user_object, "User Container"));
  PetscCall(PetscLogHandlerLogObjectState(h, (PetscObject)user_object, "Created at %e", time));
  PetscCall(PetscContainerDestroy(&user_object));
  PetscCall(PetscLogEventEnd(event_1, NULL, NULL, NULL, NULL));
  PetscCall(PetscLogEventEnd(event_2, NULL, NULL, NULL, NULL));
  PetscCall(PetscLogStagePop());
  PetscCall(PetscLogEventEnd(event_1, NULL, NULL, NULL, NULL));

  PetscCall(PetscLogHandlerStop(h));
  PetscCall(PetscLogHandlerView(h, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscLogHandlerDestroy(&h));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    requires: defined(PETSC_USE_LOG)
    suffix: 0
    filter: sed -E "s/:[^:]+:/:time_removed:/g"

TEST*/
