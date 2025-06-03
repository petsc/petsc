/*
      PETSc code to log object creation and destruction and PETSc events.

      This provides the public API used by the rest of PETSc and by users.

      These routines use a private API that is not used elsewhere in PETSc and is not
      accessible to users. The private API is defined in logimpl.h and the utils directory.

      ***

      This file, and only this file, is for functions that interact with the global logging state
*/
#include <petsc/private/logimpl.h> /*I    "petscsys.h"   I*/
#include <petsc/private/loghandlerimpl.h>
#include <petsctime.h>
#include <petscviewer.h>
#include <petscdevice.h>
#include <petsc/private/deviceimpl.h>

#if defined(PETSC_HAVE_THREADSAFETY)

PetscInt           petsc_log_gid = -1; /* Global threadId counter */
PETSC_TLS PetscInt petsc_log_tid = -1; /* Local threadId */

/* shared variables */
PetscSpinlock PetscLogSpinLock;

PetscInt PetscLogGetTid(void)
{
  if (petsc_log_tid < 0) {
    PetscCall(PetscSpinlockLock(&PetscLogSpinLock));
    petsc_log_tid = ++petsc_log_gid;
    PetscCall(PetscSpinlockUnlock(&PetscLogSpinLock));
  }
  return petsc_log_tid;
}

#endif

/* Global counters */
PetscLogDouble petsc_BaseTime        = 0.0;
PetscLogDouble petsc_TotalFlops      = 0.0; /* The number of flops */
PetscLogDouble petsc_send_ct         = 0.0; /* The number of sends */
PetscLogDouble petsc_recv_ct         = 0.0; /* The number of receives */
PetscLogDouble petsc_send_len        = 0.0; /* The total length of all sent messages */
PetscLogDouble petsc_recv_len        = 0.0; /* The total length of all received messages */
PetscLogDouble petsc_isend_ct        = 0.0; /* The number of immediate sends */
PetscLogDouble petsc_irecv_ct        = 0.0; /* The number of immediate receives */
PetscLogDouble petsc_isend_len       = 0.0; /* The total length of all immediate send messages */
PetscLogDouble petsc_irecv_len       = 0.0; /* The total length of all immediate receive messages */
PetscLogDouble petsc_wait_ct         = 0.0; /* The number of waits */
PetscLogDouble petsc_wait_any_ct     = 0.0; /* The number of anywaits */
PetscLogDouble petsc_wait_all_ct     = 0.0; /* The number of waitalls */
PetscLogDouble petsc_sum_of_waits_ct = 0.0; /* The total number of waits */
PetscLogDouble petsc_allreduce_ct    = 0.0; /* The number of reductions */
PetscLogDouble petsc_gather_ct       = 0.0; /* The number of gathers and gathervs */
PetscLogDouble petsc_scatter_ct      = 0.0; /* The number of scatters and scattervs */

/* Thread Local storage */
PETSC_TLS PetscLogDouble petsc_TotalFlops_th      = 0.0;
PETSC_TLS PetscLogDouble petsc_send_ct_th         = 0.0;
PETSC_TLS PetscLogDouble petsc_recv_ct_th         = 0.0;
PETSC_TLS PetscLogDouble petsc_send_len_th        = 0.0;
PETSC_TLS PetscLogDouble petsc_recv_len_th        = 0.0;
PETSC_TLS PetscLogDouble petsc_isend_ct_th        = 0.0;
PETSC_TLS PetscLogDouble petsc_irecv_ct_th        = 0.0;
PETSC_TLS PetscLogDouble petsc_isend_len_th       = 0.0;
PETSC_TLS PetscLogDouble petsc_irecv_len_th       = 0.0;
PETSC_TLS PetscLogDouble petsc_wait_ct_th         = 0.0;
PETSC_TLS PetscLogDouble petsc_wait_any_ct_th     = 0.0;
PETSC_TLS PetscLogDouble petsc_wait_all_ct_th     = 0.0;
PETSC_TLS PetscLogDouble petsc_sum_of_waits_ct_th = 0.0;
PETSC_TLS PetscLogDouble petsc_allreduce_ct_th    = 0.0;
PETSC_TLS PetscLogDouble petsc_gather_ct_th       = 0.0;
PETSC_TLS PetscLogDouble petsc_scatter_ct_th      = 0.0;

PetscLogDouble petsc_ctog_ct        = 0.0; /* The total number of CPU to GPU copies */
PetscLogDouble petsc_gtoc_ct        = 0.0; /* The total number of GPU to CPU copies */
PetscLogDouble petsc_ctog_sz        = 0.0; /* The total size of CPU to GPU copies */
PetscLogDouble petsc_gtoc_sz        = 0.0; /* The total size of GPU to CPU copies */
PetscLogDouble petsc_ctog_ct_scalar = 0.0; /* The total number of CPU to GPU copies */
PetscLogDouble petsc_gtoc_ct_scalar = 0.0; /* The total number of GPU to CPU copies */
PetscLogDouble petsc_ctog_sz_scalar = 0.0; /* The total size of CPU to GPU copies */
PetscLogDouble petsc_gtoc_sz_scalar = 0.0; /* The total size of GPU to CPU copies */
PetscLogDouble petsc_gflops         = 0.0; /* The flops done on a GPU */
PetscLogDouble petsc_gtime          = 0.0; /* The time spent on a GPU */

PETSC_TLS PetscLogDouble petsc_ctog_ct_th        = 0.0;
PETSC_TLS PetscLogDouble petsc_gtoc_ct_th        = 0.0;
PETSC_TLS PetscLogDouble petsc_ctog_sz_th        = 0.0;
PETSC_TLS PetscLogDouble petsc_gtoc_sz_th        = 0.0;
PETSC_TLS PetscLogDouble petsc_ctog_ct_scalar_th = 0.0;
PETSC_TLS PetscLogDouble petsc_gtoc_ct_scalar_th = 0.0;
PETSC_TLS PetscLogDouble petsc_ctog_sz_scalar_th = 0.0;
PETSC_TLS PetscLogDouble petsc_gtoc_sz_scalar_th = 0.0;
PETSC_TLS PetscLogDouble petsc_gflops_th         = 0.0;
PETSC_TLS PetscLogDouble petsc_gtime_th          = 0.0;

PetscBool PetscLogMemory = PETSC_FALSE;
PetscBool PetscLogSyncOn = PETSC_FALSE;

PetscBool PetscLogGpuTimeFlag = PETSC_FALSE;

PetscInt PetscLogNumViewersCreated   = 0;
PetscInt PetscLogNumViewersDestroyed = 0;

PetscLogState petsc_log_state = NULL;

#define PETSC_LOG_HANDLER_HOT_BLANK {NULL, NULL, NULL, NULL, NULL, NULL}

PetscLogHandlerHot PetscLogHandlers[PETSC_LOG_HANDLER_MAX] = {
  PETSC_LOG_HANDLER_HOT_BLANK,
  PETSC_LOG_HANDLER_HOT_BLANK,
  PETSC_LOG_HANDLER_HOT_BLANK,
  PETSC_LOG_HANDLER_HOT_BLANK,
};

#undef PETSC_LOG_HANDLERS_HOT_BLANK

#if defined(PETSC_USE_LOG)
  #include <../src/sys/logging/handler/impls/default/logdefault.h>

  #if defined(PETSC_HAVE_THREADSAFETY)
PetscErrorCode PetscAddLogDouble(PetscLogDouble *tot, PetscLogDouble *tot_th, PetscLogDouble tmp)
{
  *tot_th += tmp;
  PetscCall(PetscSpinlockLock(&PetscLogSpinLock));
  *tot += tmp;
  PetscCall(PetscSpinlockUnlock(&PetscLogSpinLock));
  return PETSC_SUCCESS;
}

PetscErrorCode PetscAddLogDoubleCnt(PetscLogDouble *cnt, PetscLogDouble *tot, PetscLogDouble *cnt_th, PetscLogDouble *tot_th, PetscLogDouble tmp)
{
  *cnt_th = *cnt_th + 1;
  *tot_th += tmp;
  PetscCall(PetscSpinlockLock(&PetscLogSpinLock));
  *tot += (PetscLogDouble)tmp;
  *cnt += *cnt + 1;
  PetscCall(PetscSpinlockUnlock(&PetscLogSpinLock));
  return PETSC_SUCCESS;
}

  #endif

static PetscErrorCode PetscLogTryGetHandler(PetscLogHandlerType type, PetscLogHandler *handler)
{
  PetscFunctionBegin;
  PetscAssertPointer(handler, 2);
  *handler = NULL;
  for (int i = 0; i < PETSC_LOG_HANDLER_MAX; i++) {
    PetscLogHandler h = PetscLogHandlers[i].handler;
    if (h) {
      PetscBool match;

      PetscCall(PetscObjectTypeCompare((PetscObject)h, type, &match));
      if (match) {
        *handler = PetscLogHandlers[i].handler;
        PetscFunctionReturn(PETSC_SUCCESS);
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogGetDefaultHandler - Get the default log handler if it is running.

  Not collective

  Output Parameter:
. handler - the default `PetscLogHandler`, or `NULL` if it is not running.

  Level: developer

  Notes:
  The default handler is started with `PetscLogDefaultBegin()`,
  if the options flags `-log_all` or `-log_view` is given without arguments,
  or for `-log_view :output:format` if `format` is not `ascii_xml` or `ascii_flamegraph`.

.seealso: [](ch_profiling)
@*/
PetscErrorCode PetscLogGetDefaultHandler(PetscLogHandler *handler)
{
  PetscFunctionBegin;
  PetscCall(PetscLogTryGetHandler(PETSCLOGHANDLERDEFAULT, handler));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogGetHandler(PetscLogHandlerType type, PetscLogHandler *handler)
{
  PetscFunctionBegin;
  PetscAssertPointer(handler, 2);
  PetscCall(PetscLogTryGetHandler(type, handler));
  PetscCheck(*handler != NULL, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "A PetscLogHandler of type %s has not been started.", type);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogGetState - Get the `PetscLogState` for PETSc's global logging, used
  by all default log handlers (`PetscLogDefaultBegin()`,
  `PetscLogNestedBegin()`, `PetscLogTraceBegin()`, `PetscLogMPEBegin()`,
  `PetscLogPerfstubsBegin()`).

  Collective on `PETSC_COMM_WORLD`

  Output Parameter:
. state - The `PetscLogState` changed by registrations (such as
          `PetscLogEventRegister()`) and actions (such as `PetscLogEventBegin()` or
          `PetscLogStagePush()`), or `NULL` if logging is not active

  Level: developer

.seealso: [](ch_profiling), `PetscLogState`
@*/
PetscErrorCode PetscLogGetState(PetscLogState *state)
{
  PetscFunctionBegin;
  PetscAssertPointer(state, 1);
  *state = petsc_log_state;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerCopyToHot(PetscLogHandler h, PetscLogHandlerHot *hot)
{
  PetscFunctionBegin;
  hot->handler       = h;
  hot->eventBegin    = h->ops->eventbegin;
  hot->eventEnd      = h->ops->eventend;
  hot->eventSync     = h->ops->eventsync;
  hot->objectCreate  = h->ops->objectcreate;
  hot->objectDestroy = h->ops->objectdestroy;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogHandlerStart - Connect a log handler to PETSc's global logging stream and state.

  Logically collective

  Input Parameters:
. h - a `PetscLogHandler`

  Level: developer

  Notes:
  Users should only need this if they create their own log handlers: handlers that are started
  from the command line (such as `-log_view` and `-log_trace`) or from a function like
  `PetscLogNestedBegin()` will automatically be started.

  There is a limit of `PESC_LOG_HANDLER_MAX` handlers that can be active at one time.

  To disconnect a handler from the global stream call `PetscLogHandlerStop()`.

  When a log handler is started, stages that have already been pushed with `PetscLogStagePush()`,
  will be pushed for the new log handler, but it will not be informed of any events that are
  in progress.  It is recommended to start any user-defined log handlers immediately following
  `PetscInitialize()`  before any user-defined stages are pushed.

.seealso: [](ch_profiling), `PetscLogHandler`, `PetscLogState`, `PetscLogHandlerStop()`, `PetscInitialize()`
@*/
PetscErrorCode PetscLogHandlerStart(PetscLogHandler h)
{
  PetscFunctionBegin;
  for (PetscInt i = 0; i < PETSC_LOG_HANDLER_MAX; i++) {
    if (PetscLogHandlers[i].handler == h) PetscFunctionReturn(PETSC_SUCCESS);
  }
  for (PetscInt i = 0; i < PETSC_LOG_HANDLER_MAX; i++) {
    if (PetscLogHandlers[i].handler == NULL) {
      PetscCall(PetscObjectReference((PetscObject)h));
      PetscCall(PetscLogHandlerCopyToHot(h, &PetscLogHandlers[i]));
      if (petsc_log_state) {
        PetscLogStage stack_height;
        PetscIntStack orig_stack, temp_stack;

        PetscCall(PetscLogHandlerSetState(h, petsc_log_state));
        stack_height = petsc_log_state->stage_stack->top + 1;
        PetscCall(PetscIntStackCreate(&temp_stack));
        orig_stack                     = petsc_log_state->stage_stack;
        petsc_log_state->stage_stack   = temp_stack;
        petsc_log_state->current_stage = -1;
        for (int s = 0; s < stack_height; s++) {
          PetscLogStage stage = orig_stack->stack[s];
          PetscCall(PetscLogHandlerStagePush(h, stage));
          PetscCall(PetscIntStackPush(temp_stack, stage));
          petsc_log_state->current_stage = stage;
        }
        PetscCall(PetscIntStackDestroy(temp_stack));
        petsc_log_state->stage_stack = orig_stack;
      }
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  }
  SETERRQ(PetscObjectComm((PetscObject)h), PETSC_ERR_ARG_WRONGSTATE, "%d log handlers already started, cannot start another", PETSC_LOG_HANDLER_MAX);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogHandlerStop - Disconnect a log handler from PETSc's global logging stream.

  Logically collective

  Input Parameters:
. h - a `PetscLogHandler`

  Level: developer

  Note:
  After `PetscLogHandlerStop()`, the handler can still access the global logging state
  with `PetscLogHandlerGetState()`, so that it can access the registry when post-processing
  (for instance, in `PetscLogHandlerView()`),

  When a log handler is stopped, the remaining stages will be popped before it is
  disconnected from the log stream.

.seealso: [](ch_profiling), `PetscLogHandler`, `PetscLogState`, `PetscLogHandlerStart()`
@*/
PetscErrorCode PetscLogHandlerStop(PetscLogHandler h)
{
  PetscFunctionBegin;
  for (PetscInt i = 0; i < PETSC_LOG_HANDLER_MAX; i++) {
    if (PetscLogHandlers[i].handler == h) {
      if (petsc_log_state) {
        PetscLogState state;
        PetscLogStage stack_height;
        PetscIntStack orig_stack, temp_stack;

        PetscCall(PetscLogHandlerGetState(h, &state));
        PetscCheck(state == petsc_log_state, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONGSTATE, "Called PetscLogHandlerStop() for a PetscLogHander that was not started.");
        stack_height = petsc_log_state->stage_stack->top + 1;
        PetscCall(PetscIntStackCreate(&temp_stack));
        orig_stack                   = petsc_log_state->stage_stack;
        petsc_log_state->stage_stack = temp_stack;
        for (int s = 0; s < stack_height; s++) {
          PetscLogStage stage = orig_stack->stack[s];

          PetscCall(PetscIntStackPush(temp_stack, stage));
        }
        for (int s = 0; s < stack_height; s++) {
          PetscLogStage stage;
          PetscBool     empty;

          PetscCall(PetscIntStackPop(temp_stack, &stage));
          PetscCall(PetscIntStackEmpty(temp_stack, &empty));
          if (!empty) PetscCall(PetscIntStackTop(temp_stack, &petsc_log_state->current_stage));
          else petsc_log_state->current_stage = -1;
          PetscCall(PetscLogHandlerStagePop(h, stage));
        }
        PetscCall(PetscIntStackDestroy(temp_stack));
        petsc_log_state->stage_stack = orig_stack;
        PetscCall(PetscIntStackTop(petsc_log_state->stage_stack, &petsc_log_state->current_stage));
      }
      PetscCall(PetscArrayzero(&PetscLogHandlers[i], 1));
      PetscCall(PetscObjectDereference((PetscObject)h));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogIsActive - Check if logging (profiling) is currently in progress.

  Not Collective

  Output Parameter:
. isActive - `PETSC_TRUE` if logging is in progress, `PETSC_FALSE` otherwise

  Level: beginner

.seealso: [](ch_profiling), `PetscLogDefaultBegin()`
@*/
PetscErrorCode PetscLogIsActive(PetscBool *isActive)
{
  PetscFunctionBegin;
  *isActive = PETSC_FALSE;
  if (petsc_log_state) {
    for (PetscInt i = 0; i < PETSC_LOG_HANDLER_MAX; i++) {
      if (PetscLogHandlers[i].handler) {
        *isActive = PETSC_TRUE;
        PetscFunctionReturn(PETSC_SUCCESS);
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_UNUSED static PetscErrorCode PetscLogEventBeginIsActive(PetscBool *isActive)
{
  PetscFunctionBegin;
  *isActive = PETSC_FALSE;
  if (petsc_log_state) {
    for (PetscInt i = 0; i < PETSC_LOG_HANDLER_MAX; i++) {
      if (PetscLogHandlers[i].eventBegin) {
        *isActive = PETSC_TRUE;
        PetscFunctionReturn(PETSC_SUCCESS);
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_UNUSED static PetscErrorCode PetscLogEventEndIsActive(PetscBool *isActive)
{
  PetscFunctionBegin;
  *isActive = PETSC_FALSE;
  if (petsc_log_state) {
    for (PetscInt i = 0; i < PETSC_LOG_HANDLER_MAX; i++) {
      if (PetscLogHandlers[i].eventEnd) {
        *isActive = PETSC_TRUE;
        PetscFunctionReturn(PETSC_SUCCESS);
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogTypeBegin(PetscLogHandlerType type)
{
  PetscLogHandler handler;

  PetscFunctionBegin;
  PetscCall(PetscLogTryGetHandler(type, &handler));
  if (handler) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscLogHandlerCreate(PETSC_COMM_WORLD, &handler));
  PetscCall(PetscLogHandlerSetType(handler, type));
  PetscCall(PetscLogHandlerStart(handler));
  PetscCall(PetscLogHandlerDestroy(&handler));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogDefaultBegin - Turns on logging (profiling) of PETSc code using the default log handler (profiler). This logs time, flop
  rates, and object creation and should not slow programs down too much.

  Logically Collective on `PETSC_COMM_WORLD`

  Options Database Key:
. -log_view [viewertype:filename:viewerformat] - Prints summary of flop and timing (profiling) information to the
                                                 screen (for PETSc configured with `--with-log=1` (which is the default)).
                                                 This option must be provided before `PetscInitialize()`.

  Example Usage:
.vb
      PetscInitialize(...);
      PetscLogDefaultBegin();
       ... code ...
      PetscLogView(viewer); or PetscLogDump();
      PetscFinalize();
.ve

  Level: advanced

  Notes:
  `PetscLogView()` or `PetscLogDump()` actually cause the printing of
  the logging information.

  This routine may be called more than once.

  To provide the `-log_view` option in your source code you must call  PetscCall(PetscOptionsSetValue(NULL, "-log_view", NULL));
  before you call `PetscInitialize()`

.seealso: [](ch_profiling), `PetscLogDump()`, `PetscLogView()`, `PetscLogTraceBegin()`
@*/
PetscErrorCode PetscLogDefaultBegin(void)
{
  PetscFunctionBegin;
  PetscCall(PetscLogTypeBegin(PETSCLOGHANDLERDEFAULT));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscLogTraceBegin - Begins trace logging.  Every time a PETSc event
  begins or ends, the event name is printed.

  Logically Collective on `PETSC_COMM_WORLD`, No Fortran Support

  Input Parameter:
. file - The file to print trace in (e.g. stdout)

  Options Database Key:
. -log_trace [filename] - Begins `PetscLogTraceBegin()`

  Level: intermediate

  Notes:
  `PetscLogTraceBegin()` prints the processor number, the execution time (sec),
  then "Event begin:" or "Event end:" followed by the event name.

  `PetscLogTraceBegin()` allows tracing of all PETSc calls, which is useful
  to determine where a program is hanging without running in the
  debugger.  Can be used in conjunction with the -info option.

.seealso: [](ch_profiling), `PetscLogDump()`, `PetscLogView()`, `PetscLogDefaultBegin()`
@*/
PetscErrorCode PetscLogTraceBegin(FILE *file)
{
  PetscLogHandler handler;

  PetscFunctionBegin;
  PetscCall(PetscLogTryGetHandler(PETSCLOGHANDLERTRACE, &handler));
  if (handler) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscLogHandlerCreateTrace(PETSC_COMM_WORLD, file, &handler));
  PetscCall(PetscLogHandlerStart(handler));
  PetscCall(PetscLogHandlerDestroy(&handler));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogHandlerCreate_Nested(MPI_Comm, PetscLogHandler *);

/*@
  PetscLogNestedBegin - Turns on nested logging of objects and events. This logs flop
  rates and object creation and should not slow programs down too much.

  Logically Collective on `PETSC_COMM_WORLD`, No Fortran Support

  Options Database Keys:
. -log_view :filename.xml:ascii_xml - Prints an XML summary of flop and timing information to the file

  Example Usage:
.vb
      PetscInitialize(...);
      PetscLogNestedBegin();
       ... code ...
      PetscLogView(viewer);
      PetscFinalize();
.ve

  Level: advanced

.seealso: `PetscLogDump()`, `PetscLogView()`, `PetscLogTraceBegin()`, `PetscLogDefaultBegin()`
@*/
PetscErrorCode PetscLogNestedBegin(void)
{
  PetscFunctionBegin;
  PetscCall(PetscLogTypeBegin(PETSCLOGHANDLERNESTED));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscLogLegacyCallbacksBegin - Create and start a log handler from callbacks
  matching the now deprecated function pointers `PetscLogPLB`, `PetscLogPLE`,
  `PetscLogPHC`, `PetscLogPHD`.

  Logically Collective on `PETSC_COMM_WORLD`

  Input Parameters:
+ PetscLogPLB - A callback that will be executed by `PetscLogEventBegin()` (or `NULL`)
. PetscLogPLE - A callback that will be executed by `PetscLogEventEnd()` (or `NULL`)
. PetscLogPHC - A callback that will be executed by `PetscLogObjectCreate()` (or `NULL`)
- PetscLogPHD - A callback that will be executed by `PetscLogObjectCreate()` (or `NULL`)

  Calling sequence of `PetscLogPLB`:
+ e  - a `PetscLogEvent` that is beginning
. _i - deprecated, unused
. o1 - a `PetscObject` associated with `e` (or `NULL`)
. o2 - a `PetscObject` associated with `e` (or `NULL`)
. o3 - a `PetscObject` associated with `e` (or `NULL`)
- o4 - a `PetscObject` associated with `e` (or `NULL`)

  Calling sequence of `PetscLogPLE`:
+ e  - a `PetscLogEvent` that is beginning
. _i - deprecated, unused
. o1 - a `PetscObject` associated with `e` (or `NULL`)
. o2 - a `PetscObject` associated with `e` (or `NULL`)
. o3 - a `PetscObject` associated with `e` (or `NULL`)
- o4 - a `PetscObject` associated with `e` (or `NULL`)

  Calling sequence of `PetscLogPHC`:
. o - a `PetscObject` that has just been created

  Calling sequence of `PetscLogPHD`:
. o - a `PetscObject` that is about to be destroyed

  Level: advanced

  Notes:
  This is for transitioning from the deprecated function `PetscLogSet()` and should not be used in new code.

  This should help migrate external log handlers to use `PetscLogHandler`, but
  callbacks that depend on the deprecated `PetscLogStage` datatype will have to be
  updated.

.seealso: [](ch_profiling), `PetscLogHandler`, `PetscLogHandlerStart()`, `PetscLogState`
@*/
PetscErrorCode PetscLogLegacyCallbacksBegin(PetscErrorCode (*PetscLogPLB)(PetscLogEvent e, int _i, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4), PetscErrorCode (*PetscLogPLE)(PetscLogEvent e, int _i, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4), PetscErrorCode (*PetscLogPHC)(PetscObject o), PetscErrorCode (*PetscLogPHD)(PetscObject o))
{
  PetscLogHandler handler;

  PetscFunctionBegin;
  PetscCall(PetscLogHandlerCreateLegacy(PETSC_COMM_WORLD, PetscLogPLB, PetscLogPLE, PetscLogPHC, PetscLogPHD, &handler));
  PetscCall(PetscLogHandlerStart(handler));
  PetscCall(PetscLogHandlerDestroy(&handler));
  PetscFunctionReturn(PETSC_SUCCESS);
}

  #if defined(PETSC_HAVE_MPE)
    #include <mpe.h>
static PetscBool PetscBeganMPE = PETSC_FALSE;
  #endif

/*@C
  PetscLogMPEBegin - Turns on MPE logging of events. This creates large log files and slows the
  program down.

  Collective on `PETSC_COMM_WORLD`, No Fortran Support

  Options Database Key:
. -log_mpe - Prints extensive log information

  Level: advanced

  Note:
  A related routine is `PetscLogDefaultBegin()` (with the options key `-log_view`), which is
  intended for production runs since it logs only flop rates and object creation (and should
  not significantly slow the programs).

.seealso: [](ch_profiling), `PetscLogDump()`, `PetscLogDefaultBegin()`, `PetscLogEventActivate()`,
          `PetscLogEventDeactivate()`
@*/
PetscErrorCode PetscLogMPEBegin(void)
{
  PetscFunctionBegin;
  #if defined(PETSC_HAVE_MPE)
  /* Do MPE initialization */
  if (!MPE_Initialized_logging()) { /* This function exists in mpich 1.1.2 and higher */
    PetscCall(PetscInfo(0, "Initializing MPE.\n"));
    PetscCall(MPE_Init_log());

    PetscBeganMPE = PETSC_TRUE;
  } else {
    PetscCall(PetscInfo(0, "MPE already initialized. Not attempting to reinitialize.\n"));
  }
  PetscCall(PetscLogTypeBegin(PETSCLOGHANDLERMPE));
  #else
  SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP_SYS, "PETSc was configured without MPE support, reconfigure with --with-mpe or --download-mpe");
  #endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

  #if defined(PETSC_HAVE_TAU_PERFSTUBS)
    #include <../src/sys/perfstubs/timer.h>
  #endif

/*@C
  PetscLogPerfstubsBegin - Turns on logging of events using the perfstubs interface.

  Collective on `PETSC_COMM_WORLD`, No Fortran Support

  Options Database Key:
. -log_perfstubs - use an external log handler through the perfstubs interface

  Level: advanced

.seealso: [](ch_profiling), `PetscLogDefaultBegin()`, `PetscLogEventActivate()`
@*/
PetscErrorCode PetscLogPerfstubsBegin(void)
{
  PetscFunctionBegin;
  #if defined(PETSC_HAVE_TAU_PERFSTUBS)
  PetscCall(PetscLogTypeBegin(PETSCLOGHANDLERPERFSTUBS));
  #else
  SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP_SYS, "PETSc was configured without perfstubs support, reconfigure with --with-tau-perfstubs");
  #endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogActions - Determines whether actions are logged for the default log handler.

  Not Collective

  Input Parameter:
. flag - `PETSC_TRUE` if actions are to be logged

  Options Database Key:
+ -log_exclude_actions - (deprecated) Does nothing
- -log_include_actions - Turn on action logging

  Level: intermediate

  Note:
  Logging of actions continues to consume more memory as the program
  runs. Long running programs should consider turning this feature off.

.seealso: [](ch_profiling), `PetscLogStagePush()`, `PetscLogStagePop()`, `PetscLogGetDefaultHandler()`
@*/
PetscErrorCode PetscLogActions(PetscBool flag)
{
  PetscFunctionBegin;
  for (PetscInt i = 0; i < PETSC_LOG_HANDLER_MAX; i++) {
    PetscLogHandler h = PetscLogHandlers[i].handler;

    if (h) PetscCall(PetscLogHandlerSetLogActions(h, flag));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogObjects - Determines whether objects are logged for the graphical viewer.

  Not Collective

  Input Parameter:
. flag - `PETSC_TRUE` if objects are to be logged

  Options Database Key:
+ -log_exclude_objects - (deprecated) Does nothing
- -log_include_objects - Turns on object logging

  Level: intermediate

  Note:
  Logging of objects continues to consume more memory as the program
  runs. Long running programs should consider turning this feature off.

.seealso: [](ch_profiling), `PetscLogStagePush()`, `PetscLogStagePop()`, `PetscLogGetDefaultHandler()`
@*/
PetscErrorCode PetscLogObjects(PetscBool flag)
{
  PetscFunctionBegin;
  for (PetscInt i = 0; i < PETSC_LOG_HANDLER_MAX; i++) {
    PetscLogHandler h = PetscLogHandlers[i].handler;

    if (h) PetscCall(PetscLogHandlerSetLogObjects(h, flag));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------ Stage Functions --------------------------------------------------*/
/*@
  PetscLogStageRegister - Attaches a character string name to a logging stage.

  Not Collective

  Input Parameter:
. sname - The name to associate with that stage

  Output Parameter:
. stage - The stage number or -1 if logging is not active (`PetscLogIsActive()`).

  Level: intermediate

.seealso: [](ch_profiling), `PetscLogStagePush()`, `PetscLogStagePop()`
@*/
PetscErrorCode PetscLogStageRegister(const char sname[], PetscLogStage *stage)
{
  PetscLogState state;

  PetscFunctionBegin;
  *stage = -1;
  PetscCall(PetscLogGetState(&state));
  if (state) PetscCall(PetscLogStateStageRegister(state, sname, stage));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogStagePush - This function pushes a stage on the logging stack. Events started and stopped until `PetscLogStagePop()` will be associated with the stage

  Not Collective

  Input Parameter:
. stage - The stage on which to log

  Example Usage:
  If the option -log_view is used to run the program containing the
  following code, then 2 sets of summary data will be printed during
  PetscFinalize().
.vb
      PetscInitialize(int *argc,char ***args,0,0);
      [stage 0 of code]
      PetscLogStagePush(1);
      [stage 1 of code]
      PetscLogStagePop();
      PetscBarrier(...);
      [more stage 0 of code]
      PetscFinalize();
.ve

  Level: intermediate

  Note:
  Use `PetscLogStageRegister()` to register a stage.

.seealso: [](ch_profiling), `PetscLogStagePop()`, `PetscLogStageRegister()`, `PetscBarrier()`
@*/
PetscErrorCode PetscLogStagePush(PetscLogStage stage)
{
  PetscLogState state;

  PetscFunctionBegin;
  PetscCall(PetscLogGetState(&state));
  if (!state) PetscFunctionReturn(PETSC_SUCCESS);
  for (int i = 0; i < PETSC_LOG_HANDLER_MAX; i++) {
    PetscLogHandler h = PetscLogHandlers[i].handler;
    if (h) PetscCall(PetscLogHandlerStagePush(h, stage));
  }
  PetscCall(PetscLogStateStagePush(state, stage));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogStagePop - This function pops a stage from the logging stack that was pushed with `PetscLogStagePush()`

  Not Collective

  Example Usage:
  If the option -log_view is used to run the program containing the
  following code, then 2 sets of summary data will be printed during
  PetscFinalize().
.vb
      PetscInitialize(int *argc,char ***args,0,0);
      [stage 0 of code]
      PetscLogStagePush(1);
      [stage 1 of code]
      PetscLogStagePop();
      PetscBarrier(...);
      [more stage 0 of code]
      PetscFinalize();
.ve

  Level: intermediate

.seealso: [](ch_profiling), `PetscLogStagePush()`, `PetscLogStageRegister()`, `PetscBarrier()`
@*/
PetscErrorCode PetscLogStagePop(void)
{
  PetscLogState state;
  PetscLogStage current_stage;

  PetscFunctionBegin;
  PetscCall(PetscLogGetState(&state));
  if (!state) PetscFunctionReturn(PETSC_SUCCESS);
  current_stage = state->current_stage;
  PetscCall(PetscLogStateStagePop(state));
  for (int i = 0; i < PETSC_LOG_HANDLER_MAX; i++) {
    PetscLogHandler h = PetscLogHandlers[i].handler;
    if (h) PetscCall(PetscLogHandlerStagePop(h, current_stage));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogStageSetActive - Sets if a stage is used for `PetscLogEventBegin()` and `PetscLogEventEnd()`.

  Not Collective

  Input Parameters:
+ stage    - The stage
- isActive - The activity flag, `PETSC_TRUE` for logging, else `PETSC_FALSE` (defaults to `PETSC_TRUE`)

  Level: intermediate

  Note:
  If this is set to `PETSC_FALSE` the logging acts as if the stage did not exist

.seealso: [](ch_profiling), `PetscLogStageRegister()`, `PetscLogStagePush()`, `PetscLogStagePop()`, `PetscLogEventBegin()`, `PetscLogEventEnd()`, `PetscPreLoadBegin()`, `PetscPreLoadEnd()`, `PetscPreLoadStage()`
@*/
PetscErrorCode PetscLogStageSetActive(PetscLogStage stage, PetscBool isActive)
{
  PetscLogState state;

  PetscFunctionBegin;
  PetscCall(PetscLogGetState(&state));
  if (state) PetscCall(PetscLogStateStageSetActive(state, stage, isActive));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogStageGetActive - Checks if a stage is used for `PetscLogEventBegin()` and `PetscLogEventEnd()`.

  Not Collective

  Input Parameter:
. stage - The stage

  Output Parameter:
. isActive - The activity flag, `PETSC_TRUE` for logging, else `PETSC_FALSE` (defaults to `PETSC_TRUE`)

  Level: intermediate

.seealso: [](ch_profiling), `PetscLogStageRegister()`, `PetscLogStagePush()`, `PetscLogStagePop()`, `PetscLogEventBegin()`, `PetscLogEventEnd()`, `PetscPreLoadBegin()`, `PetscPreLoadEnd()`, `PetscPreLoadStage()`
@*/
PetscErrorCode PetscLogStageGetActive(PetscLogStage stage, PetscBool *isActive)
{
  PetscLogState state;

  PetscFunctionBegin;
  *isActive = PETSC_FALSE;
  PetscCall(PetscLogGetState(&state));
  if (state) PetscCall(PetscLogStateStageGetActive(state, stage, isActive));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogStageSetVisible - Determines stage visibility in `PetscLogView()`

  Not Collective

  Input Parameters:
+ stage     - The stage
- isVisible - The visibility flag, `PETSC_TRUE` to print, else `PETSC_FALSE` (defaults to `PETSC_TRUE`)

  Level: intermediate

  Developer Notes:
  Visibility only affects the default log handler in `PetscLogView()`: stages that are
  set to invisible are suppressed from output.

.seealso: [](ch_profiling), `PetscLogStageGetVisible()`, `PetscLogStageRegister()`, `PetscLogStagePush()`, `PetscLogStagePop()`, `PetscLogView()`, `PetscLogGetDefaultHandler()`
@*/
PetscErrorCode PetscLogStageSetVisible(PetscLogStage stage, PetscBool isVisible)

{
  PetscFunctionBegin;
  for (PetscInt i = 0; i < PETSC_LOG_HANDLER_MAX; i++) {
    PetscLogHandler h = PetscLogHandlers[i].handler;

    if (h) PetscCall(PetscLogHandlerStageSetVisible(h, stage, isVisible));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogStageGetVisible - Returns stage visibility in `PetscLogView()`

  Not Collective

  Input Parameter:
. stage - The stage

  Output Parameter:
. isVisible - The visibility flag, `PETSC_TRUE` to print, else `PETSC_FALSE` (defaults to `PETSC_TRUE`)

  Level: intermediate

.seealso: [](ch_profiling), `PetscLogStageSetVisible()`, `PetscLogStageRegister()`, `PetscLogStagePush()`, `PetscLogStagePop()`, `PetscLogView()`, `PetscLogGetDefaultHandler()`
@*/
PetscErrorCode PetscLogStageGetVisible(PetscLogStage stage, PetscBool *isVisible)
{
  PetscLogHandler handler;

  PetscFunctionBegin;
  *isVisible = PETSC_FALSE;
  PetscCall(PetscLogTryGetHandler(PETSCLOGHANDLERDEFAULT, &handler));
  if (handler) PetscCall(PetscLogHandlerStageGetVisible(handler, stage, isVisible));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogStageGetId - Returns the stage id when given the stage name.

  Not Collective

  Input Parameter:
. name - The stage name

  Output Parameter:
. stage - The stage, , or -1 if no stage with that name exists

  Level: intermediate

.seealso: [](ch_profiling), `PetscLogStageRegister()`, `PetscLogStagePush()`, `PetscLogStagePop()`, `PetscPreLoadBegin()`, `PetscPreLoadEnd()`, `PetscPreLoadStage()`
@*/
PetscErrorCode PetscLogStageGetId(const char name[], PetscLogStage *stage)
{
  PetscLogState state;

  PetscFunctionBegin;
  *stage = -1;
  PetscCall(PetscLogGetState(&state));
  if (state) PetscCall(PetscLogStateGetStageFromName(state, name, stage));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogStageGetName - Returns the stage name when given the stage id.

  Not Collective

  Input Parameter:
. stage - The stage

  Output Parameter:
. name - The stage name

  Level: intermediate

.seealso: [](ch_profiling), `PetscLogStageRegister()`, `PetscLogStagePush()`, `PetscLogStagePop()`, `PetscPreLoadBegin()`, `PetscPreLoadEnd()`, `PetscPreLoadStage()`
@*/
PetscErrorCode PetscLogStageGetName(PetscLogStage stage, const char *name[])
{
  PetscLogStageInfo stage_info;
  PetscLogState     state;

  PetscFunctionBegin;
  *name = NULL;
  PetscCall(PetscLogGetState(&state));
  if (!state) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscLogStateStageGetInfo(state, stage, &stage_info));
  *name = stage_info.name;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------ Event Functions --------------------------------------------------*/

/*@
  PetscLogEventRegister - Registers an event name for logging operations

  Not Collective

  Input Parameters:
+ name    - The name associated with the event
- classid - The classid associated to the class for this event, obtain either with
           `PetscClassIdRegister()` or use a predefined one such as `KSP_CLASSID`, `SNES_CLASSID`, the predefined ones
           are only available in C code

  Output Parameter:
. event - The event id for use with `PetscLogEventBegin()` and `PetscLogEventEnd()`.

  Example Usage:
.vb
      PetscLogEvent USER_EVENT;
      PetscClassId classid;
      PetscLogDouble user_event_flops;
      PetscClassIdRegister("class name",&classid);
      PetscLogEventRegister("User event name",classid,&USER_EVENT);
      PetscLogEventBegin(USER_EVENT,0,0,0,0);
         [code segment to monitor]
         PetscLogFlops(user_event_flops);
      PetscLogEventEnd(USER_EVENT,0,0,0,0);
.ve

  Level: intermediate

  Notes:
  PETSc automatically logs library events if the code has been
  configured with --with-log (which is the default) and
  -log_view or -log_all is specified.  `PetscLogEventRegister()` is
  intended for logging user events to supplement this PETSc
  information.

  PETSc can gather data for use with the utilities Jumpshot
  (part of the MPICH distribution).  If PETSc has been compiled
  with flag -DPETSC_HAVE_MPE (MPE is an additional utility within
  MPICH), the user can employ another command line option, -log_mpe,
  to create a logfile, "mpe.log", which can be visualized
  Jumpshot.

  The classid is associated with each event so that classes of events
  can be disabled simultaneously, such as all matrix events. The user
  can either use an existing classid, such as `MAT_CLASSID`, or create
  their own as shown in the example.

  If an existing event with the same name exists, its event handle is
  returned instead of creating a new event.

.seealso: [](ch_profiling), `PetscLogStageRegister()`, `PetscLogEventBegin()`, `PetscLogEventEnd()`, `PetscLogFlops()`,
          `PetscLogEventActivate()`, `PetscLogEventDeactivate()`, `PetscClassIdRegister()`
@*/
PetscErrorCode PetscLogEventRegister(const char name[], PetscClassId classid, PetscLogEvent *event)
{
  PetscLogState state;

  PetscFunctionBegin;
  *event = -1;
  PetscCall(PetscLogGetState(&state));
  if (state) PetscCall(PetscLogStateEventRegister(state, name, classid, event));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogEventSetCollective - Indicates that a particular event is collective.

  Logically Collective

  Input Parameters:
+ event      - The event id
- collective - `PetscBool` indicating whether a particular event is collective

  Level: developer

  Notes:
  New events returned from `PetscLogEventRegister()` are collective by default.

  Collective events are handled specially if the command line option `-log_sync` is used. In that case the logging saves information about
  two parts of the event; the time for all the MPI ranks to synchronize and then the time for the actual computation/communication
  to be performed. This option is useful to debug imbalance within the computations or communications.

.seealso: [](ch_profiling), `PetscLogEventBegin()`, `PetscLogEventEnd()`, `PetscLogEventRegister()`
@*/
PetscErrorCode PetscLogEventSetCollective(PetscLogEvent event, PetscBool collective)
{
  PetscLogState state;

  PetscFunctionBegin;
  PetscCall(PetscLogGetState(&state));
  if (state) PetscCall(PetscLogStateEventSetCollective(state, event, collective));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PetscLogClassSetActiveAll - Activate or inactivate logging for all events associated with a PETSc object class in every stage.

  Not Collective

  Input Parameters:
+ classid - The object class, for example `MAT_CLASSID`, `SNES_CLASSID`, etc.
- isActive - if `PETSC_FALSE`, events associated with this class will not be send to log handlers.

  Level: developer

.seealso: [](ch_profiling), `PetscLogEventActivate()`, `PetscLogEventActivateAll()`, `PetscLogStageSetActive()`, `PetscLogEventActivateClass()`
*/
static PetscErrorCode PetscLogClassSetActiveAll(PetscClassId classid, PetscBool isActive)
{
  PetscLogState state;

  PetscFunctionBegin;
  PetscCall(PetscLogGetState(&state));
  if (state) PetscCall(PetscLogStateClassSetActiveAll(state, classid, isActive));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogEventIncludeClass - Activates event logging for a PETSc object class in every stage.

  Not Collective

  Input Parameter:
. classid - The object class, for example `MAT_CLASSID`, `SNES_CLASSID`, etc.

  Level: developer

.seealso: [](ch_profiling), `PetscLogEventActivateClass()`, `PetscLogEventDeactivateClass()`, `PetscLogEventActivate()`, `PetscLogEventDeactivate()`
@*/
PetscErrorCode PetscLogEventIncludeClass(PetscClassId classid)
{
  PetscFunctionBegin;
  PetscCall(PetscLogClassSetActiveAll(classid, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogEventExcludeClass - Deactivates event logging for a PETSc object class in every stage.

  Not Collective

  Input Parameter:
. classid - The object class, for example `MAT_CLASSID`, `SNES_CLASSID`, etc.

  Level: developer

  Note:
  If a class is excluded then events associated with that class are not logged.

.seealso: [](ch_profiling), `PetscLogEventDeactivateClass()`, `PetscLogEventActivateClass()`, `PetscLogEventDeactivate()`, `PetscLogEventActivate()`
@*/
PetscErrorCode PetscLogEventExcludeClass(PetscClassId classid)
{
  PetscFunctionBegin;
  PetscCall(PetscLogClassSetActiveAll(classid, PETSC_FALSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PetscLogEventSetActive - Activate or inactivate logging for an event in a given stage

  Not Collective

  Input Parameters:
+ stage - A registered `PetscLogStage` (or `PETSC_DEFAULT` for the current stage)
. event - A `PetscLogEvent`
- isActive - If `PETSC_FALSE`, activity from this event (`PetscLogEventBegin()`, `PetscLogEventEnd()`, `PetscLogEventSync()`) will not be sent to log handlers during this stage

  Usage:
.vb
      PetscLogEventSetActive(VEC_SetValues, PETSC_FALSE);
        [code where you do not want to log VecSetValues()]
      PetscLogEventSetActive(VEC_SetValues, PETSC_TRUE);
        [code where you do want to log VecSetValues()]
.ve

  Level: advanced

  Note:
  The event may be either a pre-defined PETSc event (found in include/petsclog.h)
  or an event number obtained with `PetscLogEventRegister()`.

.seealso: [](ch_profiling), `PetscLogEventDeactivatePush()`, `PetscLogEventDeactivatePop()`
*/
static PetscErrorCode PetscLogEventSetActive(PetscLogStage stage, PetscLogEvent event, PetscBool isActive)
{
  PetscLogState state;

  PetscFunctionBegin;
  PetscCall(PetscLogGetState(&state));
  if (state) PetscCall(PetscLogStateEventSetActive(state, stage, event, isActive));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogEventActivate - Indicates that a particular event should be logged.

  Not Collective

  Input Parameter:
. event - The event id

  Example Usage:
.vb
      PetscLogEventDeactivate(VEC_SetValues);
        [code where you do not want to log VecSetValues()]
      PetscLogEventActivate(VEC_SetValues);
        [code where you do want to log VecSetValues()]
.ve

  Level: advanced

  Note:
  The event may be either a pre-defined PETSc event (found in include/petsclog.h)
  or an event number obtained with `PetscLogEventRegister()`.

.seealso: [](ch_profiling), `PetscLogEventDeactivate()`, `PetscLogEventDeactivatePush()`, `PetscLogEventDeactivatePop()`
@*/
PetscErrorCode PetscLogEventActivate(PetscLogEvent event)
{
  PetscFunctionBegin;
  PetscCall(PetscLogEventSetActive(PETSC_DEFAULT, event, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogEventDeactivate - Indicates that a particular event should not be logged.

  Not Collective

  Input Parameter:
. event - The event id

  Example Usage:
.vb
      PetscLogEventDeactivate(VEC_SetValues);
        [code where you do not want to log VecSetValues()]
      PetscLogEventActivate(VEC_SetValues);
        [code where you do want to log VecSetValues()]
.ve

  Level: advanced

  Note:
  The event may be either a pre-defined PETSc event (found in
  include/petsclog.h) or an event number obtained with `PetscLogEventRegister()`).

.seealso: [](ch_profiling), `PetscLogEventActivate()`, `PetscLogEventDeactivatePush()`, `PetscLogEventDeactivatePop()`
@*/
PetscErrorCode PetscLogEventDeactivate(PetscLogEvent event)
{
  PetscFunctionBegin;
  PetscCall(PetscLogEventSetActive(PETSC_DEFAULT, event, PETSC_FALSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogEventDeactivatePush - Indicates that a particular event should not be logged until `PetscLogEventDeactivatePop()` is called

  Not Collective

  Input Parameter:
. event - The event id

  Example Usage:
.vb
      PetscLogEventDeactivatePush(VEC_SetValues);
        [code where you do not want to log VecSetValues()]
      PetscLogEventDeactivatePop(VEC_SetValues);
        [code where you do want to log VecSetValues()]
.ve

  Level: advanced

  Note:
  The event may be either a pre-defined PETSc event (found in
  include/petsclog.h) or an event number obtained with `PetscLogEventRegister()`).

  PETSc's default log handler (`PetscLogDefaultBegin()`) respects this function because it can make the output of `PetscLogView()` easier to interpret, but other handlers (such as the nested handler, `PetscLogNestedBegin()`) ignore it because suppressing events is not helpful in their output formats.

.seealso: [](ch_profiling), `PetscLogEventActivate()`, `PetscLogEventDeactivate()`, `PetscLogEventDeactivatePop()`
@*/
PetscErrorCode PetscLogEventDeactivatePush(PetscLogEvent event)
{
  PetscFunctionBegin;
  for (PetscInt i = 0; i < PETSC_LOG_HANDLER_MAX; i++) {
    PetscLogHandler h = PetscLogHandlers[i].handler;

    if (h) PetscCall(PetscLogHandlerEventDeactivatePush(h, PETSC_DEFAULT, event));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogEventDeactivatePop - Indicates that a particular event should again be logged after the logging was turned off with `PetscLogEventDeactivatePush()`

  Not Collective

  Input Parameter:
. event - The event id

  Example Usage:
.vb
      PetscLogEventDeactivatePush(VEC_SetValues);
        [code where you do not want to log VecSetValues()]
      PetscLogEventDeactivatePop(VEC_SetValues);
        [code where you do want to log VecSetValues()]
.ve

  Level: advanced

  Note:
  The event may be either a pre-defined PETSc event (found in
  include/petsclog.h) or an event number obtained with `PetscLogEventRegister()`).

.seealso: [](ch_profiling), `PetscLogEventActivate()`, `PetscLogEventDeactivatePush()`
@*/
PetscErrorCode PetscLogEventDeactivatePop(PetscLogEvent event)
{
  PetscFunctionBegin;
  for (PetscInt i = 0; i < PETSC_LOG_HANDLER_MAX; i++) {
    PetscLogHandler h = PetscLogHandlers[i].handler;

    if (h) PetscCall(PetscLogHandlerEventDeactivatePop(h, PETSC_DEFAULT, event));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogEventSetActiveAll - Turns on logging of all events

  Not Collective

  Input Parameters:
+ event    - The event id
- isActive - The activity flag determining whether the event is logged

  Level: advanced

.seealso: [](ch_profiling), `PetscLogEventActivate()`, `PetscLogEventDeactivate()`
@*/
PetscErrorCode PetscLogEventSetActiveAll(PetscLogEvent event, PetscBool isActive)
{
  PetscLogState state;

  PetscFunctionBegin;
  PetscCall(PetscLogGetState(&state));
  if (state) PetscCall(PetscLogStateEventSetActiveAll(state, event, isActive));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PetscLogClassSetActive - Activates event logging for a PETSc object class for the current stage

  Not Collective

  Input Parameters:
+ stage - A registered `PetscLogStage` (or `PETSC_DEFAULT` for the current stage)
. classid - The event class, for example `MAT_CLASSID`, `SNES_CLASSID`, etc.
- isActive - If `PETSC_FALSE`, events associated with this class are not sent to log handlers.

  Level: developer

.seealso: [](ch_profiling), `PetscLogEventIncludeClass()`, `PetscLogEventActivate()`, `PetscLogEventActivateAll()`, `PetscLogStageSetActive()`
*/
static PetscErrorCode PetscLogClassSetActive(PetscLogStage stage, PetscClassId classid, PetscBool isActive)
{
  PetscLogState state;

  PetscFunctionBegin;
  PetscCall(PetscLogGetState(&state));
  if (state) PetscCall(PetscLogStateClassSetActive(state, stage, classid, isActive));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogEventActivateClass - Activates event logging for a PETSc object class for the current stage

  Not Collective

  Input Parameter:
. classid - The event class, for example `MAT_CLASSID`, `SNES_CLASSID`, etc.

  Level: developer

.seealso: [](ch_profiling), `PetscLogEventIncludeClass()`, `PetscLogEventExcludeClass()`, `PetscLogEventDeactivateClass()`, `PetscLogEventActivate()`, `PetscLogEventDeactivate()`
@*/
PetscErrorCode PetscLogEventActivateClass(PetscClassId classid)
{
  PetscFunctionBegin;
  PetscCall(PetscLogClassSetActive(PETSC_DEFAULT, classid, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogEventDeactivateClass - Deactivates event logging for a PETSc object class for the current stage

  Not Collective

  Input Parameter:
. classid - The event class, for example `MAT_CLASSID`, `SNES_CLASSID`, etc.

  Level: developer

.seealso: [](ch_profiling), `PetscLogEventIncludeClass()`, `PetscLogEventExcludeClass()`, `PetscLogEventActivateClass()`, `PetscLogEventActivate()`, `PetscLogEventDeactivate()`
@*/
PetscErrorCode PetscLogEventDeactivateClass(PetscClassId classid)
{
  PetscFunctionBegin;
  PetscCall(PetscLogClassSetActive(PETSC_DEFAULT, classid, PETSC_FALSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  PetscLogEventSync - Synchronizes the beginning of a user event.

  Synopsis:
  #include <petsclog.h>
  PetscErrorCode PetscLogEventSync(PetscLogEvent e, MPI_Comm comm)

  Collective

  Input Parameters:
+ e    - `PetscLogEvent` obtained from `PetscLogEventRegister()`
- comm - an MPI communicator

  Example Usage:
.vb
  PetscLogEvent USER_EVENT;

  PetscLogEventRegister("User event", 0, &USER_EVENT);
  PetscLogEventSync(USER_EVENT, PETSC_COMM_WORLD);
  PetscLogEventBegin(USER_EVENT, 0, 0, 0, 0);
  [code segment to monitor]
  PetscLogEventEnd(USER_EVENT, 0, 0, 0 , 0);
.ve

  Level: developer

  Note:
  This routine should be called only if there is not a `PetscObject` available to pass to
  `PetscLogEventBegin()`.

.seealso: [](ch_profiling), `PetscLogEventRegister()`, `PetscLogEventBegin()`, `PetscLogEventEnd()`
M*/

/*MC
  PetscLogEventBegin - Logs the beginning of a user event.

  Synopsis:
  #include <petsclog.h>
  PetscErrorCode PetscLogEventBegin(PetscLogEvent e, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)

  Not Collective

  Input Parameters:
+ e  - `PetscLogEvent` obtained from `PetscLogEventRegister()`
. o1 - object associated with the event, or `NULL`
. o2 - object associated with the event, or `NULL`
. o3 - object associated with the event, or `NULL`
- o4 - object associated with the event, or `NULL`

  Fortran Synopsis:
  void PetscLogEventBegin(int e, PetscErrorCode ierr)

  Example Usage:
.vb
  PetscLogEvent USER_EVENT;

  PetscLogDouble user_event_flops;
  PetscLogEventRegister("User event",0, &USER_EVENT);
  PetscLogEventBegin(USER_EVENT, 0, 0, 0, 0);
  [code segment to monitor]
  PetscLogFlops(user_event_flops);
  PetscLogEventEnd(USER_EVENT, 0, 0, 0, 0);
.ve

  Level: intermediate

  Developer Note:
  `PetscLogEventBegin()` and `PetscLogEventBegin()` return error codes instead of explicitly
  handling the errors that occur in the macro directly because other packages that use this
  macros have used them in their own functions or methods that do not return error codes and it
  would be disruptive to change the current behavior.

.seealso: [](ch_profiling), `PetscLogEventRegister()`, `PetscLogEventEnd()`, `PetscLogFlops()`
M*/

/*MC
  PetscLogEventEnd - Log the end of a user event.

  Synopsis:
  #include <petsclog.h>
  PetscErrorCode PetscLogEventEnd(PetscLogEvent e, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)

  Not Collective

  Input Parameters:
+ e  - `PetscLogEvent` obtained from `PetscLogEventRegister()`
. o1 - object associated with the event, or `NULL`
. o2 - object associated with the event, or `NULL`
. o3 - object associated with the event, or `NULL`
- o4 - object associated with the event, or `NULL`

  Fortran Synopsis:
  void PetscLogEventEnd(int e, PetscErrorCode ierr)

  Example Usage:
.vb
  PetscLogEvent USER_EVENT;

  PetscLogDouble user_event_flops;
  PetscLogEventRegister("User event", 0, &USER_EVENT);
  PetscLogEventBegin(USER_EVENT, 0, 0, 0, 0);
  [code segment to monitor]
  PetscLogFlops(user_event_flops);
  PetscLogEventEnd(USER_EVENT, 0, 0, 0, 0);
.ve

  Level: intermediate

.seealso: [](ch_profiling), `PetscLogEventRegister()`, `PetscLogEventBegin()`, `PetscLogFlops()`
M*/

/*@C
  PetscLogStageGetPerfInfo - Return the performance information about the given stage

  No Fortran Support

  Input Parameters:
. stage - The stage number or `PETSC_DETERMINE` for the current stage

  Output Parameter:
. info - This structure is filled with the performance information

  Level: intermediate

  Notes:
  This is a low level routine used by the logging functions in PETSc.

  A `PETSCLOGHANDLERDEFAULT` must be running for this to work, having been started either with
  `PetscLogDefaultBegin()` or from the command line with `-log_view`.  If it was not started,
  all performance statistics in `info` will be zeroed.

.seealso: [](ch_profiling), `PetscLogEventRegister()`, `PetscLogEventBegin()`, `PetscLogEventEnd()`, `PetscLogGetDefaultHandler()`
@*/
PetscErrorCode PetscLogStageGetPerfInfo(PetscLogStage stage, PetscEventPerfInfo *info)
{
  PetscLogHandler     handler;
  PetscEventPerfInfo *event_info;

  PetscFunctionBegin;
  PetscAssertPointer(info, 2);
  PetscCall(PetscLogTryGetHandler(PETSCLOGHANDLERDEFAULT, &handler));
  if (handler) {
    PetscCall(PetscLogHandlerGetStagePerfInfo(handler, stage, &event_info));
    *info = *event_info;
  } else {
    PetscCall(PetscInfo(NULL, "Default log handler is not running, PetscLogStageGetPerfInfo() returning zeros\n"));
    PetscCall(PetscMemzero(info, sizeof(*info)));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscLogEventGetPerfInfo - Return the performance information about the given event in the given stage

  No Fortran Support

  Input Parameters:
+ stage - The stage number or `PETSC_DETERMINE` for the current stage
- event - The event number

  Output Parameter:
. info - This structure is filled with the performance information

  Level: intermediate

  Note:
  This is a low level routine used by the logging functions in PETSc

  A `PETSCLOGHANDLERDEFAULT` must be running for this to work, having been started either with
  `PetscLogDefaultBegin()` or from the command line with `-log_view`.  If it was not started,
  all performance statistics in `info` will be zeroed.

.seealso: [](ch_profiling), `PetscLogEventRegister()`, `PetscLogEventBegin()`, `PetscLogEventEnd()`, `PetscLogGetDefaultHandler()`
@*/
PetscErrorCode PetscLogEventGetPerfInfo(PetscLogStage stage, PetscLogEvent event, PetscEventPerfInfo *info)
{
  PetscLogHandler     handler;
  PetscEventPerfInfo *event_info;

  PetscFunctionBegin;
  PetscAssertPointer(info, 3);
  PetscCall(PetscLogTryGetHandler(PETSCLOGHANDLERDEFAULT, &handler));
  if (handler) {
    PetscCall(PetscLogHandlerGetEventPerfInfo(handler, stage, event, &event_info));
    *info = *event_info;
  } else {
    PetscCall(PetscInfo(NULL, "Default log handler is not running, PetscLogEventGetPerfInfo() returning zeros\n"));
    PetscCall(PetscMemzero(info, sizeof(*info)));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogEventSetDof - Set the nth number of degrees of freedom of a numerical problem associated with this event

  Not Collective

  Input Parameters:
+ event - The event id to log
. n     - The dof index, in [0, 8)
- dof   - The number of dofs

  Options Database Key:
. -log_view - Activates log summary

  Level: developer

  Note:
  This is to enable logging of convergence

.seealso: `PetscLogEventSetError()`, `PetscLogEventRegister()`, `PetscLogGetDefaultHandler()`
@*/
PetscErrorCode PetscLogEventSetDof(PetscLogEvent event, PetscInt n, PetscLogDouble dof)
{
  PetscFunctionBegin;
  PetscCheck(!(n < 0) && !(n > 7), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Error index %" PetscInt_FMT " is not in [0, 8)", n);
  for (PetscInt i = 0; i < PETSC_LOG_HANDLER_MAX; i++) {
    PetscLogHandler h = PetscLogHandlers[i].handler;

    if (h) {
      PetscEventPerfInfo *event_info;

      PetscCall(PetscLogHandlerGetEventPerfInfo(h, PETSC_DEFAULT, event, &event_info));
      if (event_info) event_info->dof[n] = dof;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogEventSetError - Set the nth error associated with a numerical problem associated with this event

  Not Collective

  Input Parameters:
+ event - The event id to log
. n     - The error index, in [0, 8)
- error - The error

  Options Database Key:
. -log_view - Activates log summary

  Level: developer

  Notes:
  This is to enable logging of convergence, and enable users to interpret the errors as they wish. For example,
  as different norms, or as errors for different fields

  This is a low level routine used by the logging functions in PETSc

.seealso: `PetscLogEventSetDof()`, `PetscLogEventRegister()`, `PetscLogGetDefaultHandler()`
@*/
PetscErrorCode PetscLogEventSetError(PetscLogEvent event, PetscInt n, PetscLogDouble error)
{
  PetscFunctionBegin;
  PetscCheck(!(n < 0) && !(n > 7), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Error index %" PetscInt_FMT " is not in [0, 8)", n);
  for (PetscInt i = 0; i < PETSC_LOG_HANDLER_MAX; i++) {
    PetscLogHandler h = PetscLogHandlers[i].handler;

    if (h) {
      PetscEventPerfInfo *event_info;

      PetscCall(PetscLogHandlerGetEventPerfInfo(h, PETSC_DEFAULT, event, &event_info));
      if (event_info) event_info->errors[n] = error;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogEventGetId - Returns the event id when given the event name.

  Not Collective

  Input Parameter:
. name - The event name

  Output Parameter:
. event - The event, or -1 if no event with that name exists

  Level: intermediate

.seealso: [](ch_profiling), `PetscLogEventBegin()`, `PetscLogEventEnd()`, `PetscLogStageGetId()`
@*/
PetscErrorCode PetscLogEventGetId(const char name[], PetscLogEvent *event)
{
  PetscLogState state;

  PetscFunctionBegin;
  *event = -1;
  PetscCall(PetscLogGetState(&state));
  if (state) PetscCall(PetscLogStateGetEventFromName(state, name, event));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogEventGetName - Returns the event name when given the event id.

  Not Collective

  Input Parameter:
. event - The event

  Output Parameter:
. name - The event name

  Level: intermediate

.seealso: [](ch_profiling), `PetscLogEventRegister()`, `PetscLogEventBegin()`, `PetscLogEventEnd()`, `PetscPreLoadBegin()`, `PetscPreLoadEnd()`, `PetscPreLoadStage()`
@*/
PetscErrorCode PetscLogEventGetName(PetscLogEvent event, const char *name[])
{
  PetscLogEventInfo event_info;
  PetscLogState     state;

  PetscFunctionBegin;
  *name = NULL;
  PetscCall(PetscLogGetState(&state));
  if (!state) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscLogStateEventGetInfo(state, event, &event_info));
  *name = event_info.name;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogEventsPause - Put event logging into "paused" mode: timers and counters for in-progress events are paused, and any events that happen before logging is resumed with `PetscLogEventsResume()` are logged in the "Main Stage" of execution.

  Not collective

  Level: advanced

  Notes:
  When an external library or runtime has is initialized it can involve lots of setup time that skews the statistics of any unrelated running events: this function is intended to isolate such calls in the default log summary (`PetscLogDefaultBegin()`, `PetscLogView()`).

  Other log handlers (such as the nested handler, `PetscLogNestedBegin()`) will ignore this function.

.seealso: [](ch_profiling), `PetscLogEventDeactivatePush()`, `PetscLogEventDeactivatePop()`, `PetscLogEventsResume()`, `PetscLogGetDefaultHandler()`
@*/
PetscErrorCode PetscLogEventsPause(void)
{
  PetscFunctionBegin;
  for (PetscInt i = 0; i < PETSC_LOG_HANDLER_MAX; i++) {
    PetscLogHandler h = PetscLogHandlers[i].handler;

    if (h) PetscCall(PetscLogHandlerEventsPause(h));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogEventsResume - Return logging to normal behavior after it was paused with `PetscLogEventsPause()`.

  Not collective

  Level: advanced

.seealso: [](ch_profiling), `PetscLogEventDeactivatePush()`, `PetscLogEventDeactivatePop()`, `PetscLogEventsPause()`, `PetscLogGetDefaultHandler()`
@*/
PetscErrorCode PetscLogEventsResume(void)
{
  PetscFunctionBegin;
  for (PetscInt i = 0; i < PETSC_LOG_HANDLER_MAX; i++) {
    PetscLogHandler h = PetscLogHandlers[i].handler;

    if (h) PetscCall(PetscLogHandlerEventsResume(h));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------ Class Functions --------------------------------------------------*/

/*MC
   PetscLogObjectCreate - Log the creation of a `PetscObject`

   Synopsis:
   #include <petsclog.h>
   PetscErrorCode PetscLogObjectCreate(PetscObject h)

   Not Collective

   Input Parameters:
.  h - A `PetscObject`

   Level: developer

   Developer Note:
     Called internally by PETSc when creating objects: users do not need to call this directly.
     Notification of the object creation is sent to each `PetscLogHandler` that is running.

.seealso: [](ch_profiling), `PetscLogHandler`, `PetscLogObjectDestroy()`
M*/

/*MC
   PetscLogObjectDestroy - Logs the destruction of a `PetscObject`

   Synopsis:
   #include <petsclog.h>
   PetscErrorCode PetscLogObjectDestroy(PetscObject h)

   Not Collective

   Input Parameters:
.  h - A `PetscObject`

   Level: developer

   Developer Note:
     Called internally by PETSc when destroying objects: users do not need to call this directly.
     Notification of the object creation is sent to each `PetscLogHandler` that is running.

.seealso: [](ch_profiling), `PetscLogHandler`, `PetscLogObjectCreate()`
M*/

/*@
  PetscLogClassGetClassId - Returns the `PetscClassId` when given the class name.

  Not Collective

  Input Parameter:
. name - The class name

  Output Parameter:
. classid - The `PetscClassId` id, or -1 if no class with that name exists

  Level: intermediate

.seealso: [](ch_profiling), `PetscLogEventBegin()`, `PetscLogEventEnd()`, `PetscLogStageGetId()`
@*/
PetscErrorCode PetscLogClassGetClassId(const char name[], PetscClassId *classid)
{
  PetscLogClass     log_class;
  PetscLogClassInfo class_info;
  PetscLogState     state;

  PetscFunctionBegin;
  *classid = -1;
  PetscCall(PetscLogGetState(&state));
  if (!state) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscLogStateGetClassFromName(state, name, &log_class));
  if (log_class < 0) {
    *classid = -1;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscLogStateClassGetInfo(state, log_class, &class_info));
  *classid = class_info.classid;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscLogClassIdGetName - Returns a `PetscClassId`'s name.

  Not Collective

  Input Parameter:
. classid - A `PetscClassId`

  Output Parameter:
. name - The class name

  Level: intermediate

.seealso: [](ch_profiling), `PetscLogClassRegister()`, `PetscLogClassBegin()`, `PetscLogClassEnd()`, `PetscPreLoadBegin()`, `PetscPreLoadEnd()`, `PetscPreLoadClass()`
@*/
PetscErrorCode PetscLogClassIdGetName(PetscClassId classid, const char **name)
{
  PetscLogClass     log_class;
  PetscLogClassInfo class_info;
  PetscLogState     state;

  PetscFunctionBegin;
  PetscCall(PetscLogGetState(&state));
  PetscCall(PetscLogStateGetClassFromClassId(state, classid, &log_class));
  PetscCall(PetscLogStateClassGetInfo(state, log_class, &class_info));
  *name = class_info.name;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------ Output Functions -------------------------------------------------*/
/*@
  PetscLogDump - Dumps logs of objects to a file. This file is intended to
  be read by bin/petscview. This program no longer exists.

  Collective on `PETSC_COMM_WORLD`

  Input Parameter:
. sname - an optional file name

  Example Usage:
.vb
  PetscInitialize(...);
  PetscLogDefaultBegin();
  // ... code ...
  PetscLogDump(filename);
  PetscFinalize();
.ve

  Level: advanced

  Note:
  The default file name is Log.<rank> where <rank> is the MPI process rank. If no name is specified,
  this file will be used.

.seealso: [](ch_profiling), `PetscLogDefaultBegin()`, `PetscLogView()`, `PetscLogGetDefaultHandler()`
@*/
PetscErrorCode PetscLogDump(const char sname[])
{
  PetscLogHandler handler;

  PetscFunctionBegin;
  PetscCall(PetscLogGetHandler(PETSCLOGHANDLERDEFAULT, &handler));
  PetscCall(PetscLogHandlerDump(handler, sname));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogMPEDump - Dumps the MPE logging info to file for later use with Jumpshot.

  Collective on `PETSC_COMM_WORLD`

  Input Parameter:
. sname - filename for the MPE logfile

  Level: advanced

.seealso: [](ch_profiling), `PetscLogDump()`, `PetscLogMPEBegin()`
@*/
PetscErrorCode PetscLogMPEDump(const char sname[])
{
  PetscFunctionBegin;
  #if defined(PETSC_HAVE_MPE)
  if (PetscBeganMPE) {
    char name[PETSC_MAX_PATH_LEN];

    PetscCall(PetscInfo(0, "Finalizing MPE.\n"));
    if (sname) {
      PetscCall(PetscStrncpy(name, sname, sizeof(name)));
    } else {
      PetscCall(PetscGetProgramName(name, sizeof(name)));
    }
    PetscCall(MPE_Finish_log(name));
  } else {
    PetscCall(PetscInfo(0, "Not finalizing MPE (not started by PETSc).\n"));
  }
  #else
  SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP_SYS, "PETSc was configured without MPE support, reconfigure with --with-mpe or --download-mpe");
  #endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogView - Prints a summary of the logging.

  Collective

  Input Parameter:
. viewer - an ASCII viewer

  Options Database Keys:
+ -log_view [:filename]                    - Prints summary of log information
. -log_view :filename.py:ascii_info_detail - Saves logging information from each process as a Python file
. -log_view :filename.xml:ascii_xml        - Saves a summary of the logging information in a nested format (see below for how to view it)
. -log_view :filename.txt:ascii_flamegraph - Saves logging information in a format suitable for visualising as a Flame Graph (see below for how to view it)
. -log_view_memory                         - Also display memory usage in each event
. -log_view_gpu_time                       - Also display time in each event for GPU kernels (Note this may slow the computation)
. -log_all                                 - Saves a file Log.rank for each MPI rank with details of each step of the computation
- -log_trace [filename]                    - Displays a trace of what each process is doing

  Level: beginner

  Notes:
  It is possible to control the logging programmatically but we recommend using the options database approach whenever possible
  By default the summary is printed to stdout.

  Before calling this routine you must have called either PetscLogDefaultBegin() or PetscLogNestedBegin()

  If PETSc is configured with --with-logging=0 then this functionality is not available

  To view the nested XML format filename.xml first copy  ${PETSC_DIR}/share/petsc/xml/performance_xml2html.xsl to the current
  directory then open filename.xml with your browser. Specific notes for certain browsers
.vb
    Firefox and Internet explorer - simply open the file
    Google Chrome - you must start up Chrome with the option --allow-file-access-from-files
    Safari - see https://ccm.net/faq/36342-safari-how-to-enable-local-file-access
.ve
  or one can use the package <http://xmlsoft.org/XSLT/xsltproc2.html> to translate the xml file to html and then open it with
  your browser.
  Alternatively, use the script ${PETSC_DIR}/lib/petsc/bin/petsc-performance-view to automatically open a new browser
  window and render the XML log file contents.

  The nested XML format was kindly donated by Koos Huijssen and Christiaan M. Klaij  MARITIME  RESEARCH  INSTITUTE  NETHERLANDS

  The Flame Graph output can be visualised using either the original Flame Graph script <https://github.com/brendangregg/FlameGraph>
  or using speedscope <https://www.speedscope.app>.
  Old XML profiles may be converted into this format using the script ${PETSC_DIR}/lib/petsc/bin/xml2flamegraph.py.

.seealso: [](ch_profiling), `PetscLogDefaultBegin()`, `PetscLogDump()`
@*/
PetscErrorCode PetscLogView(PetscViewer viewer)
{
  PetscBool         isascii;
  PetscViewerFormat format;
  int               stage;
  PetscLogState     state;
  PetscIntStack     temp_stack;
  PetscLogHandler   handler;
  PetscBool         is_empty;

  PetscFunctionBegin;
  PetscCall(PetscLogGetState(&state));
  /* Pop off any stages the user forgot to remove */
  PetscCall(PetscIntStackCreate(&temp_stack));
  PetscCall(PetscLogStateGetCurrentStage(state, &stage));
  while (stage >= 0) {
    PetscCall(PetscLogStagePop());
    PetscCall(PetscIntStackPush(temp_stack, stage));
    PetscCall(PetscLogStateGetCurrentStage(state, &stage));
  }
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  PetscCheck(isascii, PetscObjectComm((PetscObject)viewer), PETSC_ERR_SUP, "Currently can only view logging to ASCII");
  PetscCall(PetscViewerGetFormat(viewer, &format));
  if (format == PETSC_VIEWER_ASCII_XML || format == PETSC_VIEWER_ASCII_FLAMEGRAPH) {
    PetscCall(PetscLogGetHandler(PETSCLOGHANDLERNESTED, &handler));
    PetscCall(PetscLogHandlerView(handler, viewer));
  } else {
    PetscCall(PetscLogGetHandler(PETSCLOGHANDLERDEFAULT, &handler));
    PetscCall(PetscLogHandlerView(handler, viewer));
  }
  PetscCall(PetscIntStackEmpty(temp_stack, &is_empty));
  while (!is_empty) {
    PetscCall(PetscIntStackPop(temp_stack, &stage));
    PetscCall(PetscLogStagePush(stage));
    PetscCall(PetscIntStackEmpty(temp_stack, &is_empty));
  }
  PetscCall(PetscIntStackDestroy(temp_stack));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscLogViewFromOptions - Processes command line options to determine if/how a `PetscLog` is to be viewed.

  Collective on `PETSC_COMM_WORLD`

  Level: developer

.seealso: [](ch_profiling), `PetscLogView()`
@*/
PetscErrorCode PetscLogViewFromOptions(void)
{
  PetscInt          n_max = PETSC_LOG_VIEW_FROM_OPTIONS_MAX;
  PetscViewer       viewers[PETSC_LOG_VIEW_FROM_OPTIONS_MAX];
  PetscViewerFormat formats[PETSC_LOG_VIEW_FROM_OPTIONS_MAX];
  PetscBool         flg;

  PetscFunctionBegin;
  PetscCall(PetscOptionsCreateViewers(PETSC_COMM_WORLD, NULL, NULL, "-log_view", &n_max, viewers, formats, &flg));
  /*
     PetscLogHandlerView_Default_Info() wants to be sure that the only objects still around are these viewers, so keep track of how many there are
   */
  PetscLogNumViewersCreated = n_max;
  for (PetscInt i = 0; i < n_max; i++) {
    PetscInt refct;

    PetscCall(PetscViewerPushFormat(viewers[i], formats[i]));
    PetscCall(PetscLogView(viewers[i]));
    PetscCall(PetscViewerPopFormat(viewers[i]));
    PetscCall(PetscObjectGetReference((PetscObject)viewers[i], &refct));
    PetscCall(PetscViewerDestroy(&viewers[i]));
    if (refct == 1) PetscLogNumViewersDestroyed++;
  }
  PetscLogNumViewersDestroyed = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogHandlerNestedSetThreshold(PetscLogHandler, PetscLogDouble, PetscLogDouble *);

/*@
  PetscLogSetThreshold - Set the threshold time for logging the events; this is a percentage out of 100, so 1. means any event
  that takes 1 or more percent of the time.

  Logically Collective on `PETSC_COMM_WORLD`

  Input Parameter:
. newThresh - the threshold to use

  Output Parameter:
. oldThresh - the previously set threshold value

  Options Database Keys:
. -log_view :filename.xml:ascii_xml - Prints an XML summary of flop and timing information to the file

  Example Usage:
.vb
  PetscInitialize(...);
  PetscLogNestedBegin();
  PetscLogSetThreshold(0.1,&oldthresh);
  // ... code ...
  PetscLogView(viewer);
  PetscFinalize();
.ve

  Level: advanced

  Note:
  This threshold is only used by the nested log handler

.seealso: `PetscLogDump()`, `PetscLogView()`, `PetscLogTraceBegin()`, `PetscLogDefaultBegin()`,
          `PetscLogNestedBegin()`
@*/
PetscErrorCode PetscLogSetThreshold(PetscLogDouble newThresh, PetscLogDouble *oldThresh)
{
  PetscLogHandler handler;

  PetscFunctionBegin;
  PetscCall(PetscLogTryGetHandler(PETSCLOGHANDLERNESTED, &handler));
  PetscCall(PetscLogHandlerNestedSetThreshold(handler, newThresh, oldThresh));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*----------------------------------------------- Counter Functions -------------------------------------------------*/
/*@
  PetscGetFlops - Returns the number of flops used on this processor
  since the program began.

  Not Collective

  Output Parameter:
. flops - number of floating point operations

  Level: intermediate

  Notes:
  A global counter logs all PETSc flop counts.  The user can use
  `PetscLogFlops()` to increment this counter to include flops for the
  application code.

  A separate counter `PetscLogGpuFlops()` logs the flops that occur on any GPU associated with this MPI rank

.seealso: [](ch_profiling), `PetscLogGpuFlops()`, `PetscTime()`, `PetscLogFlops()`
@*/
PetscErrorCode PetscGetFlops(PetscLogDouble *flops)
{
  PetscFunctionBegin;
  *flops = petsc_TotalFlops;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscLogObjectState - Record information about an object with the default log handler

  Not Collective

  Input Parameters:
+ obj    - the `PetscObject`
. format - a printf-style format string
- ...    - printf arguments to format

  Level: developer

.seealso: [](ch_profiling), `PetscLogObjectCreate()`, `PetscLogObjectDestroy()`, `PetscLogGetDefaultHandler()`
@*/
PetscErrorCode PetscLogObjectState(PetscObject obj, const char format[], ...)
{
  PetscFunctionBegin;
  for (PetscInt i = 0; i < PETSC_LOG_HANDLER_MAX; i++) {
    PetscLogHandler h = PetscLogHandlers[i].handler;

    if (h) {
      va_list Argp;
      va_start(Argp, format);
      PetscCall(PetscLogHandlerLogObjectState_Internal(h, obj, format, Argp));
      va_end(Argp);
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  PetscLogFlops - Adds floating point operations to the global counter.

  Synopsis:
  #include <petsclog.h>
  PetscErrorCode PetscLogFlops(PetscLogDouble f)

  Not Collective

  Input Parameter:
. f - flop counter

  Example Usage:
.vb
  PetscLogEvent USER_EVENT;

  PetscLogEventRegister("User event", 0, &USER_EVENT);
  PetscLogEventBegin(USER_EVENT, 0, 0, 0, 0);
  [code segment to monitor]
  PetscLogFlops(user_flops)
  PetscLogEventEnd(USER_EVENT, 0, 0, 0, 0);
.ve

  Level: intermediate

  Note:
   A global counter logs all PETSc flop counts. The user can use PetscLogFlops() to increment
   this counter to include flops for the application code.

.seealso: [](ch_profiling), `PetscLogGpuFlops()`, `PetscLogEventRegister()`, `PetscLogEventBegin()`, `PetscLogEventEnd()`, `PetscGetFlops()`
M*/

/*MC
  PetscPreLoadBegin - Begin a segment of code that may be preloaded (run twice) to get accurate
  timings

  Synopsis:
  #include <petsclog.h>
  void PetscPreLoadBegin(PetscBool flag, char *name);

  Not Collective

  Input Parameters:
+ flag - `PETSC_TRUE` to run twice, `PETSC_FALSE` to run once, may be overridden with command
         line option `-preload true|false`
- name - name of first stage (lines of code timed separately with `-log_view`) to be preloaded

  Example Usage:
.vb
  PetscPreLoadBegin(PETSC_TRUE, "first stage");
  // lines of code
  PetscPreLoadStage("second stage");
  // lines of code
  PetscPreLoadEnd();
.ve

  Level: intermediate

  Note:
  Only works in C/C++, not Fortran

  Flags available within the macro\:
+ PetscPreLoadingUsed - `PETSC_TRUE` if we are or have done preloading
. PetscPreLoadingOn   - `PETSC_TRUE` if it is CURRENTLY doing preload
. PetscPreLoadIt      - `0` for the first computation (with preloading turned off it is only
                        `0`) `1`  for the second
- PetscPreLoadMax     - number of times it will do the computation, only one when preloading is
                        turned on

  The first two variables are available throughout the program, the second two only between the
  `PetscPreLoadBegin()` and `PetscPreLoadEnd()`

.seealso: [](ch_profiling), `PetscLogEventRegister()`, `PetscLogEventBegin()`, `PetscLogEventEnd()`, `PetscPreLoadEnd()`, `PetscPreLoadStage()`
M*/

/*MC
  PetscPreLoadEnd - End a segment of code that may be preloaded (run twice) to get accurate
  timings

  Synopsis:
  #include <petsclog.h>
  void PetscPreLoadEnd(void);

  Not Collective

  Example Usage:
.vb
  PetscPreLoadBegin(PETSC_TRUE, "first stage");
  // lines of code
  PetscPreLoadStage("second stage");
  // lines of code
  PetscPreLoadEnd();
.ve

  Level: intermediate

  Note:
  Only works in C/C++ not Fortran

.seealso: [](ch_profiling), `PetscLogEventRegister()`, `PetscLogEventBegin()`, `PetscLogEventEnd()`, `PetscPreLoadBegin()`, `PetscPreLoadStage()`
M*/

/*MC
  PetscPreLoadStage - Start a new segment of code to be timed separately to get accurate timings

  Synopsis:
  #include <petsclog.h>
  void PetscPreLoadStage(char *name);

  Not Collective

  Example Usage:
.vb
  PetscPreLoadBegin(PETSC_TRUE,"first stage");
  // lines of code
  PetscPreLoadStage("second stage");
  // lines of code
  PetscPreLoadEnd();
.ve

  Level: intermediate

  Note:
  Only works in C/C++ not Fortran

.seealso: [](ch_profiling), `PetscLogEventRegister()`, `PetscLogEventBegin()`, `PetscLogEventEnd()`, `PetscPreLoadBegin()`, `PetscPreLoadEnd()`
M*/

  #if PetscDefined(HAVE_DEVICE)
    #include <petsc/private/deviceimpl.h>

/*@
  PetscLogGpuTime - turn on the logging of GPU time for GPU kernels

  Options Database Key:
. -log_view_gpu_time - provide the GPU times for all events in the `-log_view` output

  Level: advanced

  Notes:
  Turning on the timing of the GPU kernels can slow down the entire computation and should only
  be used when studying the performance of individual operations on GPU such as vector operations and
  matrix-vector operations.

  If this option is not used then times for most of the events in the `-log_view` output will be listed as Nan, indicating the times are not available

  This routine should only be called once near the beginning of the program. Once it is started
  it cannot be turned off.

.seealso: [](ch_profiling), `PetscLogView()`, `PetscLogGpuFlops()`, `PetscLogGpuTimeEnd()`, `PetscLogGpuTimeBegin()`
@*/
PetscErrorCode PetscLogGpuTime(void)
{
  PetscFunctionBegin;
  PetscCheck(petsc_gtime == 0.0, PETSC_COMM_SELF, PETSC_ERR_SUP, "GPU logging has already been turned on");
  PetscLogGpuTimeFlag = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogGpuTimeBegin - Start timer for device

  Level: intermediate

  Notes:
  When GPU is enabled, the timer is run on the GPU, it is a separate logging of time
  devoted to GPU computations (excluding kernel launch times).

  When GPU is not available, the timer is run on the CPU, it is a separate logging of
  time devoted to GPU computations (including kernel launch times).

  There is no need to call WaitForCUDA() or WaitForHIP() between `PetscLogGpuTimeBegin()` and
  `PetscLogGpuTimeEnd()`

  This timer should NOT include times for data transfers between the GPU and CPU, nor setup
  actions such as allocating space.

  The regular logging captures the time for data transfers and any CPU activities during the
  event. It is used to compute the flop rate on the GPU as it is actively engaged in running a
  kernel.

  Developer Notes:
  The GPU event timer captures the execution time of all the kernels launched in the default
  stream by the CPU between `PetscLogGpuTimeBegin()` and `PetscLogGpuTimeEnd()`.

  `PetscLogGpuTimeBegin()` and `PetscLogGpuTimeEnd()` insert the begin and end events into the
  default stream (stream 0). The device will record a time stamp for the event when it reaches
  that event in the stream. The function xxxEventSynchronize() is called in
  `PetscLogGpuTimeEnd()` to block CPU execution, but not continued GPU execution, until the
  timer event is recorded.

.seealso: [](ch_profiling), `PetscLogView()`, `PetscLogGpuFlops()`, `PetscLogGpuTimeEnd()`, `PetscLogGpuTime()`
@*/
PetscErrorCode PetscLogGpuTimeBegin(void)
{
  PetscBool isActive;

  PetscFunctionBegin;
  PetscCall(PetscLogEventBeginIsActive(&isActive));
  if (!isActive || !PetscLogGpuTimeFlag) PetscFunctionReturn(PETSC_SUCCESS);
    #if defined(PETSC_HAVE_DEVICE) && !defined(PETSC_HAVE_KOKKOS_WITHOUT_GPU)
  {
    PetscDeviceContext dctx;

    PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
    PetscCall(PetscDeviceContextBeginTimer_Internal(dctx));
  }
    #else
  PetscCall(PetscTimeSubtract(&petsc_gtime));
    #endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscLogGpuTimeEnd - Stop timer for device

  Level: intermediate

.seealso: [](ch_profiling), `PetscLogView()`, `PetscLogGpuFlops()`, `PetscLogGpuTimeBegin()`
@*/
PetscErrorCode PetscLogGpuTimeEnd(void)
{
  PetscBool isActive;

  PetscFunctionBegin;
  PetscCall(PetscLogEventEndIsActive(&isActive));
  if (!isActive || !PetscLogGpuTimeFlag) PetscFunctionReturn(PETSC_SUCCESS);
    #if defined(PETSC_HAVE_DEVICE) && !defined(PETSC_HAVE_KOKKOS_WITHOUT_GPU)
  {
    PetscDeviceContext dctx;
    PetscLogDouble     elapsed;

    PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
    PetscCall(PetscDeviceContextEndTimer_Internal(dctx, &elapsed));
    petsc_gtime += (elapsed / 1000.0);
  }
    #else
  PetscCall(PetscTimeAdd(&petsc_gtime));
    #endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

  #endif /* end of PETSC_HAVE_DEVICE */

#endif /* PETSC_USE_LOG*/

/* -- Utility functions for logging from Fortran -- */

PETSC_EXTERN PetscErrorCode PetscASend(int count, int datatype)
{
  PetscFunctionBegin;
#if PetscDefined(USE_LOG)
  PetscCall(PetscAddLogDouble(&petsc_send_ct, &petsc_send_ct_th, 1));
  #if !defined(MPIUNI_H) && !defined(PETSC_HAVE_BROKEN_RECURSIVE_MACRO)
  PetscCall(PetscMPITypeSize(count, MPI_Type_f2c((MPI_Fint)datatype), &petsc_send_len, &petsc_send_len_th));
  #endif
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode PetscARecv(int count, int datatype)
{
  PetscFunctionBegin;
#if PetscDefined(USE_LOG)
  PetscCall(PetscAddLogDouble(&petsc_recv_ct, &petsc_recv_ct_th, 1));
  #if !defined(MPIUNI_H) && !defined(PETSC_HAVE_BROKEN_RECURSIVE_MACRO)
  PetscCall(PetscMPITypeSize(count, MPI_Type_f2c((MPI_Fint)datatype), &petsc_recv_len, &petsc_recv_len_th));
  #endif
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode PetscAReduce(void)
{
  PetscFunctionBegin;
  if (PetscDefined(USE_LOG)) PetscCall(PetscAddLogDouble(&petsc_allreduce_ct, &petsc_allreduce_ct_th, 1));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscClassId PETSC_LARGEST_CLASSID = PETSC_SMALLEST_CLASSID;
PetscClassId PETSC_OBJECT_CLASSID  = 0;

static PetscBool PetscLogInitializeCalled = PETSC_FALSE;

PETSC_INTERN PetscErrorCode PetscLogInitialize(void)
{
  int stage;

  PetscFunctionBegin;
  if (PetscLogInitializeCalled) PetscFunctionReturn(PETSC_SUCCESS);
  PetscLogInitializeCalled = PETSC_TRUE;
  if (PetscDefined(USE_LOG)) {
    /* Setup default logging structures */
    PetscCall(PetscLogStateCreate(&petsc_log_state));
    for (PetscInt i = 0; i < PETSC_LOG_HANDLER_MAX; i++) {
      if (PetscLogHandlers[i].handler) PetscCall(PetscLogHandlerSetState(PetscLogHandlers[i].handler, petsc_log_state));
    }
    PetscCall(PetscLogStateStageRegister(petsc_log_state, "Main Stage", &stage));
    PetscCall(PetscSpinlockCreate(&PetscLogSpinLock));
#if defined(PETSC_HAVE_THREADSAFETY)
    petsc_log_tid = 0;
    petsc_log_gid = 0;
#endif

    /* All processors sync here for more consistent logging */
    PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
    PetscCall(PetscTime(&petsc_BaseTime));
    PetscCall(PetscLogStagePush(stage));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogFinalize(void)
{
  PetscFunctionBegin;
  if (PetscDefined(USE_LOG)) {
    /* Resetting phase */
    // pop remaining stages
    if (petsc_log_state) {
      while (petsc_log_state->current_stage >= 0) PetscCall(PetscLogStagePop());
    }
    for (int i = 0; i < PETSC_LOG_HANDLER_MAX; i++) PetscCall(PetscLogHandlerDestroy(&PetscLogHandlers[i].handler));
    PetscCall(PetscArrayzero(PetscLogHandlers, PETSC_LOG_HANDLER_MAX));
    PetscCall(PetscLogStateDestroy(&petsc_log_state));

    petsc_TotalFlops         = 0.0;
    petsc_BaseTime           = 0.0;
    petsc_TotalFlops         = 0.0;
    petsc_send_ct            = 0.0;
    petsc_recv_ct            = 0.0;
    petsc_send_len           = 0.0;
    petsc_recv_len           = 0.0;
    petsc_isend_ct           = 0.0;
    petsc_irecv_ct           = 0.0;
    petsc_isend_len          = 0.0;
    petsc_irecv_len          = 0.0;
    petsc_wait_ct            = 0.0;
    petsc_wait_any_ct        = 0.0;
    petsc_wait_all_ct        = 0.0;
    petsc_sum_of_waits_ct    = 0.0;
    petsc_allreduce_ct       = 0.0;
    petsc_gather_ct          = 0.0;
    petsc_scatter_ct         = 0.0;
    petsc_TotalFlops_th      = 0.0;
    petsc_send_ct_th         = 0.0;
    petsc_recv_ct_th         = 0.0;
    petsc_send_len_th        = 0.0;
    petsc_recv_len_th        = 0.0;
    petsc_isend_ct_th        = 0.0;
    petsc_irecv_ct_th        = 0.0;
    petsc_isend_len_th       = 0.0;
    petsc_irecv_len_th       = 0.0;
    petsc_wait_ct_th         = 0.0;
    petsc_wait_any_ct_th     = 0.0;
    petsc_wait_all_ct_th     = 0.0;
    petsc_sum_of_waits_ct_th = 0.0;
    petsc_allreduce_ct_th    = 0.0;
    petsc_gather_ct_th       = 0.0;
    petsc_scatter_ct_th      = 0.0;

    petsc_ctog_ct    = 0.0;
    petsc_gtoc_ct    = 0.0;
    petsc_ctog_sz    = 0.0;
    petsc_gtoc_sz    = 0.0;
    petsc_gflops     = 0.0;
    petsc_gtime      = 0.0;
    petsc_ctog_ct_th = 0.0;
    petsc_gtoc_ct_th = 0.0;
    petsc_ctog_sz_th = 0.0;
    petsc_gtoc_sz_th = 0.0;
    petsc_gflops_th  = 0.0;
    petsc_gtime_th   = 0.0;
  }
  PETSC_LARGEST_CLASSID    = PETSC_SMALLEST_CLASSID;
  PETSC_OBJECT_CLASSID     = 0;
  PetscLogInitializeCalled = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscClassIdRegister - Registers a new class name for objects and logging operations in an application code.

  Not Collective

  Input Parameter:
. name - The class name

  Output Parameter:
. oclass - The class id or classid

  Level: developer

.seealso: [](ch_profiling), `PetscLogEventRegister()`
@*/
PetscErrorCode PetscClassIdRegister(const char name[], PetscClassId *oclass)
{
  PetscFunctionBegin;
  *oclass = ++PETSC_LARGEST_CLASSID;
#if defined(PETSC_USE_LOG)
  {
    PetscLogState state;
    PetscLogClass logclass;

    PetscCall(PetscLogGetState(&state));
    if (state) PetscCall(PetscLogStateClassRegister(state, name, *oclass, &logclass));
  }
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}
