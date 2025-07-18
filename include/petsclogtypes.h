#pragma once
#include <petscsystypes.h>

/* MANSEC = Sys */
/* SUBMANSEC = Log */

/*S
  PetscEventPerfInfo - statistics on how many times the event is used, how much time it takes, etc.

  Level: advanced

  Note:
  This is the data structure that describes profiling statsitics collected for an event from
  the default log handler (`PetscLogDefaultBegin()`) using `PetscLogEventGetPerfInfo()`.

.seealso(): [](ch_profiling)
S*/
typedef struct {
  int            id;                  /* The integer identifying this event / stage */
  PetscBool      active;              /* Deprecated */
  PetscBool      visible;             /* The flag to print info in summary */
  int            depth;               /* The nesting depth of the event call */
  int            count;               /* The number of times this event was executed */
  PetscLogDouble flops;               /* The flops used in this event */
  PetscLogDouble flops2;              /* The square of flops used in this event */
  PetscLogDouble flopsTmp;            /* The accumulator for flops used in this event */
  PetscLogDouble time;                /* The time taken for this event */
  PetscLogDouble time2;               /* The square of time taken for this event */
  PetscLogDouble timeTmp;             /* The accumulator for time taken for this event */
  PetscLogDouble syncTime;            /* The synchronization barrier time */
  PetscLogDouble dof[8];              /* The number of degrees of freedom associated with this event */
  PetscLogDouble errors[8];           /* The errors (user-defined) associated with this event */
  PetscLogDouble numMessages;         /* The number of messages in this event */
  PetscLogDouble messageLength;       /* The total message lengths in this event */
  PetscLogDouble numReductions;       /* The number of reductions in this event */
  PetscLogDouble memIncrease;         /* How much the resident memory has increased in this event */
  PetscLogDouble mallocIncrease;      /* How much the maximum malloced space has increased in this event */
  PetscLogDouble mallocSpace;         /* How much the space was malloced and kept during this event */
  PetscLogDouble mallocIncreaseEvent; /* Maximum of the high water mark with in event minus memory available at the end of the event */
#if defined(PETSC_HAVE_DEVICE)
  PetscLogDouble CpuToGpuCount; /* The total number of CPU to GPU copies */
  PetscLogDouble GpuToCpuCount; /* The total number of GPU to CPU copies */
  PetscLogDouble CpuToGpuSize;  /* The total size of CPU to GPU copies */
  PetscLogDouble GpuToCpuSize;  /* The total size of GPU to CPU copies */
  PetscLogDouble GpuFlops;      /* The flops done on a GPU in this event */
  PetscLogDouble GpuTime;       /* The time spent on a GPU in this event */
#endif
} PetscEventPerfInfo;

typedef struct _n_PetscIntStack *PetscIntStack;

/*MC
    PetscLogEvent - id used to identify PETSc or user events which timed portions (blocks of executable)
    code.

    Level: intermediate

.seealso: [](ch_profiling), `PetscLogEventRegister()`, `PetscLogEventBegin()`, `PetscLogEventEnd()`, `PetscLogStage`
M*/
typedef int PetscLogEvent;

/*MC
    PetscLogStage - id used to identify user stages (phases, sections) of runs - for logging

    Level: intermediate

.seealso: [](ch_profiling), `PetscLogStageRegister()`, `PetscLogStagePush()`, `PetscLogStagePop()`, `PetscLogEvent`
M*/
typedef int PetscLogStage;

/*MC
    PetscLogClass - id used to identify classes for logging purposes only.  It
    is not equal to its `PetscClassId`, which is the identifier used for other
    purposes.

    Level: developer

.seealso: [](ch_profiling), `PetscLogStateClassRegister()`
M*/
typedef int PetscLogClass;

/*S
  PetscLogHandler - Interface for performance logging.  A log handler receives a `PetscLogState` that has
  information about the events (`PetscLogEvent`) and stages (`PetscLogStage`) in the logging environment.
  When a handler is connected to PETSc's global logging stream (`PetscLogHandlerStart()`), it receives
  updates about events (`PetscLogEventBegin()` / `PetscLogEventEnd()`), stages (`PetscLogStagePush()` /
  `PetscLogStagePop()`), and objects (`PetscLogObjectCreate()` / `PetscLogObjectDestroy()`).  After
  collecting information the logger can summarize its data with `PetscLogHandlerView()`.

  Example Usage:
.vb
#include <petscsys.h>

int main() {
  UserCtx             ctx;
  PetscLogHandlerType handler_type;

  PetscInitialize(...);
  // ... fill in ctx
  PetscLogHandlerCreate(PETSC_COMM_WORLD, &handler);
  PetscLogHandlerSetType(handler, handler_type);
  PetscLogHandlerStart(handler); // connect your handler to global logging state
  // ... run code to be profiled
  PetscLogHandlerStop(handler); // disconnect your handler from the global logging state
  PetscLogHandlerView(handler, PETSC_VIEWER_STDOUT_WORLD); // view the results
  PetscLogHandlerDestroy(&handler);
  PetscFinalize();
}
.ve

  Level: developer

.seealso: [](ch_profiling),
          `PetscLogHandlerCreate()`,
          `PetscLogHandlerStart()`, `PetscLogHandlerStop()`,
          `PetscLogHandlerSetType()`, `PetscLogHandlerGetType()`,
          `PetscLogHandlerSetState()`, `PetscLogHandlerGetState()`,
          `PetscLogHandlerEventBegin()`, `PetscLogHandlerEventEnd()`,
          `PetscLogHandlerEventSync()`,
          `PetscLogHandlerObjectCreate()`, `PetscLogHandlerObjectDestroy()`,
          `PetscLogHandlerStagePush()`, `PetscLogHandlerStagePop()`,
          `PetscLogHandlerView()`,
          `PetscLogHandlerDestroy()`,
S*/
typedef struct _p_PetscLogHandler *PetscLogHandler;

/*J
  PetscLogHandlerType - String with the name of a `PetscLogHandler` type

  Level: Developer

  Note:
  Implementations included with PETSc include\:
+ `PETSCLOGHANDLERDEFAULT` (`PetscLogDefaultBegin()`)        - formats data for PETSc's default summary (`PetscLogView()`) and data-dump (`PetscLogDump()`) formats.
. `PETSCLOGHANDLERNESTED` (`PetscLogNestedBegin()`)          - formats data for XML or flamegraph output
. `PETSCLOGHANDLERTRACE` (`PetscLogTraceBegin()`)            - traces profiling events in an output stream
. `PETSCLOGHANDLERMPE` (`PetscLogMPEBegin()`)                - outputs parallel performance visualization using MPE
. `PETSCLOGHANDLERPERFSTUBS` (`PetscLogPerfstubsBegin()`)    - outputs instrumentation data for PerfStubs/TAU
. `PETSCLOGHANDLERLEGACY` (`PetscLogLegacyCallbacksBegin()`) - adapts legacy callbacks to the `PetscLogHandler` interface
- `PETSCLOGHANDLERNVTX`                                      - creates NVTX ranges for events that are visible in Nsight
- `PETSCLOGHANDLERROCTX`                                     - creates ROCTx ranges for events that are visible in rocprof

.seealso: [](ch_profiling), `PetscLogHandler`, `PetscLogHandlerSetType()`, `PetscLogHandlerGetType()`
J*/
typedef const char *PetscLogHandlerType;

#define PETSCLOGHANDLERDEFAULT   "default"
#define PETSCLOGHANDLERNESTED    "nested"
#define PETSCLOGHANDLERTRACE     "trace"
#define PETSCLOGHANDLERMPE       "mpe"
#define PETSCLOGHANDLERPERFSTUBS "perfstubs"
#define PETSCLOGHANDLERLEGACY    "legacy"
#define PETSCLOGHANDLERNVTX      "nvtx"
#define PETSCLOGHANDLERROCTX     "roctx"

typedef struct _n_PetscLogRegistry *PetscLogRegistry;

/*S
   PetscLogState - Interface for the shared state information used by `PetscLogHandler`s.

   Most users will not need to reference a `PetscLogState` directly: global logging routines
   like `PetscLogEventRegister()`  and `PetscLogStagePush()` implicitly manipulate PETSc's global
   logging state, `PetscLogGetState()`.

   Level: developer

   Notes:
   `PetscLogState` holds a registry of events (`PetscLogStateEventRegister()`), stages
   (`PetscLogStateStageRegister()`), and classes (`PetscLogStateClassRegister()`).
   It keeps track of when the user has activated events (`PetscLogStateEventSetActive()`) and
   stages (`PetscLogStateStageSetActive()`).  It also keeps a stack of running stages
   (`PetscLogStateStagePush()`, `PetscLogStateStagePop()`).

   The struct defining `PetscLogState` is in a public header so that `PetscLogEventBegin()`,
   `PetscLogEventEnd()`, `PetscLogObjectCreate()`, and `PetscLogObjectDestroy()` can be defined
   as macros rather than function calls, but users are discouraged from directly accessing
   the struct's fields, which are subject to change.

.seealso: [](ch_profiling), `PetscLogStateCreate()`, `PetscLogStateDestroy()`
S*/
typedef struct _n_PetscLogState *PetscLogState;
struct _n_PetscLogState {
  PetscLogRegistry registry;
  PetscBT          active;
  PetscIntStack    stage_stack;
  int              current_stage;
  int              bt_num_stages;
  int              bt_num_events;
  int              refct;
};

/*S
  PetscLogEventInfo - A registry entry about a logging event for `PetscLogState`.

  Level: developer

.seealso: [](ch_profiling), `PetscLogEvent`, `PetscLogState`, `PetscLogStateEventGetInfo()`
S*/
typedef struct {
  char        *name;       /* The name of this event */
  PetscClassId classid;    /* The class the event is associated with */
  PetscBool    collective; /* Flag this event as collective */
} PetscLogEventInfo;

/*S
  PetscLogClassInfo - A registry entry about a class for `PetscLogState`.

  Level: developer

.seealso: [](ch_profiling), `PetscLogClass`, `PetscLogState`, `PetscLogStateStageGetInfo()`
S*/
typedef struct {
  char        *name;    /* The class name */
  PetscClassId classid; /* The integer identifying this class */
} PetscLogClassInfo;

/*S
  PetscLogStageInfo - A registry entry about a class for `PetscLogState`.

  Level: developer

.seealso: [](ch_profiling), `PetscLogStage`, `PetscLogState`, `PetscLogStateClassGetInfo()`
S*/
typedef struct {
  char *name; /* The stage name */
} PetscLogStageInfo;
