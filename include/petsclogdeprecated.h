#pragma once

#include <petsclog.h>

/* MANSEC = Sys */
/* SUBMANSEC = Log */

/* These data structures are no longer used by any non-deprecated PETSc interface functions */

typedef struct {
  char        *name;
  PetscClassId classid;
} PetscClassRegInfo;

typedef struct _n_PetscClassRegLog *PetscClassRegLog;
struct _n_PetscClassRegLog {
  int                numClasses;
  int                maxClasses;
  PetscClassRegInfo *classInfo;
};

typedef struct {
  PetscClassId   id;
  int            creations;
  int            destructions;
  PetscLogDouble mem;
  PetscLogDouble descMem;
} PetscClassPerfInfo;

typedef struct _n_PetscClassPerfLog *PetscClassPerfLog;
struct _n_PetscClassPerfLog {
  int                 numClasses;
  int                 maxClasses;
  PetscClassPerfInfo *classInfo;
};

typedef struct {
  char        *name;
  PetscClassId classid;
  PetscBool    collective;
#if defined(PETSC_HAVE_TAU_PERFSTUBS)
  void *timer;
#endif
#if defined(PETSC_HAVE_MPE)
  int mpe_id_begin;
  int mpe_id_end;
#endif
} PetscEventRegInfo;

typedef struct _n_PetscEventRegLog *PetscEventRegLog;
struct _n_PetscEventRegLog {
  int                numEvents;
  int                maxEvents;
  PetscEventRegInfo *eventInfo; /* The registration information for each event */
};

typedef struct _n_PetscEventPerfLog *PetscEventPerfLog;
struct _n_PetscEventPerfLog {
  int                 numEvents;
  int                 maxEvents;
  PetscEventPerfInfo *eventInfo;
};

typedef struct _PetscStageInfo {
  char              *name;
  PetscBool          used;
  PetscEventPerfInfo perfInfo;
  PetscClassPerfLog  classLog;
#if defined(PETSC_HAVE_TAU_PERFSTUBS)
  void *timer;
#endif
} PetscStageInfo;

typedef struct _n_PetscStageLog *PetscStageLog;
struct _n_PetscStageLog {
  int              numStages;
  int              maxStages;
  PetscIntStack    stack;
  int              curStage;
  PetscStageInfo  *stageInfo;
  PetscEventRegLog eventLog;
  PetscClassRegLog classLog;
};

PETSC_DEPRECATED_OBJECT(3, 20, 0, "PetscLogGetState()", "PetscStageLog is no longer used.") PETSC_UNUSED static PetscStageLog petsc_stageLog = PETSC_NULLPTR;

/*@C
  PetscLogGetStageLog - Deprecated.

  Level: deprecated

  Note:
  PETSc performance logging and profiling is now split up between the logging state (`PetscLogState`) and the log handler (`PetscLogHandler`).
  The global logging state is obtained with `PetscLogGetState()`; many log handlers may be used at once (`PetscLogHandlerStart()`) and the default log handler is not directly accessible.

.seealso: [](ch_profiling), `PetscLogEventGetPerfInfo()`
@*/
PETSC_DEPRECATED_FUNCTION(3, 20, 0, "PetscLogGetState()", "PetscStageLog is no longer used.") static inline PetscErrorCode PetscLogGetStageLog(PetscStageLog *s)
{
  *s = PETSC_NULLPTR;
  return PETSC_SUCCESS;
}

/*@C
  PetscStageLogGetCurrent - Deprecated

  Level: deprecated

.seealso: [](ch_profiling)
@*/
PETSC_DEPRECATED_FUNCTION(3, 20, 0, "PetscLogStateGetCurrentStage()", "PetscStageLog is no longer used.") static inline PetscErrorCode PetscStageLogGetCurrent(PetscStageLog a, int *b)
{
  (void)a;
  *b = -1;
  return PETSC_SUCCESS;
}

/*@C
  PetscStageLogGetEventPerfLog - Deprecated

  Level: deprecated

  Note:
  PETSc performance logging and profiling is now split up between the logging state (`PetscLogState`) and the log handler (`PetscLogHandler`).
  The global logging state is obtained with `PetscLogGetState()`; many log handlers may be used at once (`PetscLogHandlerStart()`) and the default log handler is not directly accessible.

.seealso: [](ch_profiling)
@*/
PETSC_DEPRECATED_FUNCTION(3, 20, 0, "PetscLogStateEventGetInfo()", "PetscStageLog is no longer used.") static inline PetscErrorCode PetscStageLogGetEventPerfLog(PetscStageLog a, int b, PetscEventPerfLog *c)
{
  (void)a;
  (void)b;
  *c = PETSC_NULLPTR;
  return PETSC_SUCCESS;
}

PETSC_DEPRECATED_OBJECT(3, 20, 0, "PetscLogLegacyCallbacksBegin()", ) PETSC_UNUSED static PetscErrorCode (*PetscLogPLB)(PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject) = PETSC_NULLPTR;
PETSC_DEPRECATED_OBJECT(3, 20, 0, "PetscLogLegacyCallbacksBegin()", ) PETSC_UNUSED static PetscErrorCode (*PetscLogPLE)(PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject) = PETSC_NULLPTR;
PETSC_DEPRECATED_OBJECT(3, 20, 0, "PetscLogLegacyCallbacksBegin()", ) PETSC_UNUSED static PetscErrorCode (*PetscLogPHC)(PetscObject)                                                            = PETSC_NULLPTR;
PETSC_DEPRECATED_OBJECT(3, 20, 0, "PetscLogLegacyCallbacksBegin()", ) PETSC_UNUSED static PetscErrorCode (*PetscLogPHD)(PetscObject)                                                            = PETSC_NULLPTR;

PETSC_DEPRECATED_FUNCTION(3, 20, 0, "nothing", "PETSc does not guarantee a stack property of logging events.") static inline PetscErrorCode PetscLogPushCurrentEvent_Internal(PetscLogEvent e)
{
  (void)e;
  return PETSC_SUCCESS;
}

PETSC_DEPRECATED_FUNCTION(3, 20, 0, "nothing", "PETSc does not guarantee a stack property of logging events.") static inline PetscErrorCode PetscLogPopCurrentEvent_Internal(void)
{
  return PETSC_SUCCESS;
}

/*@C
  PetscLogAllBegin - Equivalent to `PetscLogDefaultBegin()`.

  Logically Collective on `PETSC_COMM_WORLD`

  Level: deprecated

  Note:
  In previous versions, PETSc's documentation stated that `PetscLogAllBegin()` "Turns on extensive logging of objects and events," which was not actually true.
  The actual way to turn on extensive logging of objects and events was, and remains, to call `PetscLogActions()` and `PetscLogObjects()`.

.seealso: [](ch_profiling), `PetscLogDump()`, `PetscLogDefaultBegin()`, `PetscLogActions()`, `PetscLogObjects()`
@*/
PETSC_DEPRECATED_FUNCTION(3, 20, 0, "PetscLogDefaultBegin()", ) static inline PetscErrorCode PetscLogAllBegin(void)
{
  return PetscLogDefaultBegin();
}

/*@C
  PetscLogSet - Deprecated.

  Level: deprecated

  Note:
  PETSc performance logging and profiling is now split up between the logging state (`PetscLogState`) and the log handler (`PetscLogHandler`).
  The global logging state is obtained with `PetscLogGetState()`; many log handlers may be used at once (`PetscLogHandlerStart()`) and the default log handler is not directly accessible.

.seealso: [](ch_profiling), `PetscLogEventGetPerfInfo()`
@*/
PETSC_DEPRECATED_FUNCTION(3, 20, 0, "PetscLogLegacyCallbacksBegin()", )
static inline PetscErrorCode PetscLogSet(PetscErrorCode (*a)(int, int, PetscObject, PetscObject, PetscObject, PetscObject), PetscErrorCode (*b)(int, int, PetscObject, PetscObject, PetscObject, PetscObject))
{
  return PetscLogLegacyCallbacksBegin(a, b, PETSC_NULLPTR, PETSC_NULLPTR);
}
