#pragma once

#include <petscsystypes.h>
#include <petsclogtypes.h>
#include <petsclogdeprecated.h>
#include <petscconf.h>

/* MANSEC = Sys */
/* SUBMANSEC = Log */

/*@
  PetscClassPerfInfoClear - Deprecated.

  Level: deprecated

  Note:
  Performance data is now controlled by `PetscLogHandler`s.  Only the default log handler (`PetscLogBeginDefault()`) exposes event performance information with
  `PetscLogEventGetPerfInfo()`.

.seealso: [](ch_profiling)
@*/
PETSC_DEPRECATED_FUNCTION(3, 20, 0, "PetscMemzero()", ) static inline PetscErrorCode PetscLogClassPerfInfoClear(PetscClassPerfInfo *a)
{
  return PetscMemzero(a, sizeof(*a));
}

/*@
  PetscClassPerfLogCreate - Deprecated.

  Level: deprecated

  Note:
  Performance data is now controlled by `PetscLogHandler`s.  Only the default log handler (`PetscLogBeginDefault()`) exposes event performance information with
  `PetscLogEventGetPerfInfo()`.

.seealso: [](ch_profiling)
@*/
PETSC_DEPRECATED_FUNCTION(3, 20, 0, "nothing", "PetscClassPerfLog is no longer used.") static inline PetscErrorCode PetscLogClassPerfLogCreate(PetscClassPerfLog *a)
{
  *a = NULL;
  return PETSC_SUCCESS;
}

/*@
  PetscClassPerfLogCreate - Deprecated.

  Level: deprecated

  Note:
  Performance data is now controlled by `PetscLogHandler`s.  Only the default log handler (`PetscLogBeginDefault()`) exposes event performance information with
  `PetscLogEventGetPerfInfo()`.

.seealso: [](ch_profiling)
@*/
PETSC_DEPRECATED_FUNCTION(3, 20, 0, "nothing", "PetscClassPerfLog is no longer used.") static inline PetscErrorCode PetscLogClassPerfLogDestroy(PetscClassPerfLog a)
{
  (void)a;
  return PETSC_SUCCESS;
}

/*@
  PetscClassPerfLogEnsureSize - Deprecated.

  Level: deprecated

  Note:
  Performance data is now controlled by `PetscLogHandler`s.  Only the default log handler (`PetscLogBeginDefault()`) exposes event performance information with
  `PetscLogEventGetPerfInfo()`.

.seealso: [](ch_profiling)
@*/
PETSC_DEPRECATED_FUNCTION(3, 20, 0, "nothing", "PetscClassPerfLog is no longer used.") static inline PetscErrorCode PetscClassPerfLogEnsureSize(PetscClassPerfLog a, int b)
{
  (void)a;
  (void)b;
  return PETSC_SUCCESS;
}

/*@
  PetscClassRegInfoDestroy - Deprecated.

  Level: deprecated

  Note:
  Registration data for logging is now controlled by `PetscLogState`.

.seealso: [](ch_profiling)
@*/
PETSC_DEPRECATED_FUNCTION(3, 20, 0, "nothing", "PetscClassRegInfo is no longer used.") static inline PetscErrorCode PetscClassRegInfoDestroy(PetscClassRegInfo *c)
{
  (void)c;
  return PETSC_SUCCESS;
}

/*@
  PetscClassRegLogCreate - Deprecated.

  Level: deprecated

  Note:
  Registration data for logging is now controlled by `PetscLogState`.

.seealso: [](ch_profiling)
@*/
PETSC_DEPRECATED_FUNCTION(3, 20, 0, "nothing", "PetscClassRegLog is no longer used.") static inline PetscErrorCode PetscClassRegLogCreate(PetscClassRegLog *c)
{
  *c = NULL;
  return PETSC_SUCCESS;
}

/*@
  PetscClassRegLogDestroy - Deprecated.

  Level: deprecated

  Note:
  Registration data for logging is now controlled by `PetscLogState`.

.seealso: [](ch_profiling)
@*/
PETSC_DEPRECATED_FUNCTION(3, 20, 0, "nothing", "PetscClassRegLog is no longer used.") static inline PetscErrorCode PetscClassRegLogDestroy(PetscClassRegLog c)
{
  (void)c;
  return PETSC_SUCCESS;
}

/*@
  PetscClassRegLogGetClass - Deprecated.

  Level: deprecated

  Note:
  Registration data for logging is now controlled by `PetscLogState`.

.seealso: [](ch_profiling), `PetscLogStateGetClassFromClassId()`
@*/
PETSC_DEPRECATED_FUNCTION(3, 20, 0, "nothing", "PetscClassRegLog is no longer used.") static inline PetscErrorCode PetscClassRegLogGetClass(PetscClassRegLog c, PetscClassId d, int *e)
{
  (void)c;
  (void)d;
  *e = -1;
  return PETSC_SUCCESS;
}

/*@
  PetscClassRegLogRegister - Deprecated.

  Level: deprecated

  Note:
  Registration data for logging is now controlled by `PetscLogState`.

.seealso: [](ch_profiling), `PetscLogStateClassRegister()`
@*/
PETSC_DEPRECATED_FUNCTION(3, 20, 0, "nothing", "PetscClassRegLog is no longer used.") static inline PetscErrorCode PetscClassRegLogRegister(PetscClassRegLog a, const char *b, PetscClassId c)
{
  (void)a;
  (void)b;
  (void)c;
  return PETSC_SUCCESS;
}

/*MC
  PetscEventPerfInfoAdd - Deprecated.

  Level: deprecated

  Note:
  `PetscEventPerfInfo` is data obtained from the default log handler with `PetscLogEventGetPerfInfo()`.  It is now "plain old data":
  PETSc provides no functions for its manipulation.

.seealso: [](ch_profiling), `PetscEventPerfInfo`
M*/

/*@
  PetscEventPerfInfoClear - Deprecated.

  Level: deprecated

  Note:
  `PetscEventPerfInfo` is data obtained from the default log handler with `PetscLogEventGetPerfInfo()`.  It is now "plain old data":
  PETSc provides no functions for its manipulation.

.seealso: [](ch_profiling), `PetscEventPerfInfo`
@*/
PETSC_DEPRECATED_FUNCTION(3, 20, 0, "PetscMemzero()", ) static inline PetscErrorCode PetscEventPerfInfoClear(PetscEventPerfInfo *a)
{
  return PetscMemzero(a, sizeof(*a));
}

/*@
  PetscEventPerfInfoCopy - Deprecated.

  Level: deprecated

  Note:
  `PetscEventPerfInfo` is data obtained from the default log handler with `PetscLogEventGetPerfInfo()`.  It is now "plain old data":
  PETSc provides no functions for its manipulation.

.seealso: [](ch_profiling), `PetscEventPerfInfo`
@*/
PETSC_DEPRECATED_FUNCTION(3, 20, 0, "PetscMemcpy()", ) static inline PetscErrorCode PetscEventPerfInfoCopy(PetscEventPerfInfo *a, PetscEventPerfInfo *b)
{
  return PetscMemcpy(a, b, sizeof(*a));
}

/*@
  PetscEventPerfLogActivate - Deprecated.

  Level: deprecated

  Note:
  Performance data is now controlled by `PetscLogHandler`s.  Only the default log handler (`PetscLogBeginDefault()`) exposes event performance information with
  `PetscLogEventGetPerfInfo()`.

.seealso: [](ch_profiling)
@*/
PETSC_DEPRECATED_FUNCTION(3, 20, 0, "nothing", "PetscEventPerfLog is no longer used.") static inline PetscErrorCode PetscEventPerfLogActivate(PetscEventPerfLog a, PetscLogEvent b)
{
  (void)a;
  (void)b;
  return PETSC_SUCCESS;
}

/*@
  PetscEventPerfLogActivateClass - Deprecated.

  Level: deprecated

  Note:
  Performance data is now controlled by `PetscLogHandler`s.  Only the default log handler (`PetscLogBeginDefault()`) exposes event performance information with
  `PetscLogEventGetPerfInfo()`.

.seealso: [](ch_profiling)
@*/
PETSC_DEPRECATED_FUNCTION(3, 20, 0, "nothing", "PetscEventPerfLog is no longer used.") static inline PetscErrorCode PetscEventPerfLogActivateClass(PetscEventPerfLog a, PetscEventRegLog b, PetscClassId c)
{
  (void)a;
  (void)b;
  (void)c;
  return PETSC_SUCCESS;
}

/*@
  PetscEventPerfLogCreate - Deprecated

  Level: deprecated

  Note:
  Performance data is now controlled by `PetscLogHandler`s.  Only the default log handler (`PetscLogBeginDefault()`) exposes event performance information with
  `PetscLogEventGetPerfInfo()`.

.seealso: [](ch_profiling)
@*/
PETSC_DEPRECATED_FUNCTION(3, 20, 0, "nothing", "PetscEventPerfLog is no longer used.") static inline PetscErrorCode PetscEventPerfLogCreate(PetscEventPerfLog *a)
{
  *a = NULL;
  return PETSC_SUCCESS;
}

/*@
  PetscEventPerfLogDeactivate - Deprecated

  Level: deprecated

  Note:
  Performance data is now controlled by `PetscLogHandler`s.  Only the default log handler (`PetscLogBeginDefault()`) exposes event performance information with
  `PetscLogEventGetPerfInfo()`.

.seealso: [](ch_profiling)
@*/
PETSC_DEPRECATED_FUNCTION(3, 20, 0, "nothing", "PetscEventPerfLog is no longer used.") static inline PetscErrorCode PetscEventPerfLogDeactivate(PetscEventPerfLog a, PetscLogEvent b)
{
  (void)a;
  (void)b;
  return PETSC_SUCCESS;
}

/*@
  PetscEventPerfLogDeactivateClass - Deprecated

  Level: deprecated

  Note:
  Performance data is now controlled by `PetscLogHandler`s.  Only the default log handler (`PetscLogBeginDefault()`) exposes event performance information with
  `PetscLogEventGetPerfInfo()`.

.seealso: [](ch_profiling)
@*/
PETSC_DEPRECATED_FUNCTION(3, 20, 0, "nothing", "PetscEventPerfLog is no longer used.") static inline PetscErrorCode PetscEventPerfLogDeactivateClass(PetscEventPerfLog a, PetscEventRegLog b, PetscClassId c)
{
  (void)a;
  (void)b;
  (void)c;
  return PETSC_SUCCESS;
}

/*@
  PetscEventPerfLogDeactivatePop - Deprecated

  Level: deprecated

  Note:
  Performance data is now controlled by `PetscLogHandler`s.  Only the default log handler (`PetscLogBeginDefault()`) exposes event performance information with
  `PetscLogEventGetPerfInfo()`.

.seealso: [](ch_profiling)
@*/
PETSC_DEPRECATED_FUNCTION(3, 20, 0, "nothing", "PetscEventPerfLog is no longer used.") static inline PetscErrorCode PetscEventPerfLogDeactivatePop(PetscEventPerfLog a, PetscLogEvent b)
{
  (void)a;
  (void)b;
  return PETSC_SUCCESS;
}

/*@
  PetscEventPerfLogDeactivatePush - Deprecated

  Level: deprecated

  Note:
  Performance data is now controlled by `PetscLogHandler`s.  Only the default log handler (`PetscLogBeginDefault()`) exposes event performance information with
  `PetscLogEventGetPerfInfo()`.

.seealso: [](ch_profiling)
@*/
PETSC_DEPRECATED_FUNCTION(3, 20, 0, "nothing", "PetscEventPerfLog is no longer used.") static inline PetscErrorCode PetscEventPerfLogDeactivatePush(PetscEventPerfLog a, PetscLogEvent b)
{
  (void)a;
  (void)b;
  return PETSC_SUCCESS;
}

/*@
  PetscEventPerfLogDestroy - Deprecated

  Level: deprecated

  Note:
  Performance data is now controlled by `PetscLogHandler`s.  Only the default log handler (`PetscLogBeginDefault()`) exposes event performance information with
  `PetscLogEventGetPerfInfo()`.

.seealso: [](ch_profiling)
@*/
PETSC_DEPRECATED_FUNCTION(3, 20, 0, "nothing", "PetscEventPerfLog is no longer used.") static inline PetscErrorCode PetscEventPerfLogDestroy(PetscEventPerfLog a)
{
  (void)a;
  return PETSC_SUCCESS;
}

/*@
  PetscEventPerfLogEnsureSize - Deprecated

  Level: deprecated

  Note:
  Performance data is now controlled by `PetscLogHandler`s.  Only the default log handler (`PetscLogBeginDefault()`) exposes event performance information with
  `PetscLogEventGetPerfInfo()`.

.seealso: [](ch_profiling)
@*/
PETSC_DEPRECATED_FUNCTION(3, 20, 0, "nothing", "PetscEventPerfLog is no longer used.") static inline PetscErrorCode PetscEventPerfLogEnsureSize(PetscEventPerfLog a, int b)
{
  (void)a;
  (void)b;
  return PETSC_SUCCESS;
}

/*@
  PetscEventPerfLogGetVisible - Deprecated

  Level: deprecated

  Note:
  Performance data is now controlled by `PetscLogHandler`s.  Only the default log handler (`PetscLogBeginDefault()`) exposes event performance information with
  `PetscLogEventGetPerfInfo()`.

.seealso: [](ch_profiling)
@*/
PETSC_DEPRECATED_FUNCTION(3, 20, 0, "nothing", "PetscEventPerfLog is no longer used.") static inline PetscErrorCode PetscEventPerfLogGetVisible(PetscEventPerfLog a, PetscLogEvent b, PetscBool *c)
{
  (void)a;
  (void)b;
  *c = PETSC_TRUE;
  return PETSC_SUCCESS;
}

/*@
  PetscEventPerfLogSetVisible - Deprecated

  Level: deprecated

  Note:
  Performance data is now controlled by `PetscLogHandler`s.  Only the default log handler (`PetscLogBeginDefault()`) exposes event performance information with
  `PetscLogEventGetPerfInfo()`.

.seealso: [](ch_profiling)
@*/
PETSC_DEPRECATED_FUNCTION(3, 20, 0, "nothing", "PetscEventPerfLog is no longer used.") static inline PetscErrorCode PetscEventPerfLogSetVisible(PetscEventPerfLog a, PetscLogEvent b, PetscBool c)
{
  (void)a;
  (void)b;
  (void)c;
  return PETSC_SUCCESS;
}

/*@
  PetscEventRegLogCreate - Deprecated

  Level: deprecated

  Note:
  Registration data for logging is now controlled by `PetscLogState`.

.seealso: [](ch_profiling)
@*/
PETSC_DEPRECATED_FUNCTION(3, 20, 0, "nothing", "PetscEventRegLog is no longer used.") static inline PetscErrorCode PetscEventRegLogCreate(PetscEventRegLog *a)
{
  *a = NULL;
  return PETSC_SUCCESS;
}

/*@
  PetscEventRegLogDestroy - Deprecated

  Level: deprecated

  Note:
  Registration data for logging is now controlled by `PetscLogState`.

.seealso: [](ch_profiling)
@*/
PETSC_DEPRECATED_FUNCTION(3, 20, 0, "nothing", "PetscEventRegLog is no longer used.") static inline PetscErrorCode PetscEventRegLogDestroy(PetscEventRegLog a)
{
  (void)a;
  return PETSC_SUCCESS;
}

/*@
  PetscEventRegLogGetEvent - Deprecated

  Level: deprecated

  Note:
  Registration data for logging is now controlled by `PetscLogState`.

.seealso: [](ch_profiling), `PetscLogEventGetId()`
@*/
PETSC_DEPRECATED_FUNCTION(3, 20, 0, "nothing", "PetscEventRegLog is no longer used.") static inline PetscErrorCode PetscEventRegLogGetEvent(PetscEventRegLog a, const char *b, PetscLogEvent *c)
{
  (void)a;
  (void)b;
  *c = -1;
  return PETSC_SUCCESS;
}

/*@
  PetscEventRegLogRegister - Deprecated

  Level: deprecated

  Note:
  Registration data for logging is now controlled by `PetscLogState`.

.seealso: [](ch_profiling), `PetscLogEventRegister()`
@*/
PETSC_DEPRECATED_FUNCTION(3, 20, 0, "nothing", "PetscEventRegLog is no longer used.") static inline PetscErrorCode PetscEventRegLogRegister(PetscEventRegLog a, const char *b, PetscClassId c, PetscLogEvent *d)
{
  (void)a;
  (void)b;
  (void)c;
  *d = -1;
  return PETSC_SUCCESS;
}

/*MC
  PetscLogMPEGetRGBColor - Deprecated.

  Level: deprecated

.seealso: [](ch_profiling)
M*/

/*MC
  PetscStageInfoDestroy - Deprecated.

  Level: deprecated

  Note:
  Registration data for logging is now controlled by `PetscLogState`.

.seealso: [](ch_profiling)
M*/

/*MC
  PetscStageLogGetEventPerfLog - Deprecated.

  Level: deprecated

  Note:
  PETSc performance logging and profiling is now split up between the logging state (`PetscLogState`) and the log handler (`PetscLogHandler`).
  The global logging state is obtained with `PetscLogGetState()`; many log handlers may be used at once (`PetscLogHandlerStart()`) and the default log handler is not directly accessible.

.seealso: [](ch_profiling), `PetscLogEventGetPerfInfo()`
M*/

/*@
  PetscStageLogCreate - Deprecated

  Level: deprecated

  Note:
  PETSc performance logging and profiling is now split up between the logging state (`PetscLogState`) and the log handler (`PetscLogHandler`).
  The global logging state is obtained with `PetscLogGetState()`; many log handlers may be used at once (`PetscLogHandlerStart()`) and the default log handler is not directly accessible.

.seealso: [](ch_profiling)
@*/
PETSC_DEPRECATED_FUNCTION(3, 20, 0, "nothing", "PetscStageLog is no longer used.") static inline PetscErrorCode PetscStageLogCreate(PetscStageLog *a)
{
  *a = NULL;
  return PETSC_SUCCESS;
}

/*@
  PetscStageLogDestroy - Deprecated

  Level: deprecated

  Note:
  PETSc performance logging and profiling is now split up between the logging state (`PetscLogState`) and the log handler (`PetscLogHandler`).
  The global logging state is obtained with `PetscLogGetState()`; many log handlers may be used at once (`PetscLogHandlerStart()`) and the default log handler is not directly accessible.

.seealso: [](ch_profiling)
@*/
PETSC_DEPRECATED_FUNCTION(3, 20, 0, "nothing", "PetscStageLog is no longer used.") static inline PetscErrorCode PetscStageLogDestroy(PetscStageLog a)
{
  (void)a;
  return PETSC_SUCCESS;
}

/*@
  PetscStageLogGetActive - Deprecated

  Level: deprecated

  Note:
  PETSc performance logging and profiling is now split up between the logging state (`PetscLogState`) and the log handler (`PetscLogHandler`).
  The global logging state is obtained with `PetscLogGetState()`; many log handlers may be used at once (`PetscLogHandlerStart()`) and the default log handler is not directly accessible.

.seealso: [](ch_profiling)
@*/
PETSC_DEPRECATED_FUNCTION(3, 20, 0, "nothing", "PetscStageLog is no longer used.") static inline PetscErrorCode PetscStageLogGetActive(PetscStageLog a, int b, PetscBool *c)
{
  (void)a;
  (void)b;
  *c = PETSC_TRUE;
  return PETSC_SUCCESS;
}

/*@
  PetscStageLogGetClassPerfLog - Deprecated

  Level: deprecated

  Note:
  PETSc performance logging and profiling is now split up between the logging state (`PetscLogState`) and the log handler (`PetscLogHandler`).
  The global logging state is obtained with `PetscLogGetState()`; many log handlers may be used at once (`PetscLogHandlerStart()`) and the default log handler is not directly accessible.

.seealso: [](ch_profiling)
@*/
PETSC_DEPRECATED_FUNCTION(3, 20, 0, "nothing", "PetscStageLog is no longer used.") static inline PetscErrorCode PetscStageLogGetClassPerfLog(PetscStageLog a, int b, PetscClassPerfLog *c)
{
  (void)a;
  (void)b;
  *c = NULL;
  return PETSC_SUCCESS;
}

/*@
  PetscStageLogGetClassRegLog - Deprecated

  Level: deprecated

  Note:
  PETSc performance logging and profiling is now split up between the logging state (`PetscLogState`) and the log handler (`PetscLogHandler`).
  The global logging state is obtained with `PetscLogGetState()`; many log handlers may be used at once (`PetscLogHandlerStart()`) and the default log handler is not directly accessible.

.seealso: [](ch_profiling)
@*/
PETSC_DEPRECATED_FUNCTION(3, 20, 0, "nothing", "PetscStageLog is no longer used.") static inline PetscErrorCode PetscStageLogGetClassRegLog(PetscStageLog a, PetscClassRegLog *c)
{
  (void)a;
  *c = NULL;
  return PETSC_SUCCESS;
}

/*@
  PetscStageLogGetEventRegLog - Deprecated

  Level: deprecated

  Note:
  PETSc performance logging and profiling is now split up between the logging state (`PetscLogState`) and the log handler (`PetscLogHandler`).
  The global logging state is obtained with `PetscLogGetState()`; many log handlers may be used at once (`PetscLogHandlerStart()`) and the default log handler is not directly accessible.

.seealso: [](ch_profiling)
@*/
PETSC_DEPRECATED_FUNCTION(3, 20, 0, "nothing", "PetscStageLog is no longer used.") static inline PetscErrorCode PetscStageLogGetEventRegLog(PetscStageLog a, PetscEventRegLog *c)
{
  (void)a;
  *c = NULL;
  return PETSC_SUCCESS;
}

/*@
  PetscStageLogGetStage - Deprecated

  Level: deprecated

  Note:
  PETSc performance logging and profiling is now split up between the logging state (`PetscLogState`) and the log handler (`PetscLogHandler`).
  The global logging state is obtained with `PetscLogGetState()`; many log handlers may be used at once (`PetscLogHandlerStart()`) and the default log handler is not directly accessible.

.seealso: [](ch_profiling)
@*/
PETSC_DEPRECATED_FUNCTION(3, 20, 0, "nothing", "PetscStageLog is no longer used.") static inline PetscErrorCode PetscStageLogGetStage(PetscStageLog a, const char *b, PetscLogStage *c)
{
  (void)a;
  (void)b;
  *c = -1;
  return PETSC_SUCCESS;
}

/*@
  PetscStageLogGetVisible - Deprecated

  Level: deprecated

  Note:
  PETSc performance logging and profiling is now split up between the logging state (`PetscLogState`) and the log handler (`PetscLogHandler`).
  The global logging state is obtained with `PetscLogGetState()`; many log handlers may be used at once (`PetscLogHandlerStart()`) and the default log handler is not directly accessible.

.seealso: [](ch_profiling)
@*/
PETSC_DEPRECATED_FUNCTION(3, 20, 0, "nothing", "PetscStageLog is no longer used.") static inline PetscErrorCode PetscStageLogGetVisible(PetscStageLog a, int b, PetscBool *c)
{
  (void)a;
  (void)b;
  *c = PETSC_TRUE;
  return PETSC_SUCCESS;
}

/*@
  PetscStageLogPop - Deprecated

  Level: deprecated

  Note:
  PETSc performance logging and profiling is now split up between the logging state (`PetscLogState`) and the log handler (`PetscLogHandler`).
  The global logging state is obtained with `PetscLogGetState()`; many log handlers may be used at once (`PetscLogHandlerStart()`) and the default log handler is not directly accessible.

.seealso: [](ch_profiling)
@*/
PETSC_DEPRECATED_FUNCTION(3, 20, 0, "nothing", "PetscStageLog is no longer used.") static inline PetscErrorCode PetscStageLogPop(PetscStageLog a)
{
  (void)a;
  return PETSC_SUCCESS;
}

/*@
  PetscStageLogPush - Deprecated

  Level: deprecated

.seealso: [](ch_profiling)
@*/
PETSC_DEPRECATED_FUNCTION(3, 20, 0, "nothing", "PetscStageLog is no longer used.") static inline PetscErrorCode PetscStageLogPush(PetscStageLog a, int b)
{
  (void)a;
  (void)b;
  return PETSC_SUCCESS;
}

/*@
  PetscStageLogRegister - Deprecated

  Level: deprecated

.seealso: [](ch_profiling)
@*/
PETSC_DEPRECATED_FUNCTION(3, 20, 0, "nothing", "PetscStageLog is no longer used.") static inline PetscErrorCode PetscStageLogRegister(PetscStageLog a, const char *b, int *c)
{
  (void)a;
  (void)b;
  *c = -1;
  return PETSC_SUCCESS;
}

/*@
  PetscStageLogSetActive - Deprecated

  Level: deprecated

.seealso: [](ch_profiling)
@*/
PETSC_DEPRECATED_FUNCTION(3, 20, 0, "nothing", "PetscStageLog is no longer used.") static inline PetscErrorCode PetscStageLogSetActive(PetscStageLog a, int b, PetscBool c)
{
  (void)a;
  (void)b;
  (void)c;
  return PETSC_SUCCESS;
}

/*@
  PetscStageLogSetVisible - Deprecated

  Level: deprecated

.seealso: [](ch_profiling)
@*/
PETSC_DEPRECATED_FUNCTION(3, 20, 0, "nothing", "PetscStageLog is no longer used.") static inline PetscErrorCode PetscStageLogSetVisible(PetscStageLog a, int b, PetscBool c)
{
  (void)a;
  (void)b;
  (void)c;
  return PETSC_SUCCESS;
}
