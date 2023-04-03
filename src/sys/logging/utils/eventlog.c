
/*
     This defines part of the private API for logging performance information. It is intended to be used only by the
   PETSc PetscLog...() interface and not elsewhere, nor by users. Hence the prototypes for these functions are NOT
   in the public PETSc include files.

*/
#include <petsc/private/logimpl.h> /*I    "petscsys.h"   I*/
#include <petscdevice.h>
#if defined(PETSC_HAVE_TAU_PERFSTUBS)
  #include <../src/sys/perfstubs/timer.h>
#endif

PetscBool PetscLogSyncOn = PETSC_FALSE;
PetscBool PetscLogMemory = PETSC_FALSE;
#if defined(PETSC_HAVE_DEVICE)
PetscBool PetscLogGpuTraffic = PETSC_FALSE;
#endif

/*----------------------------------------------- Creation Functions -------------------------------------------------*/
/* Note: these functions do not have prototypes in a public directory, so they are considered "internal" and not exported. */

/*@C
  PetscEventRegLogCreate - This creates a `PetscEventRegLog` object.

  Not collective

  Input Parameter:
. eventLog - The `PetscEventRegLog`

  Level: developer

  Note:
  This is a low level routine used by the logging functions in PETSc

.seealso: `PetscEventRegLogDestroy()`, `PetscStageLogCreate()`
@*/
PetscErrorCode PetscEventRegLogCreate(PetscEventRegLog *eventLog)
{
  PetscEventRegLog l;

  PetscFunctionBegin;
  PetscCall(PetscNew(&l));
  l->numEvents = 0;
  l->maxEvents = 100;
  PetscCall(PetscMalloc1(l->maxEvents, &l->eventInfo));
  *eventLog = l;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscEventRegLogDestroy - This destroys a `PetscEventRegLog` object.

  Not collective

  Input Parameter:
. eventLog - The `PetscEventRegLog`

  Level: developer

  Note:
  This is a low level routine used by the logging functions in PETSc

.seealso: `PetscEventRegLogCreate()`
@*/
PetscErrorCode PetscEventRegLogDestroy(PetscEventRegLog eventLog)
{
  int e;

  PetscFunctionBegin;
  for (e = 0; e < eventLog->numEvents; e++) PetscCall(PetscFree(eventLog->eventInfo[e].name));
  PetscCall(PetscFree(eventLog->eventInfo));
  PetscCall(PetscFree(eventLog));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscEventPerfLogCreate - This creates a `PetscEventPerfLog` object.

  Not collective

  Input Parameter:
. eventLog - The `PetscEventPerfLog`

  Level: developer

  Note:
  This is a low level routine used by the logging functions in PETSc

.seealso: `PetscEventPerfLogDestroy()`, `PetscStageLogCreate()`
@*/
PetscErrorCode PetscEventPerfLogCreate(PetscEventPerfLog *eventLog)
{
  PetscEventPerfLog l;

  PetscFunctionBegin;
  PetscCall(PetscNew(&l));
  l->numEvents = 0;
  l->maxEvents = 100;
  PetscCall(PetscCalloc1(l->maxEvents, &l->eventInfo));
  *eventLog = l;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscEventPerfLogDestroy - This destroys a `PetscEventPerfLog` object.

  Not collective

  Input Parameter:
. eventLog - The `PetscEventPerfLog`

  Level: developer

  Note:
  This is a low level routine used by the logging functions in PETSc

.seealso: `PetscEventPerfLogCreate()`
@*/
PetscErrorCode PetscEventPerfLogDestroy(PetscEventPerfLog eventLog)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(eventLog->eventInfo));
  PetscCall(PetscFree(eventLog));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------ General Functions -------------------------------------------------*/
/*@C
  PetscEventPerfInfoClear - This clears a `PetscEventPerfInfo` object.

  Not collective

  Input Parameter:
. eventInfo - The `PetscEventPerfInfo`

  Level: developer

  Note:
  This is a low level routine used by the logging functions in PETSc

.seealso: `PetscEventPerfLogCreate()`
@*/
PetscErrorCode PetscEventPerfInfoClear(PetscEventPerfInfo *eventInfo)
{
  PetscFunctionBegin;
  eventInfo->id            = -1;
  eventInfo->active        = PETSC_TRUE;
  eventInfo->visible       = PETSC_TRUE;
  eventInfo->depth         = 0;
  eventInfo->count         = 0;
  eventInfo->flops         = 0.0;
  eventInfo->flops2        = 0.0;
  eventInfo->flopsTmp      = 0.0;
  eventInfo->time          = 0.0;
  eventInfo->time2         = 0.0;
  eventInfo->timeTmp       = 0.0;
  eventInfo->syncTime      = 0.0;
  eventInfo->dof[0]        = -1.0;
  eventInfo->dof[1]        = -1.0;
  eventInfo->dof[2]        = -1.0;
  eventInfo->dof[3]        = -1.0;
  eventInfo->dof[4]        = -1.0;
  eventInfo->dof[5]        = -1.0;
  eventInfo->dof[6]        = -1.0;
  eventInfo->dof[7]        = -1.0;
  eventInfo->errors[0]     = -1.0;
  eventInfo->errors[1]     = -1.0;
  eventInfo->errors[2]     = -1.0;
  eventInfo->errors[3]     = -1.0;
  eventInfo->errors[4]     = -1.0;
  eventInfo->errors[5]     = -1.0;
  eventInfo->errors[6]     = -1.0;
  eventInfo->errors[7]     = -1.0;
  eventInfo->numMessages   = 0.0;
  eventInfo->messageLength = 0.0;
  eventInfo->numReductions = 0.0;
#if defined(PETSC_HAVE_DEVICE)
  eventInfo->CpuToGpuCount = 0.0;
  eventInfo->GpuToCpuCount = 0.0;
  eventInfo->CpuToGpuSize  = 0.0;
  eventInfo->GpuToCpuSize  = 0.0;
  eventInfo->GpuFlops      = 0.0;
  eventInfo->GpuTime       = 0.0;
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscEventPerfInfoCopy - Copy the activity and visibility data in eventInfo to outInfo

  Not collective

  Input Parameter:
. eventInfo - The input `PetscEventPerfInfo`

  Output Parameter:
. outInfo   - The output `PetscEventPerfInfo`

  Level: developer

  Note:
  This is a low level routine used by the logging functions in PETSc

.seealso: `PetscEventPerfInfoClear()`
@*/
PetscErrorCode PetscEventPerfInfoCopy(PetscEventPerfInfo *eventInfo, PetscEventPerfInfo *outInfo)
{
  PetscFunctionBegin;
  outInfo->id      = eventInfo->id;
  outInfo->active  = eventInfo->active;
  outInfo->visible = eventInfo->visible;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscEventPerfInfoAdd - Add data in eventInfo to outInfo

  Not collective

  Input Parameter:
. eventInfo - The input `PetscEventPerfInfo`

  Output Parameter:
. outInfo   - The output `PetscEventPerfInfo`

  Level: developer

  Note:
  This is a low level routine used by the logging functions in PETSc

.seealso: `PetscEventPerfInfoClear()`
@*/
PetscErrorCode PetscEventPerfInfoAdd(PetscEventPerfInfo *eventInfo, PetscEventPerfInfo *outInfo)
{
  PetscFunctionBegin;
  outInfo->count += eventInfo->count;
  outInfo->time += eventInfo->time;
  outInfo->time2 += eventInfo->time2;
  outInfo->flops += eventInfo->flops;
  outInfo->flops2 += eventInfo->flops2;
  outInfo->numMessages += eventInfo->numMessages;
  outInfo->messageLength += eventInfo->messageLength;
  outInfo->numReductions += eventInfo->numReductions;
#if defined(PETSC_HAVE_DEVICE)
  outInfo->CpuToGpuCount += eventInfo->CpuToGpuCount;
  outInfo->GpuToCpuCount += eventInfo->GpuToCpuCount;
  outInfo->CpuToGpuSize += eventInfo->CpuToGpuSize;
  outInfo->GpuToCpuSize += eventInfo->GpuToCpuSize;
  outInfo->GpuFlops += eventInfo->GpuFlops;
  outInfo->GpuTime += eventInfo->GpuTime;
#endif
  outInfo->memIncrease += eventInfo->memIncrease;
  outInfo->mallocSpace += eventInfo->mallocSpace;
  outInfo->mallocIncreaseEvent += eventInfo->mallocIncreaseEvent;
  outInfo->mallocIncrease += eventInfo->mallocIncrease;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscEventPerfLogEnsureSize - This ensures that a `PetscEventPerfLog` is at least of a certain size.

  Not collective

  Input Parameters:
+ eventLog - The `PetscEventPerfLog`
- size     - The size

  Level: developer

  Note:
  This is a low level routine used by the logging functions in PETSc

.seealso: `PetscEventPerfLogCreate()`
@*/
PetscErrorCode PetscEventPerfLogEnsureSize(PetscEventPerfLog eventLog, int size)
{
  PetscEventPerfInfo *eventInfo;

  PetscFunctionBegin;
  while (size > eventLog->maxEvents) {
    PetscCall(PetscCalloc1(eventLog->maxEvents * 2, &eventInfo));
    PetscCall(PetscArraycpy(eventInfo, eventLog->eventInfo, eventLog->maxEvents));
    PetscCall(PetscFree(eventLog->eventInfo));
    eventLog->eventInfo = eventInfo;
    eventLog->maxEvents *= 2;
  }
  while (eventLog->numEvents < size) PetscCall(PetscEventPerfInfoClear(&eventLog->eventInfo[eventLog->numEvents++]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if defined(PETSC_HAVE_MPE)
  #include <mpe.h>
PETSC_INTERN PetscErrorCode PetscLogMPEGetRGBColor(const char *[]);
PetscErrorCode              PetscLogEventBeginMPE(PetscLogEvent event, int t, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  PetscFunctionBegin;
  PetscCall(MPE_Log_event(petsc_stageLog->eventLog->eventInfo[event].mpe_id_begin, 0, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscLogEventEndMPE(PetscLogEvent event, int t, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  PetscFunctionBegin;
  PetscCall(MPE_Log_event(petsc_stageLog->eventLog->eventInfo[event].mpe_id_end, 0, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

/*--------------------------------------------- Registration Functions ----------------------------------------------*/
/*@C
  PetscEventRegLogRegister - Registers an event for logging operations in an application code.

  Not Collective

  Input Parameters:
+ eventLog - The `PetscEventLog`
. ename    - The name associated with the event
- classid   - The classid associated to the class for this event

  Output Parameter:
. event    - The event

  Example of Usage:
.vb
      int USER_EVENT;
      PetscLogDouble user_event_flops;
      PetscLogEventRegister("User event name",0,&USER_EVENT);
      PetscLogEventBegin(USER_EVENT,0,0,0,0);
         [code segment to monitor]
         PetscLogFlops(user_event_flops);
      PetscLogEventEnd(USER_EVENT,0,0,0,0);
.ve

  Level: developer

  Notes:
  PETSc can gather data for use with the utilities Jumpshot
  (part of the MPICH distribution).  If PETSc has been compiled
  with flag -DPETSC_HAVE_MPE (MPE is an additional utility within
  MPICH), the user can employ another command line option, -log_mpe,
  to create a logfile, "mpe.log", which can be visualized
  Jumpshot.

  This is a low level routine used by the logging functions in PETSc

.seealso: `PetscLogEventBegin()`, `PetscLogEventEnd()`, `PetscLogFlops()`,
          `PetscEventLogActivate()`, `PetscEventLogDeactivate()`
@*/
PetscErrorCode PetscEventRegLogRegister(PetscEventRegLog eventLog, const char ename[], PetscClassId classid, PetscLogEvent *event)
{
  PetscEventRegInfo *eventInfo;
  int                e;

  PetscFunctionBegin;
  PetscValidCharPointer(ename, 2);
  PetscValidIntPointer(event, 4);
  /* Should check classid I think */
  e = eventLog->numEvents++;
  if (eventLog->numEvents > eventLog->maxEvents) {
    PetscCall(PetscCalloc1(eventLog->maxEvents * 2, &eventInfo));
    PetscCall(PetscArraycpy(eventInfo, eventLog->eventInfo, eventLog->maxEvents));
    PetscCall(PetscFree(eventLog->eventInfo));
    eventLog->eventInfo = eventInfo;
    eventLog->maxEvents *= 2;
  }

  PetscCall(PetscStrallocpy(ename, &(eventLog->eventInfo[e].name)));
  eventLog->eventInfo[e].classid    = classid;
  eventLog->eventInfo[e].collective = PETSC_TRUE;
#if defined(PETSC_HAVE_TAU_PERFSTUBS)
  if (perfstubs_initialized == PERFSTUBS_SUCCESS) PetscStackCallExternalVoid("ps_timer_create_", eventLog->eventInfo[e].timer = ps_timer_create_(eventLog->eventInfo[e].name));
#endif
#if defined(PETSC_HAVE_MPE)
  if (PetscLogPLB == PetscLogEventBeginMPE) {
    const char *color;
    PetscMPIInt rank;
    int         beginID, endID;

    beginID = MPE_Log_get_event_number();
    endID   = MPE_Log_get_event_number();

    eventLog->eventInfo[e].mpe_id_begin = beginID;
    eventLog->eventInfo[e].mpe_id_end   = endID;

    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
    if (rank == 0) {
      PetscCall(PetscLogMPEGetRGBColor(&color));
      MPE_Describe_state(beginID, endID, eventLog->eventInfo[e].name, (char *)color);
    }
  }
#endif
  *event = e;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*---------------------------------------------- Activation Functions -----------------------------------------------*/
/*@C
  PetscEventPerfLogActivate - Indicates that a particular event should be logged.

  Not Collective

  Input Parameters:
+ eventLog - The `PetscEventPerfLog`
- event    - The event

   Usage:
.vb
      PetscEventPerfLogDeactivate(log, VEC_SetValues);
        [code where you do not want to log VecSetValues()]
      PetscEventPerfLogActivate(log, VEC_SetValues);
        [code where you do want to log VecSetValues()]
.ve

  Level: developer

  Notes:
  The event may be either a pre-defined PETSc event (found in
  include/petsclog.h) or an event number obtained with `PetscEventRegLogRegister()`.

  This is a low level routine used by the logging functions in PETSc

.seealso: `PetscEventPerfLogDeactivate()`, `PetscEventPerfLogDeactivatePop()`, `PetscEventPerfLogDeactivatePush()`
@*/
PetscErrorCode PetscEventPerfLogActivate(PetscEventPerfLog eventLog, PetscLogEvent event)
{
  PetscFunctionBegin;
  eventLog->eventInfo[event].active = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscEventPerfLogDeactivate - Indicates that a particular event should not be logged.

  Not Collective

  Input Parameters:
+ eventLog - The `PetscEventPerfLog`
- event    - The event

   Usage:
.vb
      PetscEventPerfLogDeactivate(log, VEC_SetValues);
        [code where you do not want to log VecSetValues()]
      PetscEventPerfLogActivate(log, VEC_SetValues);
        [code where you do want to log VecSetValues()]
.ve

   Level: developer

  Notes:
  The event may be either a pre-defined PETSc event (found in
  include/petsclog.h) or an event number obtained with `PetscEventRegLogRegister()`.

  This is a low level routine used by the logging functions in PETSc

.seealso: `PetscEventPerfLogActivate()`, `PetscEventPerfLogDeactivatePop()`, `PetscEventPerfLogDeactivatePush()`
@*/
PetscErrorCode PetscEventPerfLogDeactivate(PetscEventPerfLog eventLog, PetscLogEvent event)
{
  PetscFunctionBegin;
  eventLog->eventInfo[event].active = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscEventPerfLogDeactivatePush - Indicates that a particular event should not be logged.

  Not Collective

  Input Parameters:
+ eventLog - The `PetscEventPerfLog`
- event    - The event

   Usage:
.vb
      PetscEventPerfLogDeactivatePush(log, VEC_SetValues);
        [code where you do not want to log VecSetValues()]
      PetscEventPerfLogDeactivatePop(log, VEC_SetValues);
        [code where you do want to log VecSetValues()]
.ve

  Level: developer

  Notes:
  The event may be either a pre-defined PETSc event (found in
  include/petsclog.h) or an event number obtained with `PetscEventRegLogRegister()`.

  This is a low level routine used by the logging functions in PETSc

.seealso: `PetscEventPerfLogDeactivate()`, `PetscEventPerfLogActivate()`, `PetscEventPerfLogDeactivatePop()`
@*/
PetscErrorCode PetscEventPerfLogDeactivatePush(PetscEventPerfLog eventLog, PetscLogEvent event)
{
  PetscFunctionBegin;
  eventLog->eventInfo[event].depth++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscEventPerfLogDeactivatePop - Indicates that a particular event should  be logged.

  Not Collective

  Input Parameters:
+ eventLog - The `PetscEventPerfLog`
- event    - The event

   Usage:
.vb
      PetscEventPerfLogDeactivatePush(log, VEC_SetValues);
        [code where you do not want to log VecSetValues()]
      PetscEventPerfLogDeactivatePop(log, VEC_SetValues);
        [code where you do want to log VecSetValues()]
.ve

  Level: developer

  Notes:
  The event may be either a pre-defined PETSc event (found in
  include/petsclog.h) or an event number obtained with `PetscEventRegLogRegister()`.

  This is a low level routine used by the logging functions in PETSc

.seealso: `PetscEventPerfLogDeactivate()`, `PetscEventPerfLogActivate()`, `PetscEventPerfLogDeactivatePush()`
@*/
PetscErrorCode PetscEventPerfLogDeactivatePop(PetscEventPerfLog eventLog, PetscLogEvent event)
{
  PetscFunctionBegin;
  eventLog->eventInfo[event].depth--;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscEventPerfLogActivateClass - Activates event logging for a PETSc object class.

  Not Collective

  Input Parameters:
+ eventLog    - The `PetscEventPerfLog`
. eventRegLog - The `PetscEventRegLog`
- classid      - The class id, for example `MAT_CLASSID`, `SNES_CLASSID`

  Level: developer

  Note:
  This is a low level routine used by the logging functions in PETSc

.seealso: `PetscEventPerfLogDeactivateClass()`, `PetscEventPerfLogActivate()`, `PetscEventPerfLogDeactivate()`
@*/
PetscErrorCode PetscEventPerfLogActivateClass(PetscEventPerfLog eventLog, PetscEventRegLog eventRegLog, PetscClassId classid)
{
  int e;

  PetscFunctionBegin;
  for (e = 0; e < eventLog->numEvents; e++) {
    int c = eventRegLog->eventInfo[e].classid;
    if (c == classid) eventLog->eventInfo[e].active = PETSC_TRUE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscEventPerfLogDeactivateClass - Deactivates event logging for a PETSc object class.

  Not Collective

  Input Parameters:
+ eventLog    - The `PetscEventPerfLog`
. eventRegLog - The `PetscEventRegLog`
- classid - The class id, for example `MAT_CLASSID`, `SNES_CLASSID`

  Level: developer

  Note:
  This is a low level routine used by the logging functions in PETSc

.seealso: `PetscEventPerfLogDeactivateClass()`, `PetscEventPerfLogDeactivate()`, `PetscEventPerfLogActivate()`
@*/
PetscErrorCode PetscEventPerfLogDeactivateClass(PetscEventPerfLog eventLog, PetscEventRegLog eventRegLog, PetscClassId classid)
{
  int e;

  PetscFunctionBegin;
  for (e = 0; e < eventLog->numEvents; e++) {
    int c = eventRegLog->eventInfo[e].classid;
    if (c == classid) eventLog->eventInfo[e].active = PETSC_FALSE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------ Query Functions --------------------------------------------------*/
/*@C
  PetscEventRegLogGetEvent - This function returns the event id given the event name.

  Not Collective

  Input Parameters:
+ eventLog - The `PetscEventRegLog`
- name     - The stage name

  Output Parameter:
. event    - The event id, or -1 if not found

  Level: developer

  Note:
  This is a low level routine used by the logging functions in PETSc

.seealso: `PetscEventRegLogRegister()`
@*/
PetscErrorCode PetscEventRegLogGetEvent(PetscEventRegLog eventLog, const char name[], PetscLogEvent *event)
{
  PetscBool match;
  int       e;

  PetscFunctionBegin;
  PetscValidCharPointer(name, 2);
  PetscValidIntPointer(event, 3);
  *event = -1;
  for (e = 0; e < eventLog->numEvents; e++) {
    PetscCall(PetscStrcasecmp(eventLog->eventInfo[e].name, name, &match));
    if (match) {
      *event = e;
      break;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscEventPerfLogSetVisible - This function determines whether an event is printed during `PetscLogView()`

  Not Collective

  Input Parameters:
+ eventLog  - The `PetscEventPerfLog`
. event     - The event to log
- isVisible - The visibility flag, `PETSC_TRUE` for printing, otherwise `PETSC_FALSE` (default is `PETSC_TRUE`)

  Options Database Key:
. -log_view - Activates log summary

  Level: developer

  Note:
  This is a low level routine used by the logging functions in PETSc

.seealso: `PetscEventPerfLogGetVisible()`, `PetscEventRegLogRegister()`, `PetscStageLogGetEventLog()`
@*/
PetscErrorCode PetscEventPerfLogSetVisible(PetscEventPerfLog eventLog, PetscLogEvent event, PetscBool isVisible)
{
  PetscFunctionBegin;
  eventLog->eventInfo[event].visible = isVisible;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscEventPerfLogGetVisible - This function returns whether an event is printed during `PetscLogView()`

  Not Collective

  Input Parameters:
+ eventLog  - The `PetscEventPerfLog`
- event     - The event id to log

  Output Parameter:
. isVisible - The visibility flag, `PETSC_TRUE` for printing, otherwise `PETSC_FALSE` (default is `PETSC_TRUE`)

  Options Database Key:
. -log_view - Activates log summary

  Level: developer

  Note:
  This is a low level routine used by the logging functions in PETSc

.seealso: `PetscEventPerfLogSetVisible()`, `PetscEventRegLogRegister()`, `PetscStageLogGetEventLog()`
@*/
PetscErrorCode PetscEventPerfLogGetVisible(PetscEventPerfLog eventLog, PetscLogEvent event, PetscBool *isVisible)
{
  PetscFunctionBegin;
  PetscValidBoolPointer(isVisible, 3);
  *isVisible = eventLog->eventInfo[event].visible;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscLogEventGetPerfInfo - Return the performance information about the given event in the given stage

  Input Parameters:
+ stage - The stage number or `PETSC_DETERMINE` for the current stage
- event - The event number

  Output Parameter:
. info - This structure is filled with the performance information

  Level: Intermediate

  Note:
  This is a low level routine used by the logging functions in PETSc

.seealso: `PetscLogEventGetFlops()`
@*/
PetscErrorCode PetscLogEventGetPerfInfo(int stage, PetscLogEvent event, PetscEventPerfInfo *info)
{
  PetscStageLog     stageLog;
  PetscEventPerfLog eventLog = NULL;

  PetscFunctionBegin;
  PetscValidPointer(info, 3);
  PetscCheck(PetscLogPLB, PETSC_COMM_SELF, PETSC_ERR_SUP, "Must use -log_view or PetscLogDefaultBegin() before calling this routine");
  PetscCall(PetscLogGetStageLog(&stageLog));
  if (stage < 0) PetscCall(PetscStageLogGetCurrent(stageLog, &stage));
  PetscCall(PetscStageLogGetEventPerfLog(stageLog, stage, &eventLog));
  *info = eventLog->eventInfo[event];
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscLogEventGetFlops(PetscLogEvent event, PetscLogDouble *flops)
{
  PetscStageLog     stageLog;
  PetscEventPerfLog eventLog = NULL;
  int               stage;

  PetscFunctionBegin;
  PetscCheck(PetscLogPLB, PETSC_COMM_SELF, PETSC_ERR_SUP, "Must use -log_view or PetscLogDefaultBegin() before calling this routine");
  PetscCall(PetscLogGetStageLog(&stageLog));
  PetscCall(PetscStageLogGetCurrent(stageLog, &stage));
  PetscCall(PetscStageLogGetEventPerfLog(stageLog, stage, &eventLog));
  *flops = eventLog->eventInfo[event].flops;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscLogEventZeroFlops(PetscLogEvent event)
{
  PetscStageLog     stageLog;
  PetscEventPerfLog eventLog = NULL;
  int               stage;

  PetscFunctionBegin;
  PetscCall(PetscLogGetStageLog(&stageLog));
  PetscCall(PetscStageLogGetCurrent(stageLog, &stage));
  PetscCall(PetscStageLogGetEventPerfLog(stageLog, stage, &eventLog));

  eventLog->eventInfo[event].flops    = 0.0;
  eventLog->eventInfo[event].flops2   = 0.0;
  eventLog->eventInfo[event].flopsTmp = 0.0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscLogEventSynchronize(PetscLogEvent event, MPI_Comm comm)
{
  PetscStageLog     stageLog;
  PetscEventRegLog  eventRegLog;
  PetscEventPerfLog eventLog = NULL;
  int               stage;
  PetscLogDouble    time = 0.0;

  PetscFunctionBegin;
  if (!PetscLogSyncOn || comm == MPI_COMM_NULL) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscLogGetStageLog(&stageLog));
  PetscCall(PetscStageLogGetEventRegLog(stageLog, &eventRegLog));
  if (!eventRegLog->eventInfo[event].collective) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscStageLogGetCurrent(stageLog, &stage));
  PetscCall(PetscStageLogGetEventPerfLog(stageLog, stage, &eventLog));
  if (eventLog->eventInfo[event].depth > 0) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscTimeSubtract(&time));
  PetscCallMPI(MPI_Barrier(comm));
  PetscCall(PetscTimeAdd(&time));
  eventLog->eventInfo[event].syncTime += time;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if defined(PETSC_HAVE_CUDA)
  #include <nvToolsExt.h>
#endif
#if defined(PETSC_HAVE_THREADSAFETY)
static PetscErrorCode PetscLogGetStageEventPerfInfo_threaded(PetscLogEvent event, PetscLogStage stage, PetscEventPerfInfo **eventInfo)
{
  PetscHashIJKKey     key;
  PetscEventPerfInfo *leventInfo;

  PetscFunctionBegin;
  key.i = PetscLogGetTid();
  key.j = stage;
  key.k = event;
  PetscCall(PetscHMapEventGet(eventInfoMap_th, key, &leventInfo));
  if (!leventInfo) {
    PetscCall(PetscSpinlockLock(&PetscLogSpinLock));
    PetscCall(PetscNew(&leventInfo));
    PetscCall(PetscHMapEventSet(eventInfoMap_th, key, leventInfo));
    PetscCall(PetscSpinlockUnlock(&PetscLogSpinLock));
  }
  *eventInfo = leventInfo;
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

PetscErrorCode PetscLogEventBeginDefault(PetscLogEvent event, int t, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  PetscStageLog       stageLog;
  PetscEventPerfLog   eventLog  = NULL;
  PetscEventPerfInfo *eventInfo = NULL;
  int                 stage;

  PetscFunctionBegin;
  /* Synchronization */
  PetscCall(PetscLogEventSynchronize(event, PetscObjectComm(o1)));
  PetscCall(PetscLogGetStageLog(&stageLog));
  PetscCall(PetscStageLogGetCurrent(stageLog, &stage));
  PetscCall(PetscStageLogGetEventPerfLog(stageLog, stage, &eventLog));
#if defined(PETSC_HAVE_THREADSAFETY)
  PetscCall(PetscLogGetStageEventPerfInfo_threaded(stage, event, &eventInfo));
  if (eventInfo->depth == 0) {
    PetscCall(PetscEventPerfInfoClear(eventInfo));
    PetscCall(PetscEventPerfInfoCopy(eventLog->eventInfo + event, eventInfo));
  }
#else
  eventInfo = eventLog->eventInfo + event;
#endif
  /* Check for double counting */
  eventInfo->depth++;
  if (eventInfo->depth > 1) PetscFunctionReturn(PETSC_SUCCESS);
#if defined(PETSC_HAVE_CUDA)
  if (PetscDeviceInitialized(PETSC_DEVICE_CUDA)) {
    PetscEventRegLog eventRegLog;
    PetscCall(PetscStageLogGetEventRegLog(stageLog, &eventRegLog));
    nvtxRangePushA(eventRegLog->eventInfo[event].name);
  }
#endif
  /* Log the performance info */
#if defined(PETSC_HAVE_TAU_PERFSTUBS)
  if (perfstubs_initialized == PERFSTUBS_SUCCESS) {
    PetscEventRegLog regLog = NULL;
    PetscCall(PetscStageLogGetEventRegLog(stageLog, &regLog));
    if (regLog->eventInfo[event].timer != NULL) PetscStackCallExternalVoid("ps_timer_start_", ps_timer_start_(regLog->eventInfo[event].timer));
  }
#endif
  eventInfo->count++;
  eventInfo->timeTmp = 0.0;
  PetscCall(PetscTimeSubtract(&eventInfo->timeTmp));
  eventInfo->flopsTmp = -petsc_TotalFlops_th;
  eventInfo->numMessages -= petsc_irecv_ct_th + petsc_isend_ct_th + petsc_recv_ct_th + petsc_send_ct_th;
  eventInfo->messageLength -= petsc_irecv_len_th + petsc_isend_len_th + petsc_recv_len_th + petsc_send_len_th;
  eventInfo->numReductions -= petsc_allreduce_ct_th + petsc_gather_ct_th + petsc_scatter_ct_th;
#if defined(PETSC_HAVE_DEVICE)
  eventInfo->CpuToGpuCount -= petsc_ctog_ct_th;
  eventInfo->GpuToCpuCount -= petsc_gtoc_ct_th;
  eventInfo->CpuToGpuSize -= petsc_ctog_sz_th;
  eventInfo->GpuToCpuSize -= petsc_gtoc_sz_th;
  eventInfo->GpuFlops -= petsc_gflops_th;
  eventInfo->GpuTime -= petsc_gtime_th;
#endif
  if (PetscLogMemory) {
    PetscLogDouble usage;
    PetscCall(PetscMemoryGetCurrentUsage(&usage));
    eventInfo->memIncrease -= usage;
    PetscCall(PetscMallocGetCurrentUsage(&usage));
    eventInfo->mallocSpace -= usage;
    PetscCall(PetscMallocGetMaximumUsage(&usage));
    eventInfo->mallocIncrease -= usage;
    PetscCall(PetscMallocPushMaximumUsage((int)event));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscLogEventEndDefault(PetscLogEvent event, int t, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  PetscStageLog       stageLog;
  PetscEventPerfLog   eventLog  = NULL;
  PetscEventPerfInfo *eventInfo = NULL;
  int                 stage;

  PetscFunctionBegin;
  PetscCall(PetscLogGetStageLog(&stageLog));
  PetscCall(PetscStageLogGetCurrent(stageLog, &stage));
  PetscCall(PetscStageLogGetEventPerfLog(stageLog, stage, &eventLog));
#if defined(PETSC_HAVE_THREADSAFETY)
  PetscCall(PetscLogGetStageEventPerfInfo_threaded(stage, event, &eventInfo));
#else
  eventInfo = eventLog->eventInfo + event;
#endif
  /* Check for double counting */
  eventInfo->depth--;
  if (eventInfo->depth > 0) PetscFunctionReturn(PETSC_SUCCESS);
  else PetscCheck(eventInfo->depth == 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Logging event had unbalanced begin/end pairs");

    /* Log performance info */
#if defined(PETSC_HAVE_TAU_PERFSTUBS)
  if (perfstubs_initialized == PERFSTUBS_SUCCESS) {
    PetscEventRegLog regLog = NULL;
    PetscCall(PetscStageLogGetEventRegLog(stageLog, &regLog));
    if (regLog->eventInfo[event].timer != NULL) PetscStackCallExternalVoid("ps_timer_stop_", ps_timer_stop_(regLog->eventInfo[event].timer));
  }
#endif
  PetscCall(PetscTimeAdd(&eventInfo->timeTmp));
  eventInfo->flopsTmp += petsc_TotalFlops_th;
  eventInfo->time += eventInfo->timeTmp;
  eventInfo->time2 += eventInfo->timeTmp * eventInfo->timeTmp;
  eventInfo->flops += eventInfo->flopsTmp;
  eventInfo->flops2 += eventInfo->flopsTmp * eventInfo->flopsTmp;
  eventInfo->numMessages += petsc_irecv_ct_th + petsc_isend_ct_th + petsc_recv_ct_th + petsc_send_ct_th;
  eventInfo->messageLength += petsc_irecv_len_th + petsc_isend_len_th + petsc_recv_len + petsc_send_len_th;
  eventInfo->numReductions += petsc_allreduce_ct_th + petsc_gather_ct_th + petsc_scatter_ct_th;
#if defined(PETSC_HAVE_DEVICE)
  eventInfo->CpuToGpuCount += petsc_ctog_ct_th;
  eventInfo->GpuToCpuCount += petsc_gtoc_ct_th;
  eventInfo->CpuToGpuSize += petsc_ctog_sz_th;
  eventInfo->GpuToCpuSize += petsc_gtoc_sz_th;
  eventInfo->GpuFlops += petsc_gflops_th;
  eventInfo->GpuTime += petsc_gtime_th;
#endif
  if (PetscLogMemory) {
    PetscLogDouble usage, musage;
    PetscCall(PetscMemoryGetCurrentUsage(&usage)); /* the comments below match the column labels printed in PetscLogView_Default() */
    eventInfo->memIncrease += usage;               /* RMI */
    PetscCall(PetscMallocGetCurrentUsage(&usage));
    eventInfo->mallocSpace += usage; /* Malloc */
    PetscCall(PetscMallocPopMaximumUsage((int)event, &musage));
    eventInfo->mallocIncreaseEvent = PetscMax(musage - usage, eventInfo->mallocIncreaseEvent); /* EMalloc */
    PetscCall(PetscMallocGetMaximumUsage(&usage));
    eventInfo->mallocIncrease += usage; /* MMalloc */
  }
#if defined(PETSC_HAVE_THREADSAFETY)
  PetscCall(PetscSpinlockLock(&PetscLogSpinLock));
  PetscCall(PetscEventPerfInfoAdd(eventInfo, eventLog->eventInfo + event));
  PetscCall(PetscSpinlockUnlock(&PetscLogSpinLock));
#endif
#if defined(PETSC_HAVE_CUDA)
  if (PetscDeviceInitialized(PETSC_DEVICE_CUDA)) nvtxRangePop();
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscLogEventBeginComplete(PetscLogEvent event, int t, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  PetscStageLog     stageLog;
  PetscEventRegLog  eventRegLog;
  PetscEventPerfLog eventPerfLog = NULL;
  Action           *tmpAction;
  PetscLogDouble    start, end;
  PetscLogDouble    curTime;
  int               stage;

  PetscFunctionBegin;
  /* Dynamically enlarge logging structures */
  if (petsc_numActions >= petsc_maxActions) {
    PetscCall(PetscTime(&start));
    PetscCall(PetscCalloc1(petsc_maxActions * 2, &tmpAction));
    PetscCall(PetscArraycpy(tmpAction, petsc_actions, petsc_maxActions));
    PetscCall(PetscFree(petsc_actions));

    petsc_actions = tmpAction;
    petsc_maxActions *= 2;
    PetscCall(PetscTime(&end));
    petsc_BaseTime += (end - start);
  }
  /* Record the event */
  PetscCall(PetscLogGetStageLog(&stageLog));
  PetscCall(PetscStageLogGetCurrent(stageLog, &stage));
  PetscCall(PetscStageLogGetEventRegLog(stageLog, &eventRegLog));
  PetscCall(PetscStageLogGetEventPerfLog(stageLog, stage, &eventPerfLog));
  PetscCall(PetscTime(&curTime));
  if (petsc_logActions) {
    petsc_actions[petsc_numActions].time    = curTime - petsc_BaseTime;
    petsc_actions[petsc_numActions].action  = ACTIONBEGIN;
    petsc_actions[petsc_numActions].event   = event;
    petsc_actions[petsc_numActions].classid = eventRegLog->eventInfo[event].classid;
    if (o1) petsc_actions[petsc_numActions].id1 = o1->id;
    else petsc_actions[petsc_numActions].id1 = -1;
    if (o2) petsc_actions[petsc_numActions].id2 = o2->id;
    else petsc_actions[petsc_numActions].id2 = -1;
    if (o3) petsc_actions[petsc_numActions].id3 = o3->id;
    else petsc_actions[petsc_numActions].id3 = -1;
    petsc_actions[petsc_numActions].flops = petsc_TotalFlops;

    PetscCall(PetscMallocGetCurrentUsage(&petsc_actions[petsc_numActions].mem));
    PetscCall(PetscMallocGetMaximumUsage(&petsc_actions[petsc_numActions].maxmem));
    petsc_numActions++;
  }
  /* Check for double counting */
  eventPerfLog->eventInfo[event].depth++;
  if (eventPerfLog->eventInfo[event].depth > 1) PetscFunctionReturn(PETSC_SUCCESS);
  /* Log the performance info */
  eventPerfLog->eventInfo[event].count++;
  eventPerfLog->eventInfo[event].time -= curTime;
  eventPerfLog->eventInfo[event].flops -= petsc_TotalFlops;
  eventPerfLog->eventInfo[event].numMessages -= petsc_irecv_ct + petsc_isend_ct + petsc_recv_ct + petsc_send_ct;
  eventPerfLog->eventInfo[event].messageLength -= petsc_irecv_len + petsc_isend_len + petsc_recv_len + petsc_send_len;
  eventPerfLog->eventInfo[event].numReductions -= petsc_allreduce_ct + petsc_gather_ct + petsc_scatter_ct;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscLogEventEndComplete(PetscLogEvent event, int t, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  PetscStageLog     stageLog;
  PetscEventRegLog  eventRegLog;
  PetscEventPerfLog eventPerfLog = NULL;
  Action           *tmpAction;
  PetscLogDouble    start, end;
  PetscLogDouble    curTime;
  int               stage;

  PetscFunctionBegin;
  /* Dynamically enlarge logging structures */
  if (petsc_numActions >= petsc_maxActions) {
    PetscCall(PetscTime(&start));
    PetscCall(PetscCalloc1(petsc_maxActions * 2, &tmpAction));
    PetscCall(PetscArraycpy(tmpAction, petsc_actions, petsc_maxActions));
    PetscCall(PetscFree(petsc_actions));

    petsc_actions = tmpAction;
    petsc_maxActions *= 2;
    PetscCall(PetscTime(&end));
    petsc_BaseTime += (end - start);
  }
  /* Record the event */
  PetscCall(PetscLogGetStageLog(&stageLog));
  PetscCall(PetscStageLogGetCurrent(stageLog, &stage));
  PetscCall(PetscStageLogGetEventRegLog(stageLog, &eventRegLog));
  PetscCall(PetscStageLogGetEventPerfLog(stageLog, stage, &eventPerfLog));
  PetscCall(PetscTime(&curTime));
  if (petsc_logActions) {
    petsc_actions[petsc_numActions].time    = curTime - petsc_BaseTime;
    petsc_actions[petsc_numActions].action  = ACTIONEND;
    petsc_actions[petsc_numActions].event   = event;
    petsc_actions[petsc_numActions].classid = eventRegLog->eventInfo[event].classid;
    if (o1) petsc_actions[petsc_numActions].id1 = o1->id;
    else petsc_actions[petsc_numActions].id1 = -1;
    if (o2) petsc_actions[petsc_numActions].id2 = o2->id;
    else petsc_actions[petsc_numActions].id2 = -1;
    if (o3) petsc_actions[petsc_numActions].id3 = o3->id;
    else petsc_actions[petsc_numActions].id3 = -1;
    petsc_actions[petsc_numActions].flops = petsc_TotalFlops;

    PetscCall(PetscMallocGetCurrentUsage(&petsc_actions[petsc_numActions].mem));
    PetscCall(PetscMallocGetMaximumUsage(&petsc_actions[petsc_numActions].maxmem));
    petsc_numActions++;
  }
  /* Check for double counting */
  eventPerfLog->eventInfo[event].depth--;
  if (eventPerfLog->eventInfo[event].depth > 0) PetscFunctionReturn(PETSC_SUCCESS);
  else PetscCheck(eventPerfLog->eventInfo[event].depth >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Logging event had unbalanced begin/end pairs");
  /* Log the performance info */
  eventPerfLog->eventInfo[event].count++;
  eventPerfLog->eventInfo[event].time += curTime;
  eventPerfLog->eventInfo[event].flops += petsc_TotalFlops;
  eventPerfLog->eventInfo[event].numMessages += petsc_irecv_ct + petsc_isend_ct + petsc_recv_ct + petsc_send_ct;
  eventPerfLog->eventInfo[event].messageLength += petsc_irecv_len + petsc_isend_len + petsc_recv_len + petsc_send_len;
  eventPerfLog->eventInfo[event].numReductions += petsc_allreduce_ct + petsc_gather_ct + petsc_scatter_ct;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscLogEventBeginTrace(PetscLogEvent event, int t, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  PetscStageLog     stageLog;
  PetscEventRegLog  eventRegLog;
  PetscEventPerfLog eventPerfLog = NULL;
  PetscLogDouble    cur_time;
  PetscMPIInt       rank;
  int               stage;

  PetscFunctionBegin;
  if (!petsc_tracetime) PetscCall(PetscTime(&petsc_tracetime));

  petsc_tracelevel++;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCall(PetscLogGetStageLog(&stageLog));
  PetscCall(PetscStageLogGetCurrent(stageLog, &stage));
  PetscCall(PetscStageLogGetEventRegLog(stageLog, &eventRegLog));
  PetscCall(PetscStageLogGetEventPerfLog(stageLog, stage, &eventPerfLog));
  /* Check for double counting */
  eventPerfLog->eventInfo[event].depth++;
  if (eventPerfLog->eventInfo[event].depth > 1) PetscFunctionReturn(PETSC_SUCCESS);
  /* Log performance info */
  PetscCall(PetscTime(&cur_time));
  PetscCall(PetscFPrintf(PETSC_COMM_SELF, petsc_tracefile, "%s[%d] %g Event begin: %s\n", petsc_tracespace, rank, cur_time - petsc_tracetime, eventRegLog->eventInfo[event].name));
  PetscCall(PetscStrncpy(petsc_tracespace, petsc_traceblanks, 2 * petsc_tracelevel));

  petsc_tracespace[2 * petsc_tracelevel] = 0;

  PetscCall(PetscFFlush(petsc_tracefile));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscLogEventEndTrace(PetscLogEvent event, int t, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  PetscStageLog     stageLog;
  PetscEventRegLog  eventRegLog;
  PetscEventPerfLog eventPerfLog = NULL;
  PetscLogDouble    cur_time;
  int               stage;
  PetscMPIInt       rank;

  PetscFunctionBegin;
  petsc_tracelevel--;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCall(PetscLogGetStageLog(&stageLog));
  PetscCall(PetscStageLogGetCurrent(stageLog, &stage));
  PetscCall(PetscStageLogGetEventRegLog(stageLog, &eventRegLog));
  PetscCall(PetscStageLogGetEventPerfLog(stageLog, stage, &eventPerfLog));
  /* Check for double counting */
  eventPerfLog->eventInfo[event].depth--;
  if (eventPerfLog->eventInfo[event].depth > 0) PetscFunctionReturn(PETSC_SUCCESS);
  else PetscCheck(eventPerfLog->eventInfo[event].depth >= 0 && petsc_tracelevel >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Logging event had unbalanced begin/end pairs");

  /* Log performance info */
  if (petsc_tracelevel) PetscCall(PetscStrncpy(petsc_tracespace, petsc_traceblanks, 2 * petsc_tracelevel));
  petsc_tracespace[2 * petsc_tracelevel] = 0;
  PetscCall(PetscTime(&cur_time));
  PetscCall(PetscFPrintf(PETSC_COMM_SELF, petsc_tracefile, "%s[%d] %g Event end: %s\n", petsc_tracespace, rank, cur_time - petsc_tracetime, eventRegLog->eventInfo[event].name));
  PetscCall(PetscFFlush(petsc_tracefile));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
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

.seealso: `PetscLogEventSetError()`, `PetscEventRegLogRegister()`, `PetscStageLogGetEventLog()`
@*/
PetscErrorCode PetscLogEventSetDof(PetscLogEvent event, PetscInt n, PetscLogDouble dof)
{
  PetscStageLog     stageLog;
  PetscEventPerfLog eventLog = NULL;
  int               stage;

  PetscFunctionBegin;
  PetscCheck(!(n < 0) && !(n > 7), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Error index %" PetscInt_FMT " is not in [0, 8)", n);
  PetscCall(PetscLogGetStageLog(&stageLog));
  PetscCall(PetscStageLogGetCurrent(stageLog, &stage));
  PetscCall(PetscStageLogGetEventPerfLog(stageLog, stage, &eventLog));
  eventLog->eventInfo[event].dof[n] = dof;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
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

.seealso: `PetscLogEventSetDof()`, `PetscEventRegLogRegister()`, `PetscStageLogGetEventLog()`
@*/
PetscErrorCode PetscLogEventSetError(PetscLogEvent event, PetscInt n, PetscLogDouble error)
{
  PetscStageLog     stageLog;
  PetscEventPerfLog eventLog = NULL;
  int               stage;

  PetscFunctionBegin;
  PetscCheck(!(n < 0) && !(n > 7), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Error index %" PetscInt_FMT " is not in [0, 8)", n);
  PetscCall(PetscLogGetStageLog(&stageLog));
  PetscCall(PetscStageLogGetCurrent(stageLog, &stage));
  PetscCall(PetscStageLogGetEventPerfLog(stageLog, stage, &eventLog));
  eventLog->eventInfo[event].errors[n] = error;
  PetscFunctionReturn(PETSC_SUCCESS);
}
