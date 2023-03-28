
/*
     This defines part of the private API for logging performance information. It is intended to be used only by the
   PETSc PetscLog...() interface and not elsewhere, nor by users. Hence the prototypes for these functions are NOT
   in the public PETSc include files.

*/
#include <petsc/private/logimpl.h> /*I    "petscsys.h"   I*/

PetscStageLog petsc_stageLog = NULL;

/*@C
  PetscLogGetStageLog - This function returns the default stage logging object.

  Not collective

  Output Parameter:
. stageLog - The default PetscStageLog

  Level: developer

  Developer Note:
    Inline since called for EACH `PetscEventLogBeginDefault()` and `PetscEventLogEndDefault()`

.seealso: `PetscStageLogCreate()`
@*/
PetscErrorCode PetscLogGetStageLog(PetscStageLog *stageLog)
{
  PetscFunctionBegin;
  PetscValidPointer(stageLog, 1);
  if (!petsc_stageLog) {
    fprintf(stderr, "PETSC ERROR: Logging has not been enabled.\nYou might have forgotten to call PetscInitialize().\n");
    PETSCABORT(MPI_COMM_WORLD, PETSC_ERR_SUP);
  }
  *stageLog = petsc_stageLog;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscStageLogGetCurrent - This function returns the stage from the top of the stack.

  Not Collective

  Input Parameter:
. stageLog - The `PetscStageLog`

  Output Parameter:
. stage    - The current stage

  Note:
  If no stage is currently active, stage is set to -1.

  Level: developer

  Developer Note:
    Inline since called for EACH `PetscEventLogBeginDefault()` and `PetscEventLogEndDefault()`

.seealso: `PetscStageLogPush()`, `PetscStageLogPop()`, `PetscLogGetStageLog()`
@*/
PetscErrorCode PetscStageLogGetCurrent(PetscStageLog stageLog, int *stage)
{
  PetscBool empty;

  PetscFunctionBegin;
  PetscCall(PetscIntStackEmpty(stageLog->stack, &empty));
  if (empty) {
    *stage = -1;
  } else {
    PetscCall(PetscIntStackTop(stageLog->stack, stage));
  }
  PetscCheck(*stage == stageLog->curStage, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Inconsistency in stage log: stage %d should be %d", *stage, stageLog->curStage);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscStageLogGetEventPerfLog - This function returns the `PetscEventPerfLog` for the given stage.

  Not Collective

  Input Parameters:
+ stageLog - The `PetscStageLog`
- stage    - The stage

  Output Parameter:
. eventLog - The `PetscEventPerfLog`

  Level: developer

  Developer Note:
    Inline since called for EACH `PetscEventLogBeginDefault()` and `PetscEventLogEndDefault()`

.seealso: `PetscStageLogPush()`, `PetscStageLogPop()`, `PetscLogGetStageLog()`
@*/
PetscErrorCode PetscStageLogGetEventPerfLog(PetscStageLog stageLog, int stage, PetscEventPerfLog *eventLog)
{
  PetscFunctionBegin;
  PetscValidPointer(eventLog, 3);
  PetscCheck(!(stage < 0) && !(stage >= stageLog->numStages), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid stage %d should be in [0,%d)", stage, stageLog->numStages);
  *eventLog = stageLog->stageInfo[stage].eventLog;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscStageInfoDestroy - This destroys a `PetscStageInfo` object.

  Not collective

  Input Parameter:
. stageInfo - The `PetscStageInfo`

  Level: developer

.seealso: `PetscStageLogCreate()`
@*/
PetscErrorCode PetscStageInfoDestroy(PetscStageInfo *stageInfo)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(stageInfo->name));
  PetscCall(PetscEventPerfLogDestroy(stageInfo->eventLog));
  PetscCall(PetscClassPerfLogDestroy(stageInfo->classLog));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscStageLogDestroy - This destroys a `PetscStageLog` object.

  Not collective

  Input Parameter:
. stageLog - The `PetscStageLog`

  Level: developer

.seealso: `PetscStageLogCreate()`
@*/
PetscErrorCode PetscStageLogDestroy(PetscStageLog stageLog)
{
  int stage;

  PetscFunctionBegin;
  if (!stageLog) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscIntStackDestroy(stageLog->stack));
  PetscCall(PetscEventRegLogDestroy(stageLog->eventLog));
  PetscCall(PetscClassRegLogDestroy(stageLog->classLog));
  for (stage = 0; stage < stageLog->numStages; stage++) PetscCall(PetscStageInfoDestroy(&stageLog->stageInfo[stage]));
  PetscCall(PetscFree(stageLog->stageInfo));
  PetscCall(PetscFree(stageLog));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscStageLogRegister - Registers a stage name for logging operations in an application code.

  Not Collective

  Input Parameters:
+ stageLog - The `PetscStageLog`
- sname    - the name to associate with that stage

  Output Parameter:
. stage    - The stage index

  Level: developer

.seealso: `PetscStageLogPush()`, `PetscStageLogPop()`, `PetscStageLogCreate()`
@*/
PetscErrorCode PetscStageLogRegister(PetscStageLog stageLog, const char sname[], int *stage)
{
  PetscStageInfo *stageInfo;
  int             s;

  PetscFunctionBegin;
  PetscValidCharPointer(sname, 2);
  PetscValidIntPointer(stage, 3);
  /* Check stage already registered */
  for (s = 0; s < stageLog->numStages; ++s) {
    PetscBool same;

    PetscCall(PetscStrcmp(stageLog->stageInfo[s].name, sname, &same));
    PetscCheck(!same, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Duplicate stage name given: %s", sname);
  }
  /* Create new stage */
  s = stageLog->numStages++;
  if (stageLog->numStages > stageLog->maxStages) {
    PetscCall(PetscMalloc1(stageLog->maxStages * 2, &stageInfo));
    PetscCall(PetscArraycpy(stageInfo, stageLog->stageInfo, stageLog->maxStages));
    PetscCall(PetscFree(stageLog->stageInfo));
    stageLog->stageInfo = stageInfo;
    stageLog->maxStages *= 2;
  }
  /* Setup new stage info */
  stageInfo = &stageLog->stageInfo[s];
  PetscCall(PetscMemzero(stageInfo, sizeof(PetscStageInfo)));
  PetscCall(PetscStrallocpy(sname, &stageInfo->name));
  stageInfo->used             = PETSC_FALSE;
  stageInfo->perfInfo.active  = PETSC_TRUE;
  stageInfo->perfInfo.visible = PETSC_TRUE;
  PetscCall(PetscEventPerfLogCreate(&stageInfo->eventLog));
  PetscCall(PetscClassPerfLogCreate(&stageInfo->classLog));
  *stage = s;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscStageLogPush - This function pushes a stage on the stack.

  Not Collective

  Input Parameters:
+ stageLog   - The `PetscStageLog`
- stage - The stage to log

  Options Database Key:
. -log_view - Activates logging

  Usage:
  If the option -log_view is used to run the program containing the
  following code, then 2 sets of summary data will be printed during
  `PetscFinalize()`.
.vb
      PetscInitialize(int *argc,char ***args,0,0);
      [stage 0 of code]
      PetscStageLogPush(stageLog,1);
      [stage 1 of code]
      PetscStageLogPop(stageLog);
      PetscBarrier(...);
      [more stage 0 of code]
      PetscFinalize();
.ve

  Note;
  Use `PetscLogStageRegister()` to register a stage. All previous stages are
  accumulating time and flops, but events will only be logged in this stage.

  Level: developer

.seealso: `PetscStageLogPop()`, `PetscStageLogGetCurrent()`, `PetscStageLogRegister()`, `PetscLogGetStageLog()`
@*/
PetscErrorCode PetscStageLogPush(PetscStageLog stageLog, int stage)
{
  int       curStage = 0;
  PetscBool empty;

  PetscFunctionBegin;
  PetscCheck(!(stage < 0) && !(stage >= stageLog->numStages), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid stage %d should be in [0,%d)", stage, stageLog->numStages);

  /* Record flops/time of previous stage */
  PetscCall(PetscIntStackEmpty(stageLog->stack, &empty));
  if (!empty) {
    PetscCall(PetscIntStackTop(stageLog->stack, &curStage));
    if (stageLog->stageInfo[curStage].perfInfo.active) {
      PetscCall(PetscTimeAdd(&stageLog->stageInfo[curStage].perfInfo.time));
      stageLog->stageInfo[curStage].perfInfo.flops += petsc_TotalFlops;
      stageLog->stageInfo[curStage].perfInfo.numMessages += petsc_irecv_ct + petsc_isend_ct + petsc_recv_ct + petsc_send_ct;
      stageLog->stageInfo[curStage].perfInfo.messageLength += petsc_irecv_len + petsc_isend_len + petsc_recv_len + petsc_send_len;
      stageLog->stageInfo[curStage].perfInfo.numReductions += petsc_allreduce_ct + petsc_gather_ct + petsc_scatter_ct;
    }
  }
  /* Activate the stage */
  PetscCall(PetscIntStackPush(stageLog->stack, stage));

  stageLog->stageInfo[stage].used = PETSC_TRUE;
  stageLog->stageInfo[stage].perfInfo.count++;
  stageLog->curStage = stage;
  /* Subtract current quantities so that we obtain the difference when we pop */
  if (stageLog->stageInfo[stage].perfInfo.active) {
    PetscCall(PetscTimeSubtract(&stageLog->stageInfo[stage].perfInfo.time));
    stageLog->stageInfo[stage].perfInfo.flops -= petsc_TotalFlops;
    stageLog->stageInfo[stage].perfInfo.numMessages -= petsc_irecv_ct + petsc_isend_ct + petsc_recv_ct + petsc_send_ct;
    stageLog->stageInfo[stage].perfInfo.messageLength -= petsc_irecv_len + petsc_isend_len + petsc_recv_len + petsc_send_len;
    stageLog->stageInfo[stage].perfInfo.numReductions -= petsc_allreduce_ct + petsc_gather_ct + petsc_scatter_ct;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscStageLogPop - This function pops a stage from the stack.

  Not Collective

  Input Parameter:
. stageLog - The `PetscStageLog`

  Usage:
  If the option -log_view is used to run the program containing the
  following code, then 2 sets of summary data will be printed during
  PetscFinalize().
.vb
      PetscInitialize(int *argc,char ***args,0,0);
      [stage 0 of code]
      PetscStageLogPush(stageLog,1);
      [stage 1 of code]
      PetscStageLogPop(stageLog);
      PetscBarrier(...);
      [more stage 0 of code]
      PetscFinalize();
.ve

  Note:
  Use `PetscStageLogRegister()` to register a stage.

  Level: developer

.seealso: `PetscStageLogPush()`, `PetscStageLogGetCurrent()`, `PetscStageLogRegister()`, `PetscLogGetStageLog()`
@*/
PetscErrorCode PetscStageLogPop(PetscStageLog stageLog)
{
  int       curStage;
  PetscBool empty;

  PetscFunctionBegin;
  /* Record flops/time of current stage */
  PetscCall(PetscIntStackPop(stageLog->stack, &curStage));
  if (stageLog->stageInfo[curStage].perfInfo.active) {
    PetscCall(PetscTimeAdd(&stageLog->stageInfo[curStage].perfInfo.time));
    stageLog->stageInfo[curStage].perfInfo.flops += petsc_TotalFlops;
    stageLog->stageInfo[curStage].perfInfo.numMessages += petsc_irecv_ct + petsc_isend_ct + petsc_recv_ct + petsc_send_ct;
    stageLog->stageInfo[curStage].perfInfo.messageLength += petsc_irecv_len + petsc_isend_len + petsc_recv_len + petsc_send_len;
    stageLog->stageInfo[curStage].perfInfo.numReductions += petsc_allreduce_ct + petsc_gather_ct + petsc_scatter_ct;
  }
  PetscCall(PetscIntStackEmpty(stageLog->stack, &empty));
  if (!empty) {
    /* Subtract current quantities so that we obtain the difference when we pop */
    PetscCall(PetscIntStackTop(stageLog->stack, &curStage));
    if (stageLog->stageInfo[curStage].perfInfo.active) {
      PetscCall(PetscTimeSubtract(&stageLog->stageInfo[curStage].perfInfo.time));
      stageLog->stageInfo[curStage].perfInfo.flops -= petsc_TotalFlops;
      stageLog->stageInfo[curStage].perfInfo.numMessages -= petsc_irecv_ct + petsc_isend_ct + petsc_recv_ct + petsc_send_ct;
      stageLog->stageInfo[curStage].perfInfo.messageLength -= petsc_irecv_len + petsc_isend_len + petsc_recv_len + petsc_send_len;
      stageLog->stageInfo[curStage].perfInfo.numReductions -= petsc_allreduce_ct + petsc_gather_ct + petsc_scatter_ct;
    }
    stageLog->curStage = curStage;
  } else stageLog->curStage = -1;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscStageLogGetClassRegLog - This function returns the PetscClassRegLog for the given stage.

  Not Collective

  Input Parameter:
. stageLog - The `PetscStageLog`

  Output Parameter:
. classLog - The `PetscClassRegLog`

  Level: developer

.seealso: `PetscStageLogPush()`, `PetscStageLogPop()`, `PetscLogGetStageLog()`
@*/
PetscErrorCode PetscStageLogGetClassRegLog(PetscStageLog stageLog, PetscClassRegLog *classLog)
{
  PetscFunctionBegin;
  PetscValidPointer(classLog, 2);
  *classLog = stageLog->classLog;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscStageLogGetEventRegLog - This function returns the `PetscEventRegLog`.

  Not Collective

  Input Parameter:
. stageLog - The `PetscStageLog`

  Output Parameter:
. eventLog - The `PetscEventRegLog`

  Level: developer

.seealso: `PetscStageLogPush()`, `PetscStageLogPop()`, `PetscLogGetStageLog()`
@*/
PetscErrorCode PetscStageLogGetEventRegLog(PetscStageLog stageLog, PetscEventRegLog *eventLog)
{
  PetscFunctionBegin;
  PetscValidPointer(eventLog, 2);
  *eventLog = stageLog->eventLog;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscStageLogGetClassPerfLog - This function returns the `PetscClassPerfLog` for the given stage.

  Not Collective

  Input Parameters:
+ stageLog - The `PetscStageLog`
- stage    - The stage

  Output Parameter:
. classLog - The `PetscClassPerfLog`

  Level: developer

.seealso: `PetscStageLogPush()`, `PetscStageLogPop()`, `PetscLogGetStageLog()`
@*/
PetscErrorCode PetscStageLogGetClassPerfLog(PetscStageLog stageLog, int stage, PetscClassPerfLog *classLog)
{
  PetscFunctionBegin;
  PetscValidPointer(classLog, 3);
  PetscCheck(!(stage < 0) && !(stage >= stageLog->numStages), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid stage %d should be in [0,%d)", stage, stageLog->numStages);
  *classLog = stageLog->stageInfo[stage].classLog;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscStageLogSetActive - This function determines whether events will be logged during this state.

  Not Collective

  Input Parameters:
+ stageLog - The `PetscStageLog`
. stage    - The stage to log
- isActive - The activity flag, `PETSC_TRUE` for logging, otherwise `PETSC_FALSE` (default is `PETSC_TRUE`)

  Level: developer

.seealso: `PetscStageLogGetActive()`, `PetscStageLogGetCurrent()`, `PetscStageLogRegister()`, `PetscLogGetStageLog()`
@*/
PetscErrorCode PetscStageLogSetActive(PetscStageLog stageLog, int stage, PetscBool isActive)
{
  PetscFunctionBegin;
  PetscCheck(!(stage < 0) && !(stage >= stageLog->numStages), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid stage %d should be in [0,%d)", stage, stageLog->numStages);
  stageLog->stageInfo[stage].perfInfo.active = isActive;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscStageLogGetActive - This function returns whether events will be logged suring this stage.

  Not Collective

  Input Parameters:
+ stageLog - The `PetscStageLog`
- stage    - The stage to log

  Output Parameter:
. isActive - The activity flag, `PETSC_TRUE` for logging, otherwise `PETSC_FALSE` (default is `PETSC_TRUE`)

  Level: developer

.seealso: `PetscStageLogSetActive()`, `PetscStageLogGetCurrent()`, `PetscStageLogRegister()`, `PetscLogGetStageLog()`
@*/
PetscErrorCode PetscStageLogGetActive(PetscStageLog stageLog, int stage, PetscBool *isActive)
{
  PetscFunctionBegin;
  PetscCheck(!(stage < 0) && !(stage >= stageLog->numStages), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid stage %d should be in [0,%d)", stage, stageLog->numStages);
  PetscValidBoolPointer(isActive, 3);
  *isActive = stageLog->stageInfo[stage].perfInfo.active;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscStageLogSetVisible - This function determines whether a stage is printed during `PetscLogView()`

  Not Collective

  Input Parameters:
+ stageLog  - The `PetscStageLog`
. stage     - The stage to log
- isVisible - The visibility flag, `PETSC_TRUE` for printing, otherwise `PETSC_FALSE` (default is `PETSC_TRUE`)

  Options Database Key:
. -log_view - Activates log summary

  Level: developer

.seealso: `PetscStageLogGetVisible()`, `PetscStageLogGetCurrent()`, `PetscStageLogRegister()`, `PetscLogGetStageLog()`
@*/
PetscErrorCode PetscStageLogSetVisible(PetscStageLog stageLog, int stage, PetscBool isVisible)
{
  PetscFunctionBegin;
  PetscCheck(!(stage < 0) && !(stage >= stageLog->numStages), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid stage %d should be in [0,%d)", stage, stageLog->numStages);
  stageLog->stageInfo[stage].perfInfo.visible = isVisible;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscStageLogGetVisible - This function returns whether a stage is printed during `PetscLogView()`

  Not Collective

  Input Parameters:
+ stageLog  - The `PetscStageLog`
- stage     - The stage to log

  Output Parameter:
. isVisible - The visibility flag, `PETSC_TRUE` for printing, otherwise `PETSC_FALSE` (default is `PETSC_TRUE`)

  Options Database Key:
. -log_view - Activates log summary

  Level: developer

.seealso: `PetscStageLogSetVisible()`, `PetscStageLogGetCurrent()`, `PetscStageLogRegister()`, `PetscLogGetStageLog()`
@*/
PetscErrorCode PetscStageLogGetVisible(PetscStageLog stageLog, int stage, PetscBool *isVisible)
{
  PetscFunctionBegin;
  PetscCheck(!(stage < 0) && !(stage >= stageLog->numStages), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid stage %d should be in [0,%d)", stage, stageLog->numStages);
  PetscValidBoolPointer(isVisible, 3);
  *isVisible = stageLog->stageInfo[stage].perfInfo.visible;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscStageLogGetStage - This function returns the stage id given the stage name.

  Not Collective

  Input Parameters:
+ stageLog - The `PetscStageLog`
- name     - The stage name

  Output Parameter:
. stage    - The stage id, or -1 if it does not exist

  Level: developer

.seealso: `PetscStageLogGetCurrent()`, `PetscStageLogRegister()`, `PetscLogGetStageLog()`
@*/
PetscErrorCode PetscStageLogGetStage(PetscStageLog stageLog, const char name[], PetscLogStage *stage)
{
  PetscBool match;
  int       s;

  PetscFunctionBegin;
  PetscValidCharPointer(name, 2);
  PetscValidIntPointer(stage, 3);
  *stage = -1;
  for (s = 0; s < stageLog->numStages; s++) {
    PetscCall(PetscStrcasecmp(stageLog->stageInfo[s].name, name, &match));
    if (match) {
      *stage = s;
      break;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscStageLogCreate - This creates a `PetscStageLog` object.

  Not collective

  Output Parameter:
. stageLog - The `PetscStageLog`

  Level: developer

.seealso: `PetscStageLogCreate()`
@*/
PetscErrorCode PetscStageLogCreate(PetscStageLog *stageLog)
{
  PetscStageLog l;

  PetscFunctionBegin;
  PetscCall(PetscNew(&l));

  l->numStages = 0;
  l->maxStages = 10;
  l->curStage  = -1;

  PetscCall(PetscIntStackCreate(&l->stack));
  PetscCall(PetscMalloc1(l->maxStages, &l->stageInfo));
  PetscCall(PetscEventRegLogCreate(&l->eventLog));
  PetscCall(PetscClassRegLogCreate(&l->classLog));

  *stageLog = l;
  PetscFunctionReturn(PETSC_SUCCESS);
}
