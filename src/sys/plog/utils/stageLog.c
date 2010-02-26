#define PETSC_DLL

#include "../src/sys/plog/plog.h" /*I    "petscsys.h"   I*/

StageLog PETSC_DLLEXPORT _stageLog = 0;

#undef __FUNCT__  
#define __FUNCT__ "StageInfoDestroy"
/*@C
  StageInfoDestroy - This destroys a StageInfo object.

  Not collective

  Input Paramter:
. stageInfo - The StageInfo

  Level: beginner

.keywords: log, stage, destroy
.seealso: StageLogCreate()
@*/
PetscErrorCode PETSC_DLLEXPORT StageInfoDestroy(StageInfo *stageInfo)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(stageInfo->name);CHKERRQ(ierr);
  ierr = EventPerfLogDestroy(stageInfo->eventLog);CHKERRQ(ierr);
  ierr = ClassPerfLogDestroy(stageInfo->classLog);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "StageLogDestroy"
/*@
  StageLogDestroy - This destroys a StageLog object.

  Not collective

  Input Paramter:
. stageLog - The StageLog

  Level: beginner

.keywords: log, stage, destroy
.seealso: StageLogCreate()
@*/
PetscErrorCode PETSC_DLLEXPORT StageLogDestroy(StageLog stageLog) 
{
  int            stage;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!stageLog) PetscFunctionReturn(0);
  ierr = StackDestroy(stageLog->stack);CHKERRQ(ierr);
  ierr = EventRegLogDestroy(stageLog->eventLog);CHKERRQ(ierr);
  ierr = ClassRegLogDestroy(stageLog->classLog);CHKERRQ(ierr);
  for(stage = 0; stage < stageLog->numStages; stage++) {
    ierr = StageInfoDestroy(&stageLog->stageInfo[stage]);CHKERRQ(ierr);
  }
  ierr = PetscFree(stageLog->stageInfo);CHKERRQ(ierr);
  ierr = PetscFree(stageLog);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "StageLogRegister"
/*@
  StageLogRegister - Registers a stage name for logging operations in an application code.

  Not Collective

  Input Parameter:
+ stageLog - The StageLog
- sname    - the name to associate with that stage

  Output Parameter:
. stage    - The stage index

  Level: intermediate

.keywords: log, stage, register
.seealso: StageLogPush(), StageLogPop(), StageLogCreate()
@*/
PetscErrorCode PETSC_DLLEXPORT StageLogRegister(StageLog stageLog, const char sname[], int *stage)
{
  StageInfo *stageInfo;
  char      *str;
  int        s, st;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidCharPointer(sname,2);
  PetscValidIntPointer(stage,3);
  for(st = 0; st < stageLog->numStages; ++st) {
    PetscTruth same;

    ierr = PetscStrcmp(stageLog->stageInfo[st].name, sname, &same);CHKERRQ(ierr);
    if (same) {SETERRQ1(PETSC_ERR_ARG_WRONG, "Duplicate stage name given: %s", sname);}
  }
  s = stageLog->numStages++;
  if (stageLog->numStages > stageLog->maxStages) {
    ierr = PetscMalloc(stageLog->maxStages*2 * sizeof(StageInfo), &stageInfo);CHKERRQ(ierr);
    ierr = PetscMemcpy(stageInfo, stageLog->stageInfo, stageLog->maxStages * sizeof(StageInfo));CHKERRQ(ierr);
    ierr = PetscFree(stageLog->stageInfo);CHKERRQ(ierr);
    stageLog->stageInfo  = stageInfo;
    stageLog->maxStages *= 2;
  }
  /* Setup stage */
  ierr = PetscStrallocpy(sname, &str);CHKERRQ(ierr);
  stageLog->stageInfo[s].name                   = str;
  stageLog->stageInfo[s].used                   = PETSC_FALSE;
  stageLog->stageInfo[s].perfInfo.active        = PETSC_TRUE;
  stageLog->stageInfo[s].perfInfo.visible       = PETSC_TRUE;
  stageLog->stageInfo[s].perfInfo.count         = 0;
  stageLog->stageInfo[s].perfInfo.flops         = 0.0;
  stageLog->stageInfo[s].perfInfo.time          = 0.0;
  stageLog->stageInfo[s].perfInfo.numMessages   = 0.0;
  stageLog->stageInfo[s].perfInfo.messageLength = 0.0;
  stageLog->stageInfo[s].perfInfo.numReductions = 0.0;
  ierr = EventPerfLogCreate(&stageLog->stageInfo[s].eventLog);CHKERRQ(ierr);
  ierr = ClassPerfLogCreate(&stageLog->stageInfo[s].classLog);CHKERRQ(ierr);
  *stage = s;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "StageLogPush"
/*@
  StageLogPush - This function pushes a stage on the stack.

  Not Collective

  Input Parameters:
+ stageLog   - The StageLog
- stage - The stage to log

  Database Options:
. -log_summary - Activates logging

  Usage:
  If the option -log_sumary is used to run the program containing the 
  following code, then 2 sets of summary data will be printed during
  PetscFinalize().
.vb
      PetscInitialize(int *argc,char ***args,0,0);
      [stage 0 of code]   
      StageLogPush(stageLog,1);
      [stage 1 of code]
      StageLogPop(stageLog);
      PetscBarrier(...);
      [more stage 0 of code]   
      PetscFinalize();
.ve

  Notes:
  Use PetscLogStageRegister() to register a stage. All previous stages are
  accumulating time and flops, but events will only be logged in this stage.

  Level: intermediate

.keywords: log, push, stage
.seealso: StageLogPop(), StageLogGetCurrent(), StageLogRegister(), PetscLogGetStageLog()
@*/
PetscErrorCode PETSC_DLLEXPORT StageLogPush(StageLog stageLog, int stage)
{
  int             curStage = 0;
  PetscTruth      empty;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if ((stage < 0) || (stage >= stageLog->numStages)) {
    SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE, "Invalid stage %d should be in [0,%d)", stage, stageLog->numStages);
  }

  /* Record flops/time of previous stage */
  ierr = StackEmpty(stageLog->stack, &empty);CHKERRQ(ierr);
  if (!empty) {
    ierr = StackTop(stageLog->stack, &curStage);CHKERRQ(ierr);
    if (stageLog->stageInfo[curStage].perfInfo.active) {
      PetscTimeAdd(stageLog->stageInfo[curStage].perfInfo.time);
      stageLog->stageInfo[curStage].perfInfo.flops         += _TotalFlops;
      stageLog->stageInfo[curStage].perfInfo.numMessages   += irecv_ct  + isend_ct  + recv_ct  + send_ct;
      stageLog->stageInfo[curStage].perfInfo.messageLength += irecv_len + isend_len + recv_len + send_len;
      stageLog->stageInfo[curStage].perfInfo.numReductions += allreduce_ct;
    }
  }
  /* Activate the stage */
  ierr = StackPush(stageLog->stack, stage);CHKERRQ(ierr);
  stageLog->stageInfo[stage].used = PETSC_TRUE;
  stageLog->stageInfo[stage].perfInfo.count++;
  stageLog->curStage = stage;
  /* Subtract current quantities so that we obtain the difference when we pop */
  if (stageLog->stageInfo[stage].perfInfo.active) {
    PetscTimeSubtract(stageLog->stageInfo[stage].perfInfo.time);
    stageLog->stageInfo[stage].perfInfo.flops         -= _TotalFlops;
    stageLog->stageInfo[stage].perfInfo.numMessages   -= irecv_ct  + isend_ct  + recv_ct  + send_ct;
    stageLog->stageInfo[stage].perfInfo.messageLength -= irecv_len + isend_len + recv_len + send_len;
    stageLog->stageInfo[stage].perfInfo.numReductions -= allreduce_ct;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "StageLogPop"
/*@
  StageLogPop - This function pops a stage from the stack.

  Not Collective

  Input Parameter:
. stageLog - The StageLog

  Usage:
  If the option -log_sumary is used to run the program containing the 
  following code, then 2 sets of summary data will be printed during
  PetscFinalize().
.vb
      PetscInitialize(int *argc,char ***args,0,0);
      [stage 0 of code]   
      StageLogPush(stageLog,1);
      [stage 1 of code]
      StageLogPop(stageLog);
      PetscBarrier(...);
      [more stage 0 of code]   
      PetscFinalize();
.ve

  Notes:
  Use StageLogRegister() to register a stage.

  Level: intermediate

.keywords: log, pop, stage
.seealso: StageLogPush(), StageLogGetCurrent(), StageLogRegister(), PetscLogGetStageLog()
@*/
PetscErrorCode PETSC_DLLEXPORT StageLogPop(StageLog stageLog)
{
  int             curStage;
  PetscTruth      empty;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Record flops/time of current stage */
  ierr = StackPop(stageLog->stack, &curStage);CHKERRQ(ierr);
  if (stageLog->stageInfo[curStage].perfInfo.active) {
    PetscTimeAdd(stageLog->stageInfo[curStage].perfInfo.time);
    stageLog->stageInfo[curStage].perfInfo.flops         += _TotalFlops;
    stageLog->stageInfo[curStage].perfInfo.numMessages   += irecv_ct  + isend_ct  + recv_ct  + send_ct;
    stageLog->stageInfo[curStage].perfInfo.messageLength += irecv_len + isend_len + recv_len + send_len;
    stageLog->stageInfo[curStage].perfInfo.numReductions += allreduce_ct;
  }
  ierr = StackEmpty(stageLog->stack, &empty);CHKERRQ(ierr);
  if (!empty) {
    /* Subtract current quantities so that we obtain the difference when we pop */
    ierr = StackTop(stageLog->stack, &curStage);CHKERRQ(ierr);
    if (stageLog->stageInfo[curStage].perfInfo.active) {
      PetscTimeSubtract(stageLog->stageInfo[curStage].perfInfo.time);
      stageLog->stageInfo[curStage].perfInfo.flops         -= _TotalFlops;
      stageLog->stageInfo[curStage].perfInfo.numMessages   -= irecv_ct  + isend_ct  + recv_ct  + send_ct;
      stageLog->stageInfo[curStage].perfInfo.messageLength -= irecv_len + isend_len + recv_len + send_len;
      stageLog->stageInfo[curStage].perfInfo.numReductions -= allreduce_ct;
    }
    stageLog->curStage                           = curStage;
  } else {
    stageLog->curStage                           = -1;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "StageLogGetCurrent"
/*@
  StageLogGetCurrent - This function returns the stage from the top of the stack.

  Not Collective

  Input Parameter:
. stageLog - The StageLog

  Output Parameter:
. stage    - The current stage

  Notes:
  If no stage is currently active, stage is set to -1.

  Level: intermediate

.keywords: log, stage
.seealso: StageLogPush(), StageLogPop(), PetscLogGetStageLog()
@*/
PetscErrorCode PETSC_DLLEXPORT StageLogGetCurrent(StageLog stageLog, int *stage) 
{
  PetscTruth     empty;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = StackEmpty(stageLog->stack, &empty);CHKERRQ(ierr);
  if (empty) {
    *stage = -1;
  } else {
    ierr = StackTop(stageLog->stack, stage);CHKERRQ(ierr);
  }
#ifdef PETSC_USE_DEBUG
  if (*stage != stageLog->curStage) {
    SETERRQ2(PETSC_ERR_PLIB, "Inconsistency in stage log: stage %d should be %d", *stage, stageLog->curStage);
  }
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "StageLogGetClassRegLog"
/*@C
  StageLogGetClassRegLog - This function returns the ClassRegLog for the given stage.

  Not Collective

  Input Parameters:
. stageLog - The StageLog

  Output Parameter:
. classLog - The ClassRegLog

  Level: intermediate

.keywords: log, stage
.seealso: StageLogPush(), StageLogPop(), PetscLogGetStageLog()
@*/
PetscErrorCode PETSC_DLLEXPORT StageLogGetClassRegLog(StageLog stageLog, ClassRegLog *classLog)
{
  PetscFunctionBegin;
  PetscValidPointer(classLog,2);
  *classLog = stageLog->classLog;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "StageLogGetEventRegLog"
/*@C
  StageLogGetEventRegLog - This function returns the EventRegLog.

  Not Collective

  Input Parameters:
. stageLog - The StageLog

  Output Parameter:
. eventLog - The EventRegLog

  Level: intermediate

.keywords: log, stage
.seealso: StageLogPush(), StageLogPop(), PetscLogGetStageLog()
@*/
PetscErrorCode PETSC_DLLEXPORT StageLogGetEventRegLog(StageLog stageLog, EventRegLog *eventLog) 
{
  PetscFunctionBegin;
  PetscValidPointer(eventLog,2);
  *eventLog = stageLog->eventLog;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "StageLogGetClassPerfLog"
/*@C
  StageLogGetClassPerfLog - This function returns the ClassPerfLog for the given stage.

  Not Collective

  Input Parameters:
+ stageLog - The StageLog
- stage    - The stage

  Output Parameter:
. classLog - The ClassPerfLog

  Level: intermediate

.keywords: log, stage
.seealso: StageLogPush(), StageLogPop(), PetscLogGetStageLog()
@*/
PetscErrorCode PETSC_DLLEXPORT StageLogGetClassPerfLog(StageLog stageLog, int stage, ClassPerfLog *classLog)
{
  PetscFunctionBegin;
  PetscValidPointer(classLog,2);
  if ((stage < 0) || (stage >= stageLog->numStages)) {
    SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE, "Invalid stage %d should be in [0,%d)", stage, stageLog->numStages);
  }
  *classLog = stageLog->stageInfo[stage].classLog;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "StageLogGetEventPerfLog"
/*@C
  StageLogGetEventPerfLog - This function returns the EventPerfLog for the given stage.

  Not Collective

  Input Parameters:
+ stageLog - The StageLog
- stage    - The stage

  Output Parameter:
. eventLog - The EventPerfLog

  Level: intermediate

.keywords: log, stage
.seealso: StageLogPush(), StageLogPop(), PetscLogGetStageLog()
@*/
PetscErrorCode PETSC_DLLEXPORT StageLogGetEventPerfLog(StageLog stageLog, int stage, EventPerfLog *eventLog)
{
  PetscFunctionBegin;
  PetscValidPointer(eventLog,3);
  if ((stage < 0) || (stage >= stageLog->numStages)) {
    SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE, "Invalid stage %d should be in [0,%d)", stage, stageLog->numStages);
  }
  *eventLog = stageLog->stageInfo[stage].eventLog;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "StageLogSetActive"
/*@
  StageLogSetActive - This function determines whether events will be logged during this state.

  Not Collective

  Input Parameters:
+ stageLog - The StageLog
. stage    - The stage to log
- isActive - The activity flag, PETSC_TRUE for logging, otherwise PETSC_FALSE (default is PETSC_TRUE)

  Level: intermediate

.keywords: log, active, stage
.seealso: StageLogGetActive(), StageLogGetCurrent(), StageLogRegister(), PetscLogGetStageLog()
@*/
PetscErrorCode PETSC_DLLEXPORT StageLogSetActive(StageLog stageLog, int stage, PetscTruth isActive) 
{
  PetscFunctionBegin;
  if ((stage < 0) || (stage >= stageLog->numStages)) {
    SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE, "Invalid stage %d should be in [0,%d)", stage, stageLog->numStages);
  }
  stageLog->stageInfo[stage].perfInfo.active = isActive;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "StageLogGetActive"
/*@
  StageLogGetActive - This function returns whether events will be logged suring this stage.

  Not Collective

  Input Parameters:
+ stageLog - The StageLog
- stage    - The stage to log

  Output Parameter:
. isActive - The activity flag, PETSC_TRUE for logging, otherwise PETSC_FALSE (default is PETSC_TRUE)

  Level: intermediate

.keywords: log, visible, stage
.seealso: StageLogSetActive(), StageLogGetCurrent(), StageLogRegister(), PetscLogGetStageLog()
@*/
PetscErrorCode PETSC_DLLEXPORT StageLogGetActive(StageLog stageLog, int stage, PetscTruth *isActive)
{
  PetscFunctionBegin;
  if ((stage < 0) || (stage >= stageLog->numStages)) {
    SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE, "Invalid stage %d should be in [0,%d)", stage, stageLog->numStages);
  }
  PetscValidIntPointer(isActive,3);
  *isActive = stageLog->stageInfo[stage].perfInfo.active;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "StageLogSetVisible"
/*@
  StageLogSetVisible - This function determines whether a stage is printed during PetscLogPrintSummary()

  Not Collective

  Input Parameters:
+ stageLog  - The StageLog
. stage     - The stage to log
- isVisible - The visibility flag, PETSC_TRUE for printing, otherwise PETSC_FALSE (default is PETSC_TRUE)

  Database Options:
. -log_summary - Activates log summary

  Level: intermediate

.keywords: log, visible, stage
.seealso: StageLogGetVisible(), StageLogGetCurrent(), StageLogRegister(), PetscLogGetStageLog()
@*/
PetscErrorCode PETSC_DLLEXPORT StageLogSetVisible(StageLog stageLog, int stage, PetscTruth isVisible) 
{
  PetscFunctionBegin;
  if ((stage < 0) || (stage >= stageLog->numStages)) {
    SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE, "Invalid stage %d should be in [0,%d)", stage, stageLog->numStages);
  }
  stageLog->stageInfo[stage].perfInfo.visible = isVisible;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "StageLogGetVisible"
/*@
  StageLogGetVisible - This function returns whether a stage is printed during PetscLogPrintSummary()

  Not Collective

  Input Parameters:
+ stageLog  - The StageLog
- stage     - The stage to log

  Output Parameter:
. isVisible - The visibility flag, PETSC_TRUE for printing, otherwise PETSC_FALSE (default is PETSC_TRUE)

  Database Options:
. -log_summary - Activates log summary

  Level: intermediate

.keywords: log, visible, stage
.seealso: StageLogSetVisible(), StageLogGetCurrent(), StageLogRegister(), PetscLogGetStageLog()
@*/
PetscErrorCode PETSC_DLLEXPORT StageLogGetVisible(StageLog stageLog, int stage, PetscTruth *isVisible)
{
  PetscFunctionBegin;
  if ((stage < 0) || (stage >= stageLog->numStages)) {
    SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE, "Invalid stage %d should be in [0,%d)", stage, stageLog->numStages);
  }
  PetscValidIntPointer(isVisible,3);
  *isVisible = stageLog->stageInfo[stage].perfInfo.visible;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "StageLogGetStage"
/*@
  StageLogGetStage - This function returns the stage id given the stage name.

  Not Collective

  Input Parameters:
+ stageLog - The StageLog
- name     - The stage name

  Output Parameter:
. stage    - The stage id

  Level: intermediate

.keywords: log, stage
.seealso: StageLogGetCurrent(), StageLogRegister(), PetscLogGetStageLog()
@*/
PetscErrorCode PETSC_DLLEXPORT StageLogGetStage(StageLog stageLog, const char name[], int *stage)
{
  PetscTruth match;
  int        s;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidCharPointer(name,2);
  PetscValidIntPointer(stage,3);
  *stage = -1;
  for(s = 0; s < stageLog->numStages; s++) {
    ierr = PetscStrcasecmp(stageLog->stageInfo[s].name, name, &match);CHKERRQ(ierr);
    if (match) break;
  }
  if (s == stageLog->numStages) SETERRQ1(PETSC_ERR_ARG_WRONG, "No stage named %s", name);
  *stage = s;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "StageLogCreate"
/*@
  StageLogCreate - This creates a StageLog object.

  Not collective

  Input Parameter:
. stageLog - The StageLog

  Level: beginner

.keywords: log, stage, create
.seealso: StageLogCreate()
@*/
PetscErrorCode PETSC_DLLEXPORT StageLogCreate(StageLog *stageLog) 
{
  StageLog       l;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNew(struct _n_StageLog, &l);CHKERRQ(ierr);
  l->numStages = 0;
  l->maxStages = 10;
  l->curStage  = -1;
  ierr = StackCreate(&l->stack);CHKERRQ(ierr);
  ierr = PetscMalloc(l->maxStages * sizeof(StageInfo), &l->stageInfo);CHKERRQ(ierr);
  ierr = EventRegLogCreate(&l->eventLog);CHKERRQ(ierr);
  ierr = ClassRegLogCreate(&l->classLog);CHKERRQ(ierr);
  *stageLog = l;
  PetscFunctionReturn(0);
}
