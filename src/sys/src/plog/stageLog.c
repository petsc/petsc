/* $Id: stageLog.c,v 1.1 2000/01/10 03:28:12 knepley Exp $ */

#include "petsc.h"        /*I    "petsc.h"   I*/
#include "src/sys/src/plog/ptime.h"
#include "plog.h"

StageLog _stageLog;

#undef __FUNCT__  
#define __FUNCT__ "StageLogDestroy"
/*
  StageLogDestroy - This destroys a StageLog object.

  Not collective

  Input Paramter:
. stageLog - The StageLog

  Level: beginner

.keywords: log, stage, destroy
.seealso: StageLogCreate()
*/
int StageLogDestroy(StageLog stageLog)
{
  int stage;
  int ierr;

  PetscFunctionBegin;
  for(stage = 0; stage < stageLog->numStages; stage++) {
    ierr = PerfInfoDestroy(&stageLog->stageInfo[stage]);                                                  CHKERRQ(ierr);
    ierr = EventLogDestroy(stageLog->eventLog[stage]);                                                    CHKERRQ(ierr);
    ierr = ClassLogDestroy(stageLog->classLog[stage]);                                                    CHKERRQ(ierr);
  }
  ierr = StackDestroy(stageLog->stack);                                                                   CHKERRQ(ierr);
  ierr = PetscFree(stageLog->stageInfo);                                                                  CHKERRQ(ierr);
  ierr = PetscFree(stageLog->eventLog);                                                                   CHKERRQ(ierr);
  ierr = PetscFree(stageLog->classLog);                                                                   CHKERRQ(ierr);
  ierr = PetscFree(stageLog);                                                                             CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "StageLogRegister"
/*@C
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
int StageLogRegister(StageLog stageLog, const char sname[], int *stage)
{
  PerfInfo   *stageInfo;
  EventLog   *eventLog;
  ClassLog   *classLog;
  char       *str;
  int         s;
  int         ierr;

  PetscFunctionBegin;
  PetscValidCharPointer(sname);
  PetscValidIntPointer(stage);
  s = stageLog->numStages++;
  if (stageLog->numStages > stageLog->maxStages) {
    ierr = PetscMalloc(stageLog->maxStages*2 * sizeof(PerfInfo),   &stageInfo);                           CHKERRQ(ierr);
    ierr = PetscMalloc(stageLog->maxStages*2 * sizeof(EventLog),   &eventLog);                            CHKERRQ(ierr);
    ierr = PetscMalloc(stageLog->maxStages*2 * sizeof(ClassLog),   &classLog);                            CHKERRQ(ierr);
    ierr = PetscMemcpy(stageInfo,    stageLog->stageInfo,    stageLog->maxStages * sizeof(PerfInfo));     CHKERRQ(ierr);
    ierr = PetscMemcpy(eventLog,     stageLog->eventLog,     stageLog->maxStages * sizeof(EventLog));     CHKERRQ(ierr);
    ierr = PetscMemcpy(classLog,     stageLog->classLog,     stageLog->maxStages * sizeof(ClassLog));     CHKERRQ(ierr);
    ierr = PetscFree(stageLog->stageInfo);                                                                CHKERRQ(ierr);
    ierr = PetscFree(stageLog->eventLog);                                                                 CHKERRQ(ierr);
    ierr = PetscFree(stageLog->classLog);                                                                 CHKERRQ(ierr);
    stageLog->stageInfo    = stageInfo;
    stageLog->eventLog     = eventLog;
    stageLog->classLog     = classLog;
    stageLog->maxStages   *= 2;
  }
  /* Setup stage */
  ierr = PetscStrallocpy(sname, &str);                                                                    CHKERRQ(ierr);
  stageLog->stageInfo[s].name          = str;
  stageLog->stageInfo[s].cookie        = -1;
  stageLog->stageInfo[s].active        = PETSC_FALSE;
  stageLog->stageInfo[s].visible       = PETSC_TRUE;
  stageLog->stageInfo[s].count         = 0;
  stageLog->stageInfo[s].flops         = 0.0;
  stageLog->stageInfo[s].time          = 0.0;
  stageLog->stageInfo[s].numMessages   = 0.0;
  stageLog->stageInfo[s].messageLength = 0.0;
  stageLog->stageInfo[s].numReductions = 0.0;
  /* Create eventLog */
  if (s == 0) {
    ierr = EventLogCreate(&stageLog->eventLog[s]);                                                        CHKERRQ(ierr);
  } else {
    ierr = EventLogCopy(stageLog->eventLog[0], &stageLog->eventLog[s]);                                   CHKERRQ(ierr);
  }
  /* Create classLog */
  if (s == 0) {
    ierr = ClassLogCreate(&stageLog->classLog[s]);                                                        CHKERRQ(ierr);
  } else {
    ierr = ClassLogCopy(stageLog->classLog[0], &stageLog->classLog[s]);                                   CHKERRQ(ierr);
  }
  *stage = s;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "StageLogPush"
/*@C
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
int StageLogPush(StageLog stageLog, int stage)
{
  int        curStage = 0;
  PetscTruth empty;
  int        ierr;

  PetscFunctionBegin;
  if ((stage < 0) || (stage >= stageLog->numStages)) {
    SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE, "Invalid stage %d should be in [0,%d)", stage, stageLog->numStages);
  }

  /* Record flops/time of previous stage */
  ierr = StackEmpty(stageLog->stack, &empty);                                                             CHKERRQ(ierr);
  if (empty == PETSC_FALSE) {
    ierr = StackTop(stageLog->stack, &curStage);                                                          CHKERRQ(ierr);
    PetscTimeAdd(stageLog->stageInfo[curStage].time);
    stageLog->stageInfo[curStage].flops         += _TotalFlops;
    stageLog->stageInfo[curStage].numMessages   += irecv_ct  + isend_ct  + recv_ct  + send_ct;
    stageLog->stageInfo[curStage].messageLength += irecv_len + isend_len + recv_len + send_len;
    stageLog->stageInfo[curStage].numReductions += allreduce_ct;
  }
  /* Activate the stage */
  ierr = StackPush(stageLog->stack, stage);                                                               CHKERRQ(ierr);
  stageLog->stageInfo[stage].active = PETSC_TRUE;
  stageLog->stageInfo[stage].count++;
  stageLog->curStage = stage;
  /* Subtract current quantities so that we obtain the difference when we pop */
  PetscTimeSubtract(stageLog->stageInfo[stage].time);
  stageLog->stageInfo[stage].flops         -= _TotalFlops;
  stageLog->stageInfo[stage].numMessages   -= irecv_ct  + isend_ct  + recv_ct  + send_ct;
  stageLog->stageInfo[stage].messageLength -= irecv_len + isend_len + recv_len + send_len;
  stageLog->stageInfo[stage].numReductions -= allreduce_ct;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "StageLogPop"
/*@C
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
int StageLogPop(StageLog stageLog)
{
  int        curStage;
  PetscTruth empty;
  int        ierr;

  PetscFunctionBegin;
  /* Record flops/time of current stage */
  ierr = StackPop(stageLog->stack, &curStage);                                                            CHKERRQ(ierr);
  PetscTimeAdd(stageLog->stageInfo[curStage].time);
  stageLog->stageInfo[curStage].flops         += _TotalFlops;
  stageLog->stageInfo[curStage].numMessages   += irecv_ct  + isend_ct  + recv_ct  + send_ct;
  stageLog->stageInfo[curStage].messageLength += irecv_len + isend_len + recv_len + send_len;
  stageLog->stageInfo[curStage].numReductions += allreduce_ct;
  ierr = StackEmpty(stageLog->stack, &empty);                                                             CHKERRQ(ierr);
  if (empty == PETSC_FALSE) {
    /* Subtract current quantities so that we obtain the difference when we pop */
    ierr = StackTop(stageLog->stack, &curStage);                                                          CHKERRQ(ierr);
    PetscTimeSubtract(stageLog->stageInfo[curStage].time);
    stageLog->stageInfo[curStage].flops         -= _TotalFlops;
    stageLog->stageInfo[curStage].numMessages   -= irecv_ct  + isend_ct  + recv_ct  + send_ct;
    stageLog->stageInfo[curStage].messageLength -= irecv_len + isend_len + recv_len + send_len;
    stageLog->stageInfo[curStage].numReductions -= allreduce_ct;
    stageLog->curStage                           = curStage;
  } else {
    stageLog->curStage                           = -1;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "StageLogGetCurrent"
/*@C
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
int StageLogGetCurrent(StageLog stageLog, int *stage)
{
  PetscTruth empty;
  int        ierr;

  PetscFunctionBegin;
  ierr = StackEmpty(stageLog->stack, &empty);                                                             CHKERRQ(ierr);
  if (empty == PETSC_TRUE) {
    *stage = -1;
  } else {
    ierr = StackTop(stageLog->stack, stage);                                                              CHKERRQ(ierr);
  }
#ifdef PETSC_USE_BOPT_g
  if (*stage != stageLog->curStage) {
    SETERRQ2(PETSC_ERR_PLIB, "Inconsistency in stage log: stage %d should be %d", *stage, stageLog->curStage);
  }
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "StageLogGetClassLog"
/*@C
  StageLogGetClassLog - This function returns the ClassLog for the given stage.

  Not Collective

  Input Parameters:
+ stageLog - The StageLog
- stage    - The stage number

  Output Parameter:
. classLog - The ClassLog

  Level: intermediate

.keywords: log, stage
.seealso: StageLogPush(), StageLogPop(), PetscLogGetStageLog()
@*/
int StageLogGetClassLog(StageLog stageLog, int stage, ClassLog *classLog)
{
  PetscFunctionBegin;
  PetscValidPointer(classLog);
  if ((stage < 0) || (stage >= stageLog->numStages)) {
    SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE, "Invalid stage %d should be in [0,%d)", stage, stageLog->numStages);
  }
  *classLog = stageLog->classLog[stage];
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "StageLogGetEventLog"
/*@C
  StageLogGetEventLog - This function returns the EventLog for the given stage.

  Not Collective

  Input Parameters:
+ stageLog - The StageLog
- stage    - The stage number

  Output Parameter:
. eventLog - The EventLog

  Level: intermediate

.keywords: log, stage
.seealso: StageLogPush(), StageLogPop(), PetscLogGetStageLog()
@*/
int StageLogGetEventLog(StageLog stageLog, int stage, EventLog *eventLog)
{
  PetscFunctionBegin;
  PetscValidPointer(eventLog);
  if ((stage < 0) || (stage >= stageLog->numStages)) {
    SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE, "Invalid stage %d should be in [0,%d)", stage, stageLog->numStages);
  }
  *eventLog = stageLog->eventLog[stage];
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "StageLogSetVisible"
/*@C
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
int StageLogSetVisible(StageLog stageLog, int stage, PetscTruth isVisible)
{
  PetscFunctionBegin;
  if ((stage < 0) || (stage >= stageLog->numStages)) {
    SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE, "Invalid stage %d should be in [0,%d)", stage, stageLog->numStages);
  }
  stageLog->stageInfo[stage].visible = isVisible;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "StageLogGetVisible"
/*@C
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
int StageLogGetVisible(StageLog stageLog, int stage, PetscTruth *isVisible)
{
  PetscFunctionBegin;
  if ((stage < 0) || (stage >= stageLog->numStages)) {
    SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE, "Invalid stage %d should be in [0,%d)", stage, stageLog->numStages);
  }
  PetscValidIntPointer(isVisible);
  *isVisible = stageLog->stageInfo[stage].visible;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "StageLogGetStage"
/*@C
  StageLogGetStage - This function the stage id given the stage name.

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
int StageLogGetStage(StageLog stageLog, const char name[], int *stage)
{
  PetscTruth match;
  int        s;
  int        ierr;

  PetscFunctionBegin;
  PetscValidCharPointer(name);
  *stage = -1;
  for(s = 0; s < stageLog->numStages; s++) {
    ierr = PetscStrcasecmp(stageLog->stageInfo[s].name, name, &match);                                    CHKERRQ(ierr);
    if (match == PETSC_TRUE) break;
  }
  if (s == stageLog->numStages) SETERRQ1(PETSC_ERR_ARG_WRONG, "No stage named %s", name);
  *stage = s;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "StageLogCreate"
/*
  StageLogCreate - This creates a StageLog object.

  Not collective

  Input Parameter:
. stageLog - The StageLog

  Level: beginner

.keywords: log, stage, create
.seealso: StageLogCreate()
*/
int StageLogCreate(StageLog *stageLog)
{
  StageLog l;
  int      ierr;

  PetscFunctionBegin;
  ierr = PetscNew(struct _StageLog, &l);                                                                  CHKERRQ(ierr);
  l->numStages = 0;
  l->maxStages = 10;
  l->curStage  = -1;
  ierr = PetscMalloc(l->maxStages * sizeof(PerfInfo),   &l->stageInfo);                                   CHKERRQ(ierr);
  ierr = PetscMalloc(l->maxStages * sizeof(EventLog),   &l->eventLog);                                    CHKERRQ(ierr);
  ierr = PetscMalloc(l->maxStages * sizeof(ClassLog),   &l->classLog);                                    CHKERRQ(ierr);
  ierr = StackCreate(&l->stack);                                                                          CHKERRQ(ierr);
  *stageLog = l;
  PetscFunctionReturn(0);
}
