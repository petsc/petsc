
/*
     This defines part of the private API for logging performance information. It is intended to be used only by the
   PETSc PetscLog...() interface and not elsewhere, nor by users. Hence the prototypes for these functions are NOT
   in the public PETSc include files.

*/
#include <petsc/private/logimpl.h>  /*I    "petscsys.h"   I*/

/*----------------------------------------------- Creation Functions -------------------------------------------------*/
/* Note: these functions do not have prototypes in a public directory, so they are considered "internal" and not exported. */

/*@C
  PetscEventRegLogCreate - This creates a PetscEventRegLog object.

  Not collective

  Input Parameter:
. eventLog - The PetscEventRegLog

  Level: developer

.keywords: log, event, create
.seealso: PetscEventRegLogDestroy(), PetscStageLogCreate()
@*/
PetscErrorCode PetscEventRegLogCreate(PetscEventRegLog *eventLog)
{
  PetscEventRegLog l;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr         = PetscNew(&l);CHKERRQ(ierr);
  l->numEvents = 0;
  l->maxEvents = 100;
  ierr         = PetscMalloc1(l->maxEvents,&l->eventInfo);CHKERRQ(ierr);
  *eventLog    = l;
  PetscFunctionReturn(0);
}

/*@C
  PetscEventRegLogDestroy - This destroys a PetscEventRegLog object.

  Not collective

  Input Paramter:
. eventLog - The PetscEventRegLog

  Level: developer

.keywords: log, event, destroy
.seealso: PetscEventRegLogCreate()
@*/
PetscErrorCode PetscEventRegLogDestroy(PetscEventRegLog eventLog)
{
  int            e;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (e = 0; e < eventLog->numEvents; e++) {
    ierr = PetscFree(eventLog->eventInfo[e].name);CHKERRQ(ierr);
  }
  ierr = PetscFree(eventLog->eventInfo);CHKERRQ(ierr);
  ierr = PetscFree(eventLog);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscEventPerfLogCreate - This creates a PetscEventPerfLog object.

  Not collective

  Input Parameter:
. eventLog - The PetscEventPerfLog

  Level: developer

.keywords: log, event, create
.seealso: PetscEventPerfLogDestroy(), PetscStageLogCreate()
@*/
PetscErrorCode PetscEventPerfLogCreate(PetscEventPerfLog *eventLog)
{
  PetscEventPerfLog l;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr         = PetscNew(&l);CHKERRQ(ierr);
  l->numEvents = 0;
  l->maxEvents = 100;
  ierr         = PetscMalloc1(l->maxEvents,&l->eventInfo);CHKERRQ(ierr);
  *eventLog    = l;
  PetscFunctionReturn(0);
}

/*@C
  PetscEventPerfLogDestroy - This destroys a PetscEventPerfLog object.

  Not collective

  Input Paramter:
. eventLog - The PetscEventPerfLog

  Level: developer

.keywords: log, event, destroy
.seealso: PetscEventPerfLogCreate()
@*/
PetscErrorCode PetscEventPerfLogDestroy(PetscEventPerfLog eventLog)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(eventLog->eventInfo);CHKERRQ(ierr);
  ierr = PetscFree(eventLog);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------ General Functions -------------------------------------------------*/
/*@C
  PetscEventPerfInfoClear - This clears a PetscEventPerfInfo object.

  Not collective

  Input Paramter:
. eventInfo - The PetscEventPerfInfo

  Level: developer

.keywords: log, event, destroy
.seealso: PetscEventPerfLogCreate()
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
  eventInfo->numMessages   = 0.0;
  eventInfo->messageLength = 0.0;
  eventInfo->numReductions = 0.0;
  PetscFunctionReturn(0);
}

/*@C
  PetscEventPerfInfoCopy - Copy the activity and visibility data in eventInfo to outInfo

  Not collective

  Input Paramter:
. eventInfo - The input PetscEventPerfInfo

  Output Paramter:
. outInfo   - The output PetscEventPerfInfo

  Level: developer

.keywords: log, event, copy
.seealso: PetscEventPerfInfoClear()
@*/
PetscErrorCode PetscEventPerfInfoCopy(PetscEventPerfInfo *eventInfo,PetscEventPerfInfo *outInfo)
{
  PetscFunctionBegin;
  outInfo->id      = eventInfo->id;
  outInfo->active  = eventInfo->active;
  outInfo->visible = eventInfo->visible;
  PetscFunctionReturn(0);
}

/*@C
  PetscEventPerfLogEnsureSize - This ensures that a PetscEventPerfLog is at least of a certain size.

  Not collective

  Input Paramters:
+ eventLog - The PetscEventPerfLog
- size     - The size

  Level: developer

.keywords: log, event, size, ensure
.seealso: PetscEventPerfLogCreate()
@*/
PetscErrorCode PetscEventPerfLogEnsureSize(PetscEventPerfLog eventLog,int size)
{
  PetscEventPerfInfo *eventInfo;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  while (size > eventLog->maxEvents) {
    ierr = PetscMalloc1(eventLog->maxEvents*2,&eventInfo);CHKERRQ(ierr);
    ierr = PetscMemcpy(eventInfo,eventLog->eventInfo,eventLog->maxEvents * sizeof(PetscEventPerfInfo));CHKERRQ(ierr);
    ierr = PetscFree(eventLog->eventInfo);CHKERRQ(ierr);

    eventLog->eventInfo  = eventInfo;
    eventLog->maxEvents *= 2;
  }
  while (eventLog->numEvents < size) {
    ierr = PetscEventPerfInfoClear(&eventLog->eventInfo[eventLog->numEvents++]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_MPE)
#include <mpe.h>
PETSC_INTERN PetscErrorCode PetscLogMPEGetRGBColor(const char*[]);
PetscErrorCode PetscLogEventBeginMPE(PetscLogEvent event,int t,PetscObject o1,PetscObject o2,PetscObject o3,PetscObject o4)
{
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MPE_Log_event(petsc_stageLog->eventLog->eventInfo[event].mpe_id_begin,0,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscLogEventEndMPE(PetscLogEvent event,int t,PetscObject o1,PetscObject o2,PetscObject o3,PetscObject o4)
{
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MPE_Log_event(petsc_stageLog->eventLog->eventInfo[event].mpe_id_end,0,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

/*--------------------------------------------- Registration Functions ----------------------------------------------*/
/*@C
  PetscEventRegLogRegister - Registers an event for logging operations in an application code.

  Not Collective

  Input Parameters:
+ eventLog - The PetscEventLog
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

  Notes:

  PETSc can gather data for use with the utilities Jumpshot
  (part of the MPICH distribution).  If PETSc has been compiled
  with flag -DPETSC_HAVE_MPE (MPE is an additional utility within
  MPICH), the user can employ another command line option, -log_mpe,
  to create a logfile, "mpe.log", which can be visualized
  Jumpshot.

  Level: developer

.keywords: log, event, register
.seealso: PetscLogEventBegin(), PetscLogEventEnd(), PetscLogFlops(), 
          PetscEventLogActivate(), PetscEventLogDeactivate()
@*/
PetscErrorCode PetscEventRegLogRegister(PetscEventRegLog eventLog,const char ename[],PetscClassId classid,PetscLogEvent *event)
{
  PetscEventRegInfo *eventInfo;
  char              *str;
  int               e;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidCharPointer(ename,2);
  PetscValidIntPointer(event,4);
  /* Should check classid I think */
  e = eventLog->numEvents++;
  if (eventLog->numEvents > eventLog->maxEvents) {
    ierr = PetscMalloc1(eventLog->maxEvents*2,&eventInfo);CHKERRQ(ierr);
    ierr = PetscMemcpy(eventInfo,eventLog->eventInfo,eventLog->maxEvents * sizeof(PetscEventRegInfo));CHKERRQ(ierr);
    ierr = PetscFree(eventLog->eventInfo);CHKERRQ(ierr);

    eventLog->eventInfo  = eventInfo;
    eventLog->maxEvents *= 2;
  }
  ierr = PetscStrallocpy(ename,&str);CHKERRQ(ierr);

  eventLog->eventInfo[e].name    = str;
  eventLog->eventInfo[e].classid = classid;
#if defined(PETSC_HAVE_MPE)
  if (PetscLogPLB == PetscLogEventBeginMPE) {
    const char  *color;
    PetscMPIInt rank;
    int         beginID,endID;

    beginID = MPE_Log_get_event_number();
    endID   = MPE_Log_get_event_number();

    eventLog->eventInfo[e].mpe_id_begin = beginID;
    eventLog->eventInfo[e].mpe_id_end   = endID;

    ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
    if (!rank) {
      ierr = PetscLogMPEGetRGBColor(&color);CHKERRQ(ierr);
      MPE_Describe_state(beginID,endID,str,(char*)color);
    }
  }
#endif
  *event = e;
  PetscFunctionReturn(0);
}

/*---------------------------------------------- Activation Functions -----------------------------------------------*/
/*@C
  PetscEventPerfLogActivate - Indicates that a particular event should be logged.

  Not Collective

  Input Parameters:
+ eventLog - The PetscEventPerfLog
- event    - The event

   Usage:
.vb
      PetscEventPerfLogDeactivate(log, VEC_SetValues);
        [code where you do not want to log VecSetValues()]
      PetscEventPerfLogActivate(log, VEC_SetValues);
        [code where you do want to log VecSetValues()]
.ve

  Note:
  The event may be either a pre-defined PETSc event (found in
  include/petsclog.h) or an event number obtained with PetscEventRegLogRegister().

  Level: developer

.keywords: log, event, activate
.seealso: PetscEventPerfLogDeactivate()
@*/
PetscErrorCode PetscEventPerfLogActivate(PetscEventPerfLog eventLog,PetscLogEvent event)
{
  PetscFunctionBegin;
  eventLog->eventInfo[event].active = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@C
  PetscEventPerfLogDeactivate - Indicates that a particular event should not be logged.

  Not Collective

  Input Parameters:
+ eventLog - The PetscEventPerfLog
- event    - The event

   Usage:
.vb
      PetscEventPerfLogDeactivate(log, VEC_SetValues);
        [code where you do not want to log VecSetValues()]
      PetscEventPerfLogActivate(log, VEC_SetValues);
        [code where you do want to log VecSetValues()]
.ve

  Note:
  The event may be either a pre-defined PETSc event (found in
  include/petsclog.h) or an event number obtained with PetscEventRegLogRegister().

  Level: developer

.keywords: log, event, activate
.seealso: PetscEventPerfLogActivate()
@*/
PetscErrorCode PetscEventPerfLogDeactivate(PetscEventPerfLog eventLog,PetscLogEvent event)
{
  PetscFunctionBegin;
  eventLog->eventInfo[event].active = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
  PetscEventPerfLogActivateClass - Activates event logging for a PETSc object class.

  Not Collective

  Input Parameters:
+ eventLog    - The PetscEventPerfLog
. eventRegLog - The PetscEventRegLog
- classid      - The class id, for example MAT_CLASSID, SNES_CLASSID,

  Level: developer

.seealso: PetscEventPerfLogDeactivateClass(), PetscEventPerfLogActivate(), PetscEventPerfLogDeactivate()
@*/
PetscErrorCode PetscEventPerfLogActivateClass(PetscEventPerfLog eventLog,PetscEventRegLog eventRegLog,PetscClassId classid)
{
  int e;

  PetscFunctionBegin;
  for (e = 0; e < eventLog->numEvents; e++) {
    int c = eventRegLog->eventInfo[e].classid;
    if (c == classid) eventLog->eventInfo[e].active = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

/*@C
  PetscEventPerfLogDeactivateClass - Deactivates event logging for a PETSc object class.

  Not Collective

  Input Parameters:
+ eventLog    - The PetscEventPerfLog
. eventRegLog - The PetscEventRegLog
- classid - The class id, for example MAT_CLASSID, SNES_CLASSID,

  Level: developer

.seealso: PetscEventPerfLogDeactivateClass(), PetscEventPerfLogDeactivate(), PetscEventPerfLogActivate()
@*/
PetscErrorCode PetscEventPerfLogDeactivateClass(PetscEventPerfLog eventLog,PetscEventRegLog eventRegLog,PetscClassId classid)
{
  int e;

  PetscFunctionBegin;
  for (e = 0; e < eventLog->numEvents; e++) {
    int c = eventRegLog->eventInfo[e].classid;
    if (c == classid) eventLog->eventInfo[e].active = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------ Query Functions --------------------------------------------------*/
/*@C
  PetscEventRegLogGetEvent - This function returns the event id given the event name.

  Not Collective

  Input Parameters:
+ eventLog - The PetscEventRegLog
- name     - The stage name

  Output Parameter:
. event    - The event id, or -1 if not found

  Level: developer

.keywords: log, stage
.seealso: PetscEventRegLogRegister()
@*/
PetscErrorCode  PetscEventRegLogGetEvent(PetscEventRegLog eventLog,const char name[],PetscLogEvent *event)
{
  PetscBool      match;
  int            e;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidCharPointer(name,2);
  PetscValidIntPointer(event,3);
  *event = -1;
  for (e = 0; e < eventLog->numEvents; e++) {
    ierr = PetscStrcasecmp(eventLog->eventInfo[e].name,name,&match);CHKERRQ(ierr);
    if (match) {
      *event = e;
      break;
    }
  }
  PetscFunctionReturn(0);
}

/*@C
  PetscEventPerfLogSetVisible - This function determines whether an event is printed during PetscLogView()

  Not Collective

  Input Parameters:
+ eventLog  - The PetscEventPerfLog
. event     - The event to log
- isVisible - The visibility flag, PETSC_TRUE for printing, otherwise PETSC_FALSE (default is PETSC_TRUE)

  Database Options:
. -log_view - Activates log summary

  Level: developer

.keywords: log, visible, event
.seealso: PetscEventPerfLogGetVisible(), PetscEventRegLogRegister(), PetscStageLogGetEventLog()
@*/
PetscErrorCode PetscEventPerfLogSetVisible(PetscEventPerfLog eventLog,PetscLogEvent event,PetscBool isVisible)
{
  PetscFunctionBegin;
  eventLog->eventInfo[event].visible = isVisible;
  PetscFunctionReturn(0);
}

/*@C
  PetscEventPerfLogGetVisible - This function returns whether an event is printed during PetscLogView()

  Not Collective

  Input Parameters:
+ eventLog  - The PetscEventPerfLog
- event     - The event id to log

  Output Parameter:
. isVisible - The visibility flag, PETSC_TRUE for printing, otherwise PETSC_FALSE (default is PETSC_TRUE)

  Database Options:
. -log_view - Activates log summary

  Level: developer

.keywords: log, visible, event
.seealso: PetscEventPerfLogSetVisible(), PetscEventRegLogRegister(), PetscStageLogGetEventLog()
@*/
PetscErrorCode PetscEventPerfLogGetVisible(PetscEventPerfLog eventLog,PetscLogEvent event,PetscBool  *isVisible)
{
  PetscFunctionBegin;
  PetscValidIntPointer(isVisible,3);
  *isVisible = eventLog->eventInfo[event].visible;
  PetscFunctionReturn(0);
}

/*@C
  PetscLogEventGetPerfInfo - Return the performance information about the given event in the given stage

  Input Parameters:
+ stage - The stage number or PETSC_DETERMINE for the current stage
- event - The event number

  Output Parameters:
. info - This structure is filled with the performance information

  Level: Intermediate

.seealso: PetscLogEventGetFlops()
@*/
PetscErrorCode PetscLogEventGetPerfInfo(int stage,PetscLogEvent event,PetscEventPerfInfo *info)
{
  PetscStageLog     stageLog;
  PetscEventPerfLog eventLog = NULL;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidPointer(info,3);
  if (!PetscLogPLB) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Must use -log_view or PetscLogDefaultBegin() before calling this routine");
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  if (stage < 0) {ierr = PetscStageLogGetCurrent(stageLog,&stage);CHKERRQ(ierr);}
  ierr = PetscStageLogGetEventPerfLog(stageLog,stage,&eventLog);CHKERRQ(ierr);
  *info = eventLog->eventInfo[event];
  PetscFunctionReturn(0);
}

PetscErrorCode PetscLogEventGetFlops(PetscLogEvent event,PetscLogDouble *flops)
{
  PetscStageLog     stageLog;
  PetscEventPerfLog eventLog = NULL;
  int               stage;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (!PetscLogPLB) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Must use -log_view or PetscLogDefaultBegin() before calling this routine");
  ierr   = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  ierr   = PetscStageLogGetCurrent(stageLog,&stage);CHKERRQ(ierr);
  ierr   = PetscStageLogGetEventPerfLog(stageLog,stage,&eventLog);CHKERRQ(ierr);
  *flops = eventLog->eventInfo[event].flops;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscLogEventZeroFlops(PetscLogEvent event)
{
  PetscStageLog     stageLog;
  PetscEventPerfLog eventLog = NULL;
  int               stage;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  ierr = PetscStageLogGetCurrent(stageLog,&stage);CHKERRQ(ierr);
  ierr = PetscStageLogGetEventPerfLog(stageLog,stage,&eventLog);CHKERRQ(ierr);

  eventLog->eventInfo[event].flops    = 0.0;
  eventLog->eventInfo[event].flops2   = 0.0;
  eventLog->eventInfo[event].flopsTmp = 0.0;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscLogEventBeginDefault(PetscLogEvent event,int t,PetscObject o1,PetscObject o2,PetscObject o3,PetscObject o4)
{
  PetscStageLog     stageLog;
  PetscEventPerfLog eventLog = NULL;
  int               stage;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  ierr = PetscStageLogGetCurrent(stageLog,&stage);CHKERRQ(ierr);
  ierr = PetscStageLogGetEventPerfLog(stageLog,stage,&eventLog);CHKERRQ(ierr);
  /* Check for double counting */
  eventLog->eventInfo[event].depth++;
  if (eventLog->eventInfo[event].depth > 1) PetscFunctionReturn(0);
  /* Log performance info */
  eventLog->eventInfo[event].count++;
  eventLog->eventInfo[event].timeTmp = 0.0;
  PetscTimeSubtract(&eventLog->eventInfo[event].timeTmp);
  eventLog->eventInfo[event].flopsTmp       = 0.0;
  eventLog->eventInfo[event].flopsTmp      -= petsc_TotalFlops;
  eventLog->eventInfo[event].numMessages   -= petsc_irecv_ct  + petsc_isend_ct  + petsc_recv_ct  + petsc_send_ct;
  eventLog->eventInfo[event].messageLength -= petsc_irecv_len + petsc_isend_len + petsc_recv_len + petsc_send_len;
  eventLog->eventInfo[event].numReductions -= petsc_allreduce_ct + petsc_gather_ct + petsc_scatter_ct;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscLogEventEndDefault(PetscLogEvent event,int t,PetscObject o1,PetscObject o2,PetscObject o3,PetscObject o4)
{
  PetscStageLog     stageLog;
  PetscEventPerfLog eventLog = NULL;
  int               stage;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  ierr = PetscStageLogGetCurrent(stageLog,&stage);CHKERRQ(ierr);
  ierr = PetscStageLogGetEventPerfLog(stageLog,stage,&eventLog);CHKERRQ(ierr);
  /* Check for double counting */
  eventLog->eventInfo[event].depth--;
  if (eventLog->eventInfo[event].depth > 0) PetscFunctionReturn(0);
  else if (eventLog->eventInfo[event].depth < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Logging event had unbalanced begin/end pairs");
  /* Log performance info */
  PetscTimeAdd(&eventLog->eventInfo[event].timeTmp);
  eventLog->eventInfo[event].time          += eventLog->eventInfo[event].timeTmp;
  eventLog->eventInfo[event].time2         += eventLog->eventInfo[event].timeTmp*eventLog->eventInfo[event].timeTmp;
  eventLog->eventInfo[event].flopsTmp      += petsc_TotalFlops;
  eventLog->eventInfo[event].flops         += eventLog->eventInfo[event].flopsTmp;
  eventLog->eventInfo[event].flops2        += eventLog->eventInfo[event].flopsTmp*eventLog->eventInfo[event].flopsTmp;
  eventLog->eventInfo[event].numMessages   += petsc_irecv_ct  + petsc_isend_ct  + petsc_recv_ct  + petsc_send_ct;
  eventLog->eventInfo[event].messageLength += petsc_irecv_len + petsc_isend_len + petsc_recv_len + petsc_send_len;
  eventLog->eventInfo[event].numReductions += petsc_allreduce_ct + petsc_gather_ct + petsc_scatter_ct;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscLogEventBeginComplete(PetscLogEvent event,int t,PetscObject o1,PetscObject o2,PetscObject o3,PetscObject o4)
{
  PetscStageLog     stageLog;
  PetscEventRegLog  eventRegLog;
  PetscEventPerfLog eventPerfLog = NULL;
  Action            *tmpAction;
  PetscLogDouble    start,end;
  PetscLogDouble    curTime;
  int               stage;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  /* Dynamically enlarge logging structures */
  if (petsc_numActions >= petsc_maxActions) {
    PetscTime(&start);
    ierr = PetscMalloc1(petsc_maxActions*2,&tmpAction);CHKERRQ(ierr);
    ierr = PetscMemcpy(tmpAction,petsc_actions,petsc_maxActions * sizeof(Action));CHKERRQ(ierr);
    ierr = PetscFree(petsc_actions);CHKERRQ(ierr);

    petsc_actions     = tmpAction;
    petsc_maxActions *= 2;
    PetscTime(&end);
    petsc_BaseTime += (end - start);
  }
  /* Record the event */
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  ierr = PetscStageLogGetCurrent(stageLog,&stage);CHKERRQ(ierr);
  ierr = PetscStageLogGetEventRegLog(stageLog,&eventRegLog);CHKERRQ(ierr);
  ierr = PetscStageLogGetEventPerfLog(stageLog,stage,&eventPerfLog);CHKERRQ(ierr);
  PetscTime(&curTime);
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

    ierr = PetscMallocGetCurrentUsage(&petsc_actions[petsc_numActions].mem);CHKERRQ(ierr);
    ierr = PetscMallocGetMaximumUsage(&petsc_actions[petsc_numActions].maxmem);CHKERRQ(ierr);
    petsc_numActions++;
  }
  /* Check for double counting */
  eventPerfLog->eventInfo[event].depth++;
  if (eventPerfLog->eventInfo[event].depth > 1) PetscFunctionReturn(0);
  /* Log the performance info */
  eventPerfLog->eventInfo[event].count++;
  eventPerfLog->eventInfo[event].time          -= curTime;
  eventPerfLog->eventInfo[event].flops         -= petsc_TotalFlops;
  eventPerfLog->eventInfo[event].numMessages   -= petsc_irecv_ct  + petsc_isend_ct  + petsc_recv_ct  + petsc_send_ct;
  eventPerfLog->eventInfo[event].messageLength -= petsc_irecv_len + petsc_isend_len + petsc_recv_len + petsc_send_len;
  eventPerfLog->eventInfo[event].numReductions -= petsc_allreduce_ct + petsc_gather_ct + petsc_scatter_ct;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscLogEventEndComplete(PetscLogEvent event,int t,PetscObject o1,PetscObject o2,PetscObject o3,PetscObject o4)
{
  PetscStageLog     stageLog;
  PetscEventRegLog  eventRegLog;
  PetscEventPerfLog eventPerfLog = NULL;
  Action            *tmpAction;
  PetscLogDouble    start,end;
  PetscLogDouble    curTime;
  int               stage;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  /* Dynamically enlarge logging structures */
  if (petsc_numActions >= petsc_maxActions) {
    PetscTime(&start);
    ierr = PetscMalloc1(petsc_maxActions*2,&tmpAction);CHKERRQ(ierr);
    ierr = PetscMemcpy(tmpAction,petsc_actions,petsc_maxActions * sizeof(Action));CHKERRQ(ierr);
    ierr = PetscFree(petsc_actions);CHKERRQ(ierr);

    petsc_actions     = tmpAction;
    petsc_maxActions *= 2;
    PetscTime(&end);
    petsc_BaseTime += (end - start);
  }
  /* Record the event */
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  ierr = PetscStageLogGetCurrent(stageLog,&stage);CHKERRQ(ierr);
  ierr = PetscStageLogGetEventRegLog(stageLog,&eventRegLog);CHKERRQ(ierr);
  ierr = PetscStageLogGetEventPerfLog(stageLog,stage,&eventPerfLog);CHKERRQ(ierr);
  PetscTime(&curTime);
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

    ierr = PetscMallocGetCurrentUsage(&petsc_actions[petsc_numActions].mem);CHKERRQ(ierr);
    ierr = PetscMallocGetMaximumUsage(&petsc_actions[petsc_numActions].maxmem);CHKERRQ(ierr);
    petsc_numActions++;
  }
  /* Check for double counting */
  eventPerfLog->eventInfo[event].depth--;
  if (eventPerfLog->eventInfo[event].depth > 0) PetscFunctionReturn(0);
  else if (eventPerfLog->eventInfo[event].depth < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Logging event had unbalanced begin/end pairs");
  /* Log the performance info */
  eventPerfLog->eventInfo[event].count++;
  eventPerfLog->eventInfo[event].time          += curTime;
  eventPerfLog->eventInfo[event].flops         += petsc_TotalFlops;
  eventPerfLog->eventInfo[event].numMessages   += petsc_irecv_ct  + petsc_isend_ct  + petsc_recv_ct  + petsc_send_ct;
  eventPerfLog->eventInfo[event].messageLength += petsc_irecv_len + petsc_isend_len + petsc_recv_len + petsc_send_len;
  eventPerfLog->eventInfo[event].numReductions += petsc_allreduce_ct + petsc_gather_ct + petsc_scatter_ct;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscLogEventBeginTrace(PetscLogEvent event,int t,PetscObject o1,PetscObject o2,PetscObject o3,PetscObject o4)
{
  PetscStageLog     stageLog;
  PetscEventRegLog  eventRegLog;
  PetscEventPerfLog eventPerfLog = NULL;
  PetscLogDouble    cur_time;
  PetscMPIInt       rank;
  int               stage,err;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (!petsc_tracetime) PetscTime(&petsc_tracetime);

  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  ierr = PetscStageLogGetCurrent(stageLog,&stage);CHKERRQ(ierr);
  ierr = PetscStageLogGetEventRegLog(stageLog,&eventRegLog);CHKERRQ(ierr);
  ierr = PetscStageLogGetEventPerfLog(stageLog,stage,&eventPerfLog);CHKERRQ(ierr);
  /* Check for double counting */
  eventPerfLog->eventInfo[event].depth++;
  petsc_tracelevel++;
  if (eventPerfLog->eventInfo[event].depth > 1) PetscFunctionReturn(0);
  /* Log performance info */
  PetscTime(&cur_time);
  ierr = PetscFPrintf(PETSC_COMM_SELF,petsc_tracefile,"%s[%d] %g Event begin: %s\n",petsc_tracespace,rank,cur_time-petsc_tracetime,eventRegLog->eventInfo[event].name);CHKERRQ(ierr);
  ierr = PetscStrncpy(petsc_tracespace,petsc_traceblanks,2*petsc_tracelevel);CHKERRQ(ierr);

  petsc_tracespace[2*petsc_tracelevel] = 0;

  err = fflush(petsc_tracefile);
  if (err) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SYS,"fflush() failed on file");
  PetscFunctionReturn(0);
}

PetscErrorCode PetscLogEventEndTrace(PetscLogEvent event,int t,PetscObject o1,PetscObject o2,PetscObject o3,PetscObject o4)
{
  PetscStageLog     stageLog;
  PetscEventRegLog  eventRegLog;
  PetscEventPerfLog eventPerfLog = NULL;
  PetscLogDouble    cur_time;
  int               stage,err;
  PetscMPIInt       rank;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  petsc_tracelevel--;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  ierr = PetscStageLogGetCurrent(stageLog,&stage);CHKERRQ(ierr);
  ierr = PetscStageLogGetEventRegLog(stageLog,&eventRegLog);CHKERRQ(ierr);
  ierr = PetscStageLogGetEventPerfLog(stageLog,stage,&eventPerfLog);CHKERRQ(ierr);
  /* Check for double counting */
  eventPerfLog->eventInfo[event].depth--;
  if (eventPerfLog->eventInfo[event].depth > 0) PetscFunctionReturn(0);
  else if (eventPerfLog->eventInfo[event].depth < 0 || petsc_tracelevel < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Logging event had unbalanced begin/end pairs");

  /* Log performance info */
  ierr = PetscStrncpy(petsc_tracespace,petsc_traceblanks,2*petsc_tracelevel);CHKERRQ(ierr);

  petsc_tracespace[2*petsc_tracelevel] = 0;
  PetscTime(&cur_time);
  ierr = PetscFPrintf(PETSC_COMM_SELF,petsc_tracefile,"%s[%d] %g Event end: %s\n",petsc_tracespace,rank,cur_time-petsc_tracetime,eventRegLog->eventInfo[event].name);CHKERRQ(ierr);
  err  = fflush(petsc_tracefile);
  if (err) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SYS,"fflush() failed on file");
  PetscFunctionReturn(0);
}
