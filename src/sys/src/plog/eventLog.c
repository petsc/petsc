/* $Id: eventLog.c,v 1.3 2000/08/16 05:14:09 knepley Exp $ */

#include "petsc.h"        /*I    "petsc.h"   I*/
#include "src/sys/src/plog/ptime.h"
#include "plog.h"

/* Variables for the tracing logger */
extern FILE          *tracefile;
extern int            tracelevel;
extern char          *traceblanks;
extern char           tracespace[128];
extern PetscLogDouble tracetime;

/*----------------------------------------------- Creation Functions -------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "EventRegLogCreate"
/*
  EventRegLogCreate - This creates a EventRegLog object.

  Not collective

  Input Parameter:
. eventLog - The EventRegLog

  Level: beginner

.keywords: log, event, create
.seealso: EventRegLogDestroy(), StageLogCreate()
*/
int EventRegLogCreate(EventRegLog *eventLog) {
  EventRegLog l;
  int         ierr;

  PetscFunctionBegin;
  ierr = PetscNew(struct _EventRegLog, &l);                                                               CHKERRQ(ierr);
  l->numEvents   = 0;
  l->maxEvents   = 100;
  ierr = PetscMalloc(l->maxEvents * sizeof(EventRegInfo), &l->eventInfo);                                 CHKERRQ(ierr);
  *eventLog = l;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EventRegLogDestroy"
/*
  EventRegLogDestroy - This destroys a EventRegLog object.

  Not collective

  Input Paramter:
. eventLog - The EventRegLog

  Level: beginner

.keywords: log, event, destroy
.seealso: EventRegLogCreate()
*/
int EventRegLogDestroy(EventRegLog eventLog) {
  int ierr;

  PetscFunctionBegin;
  ierr = PetscFree(eventLog->eventInfo);                                                                  CHKERRQ(ierr);
  ierr = PetscFree(eventLog);                                                                             CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EventPerfLogCreate"
/*
  EventPerfLogCreate - This creates a EventPerfLog object.

  Not collective

  Input Parameter:
. eventLog - The EventPerfLog

  Level: beginner

.keywords: log, event, create
.seealso: EventPerfLogDestroy(), StageLogCreate()
*/
int EventPerfLogCreate(EventPerfLog *eventLog) {
  EventPerfLog l;
  int          ierr;

  PetscFunctionBegin;
  ierr = PetscNew(struct _EventPerfLog, &l);                                                              CHKERRQ(ierr);
  l->numEvents   = 0;
  l->maxEvents   = 100;
  ierr = PetscMalloc(l->maxEvents * sizeof(PerfInfo), &l->eventInfo);                                     CHKERRQ(ierr);
  *eventLog = l;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EventPerfLogDestroy"
/*
  EventPerfLogDestroy - This destroys a EventPerfLog object.

  Not collective

  Input Paramter:
. eventLog - The EventPerfLog

  Level: beginner

.keywords: log, event, destroy
.seealso: EventPerfLogCreate()
*/
int EventPerfLogDestroy(EventPerfLog eventLog) {
  int ierr;

  PetscFunctionBegin;
  ierr = PetscFree(eventLog->eventInfo);                                                                  CHKERRQ(ierr);
  ierr = PetscFree(eventLog);                                                                             CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*--------------------------------------------- Registration Functions ----------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "EventRegLogRegister"
/*@C
  EventRegLogRegister - Registers an event for logging operations in an application code.

  Not Collective

  Input Parameters:
+ eventLog - The EventLog
. ename    - The name associated with the event
- cookie   - The cookie associated to the class for this event

  Output Parameter:
. event    - The event

  Example of Usage:
.vb
      int USER_EVENT;
      int user_event_flops;
      PetscLogEventRegister(&USER_EVENT,"User event name");
      PetscLogEventBegin(USER_EVENT,0,0,0,0);
         [code segment to monitor]
         PetscLogFlops(user_event_flops);
      PetscLogEventEnd(USER_EVENT,0,0,0,0);
.ve

  Notes: 
  PETSc automatically logs library events if the code has been
  compiled with -DPETSC_USE_LOG (which is the default) and -log,
  -log_summary, or -log_all are specified.  PetscLogEventRegister() is
  intended for logging user events to supplement this PETSc
  information. 

  PETSc can gather data for use with the utilities Upshot/Nupshot
  (part of the MPICH distribution).  If PETSc has been compiled
  with flag -DPETSC_HAVE_MPE (MPE is an additional utility within
  MPICH), the user can employ another command line option, -log_mpe,
  to create a logfile, "mpe.log", which can be visualized
  Upshot/Nupshot.

  Level: intermediate

.keywords: log, event, register
.seealso: PetscLogEventBegin(), PetscLogEventEnd(), PetscLogFlops(), PetscLogEventMPEActivate(), PetscLogEventMPEDeactivate(),
          EventLogActivate(), EventLogDeactivate()
@*/
int EventRegLogRegister(EventRegLog eventLog, const char ename[], int cookie, PetscEvent *event) {
  EventRegInfo *eventInfo;
  char         *str;
  int           e;
  int           ierr;

  PetscFunctionBegin;
  PetscValidCharPointer(ename);
  PetscValidIntPointer(event);
  /* Should check cookie I think */
  e = eventLog->numEvents++;
  if (eventLog->numEvents > eventLog->maxEvents) {
    ierr = PetscMalloc(eventLog->maxEvents*2 * sizeof(EventRegInfo), &eventInfo);                         CHKERRQ(ierr);
    ierr = PetscMemcpy(eventInfo, eventLog->eventInfo, eventLog->maxEvents * sizeof(EventRegInfo));       CHKERRQ(ierr);
    ierr = PetscFree(eventLog->eventInfo);                                                                CHKERRQ(ierr);
    eventLog->eventInfo  = eventInfo;
    eventLog->maxEvents *= 2;
  }
  ierr = PetscStrallocpy(ename, &str);                                                                    CHKERRQ(ierr);
  eventLog->eventInfo[e].name   = str;
  eventLog->eventInfo[e].cookie = cookie;
#if defined(PETSC_HAVE_MPE)
  if (UseMPE) {
    char *color;
    int   rank, beginID, endID;

    beginID = MPE_Log_get_event_number();
    endID   = MPE_Log_get_event_number();
    eventLog->eventInfo[e].mpe_id_begin = beginID;
    eventLog->eventInfo[e].mpe_id_end   = endID;
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank); CHKERRQ(ierr);
    if (!rank) {
      ierr = PetscLogGetRGBColor(&color); CHKERRQ(ierr);
      MPE_Describe_state(beginID, endID, str, color);
    }
  }
#endif
  *event = e;
  PetscFunctionReturn(0);
}

/*---------------------------------------------- Activation Functions -----------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "EventPerfLogActivate"
/*@C
  EventPerfLogActivate - Indicates that a particular event should be logged.

  Not Collective

  Input Parameters:
+ eventLog - The EventPerfLog
- event    - The event

   Usage:
.vb
      EventPerfLogDeactivate(log, VEC_SetValues);
        [code where you do not want to log VecSetValues()]
      EventPerfLogActivate(log, VEC_SetValues);
        [code where you do want to log VecSetValues()]
.ve 

  Note:
  The event may be either a pre-defined PETSc event (found in 
  include/petsclog.h) or an event number obtained with EventRegLogRegister().

  Level: advanced

.keywords: log, event, activate
.seealso: PetscLogEventMPEDeactivate(), PetscLogEventMPEActivate(), EventPerfLogDeactivate()
@*/
int EventPerfLogActivate(EventPerfLog eventLog, PetscEvent event) {
  PetscFunctionBegin;
  eventLog->eventInfo[event].active = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EventPerfLogDeactivate"
/*@C
  EventPerfLogDeactivate - Indicates that a particular event should not be logged.

  Not Collective

  Input Parameters:
+ eventLog - The EventPerfLog
- event    - The event

   Usage:
.vb
      EventPerfLogDeactivate(log, VEC_SetValues);
        [code where you do not want to log VecSetValues()]
      EventPerfLogActivate(log, VEC_SetValues);
        [code where you do want to log VecSetValues()]
.ve 

  Note:
  The event may be either a pre-defined PETSc event (found in 
  include/petsclog.h) or an event number obtained with EventRegLogRegister().

  Level: advanced

.keywords: log, event, activate
.seealso: PetscLogEventMPEDeactivate(), PetscLogEventMPEActivate(), EventPerfLogActivate()
@*/
int EventPerfLogDeactivate(EventPerfLog eventLog, PetscEvent event) {
  PetscFunctionBegin;
  eventLog->eventInfo[event].active = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EventPerfLogActivateClass"
/*@C
  EventPerfLogActivateClass - Activates event logging for a PETSc object class.

  Not Collective

  Input Parameters:
+ eventLog    - The EventPerfLog
. eventRegLog - The EventRegLog
- cookie      - The class id, for example MAT_COOKIE, SNES_COOKIE,

  Level: developer

.seealso: EventPerfLogDeactivateClass(), EventPerfLogActivate(), EventPerfLogDeactivate()
@*/
int EventPerfLogActivateClass(EventPerfLog eventLog, EventRegLog eventRegLog, int cookie) {
  int c = eventRegLog->eventInfo[c].cookie;
  int e;

  PetscFunctionBegin;
  for(e = 0; e < eventLog->numEvents; e++) {
    if (c == cookie) eventLog->eventInfo[e].active = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EventPerfLogDeactivateClass"
/*@C
  EventPerfLogDeactivateClass - Deactivates event logging for a PETSc object class.

  Not Collective

  Input Parameters:
+ eventLog    - The EventPerfLog
. eventRegLog - The EventRegLog
- cookie - The class id, for example MAT_COOKIE, SNES_COOKIE,

  Level: developer

.seealso: EventPerfLogDeactivateClass(), EventPerfLogDeactivate(), EventPerfLogActivate()
@*/
int EventPerfLogDeactivateClass(EventPerfLog eventLog, EventRegLog eventRegLog, int cookie) {
  int c = eventRegLog->eventInfo[c].cookie;
  int e;

  PetscFunctionBegin;
  for(e = 0; e < eventLog->numEvents; e++) {
    if (c == cookie) eventLog->eventInfo[e].active = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------ Query Functions --------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "EventPerfLogSetVisible"
/*@C
  EventPerfLogSetVisible - This function determines whether an event is printed during PetscLogPrintSummary()

  Not Collective

  Input Parameters:
+ eventLog  - The EventPerfLog
. event     - The event to log
- isVisible - The visibility flag, PETSC_TRUE for printing, otherwise PETSC_FALSE (default is PETSC_TRUE)

  Database Options:
. -log_summary - Activates log summary

  Level: intermediate

.keywords: log, visible, event
.seealso: EventPerfLogGetVisible(), EventRegLogRegister(), StageLogGetEventLog()
@*/
int EventPerfLogSetVisible(EventPerfLog eventLog, PetscEvent event, PetscTruth isVisible) {
  PetscFunctionBegin;
  eventLog->eventInfo[event].visible = isVisible;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EventPerfLogGetVisible"
/*@C
  EventPerfLogGetVisible - This function returns whether an event is printed during PetscLogPrintSummary()

  Not Collective

  Input Parameters:
+ eventLog  - The EventPerfLog
- event     - The event id to log

  Output Parameter:
. isVisible - The visibility flag, PETSC_TRUE for printing, otherwise PETSC_FALSE (default is PETSC_TRUE)

  Database Options:
. -log_summary - Activates log summary

  Level: intermediate

.keywords: log, visible, event
.seealso: EventPerfLogSetVisible(), EventRegLogRegister(), StageLogGetEventLog()
@*/
int EventPerfLogGetVisible(EventPerfLog eventLog, PetscEvent event, PetscTruth *isVisible) {
  PetscFunctionBegin;
  PetscValidIntPointer(isVisible);
  *isVisible = eventLog->eventInfo[event].visible;
  PetscFunctionReturn(0);
}

/*------------------------------------------------ Action Functions -------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "PetscLogEventBeginDefault"
int PetscLogEventBeginDefault(PetscEvent event, int t, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4) {
  StageLog     stageLog;
  EventPerfLog eventLog;
  int          stage;
  int          ierr;

  PetscFunctionBegin;
  ierr = PetscLogGetStageLog(&stageLog);                                                                  CHKERRQ(ierr);
  ierr = StageLogGetCurrent(stageLog, &stage);                                                            CHKERRQ(ierr);
  ierr = StageLogGetEventPerfLog(stageLog, stage, &eventLog);                                             CHKERRQ(ierr);
  /* Check for double counting */
  eventLog->eventInfo[event].depth++;
  if (eventLog->eventInfo[event].depth > 1) PetscFunctionReturn(0);
  /* Log performance info */
  eventLog->eventInfo[event].count++;
  PetscTimeSubtract(eventLog->eventInfo[event].time);
  eventLog->eventInfo[event].flops         -= _TotalFlops;
  eventLog->eventInfo[event].numMessages   -= irecv_ct  + isend_ct  + recv_ct  + send_ct;
  eventLog->eventInfo[event].messageLength -= irecv_len + isend_len + recv_len + send_len;
  eventLog->eventInfo[event].numReductions -= allreduce_ct;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscLogEventEndDefault"
int PetscLogEventEndDefault(PetscEvent event, int t, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4) {
  StageLog     stageLog;
  EventPerfLog eventLog;
  int          stage;
  int          ierr;

  PetscFunctionBegin;
  ierr = PetscLogGetStageLog(&stageLog);                                                                  CHKERRQ(ierr);
  ierr = StageLogGetCurrent(stageLog, &stage);                                                            CHKERRQ(ierr);
  ierr = StageLogGetEventPerfLog(stageLog, stage, &eventLog);                                             CHKERRQ(ierr);
  /* Check for double counting */
  eventLog->eventInfo[event].depth--;
  if (eventLog->eventInfo[event].depth > 0) {
    PetscFunctionReturn(0);
  } else if (eventLog->eventInfo[event].depth < 0) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "Logging event had unbalanced begin/end pairs");
  }
  /* Log performance info */
  PetscTimeAdd(eventLog->eventInfo[event].time);
  eventLog->eventInfo[event].flops         += _TotalFlops;
  eventLog->eventInfo[event].numMessages   += irecv_ct  + isend_ct  + recv_ct  + send_ct;
  eventLog->eventInfo[event].messageLength += irecv_len + isend_len + recv_len + send_len;
  eventLog->eventInfo[event].numReductions += allreduce_ct;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscLogEventBeginComplete"
int PetscLogEventBeginComplete(PetscEvent event, int t, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4) {
  StageLog       stageLog;
  EventRegLog    eventRegLog;
  EventPerfLog   eventPerfLog;
  Action        *tmpAction;
  PetscLogDouble start, end;
  PetscLogDouble curTime;
  int            stage;
  int            ierr;

  PetscFunctionBegin;
  /* Dynamically enlarge logging structures */
  if (numActions >= maxActions) {
    PetscTime(start);
    ierr = PetscMalloc(maxActions*2 * sizeof(Action), &tmpAction);                                        CHKERRQ(ierr);
    ierr = PetscMemcpy(tmpAction, actions, maxActions * sizeof(Action));                                  CHKERRQ(ierr);
    ierr = PetscFree(actions);                                                                            CHKERRQ(ierr);
    actions     = tmpAction;
    maxActions *= 2;
    PetscTime(end);
    BaseTime += (end - start);
  }
  /* Record the event */
  ierr = PetscLogGetStageLog(&stageLog);                                                                  CHKERRQ(ierr);
  ierr = StageLogGetCurrent(stageLog, &stage);                                                            CHKERRQ(ierr);
  ierr = StageLogGetEventRegLog(stageLog, &eventRegLog);                                                  CHKERRQ(ierr);
  ierr = StageLogGetEventPerfLog(stageLog, stage, &eventPerfLog);                                         CHKERRQ(ierr);
  PetscTime(curTime);
  if (actions != PETSC_NULL) {
    actions[numActions].time     = curTime - BaseTime;
    actions[numActions++].action = ACTIONBEGIN;
    actions[numActions].event    = event;
    actions[numActions].cookie   = eventRegLog->eventInfo[event].cookie;
    if (o1) actions[numActions].id1 = o1->id; else actions[numActions].id1 = -1;
    if (o2) actions[numActions].id2 = o2->id; else actions[numActions].id2 = -1;
    if (o3) actions[numActions].id3 = o3->id; else actions[numActions].id3 = -1;
    actions[numActions].flops    = _TotalFlops;
    ierr = PetscTrSpace(&actions[numActions].mem, PETSC_NULL, &actions[numActions].maxmem);               CHKERRQ(ierr);
  }
  /* Check for double counting */
  eventPerfLog->eventInfo[event].depth++;
  if (eventPerfLog->eventInfo[event].depth > 1) PetscFunctionReturn(0);
  /* Log the performance info */
  eventPerfLog->eventInfo[event].count++;
  eventPerfLog->eventInfo[event].time          -= curTime;
  eventPerfLog->eventInfo[event].flops         -= _TotalFlops;
  eventPerfLog->eventInfo[event].numMessages   -= irecv_ct  + isend_ct  + recv_ct  + send_ct;
  eventPerfLog->eventInfo[event].messageLength -= irecv_len + isend_len + recv_len + send_len;
  eventPerfLog->eventInfo[event].numReductions -= allreduce_ct;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscLogEventEndComplete"
int PetscLogEventEndComplete(PetscEvent event, int t, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4) {
  StageLog       stageLog;
  EventRegLog    eventRegLog;
  EventPerfLog   eventPerfLog;
  Action        *tmpAction;
  PetscLogDouble start, end;
  PetscLogDouble curTime;
  int            stage;
  int            ierr;

  PetscFunctionBegin;
  /* Dynamically enlarge logging structures */
  if (numActions >= maxActions) {
    PetscTime(start);
    ierr = PetscMalloc(maxActions*2 * sizeof(Action), &tmpAction);                                        CHKERRQ(ierr);
    ierr = PetscMemcpy(tmpAction, actions, maxActions * sizeof(Action));                                  CHKERRQ(ierr);
    ierr = PetscFree(actions);                                                                            CHKERRQ(ierr);
    actions     = tmpAction;
    maxActions *= 2;
    PetscTime(end);
    BaseTime += (end - start);
  }
  /* Record the event */
  ierr = PetscLogGetStageLog(&stageLog);                                                                  CHKERRQ(ierr);
  ierr = StageLogGetCurrent(stageLog, &stage);                                                            CHKERRQ(ierr);
  ierr = StageLogGetEventRegLog(stageLog, &eventRegLog);                                                  CHKERRQ(ierr);
  ierr = StageLogGetEventPerfLog(stageLog, stage, &eventPerfLog);                                         CHKERRQ(ierr);
  PetscTime(curTime);
  if (actions != PETSC_NULL) {
    actions[numActions].time     = curTime - BaseTime;
    actions[numActions++].action = ACTIONEND;
    actions[numActions].event    = event;
    actions[numActions].cookie   = eventRegLog->eventInfo[event].cookie;
    if (o1) actions[numActions].id1 = o1->id; else actions[numActions].id1 = -1;
    if (o2) actions[numActions].id2 = o2->id; else actions[numActions].id2 = -1;
    if (o3) actions[numActions].id3 = o3->id; else actions[numActions].id3 = -1;
    actions[numActions].flops    = _TotalFlops;
    ierr = PetscTrSpace(&actions[numActions].mem, PETSC_NULL, &actions[numActions].maxmem);               CHKERRQ(ierr);
  }
  /* Check for double counting */
  eventPerfLog->eventInfo[event].depth--;
  if (eventPerfLog->eventInfo[event].depth > 0) {
    PetscFunctionReturn(0);
  } else if (eventPerfLog->eventInfo[event].depth < 0) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "Logging event had unbalanced begin/end pairs");
  }
  /* Log the performance info */
  eventPerfLog->eventInfo[event].count++;
  eventPerfLog->eventInfo[event].time          += curTime;
  eventPerfLog->eventInfo[event].flops         += _TotalFlops;
  eventPerfLog->eventInfo[event].numMessages   += irecv_ct  + isend_ct  + recv_ct  + send_ct;
  eventPerfLog->eventInfo[event].messageLength += irecv_len + isend_len + recv_len + send_len;
  eventPerfLog->eventInfo[event].numReductions += allreduce_ct;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscLogEventBeginTrace"
int PetscLogEventBeginTrace(PetscEvent event, int t, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4) {
  StageLog       stageLog;
  EventRegLog    eventRegLog;
  EventPerfLog   eventPerfLog;
  PetscLogDouble cur_time;
  int            rank, stage;
  int            ierr;

  PetscFunctionBegin;
  if (tracetime == 0.0) {PetscTime(tracetime);}

  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);                                                          CHKERRQ(ierr);
  ierr = PetscLogGetStageLog(&stageLog);                                                                  CHKERRQ(ierr);
  ierr = StageLogGetCurrent(stageLog, &stage);                                                            CHKERRQ(ierr);
  ierr = StageLogGetEventRegLog(stageLog, &eventRegLog);                                                  CHKERRQ(ierr);
  ierr = StageLogGetEventPerfLog(stageLog, stage, &eventPerfLog);                                         CHKERRQ(ierr);
  /* Check for double counting */
  eventPerfLog->eventInfo[event].depth++;
  if (eventPerfLog->eventInfo[event].depth > 1) PetscFunctionReturn(0);
  /* Log performance info */
  ierr = PetscStrncpy(tracespace, traceblanks, 2*tracelevel);                                             CHKERRQ(ierr);
  tracespace[2*tracelevel] = 0;
  PetscTime(cur_time);
  fprintf(tracefile, "%s[%d] %g Event begin: %s\n", tracespace, rank, cur_time-tracetime, eventRegLog->eventInfo[event].name);
  fflush(tracefile);
  tracelevel++;

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscLogEventEndTrace"
int PetscLogEventEndTrace(PetscEvent event,int t,PetscObject o1,PetscObject o2,PetscObject o3,PetscObject o4) {
  StageLog       stageLog;
  EventRegLog    eventRegLog;
  EventPerfLog   eventPerfLog;
  PetscLogDouble cur_time;
  int            rank, stage;
  int            ierr;

  PetscFunctionBegin;
  tracelevel--;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);                                                          CHKERRQ(ierr);
  ierr = PetscLogGetStageLog(&stageLog);                                                                  CHKERRQ(ierr);
  ierr = StageLogGetCurrent(stageLog, &stage);                                                            CHKERRQ(ierr);
  ierr = StageLogGetEventRegLog(stageLog, &eventRegLog);                                                  CHKERRQ(ierr);
  ierr = StageLogGetEventPerfLog(stageLog, stage, &eventPerfLog);                                         CHKERRQ(ierr);
  /* Check for double counting */
  eventPerfLog->eventInfo[event].depth--;
  if (eventPerfLog->eventInfo[event].depth > 0) {
    PetscFunctionReturn(0);
  } else if (eventPerfLog->eventInfo[event].depth < 0) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "Logging event had unbalanced begin/end pairs");
  }
  /* Log performance info */
  ierr = PetscStrncpy(tracespace, traceblanks, 2*tracelevel);                                             CHKERRQ(ierr);
  tracespace[2*tracelevel] = 0;
  PetscTime(cur_time);
  fprintf(tracefile, "%s[%d] %g Event end: %s\n", tracespace, rank, cur_time-tracetime, eventRegLog->eventInfo[event].name);
  fflush(tracefile);
  PetscFunctionReturn(0);
}
