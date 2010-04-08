

#include "petscsys.h"
#include "petsctime.h"
#include "petsclog.h"

/* A simple stack */
struct _n_IntStack {
  int  top;   /* The top of the stack */
  int  max;   /* The maximum stack size */
  int *stack; /* The storage */
};

#ifdef PETSC_USE_LOG
/* Stack Functions */
EXTERN PetscErrorCode StackCreate(IntStack *);
EXTERN PetscErrorCode StackDestroy(IntStack);
EXTERN PetscErrorCode StackPush(IntStack, int);
EXTERN PetscErrorCode StackPop(IntStack, int *);
EXTERN PetscErrorCode StackEmpty(IntStack, PetscTruth *);
EXTERN PetscErrorCode StackTop(IntStack, int *);
#endif /* PETSC_USE_LOG */

#undef __FUNCT__  
#define __FUNCT__ "PetscLogGetStageLog"
/*@C
  PetscLogGetStageLog - This function returns the default stage logging object.

  Not collective

  Output Parameter:
. stageLog - The default StageLog

  Level: developer

  Developer Notes: Inline since called for EACH PetscEventLogBeginDefault() and PetscEventLogEndDefault()

.keywords: log, stage
.seealso: StageLogCreate()
@*/
PETSC_STATIC_INLINE PetscErrorCode PETSC_DLLEXPORT PetscLogGetStageLog(StageLog *stageLog)
{
  PetscFunctionBegin;
  PetscValidPointer(stageLog,1);
  if (_stageLog == PETSC_NULL) {
    fprintf(stderr, "Logging has not been enabled.\nYou might have forgotten to call PetscInitialize().\n");
    MPI_Abort(MPI_COMM_WORLD, PETSC_ERR_SUP);
  }
  *stageLog = _stageLog;
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

  Level: developer

  Developer Notes: Inline since called for EACH PetscEventLogBeginDefault() and PetscEventLogEndDefault()

.keywords: log, stage
.seealso: StageLogPush(), StageLogPop(), PetscLogGetStageLog()
@*/
PETSC_STATIC_INLINE PetscErrorCode PETSC_DLLEXPORT StageLogGetCurrent(StageLog stageLog, int *stage) 
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
#define __FUNCT__ "StageLogGetEventPerfLog"
/*@C
  StageLogGetEventPerfLog - This function returns the EventPerfLog for the given stage.

  Not Collective

  Input Parameters:
+ stageLog - The StageLog
- stage    - The stage

  Output Parameter:
. eventLog - The EventPerfLog

  Level: developer

  Developer Notes: Inline since called for EACH PetscEventLogBeginDefault() and PetscEventLogEndDefault()

.keywords: log, stage
.seealso: StageLogPush(), StageLogPop(), PetscLogGetStageLog()
@*/
PETSC_STATIC_INLINE PetscErrorCode PETSC_DLLEXPORT StageLogGetEventPerfLog(StageLog stageLog, int stage, EventPerfLog *eventLog)
{
  PetscFunctionBegin;
  PetscValidPointer(eventLog,3);
  if ((stage < 0) || (stage >= stageLog->numStages)) {
    SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE, "Invalid stage %d should be in [0,%d)", stage, stageLog->numStages);
  }
  *eventLog = stageLog->stageInfo[stage].eventLog;
  PetscFunctionReturn(0);
}

/* The structure for action logging */
#define CREATE      0
#define DESTROY     1
#define ACTIONBEGIN 2
#define ACTIONEND   3
typedef struct _Action {
  int            action;        /* The type of execution */
  PetscLogEvent  event;         /* The event number */
  PetscClassId   classid;        /* The event class id */
  PetscLogDouble time;          /* The time of occurence */
  PetscLogDouble flops;         /* The cumlative flops */
  PetscLogDouble mem;           /* The current memory usage */
  PetscLogDouble maxmem;        /* The maximum memory usage */
  int            id1, id2, id3; /* The ids of associated objects */
} Action;

/* The structure for object logging */
typedef struct _Object {
  PetscObject    obj;      /* The associated PetscObject */
  int            parent;   /* The parent id */
  PetscLogDouble mem;      /* The memory associated with the object */
  char           name[64]; /* The object name */
  char           info[64]; /* The information string */
} Object;

/* Action and object logging variables */
extern Action    *actions;
extern Object    *objects;
extern PetscTruth logActions;
extern PetscTruth logObjects;
extern int        numActions, maxActions;
extern int        numObjects, maxObjects;
extern int        numObjectsDestroyed;


#ifdef PETSC_USE_LOG

/* Runtime functions */
EXTERN PetscErrorCode StageLogGetClassRegLog(StageLog, ClassRegLog *);
EXTERN PetscErrorCode StageLogGetEventRegLog(StageLog, EventRegLog *);
EXTERN PetscErrorCode StageLogGetClassPerfLog(StageLog, int, ClassPerfLog *);
EXTERN PetscErrorCode StageLogGetEventPerfLog(StageLog, int, EventPerfLog *);

/* Creation and destruction functions */
EXTERN PetscErrorCode EventRegLogCreate(EventRegLog *);
EXTERN PetscErrorCode EventRegLogDestroy(EventRegLog);
EXTERN PetscErrorCode EventPerfLogCreate(EventPerfLog *);
EXTERN PetscErrorCode EventPerfLogDestroy(EventPerfLog);
/* General functions */
EXTERN PetscErrorCode EventPerfLogEnsureSize(EventPerfLog, int);
EXTERN PetscErrorCode EventPerfInfoClear(EventPerfInfo *);
EXTERN PetscErrorCode EventPerfInfoCopy(EventPerfInfo *, EventPerfInfo *);
/* Registration functions */
EXTERN PetscErrorCode EventRegLogRegister(EventRegLog, const char [], PetscClassId, PetscLogEvent *);
/* Query functions */
EXTERN PetscErrorCode EventPerfLogSetVisible(EventPerfLog, PetscLogEvent, PetscTruth);
EXTERN PetscErrorCode EventPerfLogGetVisible(EventPerfLog, PetscLogEvent, PetscTruth *);
/* Activaton functions */
EXTERN PetscErrorCode EventPerfLogActivate(EventPerfLog, PetscLogEvent);
EXTERN PetscErrorCode EventPerfLogDeactivate(EventPerfLog, PetscLogEvent);
EXTERN PetscErrorCode EventPerfLogActivateClass(EventPerfLog, EventRegLog, PetscClassId);
EXTERN PetscErrorCode EventPerfLogDeactivateClass(EventPerfLog, EventRegLog, PetscClassId);

/* Logging functions */
EXTERN PetscErrorCode PetscLogEventBeginDefault(PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject);
EXTERN PetscErrorCode PetscLogEventEndDefault(PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject);
EXTERN PetscErrorCode PetscLogEventBeginComplete(PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject);
EXTERN PetscErrorCode PetscLogEventEndComplete(PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject);
EXTERN PetscErrorCode PetscLogEventBeginTrace(PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject);
EXTERN PetscErrorCode PetscLogEventEndTrace(PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject);

/* Creation and destruction functions */
EXTERN PetscErrorCode ClassRegLogCreate(ClassRegLog *);
EXTERN PetscErrorCode ClassRegLogDestroy(ClassRegLog);
EXTERN PetscErrorCode ClassPerfLogCreate(ClassPerfLog *);
EXTERN PetscErrorCode ClassPerfLogDestroy(ClassPerfLog);
EXTERN PetscErrorCode ClassRegInfoDestroy(ClassRegInfo *);
/* General functions */
EXTERN PetscErrorCode ClassPerfLogEnsureSize(ClassPerfLog, int);
EXTERN PetscErrorCode ClassPerfInfoClear(ClassPerfInfo *);
/* Registration functions */
EXTERN PetscErrorCode ClassRegLogRegister(ClassRegLog, const char [], PetscClassId);
/* Query functions */
EXTERN PetscErrorCode ClassRegLogGetClass(ClassRegLog, PetscClassId, int *);
/* Logging functions */
EXTERN PetscErrorCode PetscLogObjCreateDefault(PetscObject);
EXTERN PetscErrorCode PetscLogObjDestroyDefault(PetscObject);

/* Creation and destruction functions */
EXTERN PetscErrorCode PETSC_DLLEXPORT StageLogCreate(StageLog *);
EXTERN PetscErrorCode PETSC_DLLEXPORT StageLogDestroy(StageLog);
/* Registration functions */
EXTERN PetscErrorCode PETSC_DLLEXPORT StageLogRegister(StageLog, const char [], int *);
/* Runtime functions */
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscLogGetStageLog(StageLog *);
EXTERN PetscErrorCode PETSC_DLLEXPORT StageLogPush(StageLog, int);
EXTERN PetscErrorCode PETSC_DLLEXPORT StageLogPop(StageLog);
EXTERN PetscErrorCode PETSC_DLLEXPORT StageLogGetCurrent(StageLog, int *);
EXTERN PetscErrorCode PETSC_DLLEXPORT StageLogSetActive(StageLog, int, PetscTruth);
EXTERN PetscErrorCode PETSC_DLLEXPORT StageLogGetActive(StageLog, int, PetscTruth *);
EXTERN PetscErrorCode PETSC_DLLEXPORT StageLogSetVisible(StageLog, int, PetscTruth);
EXTERN PetscErrorCode PETSC_DLLEXPORT StageLogGetVisible(StageLog, int, PetscTruth *);
EXTERN PetscErrorCode PETSC_DLLEXPORT StageLogGetStage(StageLog, const char [], int *);
EXTERN PetscErrorCode PETSC_DLLEXPORT StageLogGetClassRegLog(StageLog, ClassRegLog *);
EXTERN PetscErrorCode PETSC_DLLEXPORT StageLogGetEventRegLog(StageLog, EventRegLog *);
EXTERN PetscErrorCode PETSC_DLLEXPORT StageLogGetClassPerfLog(StageLog, int, ClassPerfLog *);
EXTERN PetscErrorCode PETSC_DLLEXPORT StageLogGetEventPerfLog(StageLog, int, EventPerfLog *);

EXTERN PetscErrorCode PETSC_DLLEXPORT EventRegLogGetEvent(EventRegLog, const char [], PetscLogEvent *);


#endif /* PETSC_USE_LOG */
