

#include "petscsys.h"
#include "petsctime.h"
#include "petsclog.h"

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

/* A simple stack */
struct _n_IntStack {
  int  top;   /* The top of the stack */
  int  max;   /* The maximum stack size */
  int *stack; /* The storage */
};

#ifdef PETSC_USE_LOG

/* Runtime functions */
EXTERN PetscErrorCode StageLogGetClassRegLog(StageLog, ClassRegLog *);
EXTERN PetscErrorCode StageLogGetEventRegLog(StageLog, EventRegLog *);
EXTERN PetscErrorCode StageLogGetClassPerfLog(StageLog, int, ClassPerfLog *);
EXTERN PetscErrorCode StageLogGetEventPerfLog(StageLog, int, EventPerfLog *);
/* Stack Functions */
EXTERN PetscErrorCode StackCreate(IntStack *);
EXTERN PetscErrorCode StackDestroy(IntStack);
EXTERN PetscErrorCode StackPush(IntStack, int);
EXTERN PetscErrorCode StackPop(IntStack, int *);
EXTERN PetscErrorCode StackTop(IntStack, int *);
EXTERN PetscErrorCode StackEmpty(IntStack, PetscTruth *);

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
