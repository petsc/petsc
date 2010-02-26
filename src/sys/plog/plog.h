
/* The class naming scheme procedes as follows:

   Event:
     Events are a class which describes certain blocks of executable
     code. The corresponding instantiations of events are Actions.

   Class:
     Classes are the classes representing Petsc structures. The
     corresponding instantiations are called Objects.

   StageLog:
     This type holds information about stages of computation. These
     are understood to be chunks encompassing several events, or
     alternatively, as a covering (possibly nested) of the timeline.

   StageInfo:
     The information about each stage. This log contains an
     EventPerfLog and a ClassPerfLog.

   EventRegLog:
     This type holds the information generated for each event as
     it is registered. This information does not change and thus is
     stored separately from performance information.

   EventPerfLog:
     This type holds the performance information logged for each
     event. Usually this information is logged for only one stage.

   ClassRegLog:
     This type holds the information generated for each class as
     it is registered. This information does not change and thus is
     stored separately from performance information.

   ClassPerfLog:
     This class holds information describing class/object usage during
     a run. Usually this information is logged for only one stage.
*/

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
  PetscCookie    cookie;        /* The event class id */
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
EXTERN PetscErrorCode EventRegLogRegister(EventRegLog, const char [], PetscCookie, PetscLogEvent *);
/* Query functions */
EXTERN PetscErrorCode EventPerfLogSetVisible(EventPerfLog, PetscLogEvent, PetscTruth);
EXTERN PetscErrorCode EventPerfLogGetVisible(EventPerfLog, PetscLogEvent, PetscTruth *);
/* Activaton functions */
EXTERN PetscErrorCode EventPerfLogActivate(EventPerfLog, PetscLogEvent);
EXTERN PetscErrorCode EventPerfLogDeactivate(EventPerfLog, PetscLogEvent);
EXTERN PetscErrorCode EventPerfLogActivateClass(EventPerfLog, EventRegLog, PetscCookie);
EXTERN PetscErrorCode EventPerfLogDeactivateClass(EventPerfLog, EventRegLog, PetscCookie);

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
EXTERN PetscErrorCode ClassRegLogRegister(ClassRegLog, const char [], PetscCookie);
/* Query functions */
EXTERN PetscErrorCode ClassRegLogGetClass(ClassRegLog, PetscCookie, int *);
/* Logging functions */
EXTERN PetscErrorCode PetscLogObjCreateDefault(PetscObject);
EXTERN PetscErrorCode PetscLogObjDestroyDefault(PetscObject);

#endif /* PETSC_USE_LOG */
