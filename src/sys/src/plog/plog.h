/* $Id: plog.h,v 1.1 2000/01/10 03:28:57 knepley Exp $ */

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

/* The structure for action logging */
#define CREATE      0
#define DESTROY     1
#define ACTIONBEGIN 2
#define ACTIONEND   3
typedef struct _Action {
  int            action;        /* The type of execution */
  int            event;         /* The event number */
  int            cookie;        /* The event class id */
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

/* Global counters */
extern PetscLogDouble BaseTime;

/* A simple stack */
struct _IntStack {
  int  top;   /* The top of the stack */
  int  max;   /* The maximum stack size */
  int *stack; /* The storage */
};

#ifdef PETSC_USE_LOG

/* Creation and destruction functions */
EXTERN int StageLogCreate(StageLog *);
EXTERN int StageLogDestroy(StageLog);
/* Registration functions */
EXTERN int StageLogRegister(StageLog, const char [], int *);
/* Runtime functions */
EXTERN int PetscLogGetStageLog(StageLog *);
EXTERN int StageLogPush(StageLog, int);
EXTERN int StageLogPop(StageLog);
EXTERN int StageLogGetCurrent(StageLog, int *);
EXTERN int StageLogSetActive(StageLog, int, PetscTruth);
EXTERN int StageLogGetActive(StageLog, int, PetscTruth *);
EXTERN int StageLogSetVisible(StageLog, int, PetscTruth);
EXTERN int StageLogGetVisible(StageLog, int, PetscTruth *);
EXTERN int StageLogGetClassRegLog(StageLog, ClassRegLog *);
EXTERN int StageLogGetEventRegLog(StageLog, EventRegLog *);
EXTERN int StageLogGetClassPerfLog(StageLog, int, ClassPerfLog *);
EXTERN int StageLogGetEventPerfLog(StageLog, int, EventPerfLog *);
EXTERN int StageLogGetStage(StageLog, const char [], int *);
/* Stack Functions */
EXTERN int StackCreate(IntStack *);
EXTERN int StackDestroy(IntStack);
EXTERN int StackPush(IntStack, int);
EXTERN int StackPop(IntStack, int *);
EXTERN int StackTop(IntStack, int *);
EXTERN int StackEmpty(IntStack, PetscTruth *);

/* Creation and destruction functions */
EXTERN int EventRegLogCreate(EventRegLog *);
EXTERN int EventRegLogDestroy(EventRegLog);
EXTERN int EventPerfLogCreate(EventPerfLog *);
EXTERN int EventPerfLogDestroy(EventPerfLog);
/* General functions */
EXTERN int EventPerfLogEnsureSize(EventPerfLog, int);
EXTERN int EventPerfInfoClear(EventPerfInfo *);
/* Registration functions */
EXTERN int EventRegLogRegister(EventRegLog, const char [], int, int *);
/* Query functions */
EXTERN int EventPerfLogSetVisible(EventPerfLog, int, PetscTruth);
EXTERN int EventPerfLogGetVisible(EventPerfLog, int, PetscTruth *);
EXTERN int EventRegLogGetEvent(EventRegLog, int, int *);
/* Activaton functions */
EXTERN int EventPerfLogActivate(EventPerfLog, int);
EXTERN int EventPerfLogDeactivate(EventPerfLog, int);
EXTERN int EventPerfLogActivateClass(EventPerfLog, EventRegLog, int);
EXTERN int EventPerfLogDeactivateClass(EventPerfLog, EventRegLog, int);

/* Logging functions */
EXTERN int PetscLogEventBeginDefault(int, int, PetscObject, PetscObject, PetscObject, PetscObject);
EXTERN int PetscLogEventEndDefault(int, int, PetscObject, PetscObject, PetscObject, PetscObject);
EXTERN int PetscLogEventBeginComplete(int, int, PetscObject, PetscObject, PetscObject, PetscObject);
EXTERN int PetscLogEventEndComplete(int, int, PetscObject, PetscObject, PetscObject, PetscObject);
EXTERN int PetscLogEventBeginTrace(int, int, PetscObject, PetscObject, PetscObject, PetscObject);
EXTERN int PetscLogEventEndTrace(int, int, PetscObject, PetscObject, PetscObject, PetscObject);

/* Creation and destruction functions */
EXTERN int ClassRegLogCreate(ClassRegLog *);
EXTERN int ClassRegLogDestroy(ClassRegLog);
EXTERN int ClassPerfLogCreate(ClassPerfLog *);
EXTERN int ClassPerfLogDestroy(ClassPerfLog);
EXTERN int ClassRegInfoDestroy(ClassRegInfo *);
/* General functions */
EXTERN int ClassPerfLogEnsureSize(ClassPerfLog, int);
EXTERN int ClassPerfInfoClear(ClassPerfInfo *);
/* Registration functions */
EXTERN int ClassRegLogRegister(ClassRegLog, const char [], int *);
/* Query functions */
EXTERN int ClassRegLogGetClass(ClassRegLog, int, int *);
/* Logging functions */
EXTERN int PetscLogObjCreateDefault(PetscObject);
EXTERN int PetscLogObjDestroyDefault(PetscObject);

#endif /* PETSC_USE_LOG */
