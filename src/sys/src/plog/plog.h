/* $Id: plog.h,v 1.1 2000/01/10 03:28:57 knepley Exp $ */

/* The class naming scheme procedes as follows:

   Event:
     Events are a class which describes certain blocks of executable
     code. The corresponding instantiations of events are Actions.

   Class:
     Classes are the classes representing Petsc structures. The
     corresponding instantiations are called Objects.

   StageLog:
     This class holds information about stages of computation. These
     are understood to be chunks encompassing several events. This
     log contains a separate EventLog for each stage, as well as, a
     separate ClassLog for each stage.

   EventLog:
     This class holds the performance information logged for each
     event. Usually this information is logged for only one stage.

   ClassLog:
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
extern Action *actions;
extern Object *objects;
extern int     numActions, maxActions;
extern int     numObjects, maxObjects;
extern int     numObjectsDestroyed;

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
EXTERN int StageLogSetVisible(StageLog, int, PetscTruth);
EXTERN int StageLogGetVisible(StageLog, int, PetscTruth *);
EXTERN int StageLogGetClassLog(StageLog, int, ClassLog *);
EXTERN int StageLogGetEventLog(StageLog, int, EventLog *);
EXTERN int StageLogGetStage(StageLog, const char [], int *);
/* Stack Functions */
EXTERN int StackCreate(IntStack *);
EXTERN int StackDestroy(IntStack);
EXTERN int StackPush(IntStack, int);
EXTERN int StackPop(IntStack, int *);
EXTERN int StackTop(IntStack, int *);
EXTERN int StackEmpty(IntStack, PetscTruth *);

/* Creation and destruction functions */
EXTERN int EventLogCreate(EventLog *);
EXTERN int EventLogDestroy(EventLog);
EXTERN int EventLogCopy(EventLog, EventLog *);
/* Registration functions */
EXTERN int EventLogRegister(EventLog, const char [], int, int, int, int *);
/* Query functions */
EXTERN int EventLogSetVisible(EventLog, int, PetscTruth);
EXTERN int EventLogGetVisible(EventLog, int, PetscTruth *);
EXTERN int EventLogGetEvent(EventLog, int, int *);
/* Activaton functions */
EXTERN int EventLogActivate(EventLog, int);
EXTERN int EventLogDeactivate(EventLog, int);
EXTERN int EventLogActivateClass(EventLog, int);
EXTERN int EventLogDeactivateClass(EventLog, int);

/* Logging functions */
EXTERN int PetscLogEventBeginDefault(int, int, PetscObject, PetscObject, PetscObject, PetscObject);
EXTERN int PetscLogEventEndDefault(int, int, PetscObject, PetscObject, PetscObject, PetscObject);
EXTERN int PetscLogEventBeginComplete(int, int, PetscObject, PetscObject, PetscObject, PetscObject);
EXTERN int PetscLogEventEndComplete(int, int, PetscObject, PetscObject, PetscObject, PetscObject);
EXTERN int PetscLogEventBeginTrace(int, int, PetscObject, PetscObject, PetscObject, PetscObject);
EXTERN int PetscLogEventEndTrace(int, int, PetscObject, PetscObject, PetscObject, PetscObject);

/* Creation and destruction functions */
EXTERN int ClassLogCreate(ClassLog *);
EXTERN int ClassLogDestroy(ClassLog);
EXTERN int ClassLogCopy(ClassLog, ClassLog *);
/* Registration functions */
EXTERN int ClassLogRegister(ClassLog, const char [], int *);
/* Query functions */
EXTERN int ClassLogGetClass(ClassLog, int, int *);
/* Logging functions */
EXTERN int PetscLogObjCreateDefault(PetscObject);
EXTERN int PetscLogObjDestroyDefault(PetscObject);

/* Creation and destruction functions */
EXTERN int PerfInfoDestroy(PerfInfo *);
EXTERN int ClassInfoDestroy(ClassInfo *);

#endif /* PETSC_USE_LOG */
