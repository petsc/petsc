#include <petscsys.h>
#include <petsctime.h>

/* A simple stack */
struct _n_IntStack {
  int  top;   /* The top of the stack */
  int  max;   /* The maximum stack size */
  int *stack; /* The storage */
};

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
extern PetscBool  logActions;
extern PetscBool  logObjects;
extern int        numActions, maxActions;
extern int        numObjects, maxObjects;
extern int        numObjectsDestroyed;


#ifdef PETSC_USE_LOG

/* Runtime functions */
extern PetscErrorCode StageLogGetClassRegLog(StageLog, ClassRegLog *);
extern PetscErrorCode StageLogGetEventRegLog(StageLog, EventRegLog *);
extern PetscErrorCode StageLogGetClassPerfLog(StageLog, int, ClassPerfLog *);


/* Creation and destruction functions */
extern PetscErrorCode EventRegLogCreate(EventRegLog *);
extern PetscErrorCode EventRegLogDestroy(EventRegLog);
extern PetscErrorCode EventPerfLogCreate(EventPerfLog *);
extern PetscErrorCode EventPerfLogDestroy(EventPerfLog);
/* General functions */
extern PetscErrorCode EventPerfLogEnsureSize(EventPerfLog, int);
extern PetscErrorCode EventPerfInfoClear(EventPerfInfo *);
extern PetscErrorCode EventPerfInfoCopy(EventPerfInfo *, EventPerfInfo *);
/* Registration functions */
extern PetscErrorCode EventRegLogRegister(EventRegLog, const char [], PetscClassId, PetscLogEvent *);
/* Query functions */
extern PetscErrorCode EventPerfLogSetVisible(EventPerfLog, PetscLogEvent, PetscBool );
extern PetscErrorCode EventPerfLogGetVisible(EventPerfLog, PetscLogEvent, PetscBool  *);
/* Activaton functions */
extern PetscErrorCode EventPerfLogActivate(EventPerfLog, PetscLogEvent);
extern PetscErrorCode EventPerfLogDeactivate(EventPerfLog, PetscLogEvent);
extern PetscErrorCode EventPerfLogActivateClass(EventPerfLog, EventRegLog, PetscClassId);
extern PetscErrorCode EventPerfLogDeactivateClass(EventPerfLog, EventRegLog, PetscClassId);

/* Logging functions */
extern PetscErrorCode PetscLogEventBeginDefault(PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject);
extern PetscErrorCode PetscLogEventEndDefault(PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject);
extern PetscErrorCode PetscLogEventBeginComplete(PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject);
extern PetscErrorCode PetscLogEventEndComplete(PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject);
extern PetscErrorCode PetscLogEventBeginTrace(PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject);
extern PetscErrorCode PetscLogEventEndTrace(PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject);

/* Creation and destruction functions */
extern PetscErrorCode ClassRegLogCreate(ClassRegLog *);
extern PetscErrorCode ClassRegLogDestroy(ClassRegLog);
extern PetscErrorCode ClassPerfLogCreate(ClassPerfLog *);
extern PetscErrorCode ClassPerfLogDestroy(ClassPerfLog);
extern PetscErrorCode ClassRegInfoDestroy(ClassRegInfo *);
/* General functions */
extern PetscErrorCode ClassPerfLogEnsureSize(ClassPerfLog, int);
extern PetscErrorCode ClassPerfInfoClear(ClassPerfInfo *);
/* Registration functions */
extern PetscErrorCode ClassRegLogRegister(ClassRegLog, const char [], PetscClassId);
/* Query functions */
extern PetscErrorCode ClassRegLogGetClass(ClassRegLog, PetscClassId, int *);
/* Logging functions */
extern PetscErrorCode PetscLogObjCreateDefault(PetscObject);
extern PetscErrorCode PetscLogObjDestroyDefault(PetscObject);

/* Creation and destruction functions */
extern PetscErrorCode  StageLogCreate(StageLog *);
extern PetscErrorCode  StageLogDestroy(StageLog);
/* Registration functions */
extern PetscErrorCode  StageLogRegister(StageLog, const char [], int *);
/* Runtime functions */
extern PetscErrorCode  StageLogPush(StageLog, int);
extern PetscErrorCode  StageLogPop(StageLog);
extern PetscErrorCode  StageLogSetActive(StageLog, int, PetscBool );
extern PetscErrorCode  StageLogGetActive(StageLog, int, PetscBool  *);
extern PetscErrorCode  StageLogSetVisible(StageLog, int, PetscBool );
extern PetscErrorCode  StageLogGetVisible(StageLog, int, PetscBool  *);
extern PetscErrorCode  StageLogGetStage(StageLog, const char [], int *);
extern PetscErrorCode  StageLogGetClassRegLog(StageLog, ClassRegLog *);
extern PetscErrorCode  StageLogGetEventRegLog(StageLog, EventRegLog *);
extern PetscErrorCode  StageLogGetClassPerfLog(StageLog, int, ClassPerfLog *);

extern PetscErrorCode  EventRegLogGetEvent(EventRegLog, const char [], PetscLogEvent *);


#endif /* PETSC_USE_LOG */
