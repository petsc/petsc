#include <petscsys.h>
#include <petsctime.h>

/* A simple stack */
struct _n_PetscIntStack {
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
PETSC_EXTERN Action    *petsc_actions;
PETSC_EXTERN Object    *petsc_objects;
PETSC_EXTERN PetscBool petsc_logActions;
PETSC_EXTERN PetscBool petsc_logObjects;
PETSC_EXTERN int        petsc_numActions, petsc_maxActions;
PETSC_EXTERN int        petsc_numObjects, petsc_maxObjects;
PETSC_EXTERN int        petsc_numObjectsDestroyed;

PETSC_EXTERN FILE          *petsc_tracefile;
PETSC_EXTERN int            petsc_tracelevel;
PETSC_EXTERN const char    *petsc_traceblanks;
PETSC_EXTERN char           petsc_tracespace[128];
PETSC_EXTERN PetscLogDouble petsc_tracetime;

#ifdef PETSC_USE_LOG

/* Runtime functions */
PETSC_EXTERN PetscErrorCode PetscStageLogGetClassRegLog(PetscStageLog, PetscClassRegLog *);
PETSC_EXTERN PetscErrorCode PetscStageLogGetEventRegLog(PetscStageLog, PetscEventRegLog *);
PETSC_EXTERN PetscErrorCode PetscStageLogGetClassPerfLog(PetscStageLog, int, PetscClassPerfLog *);


/* Creation and destruction functions */
PETSC_EXTERN PetscErrorCode EventRegLogCreate(PetscEventRegLog *);
PETSC_EXTERN PetscErrorCode EventRegLogDestroy(PetscEventRegLog);
PETSC_EXTERN PetscErrorCode EventPerfLogCreate(PetscEventPerfLog *);
PETSC_EXTERN PetscErrorCode EventPerfLogDestroy(PetscEventPerfLog);
/* General functions */
PETSC_EXTERN PetscErrorCode EventPerfLogEnsureSize(PetscEventPerfLog, int);
PETSC_EXTERN PetscErrorCode EventPerfInfoClear(PetscEventPerfInfo *);
PETSC_EXTERN PetscErrorCode EventPerfInfoCopy(PetscEventPerfInfo *, PetscEventPerfInfo *);
/* Registration functions */
PETSC_EXTERN PetscErrorCode EventRegLogRegister(PetscEventRegLog, const char [], PetscClassId, PetscLogEvent *);
/* Query functions */
PETSC_EXTERN PetscErrorCode EventPerfLogSetVisible(PetscEventPerfLog, PetscLogEvent, PetscBool );
PETSC_EXTERN PetscErrorCode EventPerfLogGetVisible(PetscEventPerfLog, PetscLogEvent, PetscBool  *);
/* Activaton functions */
PETSC_EXTERN PetscErrorCode EventPerfLogActivate(PetscEventPerfLog, PetscLogEvent);
PETSC_EXTERN PetscErrorCode EventPerfLogDeactivate(PetscEventPerfLog, PetscLogEvent);
PETSC_EXTERN PetscErrorCode EventPerfLogActivateClass(PetscEventPerfLog, PetscEventRegLog, PetscClassId);
PETSC_EXTERN PetscErrorCode EventPerfLogDeactivateClass(PetscEventPerfLog, PetscEventRegLog, PetscClassId);

/* Logging functions */
PETSC_EXTERN PetscErrorCode PetscLogEventBeginDefault(PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject);
PETSC_EXTERN PetscErrorCode PetscLogEventEndDefault(PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject);
PETSC_EXTERN PetscErrorCode PetscLogEventBeginComplete(PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject);
PETSC_EXTERN PetscErrorCode PetscLogEventEndComplete(PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject);
PETSC_EXTERN PetscErrorCode PetscLogEventBeginTrace(PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject);
PETSC_EXTERN PetscErrorCode PetscLogEventEndTrace(PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject);

/* Creation and destruction functions */
PETSC_EXTERN PetscErrorCode PetscClassRegLogCreate(PetscClassRegLog *);
PETSC_EXTERN PetscErrorCode PetscClassRegLogDestroy(PetscClassRegLog);
PETSC_EXTERN PetscErrorCode ClassPerfLogCreate(PetscClassPerfLog *);
PETSC_EXTERN PetscErrorCode ClassPerfLogDestroy(PetscClassPerfLog);
PETSC_EXTERN PetscErrorCode PetscClassRegInfoDestroy(PetscClassRegInfo *);
/* General functions */
PETSC_EXTERN PetscErrorCode ClassPerfLogEnsureSize(PetscClassPerfLog, int);
PETSC_EXTERN PetscErrorCode ClassPerfInfoClear(PetscClassPerfInfo *);
/* Registration functions */
PETSC_EXTERN PetscErrorCode PetscClassRegLogRegister(PetscClassRegLog, const char [], PetscClassId);
/* Query functions */
PETSC_EXTERN PetscErrorCode PetscClassRegLogGetClass(PetscClassRegLog, PetscClassId, int *);
/* Logging functions */
PETSC_EXTERN PetscErrorCode PetscLogObjCreateDefault(PetscObject);
PETSC_EXTERN PetscErrorCode PetscLogObjDestroyDefault(PetscObject);

/* Creation and destruction functions */
PETSC_EXTERN PetscErrorCode PetscStageLogCreate(PetscStageLog *);
PETSC_EXTERN PetscErrorCode PetscStageLogDestroy(PetscStageLog);
/* Registration functions */
PETSC_EXTERN PetscErrorCode PetscStageLogRegister(PetscStageLog, const char [], int *);
/* Runtime functions */
PETSC_EXTERN PetscErrorCode PetscStageLogPush(PetscStageLog, int);
PETSC_EXTERN PetscErrorCode PetscStageLogPop(PetscStageLog);
PETSC_EXTERN PetscErrorCode PetscStageLogSetActive(PetscStageLog, int, PetscBool );
PETSC_EXTERN PetscErrorCode PetscStageLogGetActive(PetscStageLog, int, PetscBool  *);
PETSC_EXTERN PetscErrorCode PetscStageLogSetVisible(PetscStageLog, int, PetscBool );
PETSC_EXTERN PetscErrorCode PetscStageLogGetVisible(PetscStageLog, int, PetscBool  *);
PETSC_EXTERN PetscErrorCode PetscStageLogGetStage(PetscStageLog, const char [], int *);
PETSC_EXTERN PetscErrorCode PetscStageLogGetClassRegLog(PetscStageLog, PetscClassRegLog *);
PETSC_EXTERN PetscErrorCode PetscStageLogGetEventRegLog(PetscStageLog, PetscEventRegLog *);
PETSC_EXTERN PetscErrorCode PetscStageLogGetClassPerfLog(PetscStageLog, int, PetscClassPerfLog *);

PETSC_EXTERN PetscErrorCode EventRegLogGetEvent(PetscEventRegLog, const char [], PetscLogEvent *);


#endif /* PETSC_USE_LOG */
