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
extern Action    *petsc_actions;
extern Object    *petsc_objects;
extern PetscBool  petsc_logActions;
extern PetscBool  petsc_logObjects;
extern int        petsc_numActions, petsc_maxActions;
extern int        petsc_numObjects, petsc_maxObjects;
extern int        petsc_numObjectsDestroyed;

extern FILE          *petsc_tracefile;
extern int            petsc_tracelevel;
extern const char    *petsc_traceblanks;
extern char           petsc_tracespace[128];
extern PetscLogDouble petsc_tracetime;

#ifdef PETSC_USE_LOG

/* Runtime functions */
extern PetscErrorCode PetscStageLogGetClassRegLog(PetscStageLog, PetscClassRegLog *);
extern PetscErrorCode PetscStageLogGetEventRegLog(PetscStageLog, PetscEventRegLog *);
extern PetscErrorCode PetscStageLogGetClassPerfLog(PetscStageLog, int, PetscClassPerfLog *);


/* Creation and destruction functions */
extern PetscErrorCode EventRegLogCreate(PetscEventRegLog *);
extern PetscErrorCode EventRegLogDestroy(PetscEventRegLog);
extern PetscErrorCode EventPerfLogCreate(PetscEventPerfLog *);
extern PetscErrorCode EventPerfLogDestroy(PetscEventPerfLog);
/* General functions */
extern PetscErrorCode EventPerfLogEnsureSize(PetscEventPerfLog, int);
extern PetscErrorCode EventPerfInfoClear(PetscEventPerfInfo *);
extern PetscErrorCode EventPerfInfoCopy(PetscEventPerfInfo *, PetscEventPerfInfo *);
/* Registration functions */
extern PetscErrorCode EventRegLogRegister(PetscEventRegLog, const char [], PetscClassId, PetscLogEvent *);
/* Query functions */
extern PetscErrorCode EventPerfLogSetVisible(PetscEventPerfLog, PetscLogEvent, PetscBool );
extern PetscErrorCode EventPerfLogGetVisible(PetscEventPerfLog, PetscLogEvent, PetscBool  *);
/* Activaton functions */
extern PetscErrorCode EventPerfLogActivate(PetscEventPerfLog, PetscLogEvent);
extern PetscErrorCode EventPerfLogDeactivate(PetscEventPerfLog, PetscLogEvent);
extern PetscErrorCode EventPerfLogActivateClass(PetscEventPerfLog, PetscEventRegLog, PetscClassId);
extern PetscErrorCode EventPerfLogDeactivateClass(PetscEventPerfLog, PetscEventRegLog, PetscClassId);

/* Logging functions */
extern PetscErrorCode PetscLogEventBeginDefault(PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject);
extern PetscErrorCode PetscLogEventEndDefault(PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject);
extern PetscErrorCode PetscLogEventBeginComplete(PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject);
extern PetscErrorCode PetscLogEventEndComplete(PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject);
extern PetscErrorCode PetscLogEventBeginTrace(PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject);
extern PetscErrorCode PetscLogEventEndTrace(PetscLogEvent, int, PetscObject, PetscObject, PetscObject, PetscObject);

/* Creation and destruction functions */
extern PetscErrorCode PetscClassRegLogCreate(PetscClassRegLog *);
extern PetscErrorCode PetscClassRegLogDestroy(PetscClassRegLog);
extern PetscErrorCode ClassPerfLogCreate(PetscClassPerfLog *);
extern PetscErrorCode ClassPerfLogDestroy(PetscClassPerfLog);
extern PetscErrorCode PetscClassRegInfoDestroy(PetscClassRegInfo *);
/* General functions */
extern PetscErrorCode ClassPerfLogEnsureSize(PetscClassPerfLog, int);
extern PetscErrorCode ClassPerfInfoClear(PetscClassPerfInfo *);
/* Registration functions */
extern PetscErrorCode PetscClassRegLogRegister(PetscClassRegLog, const char [], PetscClassId);
/* Query functions */
extern PetscErrorCode PetscClassRegLogGetClass(PetscClassRegLog, PetscClassId, int *);
/* Logging functions */
extern PetscErrorCode PetscLogObjCreateDefault(PetscObject);
extern PetscErrorCode PetscLogObjDestroyDefault(PetscObject);

/* Creation and destruction functions */
extern PetscErrorCode  PetscStageLogCreate(PetscStageLog *);
extern PetscErrorCode  PetscStageLogDestroy(PetscStageLog);
/* Registration functions */
extern PetscErrorCode  PetscStageLogRegister(PetscStageLog, const char [], int *);
/* Runtime functions */
extern PetscErrorCode  PetscStageLogPush(PetscStageLog, int);
extern PetscErrorCode  PetscStageLogPop(PetscStageLog);
extern PetscErrorCode  PetscStageLogSetActive(PetscStageLog, int, PetscBool );
extern PetscErrorCode  PetscStageLogGetActive(PetscStageLog, int, PetscBool  *);
extern PetscErrorCode  PetscStageLogSetVisible(PetscStageLog, int, PetscBool );
extern PetscErrorCode  PetscStageLogGetVisible(PetscStageLog, int, PetscBool  *);
extern PetscErrorCode  PetscStageLogGetStage(PetscStageLog, const char [], int *);
extern PetscErrorCode  PetscStageLogGetClassRegLog(PetscStageLog, PetscClassRegLog *);
extern PetscErrorCode  PetscStageLogGetEventRegLog(PetscStageLog, PetscEventRegLog *);
extern PetscErrorCode  PetscStageLogGetClassPerfLog(PetscStageLog, int, PetscClassPerfLog *);

extern PetscErrorCode  EventRegLogGetEvent(PetscEventRegLog, const char [], PetscLogEvent *);


#endif /* PETSC_USE_LOG */
