#ifndef PETSCLOGTYPES_H
#define PETSCLOGTYPES_H
#include <petscsystypes.h>

/* SUBMANSEC = Profiling */

/*S
  PetscEventPerfInfo - statistics on how many times the event is used, how much time it takes, etc.

  Level: advanced

  Note:
  This is the data structure that describes profiling statsitics collected for an event from
  the default log handler (`PetscLogDefaultBegin()`) using `PetscLogEventGetPerfInfo()`.

.seealso(): [](ch_profiling)
S*/
typedef struct {
  int            id;                  /* The integer identifying this event / stage */
  PetscBool      active;              /* Deprecated */
  PetscBool      visible;             /* The flag to print info in summary */
  int            depth;               /* The nesting depth of the event call */
  int            count;               /* The number of times this event was executed */
  PetscLogDouble flops;               /* The flops used in this event */
  PetscLogDouble flops2;              /* The square of flops used in this event */
  PetscLogDouble flopsTmp;            /* The accumulator for flops used in this event */
  PetscLogDouble time;                /* The time taken for this event */
  PetscLogDouble time2;               /* The square of time taken for this event */
  PetscLogDouble timeTmp;             /* The accumulator for time taken for this event */
  PetscLogDouble syncTime;            /* The synchronization barrier time */
  PetscLogDouble dof[8];              /* The number of degrees of freedom associated with this event */
  PetscLogDouble errors[8];           /* The errors (user-defined) associated with this event */
  PetscLogDouble numMessages;         /* The number of messages in this event */
  PetscLogDouble messageLength;       /* The total message lengths in this event */
  PetscLogDouble numReductions;       /* The number of reductions in this event */
  PetscLogDouble memIncrease;         /* How much the resident memory has increased in this event */
  PetscLogDouble mallocIncrease;      /* How much the maximum malloced space has increased in this event */
  PetscLogDouble mallocSpace;         /* How much the space was malloced and kept during this event */
  PetscLogDouble mallocIncreaseEvent; /* Maximum of the high water mark with in event minus memory available at the end of the event */
#if defined(PETSC_HAVE_DEVICE)
  PetscLogDouble CpuToGpuCount; /* The total number of CPU to GPU copies */
  PetscLogDouble GpuToCpuCount; /* The total number of GPU to CPU copies */
  PetscLogDouble CpuToGpuSize;  /* The total size of CPU to GPU copies */
  PetscLogDouble GpuToCpuSize;  /* The total size of GPU to CPU copies */
  PetscLogDouble GpuFlops;      /* The flops done on a GPU in this event */
  PetscLogDouble GpuTime;       /* The time spent on a GPU in this event */
#endif
} PetscEventPerfInfo;

typedef struct _n_PetscIntStack *PetscIntStack;

/*MC
    PetscLogEvent - id used to identify PETSc or user events which timed portions (blocks of executable)
     code.

    Level: intermediate

.seealso: [](ch_profiling), `PetscLogEventRegister()`, `PetscLogEventBegin()`, `PetscLogEventEnd()`, `PetscLogStage`
M*/
typedef int PetscLogEvent;

/*MC
    PetscLogStage - id used to identify user stages (phases, sections) of runs - for logging

    Level: intermediate

.seealso: [](ch_profiling), `PetscLogStageRegister()`, `PetscLogStagePush()`, `PetscLogStagePop()`, `PetscLogEvent`
M*/
typedef int PetscLogStage;

#endif
