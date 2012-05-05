
#ifndef __THREADCOMMIMPL_H
#define __THREADCOMMIMPL_H

#include <petscthreadcomm.h>

#if defined(PETSC_HAVE_SCHED_H)
#ifndef __USE_GNU
#define __USE_GNU
#endif
#include <sched.h>
#endif
#if defined(PETSC_HAVE_SYS_SYSINFO_H)
#include <sys/sysinfo.h>
#endif
#if defined(PETSC_HAVE_UNISTD_H)
#include <unistd.h>
#endif
#if defined(PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#if defined(PETSC_HAVE_SYS_SYSCTL_H)
#include <sys/sysctl.h>
#endif
#if defined(PETSC_HAVE_WINDOWS_H)
#include <windows.h>
#endif

extern PetscMPIInt Petsc_ThreadComm_keyval;

/* Max. number of arguments for kernel */
#define PETSC_KERNEL_NARGS_MAX 10

/* Max. number of kernels */
#define PETSC_KERNELS_MAX 10

/* Status of threads */
#define THREADCOMM_THREAD_WAITING_FOR_NEWRED 0
#define THREADCOMM_THREAD_POSTED_LOCALRED    1
/* Status of the reduction */
#define THREADCOMM_REDUCTION_NONE           -1
#define THREADCOMM_REDUCTION_NEW             0
#define THREADCOMM_REDUCTION_COMPLETE        1

#define PetscReadOnce(type,val) (*(volatile type *)&val)

typedef struct _p_PetscThreadCommRedCtx *PetscThreadCommRedCtx;
struct _p_PetscThreadCommRedCtx{
  PetscInt                      red_status;     /* Reduction status */
  PetscInt                      nworkThreads;   /* Number of threads doing the reduction */
  PetscInt                      *thread_status; /* Reduction status of each thread */
  void                          *local_red;     /* Array to hold local reduction contribution from each thread */
};

typedef struct _p_PetscThreadCommJobCtx *PetscThreadCommJobCtx;
struct  _p_PetscThreadCommJobCtx{
  PetscThreadComm   tcomm;                         /* The thread communicator */
  PetscInt          nargs;                         /* Number of arguments for the kernel */
  PetscThreadKernel pfunc;                         /* Kernel function */
  void              *args[PETSC_KERNEL_NARGS_MAX]; /* Array of void* to hold the arguments */
};

/* Structure to manage job queue */
typedef struct _p_PetscThreadCommJobQueue *PetscThreadCommJobQueue;
struct _p_PetscThreadCommJobQueue{
  PetscInt ctr;                                  /* job counter */
  PetscThreadCommJobCtx jobs[PETSC_KERNELS_MAX]; /* queue of jobs */
};

typedef struct _PetscThreadCommOps *PetscThreadCommOps;
struct _PetscThreadCommOps {
  PetscErrorCode (*destroy)(PetscThreadComm);
  PetscErrorCode (*runkernel)(MPI_Comm,PetscThreadCommJobCtx);
  PetscErrorCode (*view)(PetscThreadComm,PetscViewer);
  PetscErrorCode (*barrier)(PetscThreadComm);
  PetscInt       (*getrank)(void);
};

struct _p_PetscThreadComm{
  PetscInt                nworkThreads; /* Number of threads in the pool */
  PetscInt                *affinities;  /* Thread affinity */
  PetscThreadCommOps      ops;          /* Operations table */ 
  void                    *data;        /* implementation specific data */
  PetscThreadCommJobQueue jobqueue;     /* Job queue */
  char                    type[256];    /* Thread model type */
  PetscInt                leader;       /* Rank of the leader thread. This thread manages
                                           the synchronization for collective operatons like reductions.
					*/
  PetscThreadCommRedCtx   red;      /* Reduction context */
  PetscInt                refcount; /* Reference count */
};

/* register thread communicator models */
extern PetscErrorCode PetscThreadCommRegister(const char[],const char[],const char[],PetscErrorCode(*)(PetscThreadComm));
extern PetscErrorCode PetscThreadCommRegisterAll(const char path[]);
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define PetscThreadCommRegisterDynamic(a,b,c,d) PetscThreadCommRegister(a,b,c,0)
#else
#define PetscThreadCommRegisterDynamic(a,b,c,d) PetscThreadCommRegister(a,b,c,d)
#endif

extern PetscErrorCode PetscRunKernel(PetscInt,PetscInt,PetscThreadCommJobCtx);
#endif
