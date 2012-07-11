
#ifndef __THREADCOMMIMPL_H
#define __THREADCOMMIMPL_H

#include <petscthreadcomm.h>

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

PETSC_EXTERN PetscMPIInt Petsc_ThreadComm_keyval;

/* Max. number of arguments for kernel */
#define PETSC_KERNEL_NARGS_MAX 10

/* Max. number of kernels */
#define PETSC_KERNELS_MAX 32

/* Reduction status of threads */
#define THREADCOMM_THREAD_WAITING_FOR_NEWRED 0
#define THREADCOMM_THREAD_POSTED_LOCALRED    1
/* Status of the reduction */
#define THREADCOMM_REDUCTION_NONE           -1
#define THREADCOMM_REDUCTION_NEW             0
#define THREADCOMM_REDUCTION_COMPLETE        1

/* Job status for threads */
#define THREAD_JOB_NONE       -1
#define THREAD_JOB_POSTED      1
#define THREAD_JOB_RECIEVED    2
#define THREAD_JOB_COMPLETED   0

#define PetscReadOnce(type,val) (*(volatile type *)&val)

/* Definitions for memory barriers and cpu_relax taken from the linux kernel source code */
#if defined(__x86_64__)
#define PetscMemoryBarrier()      asm volatile("mfence":::"memory")
#define PetscReadMemoryBarrier()  asm volatile("lfence":::"memory")
#define PetscWriteMemoryBarrier() asm volatile("sfence":::"memory")
#define PetscCPURelax()           asm volatile("rep; nop" ::: "memory")
#elif defined(__powerpc__)
#define PetscMemoryBarrier()      __asm__ __volatile__ ("sync":::"memory")
#define PetscReadMemoryBarrier()  __asm__ __volatile__ ("sync":::"memory")
#define PetscWriteMemoryBarrier() __asm__ __volatile__ ("sync":::"memory")
#define PetscCPURelax()           __asm__ __volatile__ ("" ::: "memory")
#else
#define PetscMemoryBarrier()
#define PetscReadMemoryBarrier()
#define PetscWriteMemoryBarrier()
#endif

struct _p_PetscThreadCommRedCtx{
  PetscThreadComm               tcomm;          /* The associated threadcomm */
  PetscInt                      red_status;     /* Reduction status */
  PetscInt                      nworkThreads;   /* Number of threads doing the reduction */
  PetscInt                      *thread_status; /* Reduction status of each thread */
  void                          *local_red;     /* Array to hold local reduction contribution from each thread */
  PetscThreadCommReductionOp    op;             /* The reduction operation */
  PetscDataType                 type;           /* The reduction data type */
};

typedef struct _p_PetscThreadCommJobCtx *PetscThreadCommJobCtx;
struct  _p_PetscThreadCommJobCtx{
  PetscThreadComm   tcomm;                         /* The thread communicator */
  PetscInt          nargs;                         /* Number of arguments for the kernel */
  PetscThreadKernel pfunc;                         /* Kernel function */
  void              *args[PETSC_KERNEL_NARGS_MAX]; /* Array of void* to hold the arguments */
  PetscScalar       scalars[3];                    /* Array to hold three scalar values */
  PetscInt          *job_status;                   /* Thread job status */
};

/* Structure to manage job queue */
typedef struct _p_PetscThreadCommJobQueue *PetscThreadCommJobQueue;
struct _p_PetscThreadCommJobQueue{
  PetscInt ctr;                                         /* job counter */
  PetscInt kernel_ctr;                                  /* kernel counter .. need this otherwise race conditions are unavoidable */
  PetscThreadCommJobCtx jobs[PETSC_KERNELS_MAX];        /* queue of jobs */
};

extern PetscThreadCommJobQueue PetscJobQueue;

typedef struct _PetscThreadCommOps *PetscThreadCommOps;
struct _PetscThreadCommOps {
  PetscErrorCode (*destroy)(PetscThreadComm);
  PetscErrorCode (*runkernel)(MPI_Comm,PetscThreadCommJobCtx);
  PetscErrorCode (*view)(PetscThreadComm,PetscViewer);
  PetscErrorCode (*barrier)(PetscThreadComm);
  PetscInt       (*getrank)(void);
};

struct _p_PetscThreadComm{
  PetscInt                refct;
  PetscInt                nworkThreads; /* Number of threads in the pool */
  PetscInt                *affinities;  /* Thread affinity */
  PetscThreadCommOps      ops;          /* Operations table */ 
  void                    *data;        /* implementation specific data */
  char                    type[256];    /* Thread model type */
  PetscInt                leader;       /* Rank of the leader thread. This thread manages
                                           the synchronization for collective operatons like reductions.
					*/
  PetscThreadCommRedCtx   red;          /* Reduction context */
  PetscInt                job_ctr;      /* which job is this threadcomm running in the job queue */
};

/* register thread communicator models */
PETSC_EXTERN PetscErrorCode PetscThreadCommRegister(const char[],const char[],const char[],PetscErrorCode(*)(PetscThreadComm));
PETSC_EXTERN PetscErrorCode PetscThreadCommRegisterAll(const char path[]);
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define PetscThreadCommRegisterDynamic(a,b,c,d) PetscThreadCommRegister(a,b,c,0)
#else
#define PetscThreadCommRegisterDynamic(a,b,c,d) PetscThreadCommRegister(a,b,c,d)
#endif

PETSC_EXTERN PetscErrorCode PetscThreadCommReductionCreate(PetscThreadComm,PetscThreadCommRedCtx*);
PETSC_EXTERN PetscErrorCode PetscThreadCommReductionDestroy(PetscThreadCommRedCtx);
PETSC_EXTERN PetscErrorCode PetscRunKernel(PetscInt,PetscInt,PetscThreadCommJobCtx);

PETSC_EXTERN PetscLogEvent ThreadComm_Init, ThreadComm_RunKernel, ThreadComm_Barrier;
#endif
