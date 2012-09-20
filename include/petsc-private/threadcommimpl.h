
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

#if defined(PETSC_MEMORY_BARRIER)
#define PetscMemoryBarrier() do {PETSC_MEMORY_BARRIER();} while(0)
#else
#define PetscMemoryBarrier()
#endif
#if defined(PETSC_READ_MEMORY_BARRIER)
#define PetscReadMemoryBarrier() do {PETSC_READ_MEMORY_BARRIER();} while(0)
#else
#define PetscReadMemoryBarrier()
#endif
#if defined(PETSC_WRITE_MEMORY_BARRIER)
#define PetscWriteMemoryBarrier() do {PETSC_WRITE_MEMORY_BARRIER();} while(0)
#else
#define PetscWriteMemoryBarrier()
#endif

typedef struct _p_PetscThreadCommRedCtx *PetscThreadCommRedCtx;
struct _p_PetscThreadCommRedCtx{
  PetscThreadComm               tcomm;          /* The associated threadcomm */
  PetscInt                      red_status;     /* Reduction status */
  PetscInt                      *thread_status; /* Reduction status of each thread */
  void                          *local_red;     /* Array to hold local reduction contribution from each thread */
  PetscThreadCommReductionOp    op;             /* The reduction operation */
  PetscDataType                 type;           /* The reduction data type */
};

struct _p_PetscThreadCommReduction{
  PetscInt              nreds;                              /* Number of reductions in operation */
  PetscThreadCommRedCtx redctx[PETSC_REDUCTIONS_MAX];       /* Reduction objects */
  PetscInt               ctr;                               /* Global Reduction counter */
  PetscInt              *thread_ctr;                        /* Reduction counter for each thread */
};

typedef struct _p_PetscThreadCommJobCtx* PetscThreadCommJobCtx;
struct  _p_PetscThreadCommJobCtx{
  PetscThreadComm   tcomm;                         /* The thread communicator */
  PetscInt          nargs;                         /* Number of arguments for the kernel */
  PetscThreadKernel pfunc;                         /* Kernel function */
  void              *args[PETSC_KERNEL_NARGS_MAX]; /* Array of void* to hold the arguments */
  PetscScalar       scalars[3];                    /* Array to hold three scalar values */
  PetscInt          ints[3];                       /* Array to hold three integer values */
  PetscInt          *job_status;                   /* Thread job status */
};

/* Structure to manage job queue */
typedef struct _p_PetscThreadCommJobQueue* PetscThreadCommJobQueue;
struct _p_PetscThreadCommJobQueue{
  PetscInt ctr;                                         /* job counter */
  PetscInt kernel_ctr;                                  /* kernel counter .. need this otherwise race conditions are unavoidable */
  PetscThreadCommJobCtx jobs[PETSC_KERNELS_MAX];        /* queue of jobs */
};

extern PetscThreadCommJobQueue PetscJobQueue;

typedef struct _PetscThreadCommOps* PetscThreadCommOps;
struct _PetscThreadCommOps {
  PetscErrorCode (*destroy)(PetscThreadComm);
  PetscErrorCode (*runkernel)(MPI_Comm,PetscThreadCommJobCtx);
  PetscErrorCode (*view)(PetscThreadComm,PetscViewer);
  PetscErrorCode (*barrier)(PetscThreadComm);
  PetscErrorCode (*getrank)(PetscInt*);
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
  PetscThreadCommReduction red;          /* Reduction context */
  PetscInt                job_ctr;      /* which job is this threadcomm running in the job queue */
  PetscBool               isnothread;   /* No threading model used */
};

/* register thread communicator models */
PETSC_EXTERN PetscErrorCode PetscThreadCommRegister(const char[],const char[],const char[],PetscErrorCode(*)(PetscThreadComm));
PETSC_EXTERN PetscErrorCode PetscThreadCommRegisterAll(const char path[]);
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define PetscThreadCommRegisterDynamic(a,b,c,d) PetscThreadCommRegister(a,b,c,0)
#else
#define PetscThreadCommRegisterDynamic(a,b,c,d) PetscThreadCommRegister(a,b,c,d)
#endif

#undef __FUNCT__
#define __FUNCT__
PETSC_STATIC_INLINE PetscErrorCode PetscRunKernel(PetscInt trank,PetscInt nargs,PetscThreadCommJobCtx job)
{
  switch(nargs) {
  case 0:
    (*job->pfunc)(trank);
    break;
  case 1:
    (*job->pfunc)(trank,job->args[0]);
    break;
  case 2:
    (*job->pfunc)(trank,job->args[0],job->args[1]);
    break;
  case 3:
    (*job->pfunc)(trank,job->args[0],job->args[1],job->args[2]);
    break;
  case 4:
    (*job->pfunc)(trank,job->args[0],job->args[1],job->args[2],job->args[3]);
    break;
  case 5:
    (*job->pfunc)(trank,job->args[0],job->args[1],job->args[2],job->args[3],job->args[4]);
    break;
  case 6:
    (*job->pfunc)(trank,job->args[0],job->args[1],job->args[2],job->args[3],job->args[4],job->args[5]);
    break;
  case 7:
    (*job->pfunc)(trank,job->args[0],job->args[1],job->args[2],job->args[3],job->args[4],job->args[5],job->args[6]);
    break;
  case 8:
    (*job->pfunc)(trank,job->args[0],job->args[1],job->args[2],job->args[3],job->args[4],job->args[5],job->args[6],job->args[7]);
    break;
  case 9:
    (*job->pfunc)(trank,job->args[0],job->args[1],job->args[2],job->args[3],job->args[4],job->args[5],job->args[6],job->args[7],job->args[8]);
    break;
  case 10:
    (*job->pfunc)(trank,job->args[0],job->args[1],job->args[2],job->args[3],job->args[4],job->args[5],job->args[6],job->args[7],job->args[8],job->args[9]);
    break;
  }
  return 0;
}

PETSC_EXTERN PetscErrorCode PetscThreadCommReductionCreate(PetscThreadComm,PetscThreadCommReduction*);
PETSC_EXTERN PetscErrorCode PetscThreadCommReductionDestroy(PetscThreadCommReduction);

PETSC_EXTERN PetscLogEvent ThreadComm_RunKernel, ThreadComm_Barrier;
#endif
