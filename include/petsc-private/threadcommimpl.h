
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
/* x86_64 */
#if defined(__x86_64__) || defined(__x86_64)
#if defined (__GNUC__)
#define PetscMemoryBarrier()      asm volatile("mfence":::"memory")
#define PetscReadMemoryBarrier()  asm volatile("lfence":::"memory")
#define PetscWriteMemoryBarrier() asm volatile("sfence":::"memory")
#define PetscCPURelax()           asm volatile("rep; nop" ::: "memory")
#elif defined(__INTEL_COMPILER)
/* Intel ECC compiler doesn't support gcc specific asm stmts.
   It uses intrinsics to do the equivalent things
*/
#define PetscMemoryBarrier()      __memory_barrier()
#define PetscReadMemoryBarrier()  __memory_barrier()
#define PetscWriteMemoryBarrier() __memory_barrier()
#define PetscCPURelax()
#else
#define PetscMemoryBarrier()
#define PetscReadMemoryBarrier()
#define PetscWriteMemoryBarrier()
#define PetscCPURelax()
#endif
/* x86_32 */
#elif defined(__i386__) || defined(i386)
#if defined(__GNUC__)
/* alternative assembly primitive: */
#ifndef ALTERNATIVE
#define ALTERNATIVE(oldinstr, newinstr, feature)			\
									\
      "661:\n\t" oldinstr "\n662:\n"					\
      ".section .altinstructions,\"a\"\n"				\
      "	 .long 661b - .\n"			/* label           */	\
      "	 .long 663f - .\n"			/* new instruction */	\
      "	 .word " __stringify(feature) "\n"	/* feature bit     */	\
      "	 .byte 662b-661b\n"			/* sourcelen       */	\
      "	 .byte 664f-663f\n"			/* replacementlen  */	\
      ".previous\n"							\
      ".section .discard,\"aw\",@progbits\n"				\
      "	 .byte 0xff + (664f-663f) - (662b-661b)\n" /* rlen <= slen */	\
      ".previous\n"							\
      ".section .altinstr_replacement, \"ax\"\n"			\
      "663:\n\t" newinstr "\n664:\n"		/* replacement     */	\
      ".previous"
#endif
/*
 * Alternative instructions for different CPU types or capabilities.
 *
 * This allows to use optimized instructions even on generic binary
 * kernels.
 *
 * length of oldinstr must be longer or equal the length of newinstr
 * It can be padded with nops as needed.
 *
 * For non barrier like inlines please define new variants
 * without volatile and memory clobber.
 */
#ifndef alternative
#define alternative(oldinstr, newinstr, feature)			\
	asm volatile (ALTERNATIVE(oldinstr, newinstr, feature) : : : "memory")
#endif
#ifndef X86_FEATURE_XMM
#define X86_FEATURE_XMM		(0*32+25) /* "sse" */
#endif
#ifndef X86_FEATURE_XMM2
#define X86_FEATURE_XMM2	(0*32+26) /* "sse2" */
#endif
#define PetscMemoryBarrier() alternative("lock; addl $0,0(%%esp)", "mfence", X86_FEATURE_XMM2)
#define PetscReadMemoryBarrier() alternative("lock; addl $0,0(%%esp)", "lfence", X86_FEATURE_XMM2)
#define PetscWriteMemoryBarrier() alternative("lock; addl $0,0(%%esp)", "sfence", X86_FEATURE_XMM)
#define PetscCPURelax()           asm volatile("rep; nop" ::: "memory")
#elif defined(__INTEL_COMPILER)
/* Intel ECC compiler doesn't support gcc specific asm stmts.
   It uses intrinsics to do the equivalent things
*/
#define PetscMemoryBarrier()      __memory_barrier()
#define PetscReadMemoryBarrier()  __memory_barrier()
#define PetscWriteMemoryBarrier() __memory_barrier()
#define PetscCPURelax()
#else
#define PetscMemoryBarrier()
#define PetscReadMemoryBarrier()
#define PetscWriteMemoryBarrier()
#define PetscCPURelax()
#endif
#elif defined(__powerpc__)
#define PetscMemoryBarrier()      __asm__ __volatile__ ("sync":::"memory")
#define PetscReadMemoryBarrier()  __asm__ __volatile__ ("sync":::"memory")
#define PetscWriteMemoryBarrier() __asm__ __volatile__ ("sync":::"memory")
#define PetscCPURelax()           __asm__ __volatile__ ("" ::: "memory")
#else
#define PetscMemoryBarrier()
#define PetscReadMemoryBarrier()
#define PetscWriteMemoryBarrier()
#define PetscCPURelax()
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

PETSC_EXTERN PetscLogEvent ThreadComm_RunKernel, ThreadComm_Barrier;
#endif
