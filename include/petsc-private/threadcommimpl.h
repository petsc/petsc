
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

typedef struct _p_PetscThreadCommJobCtx *PetscThreadCommJobCtx;
struct  _p_PetscThreadCommJobCtx{
  PetscInt          nargs;                         /* Number of arguments for the kernel */
  PetscThreadKernel pfunc;                         /* Kernel function */
  void              *args[PETSC_KERNEL_NARGS_MAX]; /* Array of void* to hold the arguments */
};

typedef struct _PetscThreadCommOps *PetscThreadCommOps;
struct _PetscThreadCommOps {
  PetscErrorCode (*destroy)(PetscThreadComm);
  PetscErrorCode (*runkernel)(MPI_Comm,PetscThreadCommJobCtx);
  PetscErrorCode (*view)(PetscThreadComm,PetscViewer);
};

struct _p_PetscThreadComm{
  PetscInt              nworkThreads; /* Number of threads in the pool */
  PetscInt              *affinities;  /* Thread affinity */
  PetscThreadCommOps    ops;          /* Operations table */ 
  void                  *data;        /* implementation specific data */
  PetscThreadCommJobCtx jobctx;       /* Job context */
  char                  type[256];    /* Thread model type */
};

extern PetscErrorCode PetscCommGetThreadComm(MPI_Comm,PetscThreadComm*);

#endif
