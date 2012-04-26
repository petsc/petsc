
#if !defined(__PETSCTHREADCOMM_H)
#define __PETSCTHREADCOMM_H
#include "petscsys.h"
PETSC_EXTERN_CXX_BEGIN

/* This key should be in petsc-private/threadcommimpl.h */
extern PetscMPIInt Petsc_ThreadComm_keyval;

/* Function pointer cast for the kernel function */
typedef PetscErrorCode (*PetscThreadKernel)(PetscInt,...);

/* Max. number of arguments for kernel */
#define PETSC_KERNEL_NARGS_MAX 10

/* Max. number of kernels */
#define PETSC_KERNELS_MAX 10

/*
  ThreadComm - Abstract object that manages all thread communication models

  Level: developer

  Concepts: threads

.seealso: PetscThreadCommCreate(), PetscThreadCommDestroy
S*/
typedef struct _p_PetscThreadComm* PetscThreadComm;

extern PetscInt N_CORES; /* Number of available cores */

#define PetscThreadCommType char*
#define PTHREAD             "pthread"

extern PetscFList PetscThreadCommList;

extern PetscErrorCode PetscThreadComm_Init();
extern PetscErrorCode PetscThreadCommGetNThreads(MPI_Comm,PetscInt*);
extern PetscErrorCode PetscThreadCommGetAffinities(MPI_Comm,PetscInt[]);
extern PetscErrorCode PetscThreadCommView(MPI_Comm,PetscViewer);
extern PetscErrorCode PetscThreadCommRunKernel(MPI_Comm,PetscErrorCode (*)(PetscInt,...),PetscInt,...);
extern PetscErrorCode PetscThreadCommBarrier(MPI_Comm);

/* register thread communicator models */
extern PetscErrorCode PetscThreadCommRegister(const char[],const char[],const char[],PetscErrorCode(*)(PetscThreadComm));
extern PetscErrorCode PetscThreadCommRegisterAll(const char path[]);
extern PetscErrorCode PetscThreadCommRegisterDestroy(void);

#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define PetscThreadCommRegisterDynamic(a,b,c,d) PetscThreadCommRegister(a,b,c,0)
#else
#define PetscThreadCommRegisterDynamic(a,b,c,d) PetscThreadCommRegister(a,b,c,d)
#endif

PETSC_EXTERN_CXX_END
#endif
