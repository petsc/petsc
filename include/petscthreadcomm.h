
#if !defined(__PETSCTHREADCOMM_H)
#define __PETSCTHREADCOMM_H
#include "petscsys.h"

/* Function pointer cast for the kernel function */
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode (*PetscThreadKernel)(PetscInt,...);

/*
  PetscThreadComm - Abstract object that manages all thread communication models

  Level: developer

  Concepts: threads

.seealso: PetscThreadCommCreate(), PetscThreadCommDestroy
*/
typedef struct _p_PetscThreadComm* PetscThreadComm;

/*
   PetscThreadCommRedCtx - Context used for doing threaded reductions

   Level: developer
*/
typedef struct _p_PetscThreadCommRedCtx *PetscThreadCommRedCtx;

#define PetscThreadCommType char*
#define PTHREAD             "pthread"
#define NOTHREAD            "nothread"
#define OPENMP              "openmp"

PETSC_EXTERN PetscFList PetscThreadCommList;

typedef enum {THREADCOMM_SUM,THREADCOMM_PROD} PetscThreadCommReductionOp;
PETSC_EXTERN const char *const PetscThreadCommReductionOps[];

PETSC_EXTERN PetscErrorCode PetscGetNCores(PetscInt*);
PETSC_EXTERN PetscErrorCode PetscCommGetThreadComm(MPI_Comm,PetscThreadComm*);
PETSC_EXTERN PetscErrorCode PetscThreadCommInitializePackage(const char *path);
PETSC_EXTERN PetscErrorCode PetscThreadCommFinalizePackage(void);
PETSC_EXTERN PetscErrorCode PetscThreadCommInitialize(void);
PETSC_EXTERN PetscErrorCode PetscThreadCommGetNThreads(MPI_Comm,PetscInt*);
PETSC_EXTERN PetscErrorCode PetscThreadCommGetAffinities(MPI_Comm,PetscInt[]);
PETSC_EXTERN PetscErrorCode PetscThreadCommView(MPI_Comm,PetscViewer);
PETSC_EXTERN PetscErrorCode PetscThreadCommGetScalars(MPI_Comm,PetscScalar**,PetscScalar**,PetscScalar**);
PETSC_EXTERN PetscErrorCode PetscThreadCommRunKernel(MPI_Comm,PetscErrorCode (*)(PetscInt,...),PetscInt,...);
PETSC_EXTERN PetscErrorCode PetscThreadCommBarrier(MPI_Comm);
PETSC_EXTERN PetscErrorCode PetscThreadCommGetOwnershipRanges(MPI_Comm,PetscInt,PetscInt*[]);
PETSC_EXTERN PetscErrorCode PetscThreadCommRegisterDestroy(void);
PETSC_EXTERN PetscInt  PetscThreadCommGetRank(PetscThreadComm);

/* Reduction operations */
PETSC_EXTERN PetscErrorCode PetscThreadReductionKernelBegin(PetscInt,PetscThreadCommRedCtx,void*);
PETSC_EXTERN PetscErrorCode PetscThreadReductionKernelEnd(PetscInt,PetscThreadCommRedCtx,void*);
PETSC_EXTERN PetscErrorCode PetscThreadReductionBegin(MPI_Comm,PetscThreadCommReductionOp,PetscDataType,PetscThreadCommRedCtx*);
PETSC_EXTERN PetscErrorCode PetscThreadReductionEnd(PetscThreadCommRedCtx,void*);



#endif
