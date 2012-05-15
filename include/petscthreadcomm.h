
#if !defined(__PETSCTHREADCOMM_H)
#define __PETSCTHREADCOMM_H
#include "petscsys.h"
PETSC_EXTERN_CXX_BEGIN

/* Function pointer cast for the kernel function */
typedef PetscErrorCode (*PetscThreadKernel)(PetscInt,...);

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

extern PetscFList PetscThreadCommList;

typedef enum {THREADCOMM_SUM,THREADCOMM_PROD} PetscThreadCommReductionOp;
extern const char *const PetscThreadCommReductionOps[];

extern PetscErrorCode PetscGetNCores(PetscInt*);
extern PetscErrorCode PetscCommGetThreadComm(MPI_Comm,PetscThreadComm*);
extern PetscErrorCode PetscThreadCommInitializePackage(const char *path);
extern PetscErrorCode PetscThreadCommFinalizePackage(void);
extern PetscErrorCode PetscThreadCommInitialize(void);
extern PetscErrorCode PetscThreadCommGetNThreads(MPI_Comm,PetscInt*);
extern PetscErrorCode PetscThreadCommGetAffinities(MPI_Comm,PetscInt[]);
extern PetscErrorCode PetscThreadCommView(MPI_Comm,PetscViewer);
extern PetscErrorCode PetscThreadCommGetScalars(MPI_Comm,PetscScalar**,PetscScalar**,PetscScalar**);
extern PetscErrorCode PetscThreadCommRunKernel(MPI_Comm,PetscErrorCode (*)(PetscInt,...),PetscInt,...);
extern PetscErrorCode PetscThreadCommBarrier(MPI_Comm);
extern PetscErrorCode PetscThreadCommGetOwnershipRanges(MPI_Comm,PetscInt,PetscInt*[]);
extern PetscErrorCode PetscThreadCommRegisterDestroy(void);
extern PetscInt PetscThreadCommGetRank(PetscThreadComm);

/* Reduction operations */
extern PetscErrorCode PetscThreadReductionKernelBegin(PetscInt,PetscThreadCommRedCtx,void*);
extern PetscErrorCode PetscThreadReductionKernelEnd(PetscInt,PetscThreadCommRedCtx,void*);
extern PetscErrorCode PetscThreadReductionBegin(MPI_Comm,PetscThreadCommReductionOp,PetscDataType,PetscThreadCommRedCtx*);
extern PetscErrorCode PetscThreadReductionEnd(PetscThreadCommRedCtx,void*);



PETSC_EXTERN_CXX_END
#endif
