
#if !defined(__PETSCTHREADCOMM_H)
#define __PETSCTHREADCOMM_H
#include "petscsys.h"
PETSC_EXTERN_CXX_BEGIN

/* Function pointer cast for the kernel function */
typedef PetscErrorCode (*PetscThreadKernel)(PetscInt,...);

/*
  ThreadComm - Abstract object that manages all thread communication models

  Level: developer

  Concepts: threads

.seealso: PetscThreadCommCreate(), PetscThreadCommDestroy
S*/
typedef struct _p_PetscThreadComm* PetscThreadComm;

#define PetscThreadCommType char*
#define PTHREAD             "pthread"
#define NOTHREAD            "nothread"
#define OPENMP              "openmp"

extern PetscFList PetscThreadCommList;

typedef enum {THREADCOMM_SUM,THREADCOMM_PROD} PetscThreadCommReductionType;
extern const char *const PetscThreadCommReductionTypes[];

extern PetscErrorCode PetscGetNCores(PetscInt*);
extern PetscErrorCode PetscCommGetThreadComm(MPI_Comm,PetscThreadComm*);
extern PetscErrorCode PetscThreadCommInitializePackage(const char *path);
extern PetscErrorCode PetscThreadCommFinalizePackage();
extern PetscErrorCode PetscThreadCommInitialize();
extern PetscErrorCode PetscThreadCommGetNThreads(MPI_Comm,PetscInt*);
extern PetscErrorCode PetscThreadCommGetAffinities(MPI_Comm,PetscInt[]);
extern PetscErrorCode PetscThreadCommView(MPI_Comm,PetscViewer);
extern PetscErrorCode PetscThreadCommRunKernel(MPI_Comm,PetscErrorCode (*)(PetscInt,...),PetscInt,...);
extern PetscErrorCode PetscThreadCommBarrier(MPI_Comm);

extern PetscErrorCode PetscThreadCommRegisterDestroy(void);

extern PetscErrorCode PetscThreadReductionKernelBegin(PetscInt,PetscThreadComm,PetscThreadCommReductionType,PetscDataType,void*,void*);
extern PetscErrorCode PetscThreadReductionKernelEnd(PetscInt,PetscThreadComm,PetscThreadCommReductionType,PetscDataType,void*,void*);
extern PetscErrorCode PetscThreadCommGetOwnershipRanges(MPI_Comm,PetscInt,PetscInt*[]);
extern PetscInt PetscThreadCommGetRank(PetscThreadComm);

PETSC_EXTERN_CXX_END
#endif
