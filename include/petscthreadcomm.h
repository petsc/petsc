
#if !defined(__PETSCTHREADCOMM_H)
#define __PETSCTHREADCOMM_H
#include <petscsys.h>

/* Function pointer cast for the kernel function */
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode (*PetscThreadKernel)(PetscInt,...);

/*
  PetscThreadComm - Abstract object that manages all thread communication models

  Level: developer

  Concepts: threads

.seealso: PetscThreadCommCreate(), PetscThreadCommDestroy
*/
typedef struct _p_PetscThreadComm *PetscThreadComm;

/*
   PetscThreadCommReduction - Context used for managing threaded reductions

   Level: developer
*/
typedef struct _p_PetscThreadCommReduction *PetscThreadCommReduction;

typedef const char* PetscThreadCommType;
#define PTHREAD             "pthread"
#define NOTHREAD            "nothread"
#define OPENMP              "openmp"

PETSC_EXTERN PetscFList PetscThreadCommList;

typedef enum {THREADCOMM_SUM,THREADCOMM_PROD,THREADCOMM_MAX,THREADCOMM_MIN,THREADCOMM_MAXLOC,THREADCOMM_MINLOC} PetscThreadCommReductionOp;
PETSC_EXTERN const char* const PetscThreadCommReductionOps[];

/* Max. number of reductions */
#define PETSC_REDUCTIONS_MAX 32

PETSC_EXTERN PetscErrorCode PetscGetNCores(PetscInt*);
PETSC_EXTERN PetscErrorCode PetscCommGetThreadComm(MPI_Comm,PetscThreadComm*);
PETSC_EXTERN PetscErrorCode PetscThreadCommInitializePackage(const char *path);
PETSC_EXTERN PetscErrorCode PetscThreadCommFinalizePackage(void);
PETSC_EXTERN PetscErrorCode PetscThreadCommInitialize(void);
PETSC_EXTERN PetscErrorCode PetscThreadCommGetNThreads(MPI_Comm,PetscInt*);
PETSC_EXTERN PetscErrorCode PetscThreadCommGetAffinities(MPI_Comm,PetscInt[]);
PETSC_EXTERN PetscErrorCode PetscThreadCommView(MPI_Comm,PetscViewer);
PETSC_EXTERN PetscErrorCode PetscThreadCommGetScalars(MPI_Comm,PetscScalar**,PetscScalar**,PetscScalar**);
PETSC_EXTERN PetscErrorCode PetscThreadCommGetInts(MPI_Comm,PetscInt**,PetscInt**,PetscInt**);
PETSC_EXTERN PetscErrorCode PetscThreadCommRunKernel(MPI_Comm,PetscErrorCode (*)(PetscInt,...),PetscInt,...);
PETSC_EXTERN PetscErrorCode PetscThreadCommRunKernel0(MPI_Comm,PetscErrorCode (*)(PetscInt,...));
PETSC_EXTERN PetscErrorCode PetscThreadCommRunKernel1(MPI_Comm,PetscErrorCode (*)(PetscInt,...),void*);
PETSC_EXTERN PetscErrorCode PetscThreadCommRunKernel2(MPI_Comm,PetscErrorCode (*)(PetscInt,...),void*,void*);
PETSC_EXTERN PetscErrorCode PetscThreadCommRunKernel3(MPI_Comm,PetscErrorCode (*)(PetscInt,...),void*,void*,void*);
PETSC_EXTERN PetscErrorCode PetscThreadCommRunKernel4(MPI_Comm,PetscErrorCode (*)(PetscInt,...),void*,void*,void*,void*);
PETSC_EXTERN PetscErrorCode PetscThreadCommRunKernel6(MPI_Comm,PetscErrorCode (*)(PetscInt,...),void*,void*,void*,void*,void*,void*);
PETSC_EXTERN PetscErrorCode PetscThreadCommBarrier(MPI_Comm);
PETSC_EXTERN PetscErrorCode PetscThreadCommGetOwnershipRanges(MPI_Comm,PetscInt,PetscInt*[]);
PETSC_EXTERN PetscErrorCode PetscThreadCommRegisterDestroy(void);
PETSC_EXTERN PetscErrorCode PetscThreadCommGetRank(PetscThreadComm,PetscInt*);

/* Reduction operations */
PETSC_EXTERN PetscErrorCode PetscThreadReductionKernelPost(PetscInt,PetscThreadCommReduction,void*);
PETSC_EXTERN PetscErrorCode PetscThreadReductionKernelEnd(PetscInt,PetscThreadCommReduction,void*);
PETSC_EXTERN PetscErrorCode PetscThreadReductionBegin(MPI_Comm,PetscThreadCommReductionOp,PetscDataType,PetscInt,PetscThreadCommReduction*);
PETSC_EXTERN PetscErrorCode PetscThreadReductionEnd(PetscThreadCommReduction,void*);

#endif
