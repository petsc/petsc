#pragma once

#include <petscmat.h>

/* MANSEC = ML */
/* SUBMANSEC = PetscDA */

/*S
   PetscDA - Abstract PETSc object that manages data assimilation

   Level: beginner

   Notes:
   This is new code, please independently verify all results you obtain using it.

   Some planned work for `PetscDA` is available as GitLab Issue #1882

   Currently we supply two ensemble-based assimilators: `PETSCDAETKF` and `PETSCDALETKF`

.seealso: [](ch_da), `PetscDAType`, `PETSCDAETKF`, `PETSCDALETKF`, `PetscDASqrtType`, `PetscDACreate()`, `PetscDASetType()`,
          `PetscDASetSizes()`, `PetscDAEnsembleSetSize()`, `PetscDAEnsembleAnalysis()`, `PetscDAEnsembleForecast()`,
          `PetscDADestroy()`, `PetscDAView()`
S*/
typedef struct _p_PetscDA *PetscDA;

/*E
  PetscDASqrtType - Type of square root of matrices to use the data assimilation algorithms

  Values:
+  `PETSCDA_SQRT_CHOLESKY` - Use the Cholesky factorization
-  `PETSCDA_SQRT_EIGEN`    - Use the eigenvalue decomposition

  Option Database Key:
. -petscda_ensemble_sqrt_type <cholesky, eigen> - select the square root type at run time

  Level: intermediate

.seealso: [](ch_da), `PetscDA`, `PetscDAEnsembleSetSqrtType()`, `PetscDAEnsembleGetSqrtType()`
E*/
typedef enum {
  PETSCDA_SQRT_CHOLESKY = 0,
  PETSCDA_SQRT_EIGEN    = 1
} PetscDASqrtType;

/*J
  PetscDAType - String with the name of a PETSc data assimilation method

  Level: beginner

.seealso: [](ch_da), `PetscDA`, `PetscDASetType()`, `PETSCDAETKF`, `PETSCDALETKF`
J*/
typedef const char *PetscDAType;
#define PETSCDAETKF  "etkf"
#define PETSCDALETKF "letkf"

PETSC_EXTERN PetscErrorCode PetscDAInitializePackage(void);
PETSC_EXTERN PetscErrorCode PetscDAFinalizePackage(void);

PETSC_EXTERN PetscErrorCode PetscDARegister(const char[], PetscErrorCode (*)(PetscDA));
PETSC_EXTERN PetscErrorCode PetscDARegisterAll(void);

PETSC_EXTERN PetscErrorCode PetscDACreate(MPI_Comm, PetscDA *);
PETSC_EXTERN PetscErrorCode PetscDADestroy(PetscDA *);
PETSC_EXTERN PetscErrorCode PetscDASetType(PetscDA, PetscDAType);
PETSC_EXTERN PetscErrorCode PetscDAGetType(PetscDA, PetscDAType *);
PETSC_EXTERN PetscErrorCode PetscDAView(PetscDA, PetscViewer);
PETSC_EXTERN PetscErrorCode PetscDAViewFromOptions(PetscDA, PetscObject, const char[]);
PETSC_EXTERN PetscErrorCode PetscDASetFromOptions(PetscDA);

PETSC_EXTERN PetscErrorCode PetscDASetSizes(PetscDA, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode PetscDASetLocalSizes(PetscDA, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode PetscDAGetSizes(PetscDA, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscDASetNDOF(PetscDA, PetscInt);
PETSC_EXTERN PetscErrorCode PetscDAGetNDOF(PetscDA, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscDASetUp(PetscDA);

PETSC_EXTERN PetscErrorCode PetscDASetObsErrorVariance(PetscDA, Vec);
PETSC_EXTERN PetscErrorCode PetscDAGetObsErrorVariance(PetscDA, Vec *);

PETSC_EXTERN PetscErrorCode PetscDASetOptionsPrefix(PetscDA, const char[]);
PETSC_EXTERN PetscErrorCode PetscDAAppendOptionsPrefix(PetscDA, const char[]);
PETSC_EXTERN PetscErrorCode PetscDAGetOptionsPrefix(PetscDA, const char *[]);

PETSC_EXTERN PetscErrorCode PetscDAEnsembleSetSize(PetscDA, PetscInt);
PETSC_EXTERN PetscErrorCode PetscDAEnsembleGetSize(PetscDA, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscDAEnsembleSetInflation(PetscDA, PetscReal);
PETSC_EXTERN PetscErrorCode PetscDAEnsembleGetInflation(PetscDA, PetscReal *);

PETSC_EXTERN PetscErrorCode PetscDAEnsembleGetMember(PetscDA, PetscInt, Vec *);
PETSC_EXTERN PetscErrorCode PetscDAEnsembleRestoreMember(PetscDA, PetscInt, Vec *);
PETSC_EXTERN PetscErrorCode PetscDAEnsembleSetMember(PetscDA, PetscInt, Vec);

PETSC_EXTERN PetscErrorCode PetscDAEnsembleComputeMean(PetscDA, Vec);
PETSC_EXTERN PetscErrorCode PetscDAEnsembleComputeAnomalies(PetscDA, Vec, Mat *);
PETSC_EXTERN PetscErrorCode PetscDAEnsembleAnalysis(PetscDA, Vec, Mat);
PETSC_EXTERN PetscErrorCode PetscDAEnsembleForecast(PetscDA, PetscErrorCode (*)(Vec, Vec, PetscCtx), PetscCtx);
PETSC_EXTERN PetscErrorCode PetscDAEnsembleInitialize(PetscDA, Vec, PetscReal, PetscRandom);

PETSC_EXTERN PetscErrorCode PetscDAEnsembleComputeNormalizedInnovationMatrix(Mat, Vec, Vec, PetscInt, PetscScalar, Mat);
PETSC_EXTERN PetscErrorCode PetscDAEnsembleSetSqrtType(PetscDA, PetscDASqrtType);
PETSC_EXTERN PetscErrorCode PetscDAEnsembleGetSqrtType(PetscDA, PetscDASqrtType *);
PETSC_EXTERN PetscErrorCode PetscDAEnsembleTFactor(PetscDA, Mat);
PETSC_EXTERN PetscErrorCode PetscDAEnsembleApplyTInverse(PetscDA, Vec, Vec);
PETSC_EXTERN PetscErrorCode PetscDAEnsembleApplySqrtTInverse(PetscDA, Mat, Mat);

PETSC_EXTERN PetscErrorCode PetscDALETKFSetLocalization(PetscDA, Mat, Mat);
PETSC_EXTERN PetscErrorCode PetscDALETKFSetObsPerVertex(PetscDA, PetscInt);
PETSC_EXTERN PetscErrorCode PetscDALETKFGetObsPerVertex(PetscDA, PetscInt *);

#if defined(PETSC_HAVE_KOKKOS_KERNELS)
PETSC_EXTERN PetscErrorCode PetscDALETKFGetLocalizationMatrix(const PetscInt, const PetscInt, Vec[3], PetscReal[3], Mat, Mat *);
#endif
