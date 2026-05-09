#pragma once

#include <petscmat.h>

/* MANSEC = ML */
/* SUBMANSEC = PetscDA */

/*S
   PetscDA - Abstract PETSc object that manages data assimilation

   Level: beginner

   Notes:
   This is new code, please independently verify all results you obtain using it.
   Planned work for `PetscDA` is tracked on GitLab as Issue #1882.
   The only ensemble-based assimilator currently supplied is `PETSCDALETKF`.

.seealso: [](ch_da), `PetscDAType`, `PETSCDALETKF`, `PetscDACreate()`, `PetscDASetType()`,
          `PetscDASetSizes()`, `PetscDAEnsembleSetSize()`, `PetscDAEnsembleAnalysis()`, `PetscDAEnsembleForecast()`,
          `PetscDADestroy()`, `PetscDAView()`
S*/
typedef struct _p_PetscDA *PetscDA;

/*E
  PetscDALETKFLocalizationType - Type of localization kernel used by `PETSCDALETKF`

  Values:
+ `PETSCDA_LETKF_LOC_NONE`         - No localization. Each vertex sees every observation with weight one;
                                     the per-vertex loop reduces to a single global analysis (the classic ETKF).
. `PETSCDA_LETKF_LOC_GASPARI_COHN` - Gaspari-Cohn fifth-order piecewise rational kernel with compact support at twice the radius
. `PETSCDA_LETKF_LOC_GAUSSIAN`     - Gaussian kernel exp(-d^2 / (2 r^2)) truncated at twice the radius
- `PETSCDA_LETKF_LOC_BOXCAR`       - Uniform weight one inside the radius, zero outside

  Options Database Keys:
. -petscda_letkf_localization_type (none|gaspari_cohn|gaussian|boxcar) - select the localization kernel at run time

  Level: intermediate

.seealso: [](ch_da), `PETSCDALETKF`, `PetscDALETKFSetLocalizationType()`, `PetscDALETKFGetLocalizationType()`,
          `PetscDALETKFSetLocalizationRadius()`, `PetscDALETKFSetLocalizationCoordinates()`
E*/
typedef enum {
  PETSCDA_LETKF_LOC_NONE         = 0,
  PETSCDA_LETKF_LOC_GASPARI_COHN = 1,
  PETSCDA_LETKF_LOC_GAUSSIAN     = 2,
  PETSCDA_LETKF_LOC_BOXCAR       = 3,
  PETSCDA_LETKF_LOC_NUM_TYPES
} PetscDALETKFLocalizationType;

/*J
  PetscDAType - String with the name of a PETSc data assimilation method

  Level: beginner

.seealso: [](ch_da), `PetscDA`, `PetscDASetType()`, `PETSCDALETKF`
J*/
typedef const char *PetscDAType;
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

/*S
  PetscDAEnsembleForecastFn - A prototype of an ensemble forecast model callback passed to `PetscDAEnsembleForecast()`

  Calling Sequence:
+ ensemble - ensemble matrix advanced in place; on entry its columns are the current members, on
             return its columns are the members at the next time
- ctx      - [optional] user-defined model context

  Level: intermediate

  Note:
  Per-member integrators that read and overwrite each column (the common case, e.g. a `TS`-driven step)
  iterate with `MatDenseGetColumnVec()` / `MatDenseRestoreColumnVec()`. Use the read-only or write-only
  variants (`MatDenseGetColumnVecRead()` / `MatDenseGetColumnVecWrite()`) only when the kernel truly
  does not need the other half (e.g. a write-only resampler).

.seealso: [](ch_da), `PetscDA`, `PETSCDALETKF`, `PetscDAEnsembleForecast()`
S*/
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode PetscDAEnsembleForecastFn(Mat ensemble, PetscCtx ctx);

PETSC_EXTERN PetscErrorCode PetscDAEnsembleForecast(PetscDA, PetscDAEnsembleForecastFn *, PetscCtx);
PETSC_EXTERN PetscErrorCode PetscDAEnsembleInitialize(PetscDA, Vec, PetscReal, PetscRandom);

PETSC_EXTERN PetscErrorCode PetscDAEnsembleComputeNormalizedInnovationMatrix(Mat, Vec, Vec, PetscInt, PetscScalar, Mat);
PETSC_EXTERN PetscErrorCode PetscDAEnsembleTFactor(PetscDA, Mat);
PETSC_EXTERN PetscErrorCode PetscDAEnsembleApplyTInverse(PetscDA, Vec, Vec);
PETSC_EXTERN PetscErrorCode PetscDAEnsembleApplySqrtTInverse(PetscDA, Mat, Mat);

PETSC_EXTERN PetscErrorCode PetscDALETKFSetLocalizationRadius(PetscDA, PetscReal);
PETSC_EXTERN PetscErrorCode PetscDALETKFGetLocalizationRadius(PetscDA, PetscReal *);
PETSC_EXTERN PetscErrorCode PetscDALETKFSetLocalizationType(PetscDA, PetscDALETKFLocalizationType);
PETSC_EXTERN PetscErrorCode PetscDALETKFGetLocalizationType(PetscDA, PetscDALETKFLocalizationType *);
PETSC_EXTERN PetscErrorCode PetscDALETKFSetLocalizationCoordinates(PetscDA, const Vec[3], const PetscReal[3], Mat);
