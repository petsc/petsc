#ifndef __TAO_UTIL_H
#define __TAO_UTIL_H
#include "petscvec.h"
#include "petscmat.h"
#include "petscksp.h"

PetscErrorCode VecPow(Vec, PetscReal);
PetscErrorCode VecMedian(Vec, Vec, Vec, Vec);
PetscErrorCode VecFischer(Vec, Vec, Vec, Vec, Vec);
PetscErrorCode VecSFischer(Vec, Vec, Vec, Vec, PetscReal, Vec);
PetscErrorCode D_Fischer(Mat, Vec, Vec, Vec, Vec, Vec, Vec, Vec, Vec);
PetscErrorCode D_SFischer(Mat, Vec, Vec, Vec, Vec, PetscReal, Vec, Vec, Vec, Vec, Vec);

PetscErrorCode VecBoundGradientProjection(Vec, Vec, Vec, Vec, Vec);
PetscErrorCode VecStepBoundInfo(Vec,Vec,Vec,Vec,PetscReal*, PetscReal*,PetscReal*);
PetscErrorCode VecStepMax(Vec, Vec, PetscReal *);


/*E
  TaoSubsetType - PetscInt representing the way TAO handles active sets

+ TAO_SUBSET_SUBVEC - TAO uses PETSc's MatGetSubMatrix and VecGetSubVector
. TAO_SUBSET_MASK - Matrices are zeroed out corresponding to active set entries
- TAO_SUBSET_MATRIXFREE - Same as TAO_SUBSET_MASK, but can be applied to matrix-free operators

  Options database keys:
. -different_hessian - TAO will use a copy of the hessian operator for masking.  By default
                       TAO will directly alter the hessian operator.

E*/
typedef enum {TAO_SUBSET_SUBVEC,TAO_SUBSET_MASK,TAO_SUBSET_MATRIXFREE,TAO_SUBSET_TYPES} TaoSubsetType;
PETSC_EXTERN const char *const TaoSubsetTypes[];

PetscErrorCode VecWhichBetween(Vec, Vec, Vec, IS *);
PetscErrorCode VecWhichBetweenOrEqual(Vec, Vec, Vec, IS *);
PetscErrorCode VecWhichGreaterThan(Vec, Vec, IS * );
PetscErrorCode VecWhichLessThan(Vec, Vec, IS *);
PetscErrorCode VecWhichEqual(Vec, Vec, IS *);

PetscErrorCode VecGetSubVec(Vec, IS, PetscInt, PetscReal, Vec*);
PetscErrorCode VecReducedXPY(Vec, Vec, IS);
PetscErrorCode ISCreateComplement(IS, Vec, IS *);
PetscErrorCode VecISSetToConstant(IS, PetscReal, Vec);
PetscErrorCode MatGetSubMat(Mat, IS, Vec, TaoSubsetType, Mat*);
#endif /* defined __TAOUTIL_H */
