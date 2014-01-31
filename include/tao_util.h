#ifndef __TAO_UTIL_H
#define __TAO_UTIL_H
#include <petscvec.h>
#include <petscmat.h>
#include <petscksp.h>

PetscErrorCode VecFischer(Vec, Vec, Vec, Vec, Vec);
PetscErrorCode VecSFischer(Vec, Vec, Vec, Vec, PetscReal, Vec);
PetscErrorCode D_Fischer(Mat, Vec, Vec, Vec, Vec, Vec, Vec, Vec, Vec);
PetscErrorCode D_SFischer(Mat, Vec, Vec, Vec, Vec, PetscReal, Vec, Vec, Vec, Vec, Vec);


/*E
  TaoSubsetType - PetscInt representing the way TAO handles active sets

+ TAO_SUBSET_SUBVEC - TAO uses PETSc's MatGetSubMatrix and VecGetSubVector
. TAO_SUBSET_MASK - Matrices are zeroed out corresponding to active set entries
- TAO_SUBSET_MATRIXFREE - Same as TAO_SUBSET_MASK, but can be applied to matrix-free operators

  Options database keys:
. -different_hessian - TAO will use a copy of the hessian operator for masking.  By default
                       TAO will directly alter the hessian operator.

E*/
typedef enum {TAO_SUBSET_SUBVEC,TAO_SUBSET_MASK,TAO_SUBSET_MATRIXFREE} TaoSubsetType;
PETSC_EXTERN const char *const TaoSubsetTypes[];



PetscErrorCode VecGetSubVec(Vec, IS, PetscInt, PetscReal, Vec*);
PetscErrorCode MatGetSubMat(Mat, IS, Vec, TaoSubsetType, Mat*);
#endif /* defined __TAOUTIL_H */
