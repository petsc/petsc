#if !defined(__LMVM_H)
#define __LMVM_H
#include <petsc/private/matimpl.h>
#include <petscksp.h>

/*
  MATLMVM format - a matrix-type that represents a Limited Memory Variable Metric approximation of a Jacobian.
*/

typedef struct {
  PetscBool allocated;
  PetscBool recycle;
  PetscInt m, k;
  Vec *S;
  Vec *Y;
  Vec Xprev;
  Vec Fprev;
  Mat H0;
  KSP H0_KSP;
} Mat_LMVM;

PETSC_EXTERN PetscErrorCode MatCreate_LMVM(Mat);
PETSC_EXTERN PetscErrorCode MatUpdate_LMVM(Mat, Vec, Vec);
PETSC_EXTERN PetscErrorCode MatSetH0_LMVM(Mat, Mat);
PETSC_EXTERN PetscErrorCode MatSetH0KSP_LMVM(Mat, KSP);

PETSC_INTERN PetscErrorCode MatSetUp_LMVM(Mat);
PETSC_INTERN PetscErrorCode MatDestroy_LMVM(Mat);
PETSC_INTERN PetscErrorCode MatView_LMVM(Mat);

#endif