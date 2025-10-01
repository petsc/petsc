#pragma once

#include <../src/ksp/ksp/utils/lmvm/lmvm.h>

/*
  Limited-memory Broyden's method for approximating the inverse of
  a Jacobian.
*/

// Bases used by Broyden & Bad Broyden algorithms beyond those provided in Mat_LMVM
enum {
  BROYDEN_BASIS_Y_MINUS_BKS = 0, // storage for the vectors Y_k - B_k S_k
  BROYDEN_BASIS_S_MINUS_HKY = 1, // dual to the above, S_k - H_k Y_k
  BROYDEN_BASIS_COUNT
};

typedef PetscInt BroydenBasisType;

// Products used by Broyden & Bad Broyden algorithms beyond those provided in Mat_LMVM
enum {
  BROYDEN_PRODUCTS_STHKY           = 0, // diagonal S_k^T (H_k Y_k) values for recursive algorithms
  BROYDEN_PRODUCTS_YTBKS           = 1, // dual to the above, diagonal Y_k^T (B_K S_K) values
  BROYDEN_PRODUCTS_STH0Y_MINUS_STS = 2, // stores and factors S^T B_0 Y - stril(S^T S) for compact algorithms
  BROYDEN_PRODUCTS_YTB0S_MINUS_YTY = 3, // dual to the above, Y^T H_0 S - stril(Y^T Y) for compact algorithms
  BROYDEN_PRODUCTS_COUNT
};

typedef PetscInt BroydenProductsType;

typedef struct {
  LMBasis    basis[BROYDEN_BASIS_COUNT];
  LMProducts products[BROYDEN_PRODUCTS_COUNT];
  Vec        YtFprev;
} Mat_Brdn;

PETSC_INTERN PetscErrorCode BroydenKernel_Recursive(Mat, MatLMVMMode, Vec, Vec);
PETSC_INTERN PetscErrorCode BroydenKernel_CompactDense(Mat, MatLMVMMode, Vec, Vec);
PETSC_INTERN PetscErrorCode BroydenKernel_Dense(Mat, MatLMVMMode, Vec, Vec);
PETSC_INTERN PetscErrorCode BroydenKernelHermitianTranspose_Recursive(Mat, MatLMVMMode, Vec, Vec);
PETSC_INTERN PetscErrorCode BroydenKernelHermitianTranspose_CompactDense(Mat, MatLMVMMode, Vec, Vec);
PETSC_INTERN PetscErrorCode BroydenKernelHermitianTranspose_Dense(Mat, MatLMVMMode, Vec, Vec);
PETSC_INTERN PetscErrorCode BadBroydenKernel_Recursive(Mat, MatLMVMMode, Vec, Vec);
PETSC_INTERN PetscErrorCode BadBroydenKernel_CompactDense(Mat, MatLMVMMode, Vec, Vec);
PETSC_INTERN PetscErrorCode BadBroydenKernel_Dense(Mat, MatLMVMMode, Vec, Vec);
PETSC_INTERN PetscErrorCode BadBroydenKernelHermitianTranspose_Recursive(Mat, MatLMVMMode, Vec, Vec);
PETSC_INTERN PetscErrorCode BadBroydenKernelHermitianTranspose_CompactDense(Mat, MatLMVMMode, Vec, Vec);
PETSC_INTERN PetscErrorCode BadBroydenKernelHermitianTranspose_Dense(Mat, MatLMVMMode, Vec, Vec);
