#pragma once

#include <../src/ksp/ksp/utils/lmvm/lmvm.h>
#include <../src/ksp/ksp/utils/lmvm/rescale/symbrdnrescale.h>

/*
  Limited-memory Symmetric Broyden method for approximating both
  the forward product and inverse application of a Jacobian.
*/

// bases needed by symmetric [bad] Broyden algorithms beyond those in Mat_LMVM
enum {
  SYMBROYDEN_BASIS_BKS = 0, // B_k S_k for recursive algorithms
  SYMBROYDEN_BASIS_HKY = 1, // dual to the above, H_k Y_k
  SYMBROYDEN_BASIS_COUNT
};

typedef PetscInt SymBroydenBasisType;

// products needed by symmetric [bad] Broyden algorithms beyond those in Mat_LMVM
enum {
  SYMBROYDEN_PRODUCTS_PHI   = 0, // diagonal: either phi_k = phi_scalar (symm. Broyden), or phi_k is different for every k (symm. bad Broyden)
  SYMBROYDEN_PRODUCTS_PSI   = 1, // diagonal: either psi_k = psi_scalar (symm. bad Broyden), or psi_k is different for every k (symm. Broyden)
  SYMBROYDEN_PRODUCTS_STBKS = 2, // diagonal S_k^T B_k S_k values for recursive algorithms
  SYMBROYDEN_PRODUCTS_YTHKY = 3, // dual to the above: diagonal Y_k^T H_k Y_k values
  SYMBROYDEN_PRODUCTS_M00   = 4, // matrix that appears in (B_* S) M_00 (B_* S)^T rank-m updates, either diagonal (recursive) or full (compact)
  SYMBROYDEN_PRODUCTS_N00   = 5, // dual to the above, appears in (H_* Y) N_00 (H_* Y)^T rank-m updates
  SYMBROYDEN_PRODUCTS_M01   = 6, // matrix that appears in (B_* S) M_01 Y^T rank-m updates, either diagonal (recursive) or full (compact)
  SYMBROYDEN_PRODUCTS_N01   = 7, // dual to the above, appears in (H_* Y) N_01 S^T rank-m updates
  SYMBROYDEN_PRODUCTS_M11   = 8, // matrix that appears in Y M_11 Y^T rank-m updates, either diagonal (recursive) or full (compact)
  SYMBROYDEN_PRODUCTS_N11   = 9, // dual to the above, appears in S N_11 S^T rank-m updates
  SYMBROYDEN_PRODUCTS_COUNT
};

typedef PetscInt SymBroydenProductsType;

typedef struct {
  PetscReal         phi_scalar, psi_scalar;
  PetscInt          watchdog, max_seq_rejects; /* tracker to reset after a certain # of consecutive rejects */
  SymBroydenRescale rescale;                   /* context for diagonal or scalar rescaling */
  LMBasis           basis[SYMBROYDEN_BASIS_COUNT];
  LMProducts        products[SYMBROYDEN_PRODUCTS_COUNT];
  Vec               StFprev, YtH0Fprev;
} Mat_SymBrdn;

PETSC_INTERN PetscErrorCode SymBroydenKernel_Recursive(Mat, MatLMVMMode, Vec, Vec, PetscBool);
PETSC_INTERN PetscErrorCode SymBroydenKernel_CompactDense(Mat, MatLMVMMode, Vec, Vec, PetscBool);

PETSC_INTERN PetscErrorCode DFPKernel_Recursive(Mat, MatLMVMMode, Vec, Vec);
PETSC_INTERN PetscErrorCode DFPKernel_CompactDense(Mat, MatLMVMMode, Vec, Vec);
PETSC_INTERN PetscErrorCode DFPKernel_Dense(Mat, MatLMVMMode, Vec, Vec);

PETSC_INTERN PetscErrorCode BFGSKernel_Recursive(Mat, MatLMVMMode, Vec, Vec);
PETSC_INTERN PetscErrorCode BFGSKernel_CompactDense(Mat, MatLMVMMode, Vec, Vec);

PETSC_INTERN PetscErrorCode SymBroydenCompactDenseKernelUseB0S(Mat, MatLMVMMode, Vec, PetscBool *);
