#pragma once
#include <petscksp.h>
#include <petsc/private/matimpl.h>
#include <petsc/private/vecimpl.h>
#include "lmbasis.h"
#include "lmproducts.h"

PETSC_INTERN PetscLogEvent MATLMVM_Update;

/*
  MATLMVM format - a base matrix-type that represents Limited-Memory
  Variable Metric (LMVM) approximations of a Jacobian.

  LMVM approximations can be symmetric, symmetric positive-definite,
  rectangular, or otherwise square with no determinable properties.
  Each derived LMVM type should automatically set its matrix properties
  if its construction can guarantee symmetry (MAT_SYMMETRIC) or symmetric
  positive-definiteness (MAT_SPD).
*/

/* MatLMVMReset(Mat, PetscBool) has a simple boolean for destructive/nondestructive reset,
   but internally it is helpful to have more control
 */
enum {
  MAT_LMVM_RESET_HISTORY = 0x0,
  MAT_LMVM_RESET_BASES   = 0x1,
  MAT_LMVM_RESET_J0      = 0x2,
  MAT_LMVM_RESET_VECS    = 0x4,
  MAT_LMVM_RESET_ALL     = 0xf,
};

typedef PetscInt MatLMVMResetMode;

#define MatLMVMResetClearsBases(mode) ((mode) & MAT_LMVM_RESET_BASES)
#define MatLMVMResetClearsJ0(mode)    ((mode) & MAT_LMVM_RESET_J0)
#define MatLMVMResetClearsVecs(mode)  ((mode) & MAT_LMVM_RESET_VECS)
#define MatLMVMResetClearsAll(mode)   ((mode) == MAT_LMVM_RESET_ALL)

typedef struct _MatOps_LMVM *MatOps_LMVM;
struct _MatOps_LMVM {
  PetscErrorCode (*update)(Mat, Vec, Vec);
  PetscErrorCode (*reset)(Mat, MatLMVMResetMode);
  PetscErrorCode (*mult)(Mat, Vec, Vec);
  PetscErrorCode (*multht)(Mat, Vec, Vec);
  PetscErrorCode (*solve)(Mat, Vec, Vec);
  PetscErrorCode (*solveht)(Mat, Vec, Vec);
  PetscErrorCode (*copy)(Mat, Mat, MatStructure);
  PetscErrorCode (*setmultalgorithm)(Mat);
};

/* Identifies vector bases used internally.

   - bases are listed in "dual pairs": e.g. if an algorithm for MatMult() uses basis X somewhere, then the dual
     algorithm for MatSolve will use the dual basis X' in the same place

   - bases are stored in the `basis[]` array in Mat_LMVM, rather than as struct members with names, to enable "modal"
     access: code of the form `basis[LMVMModeMap(LMBASIS_S, mode)]` can be used for
     the primal algorithm (`mode == MATLMVM_MODE_PRIMAL`) and the dual algorithm (`mode == MATLMMV_MODE_DUAL`).
 */
enum {
  LMBASIS_S           = 0, // differences between solutions, S_i = (X_{i+1} - X_i)
  LMBASIS_Y           = 1, // differences in function values, Y_i = (F_{i+1} - F_i)
  LMBASIS_H0Y         = 2, // H_0 = J_0^{-1}
  LMBASIS_B0S         = 3, // B_0 is the symbol used instead of J_0 in many textbooks and papers, we use it internally
  LMBASIS_S_MINUS_H0Y = 4,
  LMBASIS_Y_MINUS_B0S = 5,
  LMBASIS_END
};

typedef PetscInt MatLMVMBasisType;

typedef enum {
  MATLMVM_MODE_PRIMAL = 0,
  MATLMVM_MODE_DUAL   = 1,
} MatLMVMMode;

#define LMVMModeMap(a, mode)                       ((a) ^ (PetscInt)(mode))
#define MatLMVMApplyJ0Mode(mode)                   ((mode) == MATLMVM_MODE_PRIMAL ? MatLMVMApplyJ0Fwd : MatLMVMApplyJ0Inv)
#define MatLMVMApplyJ0HermitianTransposeMode(mode) ((mode) == MATLMVM_MODE_PRIMAL ? MatLMVMApplyJ0HermitianTranspose : MatLMVMApplyJ0InvHermitianTranspose)
#define MatLMVMBasisSizeOf(type)                   ((type) & LMBASIS_Y)

typedef struct {
  /* Core data structures for stored updates */
  struct _MatOps_LMVM ops[1];
  PetscBool           prev_set;
  PetscInt            m, k, nupdates, nrejects, nresets;
  LMBasis             basis[LMBASIS_END];
  LMProducts          products[LMBLOCK_END][LMBASIS_END][LMBASIS_END];
  Vec                 Xprev, Fprev;

  /* User-defined initial Jacobian tools */
  PetscScalar shift;
  Mat         J0;
  KSP         J0ksp;
  PetscBool   disable_ksp_viewers;
  PetscBool   created_J0;
  PetscBool   created_J0ksp;
  PetscBool   do_not_cache_J0_products;
  PetscBool   cache_gradient_products;

  /* Miscellenous parameters */
  PetscReal            eps; /* (default: PetscPowReal(PETSC_MACHINE_EPSILON, 2.0/3.0)) */
  MatLMVMMultAlgorithm mult_alg;
  void                *ctx; /* implementation specific context */
  PetscBool            debug;
} Mat_LMVM;

/* Shared internal functions for LMVM matrices */
PETSC_INTERN PetscErrorCode MatUpdateKernel_LMVM(Mat, Vec, Vec);
PETSC_INTERN PetscErrorCode MatUpdate_LMVM(Mat, Vec, Vec);
PETSC_INTERN PetscErrorCode MatAllocate_LMVM(Mat, Vec, Vec);
PETSC_INTERN PetscErrorCode MatLMVMAllocateBases(Mat);
PETSC_INTERN PetscErrorCode MatLMVMAllocateVecs(Mat);
PETSC_INTERN PetscErrorCode MatLMVMReset_Internal(Mat, MatLMVMResetMode);
PETSC_INTERN PetscErrorCode MatReset_LMVM(Mat, MatLMVMResetMode);

/* LMVM implementations of core Mat functionality */
PETSC_INTERN PetscErrorCode MatSetFromOptions_LMVM(Mat, PetscOptionItems PetscOptionsObject);
PETSC_INTERN PetscErrorCode MatSetUp_LMVM(Mat);
PETSC_INTERN PetscErrorCode MatView_LMVM(Mat, PetscViewer);
PETSC_INTERN PetscErrorCode MatDestroy_LMVM(Mat);
PETSC_INTERN PetscErrorCode MatCreate_LMVM(Mat);

/* Create functions for derived LMVM types */
PETSC_EXTERN PetscErrorCode MatCreate_LMVMDFP(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_LMVMDDFP(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_LMVMBFGS(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_LMVMDBFGS(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_LMVMDQN(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_LMVMSR1(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_LMVMBrdn(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_LMVMBadBrdn(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_LMVMSymBrdn(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_LMVMSymBadBrdn(Mat);
PETSC_EXTERN PetscErrorCode MatCreate_LMVMDiagBrdn(Mat);

PETSC_INTERN PetscErrorCode MatLMVMGetJ0InvDiag(Mat, Vec *);
PETSC_INTERN PetscErrorCode MatLMVMRestoreJ0InvDiag(Mat, Vec *);

PETSC_INTERN PetscErrorCode MatLMVMGetRange(Mat, PetscInt *, PetscInt *);

PETSC_INTERN PetscErrorCode MatLMVMApplyJ0HermitianTranspose(Mat, Vec, Vec);
PETSC_INTERN PetscErrorCode MatLMVMApplyJ0InvHermitianTranspose(Mat, Vec, Vec);

PETSC_INTERN PetscErrorCode MatLMVMGetJ0Scalar(Mat, PetscBool *, PetscScalar *);
PETSC_INTERN PetscErrorCode MatLMVMJ0KSPIsExact(Mat, PetscBool *);

PETSC_INTERN PetscErrorCode MatLMVMUseVecLayoutsIfCompatible(Mat, Vec, Vec);

PETSC_INTERN PetscErrorCode MatLMVMGetUpdatedBasis(Mat, MatLMVMBasisType, LMBasis *, MatLMVMBasisType *, PetscScalar *);
PETSC_INTERN PetscErrorCode MatLMVMGetWorkRow(Mat, Vec *);
PETSC_INTERN PetscErrorCode MatLMVMRestoreWorkRow(Mat, Vec *);
PETSC_INTERN PetscErrorCode MatLMVMBasisGetVecRead(Mat, MatLMVMBasisType, PetscInt, Vec *, PetscScalar *);
PETSC_INTERN PetscErrorCode MatLMVMBasisRestoreVecRead(Mat, MatLMVMBasisType, PetscInt, Vec *, PetscScalar *);
PETSC_INTERN PetscErrorCode MatLMVMBasisGEMVH(Mat, MatLMVMBasisType, PetscInt, PetscInt, PetscScalar, Vec, PetscScalar, Vec);
PETSC_INTERN PetscErrorCode MatLMVMBasisGEMV(Mat, MatLMVMBasisType, PetscInt, PetscInt, PetscScalar, Vec, PetscScalar, Vec);

PETSC_INTERN PetscErrorCode MatLMVMCreateProducts(Mat, LMBlockType, LMProducts *);
PETSC_INTERN PetscErrorCode MatLMVMGetUpdatedProducts(Mat, MatLMVMBasisType, MatLMVMBasisType, LMBlockType, LMProducts *);
PETSC_INTERN PetscErrorCode MatLMVMProductsInsertDiagonalValue(Mat, MatLMVMBasisType, MatLMVMBasisType, PetscInt, PetscScalar);
PETSC_INTERN PetscErrorCode MatLMVMProductsGetDiagonalValue(Mat, MatLMVMBasisType, MatLMVMBasisType, PetscInt, PetscScalar *);

PETSC_INTERN PetscBool  ByrdNocedalSchnabelCite;
PETSC_INTERN const char ByrdNocedalSchnabelCitation[];
