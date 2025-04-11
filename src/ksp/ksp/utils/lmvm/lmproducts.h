#pragma once
#include "lmbasis.h"

PETSC_INTERN PetscLogEvent LMPROD_Mult;
PETSC_INTERN PetscLogEvent LMPROD_Solve;
PETSC_INTERN PetscLogEvent LMPROD_Update;

// Refers to blocks of LMProducts
typedef enum {
  LMBLOCK_DIAGONAL              = 0,
  LMBLOCK_UPPER_TRIANGLE        = 1,
  LMBLOCK_STRICT_UPPER_TRIANGLE = 2,
  LMBLOCK_FULL                  = 3,
  LMBLOCK_END,
} LMBlockType;

// inner-products of LMBasis vectors
typedef struct _n_LMProducts *LMProducts;
struct _n_LMProducts {
  PetscInt         m;
  PetscInt         k;
  PetscInt         m_local; // rank 0 will have all values (m_local = m), others have none (m_local = 0)
  Mat              full;
  Vec              diagonal_dup;    // duplicated on each host process
  Vec              diagonal_global; // matches the memory location and layout of an LMBasis
  Vec              diagonal_local;  // matches the memory location and layout of an LMBasis
  PetscBool        update_diagonal_global;
  LMBlockType      block_type;
  PetscObjectId    operator_id;
  PetscObjectState operator_state;
  PetscBool        debug;
  Vec              rhs_local, lhs_local;
};

PETSC_INTERN PetscErrorCode LMProductsCreate(LMBasis, LMBlockType, LMProducts *);
PETSC_INTERN PetscErrorCode LMProductsDestroy(LMProducts *);
PETSC_INTERN PetscErrorCode LMProductsReset(LMProducts);
PETSC_INTERN PetscErrorCode LMProductsPrepare(LMProducts, Mat, PetscInt, PetscInt);
PETSC_INTERN PetscErrorCode LMProductsInsertNextDiagonalValue(LMProducts, PetscInt, PetscScalar);
PETSC_INTERN PetscErrorCode LMProductsGetDiagonalValue(LMProducts, PetscInt, PetscScalar *);
PETSC_INTERN PetscErrorCode LMProductsUpdate(LMProducts, LMBasis, LMBasis);
PETSC_INTERN PetscErrorCode LMProductsCopy(LMProducts, LMProducts);
PETSC_INTERN PetscErrorCode LMProductsScale(LMProducts, PetscScalar);
PETSC_INTERN PetscErrorCode LMProductsGetLocalMatrix(LMProducts, Mat *, PetscInt *, PetscBool *);
PETSC_INTERN PetscErrorCode LMProductsRestoreLocalMatrix(LMProducts, Mat *, PetscInt *);
PETSC_INTERN PetscErrorCode LMProductsGetLocalDiagonal(LMProducts, Vec *);
PETSC_INTERN PetscErrorCode LMProductsRestoreLocalDiagonal(LMProducts, Vec *);
PETSC_INTERN PetscErrorCode LMProductsSolve(LMProducts, PetscInt, PetscInt, Vec, Vec, PetscBool);
PETSC_INTERN PetscErrorCode LMProductsMult(LMProducts, PetscInt, PetscInt, PetscScalar, Vec, PetscScalar, Vec, PetscBool);
PETSC_INTERN PetscErrorCode LMProductsMultHermitian(LMProducts, PetscInt, PetscInt, PetscScalar, Vec, PetscScalar, Vec);
PETSC_INTERN PetscErrorCode LMProductsGetNextColumn(LMProducts, Vec *);
PETSC_INTERN PetscErrorCode LMProductsRestoreNextColumn(LMProducts, Vec *);
PETSC_INTERN PetscErrorCode LMProductsMakeHermitian(Mat, PetscInt, PetscInt);
PETSC_INTERN PetscErrorCode LMProductsOnesOnUnusedDiagonal(Mat, PetscInt, PetscInt);
