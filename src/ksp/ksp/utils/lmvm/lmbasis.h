#pragma once
#include <petscmat.h>

PETSC_INTERN PetscLogEvent LMBASIS_GEMM;
PETSC_INTERN PetscLogEvent LMBASIS_GEMV;
PETSC_INTERN PetscLogEvent LMBASIS_GEMVH;

typedef struct _n_VecLink *VecLink;

struct _n_VecLink {
  Vec     vec;
  VecLink next;
};

// Limited Memory Basis
typedef struct _n_LMBasis *LMBasis;
struct _n_LMBasis {
  PetscInt         m;                   // Number of vectors in the limited memory window
  PetscInt         k;                   // Index of the history-order next vector to be inserted
  Mat              vecs;                // Dense matrix backing storage of vectors
  PetscObjectId    operator_id;         // If these vecs include the output of an operator (like B0 * S), the id of the operator
  PetscObjectState operator_state;      // The state of the operator when vectors in S were computed, to determine when basis vectors are stale because B0 has changed
  Vec              cached_product;      // Some methods will cache v <- S^T g during MatLMVMUpdate(B, x, g) for use in MatSolve(B, g, p), this is that v
  PetscObjectId    cached_vec_id;       // The id of g, to help determine when an input vector is g that was used to compute v
  PetscObjectState cached_vec_state;    // The state of g when v was computed, to determine when v is stale because g has changed
  VecLink          work_vecs_available; // work vectors the layout of column vectors of S
  VecLink          work_vecs_in_use;
  VecLink          work_rows_available; // work vectors the layout of row vectors of S
  VecLink          work_rows_in_use;
};

PETSC_INTERN PetscErrorCode LMBasisCreate(Vec, PetscInt, LMBasis *);
PETSC_INTERN PetscErrorCode LMBasisDestroy(LMBasis *);
PETSC_INTERN PetscErrorCode LMBasisReset(LMBasis);
PETSC_INTERN PetscErrorCode LMBasisGetNextVec(LMBasis, Vec *);
PETSC_INTERN PetscErrorCode LMBasisRestoreNextVec(LMBasis, Vec *);
PETSC_INTERN PetscErrorCode LMBasisSetNextVec(LMBasis, Vec);
PETSC_INTERN PetscErrorCode LMBasisGetVecRead(LMBasis, PetscInt, Vec *);
PETSC_INTERN PetscErrorCode LMBasisRestoreVecRead(LMBasis, PetscInt, Vec *);
PETSC_INTERN PetscErrorCode LMBasisCopy(LMBasis, LMBasis);
PETSC_INTERN PetscErrorCode LMBasisCreateRow(LMBasis, Vec *);
PETSC_INTERN PetscErrorCode LMBasisGetWorkRow(LMBasis, Vec *);
PETSC_INTERN PetscErrorCode LMBasisRestoreWorkRow(LMBasis, Vec *);
PETSC_INTERN PetscErrorCode LMBasisGetWorkVec(LMBasis, Vec *);
PETSC_INTERN PetscErrorCode LMBasisRestoreWorkVec(LMBasis, Vec *);
PETSC_INTERN PetscErrorCode LMBasisGetRange(LMBasis, PetscInt *, PetscInt *);
PETSC_INTERN PetscErrorCode LMBasisGEMV(LMBasis, PetscInt, PetscInt, PetscScalar, Vec, PetscScalar, Vec);
PETSC_INTERN PetscErrorCode LMBasisGEMVH(LMBasis, PetscInt, PetscInt, PetscScalar, Vec, PetscScalar, Vec);
PETSC_INTERN PetscErrorCode LMBasisGEMMH(LMBasis, PetscInt, PetscInt, LMBasis, PetscInt, PetscInt, PetscScalar, PetscScalar, Mat);
PETSC_INTERN PetscErrorCode LMBasisSetCachedProduct(LMBasis, Vec, Vec);
