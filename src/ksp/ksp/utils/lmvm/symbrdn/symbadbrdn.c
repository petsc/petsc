#include "symbrdn.h" /*I "petscksp.h" I*/

static PetscErrorCode MatMult_LMVMSymBadBrdn_Recursive(Mat B, Vec X, Vec Y)
{
  Mat_LMVM    *lmvm = (Mat_LMVM *)B->data;
  Mat_SymBrdn *lsb  = (Mat_SymBrdn *)lmvm->ctx;

  PetscFunctionBegin;
  PetscCheck(lsb->psi_scalar != PETSC_DETERMINE, PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_WRONGSTATE, "Psi must first be set using MatLMVMSymBadBroydenSetPsi()");
  if (lsb->psi_scalar == 0.0) {
    PetscCall(DFPKernel_Recursive(B, MATLMVM_MODE_PRIMAL, X, Y));
  } else if (lsb->psi_scalar == 1.0) {
    PetscCall(BFGSKernel_Recursive(B, MATLMVM_MODE_PRIMAL, X, Y));
  } else {
    PetscCall(SymBroydenKernel_Recursive(B, MATLMVM_MODE_PRIMAL, X, Y, PETSC_TRUE));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSolve_LMVMSymBadBrdn_Recursive(Mat B, Vec X, Vec Y)
{
  Mat_LMVM    *lmvm = (Mat_LMVM *)B->data;
  Mat_SymBrdn *lsb  = (Mat_SymBrdn *)lmvm->ctx;

  PetscFunctionBegin;
  PetscCheck(lsb->psi_scalar != PETSC_DETERMINE, PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_WRONGSTATE, "Psi must first be set using MatLMVMSymBadBroydenSetPsi()");
  if (lsb->psi_scalar == 0.0) {
    PetscCall(BFGSKernel_Recursive(B, MATLMVM_MODE_DUAL, X, Y));
  } else if (lsb->psi_scalar == 1.0) {
    PetscCall(DFPKernel_Recursive(B, MATLMVM_MODE_DUAL, X, Y));
  } else {
    PetscCall(SymBroydenKernel_Recursive(B, MATLMVM_MODE_DUAL, X, Y, PETSC_FALSE));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMult_LMVMSymBadBrdn_CompactDense(Mat B, Vec X, Vec Y)
{
  Mat_LMVM    *lmvm = (Mat_LMVM *)B->data;
  Mat_SymBrdn *lsb  = (Mat_SymBrdn *)lmvm->ctx;

  PetscFunctionBegin;
  PetscCheck(lsb->psi_scalar != PETSC_DETERMINE, PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_WRONGSTATE, "Psi must first be set using MatLMVMSymBadBroydenSetPsi()");
  if (lsb->psi_scalar == 0.0) {
    PetscCall(DFPKernel_CompactDense(B, MATLMVM_MODE_PRIMAL, X, Y));
  } else if (lsb->psi_scalar == 1.0) {
    PetscCall(BFGSKernel_CompactDense(B, MATLMVM_MODE_PRIMAL, X, Y));
  } else {
    PetscCall(SymBroydenKernel_CompactDense(B, MATLMVM_MODE_PRIMAL, X, Y, PETSC_TRUE));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSolve_LMVMSymBadBrdn_CompactDense(Mat B, Vec X, Vec Y)
{
  Mat_LMVM    *lmvm = (Mat_LMVM *)B->data;
  Mat_SymBrdn *lsb  = (Mat_SymBrdn *)lmvm->ctx;

  PetscFunctionBegin;
  PetscCheck(lsb->psi_scalar != PETSC_DETERMINE, PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_WRONGSTATE, "Psi must first be set using MatLMVMSymBadBroydenSetPsi()");
  if (lsb->psi_scalar == 0.0) {
    PetscCall(BFGSKernel_CompactDense(B, MATLMVM_MODE_DUAL, X, Y));
  } else if (lsb->psi_scalar == 1.0) {
    PetscCall(DFPKernel_CompactDense(B, MATLMVM_MODE_DUAL, X, Y));
  } else {
    PetscCall(SymBroydenKernel_CompactDense(B, MATLMVM_MODE_DUAL, X, Y, PETSC_FALSE));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatLMVMSetMultAlgorithm_SymBadBrdn(Mat B)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;

  PetscFunctionBegin;
  switch (lmvm->mult_alg) {
  case MAT_LMVM_MULT_RECURSIVE:
    lmvm->ops->mult  = MatMult_LMVMSymBadBrdn_Recursive;
    lmvm->ops->solve = MatSolve_LMVMSymBadBrdn_Recursive;
    break;
  case MAT_LMVM_MULT_DENSE:
  case MAT_LMVM_MULT_COMPACT_DENSE:
    lmvm->ops->mult  = MatMult_LMVMSymBadBrdn_CompactDense;
    lmvm->ops->solve = MatSolve_LMVMSymBadBrdn_CompactDense;
    break;
  }
  lmvm->ops->multht  = lmvm->ops->mult;
  lmvm->ops->solveht = lmvm->ops->solve;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetFromOptions_LMVMSymBadBrdn(Mat B, PetscOptionItems PetscOptionsObject)
{
  Mat_LMVM                  *lmvm = (Mat_LMVM *)B->data;
  Mat_SymBrdn               *lsb  = (Mat_SymBrdn *)lmvm->ctx;
  MatLMVMSymBroydenScaleType stype;

  PetscFunctionBegin;
  PetscCall(MatSetFromOptions_LMVM(B, PetscOptionsObject));
  PetscOptionsHeadBegin(PetscOptionsObject, "Restricted/Symmetric Bad Broyden method for approximating SPD Jacobian actions (MATLMVMSYMBADBROYDEN)");
  PetscCall(PetscOptionsReal("-mat_lmvm_psi", "convex ratio between DFP and BFGS components of the update", "", lsb->psi_scalar, &lsb->psi_scalar, NULL));
  PetscCheck(lsb->psi_scalar >= 0.0 && lsb->psi_scalar <= 1.0, PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_OUTOFRANGE, "convex ratio for the update formula cannot be outside the range of [0, 1]");
  PetscCall(SymBroydenRescaleSetFromOptions(B, lsb->rescale, PetscOptionsObject));
  PetscOptionsHeadEnd();
  PetscCall(SymBroydenRescaleGetType(lsb->rescale, &stype));
  if (stype == MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL) PetscCall(SymBroydenRescaleSetDiagonalMode(lsb->rescale, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatCreate_LMVMSymBadBrdn(Mat B)
{
  Mat_LMVM    *lmvm;
  Mat_SymBrdn *lsb;

  PetscFunctionBegin;
  PetscCall(MatCreate_LMVMSymBrdn(B));
  PetscCall(PetscObjectChangeTypeName((PetscObject)B, MATLMVMSYMBADBROYDEN));
  B->ops->setfromoptions = MatSetFromOptions_LMVMSymBadBrdn;

  lmvm                        = (Mat_LMVM *)B->data;
  lmvm->ops->setmultalgorithm = MatLMVMSetMultAlgorithm_SymBadBrdn;

  lsb                   = (Mat_SymBrdn *)lmvm->ctx;
  lsb->psi_scalar       = 0.875;
  lsb->phi_scalar       = PETSC_DETERMINE;
  lsb->rescale->forward = PETSC_FALSE;
  lsb->rescale->theta   = 1.0 - lsb->psi_scalar;

  PetscCall(MatLMVMSetMultAlgorithm_SymBadBrdn(B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatCreateLMVMSymBadBroyden - Creates a limited-memory Symmetric "Bad" Broyden-type matrix used
  for approximating Jacobians.

  Collective

  Input Parameters:
+ comm - MPI communicator
. n    - number of local rows for storage vectors
- N    - global size of the storage vectors

  Output Parameter:
. B - the matrix

  Options Database Keys:
+ -mat_lmvm_hist_size         - the number of history vectors to keep
. -mat_lmvm_psi               - convex ratio between BFGS and DFP components of the update
. -mat_lmvm_scale_type        - type of scaling applied to J0 (none, scalar, diagonal)
. -mat_lmvm_mult_algorithm    - the algorithm to use for multiplication (recursive, dense, compact_dense)
. -mat_lmvm_cache_J0_products - whether products between the base Jacobian J0 and history vectors should be cached or recomputed
. -mat_lmvm_eps               - (developer) numerical zero tolerance for testing when an update should be skipped
. -mat_lmvm_debug             - (developer) perform internal debugging checks
. -mat_lmvm_theta             - (developer) convex ratio between BFGS and DFP components of the diagonal J0 scaling
. -mat_lmvm_rho               - (developer) update limiter for the J0 scaling
. -mat_lmvm_alpha             - (developer) coefficient factor for the quadratic subproblem in J0 scaling
. -mat_lmvm_beta              - (developer) exponential factor for the diagonal J0 scaling
- -mat_lmvm_sigma_hist        - (developer) number of past updates to use in J0 scaling

  Level: intermediate

  Notes:
  L-SymBadBrdn is a convex combination of L-DFP and L-BFGS such that $B^{-1} = (1 - \psi)*B_{\text{DFP}}^{-1} +
  \psi*B_{\text{BFGS}}^{-1}$. The combination factor $\psi$ is restricted to the range $[0, 1]$, where the L-SymBadBrdn matrix
  is guaranteed to be symmetric positive-definite. Note that this combination is on the inverses and not on the
  forwards. For forward convex combinations, use the L-SymBrdn matrix (`MATLMVMSYMBROYDEN`).

  To use the L-SymBrdn matrix with other vector types, the matrix must be created using `MatCreate()` and
  `MatSetType()`, followed by `MatLMVMAllocate()`.  This ensures that the internal storage and work vectors are
  duplicated from the correct type of vector.

  It is recommended that one use the `MatCreate()`, `MatSetType()` and/or `MatSetFromOptions()` paradigm instead of this
  routine directly.

.seealso: [](ch_ksp), [LMVM Matrices](sec_matlmvm), `MatCreate()`, `MATLMVM`, `MATLMVMSYMBROYDEN`, `MatCreateLMVMDFP()`, `MatCreateLMVMSR1()`,
          `MatCreateLMVMBFGS()`, `MatCreateLMVMBroyden()`, `MatCreateLMVMBadBroyden()`
@*/
PetscErrorCode MatCreateLMVMSymBadBroyden(MPI_Comm comm, PetscInt n, PetscInt N, Mat *B)
{
  PetscFunctionBegin;
  PetscCall(KSPInitializePackage());
  PetscCall(MatCreate(comm, B));
  PetscCall(MatSetSizes(*B, n, n, N, N));
  PetscCall(MatSetType(*B, MATLMVMSYMBADBROYDEN));
  PetscCall(MatSetUp(*B));
  PetscFunctionReturn(PETSC_SUCCESS);
}
