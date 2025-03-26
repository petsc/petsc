#include <../src/ksp/ksp/utils/lmvm/symbrdn/symbrdn.h> /*I "petscksp.h" I*/

/* The DFP update can be written as

     B_{k+1} = (I - y_k (y_k^T s_k)^{-1} s_k^T) B_k (I - s_k (y_k^T s_k)^{-1} y_k^T) + y_k (y_k^T s_k)^{-1} y_k^T

   So B_{k+1}x can be computed in the following way

     a_k = (y_k^T s_k)^{-1} y_k^T x
     g = B_k (x - a s_k) <--- recursion
     B_{k+1}x = g + y_k(a - (y_k^T s_k)^{-1} s_k^T g)
 */
PETSC_INTERN PetscErrorCode DFPKernel_Recursive(Mat B, MatLMVMMode mode, Vec X, Vec BX)
{
  MatLMVMBasisType S_t = LMVMModeMap(LMBASIS_S, mode);
  MatLMVMBasisType Y_t = LMVMModeMap(LMBASIS_Y, mode);
  LMBasis          S, Y;
  LMProducts       YtS;
  Vec              G     = X;
  PetscScalar     *alpha = NULL;
  PetscInt         oldest, next;

  PetscFunctionBegin;
  PetscCall(MatLMVMGetRange(B, &oldest, &next));
  PetscCall(MatLMVMGetUpdatedBasis(B, S_t, &S, NULL, NULL));
  PetscCall(MatLMVMGetUpdatedBasis(B, Y_t, &Y, NULL, NULL));
  PetscCall(MatLMVMGetUpdatedProducts(B, Y_t, S_t, LMBLOCK_DIAGONAL, &YtS));
  if (next > oldest) {
    PetscCall(LMBasisGetWorkVec(S, &G));
    PetscCall(VecCopy(X, G));
    PetscCall(PetscMalloc1(next - oldest, &alpha));
  }
  for (PetscInt i = next - 1; i >= oldest; i--) {
    Vec         s_i, y_i;
    PetscScalar yitsi, yitx, a;

    PetscCall(LMBasisGetVecRead(Y, i, &y_i));
    PetscCall(VecDot(G, y_i, &yitx));
    PetscCall(LMBasisRestoreVecRead(Y, i, &y_i));
    PetscCall(LMProductsGetDiagonalValue(YtS, i, &yitsi));
    alpha[i - oldest] = a = yitx / yitsi;
    PetscCall(LMBasisGetVecRead(S, i, &s_i));
    PetscCall(VecAXPY(G, -a, s_i));
    PetscCall(LMBasisRestoreVecRead(S, i, &s_i));
  }
  PetscCall(MatLMVMApplyJ0Mode(mode)(B, G, BX));
  for (PetscInt i = oldest; i < next; i++) {
    Vec         s_i, y_i;
    PetscScalar yitsi, sitbx, a, b;

    PetscCall(LMBasisGetVecRead(S, i, &s_i));
    PetscCall(VecDot(BX, s_i, &sitbx));
    PetscCall(LMBasisRestoreVecRead(S, i, &s_i));
    PetscCall(LMProductsGetDiagonalValue(YtS, i, &yitsi));
    a = alpha[i - oldest];
    b = sitbx / yitsi;
    PetscCall(LMBasisGetVecRead(Y, i, &y_i));
    PetscCall(VecAXPY(BX, a - b, y_i));
    PetscCall(LMBasisRestoreVecRead(Y, i, &y_i));
  }
  if (next > oldest) {
    PetscCall(PetscFree(alpha));
    PetscCall(LMBasisRestoreWorkVec(S, &G));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode DFPKernel_CompactDense(Mat B, MatLMVMMode mode, Vec X, Vec BX)
{
  PetscInt oldest, next;

  PetscFunctionBegin;
  PetscCall(MatLMVMApplyJ0Mode(mode)(B, X, BX));
  PetscCall(MatLMVMGetRange(B, &oldest, &next));
  if (next > oldest) {
    MatLMVMBasisType S_t   = LMVMModeMap(LMBASIS_S, mode);
    MatLMVMBasisType Y_t   = LMVMModeMap(LMBASIS_Y, mode);
    MatLMVMBasisType B0S_t = LMVMModeMap(LMBASIS_B0S, mode);
    Vec              StB0X, YtX, u, v;
    PetscBool        use_B0S;
    LMBasis          S, Y;
    LMProducts       YtS, StB0S, D;

    PetscCall(MatLMVMGetUpdatedBasis(B, S_t, &S, NULL, NULL));
    PetscCall(MatLMVMGetUpdatedBasis(B, Y_t, &Y, NULL, NULL));
    PetscCall(MatLMVMGetUpdatedProducts(B, Y_t, S_t, LMBLOCK_UPPER_TRIANGLE, &YtS));
    PetscCall(MatLMVMGetUpdatedProducts(B, Y_t, S_t, LMBLOCK_DIAGONAL, &D));
    PetscCall(MatLMVMGetUpdatedProducts(B, S_t, B0S_t, LMBLOCK_UPPER_TRIANGLE, &StB0S));

    PetscCall(MatLMVMGetWorkRow(B, &StB0X));
    PetscCall(MatLMVMGetWorkRow(B, &YtX));
    PetscCall(MatLMVMGetWorkRow(B, &u));
    PetscCall(MatLMVMGetWorkRow(B, &v));

    PetscCall(SymBroydenCompactDenseKernelUseB0S(B, mode, X, &use_B0S));
    if (use_B0S) PetscCall(MatLMVMBasisGEMVH(B, B0S_t, oldest, next, 1.0, X, 0.0, StB0X));
    else PetscCall(LMBasisGEMVH(S, oldest, next, 1.0, BX, 0.0, StB0X));

    PetscCall(LMBasisGEMVH(Y, oldest, next, 1.0, X, 0.0, YtX));
    PetscCall(LMProductsSolve(YtS, oldest, next, YtX, YtX, /* ^H */ PETSC_FALSE));

    PetscCall(VecAXPBY(u, -1.0, 0.0, YtX));
    PetscCall(LMProductsMult(D, oldest, next, 1.0, YtX, 0.0, v, /* ^H */ PETSC_FALSE));
    PetscCall(LMProductsMultHermitian(StB0S, oldest, next, 1.0, YtX, 1.0, v));
    PetscCall(VecAXPY(v, -1.0, StB0X));

    PetscCall(LMProductsSolve(YtS, oldest, next, v, v, /* ^H */ PETSC_TRUE));
    PetscCall(LMBasisGEMV(Y, oldest, next, 1.0, v, 1.0, BX));
    PetscCall(MatLMVMBasisGEMV(B, B0S_t, oldest, next, 1.0, u, 1.0, BX));

    PetscCall(MatLMVMRestoreWorkRow(B, &v));
    PetscCall(MatLMVMRestoreWorkRow(B, &u));
    PetscCall(MatLMVMRestoreWorkRow(B, &YtX));
    PetscCall(MatLMVMRestoreWorkRow(B, &StB0X));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode DFPKernel_Dense(Mat B, MatLMVMMode mode, Vec X, Vec BX)
{
  Vec        G   = X;
  Vec        YtX = NULL, u = NULL;
  PetscInt   oldest, next;
  LMProducts YtS = NULL, D = NULL;
  LMBasis    S = NULL, Y = NULL;

  PetscFunctionBegin;
  PetscCall(MatLMVMGetRange(B, &oldest, &next));
  if (next > oldest) {
    MatLMVMBasisType S_t = LMVMModeMap(LMBASIS_S, mode);
    MatLMVMBasisType Y_t = LMVMModeMap(LMBASIS_Y, mode);

    PetscCall(MatLMVMGetUpdatedBasis(B, S_t, &S, NULL, NULL));
    PetscCall(MatLMVMGetUpdatedBasis(B, Y_t, &Y, NULL, NULL));
    PetscCall(MatLMVMGetUpdatedProducts(B, Y_t, S_t, LMBLOCK_UPPER_TRIANGLE, &YtS));
    PetscCall(MatLMVMGetUpdatedProducts(B, Y_t, S_t, LMBLOCK_DIAGONAL, &D));
    PetscCall(LMBasisGetWorkVec(Y, &G));
    PetscCall(VecCopy(X, G));

    PetscCall(MatLMVMGetWorkRow(B, &YtX));
    PetscCall(MatLMVMGetWorkRow(B, &u));
    PetscCall(LMBasisGEMVH(Y, oldest, next, 1.0, X, 0.0, YtX));
    PetscCall(LMProductsSolve(YtS, oldest, next, YtX, YtX, /* ^H */ PETSC_FALSE));
    PetscCall(LMBasisGEMV(S, oldest, next, -1.0, YtX, 1.0, G));
  }
  PetscCall(MatLMVMApplyJ0Mode(mode)(B, G, BX));
  if (next > oldest) {
    PetscCall(LMProductsMult(D, oldest, next, 1.0, YtX, 0.0, u, /* ^H */ PETSC_FALSE));
    PetscCall(LMBasisGEMVH(S, oldest, next, -1.0, BX, 1.0, u));
    PetscCall(LMProductsSolve(YtS, oldest, next, u, u, /* ^H */ PETSC_TRUE));
    PetscCall(LMBasisGEMV(Y, oldest, next, 1.0, u, 1.0, BX));
    PetscCall(MatLMVMRestoreWorkRow(B, &u));
    PetscCall(MatLMVMRestoreWorkRow(B, &YtX));
    PetscCall(LMBasisRestoreWorkVec(Y, &G));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMult_LMVMDFP_Recursive(Mat B, Vec X, Vec BX)
{
  PetscFunctionBegin;
  PetscCall(DFPKernel_Recursive(B, MATLMVM_MODE_PRIMAL, X, BX));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMult_LMVMDFP_CompactDense(Mat B, Vec X, Vec BX)
{
  PetscFunctionBegin;
  PetscCall(DFPKernel_CompactDense(B, MATLMVM_MODE_PRIMAL, X, BX));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMult_LMVMDFP_Dense(Mat B, Vec X, Vec BX)
{
  PetscFunctionBegin;
  PetscCall(DFPKernel_Dense(B, MATLMVM_MODE_PRIMAL, X, BX));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSolve_LMVMDFP_Recursive(Mat B, Vec X, Vec HX)
{
  PetscFunctionBegin;
  PetscCall(BFGSKernel_Recursive(B, MATLMVM_MODE_DUAL, X, HX));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSolve_LMVMDFP_CompactDense(Mat B, Vec X, Vec HX)
{
  PetscFunctionBegin;
  PetscCall(BFGSKernel_CompactDense(B, MATLMVM_MODE_DUAL, X, HX));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetFromOptions_LMVMDFP(Mat B, PetscOptionItems PetscOptionsObject)
{
  Mat_LMVM    *lmvm = (Mat_LMVM *)B->data;
  Mat_SymBrdn *ldfp = (Mat_SymBrdn *)lmvm->ctx;

  PetscFunctionBegin;
  PetscCall(MatSetFromOptions_LMVM(B, PetscOptionsObject));
  PetscOptionsHeadBegin(PetscOptionsObject, "DFP method for approximating SPD Jacobian actions (MATLMVMDFP)");
  PetscCall(SymBroydenRescaleSetFromOptions(B, ldfp->rescale, PetscOptionsObject));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatLMVMSetMultAlgorithm_DFP(Mat B)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;

  PetscFunctionBegin;
  switch (lmvm->mult_alg) {
  case MAT_LMVM_MULT_RECURSIVE:
    lmvm->ops->mult  = MatMult_LMVMDFP_Recursive;
    lmvm->ops->solve = MatSolve_LMVMDFP_Recursive;
    break;
  case MAT_LMVM_MULT_DENSE:
    lmvm->ops->mult  = MatMult_LMVMDFP_Dense;
    lmvm->ops->solve = MatSolve_LMVMDFP_CompactDense;
    break;
  case MAT_LMVM_MULT_COMPACT_DENSE:
    lmvm->ops->mult  = MatMult_LMVMDFP_CompactDense;
    lmvm->ops->solve = MatSolve_LMVMDFP_CompactDense;
    break;
  }
  lmvm->ops->multht  = lmvm->ops->mult;
  lmvm->ops->solveht = lmvm->ops->solve;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatCreate_LMVMDFP(Mat B)
{
  Mat_LMVM    *lmvm;
  Mat_SymBrdn *ldfp;

  PetscFunctionBegin;
  PetscCall(MatCreate_LMVMSymBrdn(B));
  PetscCall(PetscObjectChangeTypeName((PetscObject)B, MATLMVMDFP));
  B->ops->setfromoptions = MatSetFromOptions_LMVMDFP;

  lmvm                        = (Mat_LMVM *)B->data;
  lmvm->ops->setmultalgorithm = MatLMVMSetMultAlgorithm_DFP;
  PetscCall(MatLMVMSetMultAlgorithm_DFP(B));

  ldfp             = (Mat_SymBrdn *)lmvm->ctx;
  ldfp->phi_scalar = 1.0;
  ldfp->psi_scalar = 0.0;
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatLMVMSymBadBroydenSetPsi_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatCreateLMVMDFP - Creates a limited-memory Davidon-Fletcher-Powell (DFP) matrix
  used for approximating Jacobians. L-DFP is symmetric positive-definite by
  construction, and is the dual of L-BFGS where Y and S vectors swap roles.

  To use the L-DFP matrix with other vector types, the matrix must be
  created using `MatCreate()` and `MatSetType()`, followed by `MatLMVMAllocate()`.
  This ensures that the internal storage and work vectors are duplicated from the
  correct type of vector.

  Collective

  Input Parameters:
+ comm - MPI communicator
. n    - number of local rows for storage vectors
- N    - global size of the storage vectors

  Output Parameter:
. B - the matrix

  Options Database Keys:
+ -mat_lmvm_scale_type - (developer) type of scaling applied to J0 (none, scalar, diagonal)
. -mat_lmvm_theta      - (developer) convex ratio between BFGS and DFP components of the diagonal J0 scaling
. -mat_lmvm_rho        - (developer) update limiter for the J0 scaling
. -mat_lmvm_alpha      - (developer) coefficient factor for the quadratic subproblem in J0 scaling
. -mat_lmvm_beta       - (developer) exponential factor for the diagonal J0 scaling
- -mat_lmvm_sigma_hist - (developer) number of past updates to use in J0 scaling

  Level: intermediate

  Note:
  It is recommended that one use the `MatCreate()`, `MatSetType()` and/or `MatSetFromOptions()`
  paradigm instead of this routine directly.

.seealso: [](ch_ksp), `MatCreate()`, `MATLMVM`, `MATLMVMDFP`, `MatCreateLMVMBFGS()`, `MatCreateLMVMSR1()`,
          `MatCreateLMVMBroyden()`, `MatCreateLMVMBadBroyden()`, `MatCreateLMVMSymBroyden()`
@*/
PetscErrorCode MatCreateLMVMDFP(MPI_Comm comm, PetscInt n, PetscInt N, Mat *B)
{
  PetscFunctionBegin;
  PetscCall(KSPInitializePackage());
  PetscCall(MatCreate(comm, B));
  PetscCall(MatSetSizes(*B, n, n, N, N));
  PetscCall(MatSetType(*B, MATLMVMDFP));
  PetscCall(MatSetUp(*B));
  PetscFunctionReturn(PETSC_SUCCESS);
}
