#include <../src/ksp/ksp/utils/lmvm/symbrdn/symbrdn.h> /*I "petscksp.h" I*/
#include <petsc/private/vecimpl.h>
#include <petscdevice.h>

/* The BFGS update can be written

   B_{k+1} = B_k + y_k (y_k^T s_k)^{-1} y_k^T - B_k s_k (s_k^T B_k s_k)^{-1} s_k^T B_k + y_k (y_k^T s_k)^{-1} y_k^T

   Which can be unrolled as a parallel sum

   B_{k+1} = B_0 + \sum_i B_i y_i (y_i^T s_i)^{-1} y_i^T - s_i (s_i^T B_i s_i)^{-1} s_i^T B_i

   Once the (B_i y_i) vectors, (y_i^T s_i), and (s_i^T B_i s_i) products have been computed
 */
static PetscErrorCode BFGSKernel_Recursive_Inner(Mat B, MatLMVMMode mode, PetscInt oldest, PetscInt next, Vec X, Vec B0X)
{
  Mat_LMVM        *lmvm = (Mat_LMVM *)B->data;
  Mat_SymBrdn     *lsb  = (Mat_SymBrdn *)lmvm->ctx;
  MatLMVMBasisType Y_t  = LMVMModeMap(LMBASIS_Y, mode);
  LMBasis          BkS  = lsb->basis[LMVMModeMap(SYMBROYDEN_BASIS_BKS, mode)];
  LMProducts       YtS;
  LMProducts       StBkS = lsb->products[LMVMModeMap(SYMBROYDEN_PRODUCTS_STBKS, mode)];
  LMBasis          Y;
  Vec              StBkX, YtX;

  PetscFunctionBegin;
  PetscCall(MatLMVMGetUpdatedBasis(B, Y_t, &Y, NULL, NULL));
  PetscCall(MatLMVMGetUpdatedProducts(B, LMBASIS_Y, LMBASIS_S, LMBLOCK_DIAGONAL, &YtS));
  PetscCall(MatLMVMGetWorkRow(B, &StBkX));
  PetscCall(MatLMVMGetWorkRow(B, &YtX));
  PetscCall(LMBasisGEMVH(BkS, oldest, next, 1.0, X, 0.0, StBkX));
  PetscCall(LMProductsSolve(StBkS, oldest, next, StBkX, StBkX, /* ^H */ PETSC_FALSE));
  PetscCall(LMBasisGEMV(BkS, oldest, next, -1.0, StBkX, 1.0, B0X));
  PetscCall(LMBasisGEMVH(Y, oldest, next, 1.0, X, 0.0, YtX));
  PetscCall(LMProductsSolve(YtS, oldest, next, YtX, YtX, /* ^H */ PETSC_FALSE));
  PetscCall(LMBasisGEMV(Y, oldest, next, 1.0, YtX, 1.0, B0X));
  PetscCall(MatLMVMRestoreWorkRow(B, &YtX));
  PetscCall(MatLMVMRestoreWorkRow(B, &StBkX));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   The B_i s_i vectors and (s_i^T B_i s_i) products are computed recursively
 */
static PetscErrorCode BFGSRecursiveBasisUpdate(Mat B, MatLMVMMode mode)
{
  Mat_LMVM              *lmvm    = (Mat_LMVM *)B->data;
  Mat_SymBrdn           *lsb     = (Mat_SymBrdn *)lmvm->ctx;
  MatLMVMBasisType       S_t     = LMVMModeMap(LMBASIS_S, mode);
  MatLMVMBasisType       B0S_t   = LMVMModeMap(LMBASIS_B0S, mode);
  SymBroydenProductsType StBkS_t = LMVMModeMap(SYMBROYDEN_PRODUCTS_STBKS, mode);
  SymBroydenBasisType    BkS_t   = LMVMModeMap(SYMBROYDEN_BASIS_BKS, mode);
  LMBasis                BkS;
  LMProducts             StBkS, YtS;
  PetscInt               oldest, start, next;
  PetscInt               products_oldest;
  LMBasis                S;

  PetscFunctionBegin;
  PetscCall(MatLMVMGetRange(B, &oldest, &next));
  if (!lsb->basis[BkS_t]) PetscCall(LMBasisCreate(MatLMVMBasisSizeOf(B0S_t) == LMBASIS_S ? lmvm->Xprev : lmvm->Fprev, lmvm->m, &lsb->basis[BkS_t]));
  BkS = lsb->basis[BkS_t];
  if (!lsb->products[StBkS_t]) PetscCall(MatLMVMCreateProducts(B, LMBLOCK_DIAGONAL, &lsb->products[StBkS_t]));
  StBkS = lsb->products[StBkS_t];
  PetscCall(LMProductsPrepare(StBkS, lmvm->J0, oldest, next));
  products_oldest = PetscMax(0, StBkS->k - lmvm->m);
  if (oldest > products_oldest) {
    // recursion is starting from a different starting index, it must be recomputed
    StBkS->k = oldest;
  }
  BkS->k = start = StBkS->k;
  if (start == next) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(MatLMVMGetUpdatedBasis(B, S_t, &S, NULL, NULL));
  // make sure YtS is updated before entering the loop
  PetscCall(MatLMVMGetUpdatedProducts(B, LMBASIS_Y, LMBASIS_S, LMBLOCK_DIAGONAL, &YtS));
  for (PetscInt j = start; j < next; j++) {
    Vec         p_j, s_j, B0s_j;
    PetscScalar alpha, sjtbjsj;

    PetscCall(LMBasisGetWorkVec(BkS, &p_j));
    // p_j starts as B_0 * s_j
    PetscCall(MatLMVMBasisGetVecRead(B, B0S_t, j, &B0s_j, &alpha));
    PetscCall(VecAXPBY(p_j, alpha, 0.0, B0s_j));
    PetscCall(MatLMVMBasisRestoreVecRead(B, B0S_t, j, &B0s_j, &alpha));

    // Use the matmult kernel to compute p_j = B_j * p_j
    PetscCall(LMBasisGetVecRead(S, j, &s_j));
    if (j > oldest) PetscCall(BFGSKernel_Recursive_Inner(B, mode, oldest, j, s_j, p_j));
    PetscCall(VecDot(p_j, s_j, &sjtbjsj));
    PetscCall(LMBasisRestoreVecRead(S, j, &s_j));
    PetscCall(LMProductsInsertNextDiagonalValue(StBkS, j, sjtbjsj));
    PetscCall(LMBasisSetNextVec(BkS, p_j));
    PetscCall(LMBasisRestoreWorkVec(BkS, &p_j));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode BFGSKernel_Recursive(Mat B, MatLMVMMode mode, Vec X, Vec Y)
{
  PetscInt oldest, next;

  PetscFunctionBegin;
  PetscCall(MatLMVMApplyJ0Mode(mode)(B, X, Y));
  PetscCall(MatLMVMGetRange(B, &oldest, &next));
  if (next > oldest) {
    PetscCall(BFGSRecursiveBasisUpdate(B, mode));
    PetscCall(BFGSKernel_Recursive_Inner(B, mode, oldest, next, X, Y));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode BFGSCompactDenseProductsUpdate(Mat B, MatLMVMMode mode)
{
  Mat_LMVM              *lmvm = (Mat_LMVM *)B->data;
  Mat_SymBrdn           *lsb  = (Mat_SymBrdn *)lmvm->ctx;
  PetscInt               oldest, next, k;
  MatLMVMBasisType       S_t   = LMVMModeMap(LMBASIS_S, mode);
  MatLMVMBasisType       B0S_t = LMVMModeMap(LMBASIS_B0S, mode);
  MatLMVMBasisType       Y_t   = LMVMModeMap(LMBASIS_Y, mode);
  SymBroydenProductsType M00_t = LMVMModeMap(SYMBROYDEN_PRODUCTS_M00, mode);
  LMProducts             M00, StB0S, YtS, D;
  Mat                    YtS_local, StB0S_local, M00_local;
  Vec                    D_local;
  PetscBool              local_is_nonempty;

  PetscFunctionBegin;
  PetscCall(MatLMVMGetRange(B, &oldest, &next));
  if (lsb->products[M00_t] && lsb->products[M00_t]->block_type != LMBLOCK_FULL) PetscCall(LMProductsDestroy(&lsb->products[M00_t]));
  if (!lsb->products[M00_t]) PetscCall(MatLMVMCreateProducts(B, LMBLOCK_FULL, &lsb->products[M00_t]));
  M00 = lsb->products[M00_t];
  PetscCall(LMProductsPrepare(M00, lmvm->J0, oldest, next));
  PetscCall(LMProductsGetLocalMatrix(M00, &M00_local, &k, &local_is_nonempty));
  if (k < next) {
    PetscCall(MatLMVMGetUpdatedProducts(B, Y_t, S_t, LMBLOCK_STRICT_UPPER_TRIANGLE, &YtS));
    PetscCall(MatLMVMGetUpdatedProducts(B, LMBASIS_Y, LMBASIS_S, LMBLOCK_DIAGONAL, &D));
    PetscCall(MatLMVMGetUpdatedProducts(B, S_t, B0S_t, LMBLOCK_UPPER_TRIANGLE, &StB0S));

    PetscCall(LMProductsGetLocalMatrix(StB0S, &StB0S_local, NULL, NULL));
    PetscCall(LMProductsGetLocalMatrix(YtS, &YtS_local, NULL, NULL));
    PetscCall(LMProductsGetLocalDiagonal(D, &D_local));
    if (local_is_nonempty) {
      Vec invD;
      Mat stril_StY;

      PetscCall(MatSetUnfactored(M00_local));
      PetscCall(MatCopy(StB0S_local, M00_local, SAME_NONZERO_PATTERN));
      PetscCall(VecDuplicate(D_local, &invD));
      PetscCall(VecCopy(D_local, invD));
      PetscCall(VecReciprocal(invD));
      PetscCall(MatTranspose(YtS_local, MAT_INITIAL_MATRIX, &stril_StY));
      if (PetscDefined(USE_COMPLEX)) PetscCall(MatConjugate(stril_StY));

      PetscCall(MatDiagonalScale(stril_StY, NULL, invD));
      PetscCall(MatMatMult(stril_StY, YtS_local, MAT_REUSE_MATRIX, PETSC_DETERMINE, &M00_local));
      PetscCall(MatAXPY(M00_local, 1.0, StB0S_local, UNKNOWN_NONZERO_PATTERN));
      PetscCall(LMProductsMakeHermitian(M00_local, oldest, next));
      PetscCall(LMProductsOnesOnUnusedDiagonal(M00_local, oldest, next));
      PetscCall(MatSetOption(M00_local, MAT_HERMITIAN, PETSC_TRUE));
      PetscCall(MatSetOption(M00_local, MAT_SPD, PETSC_TRUE));
      PetscCall(MatCholeskyFactor(M00_local, NULL, NULL));
      PetscCall(MatDestroy(&stril_StY));
      PetscCall(VecDestroy(&invD));
    }
    PetscCall(LMProductsRestoreLocalDiagonal(D, &D_local));
    PetscCall(LMProductsRestoreLocalMatrix(YtS, &YtS_local, NULL));
    PetscCall(LMProductsRestoreLocalMatrix(StB0S, &StB0S_local, NULL));
  }
  PetscCall(LMProductsRestoreLocalMatrix(M00, &M00_local, &next));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode BFGSKernel_CompactDense(Mat B, MatLMVMMode mode, Vec X, Vec BX)
{
  PetscInt oldest, next;

  PetscFunctionBegin;
  PetscCall(MatLMVMApplyJ0Mode(mode)(B, X, BX));
  PetscCall(MatLMVMGetRange(B, &oldest, &next));
  if (next > oldest) {
    Mat_LMVM              *lmvm  = (Mat_LMVM *)B->data;
    Mat_SymBrdn           *bfgs  = (Mat_SymBrdn *)lmvm->ctx;
    MatLMVMBasisType       S_t   = LMVMModeMap(LMBASIS_S, mode);
    MatLMVMBasisType       Y_t   = LMVMModeMap(LMBASIS_Y, mode);
    MatLMVMBasisType       B0S_t = LMVMModeMap(LMBASIS_B0S, mode);
    SymBroydenProductsType M00_t = LMVMModeMap(SYMBROYDEN_PRODUCTS_M00, mode);
    LMBasis                S, Y;
    PetscBool              use_B0S;
    Vec                    YtX, StB0X, u, v;
    LMProducts             M00, YtS, D;

    PetscCall(BFGSCompactDenseProductsUpdate(B, mode));
    PetscCall(MatLMVMGetUpdatedBasis(B, S_t, &S, NULL, NULL));
    PetscCall(MatLMVMGetUpdatedBasis(B, Y_t, &Y, NULL, NULL));
    PetscCall(MatLMVMGetUpdatedProducts(B, Y_t, S_t, LMBLOCK_STRICT_UPPER_TRIANGLE, &YtS));
    PetscCall(MatLMVMGetUpdatedProducts(B, LMBASIS_Y, LMBASIS_S, LMBLOCK_DIAGONAL, &D));
    M00 = bfgs->products[M00_t];

    PetscCall(MatLMVMGetWorkRow(B, &YtX));
    PetscCall(MatLMVMGetWorkRow(B, &StB0X));
    PetscCall(MatLMVMGetWorkRow(B, &u));
    PetscCall(MatLMVMGetWorkRow(B, &v));

    PetscCall(LMBasisGEMVH(Y, oldest, next, 1.0, X, 0.0, YtX));
    PetscCall(SymBroydenCompactDenseKernelUseB0S(B, mode, X, &use_B0S));
    if (use_B0S) PetscCall(MatLMVMBasisGEMVH(B, B0S_t, oldest, next, 1.0, X, 0.0, StB0X));
    else PetscCall(LMBasisGEMVH(S, oldest, next, 1.0, BX, 0.0, StB0X));

    PetscCall(LMProductsSolve(D, oldest, next, YtX, YtX, /* ^H */ PETSC_FALSE));
    PetscCall(LMProductsMult(YtS, oldest, next, 1.0, YtX, 1.0, StB0X, /* ^H */ PETSC_TRUE));
    PetscCall(LMProductsSolve(M00, oldest, next, StB0X, u, PETSC_FALSE));
    PetscCall(VecScale(u, -1.0));
    PetscCall(LMProductsMult(YtS, oldest, next, 1.0, u, 0.0, v, /* ^H */ PETSC_FALSE));
    PetscCall(LMProductsSolve(D, oldest, next, v, v, PETSC_FALSE));
    PetscCall(VecAXPY(v, 1.0, YtX));

    PetscCall(LMBasisGEMV(Y, oldest, next, 1.0, v, 1.0, BX));
    PetscCall(MatLMVMBasisGEMV(B, B0S_t, oldest, next, 1.0, u, 1.0, BX));

    PetscCall(MatLMVMRestoreWorkRow(B, &v));
    PetscCall(MatLMVMRestoreWorkRow(B, &u));
    PetscCall(MatLMVMRestoreWorkRow(B, &StB0X));
    PetscCall(MatLMVMRestoreWorkRow(B, &YtX));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMult_LMVMBFGS_Recursive(Mat B, Vec X, Vec Y)
{
  PetscFunctionBegin;
  PetscCall(BFGSKernel_Recursive(B, MATLMVM_MODE_PRIMAL, X, Y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMult_LMVMBFGS_CompactDense(Mat B, Vec X, Vec Y)
{
  PetscFunctionBegin;
  PetscCall(BFGSKernel_CompactDense(B, MATLMVM_MODE_PRIMAL, X, Y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSolve_LMVMBFGS_Recursive(Mat B, Vec X, Vec HX)
{
  PetscFunctionBegin;
  PetscCall(DFPKernel_Recursive(B, MATLMVM_MODE_DUAL, X, HX));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSolve_LMVMBFGS_CompactDense(Mat B, Vec X, Vec HX)
{
  PetscFunctionBegin;
  PetscCall(DFPKernel_CompactDense(B, MATLMVM_MODE_DUAL, X, HX));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSolve_LMVMBFGS_Dense(Mat B, Vec X, Vec HX)
{
  PetscFunctionBegin;
  PetscCall(DFPKernel_Dense(B, MATLMVM_MODE_DUAL, X, HX));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetFromOptions_LMVMBFGS(Mat B, PetscOptionItems PetscOptionsObject)
{
  Mat_LMVM    *lmvm  = (Mat_LMVM *)B->data;
  Mat_SymBrdn *lbfgs = (Mat_SymBrdn *)lmvm->ctx;

  PetscFunctionBegin;
  PetscCall(MatSetFromOptions_LMVM(B, PetscOptionsObject));
  PetscOptionsHeadBegin(PetscOptionsObject, "L-BFGS method for approximating SPD Jacobian actions (MATLMVMBFGS)");
  PetscCall(SymBroydenRescaleSetFromOptions(B, lbfgs->rescale, PetscOptionsObject));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatLMVMSetMultAlgorithm_BFGS(Mat B)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;

  PetscFunctionBegin;
  switch (lmvm->mult_alg) {
  case MAT_LMVM_MULT_RECURSIVE:
    lmvm->ops->mult  = MatMult_LMVMBFGS_Recursive;
    lmvm->ops->solve = MatSolve_LMVMBFGS_Recursive;
    break;
  case MAT_LMVM_MULT_DENSE:
    lmvm->ops->mult  = MatMult_LMVMBFGS_CompactDense;
    lmvm->ops->solve = MatSolve_LMVMBFGS_Dense;
    break;
  case MAT_LMVM_MULT_COMPACT_DENSE:
    lmvm->ops->mult  = MatMult_LMVMBFGS_CompactDense;
    lmvm->ops->solve = MatSolve_LMVMBFGS_CompactDense;
    break;
  }
  lmvm->ops->multht  = lmvm->ops->mult;
  lmvm->ops->solveht = lmvm->ops->solve;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatCreate_LMVMBFGS(Mat B)
{
  Mat_LMVM    *lmvm;
  Mat_SymBrdn *lbfgs;

  PetscFunctionBegin;
  PetscCall(MatCreate_LMVMSymBrdn(B));
  PetscCall(PetscObjectChangeTypeName((PetscObject)B, MATLMVMBFGS));
  B->ops->setfromoptions = MatSetFromOptions_LMVMBFGS;

  lmvm                        = (Mat_LMVM *)B->data;
  lmvm->ops->setmultalgorithm = MatLMVMSetMultAlgorithm_BFGS;
  PetscCall(MatLMVMSetMultAlgorithm_BFGS(B));

  lbfgs = (Mat_SymBrdn *)lmvm->ctx;

  lbfgs->phi_scalar = 0.0;
  lbfgs->psi_scalar = 1.0;
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatLMVMSymBroydenSetPhi_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatCreateLMVMBFGS - Creates a limited-memory Broyden-Fletcher-Goldfarb-Shano (BFGS)
  matrix used for approximating Jacobians. L-BFGS is symmetric positive-definite by
  construction, and is commonly used to approximate Hessians in optimization
  problems.

  To use the L-BFGS matrix with other vector types, the matrix must be
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

.seealso: [](ch_ksp), `MatCreate()`, `MATLMVM`, `MATLMVMBFGS`, `MatCreateLMVMDFP()`, `MatCreateLMVMSR1()`,
          `MatCreateLMVMBroyden()`, `MatCreateLMVMBadBroyden()`, `MatCreateLMVMSymBroyden()`
@*/
PetscErrorCode MatCreateLMVMBFGS(MPI_Comm comm, PetscInt n, PetscInt N, Mat *B)
{
  PetscFunctionBegin;
  PetscCall(KSPInitializePackage());
  PetscCall(MatCreate(comm, B));
  PetscCall(MatSetSizes(*B, n, n, N, N));
  PetscCall(MatSetType(*B, MATLMVMBFGS));
  PetscCall(MatSetUp(*B));
  PetscFunctionReturn(PETSC_SUCCESS);
}
