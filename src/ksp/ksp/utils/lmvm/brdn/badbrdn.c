#include <../src/ksp/ksp/utils/lmvm/brdn/brdn.h> /*I "petscksp.h" I*/

// Bad Broyden is dual to Broyden: MatSolve routines are dual to Broyden MatMult routines

static PetscErrorCode MatSolve_LMVMBadBrdn_Recursive(Mat B, Vec F, Vec dX)
{
  PetscFunctionBegin;
  PetscCall(BroydenKernel_Recursive(B, MATLMVM_MODE_DUAL, F, dX));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSolve_LMVMBadBrdn_CompactDense(Mat B, Vec F, Vec dX)
{
  PetscFunctionBegin;
  PetscCall(BroydenKernel_CompactDense(B, MATLMVM_MODE_DUAL, F, dX));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_UNUSED static PetscErrorCode MatSolve_LMVMBadBrdn_Dense(Mat B, Vec F, Vec dX)
{
  PetscFunctionBegin;
  PetscCall(BroydenKernel_Dense(B, MATLMVM_MODE_DUAL, F, dX));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSolveHermitianTranspose_LMVMBadBrdn_Recursive(Mat B, Vec F, Vec dX)
{
  PetscFunctionBegin;
  PetscCall(BroydenKernelHermitianTranspose_Recursive(B, MATLMVM_MODE_DUAL, F, dX));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSolveHermitianTranspose_LMVMBadBrdn_CompactDense(Mat B, Vec F, Vec dX)
{
  PetscFunctionBegin;
  PetscCall(BroydenKernelHermitianTranspose_CompactDense(B, MATLMVM_MODE_DUAL, F, dX));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_UNUSED static PetscErrorCode MatSolveHermitianTranspose_LMVMBadBrdn_Dense(Mat B, Vec F, Vec dX)
{
  PetscFunctionBegin;
  PetscCall(BroydenKernelHermitianTranspose_Dense(B, MATLMVM_MODE_DUAL, F, dX));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   The bad Broyden kernel can be written as

     B_{k+1} x = B_k x + (y_k - B_k s_k)^T * (y_k^T B_k s_k)^{-1} y_k^T B_k x
               = (I + (y_k - B_k s_k)^T (y_k^T B_k s_k)^{-1} y_k^T) (B_k x)
                 \______________________ _________________________/
                                        V
                               recursive rank-1 update

   When the basis (y_k - B_k s_k) and the products (y_k^T B_k s_k) have been computed, the product can be computed by
   application of rank-1 updates from oldest to newest
 */

static PetscErrorCode BadBroydenKernel_Recursive_Inner(Mat B, MatLMVMMode mode, PetscInt oldest, PetscInt next, Vec B0X)
{
  Mat_LMVM        *lmvm        = (Mat_LMVM *)B->data;
  Mat_Brdn        *lbrdn       = (Mat_Brdn *)lmvm->ctx;
  MatLMVMBasisType Y_t         = LMVMModeMap(LMBASIS_Y, mode);
  LMBasis          Y_minus_BkS = lbrdn->basis[LMVMModeMap(BROYDEN_BASIS_Y_MINUS_BKS, mode)];
  LMBasis          Y;
  LMProducts       YtBkS = lbrdn->products[LMVMModeMap(BROYDEN_PRODUCTS_YTBKS, mode)];

  PetscFunctionBegin;
  PetscCall(MatLMVMGetUpdatedBasis(B, Y_t, &Y, NULL, NULL));
  // These cannot be parallelized, notice the data dependence
  for (PetscInt i = oldest; i < next; i++) {
    Vec         y_i, yimbisi;
    PetscScalar yitbix;
    PetscScalar yitbisi;

    PetscCall(LMBasisGetVecRead(Y, i, &y_i));
    PetscCall(VecDot(B0X, y_i, &yitbix));
    PetscCall(LMBasisRestoreVecRead(Y, i, &y_i));
    PetscCall(LMProductsGetDiagonalValue(YtBkS, i, &yitbisi));
    PetscCall(LMBasisGetVecRead(Y_minus_BkS, i, &yimbisi));
    PetscCall(VecAXPY(B0X, yitbix / yitbisi, yimbisi));
    PetscCall(LMBasisRestoreVecRead(Y_minus_BkS, i, &yimbisi));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   Compute the basis vectors (y_k - B_k s_k) and dot products (y_k^T B_k s_k) recursively
 */

static PetscErrorCode BadBroydenRecursiveBasisUpdate(Mat B, MatLMVMMode mode)
{
  Mat_LMVM           *lmvm  = (Mat_LMVM *)B->data;
  Mat_Brdn           *lbrdn = (Mat_Brdn *)lmvm->ctx;
  MatLMVMBasisType    Y_t   = LMVMModeMap(LMBASIS_Y, mode);
  MatLMVMBasisType    B0S_t = LMVMModeMap(LMBASIS_B0S, mode);
  LMBasis             Y_minus_BkS;
  LMProducts          YtBkS;
  BroydenBasisType    Y_minus_BkS_t = LMVMModeMap(BROYDEN_BASIS_Y_MINUS_BKS, mode);
  BroydenProductsType YtBkS_t       = LMVMModeMap(BROYDEN_PRODUCTS_YTBKS, mode);
  PetscInt            oldest, next;
  PetscInt            products_oldest;
  LMBasis             Y;
  PetscInt            start;

  PetscFunctionBegin;
  if (!lbrdn->basis[Y_minus_BkS_t]) PetscCall(LMBasisCreate(Y_t == LMBASIS_Y ? lmvm->Fprev : lmvm->Xprev, lmvm->m, &lbrdn->basis[Y_minus_BkS_t]));
  Y_minus_BkS = lbrdn->basis[Y_minus_BkS_t];
  if (!lbrdn->products[YtBkS_t]) PetscCall(MatLMVMCreateProducts(B, LMBLOCK_DIAGONAL, &lbrdn->products[YtBkS_t]));
  YtBkS = lbrdn->products[YtBkS_t];
  PetscCall(MatLMVMGetUpdatedBasis(B, Y_t, &Y, NULL, NULL));
  PetscCall(MatLMVMGetRange(B, &oldest, &next));
  // invalidate computed values if J0 has changed
  PetscCall(LMProductsPrepare(YtBkS, lmvm->J0, oldest, next));
  products_oldest = PetscMax(0, YtBkS->k - lmvm->m);
  if (oldest > products_oldest) {
    // recursion is starting from a different starting index, it must be recomputed
    YtBkS->k = oldest;
  }
  Y_minus_BkS->k = start = YtBkS->k;
  // recompute each column vector in Y_minus_BkS, and the product y_k^T B_k s_k, in order
  for (PetscInt j = start; j < next; j++) {
    Vec         p_j, y_j, B0s_j;
    PetscScalar yjtbjsj, alpha;

    PetscCall(LMBasisGetWorkVec(Y_minus_BkS, &p_j));

    // p_j starts as B_0 * s_j
    PetscCall(MatLMVMBasisGetVecRead(B, B0S_t, j, &B0s_j, &alpha));
    PetscCall(VecAXPBY(p_j, alpha, 0.0, B0s_j));
    PetscCall(MatLMVMBasisRestoreVecRead(B, B0S_t, j, &B0s_j, &alpha));

    /* Use the matmult kernel to compute p_j = B_j * p_j
       (if j == oldest, p_j is already correct) */
    if (j > oldest) PetscCall(BadBroydenKernel_Recursive_Inner(B, mode, oldest, j, p_j));
    PetscCall(LMBasisGetVecRead(Y, j, &y_j));
    PetscCall(VecDot(p_j, y_j, &yjtbjsj));
    PetscCall(LMProductsInsertNextDiagonalValue(YtBkS, j, yjtbjsj));
    PetscCall(VecAYPX(p_j, -1.0, y_j));
    PetscCall(LMBasisRestoreVecRead(Y, j, &y_j));
    PetscCall(LMBasisSetNextVec(Y_minus_BkS, p_j));
    PetscCall(LMBasisRestoreWorkVec(Y_minus_BkS, &p_j));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode BadBroydenKernel_Recursive(Mat B, MatLMVMMode mode, Vec X, Vec Y)
{
  PetscInt oldest, next;

  PetscFunctionBegin;
  PetscCall(MatLMVMApplyJ0Mode(mode)(B, X, Y));
  PetscCall(MatLMVMGetRange(B, &oldest, &next));
  if (next > oldest) {
    PetscCall(BadBroydenRecursiveBasisUpdate(B, mode));
    PetscCall(BadBroydenKernel_Recursive_Inner(B, mode, oldest, next, Y));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   The adjoint of the recursive bad Broyden kernel is

     B_{k+1}^T x = B_k^T (I + y_k (s_k^T B_k^T y_k)^{-1} (y_k - B_k s_k)^T) x
                         \____________________ ___________________________/
                                            V
                                recursive rank-1 update

    which can be computed by application of rank-1 updates from newest to oldest
 */

static PetscErrorCode BadBroydenKernelHermitianTranspose_Recursive_Inner(Mat B, MatLMVMMode mode, PetscInt oldest, PetscInt next, Vec X)
{
  MatLMVMBasisType    Y_t           = LMVMModeMap(LMBASIS_Y, mode);
  BroydenBasisType    Y_minus_BkS_t = LMVMModeMap(BROYDEN_BASIS_Y_MINUS_BKS, mode);
  BroydenProductsType YtBkS_t       = LMVMModeMap(BROYDEN_PRODUCTS_YTBKS, mode);
  Mat_LMVM           *lmvm          = (Mat_LMVM *)B->data;
  Mat_Brdn           *lbrdn         = (Mat_Brdn *)lmvm->ctx;
  LMBasis             Y_minus_BkS   = lbrdn->basis[Y_minus_BkS_t];
  LMBasis             Y;
  LMProducts          YtBkS = lbrdn->products[YtBkS_t];

  PetscFunctionBegin;
  PetscCall(MatLMVMGetUpdatedBasis(B, Y_t, &Y, NULL, NULL));
  // These cannot be parallelized, notice the data dependence
  for (PetscInt i = next - 1; i >= oldest; i--) {
    Vec         yimbisi, y_i;
    PetscScalar yimbisitx;
    PetscScalar yitbisi;

    PetscCall(LMBasisGetVecRead(Y_minus_BkS, i, &yimbisi));
    PetscCall(VecDot(X, yimbisi, &yimbisitx));
    PetscCall(LMBasisRestoreVecRead(Y_minus_BkS, i, &yimbisi));
    PetscCall(LMProductsGetDiagonalValue(YtBkS, i, &yitbisi));
    PetscCall(LMBasisGetVecRead(Y, i, &y_i));
    PetscCall(VecAXPY(X, yimbisitx / PetscConj(yitbisi), y_i));
    PetscCall(LMBasisRestoreVecRead(Y, i, &y_i));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode BadBroydenKernelHermitianTranspose_Recursive(Mat B, MatLMVMMode mode, Vec X, Vec BX)
{
  MatLMVMBasisType Y_t = LMVMModeMap(LMBASIS_Y, mode);
  PetscInt         oldest, next;
  Vec              G = X;
  LMBasis          Y;

  PetscFunctionBegin;
  PetscCall(MatLMVMGetUpdatedBasis(B, Y_t, &Y, NULL, NULL));
  PetscCall(MatLMVMGetRange(B, &oldest, &next));
  if (next > oldest) {
    PetscCall(LMBasisGetWorkVec(Y, &G));
    PetscCall(VecCopy(X, G));
    PetscCall(BadBroydenRecursiveBasisUpdate(B, mode));
    PetscCall(BadBroydenKernelHermitianTranspose_Recursive_Inner(B, mode, oldest, next, G));
  }
  PetscCall(MatLMVMApplyJ0HermitianTransposeMode(mode)(B, G, BX));
  if (next > oldest) PetscCall(LMBasisRestoreWorkVec(Y, &G));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   The bad Broyden kernel can be written as

     B_k = B_0 + (Y - B_0 S) (Y^H B_0 S - stril(Y^H Y))^{-1} Y^H B_0
         = (I + (Y - B_0 S) (Y^H B_0 S - stril(Y^H Y))^{-1} Y^H) B_0

   where stril is the strictly lower triangular component.  We compute and factorize
   the small matrix in order to apply a single rank m update
 */

static PetscErrorCode BadBroydenCompactProductsUpdate(Mat B, MatLMVMMode mode)
{
  MatLMVMBasisType    Y_t               = LMVMModeMap(LMBASIS_Y, mode);
  MatLMVMBasisType    B0S_t             = LMVMModeMap(LMBASIS_B0S, mode);
  BroydenProductsType YtB0S_minus_YtY_t = LMVMModeMap(BROYDEN_PRODUCTS_YTB0S_MINUS_YTY, mode);
  Mat_LMVM           *lmvm              = (Mat_LMVM *)B->data;
  Mat_Brdn           *lbrdn             = (Mat_Brdn *)lmvm->ctx;
  LMProducts          YtB0S, YtY, YtB0S_minus_YtY;
  LMBasis             Y, B0S;
  PetscScalar         alpha;
  PetscInt            oldest, k, next;
  PetscBool           local_is_nonempty;
  Mat                 ytbsmyty_local;

  PetscFunctionBegin;
  if (!lbrdn->products[YtB0S_minus_YtY_t]) PetscCall(MatLMVMCreateProducts(B, LMBLOCK_FULL, &lbrdn->products[YtB0S_minus_YtY_t]));
  YtB0S_minus_YtY = lbrdn->products[YtB0S_minus_YtY_t];
  PetscCall(MatLMVMGetUpdatedBasis(B, Y_t, &Y, NULL, NULL));
  PetscCall(MatLMVMGetUpdatedBasis(B, B0S_t, &B0S, &B0S_t, &alpha));
  PetscCall(MatLMVMGetRange(B, &oldest, &next));
  // invalidate computed values if J0 has changed
  PetscCall(LMProductsPrepare(YtB0S_minus_YtY, lmvm->J0, oldest, next));
  PetscCall(LMProductsGetLocalMatrix(YtB0S_minus_YtY, &ytbsmyty_local, &k, &local_is_nonempty));
  if (k < next) {
    Mat yty_local, ytbs_local;

    PetscCall(MatLMVMGetUpdatedProducts(B, Y_t, B0S_t, LMBLOCK_FULL, &YtB0S));
    PetscCall(MatLMVMGetUpdatedProducts(B, Y_t, Y_t, LMBLOCK_STRICT_UPPER_TRIANGLE, &YtY));
    PetscCall(LMProductsGetLocalMatrix(YtB0S, &ytbs_local, NULL, NULL));
    PetscCall(LMProductsGetLocalMatrix(YtY, &yty_local, NULL, NULL));
    if (local_is_nonempty) {
      PetscCall(MatSetUnfactored(ytbsmyty_local));
      PetscCall(MatCopy(yty_local, ytbsmyty_local, SAME_NONZERO_PATTERN));
      PetscCall(MatTranspose(ytbsmyty_local, MAT_INPLACE_MATRIX, &ytbsmyty_local));
      if (PetscDefined(USE_COMPLEX)) PetscCall(MatConjugate(ytbsmyty_local));
      PetscCall(MatScale(ytbsmyty_local, -1.0));
      PetscCall(MatAXPY(ytbsmyty_local, alpha, ytbs_local, SAME_NONZERO_PATTERN));
      PetscCall(LMProductsOnesOnUnusedDiagonal(ytbsmyty_local, oldest, next));
      PetscCall(MatLUFactor(ytbsmyty_local, NULL, NULL, NULL));
    }
    PetscCall(LMProductsRestoreLocalMatrix(YtY, &yty_local, NULL));
    PetscCall(LMProductsRestoreLocalMatrix(YtB0S, &ytbs_local, NULL));
  }
  PetscCall(LMProductsRestoreLocalMatrix(YtB0S_minus_YtY, &ytbsmyty_local, &next));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode BadBroydenKernel_CompactDense(Mat B, MatLMVMMode mode, Vec X, Vec BX)
{
  PetscInt oldest, next;

  PetscFunctionBegin;
  PetscCall(MatLMVMApplyJ0Mode(mode)(B, X, BX));
  PetscCall(MatLMVMGetRange(B, &oldest, &next));
  if (next > oldest) {
    Mat_LMVM           *lmvm              = (Mat_LMVM *)B->data;
    Mat_Brdn           *lbrdn             = (Mat_Brdn *)lmvm->ctx;
    MatLMVMBasisType    Y_t               = LMVMModeMap(LMBASIS_Y, mode);
    MatLMVMBasisType    Y_minus_B0S_t     = LMVMModeMap(LMBASIS_Y_MINUS_B0S, mode);
    BroydenProductsType YtB0S_minus_YtY_t = LMVMModeMap(BROYDEN_PRODUCTS_YTB0S_MINUS_YTY, mode);
    LMProducts          YtB0S_minus_YtY;
    LMBasis             Y;
    Vec                 YtB0X, v;

    PetscCall(BadBroydenCompactProductsUpdate(B, mode));
    PetscCall(MatLMVMGetUpdatedBasis(B, Y_t, &Y, NULL, NULL));
    YtB0S_minus_YtY = lbrdn->products[YtB0S_minus_YtY_t];
    PetscCall(MatLMVMGetWorkRow(B, &YtB0X));
    PetscCall(MatLMVMGetWorkRow(B, &v));
    PetscCall(LMBasisGEMVH(Y, oldest, next, 1.0, BX, 0.0, YtB0X));
    PetscCall(LMProductsSolve(YtB0S_minus_YtY, oldest, next, YtB0X, v, /* ^H */ PETSC_FALSE));
    PetscCall(MatLMVMBasisGEMV(B, Y_minus_B0S_t, oldest, next, 1.0, v, 1.0, BX));
    PetscCall(MatLMVMRestoreWorkRow(B, &v));
    PetscCall(MatLMVMRestoreWorkRow(B, &YtB0X));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   The adjoint of the above formula for the bad Broyden kernel is

     B_k^H = B_0^H + B_0^H Y (Y^H B_0 S - stril(Y^H Y))^{-H} (Y - B_0 S)^H
           = B_0^H (I + Y (Y^H B_0 S - stril(Y^H Y))^{-H} (Y - B_0 S)^H)
 */

PETSC_INTERN PetscErrorCode BadBroydenKernelHermitianTranspose_CompactDense(Mat B, MatLMVMMode mode, Vec X, Vec BHX)
{
  MatLMVMBasisType Y_t = LMVMModeMap(LMBASIS_Y, mode);
  PetscInt         oldest, next;
  Vec              G = X;
  LMBasis          Y = NULL;

  PetscFunctionBegin;
  PetscCall(MatLMVMGetRange(B, &oldest, &next));
  if (next > oldest) {
    Mat_LMVM           *lmvm              = (Mat_LMVM *)B->data;
    Mat_Brdn           *lbrdn             = (Mat_Brdn *)lmvm->ctx;
    MatLMVMBasisType    Y_minus_B0S_t     = LMVMModeMap(LMBASIS_Y_MINUS_B0S, mode);
    BroydenProductsType YtB0S_minus_YtY_t = LMVMModeMap(BROYDEN_PRODUCTS_YTB0S_MINUS_YTY, mode);
    LMProducts          YtB0S_minus_YtY;
    Vec                 YmB0StG, v;

    PetscCall(MatLMVMGetUpdatedBasis(B, Y_t, &Y, NULL, NULL));
    PetscCall(LMBasisGetWorkVec(Y, &G));
    PetscCall(VecCopy(X, G));
    PetscCall(BadBroydenCompactProductsUpdate(B, mode));
    YtB0S_minus_YtY = lbrdn->products[YtB0S_minus_YtY_t];
    PetscCall(MatLMVMGetWorkRow(B, &YmB0StG));
    PetscCall(MatLMVMGetWorkRow(B, &v));
    PetscCall(MatLMVMBasisGEMVH(B, Y_minus_B0S_t, oldest, next, 1.0, G, 0.0, YmB0StG));
    PetscCall(LMProductsSolve(YtB0S_minus_YtY, oldest, next, YmB0StG, v, /* ^H */ PETSC_TRUE));
    PetscCall(LMBasisGEMV(Y, oldest, next, 1.0, v, 1.0, G));
    PetscCall(MatLMVMRestoreWorkRow(B, &v));
    PetscCall(MatLMVMRestoreWorkRow(B, &YmB0StG));
  }
  PetscCall(MatLMVMApplyJ0HermitianTransposeMode(mode)(B, G, BHX));
  if (next > oldest) PetscCall(LMBasisRestoreWorkVec(Y, &G));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMult_LMVMBadBrdn_Recursive(Mat B, Vec F, Vec dX)
{
  PetscFunctionBegin;
  PetscCall(BadBroydenKernel_Recursive(B, MATLMVM_MODE_PRIMAL, F, dX));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMult_LMVMBadBrdn_CompactDense(Mat B, Vec F, Vec dX)
{
  PetscFunctionBegin;
  PetscCall(BadBroydenKernel_CompactDense(B, MATLMVM_MODE_PRIMAL, F, dX));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMultHermitianTranspose_LMVMBadBrdn_Recursive(Mat B, Vec F, Vec dX)
{
  PetscFunctionBegin;
  PetscCall(BadBroydenKernelHermitianTranspose_Recursive(B, MATLMVM_MODE_PRIMAL, F, dX));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMultHermitianTranspose_LMVMBadBrdn_CompactDense(Mat B, Vec F, Vec dX)
{
  PetscFunctionBegin;
  PetscCall(BadBroydenKernelHermitianTranspose_CompactDense(B, MATLMVM_MODE_PRIMAL, F, dX));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatLMVMSetMultAlgorithm_BadBrdn(Mat B)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;

  PetscFunctionBegin;
  switch (lmvm->mult_alg) {
  case MAT_LMVM_MULT_RECURSIVE:
    lmvm->ops->mult    = MatMult_LMVMBadBrdn_Recursive;
    lmvm->ops->multht  = MatMultHermitianTranspose_LMVMBadBrdn_Recursive;
    lmvm->ops->solve   = MatSolve_LMVMBadBrdn_Recursive;
    lmvm->ops->solveht = MatSolveHermitianTranspose_LMVMBadBrdn_Recursive;
    break;
  case MAT_LMVM_MULT_DENSE:
    lmvm->ops->mult    = MatMult_LMVMBadBrdn_CompactDense;
    lmvm->ops->multht  = MatMultHermitianTranspose_LMVMBadBrdn_CompactDense;
    lmvm->ops->solve   = MatSolve_LMVMBadBrdn_Dense;
    lmvm->ops->solveht = MatSolveHermitianTranspose_LMVMBadBrdn_Dense;
    break;
  case MAT_LMVM_MULT_COMPACT_DENSE:
    lmvm->ops->mult    = MatMult_LMVMBadBrdn_CompactDense;
    lmvm->ops->multht  = MatMultHermitianTranspose_LMVMBadBrdn_CompactDense;
    lmvm->ops->solve   = MatSolve_LMVMBadBrdn_CompactDense;
    lmvm->ops->solveht = MatSolveHermitianTranspose_LMVMBadBrdn_CompactDense;
    break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatCreate_LMVMBadBrdn(Mat B)
{
  Mat_LMVM *lmvm;

  PetscFunctionBegin;
  PetscCall(MatCreate_LMVMBrdn(B));
  PetscCall(PetscObjectChangeTypeName((PetscObject)B, MATLMVMBADBROYDEN));
  lmvm                          = (Mat_LMVM *)B->data;
  lmvm->ops->setmultalgorithm   = MatLMVMSetMultAlgorithm_BadBrdn;
  lmvm->cache_gradient_products = PETSC_TRUE;
  PetscCall(MatLMVMSetMultAlgorithm_BadBrdn(B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatCreateLMVMBadBroyden - Creates a limited-memory modified (aka "bad") Broyden-type
  approximation matrix used for a Jacobian. L-BadBrdn is not guaranteed to be
  symmetric or positive-definite.

  To use the L-BadBrdn matrix with other vector types, the matrix must be
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
+ -mat_lmvm_hist_size         - the number of history vectors to keep
. -mat_lmvm_mult_algorithm    - the algorithm to use for multiplication (recursive, dense, compact_dense)
. -mat_lmvm_cache_J0_products - whether products between the base Jacobian J0 and history vectors should be cached or recomputed
- -mat_lmvm_debug             - (developer) perform internal debugging checks

  Level: intermediate

  Note:
  It is recommended that one use the `MatCreate()`, `MatSetType()` and/or `MatSetFromOptions()`
  paradigm instead of this routine directly.

.seealso: [](ch_ksp), `MatCreate()`, `MATLMVM`, `MATLMVMBADBRDN`, `MatCreateLMVMDFP()`, `MatCreateLMVMSR1()`,
          `MatCreateLMVMBFGS()`, `MatCreateLMVMBroyden()`, `MatCreateLMVMSymBroyden()`
@*/
PetscErrorCode MatCreateLMVMBadBroyden(MPI_Comm comm, PetscInt n, PetscInt N, Mat *B)
{
  PetscFunctionBegin;
  PetscCall(KSPInitializePackage());
  PetscCall(MatCreate(comm, B));
  PetscCall(MatSetSizes(*B, n, n, N, N));
  PetscCall(MatSetType(*B, MATLMVMBADBROYDEN));
  PetscCall(MatSetUp(*B));
  PetscFunctionReturn(PETSC_SUCCESS);
}
