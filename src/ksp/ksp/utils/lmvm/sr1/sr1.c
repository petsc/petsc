#include <../src/ksp/ksp/utils/lmvm/lmvm.h> /*I "petscksp.h" I*/

/*
  Limited-memory Symmetric-Rank-1 method for approximating both
  the forward product and inverse application of a Jacobian.
*/

// bases needed by SR1 algorithms beyond those in Mat_LMVM
enum {
  SR1_BASIS_Y_MINUS_BKS = 0, // Y_k - B_k S_k for recursive algorithms
  SR1_BASIS_S_MINUS_HKY = 1, // dual to the above, S_k - H_k Y_k
  SR1_BASIS_COUNT
};

typedef PetscInt SR1BasisType;

// products needed by SR1 algorithms beyond those in Mat_LMVM
enum {
  SR1_PRODUCTS_YTS_MINUS_STB0S = 0, // stores and factors symm(triu((Y - B_0 S)^T S)) for compact algorithms
  SR1_PRODUCTS_STY_MINUS_YTH0Y = 1, // dual to the above, stores and factors symm(triu((S - H_0 Y)^T Y))
  SR1_PRODUCTS_YTS_MINUS_STBKS = 2, // diagonal (Y_k - B_k S_k)^T S_k values for recursive algorithms
  SR1_PRODUCTS_STY_MINUS_YTHKY = 3, // dual to the above, diagonal (S_k - H_k Y_k)^T Y_k
  SR1_PRODUCTS_COUNT
};

typedef PetscInt SR1ProductsType;

typedef struct {
  LMBasis    basis[SR1_BASIS_COUNT];
  LMProducts products[SR1_PRODUCTS_COUNT];
  Vec        StFprev, SmH0YtFprev;
} Mat_LSR1;

/* The SR1 kernel can be written as

     B_{k+1} = B_k + (y_k - B_k s_k) ((y_k - B_k s_k)^T s_k)^{-1} (y_k - B_k s_k)^T

   this unrolls to a rank-m update

     B_{k+1} = B_0 + \sum_{i = k-m+1}^k (y_i - B_i s_i) ((y_i - B_i s_i)^T s_i)^{-1} (y_i - B_i s_i)^T

   This inner kernel assumes the (y_i - B_i s_i) vectors and the ((y_i - B_i s_i)^T s_i) products have been computed
 */

static PetscErrorCode SR1Kernel_Recursive_Inner(Mat B, MatLMVMMode mode, PetscInt oldest, PetscInt next, Vec X, Vec BX)
{
  Mat_LMVM       *lmvm              = (Mat_LMVM *)B->data;
  Mat_LSR1       *lsr1              = (Mat_LSR1 *)lmvm->ctx;
  SR1BasisType    Y_minus_BkS_t     = LMVMModeMap(SR1_BASIS_Y_MINUS_BKS, mode);
  SR1ProductsType YtS_minus_StBkS_t = LMVMModeMap(SR1_PRODUCTS_STY_MINUS_YTHKY, mode);
  LMBasis         Y_minus_BkS       = lsr1->basis[Y_minus_BkS_t];
  LMProducts      YtS_minus_StBkS   = lsr1->products[YtS_minus_StBkS_t];
  Vec             YmBkStX;

  PetscFunctionBegin;
  PetscCall(MatLMVMGetWorkRow(B, &YmBkStX));
  PetscCall(LMBasisGEMVH(Y_minus_BkS, oldest, next, 1.0, X, 0.0, YmBkStX));
  PetscCall(LMProductsSolve(YtS_minus_StBkS, oldest, next, YmBkStX, YmBkStX, /* ^H */ PETSC_FALSE));
  PetscCall(LMBasisGEMV(Y_minus_BkS, oldest, next, 1.0, YmBkStX, 1.0, BX));
  PetscCall(MatLMVMRestoreWorkRow(B, &YmBkStX));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Recursively compute the (y_i - B_i s_i) vectors and ((y_i - B_i s_i)^T s_i) products */

static PetscErrorCode SR1RecursiveBasisUpdate(Mat B, MatLMVMMode mode)
{
  Mat_LMVM        *lmvm              = (Mat_LMVM *)B->data;
  Mat_LSR1        *lsr1              = (Mat_LSR1 *)lmvm->ctx;
  MatLMVMBasisType B0S_t             = LMVMModeMap(LMBASIS_B0S, mode);
  MatLMVMBasisType S_t               = LMVMModeMap(LMBASIS_S, mode);
  MatLMVMBasisType Y_t               = LMVMModeMap(LMBASIS_Y, mode);
  SR1BasisType     Y_minus_BkS_t     = LMVMModeMap(SR1_BASIS_Y_MINUS_BKS, mode);
  SR1ProductsType  YtS_minus_StBkS_t = LMVMModeMap(SR1_PRODUCTS_STY_MINUS_YTHKY, mode);
  LMBasis          Y_minus_BkS;
  LMProducts       YtS_minus_StBkS;
  PetscInt         oldest, next;
  PetscInt         products_oldest;
  LMBasis          S, Y;
  PetscInt         start;

  PetscFunctionBegin;
  if (!lsr1->basis[Y_minus_BkS_t]) PetscCall(LMBasisCreate(mode == MATLMVM_MODE_PRIMAL ? lmvm->Fprev : lmvm->Xprev, lmvm->m, &lsr1->basis[Y_minus_BkS_t]));
  Y_minus_BkS = lsr1->basis[Y_minus_BkS_t];
  if (!lsr1->products[YtS_minus_StBkS_t]) PetscCall(MatLMVMCreateProducts(B, LMBLOCK_DIAGONAL, &lsr1->products[YtS_minus_StBkS_t]));
  YtS_minus_StBkS = lsr1->products[YtS_minus_StBkS_t];
  PetscCall(MatLMVMGetUpdatedBasis(B, S_t, &S, NULL, NULL));
  PetscCall(MatLMVMGetUpdatedBasis(B, Y_t, &Y, NULL, NULL));
  PetscCall(MatLMVMGetRange(B, &oldest, &next));
  // invalidate computed values if J0 has changed
  PetscCall(LMProductsPrepare(YtS_minus_StBkS, lmvm->J0, oldest, next));
  products_oldest = PetscMax(0, YtS_minus_StBkS->k - lmvm->m);
  if (oldest > products_oldest) {
    // recursion is starting from a different starting index, it must be recomputed
    YtS_minus_StBkS->k = oldest;
  }
  Y_minus_BkS->k = start = YtS_minus_StBkS->k;
  // recompute each column in Y_minus_BkS in order
  for (PetscInt j = start; j < next; j++) {
    Vec         s_j, B0s_j, p_j, y_j;
    PetscScalar alpha, ymbksts;

    PetscCall(LMBasisGetWorkVec(Y_minus_BkS, &p_j));

    // p_j starts as B_0 * s_j
    PetscCall(MatLMVMBasisGetVecRead(B, B0S_t, j, &B0s_j, &alpha));
    PetscCall(VecAXPBY(p_j, alpha, 0.0, B0s_j));
    PetscCall(MatLMVMBasisRestoreVecRead(B, B0S_t, j, &B0s_j, &alpha));

    // Use the matmult kernel to compute p_j = B_j * p_j
    PetscCall(LMBasisGetVecRead(S, j, &s_j));
    // if j == oldest p_j is already correct
    if (j > oldest) PetscCall(SR1Kernel_Recursive_Inner(B, mode, oldest, j, s_j, p_j));
    PetscCall(LMBasisGetVecRead(Y, j, &y_j));
    PetscCall(VecAYPX(p_j, -1.0, y_j));
    PetscCall(VecDot(s_j, p_j, &ymbksts));
    PetscCall(LMProductsInsertNextDiagonalValue(YtS_minus_StBkS, j, ymbksts));
    PetscCall(LMBasisRestoreVecRead(S, j, &s_j));
    PetscCall(LMBasisRestoreVecRead(Y, j, &y_j));
    PetscCall(LMBasisSetNextVec(Y_minus_BkS, p_j));
    PetscCall(LMBasisRestoreWorkVec(Y_minus_BkS, &p_j));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SR1Kernel_Recursive(Mat B, MatLMVMMode mode, Vec X, Vec BX)
{
  PetscInt oldest, next;

  PetscFunctionBegin;
  PetscCall(MatLMVMApplyJ0Mode(mode)(B, X, BX));
  PetscCall(MatLMVMGetRange(B, &oldest, &next));
  if (next > oldest) {
    PetscCall(SR1RecursiveBasisUpdate(B, mode));
    PetscCall(SR1Kernel_Recursive_Inner(B, mode, oldest, next, X, BX));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* The SR1 kernel can be written as (See Byrd, Schnabel & Nocedal 1994)

     B_{k+1} = B_0 + (Y - B_0 S) (diag(S^T Y) + stril(S^T Y) + stril(S^T Y)^T - S^T B_0 S)^{-1} (Y - B_0 S)^T
                                 \___________________________ ___________________________/
                                                             V
                                                             M

   M is symmetric indefinite (stril is the strictly lower triangular part)

   M can be computed by computed triu((Y - B_0 S)^T S) and filling in the lower triangle
 */

static PetscErrorCode SR1CompactProductsUpdate(Mat B, MatLMVMMode mode)
{
  Mat_LMVM        *lmvm              = (Mat_LMVM *)B->data;
  Mat_LSR1        *lsr1              = (Mat_LSR1 *)lmvm->ctx;
  MatLMVMBasisType S_t               = LMVMModeMap(LMBASIS_S, mode);
  MatLMVMBasisType YmB0S_t           = LMVMModeMap(LMBASIS_Y_MINUS_B0S, mode);
  SR1ProductsType  YtS_minus_StB0S_t = LMVMModeMap(SR1_PRODUCTS_YTS_MINUS_STB0S, mode);
  LMProducts       YtS_minus_StB0S;
  Mat              local;
  PetscInt         oldest, next, k;
  PetscBool        local_is_nonempty;

  PetscFunctionBegin;
  if (!lsr1->products[YtS_minus_StB0S_t]) PetscCall(MatLMVMCreateProducts(B, LMBLOCK_FULL, &lsr1->products[YtS_minus_StB0S_t]));
  YtS_minus_StB0S = lsr1->products[YtS_minus_StB0S_t];
  PetscCall(MatLMVMGetRange(B, &oldest, &next));
  PetscCall(LMProductsPrepare(YtS_minus_StB0S, lmvm->J0, oldest, next));
  PetscCall(LMProductsGetLocalMatrix(YtS_minus_StB0S, &local, &k, &local_is_nonempty));
  if (YtS_minus_StB0S->k < next) {
    // copy to factor in place
    LMProducts YmB0StS;
    Mat        ymb0sts_local;

    PetscCall(PetscCitationsRegister(ByrdNocedalSchnabelCitation, &ByrdNocedalSchnabelCite));
    YtS_minus_StB0S->k = next;
    PetscCall(MatLMVMGetUpdatedProducts(B, YmB0S_t, S_t, LMBLOCK_UPPER_TRIANGLE, &YmB0StS));
    PetscCall(LMProductsGetLocalMatrix(YmB0StS, &ymb0sts_local, NULL, NULL));
    if (local_is_nonempty) {
      PetscErrorCode ierr;

      PetscCall(MatSetUnfactored(local));
      PetscCall(MatCopy(ymb0sts_local, local, SAME_NONZERO_PATTERN));
      PetscCall(LMProductsMakeHermitian(local, oldest, next));
      PetscCall(LMProductsOnesOnUnusedDiagonal(local, oldest, next));
      PetscCall(MatSetOption(local, MAT_HERMITIAN, PETSC_TRUE));
      // Set not spd so that "Cholesky" factorization is actually the symmetric indefinite Bunch Kaufman factorization
      PetscCall(MatSetOption(local, MAT_SPD, PETSC_FALSE));

      PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler, NULL));
      ierr = MatCholeskyFactor(local, NULL, NULL);
      PetscCall(PetscPopErrorHandler());
      PetscCheck(ierr == PETSC_SUCCESS || ierr == PETSC_ERR_SUP, PETSC_COMM_SELF, ierr, "Error in Bunch-Kaufman factorization");
      // cusolver does not provide Bunch Kaufman, resort to LU if it is unavailable
      if (ierr == PETSC_ERR_SUP) PetscCall(MatLUFactor(local, NULL, NULL, NULL));
    }
    PetscCall(LMProductsRestoreLocalMatrix(YmB0StS, &ymb0sts_local, NULL));
  }
  PetscCall(LMProductsRestoreLocalMatrix(YtS_minus_StB0S, &local, &next));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SR1Kernel_CompactDense(Mat B, MatLMVMMode mode, Vec X, Vec BX)
{
  PetscInt oldest, next;

  PetscFunctionBegin;
  PetscCall(MatLMVMApplyJ0Mode(mode)(B, X, BX));
  PetscCall(MatLMVMGetRange(B, &oldest, &next));
  if (next > oldest) {
    Mat_LMVM        *lmvm              = (Mat_LMVM *)B->data;
    Mat_LSR1        *lsr1              = (Mat_LSR1 *)lmvm->ctx;
    MatLMVMBasisType Y_minus_B0S_t     = LMVMModeMap(LMBASIS_Y_MINUS_B0S, mode);
    SR1ProductsType  YtS_minus_StB0S_t = LMVMModeMap(SR1_PRODUCTS_YTS_MINUS_STB0S, mode);
    LMProducts       YtS_minus_StB0S;
    Vec              YmB0StX, v;

    PetscCall(SR1CompactProductsUpdate(B, mode));
    YtS_minus_StB0S = lsr1->products[YtS_minus_StB0S_t];
    PetscCall(MatLMVMGetWorkRow(B, &YmB0StX));
    PetscCall(MatLMVMGetWorkRow(B, &v));
    if (lmvm->do_not_cache_J0_products) {
      /* the initial (Y - B_0 S)^T x inner product can be computed as Y^T x - S^T (B_0 x)
         if we are not caching B_0 S products */
      MatLMVMBasisType S_t = LMVMModeMap(LMBASIS_S, mode);
      MatLMVMBasisType Y_t = LMVMModeMap(LMBASIS_Y, mode);
      LMBasis          S, Y;

      PetscCall(MatLMVMGetUpdatedBasis(B, S_t, &S, NULL, NULL));
      PetscCall(MatLMVMGetUpdatedBasis(B, Y_t, &Y, NULL, NULL));
      PetscCall(LMBasisGEMVH(Y, oldest, next, 1.0, X, 0.0, YmB0StX));
      PetscCall(LMBasisGEMVH(S, oldest, next, -1.0, BX, 1.0, YmB0StX));
    } else PetscCall(MatLMVMBasisGEMVH(B, Y_minus_B0S_t, oldest, next, 1.0, X, 0.0, YmB0StX));
    PetscCall(LMProductsSolve(YtS_minus_StB0S, oldest, next, YmB0StX, v, PETSC_FALSE));
    PetscCall(MatLMVMBasisGEMV(B, Y_minus_B0S_t, oldest, next, 1.0, v, 1.0, BX));
    PetscCall(MatLMVMRestoreWorkRow(B, &v));
    PetscCall(MatLMVMRestoreWorkRow(B, &YmB0StX));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMult_LMVMSR1_CompactDense(Mat B, Vec X, Vec BX)
{
  PetscFunctionBegin;
  PetscCall(SR1Kernel_CompactDense(B, MATLMVM_MODE_PRIMAL, X, BX));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSolve_LMVMSR1_CompactDense(Mat B, Vec X, Vec BX)
{
  PetscFunctionBegin;
  PetscCall(SR1Kernel_CompactDense(B, MATLMVM_MODE_DUAL, X, BX));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMult_LMVMSR1_Recursive(Mat B, Vec X, Vec Z)
{
  PetscFunctionBegin;
  PetscCall(SR1Kernel_Recursive(B, MATLMVM_MODE_PRIMAL, X, Z));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSolve_LMVMSR1_Recursive(Mat B, Vec F, Vec dX)
{
  PetscFunctionBegin;
  PetscCall(SR1Kernel_Recursive(B, MATLMVM_MODE_DUAL, F, dX));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatUpdate_LMVMSR1(Mat B, Vec X, Vec F)
{
  Mat_LMVM *lmvm          = (Mat_LMVM *)B->data;
  Mat_LSR1 *sr1           = (Mat_LSR1 *)lmvm->ctx;
  PetscBool cache_SmH0YtF = (lmvm->mult_alg != MAT_LMVM_MULT_RECURSIVE && !lmvm->do_not_cache_J0_products) ? lmvm->cache_gradient_products : PETSC_FALSE;

  PetscFunctionBegin;
  if (!lmvm->m) PetscFunctionReturn(PETSC_SUCCESS);
  if (lmvm->prev_set) {
    PetscReal   snorm, pnorm;
    PetscScalar sktw;
    Vec         work;
    Vec         Fprev_old       = NULL;
    Vec         SmH0YtFprev_old = NULL;
    LMProducts  SmH0YtY         = NULL;
    PetscInt    oldest, next;
    LMBasis     SmH0Y = NULL;
    LMBasis     Y;

    PetscCall(MatLMVMGetRange(B, &oldest, &next));
    if (cache_SmH0YtF) {
      PetscCall(MatLMVMGetUpdatedBasis(B, LMBASIS_S_MINUS_H0Y, &SmH0Y, NULL, NULL));
      if (!sr1->SmH0YtFprev) PetscCall(LMBasisCreateRow(SmH0Y, &sr1->SmH0YtFprev));
      PetscCall(LMBasisGetWorkVec(SmH0Y, &Fprev_old));
      PetscCall(MatLMVMGetUpdatedProducts(B, LMBASIS_S_MINUS_H0Y, LMBASIS_Y, LMBLOCK_UPPER_TRIANGLE, &SmH0YtY));
      PetscCall(LMProductsGetNextColumn(SmH0YtY, &SmH0YtFprev_old));
      PetscCall(VecCopy(lmvm->Fprev, Fprev_old));
      if (sr1->SmH0YtFprev == SmH0Y->cached_product) {
        PetscCall(VecCopy(sr1->SmH0YtFprev, SmH0YtFprev_old));
      } else {
        if (next > oldest) {
          // need to recalculate
          PetscCall(LMBasisGEMVH(SmH0Y, oldest, next, 1.0, Fprev_old, 0.0, SmH0YtFprev_old));
        } else {
          PetscCall(VecZeroEntries(SmH0YtFprev_old));
        }
      }
    }

    /* Compute the new (S = X - Xprev) and (Y = F - Fprev) vectors */
    PetscCall(VecAYPX(lmvm->Xprev, -1.0, X));
    PetscCall(VecAYPX(lmvm->Fprev, -1.0, F));

    /* See if the updates can be accepted
       NOTE: This tests abs(S[k]^T (Y[k] - B_k*S[k])) >= eps * norm(S[k]) * norm(Y[k] - B_k*S[k])

       Note that this test is flawed because this is a **limited memory** SR1 method: we are testing

         abs(S[k]^T (Y[k] - B_{k,m}*S[k])) >= eps * norm(S[k]) * norm(Y[k] - B_{k,m}*S[k])

       when the oldest pair of vectors in the definition of B_{k,m}, (s_{k-m}, y_{k-m}), will be dropped if we add a new
       pair.  To really ensure that B_{k+1} = B_{k+1,m} is nonsingular, you need to test

         abs(S[k]^T (Y[k] - B_{k,m-1}*S[k])) >= eps * norm(S[k]) * norm(Y[k] - B_{k,m-1}*S[k])

       But the product B_{k,m-1}*S[k] is not readily computable (see e.g. Lu, Xuehua, "A study of the limited memory SR1
       method in practice", 1996).
     */
    PetscCall(MatLMVMGetUpdatedBasis(B, LMBASIS_Y, &Y, NULL, NULL));
    PetscCall(LMBasisGetWorkVec(Y, &work));
    PetscCall(MatMult(B, lmvm->Xprev, work));
    PetscCall(VecAYPX(work, -1.0, lmvm->Fprev));
    PetscCall(VecDot(lmvm->Xprev, work, &sktw));
    PetscCall(VecNorm(lmvm->Xprev, NORM_2, &snorm));
    PetscCall(VecNorm(work, NORM_2, &pnorm));
    PetscCall(LMBasisRestoreWorkVec(Y, &work));
    if (PetscAbsReal(PetscRealPart(sktw)) >= lmvm->eps * snorm * pnorm) {
      /* Update is good, accept it */
      PetscCall(MatUpdateKernel_LMVM(B, lmvm->Xprev, lmvm->Fprev));
      if (cache_SmH0YtF) {
        PetscInt oldest_new, next_new;

        PetscCall(MatLMVMGetUpdatedBasis(B, LMBASIS_S_MINUS_H0Y, &SmH0Y, NULL, NULL));
        PetscCall(MatLMVMGetRange(B, &oldest_new, &next_new));
        PetscCall(LMBasisGEMVH(SmH0Y, next, next_new, 1.0, Fprev_old, 0.0, SmH0YtFprev_old));
        PetscCall(LMBasisGEMVH(SmH0Y, oldest_new, next_new, 1.0, F, 0.0, sr1->SmH0YtFprev));
        PetscCall(LMBasisSetCachedProduct(SmH0Y, F, sr1->SmH0YtFprev));
        PetscCall(VecAXPBY(SmH0YtFprev_old, 1.0, -1.0, sr1->SmH0YtFprev));
        PetscCall(LMProductsRestoreNextColumn(SmH0YtY, &SmH0YtFprev_old));
      }
    } else {
      /* Update is bad, skip it */
      lmvm->nrejects++;
      if (cache_SmH0YtF) {
        // we still need to update the cached product
        PetscCall(LMBasisGEMVH(SmH0Y, oldest, next, 1.0, F, 0.0, sr1->SmH0YtFprev));
        PetscCall(LMBasisSetCachedProduct(SmH0Y, F, sr1->SmH0YtFprev));
      }
    }
    if (cache_SmH0YtF) PetscCall(LMBasisRestoreWorkVec(SmH0Y, &Fprev_old));
  }
  /* Save the solution and function to be used in the next update */
  PetscCall(VecCopy(X, lmvm->Xprev));
  PetscCall(VecCopy(F, lmvm->Fprev));
  lmvm->prev_set = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatReset_LMVMSR1(Mat B, MatLMVMResetMode mode)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  Mat_LSR1 *lsr1 = (Mat_LSR1 *)lmvm->ctx;

  PetscFunctionBegin;
  if (MatLMVMResetClearsBases(mode)) {
    for (PetscInt i = 0; i < SR1_BASIS_COUNT; i++) PetscCall(LMBasisDestroy(&lsr1->basis[i]));
    for (PetscInt i = 0; i < SR1_PRODUCTS_COUNT; i++) PetscCall(LMProductsDestroy(&lsr1->products[i]));
    PetscCall(VecDestroy(&lsr1->StFprev));
    PetscCall(VecDestroy(&lsr1->SmH0YtFprev));
  } else {
    for (PetscInt i = 0; i < SR1_BASIS_COUNT; i++) PetscCall(LMBasisReset(lsr1->basis[i]));
    for (PetscInt i = 0; i < SR1_PRODUCTS_COUNT; i++) PetscCall(LMProductsReset(lsr1->products[i]));
    if (lsr1->StFprev) PetscCall(VecZeroEntries(lsr1->StFprev));
    if (lsr1->SmH0YtFprev) PetscCall(VecZeroEntries(lsr1->SmH0YtFprev));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDestroy_LMVMSR1(Mat B)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;

  PetscFunctionBegin;
  PetscCall(MatReset_LMVMSR1(B, MAT_LMVM_RESET_ALL));
  PetscCall(PetscFree(lmvm->ctx));
  PetscCall(MatDestroy_LMVM(B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatLMVMSetMultAlgorithm_SR1(Mat B)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;

  PetscFunctionBegin;
  switch (lmvm->mult_alg) {
  case MAT_LMVM_MULT_RECURSIVE:
    lmvm->ops->mult  = MatMult_LMVMSR1_Recursive;
    lmvm->ops->solve = MatSolve_LMVMSR1_Recursive;
    break;
  case MAT_LMVM_MULT_DENSE:
  case MAT_LMVM_MULT_COMPACT_DENSE:
    lmvm->ops->mult  = MatMult_LMVMSR1_CompactDense;
    lmvm->ops->solve = MatSolve_LMVMSR1_CompactDense;
    break;
  }
  lmvm->ops->multht  = lmvm->ops->mult;
  lmvm->ops->solveht = lmvm->ops->solve;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatCreate_LMVMSR1(Mat B)
{
  Mat_LMVM *lmvm;
  Mat_LSR1 *lsr1;

  PetscFunctionBegin;
  PetscCall(MatCreate_LMVM(B));
  PetscCall(PetscObjectChangeTypeName((PetscObject)B, MATLMVMSR1));
  PetscCall(MatSetOption(B, MAT_HERMITIAN, PETSC_TRUE));
  B->ops->destroy = MatDestroy_LMVMSR1;

  lmvm                          = (Mat_LMVM *)B->data;
  lmvm->ops->reset              = MatReset_LMVMSR1;
  lmvm->ops->update             = MatUpdate_LMVMSR1;
  lmvm->ops->setmultalgorithm   = MatLMVMSetMultAlgorithm_SR1;
  lmvm->cache_gradient_products = PETSC_TRUE;
  PetscCall(MatLMVMSetMultAlgorithm_SR1(B));
  PetscCall(PetscNew(&lsr1));
  lmvm->ctx = (void *)lsr1;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatCreateLMVMSR1 - Creates a limited-memory Symmetric-Rank-1 approximation
  matrix used for a Jacobian. L-SR1 is symmetric by construction, but is not
  guaranteed to be positive-definite.

  To use the L-SR1 matrix with other vector types, the matrix must be
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
. -mat_lmvm_eps               - (developer) numerical zero tolerance for testing when an update should be skipped
- -mat_lmvm_debug             - (developer) perform internal debugging checks

  Level: intermediate

  Note:
  It is recommended that one use the `MatCreate()`, `MatSetType()` and/or `MatSetFromOptions()`
  paradigm instead of this routine directly.

.seealso: [](ch_ksp), `MatCreate()`, `MATLMVM`, `MATLMVMSR1`, `MatCreateLMVMBFGS()`, `MatCreateLMVMDFP()`,
          `MatCreateLMVMBroyden()`, `MatCreateLMVMBadBroyden()`, `MatCreateLMVMSymBroyden()`
@*/
PetscErrorCode MatCreateLMVMSR1(MPI_Comm comm, PetscInt n, PetscInt N, Mat *B)
{
  PetscFunctionBegin;
  PetscCall(KSPInitializePackage());
  PetscCall(MatCreate(comm, B));
  PetscCall(MatSetSizes(*B, n, n, N, N));
  PetscCall(MatSetType(*B, MATLMVMSR1));
  PetscCall(MatSetUp(*B));
  PetscFunctionReturn(PETSC_SUCCESS);
}
