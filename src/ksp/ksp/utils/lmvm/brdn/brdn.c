#include <../src/ksp/ksp/utils/lmvm/brdn/brdn.h> /*I "petscksp.h" I*/
#include <petscblaslapack.h>

// Broyden is dual to bad Broyden: MatSolve routines are dual to bad Broyden MatMult routines

static PetscErrorCode MatSolve_LMVMBrdn_Recursive(Mat B, Vec F, Vec dX)
{
  PetscFunctionBegin;
  PetscCall(BadBroydenKernel_Recursive(B, MATLMVM_MODE_DUAL, F, dX));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSolveHermitianTranspose_LMVMBrdn_Recursive(Mat B, Vec F, Vec dX)
{
  PetscFunctionBegin;
  PetscCall(BadBroydenKernelHermitianTranspose_Recursive(B, MATLMVM_MODE_DUAL, F, dX));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSolve_LMVMBrdn_CompactDense(Mat B, Vec F, Vec dX)
{
  PetscFunctionBegin;
  PetscCall(BadBroydenKernel_CompactDense(B, MATLMVM_MODE_DUAL, F, dX));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSolveHermitianTranspose_LMVMBrdn_CompactDense(Mat B, Vec F, Vec dX)
{
  PetscFunctionBegin;
  PetscCall(BadBroydenKernelHermitianTranspose_CompactDense(B, MATLMVM_MODE_DUAL, F, dX));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* The Broyden kernel can be written as

     B_k = B_{k-1} + (y_{k-1} - B_{k-1} s_{k-1}) (s_{k-1}^H s_{k-1})^{-1} s_{k-1}^H

   This means that we can write B_k x using an intermediate variable alpha_k as

     alpha_k = (s_{k-1}^H s_{k-1})^{-1} s_{k-1}^H x
     B_k x   = y_{k-1} alpha_k + B_{k-1}(x - s_{k-1} alpha_k)
                                 \_____________ ____________/
                                               V
                                     recursive application
 */

PETSC_INTERN PetscErrorCode BroydenKernel_Recursive(Mat B, MatLMVMMode mode, Vec X, Vec BX)
{
  Vec              G   = X;
  Vec              W   = BX;
  MatLMVMBasisType S_t = LMVMModeMap(LMBASIS_S, mode);
  MatLMVMBasisType Y_t = LMVMModeMap(LMBASIS_Y, mode);
  LMBasis          S = NULL, Y = NULL;
  PetscInt         oldest, next;

  PetscFunctionBegin;
  PetscCall(MatLMVMGetRange(B, &oldest, &next));
  if (next > oldest) {
    LMProducts StS;

    PetscCall(MatLMVMGetUpdatedBasis(B, S_t, &S, NULL, NULL));
    PetscCall(MatLMVMGetUpdatedBasis(B, Y_t, &Y, NULL, NULL));
    PetscCall(MatLMVMGetUpdatedProducts(B, S_t, S_t, LMBLOCK_DIAGONAL, &StS));

    PetscCall(LMBasisGetWorkVec(S, &G));
    PetscCall(VecCopy(X, G));
    PetscCall(VecZeroEntries(BX));

    for (PetscInt i = next - 1; i >= oldest; i--) {
      Vec         s_i, y_i;
      PetscScalar sitg, sitsi, alphai;

      PetscCall(LMBasisGetVecRead(S, i, &s_i));
      PetscCall(LMBasisGetVecRead(Y, i, &y_i));
      PetscCall(LMProductsGetDiagonalValue(StS, i, &sitsi));

      PetscCall(VecDot(G, s_i, &sitg));
      alphai = sitg / sitsi;
      PetscCall(VecAXPY(BX, alphai, y_i));
      PetscCall(VecAXPY(G, -alphai, s_i));

      PetscCall(LMBasisRestoreVecRead(Y, i, &y_i));
      PetscCall(LMBasisRestoreVecRead(S, i, &s_i));
    }

    PetscCall(LMBasisGetWorkVec(Y, &W));
  }
  PetscCall(MatLMVMApplyJ0Mode(mode)(B, G, W));
  if (next > oldest) {
    PetscCall(VecAXPY(BX, 1.0, W));
    PetscCall(LMBasisRestoreWorkVec(Y, &W));
    PetscCall(LMBasisRestoreWorkVec(S, &G));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* The adjoint of the kernel is

     B_k^H = B_{k-1}^H + s_{k-1} (s_{k-1}^H s_{k-1})^{-1} (y_{k-1} - B_{k-1} s_{k-1})^H

   This means that we can write B_k^H x using an intermediate variable w_k = B_{k-1}^H x

     w_k = B_{k-1}^H x <-- recursive application
     B_k^H x = w_k + s_{k-1} (s_{k-1}^H s_{k-1})^{-1} (y_{k-1}^H x - s_k^H w_k)
 */

PETSC_INTERN PetscErrorCode BroydenKernelHermitianTranspose_Recursive(Mat B, MatLMVMMode mode, Vec X, Vec BHX)
{
  PetscInt oldest, next;

  PetscFunctionBegin;
  PetscCall(MatLMVMApplyJ0HermitianTransposeMode(mode)(B, X, BHX));
  PetscCall(MatLMVMGetRange(B, &oldest, &next));
  if (next > oldest) {
    MatLMVMBasisType S_t = LMVMModeMap(LMBASIS_S, mode);
    MatLMVMBasisType Y_t = LMVMModeMap(LMBASIS_Y, mode);
    LMBasis          S, Y;
    LMProducts       StS;

    PetscCall(MatLMVMGetUpdatedProducts(B, S_t, S_t, LMBLOCK_DIAGONAL, &StS));
    PetscCall(MatLMVMGetUpdatedBasis(B, S_t, &S, NULL, NULL));
    PetscCall(MatLMVMGetUpdatedBasis(B, Y_t, &Y, NULL, NULL));

    for (PetscInt i = oldest; i < next; i++) {
      Vec         s_i, y_i;
      PetscScalar sitBHX, sitsi, yitx;

      PetscCall(LMBasisGetVecRead(S, i, &s_i));
      PetscCall(LMBasisGetVecRead(Y, i, &y_i));
      PetscCall(LMProductsGetDiagonalValue(StS, i, &sitsi));

      PetscCall(VecDotBegin(BHX, s_i, &sitBHX));
      PetscCall(VecDotBegin(X, y_i, &yitx));
      PetscCall(VecDotEnd(BHX, s_i, &sitBHX));
      PetscCall(VecDotEnd(X, y_i, &yitx));
      PetscCall(VecAXPY(BHX, (yitx - sitBHX) / sitsi, s_i));
      PetscCall(LMBasisRestoreVecRead(Y, i, &y_i));
      PetscCall(LMBasisRestoreVecRead(S, i, &s_i));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   The Broyden kernel can be written as

     B_k = B_0 + (Y - B_0 S) (triu(S^H S))^{-1} S^H

   where triu is the upper triangular component.  We solve by back substitution each time
   we apply
 */

PETSC_INTERN PetscErrorCode BroydenKernel_CompactDense(Mat B, MatLMVMMode mode, Vec X, Vec BX)
{
  PetscInt oldest, next;

  PetscFunctionBegin;
  PetscCall(MatLMVMApplyJ0Mode(mode)(B, X, BX));
  PetscCall(MatLMVMGetRange(B, &oldest, &next));
  if (next > oldest) {
    MatLMVMBasisType S_t           = LMVMModeMap(LMBASIS_S, mode);
    MatLMVMBasisType Y_minus_B0S_t = LMVMModeMap(LMBASIS_Y_MINUS_B0S, mode);
    LMBasis          S;
    LMProducts       StS;
    Vec              StX;

    PetscCall(MatLMVMGetUpdatedBasis(B, S_t, &S, NULL, NULL));
    PetscCall(MatLMVMGetUpdatedProducts(B, S_t, S_t, LMBLOCK_UPPER_TRIANGLE, &StS));
    PetscCall(MatLMVMGetWorkRow(B, &StX));
    PetscCall(LMBasisGEMVH(S, oldest, next, 1.0, X, 0.0, StX));
    PetscCall(LMProductsSolve(StS, oldest, next, StX, StX, /* ^H */ PETSC_FALSE));
    PetscCall(MatLMVMBasisGEMV(B, Y_minus_B0S_t, oldest, next, 1.0, StX, 1.0, BX));
    PetscCall(MatLMVMRestoreWorkRow(B, &StX));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   The adjoint of the above formula is

     B_k^H = B_0^H + S^H (triu(S^H S))^{-H} (Y - B_0 S)^H
 */

PETSC_INTERN PetscErrorCode BroydenKernelHermitianTranspose_CompactDense(Mat B, MatLMVMMode mode, Vec X, Vec BHX)
{
  PetscInt oldest, next;

  PetscFunctionBegin;
  PetscCall(MatLMVMApplyJ0HermitianTransposeMode(mode)(B, X, BHX));
  PetscCall(MatLMVMGetRange(B, &oldest, &next));
  if (next > oldest) {
    MatLMVMBasisType S_t           = LMVMModeMap(LMBASIS_S, mode);
    MatLMVMBasisType Y_minus_B0S_t = LMVMModeMap(LMBASIS_Y_MINUS_B0S, mode);
    LMProducts       StS;
    LMBasis          S;
    Vec              YmB0StX;

    PetscCall(MatLMVMGetUpdatedBasis(B, S_t, &S, NULL, NULL));
    PetscCall(MatLMVMGetUpdatedProducts(B, S_t, S_t, LMBLOCK_UPPER_TRIANGLE, &StS));
    PetscCall(MatLMVMGetWorkRow(B, &YmB0StX));
    PetscCall(MatLMVMBasisGEMVH(B, Y_minus_B0S_t, oldest, next, 1.0, X, 0.0, YmB0StX));
    PetscCall(LMProductsSolve(StS, oldest, next, YmB0StX, YmB0StX, /* ^H */ PETSC_TRUE));
    PetscCall(LMBasisGEMV(S, oldest, next, 1.0, YmB0StX, 1.0, BHX));
    PetscCall(MatLMVMRestoreWorkRow(B, &YmB0StX));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   The dense Broyden formula use for CompactDense can be written in a non-update form as

     B_k = [B_0 | Y] [ I | -S ] [          I           ]
                     [---+----] [----------------------]
                     [ 0 |  I ] [ triu(S^H S)^{-1} S^H ]

   The advantage of this form is B_0 appears only once and the (Y - B_0 S) vectors are not needed
 */

PETSC_INTERN PetscErrorCode BroydenKernel_Dense(Mat B, MatLMVMMode mode, Vec X, Vec BX)
{
  Vec              G   = X;
  Vec              W   = BX;
  MatLMVMBasisType S_t = LMVMModeMap(LMBASIS_S, mode);
  MatLMVMBasisType Y_t = LMVMModeMap(LMBASIS_Y, mode);
  LMBasis          S = NULL, Y = NULL;
  PetscInt         oldest, next;

  PetscFunctionBegin;
  PetscCall(MatLMVMGetRange(B, &oldest, &next));
  if (next > oldest) {
    LMProducts StS;
    Vec        StX;

    PetscCall(MatLMVMGetUpdatedBasis(B, S_t, &S, NULL, NULL));
    PetscCall(MatLMVMGetUpdatedBasis(B, Y_t, &Y, NULL, NULL));
    PetscCall(MatLMVMGetUpdatedProducts(B, S_t, S_t, LMBLOCK_UPPER_TRIANGLE, &StS));
    PetscCall(LMBasisGetWorkVec(S, &G));
    PetscCall(MatLMVMGetWorkRow(B, &StX));
    PetscCall(LMBasisGEMVH(S, oldest, next, 1.0, X, 0.0, StX));
    PetscCall(LMProductsSolve(StS, oldest, next, StX, StX, PETSC_FALSE));
    PetscCall(VecCopy(X, G));
    PetscCall(LMBasisGEMV(S, oldest, next, -1.0, StX, 1.0, G));
    PetscCall(LMBasisGEMV(Y, oldest, next, 1.0, StX, 0.0, BX));
    PetscCall(MatLMVMRestoreWorkRow(B, &StX));
    PetscCall(LMBasisGetWorkVec(Y, &W));
  }
  PetscCall(MatLMVMApplyJ0Mode(mode)(B, G, W));
  if (next > oldest) {
    PetscCall(VecAXPY(BX, 1.0, W));
    PetscCall(LMBasisRestoreWorkVec(Y, &W));
    PetscCall(LMBasisRestoreWorkVec(S, &G));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   The adoint of the above formula is

     B_k^h = [I | S triu(S^H S)^{-H} ] [  I   | 0 ] [ B_0^H ]
                                       [------+---] [-------]
                                       [ -S^H | I ] [  Y^H  ]
 */

PETSC_INTERN PetscErrorCode BroydenKernelHermitianTranspose_Dense(Mat B, MatLMVMMode mode, Vec X, Vec BHX)
{
  PetscInt oldest, next;

  PetscFunctionBegin;
  PetscCall(MatLMVMApplyJ0HermitianTransposeMode(mode)(B, X, BHX));
  PetscCall(MatLMVMGetRange(B, &oldest, &next));
  if (next > oldest) {
    MatLMVMBasisType S_t = LMVMModeMap(LMBASIS_S, mode);
    MatLMVMBasisType Y_t = LMVMModeMap(LMBASIS_Y, mode);
    LMBasis          S, Y;
    LMProducts       StS;
    Vec              YtX, StBHX;

    PetscCall(MatLMVMGetUpdatedProducts(B, S_t, S_t, LMBLOCK_UPPER_TRIANGLE, &StS));
    PetscCall(MatLMVMGetUpdatedBasis(B, S_t, &S, NULL, NULL));
    PetscCall(MatLMVMGetUpdatedBasis(B, Y_t, &Y, NULL, NULL));
    PetscCall(MatLMVMGetWorkRow(B, &YtX));
    PetscCall(MatLMVMGetWorkRow(B, &StBHX));
    PetscCall(LMBasisGEMVH(S, oldest, next, 1.0, BHX, 0.0, StBHX));
    PetscCall(LMBasisGEMVH(Y, oldest, next, 1.0, X, 0.0, YtX));
    PetscCall(VecAXPY(YtX, -1.0, StBHX));
    PetscCall(LMProductsSolve(StS, oldest, next, YtX, YtX, PETSC_TRUE));
    PetscCall(LMBasisGEMV(S, oldest, next, 1.0, YtX, 1.0, BHX));
    PetscCall(MatLMVMRestoreWorkRow(B, &StBHX));
    PetscCall(MatLMVMRestoreWorkRow(B, &YtX));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMult_LMVMBrdn_Recursive(Mat B, Vec X, Vec Z)
{
  PetscFunctionBegin;
  PetscCall(BroydenKernel_Recursive(B, MATLMVM_MODE_PRIMAL, X, Z));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMultHermitianTranspose_LMVMBrdn_Recursive(Mat B, Vec X, Vec Z)
{
  PetscFunctionBegin;
  PetscCall(BroydenKernelHermitianTranspose_Recursive(B, MATLMVM_MODE_PRIMAL, X, Z));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMult_LMVMBrdn_CompactDense(Mat B, Vec X, Vec Z)
{
  PetscFunctionBegin;
  PetscCall(BroydenKernel_CompactDense(B, MATLMVM_MODE_PRIMAL, X, Z));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMultHermitianTranspose_LMVMBrdn_CompactDense(Mat B, Vec X, Vec Z)
{
  PetscFunctionBegin;
  PetscCall(BroydenKernelHermitianTranspose_CompactDense(B, MATLMVM_MODE_PRIMAL, X, Z));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_UNUSED static PetscErrorCode MatMult_LMVMBrdn_Dense(Mat B, Vec X, Vec Z)
{
  PetscFunctionBegin;
  PetscCall(BroydenKernel_Dense(B, MATLMVM_MODE_PRIMAL, X, Z));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_UNUSED static PetscErrorCode MatMultHermitianTranspose_LMVMBrdn_Dense(Mat B, Vec X, Vec Z)
{
  PetscFunctionBegin;
  PetscCall(BroydenKernelHermitianTranspose_Dense(B, MATLMVM_MODE_PRIMAL, X, Z));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatUpdate_LMVMBrdn(Mat B, Vec X, Vec F)
{
  Mat_LMVM *lmvm          = (Mat_LMVM *)B->data;
  Mat_Brdn *brdn          = (Mat_Brdn *)lmvm->ctx;
  PetscBool cache_YtFprev = (lmvm->mult_alg != MAT_LMVM_MULT_RECURSIVE) ? lmvm->cache_gradient_products : PETSC_FALSE;

  PetscFunctionBegin;
  if (!lmvm->m) PetscFunctionReturn(PETSC_SUCCESS);
  if (lmvm->prev_set) {
    LMBasis    Y;
    PetscInt   oldest, next;
    Vec        Fprev_old   = NULL;
    Vec        YtFprev_old = NULL;
    LMProducts YtY         = NULL;

    PetscCall(MatLMVMGetRange(B, &oldest, &next));
    PetscCall(MatLMVMGetUpdatedBasis(B, LMBASIS_Y, &Y, NULL, NULL));

    if (cache_YtFprev) {
      if (!brdn->YtFprev) PetscCall(LMBasisCreateRow(Y, &brdn->YtFprev));
      PetscCall(LMBasisGetWorkVec(Y, &Fprev_old));
      PetscCall(MatLMVMGetUpdatedProducts(B, LMBASIS_Y, LMBASIS_Y, LMBLOCK_UPPER_TRIANGLE, &YtY));
      PetscCall(LMProductsGetNextColumn(YtY, &YtFprev_old));
      PetscCall(VecCopy(lmvm->Fprev, Fprev_old));
      PetscCall(VecCopy(brdn->YtFprev, YtFprev_old));
    }
    /* Compute the new (S = X - Xprev) and (Y = F - Fprev) vectors */
    PetscCall(VecAYPX(lmvm->Xprev, -1.0, X));
    PetscCall(VecAYPX(lmvm->Fprev, -1.0, F));
    /* Accept the update */
    PetscCall(MatUpdateKernel_LMVM(B, lmvm->Xprev, lmvm->Fprev));
    if (cache_YtFprev) {
      PetscInt oldest_new, next_new;

      PetscCall(MatLMVMGetRange(B, &oldest_new, &next_new));
      // Compute the one new Y_i^T Fprev_old value
      PetscCall(LMBasisGEMVH(Y, next, next_new, 1.0, Fprev_old, 0.0, YtFprev_old));
      PetscCall(LMBasisGEMVH(Y, oldest_new, next_new, 1.0, F, 0.0, brdn->YtFprev));
      PetscCall(LMBasisSetCachedProduct(Y, F, brdn->YtFprev));
      PetscCall(VecAXPBY(YtFprev_old, 1.0, -1.0, brdn->YtFprev));
      PetscCall(LMProductsRestoreNextColumn(YtY, &YtFprev_old));
      PetscCall(LMBasisRestoreWorkVec(Y, &Fprev_old));
    }
  } else if (cache_YtFprev) {
    LMBasis Y;

    PetscCall(MatLMVMGetUpdatedBasis(B, LMBASIS_Y, &Y, NULL, NULL));
    if (!brdn->YtFprev) PetscCall(LMBasisCreateRow(Y, &brdn->YtFprev));
  }
  /* Save the solution and function to be used in the next update */
  PetscCall(VecCopy(X, lmvm->Xprev));
  PetscCall(VecCopy(F, lmvm->Fprev));
  lmvm->prev_set = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatReset_LMVMBrdn(Mat B, MatLMVMResetMode mode)
{
  Mat_LMVM *lmvm  = (Mat_LMVM *)B->data;
  Mat_Brdn *lbrdn = (Mat_Brdn *)lmvm->ctx;

  PetscFunctionBegin;
  if (MatLMVMResetClearsBases(mode)) {
    for (PetscInt i = 0; i < BROYDEN_BASIS_COUNT; i++) PetscCall(LMBasisDestroy(&lbrdn->basis[i]));
    for (PetscInt i = 0; i < BROYDEN_PRODUCTS_COUNT; i++) PetscCall(LMProductsDestroy(&lbrdn->products[i]));
    PetscCall(VecDestroy(&lbrdn->YtFprev));
  } else {
    for (PetscInt i = 0; i < BROYDEN_BASIS_COUNT; i++) PetscCall(LMBasisReset(lbrdn->basis[i]));
    for (PetscInt i = 0; i < BROYDEN_PRODUCTS_COUNT; i++) PetscCall(LMProductsReset(lbrdn->products[i]));
    if (lbrdn->YtFprev) PetscCall(VecZeroEntries(lbrdn->YtFprev));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDestroy_LMVMBrdn(Mat B)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;

  PetscFunctionBegin;
  PetscCall(MatReset_LMVMBrdn(B, MAT_LMVM_RESET_ALL));
  PetscCall(PetscFree(lmvm->ctx));
  PetscCall(MatDestroy_LMVM(B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatLMVMSetMultAlgorithm_Brdn(Mat B)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;

  PetscFunctionBegin;
  switch (lmvm->mult_alg) {
  case MAT_LMVM_MULT_RECURSIVE:
    lmvm->ops->mult    = MatMult_LMVMBrdn_Recursive;
    lmvm->ops->multht  = MatMultHermitianTranspose_LMVMBrdn_Recursive;
    lmvm->ops->solve   = MatSolve_LMVMBrdn_Recursive;
    lmvm->ops->solveht = MatSolveHermitianTranspose_LMVMBrdn_Recursive;
    break;
  case MAT_LMVM_MULT_DENSE:
    lmvm->ops->mult    = MatMult_LMVMBrdn_Dense;
    lmvm->ops->multht  = MatMultHermitianTranspose_LMVMBrdn_Dense;
    lmvm->ops->solve   = MatSolve_LMVMBrdn_CompactDense;
    lmvm->ops->solveht = MatSolveHermitianTranspose_LMVMBrdn_CompactDense;
    break;
  case MAT_LMVM_MULT_COMPACT_DENSE:
    lmvm->ops->mult    = MatMult_LMVMBrdn_CompactDense;
    lmvm->ops->multht  = MatMultHermitianTranspose_LMVMBrdn_CompactDense;
    lmvm->ops->solve   = MatSolve_LMVMBrdn_CompactDense;
    lmvm->ops->solveht = MatSolveHermitianTranspose_LMVMBrdn_CompactDense;
    break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatCreate_LMVMBrdn(Mat B)
{
  Mat_LMVM *lmvm;
  Mat_Brdn *lbrdn;

  PetscFunctionBegin;
  PetscCall(MatCreate_LMVM(B));
  PetscCall(PetscObjectChangeTypeName((PetscObject)B, MATLMVMBROYDEN));
  B->ops->destroy = MatDestroy_LMVMBrdn;

  lmvm                        = (Mat_LMVM *)B->data;
  lmvm->ops->reset            = MatReset_LMVMBrdn;
  lmvm->ops->update           = MatUpdate_LMVMBrdn;
  lmvm->ops->setmultalgorithm = MatLMVMSetMultAlgorithm_Brdn;

  PetscCall(PetscNew(&lbrdn));
  lmvm->ctx = (void *)lbrdn;

  PetscCall(MatLMVMSetMultAlgorithm_Brdn(B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatCreateLMVMBroyden - Creates a limited-memory "good" Broyden-type approximation
  matrix used for a Jacobian. L-Brdn is not guaranteed to be symmetric or
  positive-definite.

  To use the L-Brdn matrix with other vector types, the matrix must be
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

.seealso: [](ch_ksp), `MatCreate()`, `MATLMVM`, `MATLMVMBRDN`, `MatCreateLMVMDFP()`, `MatCreateLMVMSR1()`,
          `MatCreateLMVMBFGS()`, `MatCreateLMVMBadBroyden()`, `MatCreateLMVMSymBroyden()`
@*/
PetscErrorCode MatCreateLMVMBroyden(MPI_Comm comm, PetscInt n, PetscInt N, Mat *B)
{
  PetscFunctionBegin;
  PetscCall(KSPInitializePackage());
  PetscCall(MatCreate(comm, B));
  PetscCall(MatSetSizes(*B, n, n, N, N));
  PetscCall(MatSetType(*B, MATLMVMBROYDEN));
  PetscCall(MatSetUp(*B));
  PetscFunctionReturn(PETSC_SUCCESS);
}
