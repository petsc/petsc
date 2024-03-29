#include <../src/ksp/ksp/utils/lmvm/brdn/brdn.h> /*I "petscksp.h" I*/

/*
  The solution method is the matrix-free implementation of the inverse Hessian in
  Equation 6 on page 312 of Griewank "Broyden Updating, The Good and The Bad!"
  (http://www.emis.ams.org/journals/DMJDMV/vol-ismp/45_griewank-andreas-broyden.pdf).

  Q[i] = (B_i)^{-1}*S[i] terms are computed ahead of time whenever
  the matrix is updated with a new (S[i], Y[i]) pair. This allows
  repeated calls of MatSolve without incurring redundant computation.

  dX <- J0^{-1} * F

  for i=0,1,2,...,k
    # Q[i] = (B_i)^{-1} * Y[i]
    tau = (Y[i]^T F) / (Y[i]^T Y[i])
    dX <- dX + (tau * (S[i] - Q[i]))
  end
 */

static PetscErrorCode MatSolve_LMVMBadBrdn(Mat B, Vec F, Vec dX)
{
  Mat_LMVM   *lmvm = (Mat_LMVM *)B->data;
  Mat_Brdn   *lbb  = (Mat_Brdn *)lmvm->ctx;
  PetscInt    i, j;
  PetscScalar yjtyi, ytf;

  PetscFunctionBegin;
  VecCheckSameSize(F, 2, dX, 3);
  VecCheckMatCompatible(B, dX, 3, F, 2);

  if (lbb->needQ) {
    /* Pre-compute (Q[i] = (B_i)^{-1} * Y[i]) */
    for (i = 0; i <= lmvm->k; ++i) {
      PetscCall(MatLMVMApplyJ0Inv(B, lmvm->Y[i], lbb->Q[i]));
      for (j = 0; j <= i - 1; ++j) {
        PetscCall(VecDot(lmvm->Y[j], lmvm->Y[i], &yjtyi));
        PetscCall(VecAXPBYPCZ(lbb->Q[i], PetscRealPart(yjtyi) / lbb->yty[j], -PetscRealPart(yjtyi) / lbb->yty[j], 1.0, lmvm->S[j], lbb->Q[j]));
      }
    }
    lbb->needQ = PETSC_FALSE;
  }

  PetscCall(MatLMVMApplyJ0Inv(B, F, dX));
  for (i = 0; i <= lmvm->k; ++i) {
    PetscCall(VecDot(lmvm->Y[i], F, &ytf));
    PetscCall(VecAXPBYPCZ(dX, PetscRealPart(ytf) / lbb->yty[i], -PetscRealPart(ytf) / lbb->yty[i], 1.0, lmvm->S[i], lbb->Q[i]));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  The forward product is the matrix-free implementation of the direct update in
  Equation 6 on page 302 of Griewank "Broyden Updating, The Good and The Bad!"
  (http://www.emis.ams.org/journals/DMJDMV/vol-ismp/45_griewank-andreas-broyden.pdf).

  P[i] = (B_i)*S[i] terms are computed ahead of time whenever
  the matrix is updated with a new (S[i], Y[i]) pair. This allows
  repeated calls of MatMult inside KSP solvers without unnecessarily
  recomputing P[i] terms in expensive nested-loops.

  Z <- J0 * X

  for i=0,1,2,...,k
    # P[i] = B_i * S[i]
    tau = (Y[i]^T X) / (Y[i]^T S[i])
    dX <- dX + (tau * (Y[i] - P[i]))
  end
 */

static PetscErrorCode MatMult_LMVMBadBrdn(Mat B, Vec X, Vec Z)
{
  Mat_LMVM   *lmvm = (Mat_LMVM *)B->data;
  Mat_Brdn   *lbb  = (Mat_Brdn *)lmvm->ctx;
  PetscInt    i, j;
  PetscScalar yjtsi, ytx;

  PetscFunctionBegin;
  VecCheckSameSize(X, 2, Z, 3);
  VecCheckMatCompatible(B, X, 2, Z, 3);

  if (lbb->needP) {
    /* Pre-compute (P[i] = (B_i) * S[i]) */
    for (i = 0; i <= lmvm->k; ++i) {
      PetscCall(MatLMVMApplyJ0Fwd(B, lmvm->S[i], lbb->P[i]));
      for (j = 0; j <= i - 1; ++j) {
        PetscCall(VecDot(lmvm->Y[j], lmvm->S[i], &yjtsi));
        PetscCall(VecAXPBYPCZ(lbb->P[i], PetscRealPart(yjtsi) / lbb->yts[j], -PetscRealPart(yjtsi) / lbb->yts[j], 1.0, lmvm->Y[j], lbb->P[j]));
      }
    }
    lbb->needP = PETSC_FALSE;
  }

  PetscCall(MatLMVMApplyJ0Fwd(B, X, Z));
  for (i = 0; i <= lmvm->k; ++i) {
    PetscCall(VecDot(lmvm->Y[i], X, &ytx));
    PetscCall(VecAXPBYPCZ(Z, PetscRealPart(ytx) / lbb->yts[i], -PetscRealPart(ytx) / lbb->yts[i], 1.0, lmvm->Y[i], lbb->P[i]));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatUpdate_LMVMBadBrdn(Mat B, Vec X, Vec F)
{
  Mat_LMVM   *lmvm = (Mat_LMVM *)B->data;
  Mat_Brdn   *lbb  = (Mat_Brdn *)lmvm->ctx;
  PetscInt    old_k, i;
  PetscScalar yty, yts;

  PetscFunctionBegin;
  if (!lmvm->m) PetscFunctionReturn(PETSC_SUCCESS);
  if (lmvm->prev_set) {
    /* Compute the new (S = X - Xprev) and (Y = F - Fprev) vectors */
    PetscCall(VecAYPX(lmvm->Xprev, -1.0, X));
    PetscCall(VecAYPX(lmvm->Fprev, -1.0, F));
    /* Accept the update */
    lbb->needP = lbb->needQ = PETSC_TRUE;
    old_k                   = lmvm->k;
    PetscCall(MatUpdateKernel_LMVM(B, lmvm->Xprev, lmvm->Fprev));
    /* If we hit the memory limit, shift the yty and yts arrays */
    if (old_k == lmvm->k) {
      for (i = 0; i <= lmvm->k - 1; ++i) {
        lbb->yty[i] = lbb->yty[i + 1];
        lbb->yts[i] = lbb->yts[i + 1];
      }
    }
    /* Accumulate the latest yTy and yTs dot products */
    PetscCall(VecDotBegin(lmvm->Y[lmvm->k], lmvm->Y[lmvm->k], &yty));
    PetscCall(VecDotBegin(lmvm->Y[lmvm->k], lmvm->S[lmvm->k], &yts));
    PetscCall(VecDotEnd(lmvm->Y[lmvm->k], lmvm->Y[lmvm->k], &yty));
    PetscCall(VecDotEnd(lmvm->Y[lmvm->k], lmvm->S[lmvm->k], &yts));
    lbb->yty[lmvm->k] = PetscRealPart(yty);
    lbb->yts[lmvm->k] = PetscRealPart(yts);
  }
  /* Save the solution and function to be used in the next update */
  PetscCall(VecCopy(X, lmvm->Xprev));
  PetscCall(VecCopy(F, lmvm->Fprev));
  lmvm->prev_set = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatCopy_LMVMBadBrdn(Mat B, Mat M, MatStructure str)
{
  Mat_LMVM *bdata = (Mat_LMVM *)B->data;
  Mat_Brdn *bctx  = (Mat_Brdn *)bdata->ctx;
  Mat_LMVM *mdata = (Mat_LMVM *)M->data;
  Mat_Brdn *mctx  = (Mat_Brdn *)mdata->ctx;
  PetscInt  i;

  PetscFunctionBegin;
  mctx->needP = bctx->needP;
  mctx->needQ = bctx->needQ;
  for (i = 0; i <= bdata->k; ++i) {
    mctx->yty[i] = bctx->yty[i];
    mctx->yts[i] = bctx->yts[i];
    PetscCall(VecCopy(bctx->P[i], mctx->P[i]));
    PetscCall(VecCopy(bctx->Q[i], mctx->Q[i]));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatReset_LMVMBadBrdn(Mat B, PetscBool destructive)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  Mat_Brdn *lbb  = (Mat_Brdn *)lmvm->ctx;

  PetscFunctionBegin;
  lbb->needP = lbb->needQ = PETSC_TRUE;
  if (destructive && lbb->allocated) {
    PetscCall(PetscFree2(lbb->yty, lbb->yts));
    PetscCall(VecDestroyVecs(lmvm->m, &lbb->P));
    PetscCall(VecDestroyVecs(lmvm->m, &lbb->Q));
    lbb->allocated = PETSC_FALSE;
  }
  PetscCall(MatReset_LMVM(B, destructive));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatAllocate_LMVMBadBrdn(Mat B, Vec X, Vec F)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  Mat_Brdn *lbb  = (Mat_Brdn *)lmvm->ctx;

  PetscFunctionBegin;
  PetscCall(MatAllocate_LMVM(B, X, F));
  if (!lbb->allocated) {
    PetscCall(PetscMalloc2(lmvm->m, &lbb->yty, lmvm->m, &lbb->yts));
    if (lmvm->m > 0) {
      PetscCall(VecDuplicateVecs(X, lmvm->m, &lbb->P));
      PetscCall(VecDuplicateVecs(X, lmvm->m, &lbb->Q));
    }
    lbb->allocated = PETSC_TRUE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDestroy_LMVMBadBrdn(Mat B)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  Mat_Brdn *lbb  = (Mat_Brdn *)lmvm->ctx;

  PetscFunctionBegin;
  if (lbb->allocated) {
    PetscCall(PetscFree2(lbb->yty, lbb->yts));
    PetscCall(VecDestroyVecs(lmvm->m, &lbb->P));
    PetscCall(VecDestroyVecs(lmvm->m, &lbb->Q));
    lbb->allocated = PETSC_FALSE;
  }
  PetscCall(PetscFree(lmvm->ctx));
  PetscCall(MatDestroy_LMVM(B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetUp_LMVMBadBrdn(Mat B)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  Mat_Brdn *lbb  = (Mat_Brdn *)lmvm->ctx;

  PetscFunctionBegin;
  PetscCall(MatSetUp_LMVM(B));
  if (!lbb->allocated) {
    PetscCall(PetscMalloc2(lmvm->m, &lbb->yty, lmvm->m, &lbb->yts));
    if (lmvm->m > 0) {
      PetscCall(VecDuplicateVecs(lmvm->Xprev, lmvm->m, &lbb->P));
      PetscCall(VecDuplicateVecs(lmvm->Xprev, lmvm->m, &lbb->Q));
    }
    lbb->allocated = PETSC_TRUE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatCreate_LMVMBadBrdn(Mat B)
{
  Mat_LMVM *lmvm;
  Mat_Brdn *lbb;

  PetscFunctionBegin;
  PetscCall(MatCreate_LMVM(B));
  PetscCall(PetscObjectChangeTypeName((PetscObject)B, MATLMVMBADBROYDEN));
  B->ops->setup   = MatSetUp_LMVMBadBrdn;
  B->ops->destroy = MatDestroy_LMVMBadBrdn;
  B->ops->solve   = MatSolve_LMVMBadBrdn;

  lmvm                = (Mat_LMVM *)B->data;
  lmvm->square        = PETSC_TRUE;
  lmvm->ops->allocate = MatAllocate_LMVMBadBrdn;
  lmvm->ops->reset    = MatReset_LMVMBadBrdn;
  lmvm->ops->mult     = MatMult_LMVMBadBrdn;
  lmvm->ops->solve    = MatSolve_LMVMBadBrdn;
  lmvm->ops->update   = MatUpdate_LMVMBadBrdn;
  lmvm->ops->copy     = MatCopy_LMVMBadBrdn;

  PetscCall(PetscNew(&lbb));
  lmvm->ctx      = (void *)lbb;
  lbb->allocated = PETSC_FALSE;
  lbb->needP = lbb->needQ = PETSC_TRUE;
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

.seealso: [](ch_ksp), `MatCreate()`, `MATLMVM`, `MATLMVMBADBRDN`, `MatCreateLMVMDFP()`, `MatCreateLMVMSR1()`,
          `MatCreateLMVMBFGS()`, `MatCreateLMVMBrdn()`, `MatCreateLMVMSymBrdn()`
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
