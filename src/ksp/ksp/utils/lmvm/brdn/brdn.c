#include <../src/ksp/ksp/utils/lmvm/brdn/brdn.h> /*I "petscksp.h" I*/

/*------------------------------------------------------------*/

/*
  The solution method is the matrix-free implementation of the inverse Hessian
  representation in page 312 of Griewank "Broyden Updating, The Good and The Bad!"
  (http://www.emis.ams.org/journals/DMJDMV/vol-ismp/45_griewank-andreas-broyden.pdf).

  Q[i] = (B_i)^{-1}*S[i] terms are computed ahead of time whenever
  the matrix is updated with a new (S[i], Y[i]) pair. This allows
  repeated calls of MatSolve without incurring redundant computation.

  dX <- J0^{-1} * F

  for i=0,1,2,...,k
    # Q[i] = (B_i)^{-1} * Y[i]
    tau = (S[i]^T dX) / (S[i]^T Q[i])
    dX <- dX + (tau * (S[i] - Q[i]))
  end
 */

static PetscErrorCode MatSolve_LMVMBrdn(Mat B, Vec F, Vec dX)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_Brdn          *lbrdn = (Mat_Brdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  PetscInt          i, j;
  PetscScalar       sjtqi, stx, stq;

  PetscFunctionBegin;
  VecCheckSameSize(F, 2, dX, 3);
  VecCheckMatCompatible(B, dX, 3, F, 2);

  if (lbrdn->needQ) {
    /* Pre-compute (Q[i] = (B_i)^{-1} * Y[i]) */
    for (i = 0; i <= lmvm->k; ++i) {
      ierr = MatLMVMApplyJ0Inv(B, lmvm->Y[i], lbrdn->Q[i]);CHKERRQ(ierr);
      for (j = 0; j <= i-1; ++j) {
        ierr = VecDot(lmvm->S[j], lbrdn->Q[i], &sjtqi);CHKERRQ(ierr);
        ierr = VecAXPBYPCZ(lbrdn->Q[i], PetscRealPart(sjtqi)/lbrdn->stq[j], -PetscRealPart(sjtqi)/lbrdn->stq[j], 1.0, lmvm->S[j], lbrdn->Q[j]);CHKERRQ(ierr);
      }
      ierr = VecDot(lmvm->S[i], lbrdn->Q[i], &stq);CHKERRQ(ierr);
      lbrdn->stq[i] = PetscRealPart(stq);
    }
    lbrdn->needQ = PETSC_FALSE;
  }

  ierr = MatLMVMApplyJ0Inv(B, F, dX);CHKERRQ(ierr);
  for (i = 0; i <= lmvm->k; ++i) {
    ierr = VecDot(lmvm->S[i], dX, &stx);CHKERRQ(ierr);
    ierr = VecAXPBYPCZ(dX, PetscRealPart(stx)/lbrdn->stq[i], -PetscRealPart(stx)/lbrdn->stq[i], 1.0, lmvm->S[i], lbrdn->Q[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/*
  The forward product is the matrix-free implementation of Equation 2 in
  page 302 of Griewank "Broyden Updating, The Good and The Bad!"
  (http://www.emis.ams.org/journals/DMJDMV/vol-ismp/45_griewank-andreas-broyden.pdf).

  P[i] = (B_i)*S[i] terms are computed ahead of time whenever
  the matrix is updated with a new (S[i], Y[i]) pair. This allows
  repeated calls of MatMult inside KSP solvers without unnecessarily
  recomputing P[i] terms in expensive nested-loops.

  Z <- J0 * X

  for i=0,1,2,...,k
    # P[i] = B_i * S[i]
    tau = (S[i]^T X) / (S[i]^T S[i])
    dX <- dX + (tau * (Y[i] - P[i]))
  end
 */

static PetscErrorCode MatMult_LMVMBrdn(Mat B, Vec X, Vec Z)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_Brdn          *lbrdn = (Mat_Brdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  PetscInt          i, j;
  PetscScalar       sjtsi, stx;

  PetscFunctionBegin;
  VecCheckSameSize(X, 2, Z, 3);
  VecCheckMatCompatible(B, X, 2, Z, 3);

  if (lbrdn->needP) {
    /* Pre-compute (P[i] = (B_i) * S[i]) */
    for (i = 0; i <= lmvm->k; ++i) {
      ierr = MatLMVMApplyJ0Fwd(B, lmvm->S[i], lbrdn->P[i]);CHKERRQ(ierr);
      for (j = 0; j <= i-1; ++j) {
        ierr = VecDot(lmvm->S[j], lmvm->S[i], &sjtsi);CHKERRQ(ierr);
        ierr = VecAXPBYPCZ(lbrdn->P[i], PetscRealPart(sjtsi)/lbrdn->sts[j], -PetscRealPart(sjtsi)/lbrdn->sts[j], 1.0, lmvm->Y[j], lbrdn->P[j]);CHKERRQ(ierr);
      }
    }
    lbrdn->needP = PETSC_FALSE;
  }

  ierr = MatLMVMApplyJ0Fwd(B, X, Z);CHKERRQ(ierr);
  for (i = 0; i <= lmvm->k; ++i) {
    ierr = VecDot(lmvm->S[i], X, &stx);CHKERRQ(ierr);
    ierr = VecAXPBYPCZ(Z, PetscRealPart(stx)/lbrdn->sts[i], -PetscRealPart(stx)/lbrdn->sts[i], 1.0, lmvm->Y[i], lbrdn->P[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatUpdate_LMVMBrdn(Mat B, Vec X, Vec F)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_Brdn          *lbrdn = (Mat_Brdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  PetscInt          old_k, i;
  PetscScalar       sts;

  PetscFunctionBegin;
  if (!lmvm->m) PetscFunctionReturn(0);
  if (lmvm->prev_set) {
    /* Compute the new (S = X - Xprev) and (Y = F - Fprev) vectors */
    ierr = VecAYPX(lmvm->Xprev, -1.0, X);CHKERRQ(ierr);
    ierr = VecAYPX(lmvm->Fprev, -1.0, F);CHKERRQ(ierr);
    /* Accept the update */
    lbrdn->needP = lbrdn->needQ = PETSC_TRUE;
    old_k = lmvm->k;
    ierr = MatUpdateKernel_LMVM(B, lmvm->Xprev, lmvm->Fprev);CHKERRQ(ierr);
    /* If we hit the memory limit, shift the sts array */
    if (old_k == lmvm->k) {
      for (i = 0; i <= lmvm->k-1; ++i) {
        lbrdn->sts[i] = lbrdn->sts[i+1];
      }
    }
    ierr = VecDot(lmvm->S[lmvm->k], lmvm->S[lmvm->k], &sts);CHKERRQ(ierr);
    lbrdn->sts[lmvm->k] = PetscRealPart(sts);
  }
  /* Save the solution and function to be used in the next update */
  ierr = VecCopy(X, lmvm->Xprev);CHKERRQ(ierr);
  ierr = VecCopy(F, lmvm->Fprev);CHKERRQ(ierr);
  lmvm->prev_set = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatCopy_LMVMBrdn(Mat B, Mat M, MatStructure str)
{
  Mat_LMVM          *bdata = (Mat_LMVM*)B->data;
  Mat_Brdn          *bctx = (Mat_Brdn*)bdata->ctx;
  Mat_LMVM          *mdata = (Mat_LMVM*)M->data;
  Mat_Brdn          *mctx = (Mat_Brdn*)mdata->ctx;
  PetscErrorCode    ierr;
  PetscInt          i;

  PetscFunctionBegin;
  mctx->needP = bctx->needP;
  mctx->needQ = bctx->needQ;
  for (i=0; i<=bdata->k; ++i) {
    mctx->sts[i] = bctx->sts[i];
    mctx->stq[i] = bctx->stq[i];
    ierr = VecCopy(bctx->P[i], mctx->P[i]);CHKERRQ(ierr);
    ierr = VecCopy(bctx->Q[i], mctx->Q[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatReset_LMVMBrdn(Mat B, PetscBool destructive)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_Brdn          *lbrdn = (Mat_Brdn*)lmvm->ctx;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  lbrdn->needP = lbrdn->needQ = PETSC_TRUE;
  if (destructive && lbrdn->allocated) {
    ierr = PetscFree2(lbrdn->sts, lbrdn->stq);CHKERRQ(ierr);
    ierr = VecDestroyVecs(lmvm->m, &lbrdn->P);CHKERRQ(ierr);
    ierr = VecDestroyVecs(lmvm->m, &lbrdn->Q);CHKERRQ(ierr);
    lbrdn->allocated = PETSC_FALSE;
  }
  ierr = MatReset_LMVM(B, destructive);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatAllocate_LMVMBrdn(Mat B, Vec X, Vec F)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_Brdn          *lbrdn = (Mat_Brdn*)lmvm->ctx;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatAllocate_LMVM(B, X, F);CHKERRQ(ierr);
  if (!lbrdn->allocated) {
    ierr = PetscMalloc2(lmvm->m, &lbrdn->sts, lmvm->m, &lbrdn->stq);CHKERRQ(ierr);
    if (lmvm->m > 0) {
      ierr = VecDuplicateVecs(X, lmvm->m, &lbrdn->P);CHKERRQ(ierr);
      ierr = VecDuplicateVecs(X, lmvm->m, &lbrdn->Q);CHKERRQ(ierr);
    }
    lbrdn->allocated = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatDestroy_LMVMBrdn(Mat B)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_Brdn          *lbrdn = (Mat_Brdn*)lmvm->ctx;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (lbrdn->allocated) {
    ierr = PetscFree2(lbrdn->sts, lbrdn->stq);CHKERRQ(ierr);
    ierr = VecDestroyVecs(lmvm->m, &lbrdn->P);CHKERRQ(ierr);
    ierr = VecDestroyVecs(lmvm->m, &lbrdn->Q);CHKERRQ(ierr);
    lbrdn->allocated = PETSC_FALSE;
  }
  ierr = PetscFree(lmvm->ctx);CHKERRQ(ierr);
  ierr = MatDestroy_LMVM(B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatSetUp_LMVMBrdn(Mat B)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_Brdn          *lbrdn = (Mat_Brdn*)lmvm->ctx;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatSetUp_LMVM(B);CHKERRQ(ierr);
  if (!lbrdn->allocated) {
    ierr = PetscMalloc2(lmvm->m, &lbrdn->sts, lmvm->m, &lbrdn->stq);CHKERRQ(ierr);
    if (lmvm->m > 0) {
      ierr = VecDuplicateVecs(lmvm->Xprev, lmvm->m, &lbrdn->P);CHKERRQ(ierr);
      ierr = VecDuplicateVecs(lmvm->Xprev, lmvm->m, &lbrdn->Q);CHKERRQ(ierr);
    }
    lbrdn->allocated = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode MatCreate_LMVMBrdn(Mat B)
{
  Mat_LMVM          *lmvm;
  Mat_Brdn          *lbrdn;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatCreate_LMVM(B);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B, MATLMVMBROYDEN);CHKERRQ(ierr);
  B->ops->setup = MatSetUp_LMVMBrdn;
  B->ops->destroy = MatDestroy_LMVMBrdn;
  B->ops->solve = MatSolve_LMVMBrdn;

  lmvm = (Mat_LMVM*)B->data;
  lmvm->square = PETSC_TRUE;
  lmvm->ops->allocate = MatAllocate_LMVMBrdn;
  lmvm->ops->reset = MatReset_LMVMBrdn;
  lmvm->ops->mult = MatMult_LMVMBrdn;
  lmvm->ops->update = MatUpdate_LMVMBrdn;
  lmvm->ops->copy = MatCopy_LMVMBrdn;

  ierr = PetscNewLog(B, &lbrdn);CHKERRQ(ierr);
  lmvm->ctx = (void*)lbrdn;
  lbrdn->allocated = PETSC_FALSE;
  lbrdn->needP = lbrdn->needQ = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/*@
   MatCreateLMVMBroyden - Creates a limited-memory "good" Broyden-type approximation
   matrix used for a Jacobian. L-Brdn is not guaranteed to be symmetric or
   positive-definite.

   The provided local and global sizes must match the solution and function vectors
   used with MatLMVMUpdate() and MatSolve(). The resulting L-Brdn matrix will have
   storage vectors allocated with VecCreateSeq() in serial and VecCreateMPI() in
   parallel. To use the L-Brdn matrix with other vector types, the matrix must be
   created using MatCreate() and MatSetType(), followed by MatLMVMAllocate().
   This ensures that the internal storage and work vectors are duplicated from the
   correct type of vector.

   Collective

   Input Parameters:
+  comm - MPI communicator, set to PETSC_COMM_SELF
.  n - number of local rows for storage vectors
-  N - global size of the storage vectors

   Output Parameter:
.  B - the matrix

   It is recommended that one use the MatCreate(), MatSetType() and/or MatSetFromOptions()
   paradigm instead of this routine directly.

   Options Database Keys:
.   -mat_lmvm_num_vecs - maximum number of correction vectors (i.e.: updates) stored

   Level: intermediate

.seealso: MatCreate(), MATLMVM, MATLMVMBRDN, MatCreateLMVMDFP(), MatCreateLMVMSR1(),
         MatCreateLMVMBFGS(), MatCreateLMVMBadBrdn(), MatCreateLMVMSymBrdn()
@*/
PetscErrorCode MatCreateLMVMBroyden(MPI_Comm comm, PetscInt n, PetscInt N, Mat *B)
{
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatCreate(comm, B);CHKERRQ(ierr);
  ierr = MatSetSizes(*B, n, n, N, N);CHKERRQ(ierr);
  ierr = MatSetType(*B, MATLMVMBROYDEN);CHKERRQ(ierr);
  ierr = MatSetUp(*B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
