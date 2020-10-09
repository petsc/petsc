#include <../src/ksp/ksp/utils/lmvm/lmvm.h> /*I "petscksp.h" I*/

/*
  Limited-memory Symmetric-Rank-1 method for approximating both
  the forward product and inverse application of a Jacobian.
*/

typedef struct {
  Vec *P, *Q;
  Vec work;
  PetscBool allocated, needP, needQ;
  PetscReal *stp, *ytq;
} Mat_LSR1;

/*------------------------------------------------------------*/

/*
  The solution method is adapted from Algorithm 8 of Erway and Marcia
  "On Solving Large-Scale Limited-Memory Quasi-Newton Equations"
  (https://arxiv.org/abs/1510.06378).

  Q[i] = S[i] - (B_i)^{-1}*Y[i] terms are computed ahead of time whenever
  the matrix is updated with a new (S[i], Y[i]) pair. This allows
  repeated calls of MatMult inside KSP solvers without unnecessarily
  recomputing Q[i] terms in expensive nested-loops.

  dX <- J0^{-1} * F
  for i = 0,1,2,...,k
    # Q[i] = S[i] - (B_i)^{-1}*Y[i]
    zeta = (Q[i]^T F) / (Q[i]^T Y[i])
    dX <- dX + (zeta * Q[i])
  end
*/
static PetscErrorCode MatSolve_LMVMSR1(Mat B, Vec F, Vec dX)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_LSR1          *lsr1 = (Mat_LSR1*)lmvm->ctx;
  PetscErrorCode    ierr;
  PetscInt          i, j;
  PetscScalar       qjtyi, qtf, ytq;

  PetscFunctionBegin;
  VecCheckSameSize(F, 2, dX, 3);
  VecCheckMatCompatible(B, dX, 3, F, 2);

  if (lsr1->needQ) {
    /* Pre-compute (Q[i] = S[i] - (B_i)^{-1} * Y[i]) and (Y[i]^T Q[i]) */
    for (i = 0; i <= lmvm->k; ++i) {
      ierr = MatLMVMApplyJ0Inv(B, lmvm->Y[i], lsr1->Q[i]);CHKERRQ(ierr);
      ierr = VecAYPX(lsr1->Q[i], -1.0, lmvm->S[i]);CHKERRQ(ierr);
      for (j = 0; j <= i-1; ++j) {
        ierr = VecDot(lsr1->Q[j], lmvm->Y[i], &qjtyi);CHKERRQ(ierr);
        ierr = VecAXPY(lsr1->Q[i], -PetscRealPart(qjtyi)/lsr1->ytq[j], lsr1->Q[j]);CHKERRQ(ierr);
      }
      ierr = VecDot(lmvm->Y[i], lsr1->Q[i], &ytq);CHKERRQ(ierr);
      lsr1->ytq[i] = PetscRealPart(ytq);
    }
    lsr1->needQ = PETSC_FALSE;
  }

  /* Invert the initial Jacobian onto F (or apply scaling) */
  ierr = MatLMVMApplyJ0Inv(B, F, dX);CHKERRQ(ierr);
  /* Start outer loop */
  for (i = 0; i <= lmvm->k; ++i) {
    ierr = VecDot(lsr1->Q[i], F, &qtf);CHKERRQ(ierr);
    ierr = VecAXPY(dX, PetscRealPart(qtf)/lsr1->ytq[i], lsr1->Q[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/*
  The forward product is the matrix-free implementation of
  Equation (6.24) in Nocedal and Wright "Numerical Optimization"
  2nd edition, pg 144.

  Note that the structure of the forward product is identical to
  the solution, with S and Y exchanging roles.

  P[i] = Y[i] - (B_i)*S[i] terms are computed ahead of time whenever
  the matrix is updated with a new (S[i], Y[i]) pair. This allows
  repeated calls of MatMult inside KSP solvers without unnecessarily
  recomputing P[i] terms in expensive nested-loops.

  Z <- J0 * X
  for i = 0,1,2,...,k
    # P[i] = Y[i] - (B_i)*S[i]
    zeta = (P[i]^T X) / (P[i]^T S[i])
    Z <- Z + (zeta * P[i])
  end
*/
static PetscErrorCode MatMult_LMVMSR1(Mat B, Vec X, Vec Z)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_LSR1          *lsr1 = (Mat_LSR1*)lmvm->ctx;
  PetscErrorCode    ierr;
  PetscInt          i, j;
  PetscScalar       pjtsi, ptx, stp;

  PetscFunctionBegin;
  VecCheckSameSize(X, 2, Z, 3);
  VecCheckMatCompatible(B, X, 2, Z, 3);

  if (lsr1->needP) {
    /* Pre-compute (P[i] = Y[i] - (B_i) * S[i]) and (S[i]^T P[i]) */
    for (i = 0; i <= lmvm->k; ++i) {
      ierr = MatLMVMApplyJ0Fwd(B, lmvm->S[i], lsr1->P[i]);CHKERRQ(ierr);
      ierr = VecAYPX(lsr1->P[i], -1.0, lmvm->Y[i]);CHKERRQ(ierr);
      for (j = 0; j <= i-1; ++j) {
        ierr = VecDot(lsr1->P[j], lmvm->S[i], &pjtsi);CHKERRQ(ierr);
        ierr = VecAXPY(lsr1->P[i], -PetscRealPart(pjtsi)/lsr1->stp[j], lsr1->P[j]);CHKERRQ(ierr);
      }
      ierr = VecDot(lmvm->S[i], lsr1->P[i], &stp);CHKERRQ(ierr);
      lsr1->stp[i] = PetscRealPart(stp);
    }
    lsr1->needP = PETSC_FALSE;
  }

  ierr = MatLMVMApplyJ0Fwd(B, X, Z);CHKERRQ(ierr);
  for (i = 0; i <= lmvm->k; ++i) {
    ierr = VecDot(lsr1->P[i], X, &ptx);CHKERRQ(ierr);
    ierr = VecAXPY(Z, PetscRealPart(ptx)/lsr1->stp[i], lsr1->P[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatUpdate_LMVMSR1(Mat B, Vec X, Vec F)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_LSR1          *lsr1 = (Mat_LSR1*)lmvm->ctx;
  PetscErrorCode    ierr;
  PetscReal         snorm, pnorm;
  PetscScalar       sktw;

  PetscFunctionBegin;
  if (!lmvm->m) PetscFunctionReturn(0);
  if (lmvm->prev_set) {
    /* Compute the new (S = X - Xprev) and (Y = F - Fprev) vectors */
    ierr = VecAYPX(lmvm->Xprev, -1.0, X);CHKERRQ(ierr);
    ierr = VecAYPX(lmvm->Fprev, -1.0, F);CHKERRQ(ierr);
    /* See if the updates can be accepted
       NOTE: This tests abs(S[k]^T (Y[k] - B_k*S[k])) >= eps * norm(S[k]) * norm(Y[k] - B_k*S[k]) */
    ierr = MatMult(B, lmvm->Xprev, lsr1->work);CHKERRQ(ierr);
    ierr = VecAYPX(lsr1->work, -1.0, lmvm->Fprev);CHKERRQ(ierr);
    ierr = VecDot(lmvm->Xprev, lsr1->work, &sktw);CHKERRQ(ierr);
    ierr = VecNorm(lmvm->Xprev, NORM_2, &snorm);CHKERRQ(ierr);
    ierr = VecNorm(lsr1->work, NORM_2, &pnorm);CHKERRQ(ierr);
    if (PetscAbsReal(PetscRealPart(sktw)) >= lmvm->eps * snorm * pnorm) {
      /* Update is good, accept it */
      lsr1->needP = lsr1->needQ = PETSC_TRUE;
      ierr = MatUpdateKernel_LMVM(B, lmvm->Xprev, lmvm->Fprev);CHKERRQ(ierr);
    } else {
      /* Update is bad, skip it */
      ++lmvm->nrejects;
    }
  }
  /* Save the solution and function to be used in the next update */
  ierr = VecCopy(X, lmvm->Xprev);CHKERRQ(ierr);
  ierr = VecCopy(F, lmvm->Fprev);CHKERRQ(ierr);
  lmvm->prev_set = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatCopy_LMVMSR1(Mat B, Mat M, MatStructure str)
{
  Mat_LMVM          *bdata = (Mat_LMVM*)B->data;
  Mat_LSR1          *bctx = (Mat_LSR1*)bdata->ctx;
  Mat_LMVM          *mdata = (Mat_LMVM*)M->data;
  Mat_LSR1          *mctx = (Mat_LSR1*)mdata->ctx;
  PetscErrorCode    ierr;
  PetscInt          i;

  PetscFunctionBegin;
  mctx->needP = bctx->needP;
  mctx->needQ = bctx->needQ;
  for (i=0; i<=bdata->k; ++i) {
    mctx->stp[i] = bctx->stp[i];
    mctx->ytq[i] = bctx->ytq[i];
    ierr = VecCopy(bctx->P[i], mctx->P[i]);CHKERRQ(ierr);
    ierr = VecCopy(bctx->Q[i], mctx->Q[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatReset_LMVMSR1(Mat B, PetscBool destructive)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_LSR1          *lsr1 = (Mat_LSR1*)lmvm->ctx;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  lsr1->needP = lsr1->needQ = PETSC_TRUE;
  if (destructive && lsr1->allocated) {
    ierr = VecDestroy(&lsr1->work);CHKERRQ(ierr);
    ierr = PetscFree2(lsr1->stp, lsr1->ytq);CHKERRQ(ierr);
    ierr = VecDestroyVecs(lmvm->m, &lsr1->P);CHKERRQ(ierr);
    ierr = VecDestroyVecs(lmvm->m, &lsr1->Q);CHKERRQ(ierr);
    lsr1->allocated = PETSC_FALSE;
  }
  ierr = MatReset_LMVM(B, destructive);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatAllocate_LMVMSR1(Mat B, Vec X, Vec F)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_LSR1          *lsr1 = (Mat_LSR1*)lmvm->ctx;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatAllocate_LMVM(B, X, F);CHKERRQ(ierr);
  if (!lsr1->allocated) {
    ierr = VecDuplicate(X, &lsr1->work);CHKERRQ(ierr);
    ierr = PetscMalloc2(lmvm->m, &lsr1->stp, lmvm->m, &lsr1->ytq);CHKERRQ(ierr);
    if (lmvm->m > 0) {
      ierr = VecDuplicateVecs(X, lmvm->m, &lsr1->P);CHKERRQ(ierr);
      ierr = VecDuplicateVecs(X, lmvm->m, &lsr1->Q);CHKERRQ(ierr);
    }
    lsr1->allocated = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatDestroy_LMVMSR1(Mat B)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_LSR1          *lsr1 = (Mat_LSR1*)lmvm->ctx;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (lsr1->allocated) {
    ierr = VecDestroy(&lsr1->work);CHKERRQ(ierr);
    ierr = PetscFree2(lsr1->stp, lsr1->ytq);CHKERRQ(ierr);
    ierr = VecDestroyVecs(lmvm->m, &lsr1->P);CHKERRQ(ierr);
    ierr = VecDestroyVecs(lmvm->m, &lsr1->Q);CHKERRQ(ierr);
    lsr1->allocated = PETSC_FALSE;
  }
  ierr = PetscFree(lmvm->ctx);CHKERRQ(ierr);
  ierr = MatDestroy_LMVM(B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatSetUp_LMVMSR1(Mat B)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_LSR1          *lsr1 = (Mat_LSR1*)lmvm->ctx;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatSetUp_LMVM(B);CHKERRQ(ierr);
  if (!lsr1->allocated && lmvm->m > 0) {
    ierr = VecDuplicate(lmvm->Xprev, &lsr1->work);CHKERRQ(ierr);
    ierr = PetscMalloc2(lmvm->m, &lsr1->stp, lmvm->m, &lsr1->ytq);CHKERRQ(ierr);
    if (lmvm->m > 0) {
      ierr = VecDuplicateVecs(lmvm->Xprev, lmvm->m, &lsr1->P);CHKERRQ(ierr);
      ierr = VecDuplicateVecs(lmvm->Xprev, lmvm->m, &lsr1->Q);CHKERRQ(ierr);
    }
    lsr1->allocated = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode MatCreate_LMVMSR1(Mat B)
{
  Mat_LMVM          *lmvm;
  Mat_LSR1          *lsr1;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatCreate_LMVM(B);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B, MATLMVMSR1);CHKERRQ(ierr);
  ierr = MatSetOption(B, MAT_SYMMETRIC, PETSC_TRUE);CHKERRQ(ierr);
  B->ops->setup = MatSetUp_LMVMSR1;
  B->ops->destroy = MatDestroy_LMVMSR1;
  B->ops->solve = MatSolve_LMVMSR1;

  lmvm = (Mat_LMVM*)B->data;
  lmvm->square = PETSC_TRUE;
  lmvm->ops->allocate = MatAllocate_LMVMSR1;
  lmvm->ops->reset = MatReset_LMVMSR1;
  lmvm->ops->update = MatUpdate_LMVMSR1;
  lmvm->ops->mult = MatMult_LMVMSR1;
  lmvm->ops->copy = MatCopy_LMVMSR1;

  ierr = PetscNewLog(B, &lsr1);CHKERRQ(ierr);
  lmvm->ctx = (void*)lsr1;
  lsr1->allocated = PETSC_FALSE;
  lsr1->needP = lsr1->needQ = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/*@
   MatCreateLMVMSR1 - Creates a limited-memory Symmetric-Rank-1 approximation
   matrix used for a Jacobian. L-SR1 is symmetric by construction, but is not
   guaranteed to be positive-definite.

   The provided local and global sizes must match the solution and function vectors
   used with MatLMVMUpdate() and MatSolve(). The resulting L-SR1 matrix will have
   storage vectors allocated with VecCreateSeq() in serial and VecCreateMPI() in
   parallel. To use the L-SR1 matrix with other vector types, the matrix must be
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

.seealso: MatCreate(), MATLMVM, MATLMVMSR1, MatCreateLMVMBFGS(), MatCreateLMVMDFP(),
          MatCreateLMVMBrdn(), MatCreateLMVMBadBrdn(), MatCreateLMVMSymBrdn()
@*/
PetscErrorCode MatCreateLMVMSR1(MPI_Comm comm, PetscInt n, PetscInt N, Mat *B)
{
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatCreate(comm, B);CHKERRQ(ierr);
  ierr = MatSetSizes(*B, n, n, N, N);CHKERRQ(ierr);
  ierr = MatSetType(*B, MATLMVMSR1);CHKERRQ(ierr);
  ierr = MatSetUp(*B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
