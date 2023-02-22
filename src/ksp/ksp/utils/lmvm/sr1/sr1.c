#include <../src/ksp/ksp/utils/lmvm/lmvm.h> /*I "petscksp.h" I*/

/*
  Limited-memory Symmetric-Rank-1 method for approximating both
  the forward product and inverse application of a Jacobian.
*/

typedef struct {
  Vec       *P, *Q;
  Vec        work;
  PetscBool  allocated, needP, needQ;
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
  Mat_LMVM   *lmvm = (Mat_LMVM *)B->data;
  Mat_LSR1   *lsr1 = (Mat_LSR1 *)lmvm->ctx;
  PetscInt    i, j;
  PetscScalar qjtyi, qtf, ytq;

  PetscFunctionBegin;
  VecCheckSameSize(F, 2, dX, 3);
  VecCheckMatCompatible(B, dX, 3, F, 2);

  if (lsr1->needQ) {
    /* Pre-compute (Q[i] = S[i] - (B_i)^{-1} * Y[i]) and (Y[i]^T Q[i]) */
    for (i = 0; i <= lmvm->k; ++i) {
      PetscCall(MatLMVMApplyJ0Inv(B, lmvm->Y[i], lsr1->Q[i]));
      PetscCall(VecAYPX(lsr1->Q[i], -1.0, lmvm->S[i]));
      for (j = 0; j <= i - 1; ++j) {
        PetscCall(VecDot(lsr1->Q[j], lmvm->Y[i], &qjtyi));
        PetscCall(VecAXPY(lsr1->Q[i], -PetscRealPart(qjtyi) / lsr1->ytq[j], lsr1->Q[j]));
      }
      PetscCall(VecDot(lmvm->Y[i], lsr1->Q[i], &ytq));
      lsr1->ytq[i] = PetscRealPart(ytq);
    }
    lsr1->needQ = PETSC_FALSE;
  }

  /* Invert the initial Jacobian onto F (or apply scaling) */
  PetscCall(MatLMVMApplyJ0Inv(B, F, dX));
  /* Start outer loop */
  for (i = 0; i <= lmvm->k; ++i) {
    PetscCall(VecDot(lsr1->Q[i], F, &qtf));
    PetscCall(VecAXPY(dX, PetscRealPart(qtf) / lsr1->ytq[i], lsr1->Q[i]));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
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
  Mat_LMVM   *lmvm = (Mat_LMVM *)B->data;
  Mat_LSR1   *lsr1 = (Mat_LSR1 *)lmvm->ctx;
  PetscInt    i, j;
  PetscScalar pjtsi, ptx, stp;

  PetscFunctionBegin;
  VecCheckSameSize(X, 2, Z, 3);
  VecCheckMatCompatible(B, X, 2, Z, 3);

  if (lsr1->needP) {
    /* Pre-compute (P[i] = Y[i] - (B_i) * S[i]) and (S[i]^T P[i]) */
    for (i = 0; i <= lmvm->k; ++i) {
      PetscCall(MatLMVMApplyJ0Fwd(B, lmvm->S[i], lsr1->P[i]));
      PetscCall(VecAYPX(lsr1->P[i], -1.0, lmvm->Y[i]));
      for (j = 0; j <= i - 1; ++j) {
        PetscCall(VecDot(lsr1->P[j], lmvm->S[i], &pjtsi));
        PetscCall(VecAXPY(lsr1->P[i], -PetscRealPart(pjtsi) / lsr1->stp[j], lsr1->P[j]));
      }
      PetscCall(VecDot(lmvm->S[i], lsr1->P[i], &stp));
      lsr1->stp[i] = PetscRealPart(stp);
    }
    lsr1->needP = PETSC_FALSE;
  }

  PetscCall(MatLMVMApplyJ0Fwd(B, X, Z));
  for (i = 0; i <= lmvm->k; ++i) {
    PetscCall(VecDot(lsr1->P[i], X, &ptx));
    PetscCall(VecAXPY(Z, PetscRealPart(ptx) / lsr1->stp[i], lsr1->P[i]));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatUpdate_LMVMSR1(Mat B, Vec X, Vec F)
{
  Mat_LMVM   *lmvm = (Mat_LMVM *)B->data;
  Mat_LSR1   *lsr1 = (Mat_LSR1 *)lmvm->ctx;
  PetscReal   snorm, pnorm;
  PetscScalar sktw;

  PetscFunctionBegin;
  if (!lmvm->m) PetscFunctionReturn(PETSC_SUCCESS);
  if (lmvm->prev_set) {
    /* Compute the new (S = X - Xprev) and (Y = F - Fprev) vectors */
    PetscCall(VecAYPX(lmvm->Xprev, -1.0, X));
    PetscCall(VecAYPX(lmvm->Fprev, -1.0, F));
    /* See if the updates can be accepted
       NOTE: This tests abs(S[k]^T (Y[k] - B_k*S[k])) >= eps * norm(S[k]) * norm(Y[k] - B_k*S[k]) */
    PetscCall(MatMult(B, lmvm->Xprev, lsr1->work));
    PetscCall(VecAYPX(lsr1->work, -1.0, lmvm->Fprev));
    PetscCall(VecDot(lmvm->Xprev, lsr1->work, &sktw));
    PetscCall(VecNorm(lmvm->Xprev, NORM_2, &snorm));
    PetscCall(VecNorm(lsr1->work, NORM_2, &pnorm));
    if (PetscAbsReal(PetscRealPart(sktw)) >= lmvm->eps * snorm * pnorm) {
      /* Update is good, accept it */
      lsr1->needP = lsr1->needQ = PETSC_TRUE;
      PetscCall(MatUpdateKernel_LMVM(B, lmvm->Xprev, lmvm->Fprev));
    } else {
      /* Update is bad, skip it */
      ++lmvm->nrejects;
    }
  }
  /* Save the solution and function to be used in the next update */
  PetscCall(VecCopy(X, lmvm->Xprev));
  PetscCall(VecCopy(F, lmvm->Fprev));
  lmvm->prev_set = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatCopy_LMVMSR1(Mat B, Mat M, MatStructure str)
{
  Mat_LMVM *bdata = (Mat_LMVM *)B->data;
  Mat_LSR1 *bctx  = (Mat_LSR1 *)bdata->ctx;
  Mat_LMVM *mdata = (Mat_LMVM *)M->data;
  Mat_LSR1 *mctx  = (Mat_LSR1 *)mdata->ctx;
  PetscInt  i;

  PetscFunctionBegin;
  mctx->needP = bctx->needP;
  mctx->needQ = bctx->needQ;
  for (i = 0; i <= bdata->k; ++i) {
    mctx->stp[i] = bctx->stp[i];
    mctx->ytq[i] = bctx->ytq[i];
    PetscCall(VecCopy(bctx->P[i], mctx->P[i]));
    PetscCall(VecCopy(bctx->Q[i], mctx->Q[i]));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatReset_LMVMSR1(Mat B, PetscBool destructive)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  Mat_LSR1 *lsr1 = (Mat_LSR1 *)lmvm->ctx;

  PetscFunctionBegin;
  lsr1->needP = lsr1->needQ = PETSC_TRUE;
  if (destructive && lsr1->allocated) {
    PetscCall(VecDestroy(&lsr1->work));
    PetscCall(PetscFree2(lsr1->stp, lsr1->ytq));
    PetscCall(VecDestroyVecs(lmvm->m, &lsr1->P));
    PetscCall(VecDestroyVecs(lmvm->m, &lsr1->Q));
    lsr1->allocated = PETSC_FALSE;
  }
  PetscCall(MatReset_LMVM(B, destructive));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatAllocate_LMVMSR1(Mat B, Vec X, Vec F)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  Mat_LSR1 *lsr1 = (Mat_LSR1 *)lmvm->ctx;

  PetscFunctionBegin;
  PetscCall(MatAllocate_LMVM(B, X, F));
  if (!lsr1->allocated) {
    PetscCall(VecDuplicate(X, &lsr1->work));
    PetscCall(PetscMalloc2(lmvm->m, &lsr1->stp, lmvm->m, &lsr1->ytq));
    if (lmvm->m > 0) {
      PetscCall(VecDuplicateVecs(X, lmvm->m, &lsr1->P));
      PetscCall(VecDuplicateVecs(X, lmvm->m, &lsr1->Q));
    }
    lsr1->allocated = PETSC_TRUE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatDestroy_LMVMSR1(Mat B)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  Mat_LSR1 *lsr1 = (Mat_LSR1 *)lmvm->ctx;

  PetscFunctionBegin;
  if (lsr1->allocated) {
    PetscCall(VecDestroy(&lsr1->work));
    PetscCall(PetscFree2(lsr1->stp, lsr1->ytq));
    PetscCall(VecDestroyVecs(lmvm->m, &lsr1->P));
    PetscCall(VecDestroyVecs(lmvm->m, &lsr1->Q));
    lsr1->allocated = PETSC_FALSE;
  }
  PetscCall(PetscFree(lmvm->ctx));
  PetscCall(MatDestroy_LMVM(B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatSetUp_LMVMSR1(Mat B)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  Mat_LSR1 *lsr1 = (Mat_LSR1 *)lmvm->ctx;

  PetscFunctionBegin;
  PetscCall(MatSetUp_LMVM(B));
  if (!lsr1->allocated && lmvm->m > 0) {
    PetscCall(VecDuplicate(lmvm->Xprev, &lsr1->work));
    PetscCall(PetscMalloc2(lmvm->m, &lsr1->stp, lmvm->m, &lsr1->ytq));
    if (lmvm->m > 0) {
      PetscCall(VecDuplicateVecs(lmvm->Xprev, lmvm->m, &lsr1->P));
      PetscCall(VecDuplicateVecs(lmvm->Xprev, lmvm->m, &lsr1->Q));
    }
    lsr1->allocated = PETSC_TRUE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

PetscErrorCode MatCreate_LMVMSR1(Mat B)
{
  Mat_LMVM *lmvm;
  Mat_LSR1 *lsr1;

  PetscFunctionBegin;
  PetscCall(MatCreate_LMVM(B));
  PetscCall(PetscObjectChangeTypeName((PetscObject)B, MATLMVMSR1));
  PetscCall(MatSetOption(B, MAT_SYMMETRIC, PETSC_TRUE));
  B->ops->setup   = MatSetUp_LMVMSR1;
  B->ops->destroy = MatDestroy_LMVMSR1;
  B->ops->solve   = MatSolve_LMVMSR1;

  lmvm                = (Mat_LMVM *)B->data;
  lmvm->square        = PETSC_TRUE;
  lmvm->ops->allocate = MatAllocate_LMVMSR1;
  lmvm->ops->reset    = MatReset_LMVMSR1;
  lmvm->ops->update   = MatUpdate_LMVMSR1;
  lmvm->ops->mult     = MatMult_LMVMSR1;
  lmvm->ops->copy     = MatCopy_LMVMSR1;

  PetscCall(PetscNew(&lsr1));
  lmvm->ctx       = (void *)lsr1;
  lsr1->allocated = PETSC_FALSE;
  lsr1->needP = lsr1->needQ = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

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
+  comm - MPI communicator
.  n - number of local rows for storage vectors
-  N - global size of the storage vectors

   Output Parameter:
.  B - the matrix

   Level: intermediate

   Note:
   It is recommended that one use the `MatCreate()`, `MatSetType()` and/or `MatSetFromOptions()`
   paradigm instead of this routine directly.

.seealso: [](chapter_ksp), `MatCreate()`, `MATLMVM`, `MATLMVMSR1`, `MatCreateLMVMBFGS()`, `MatCreateLMVMDFP()`,
          `MatCreateLMVMBrdn()`, `MatCreateLMVMBadBrdn()`, `MatCreateLMVMSymBrdn()`
@*/
PetscErrorCode MatCreateLMVMSR1(MPI_Comm comm, PetscInt n, PetscInt N, Mat *B)
{
  PetscFunctionBegin;
  PetscCall(MatCreate(comm, B));
  PetscCall(MatSetSizes(*B, n, n, N, N));
  PetscCall(MatSetType(*B, MATLMVMSR1));
  PetscCall(MatSetUp(*B));
  PetscFunctionReturn(PETSC_SUCCESS);
}
