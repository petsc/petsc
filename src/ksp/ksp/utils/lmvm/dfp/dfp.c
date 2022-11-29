#include <../src/ksp/ksp/utils/lmvm/symbrdn/symbrdn.h> /*I "petscksp.h" I*/
#include <../src/ksp/ksp/utils/lmvm/diagbrdn/diagbrdn.h>

/*
  Limited-memory Davidon-Fletcher-Powell method for approximating both
  the forward product and inverse application of a Jacobian.
 */

/*------------------------------------------------------------*/

/*
  The solution method (approximate inverse Jacobian application) is
  matrix-vector product version of the recursive formula given in
  Equation (6.15) of Nocedal and Wright "Numerical Optimization" 2nd
  edition, pg 139.

  Note: Q[i] = (B_i)^{-1}*S[i] terms are computed ahead of time whenever
  the matrix is updated with a new (S[i], Y[i]) pair. This allows
  repeated calls of MatSolve without incurring redundant computation.

  dX <- J0^{-1} * F

  for i = 0,1,2,...,k
    # Q[i] = (B_i)^{-1} * Y[i]
    gamma = (S[i]^T F) / (Y[i]^T S[i])
    zeta = (Y[i]^T dX) / (Y[i]^T Q[i])
    dX <- dX + (gamma * S[i]) - (zeta * Q[i])
  end
*/
PetscErrorCode MatSolve_LMVMDFP(Mat B, Vec F, Vec dX)
{
  Mat_LMVM    *lmvm = (Mat_LMVM *)B->data;
  Mat_SymBrdn *ldfp = (Mat_SymBrdn *)lmvm->ctx;
  PetscInt     i, j;
  PetscScalar  yjtqi, sjtyi, ytx, stf, ytq;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(F, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(dX, VEC_CLASSID, 3);
  VecCheckSameSize(F, 2, dX, 3);
  VecCheckMatCompatible(B, dX, 3, F, 2);

  if (ldfp->needQ) {
    /* Start the loop for (Q[k] = (B_k)^{-1} * Y[k]) */
    for (i = 0; i <= lmvm->k; ++i) {
      PetscCall(MatSymBrdnApplyJ0Inv(B, lmvm->Y[i], ldfp->Q[i]));
      for (j = 0; j <= i - 1; ++j) {
        /* Compute the necessary dot products */
        PetscCall(VecDotBegin(lmvm->Y[j], ldfp->Q[i], &yjtqi));
        PetscCall(VecDotBegin(lmvm->S[j], lmvm->Y[i], &sjtyi));
        PetscCall(VecDotEnd(lmvm->Y[j], ldfp->Q[i], &yjtqi));
        PetscCall(VecDotEnd(lmvm->S[j], lmvm->Y[i], &sjtyi));
        /* Compute the pure DFP component of the inverse application*/
        PetscCall(VecAXPBYPCZ(ldfp->Q[i], -PetscRealPart(yjtqi) / ldfp->ytq[j], PetscRealPart(sjtyi) / ldfp->yts[j], 1.0, ldfp->Q[j], lmvm->S[j]));
      }
      PetscCall(VecDot(lmvm->Y[i], ldfp->Q[i], &ytq));
      ldfp->ytq[i] = PetscRealPart(ytq);
    }
    ldfp->needQ = PETSC_FALSE;
  }

  /* Start the outer loop (i) for the recursive formula */
  PetscCall(MatSymBrdnApplyJ0Inv(B, F, dX));
  for (i = 0; i <= lmvm->k; ++i) {
    /* Get all the dot products we need */
    PetscCall(VecDotBegin(lmvm->Y[i], dX, &ytx));
    PetscCall(VecDotBegin(lmvm->S[i], F, &stf));
    PetscCall(VecDotEnd(lmvm->Y[i], dX, &ytx));
    PetscCall(VecDotEnd(lmvm->S[i], F, &stf));
    /* Update dX_{i+1} = (B^{-1})_{i+1} * f */
    PetscCall(VecAXPBYPCZ(dX, -PetscRealPart(ytx) / ldfp->ytq[i], PetscRealPart(stf) / ldfp->yts[i], 1.0, ldfp->Q[i], lmvm->S[i]));
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/*
  The forward product for the approximate Jacobian is the matrix-free
  implementation of the recursive formula given in Equation 6.13 of
  Nocedal and Wright "Numerical Optimization" 2nd edition, pg 139.

  This forward product has a two-loop form similar to the BFGS two-loop
  formulation for the inverse Jacobian application. However, the S and
  Y vectors have interchanged roles.

  work <- X

  for i = k,k-1,k-2,...,0
    rho[i] = 1 / (Y[i]^T S[i])
    alpha[i] = rho[i] * (Y[i]^T work)
    work <- work - (alpha[i] * S[i])
  end

  Z <- J0 * work

  for i = 0,1,2,...,k
    beta = rho[i] * (S[i]^T Y)
    Z <- Z + ((alpha[i] - beta) * Y[i])
  end
*/
PetscErrorCode MatMult_LMVMDFP(Mat B, Vec X, Vec Z)
{
  Mat_LMVM    *lmvm = (Mat_LMVM *)B->data;
  Mat_SymBrdn *ldfp = (Mat_SymBrdn *)lmvm->ctx;
  PetscInt     i;
  PetscReal   *alpha, beta;
  PetscScalar  ytx, stz;

  PetscFunctionBegin;
  /* Copy the function into the work vector for the first loop */
  PetscCall(VecCopy(X, ldfp->work));

  /* Start the first loop */
  PetscCall(PetscMalloc1(lmvm->k + 1, &alpha));
  for (i = lmvm->k; i >= 0; --i) {
    PetscCall(VecDot(lmvm->Y[i], ldfp->work, &ytx));
    alpha[i] = PetscRealPart(ytx) / ldfp->yts[i];
    PetscCall(VecAXPY(ldfp->work, -alpha[i], lmvm->S[i]));
  }

  /* Apply the forward product with initial Jacobian */
  PetscCall(MatSymBrdnApplyJ0Fwd(B, ldfp->work, Z));

  /* Start the second loop */
  for (i = 0; i <= lmvm->k; ++i) {
    PetscCall(VecDot(lmvm->S[i], Z, &stz));
    beta = PetscRealPart(stz) / ldfp->yts[i];
    PetscCall(VecAXPY(Z, alpha[i] - beta, lmvm->Y[i]));
  }
  PetscCall(PetscFree(alpha));
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatUpdate_LMVMDFP(Mat B, Vec X, Vec F)
{
  Mat_LMVM     *lmvm = (Mat_LMVM *)B->data;
  Mat_SymBrdn  *ldfp = (Mat_SymBrdn *)lmvm->ctx;
  Mat_LMVM     *dbase;
  Mat_DiagBrdn *dctx;
  PetscInt      old_k, i;
  PetscReal     curvtol;
  PetscScalar   curvature, ytytmp, ststmp;

  PetscFunctionBegin;
  if (!lmvm->m) PetscFunctionReturn(0);
  if (lmvm->prev_set) {
    /* Compute the new (S = X - Xprev) and (Y = F - Fprev) vectors */
    PetscCall(VecAYPX(lmvm->Xprev, -1.0, X));
    PetscCall(VecAYPX(lmvm->Fprev, -1.0, F));
    /* Test if the updates can be accepted */
    PetscCall(VecDotBegin(lmvm->Xprev, lmvm->Fprev, &curvature));
    PetscCall(VecDotBegin(lmvm->Xprev, lmvm->Xprev, &ststmp));
    PetscCall(VecDotEnd(lmvm->Xprev, lmvm->Fprev, &curvature));
    PetscCall(VecDotEnd(lmvm->Xprev, lmvm->Xprev, &ststmp));
    if (PetscRealPart(ststmp) < lmvm->eps) {
      curvtol = 0.0;
    } else {
      curvtol = lmvm->eps * PetscRealPart(ststmp);
    }
    if (PetscRealPart(curvature) > curvtol) {
      /* Update is good, accept it */
      ldfp->watchdog = 0;
      ldfp->needQ    = PETSC_TRUE;
      old_k          = lmvm->k;
      PetscCall(MatUpdateKernel_LMVM(B, lmvm->Xprev, lmvm->Fprev));
      /* If we hit the memory limit, shift the yts, yty and sts arrays */
      if (old_k == lmvm->k) {
        for (i = 0; i <= lmvm->k - 1; ++i) {
          ldfp->yts[i] = ldfp->yts[i + 1];
          ldfp->yty[i] = ldfp->yty[i + 1];
          ldfp->sts[i] = ldfp->sts[i + 1];
        }
      }
      /* Update history of useful scalars */
      PetscCall(VecDot(lmvm->Y[lmvm->k], lmvm->Y[lmvm->k], &ytytmp));
      ldfp->yts[lmvm->k] = PetscRealPart(curvature);
      ldfp->yty[lmvm->k] = PetscRealPart(ytytmp);
      ldfp->sts[lmvm->k] = PetscRealPart(ststmp);
      /* Compute the scalar scale if necessary */
      if (ldfp->scale_type == MAT_LMVM_SYMBROYDEN_SCALE_SCALAR) PetscCall(MatSymBrdnComputeJ0Scalar(B));
    } else {
      /* Update is bad, skip it */
      ++lmvm->nrejects;
      ++ldfp->watchdog;
    }
  } else {
    switch (ldfp->scale_type) {
    case MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL:
      dbase = (Mat_LMVM *)ldfp->D->data;
      dctx  = (Mat_DiagBrdn *)dbase->ctx;
      PetscCall(VecSet(dctx->invD, ldfp->delta));
      break;
    case MAT_LMVM_SYMBROYDEN_SCALE_SCALAR:
      ldfp->sigma = ldfp->delta;
      break;
    case MAT_LMVM_SYMBROYDEN_SCALE_NONE:
      ldfp->sigma = 1.0;
      break;
    default:
      break;
    }
  }

  /* Update the scaling */
  if (ldfp->scale_type == MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL) PetscCall(MatLMVMUpdate(ldfp->D, X, F));

  if (ldfp->watchdog > ldfp->max_seq_rejects) {
    PetscCall(MatLMVMReset(B, PETSC_FALSE));
    if (ldfp->scale_type == MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL) PetscCall(MatLMVMReset(ldfp->D, PETSC_FALSE));
  }

  /* Save the solution and function to be used in the next update */
  PetscCall(VecCopy(X, lmvm->Xprev));
  PetscCall(VecCopy(F, lmvm->Fprev));
  lmvm->prev_set = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatCopy_LMVMDFP(Mat B, Mat M, MatStructure str)
{
  Mat_LMVM    *bdata = (Mat_LMVM *)B->data;
  Mat_SymBrdn *bctx  = (Mat_SymBrdn *)bdata->ctx;
  Mat_LMVM    *mdata = (Mat_LMVM *)M->data;
  Mat_SymBrdn *mctx  = (Mat_SymBrdn *)mdata->ctx;
  PetscInt     i;

  PetscFunctionBegin;
  mctx->needQ = bctx->needQ;
  for (i = 0; i <= bdata->k; ++i) {
    mctx->ytq[i] = bctx->ytq[i];
    mctx->yts[i] = bctx->yts[i];
    PetscCall(VecCopy(bctx->Q[i], mctx->Q[i]));
  }
  mctx->scale_type      = bctx->scale_type;
  mctx->alpha           = bctx->alpha;
  mctx->beta            = bctx->beta;
  mctx->rho             = bctx->rho;
  mctx->sigma_hist      = bctx->sigma_hist;
  mctx->watchdog        = bctx->watchdog;
  mctx->max_seq_rejects = bctx->max_seq_rejects;
  switch (bctx->scale_type) {
  case MAT_LMVM_SYMBROYDEN_SCALE_SCALAR:
    mctx->sigma = bctx->sigma;
    break;
  case MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL:
    PetscCall(MatCopy(bctx->D, mctx->D, SAME_NONZERO_PATTERN));
    break;
  case MAT_LMVM_SYMBROYDEN_SCALE_NONE:
    mctx->sigma = 1.0;
    break;
  default:
    break;
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatReset_LMVMDFP(Mat B, PetscBool destructive)
{
  Mat_LMVM     *lmvm = (Mat_LMVM *)B->data;
  Mat_SymBrdn  *ldfp = (Mat_SymBrdn *)lmvm->ctx;
  Mat_LMVM     *dbase;
  Mat_DiagBrdn *dctx;

  PetscFunctionBegin;
  ldfp->watchdog = 0;
  ldfp->needQ    = PETSC_TRUE;
  if (ldfp->allocated) {
    if (destructive) {
      PetscCall(VecDestroy(&ldfp->work));
      PetscCall(PetscFree4(ldfp->ytq, ldfp->yts, ldfp->yty, ldfp->sts));
      PetscCall(VecDestroyVecs(lmvm->m, &ldfp->Q));
      switch (ldfp->scale_type) {
      case MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL:
        PetscCall(MatLMVMReset(ldfp->D, PETSC_TRUE));
        break;
      default:
        break;
      }
      ldfp->allocated = PETSC_FALSE;
    } else {
      switch (ldfp->scale_type) {
      case MAT_LMVM_SYMBROYDEN_SCALE_SCALAR:
        ldfp->sigma = ldfp->delta;
        break;
      case MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL:
        PetscCall(MatLMVMReset(ldfp->D, PETSC_FALSE));
        dbase = (Mat_LMVM *)ldfp->D->data;
        dctx  = (Mat_DiagBrdn *)dbase->ctx;
        PetscCall(VecSet(dctx->invD, ldfp->delta));
        break;
      case MAT_LMVM_SYMBROYDEN_SCALE_NONE:
        ldfp->sigma = 1.0;
        break;
      default:
        break;
      }
    }
  }
  PetscCall(MatReset_LMVM(B, destructive));
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatAllocate_LMVMDFP(Mat B, Vec X, Vec F)
{
  Mat_LMVM    *lmvm = (Mat_LMVM *)B->data;
  Mat_SymBrdn *ldfp = (Mat_SymBrdn *)lmvm->ctx;

  PetscFunctionBegin;
  PetscCall(MatAllocate_LMVM(B, X, F));
  if (!ldfp->allocated) {
    PetscCall(VecDuplicate(X, &ldfp->work));
    PetscCall(PetscMalloc4(lmvm->m, &ldfp->ytq, lmvm->m, &ldfp->yts, lmvm->m, &ldfp->yty, lmvm->m, &ldfp->sts));
    if (lmvm->m > 0) PetscCall(VecDuplicateVecs(X, lmvm->m, &ldfp->Q));
    switch (ldfp->scale_type) {
    case MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL:
      PetscCall(MatLMVMAllocate(ldfp->D, X, F));
      break;
    default:
      break;
    }
    ldfp->allocated = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatDestroy_LMVMDFP(Mat B)
{
  Mat_LMVM    *lmvm = (Mat_LMVM *)B->data;
  Mat_SymBrdn *ldfp = (Mat_SymBrdn *)lmvm->ctx;

  PetscFunctionBegin;
  if (ldfp->allocated) {
    PetscCall(VecDestroy(&ldfp->work));
    PetscCall(PetscFree4(ldfp->ytq, ldfp->yts, ldfp->yty, ldfp->sts));
    PetscCall(VecDestroyVecs(lmvm->m, &ldfp->Q));
    ldfp->allocated = PETSC_FALSE;
  }
  PetscCall(MatDestroy(&ldfp->D));
  PetscCall(PetscFree(lmvm->ctx));
  PetscCall(MatDestroy_LMVM(B));
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatSetUp_LMVMDFP(Mat B)
{
  Mat_LMVM    *lmvm = (Mat_LMVM *)B->data;
  Mat_SymBrdn *ldfp = (Mat_SymBrdn *)lmvm->ctx;
  PetscInt     n, N;

  PetscFunctionBegin;
  PetscCall(MatSetUp_LMVM(B));
  if (!ldfp->allocated) {
    PetscCall(VecDuplicate(lmvm->Xprev, &ldfp->work));
    PetscCall(PetscMalloc4(lmvm->m, &ldfp->ytq, lmvm->m, &ldfp->yts, lmvm->m, &ldfp->yty, lmvm->m, &ldfp->sts));
    if (lmvm->m > 0) PetscCall(VecDuplicateVecs(lmvm->Xprev, lmvm->m, &ldfp->Q));
    switch (ldfp->scale_type) {
    case MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL:
      PetscCall(MatGetLocalSize(B, &n, &n));
      PetscCall(MatGetSize(B, &N, &N));
      PetscCall(MatSetSizes(ldfp->D, n, n, N, N));
      PetscCall(MatSetUp(ldfp->D));
      break;
    default:
      break;
    }
    ldfp->allocated = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatSetFromOptions_LMVMDFP(Mat B, PetscOptionItems *PetscOptionsObject)
{
  PetscFunctionBegin;
  PetscCall(MatSetFromOptions_LMVM(B, PetscOptionsObject));
  PetscOptionsHeadBegin(PetscOptionsObject, "DFP method for approximating SPD Jacobian actions (MATLMVMDFP)");
  PetscCall(MatSetFromOptions_LMVMSymBrdn_Private(B, PetscOptionsObject));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode MatCreate_LMVMDFP(Mat B)
{
  Mat_LMVM    *lmvm;
  Mat_SymBrdn *ldfp;

  PetscFunctionBegin;
  PetscCall(MatCreate_LMVMSymBrdn(B));
  PetscCall(PetscObjectChangeTypeName((PetscObject)B, MATLMVMDFP));
  B->ops->setup          = MatSetUp_LMVMDFP;
  B->ops->destroy        = MatDestroy_LMVMDFP;
  B->ops->setfromoptions = MatSetFromOptions_LMVMDFP;
  B->ops->solve          = MatSolve_LMVMDFP;

  lmvm                = (Mat_LMVM *)B->data;
  lmvm->ops->allocate = MatAllocate_LMVMDFP;
  lmvm->ops->reset    = MatReset_LMVMDFP;
  lmvm->ops->update   = MatUpdate_LMVMDFP;
  lmvm->ops->mult     = MatMult_LMVMDFP;
  lmvm->ops->copy     = MatCopy_LMVMDFP;

  ldfp        = (Mat_SymBrdn *)lmvm->ctx;
  ldfp->needP = PETSC_FALSE;
  ldfp->phi   = 1.0;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/*@
   MatCreateLMVMDFP - Creates a limited-memory Davidon-Fletcher-Powell (DFP) matrix
   used for approximating Jacobians. L-DFP is symmetric positive-definite by
   construction, and is the dual of L-BFGS where Y and S vectors swap roles.

   The provided local and global sizes must match the solution and function vectors
   used with `MatLMVMUpdate()` and `MatSolve()`. The resulting L-DFP matrix will have
   storage vectors allocated with `VecCreateSeq()` in serial and `VecCreateMPI()` in
   parallel. To use the L-DFP matrix with other vector types, the matrix must be
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

   Options Database Keys:
+   -mat_lmvm_scale_type - (developer) type of scaling applied to J0 (none, scalar, diagonal)
.   -mat_lmvm_theta - (developer) convex ratio between BFGS and DFP components of the diagonal J0 scaling
.   -mat_lmvm_rho - (developer) update limiter for the J0 scaling
.   -mat_lmvm_alpha - (developer) coefficient factor for the quadratic subproblem in J0 scaling
.   -mat_lmvm_beta - (developer) exponential factor for the diagonal J0 scaling
-   -mat_lmvm_sigma_hist - (developer) number of past updates to use in J0 scaling

   Level: intermediate

   Note:
   It is recommended that one use the `MatCreate()`, `MatSetType()` and/or `MatSetFromOptions()`
   paradigm instead of this routine directly.

.seealso: [](chapter_ksp), `MatCreate()`, `MATLMVM`, `MATLMVMDFP`, `MatCreateLMVMBFGS()`, `MatCreateLMVMSR1()`,
          `MatCreateLMVMBrdn()`, `MatCreateLMVMBadBrdn()`, `MatCreateLMVMSymBrdn()`
@*/
PetscErrorCode MatCreateLMVMDFP(MPI_Comm comm, PetscInt n, PetscInt N, Mat *B)
{
  PetscFunctionBegin;
  PetscCall(MatCreate(comm, B));
  PetscCall(MatSetSizes(*B, n, n, N, N));
  PetscCall(MatSetType(*B, MATLMVMDFP));
  PetscCall(MatSetUp(*B));
  PetscFunctionReturn(0);
}
