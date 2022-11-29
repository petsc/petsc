#include <../src/ksp/ksp/utils/lmvm/symbrdn/symbrdn.h> /*I "petscksp.h" I*/
#include <../src/ksp/ksp/utils/lmvm/diagbrdn/diagbrdn.h>

/*
  Limited-memory Broyden-Fletcher-Goldfarb-Shano method for approximating both
  the forward product and inverse application of a Jacobian.
*/

/*------------------------------------------------------------*/

/*
  The solution method (approximate inverse Jacobian application) is adapted
   from Algorithm 7.4 on page 178 of Nocedal and Wright "Numerical Optimization"
   2nd edition (https://doi.org/10.1007/978-0-387-40065-5). The initial inverse
   Jacobian application falls back onto the gamma scaling recommended in equation
   (7.20) if the user has not provided any estimation of the initial Jacobian or
   its inverse.

   work <- F

   for i = k,k-1,k-2,...,0
     rho[i] = 1 / (Y[i]^T S[i])
     alpha[i] = rho[i] * (S[i]^T work)
     Fwork <- work - (alpha[i] * Y[i])
   end

   dX <- J0^{-1} * work

   for i = 0,1,2,...,k
     beta = rho[i] * (Y[i]^T dX)
     dX <- dX + ((alpha[i] - beta) * S[i])
   end
*/
PetscErrorCode MatSolve_LMVMBFGS(Mat B, Vec F, Vec dX)
{
  Mat_LMVM    *lmvm  = (Mat_LMVM *)B->data;
  Mat_SymBrdn *lbfgs = (Mat_SymBrdn *)lmvm->ctx;
  PetscInt     i;
  PetscReal   *alpha, beta;
  PetscScalar  stf, ytx;

  PetscFunctionBegin;
  VecCheckSameSize(F, 2, dX, 3);
  VecCheckMatCompatible(B, dX, 3, F, 2);

  /* Copy the function into the work vector for the first loop */
  PetscCall(VecCopy(F, lbfgs->work));

  /* Start the first loop */
  PetscCall(PetscMalloc1(lmvm->k + 1, &alpha));
  for (i = lmvm->k; i >= 0; --i) {
    PetscCall(VecDot(lmvm->S[i], lbfgs->work, &stf));
    alpha[i] = PetscRealPart(stf) / lbfgs->yts[i];
    PetscCall(VecAXPY(lbfgs->work, -alpha[i], lmvm->Y[i]));
  }

  /* Invert the initial Jacobian onto the work vector (or apply scaling) */
  PetscCall(MatSymBrdnApplyJ0Inv(B, lbfgs->work, dX));

  /* Start the second loop */
  for (i = 0; i <= lmvm->k; ++i) {
    PetscCall(VecDot(lmvm->Y[i], dX, &ytx));
    beta = PetscRealPart(ytx) / lbfgs->yts[i];
    PetscCall(VecAXPY(dX, alpha[i] - beta, lmvm->S[i]));
  }
  PetscCall(PetscFree(alpha));
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/*
  The forward product for the approximate Jacobian is the matrix-free
  implementation of Equation (6.19) in Nocedal and Wright "Numerical
  Optimization" 2nd Edition, pg 140.

  This forward product has the same structure as the inverse Jacobian
  application in the DFP formulation, except with S and Y exchanging
  roles.

  Note: P[i] = (B_i)*S[i] terms are computed ahead of time whenever
  the matrix is updated with a new (S[i], Y[i]) pair. This allows
  repeated calls of MatMult inside KSP solvers without unnecessarily
  recomputing P[i] terms in expensive nested-loops.

  Z <- J0 * X

  for i = 0,1,2,...,k
    P[i] <- J0 * S[i]
    for j = 0,1,2,...,(i-1)
      gamma = (Y[j]^T S[i]) / (Y[j]^T S[j])
      zeta = (S[j]^ P[i]) / (S[j]^T P[j])
      P[i] <- P[i] - (zeta * P[j]) + (gamma * Y[j])
    end
    gamma = (Y[i]^T X) / (Y[i]^T S[i])
    zeta = (S[i]^T Z) / (S[i]^T P[i])
    Z <- Z - (zeta * P[i]) + (gamma * Y[i])
  end
*/
PetscErrorCode MatMult_LMVMBFGS(Mat B, Vec X, Vec Z)
{
  Mat_LMVM    *lmvm  = (Mat_LMVM *)B->data;
  Mat_SymBrdn *lbfgs = (Mat_SymBrdn *)lmvm->ctx;
  PetscInt     i, j;
  PetscScalar  sjtpi, yjtsi, ytx, stz, stp;

  PetscFunctionBegin;
  VecCheckSameSize(X, 2, Z, 3);
  VecCheckMatCompatible(B, X, 2, Z, 3);

  if (lbfgs->needP) {
    /* Pre-compute (P[i] = B_i * S[i]) */
    for (i = 0; i <= lmvm->k; ++i) {
      PetscCall(MatSymBrdnApplyJ0Fwd(B, lmvm->S[i], lbfgs->P[i]));
      for (j = 0; j <= i - 1; ++j) {
        /* Compute the necessary dot products */
        PetscCall(VecDotBegin(lmvm->S[j], lbfgs->P[i], &sjtpi));
        PetscCall(VecDotBegin(lmvm->Y[j], lmvm->S[i], &yjtsi));
        PetscCall(VecDotEnd(lmvm->S[j], lbfgs->P[i], &sjtpi));
        PetscCall(VecDotEnd(lmvm->Y[j], lmvm->S[i], &yjtsi));
        /* Compute the pure BFGS component of the forward product */
        PetscCall(VecAXPBYPCZ(lbfgs->P[i], -PetscRealPart(sjtpi) / lbfgs->stp[j], PetscRealPart(yjtsi) / lbfgs->yts[j], 1.0, lbfgs->P[j], lmvm->Y[j]));
      }
      PetscCall(VecDot(lmvm->S[i], lbfgs->P[i], &stp));
      lbfgs->stp[i] = PetscRealPart(stp);
    }
    lbfgs->needP = PETSC_FALSE;
  }

  /* Start the outer loop (i) for the recursive formula */
  PetscCall(MatSymBrdnApplyJ0Fwd(B, X, Z));
  for (i = 0; i <= lmvm->k; ++i) {
    /* Get all the dot products we need */
    PetscCall(VecDotBegin(lmvm->S[i], Z, &stz));
    PetscCall(VecDotBegin(lmvm->Y[i], X, &ytx));
    PetscCall(VecDotEnd(lmvm->S[i], Z, &stz));
    PetscCall(VecDotEnd(lmvm->Y[i], X, &ytx));
    /* Update Z_{i+1} = B_{i+1} * X */
    PetscCall(VecAXPBYPCZ(Z, -PetscRealPart(stz) / lbfgs->stp[i], PetscRealPart(ytx) / lbfgs->yts[i], 1.0, lbfgs->P[i], lmvm->Y[i]));
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatUpdate_LMVMBFGS(Mat B, Vec X, Vec F)
{
  Mat_LMVM     *lmvm  = (Mat_LMVM *)B->data;
  Mat_SymBrdn  *lbfgs = (Mat_SymBrdn *)lmvm->ctx;
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
      lbfgs->watchdog = 0;
      lbfgs->needP    = PETSC_TRUE;
      old_k           = lmvm->k;
      PetscCall(MatUpdateKernel_LMVM(B, lmvm->Xprev, lmvm->Fprev));
      /* If we hit the memory limit, shift the yts, yty and sts arrays */
      if (old_k == lmvm->k) {
        for (i = 0; i <= lmvm->k - 1; ++i) {
          lbfgs->yts[i] = lbfgs->yts[i + 1];
          lbfgs->yty[i] = lbfgs->yty[i + 1];
          lbfgs->sts[i] = lbfgs->sts[i + 1];
        }
      }
      /* Update history of useful scalars */
      PetscCall(VecDot(lmvm->Y[lmvm->k], lmvm->Y[lmvm->k], &ytytmp));
      lbfgs->yts[lmvm->k] = PetscRealPart(curvature);
      lbfgs->yty[lmvm->k] = PetscRealPart(ytytmp);
      lbfgs->sts[lmvm->k] = PetscRealPart(ststmp);
      /* Compute the scalar scale if necessary */
      if (lbfgs->scale_type == MAT_LMVM_SYMBROYDEN_SCALE_SCALAR) PetscCall(MatSymBrdnComputeJ0Scalar(B));
    } else {
      /* Update is bad, skip it */
      ++lmvm->nrejects;
      ++lbfgs->watchdog;
    }
  } else {
    switch (lbfgs->scale_type) {
    case MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL:
      dbase = (Mat_LMVM *)lbfgs->D->data;
      dctx  = (Mat_DiagBrdn *)dbase->ctx;
      PetscCall(VecSet(dctx->invD, lbfgs->delta));
      break;
    case MAT_LMVM_SYMBROYDEN_SCALE_SCALAR:
      lbfgs->sigma = lbfgs->delta;
      break;
    case MAT_LMVM_SYMBROYDEN_SCALE_NONE:
      lbfgs->sigma = 1.0;
      break;
    default:
      break;
    }
  }

  /* Update the scaling */
  if (lbfgs->scale_type == MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL) PetscCall(MatLMVMUpdate(lbfgs->D, X, F));

  if (lbfgs->watchdog > lbfgs->max_seq_rejects) {
    PetscCall(MatLMVMReset(B, PETSC_FALSE));
    if (lbfgs->scale_type == MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL) PetscCall(MatLMVMReset(lbfgs->D, PETSC_FALSE));
  }

  /* Save the solution and function to be used in the next update */
  PetscCall(VecCopy(X, lmvm->Xprev));
  PetscCall(VecCopy(F, lmvm->Fprev));
  lmvm->prev_set = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatCopy_LMVMBFGS(Mat B, Mat M, MatStructure str)
{
  Mat_LMVM    *bdata = (Mat_LMVM *)B->data;
  Mat_SymBrdn *bctx  = (Mat_SymBrdn *)bdata->ctx;
  Mat_LMVM    *mdata = (Mat_LMVM *)M->data;
  Mat_SymBrdn *mctx  = (Mat_SymBrdn *)mdata->ctx;
  PetscInt     i;

  PetscFunctionBegin;
  mctx->needP = bctx->needP;
  for (i = 0; i <= bdata->k; ++i) {
    mctx->stp[i] = bctx->stp[i];
    mctx->yts[i] = bctx->yts[i];
    PetscCall(VecCopy(bctx->P[i], mctx->P[i]));
  }
  mctx->scale_type      = bctx->scale_type;
  mctx->alpha           = bctx->alpha;
  mctx->beta            = bctx->beta;
  mctx->rho             = bctx->rho;
  mctx->delta           = bctx->delta;
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

static PetscErrorCode MatReset_LMVMBFGS(Mat B, PetscBool destructive)
{
  Mat_LMVM     *lmvm  = (Mat_LMVM *)B->data;
  Mat_SymBrdn  *lbfgs = (Mat_SymBrdn *)lmvm->ctx;
  Mat_LMVM     *dbase;
  Mat_DiagBrdn *dctx;

  PetscFunctionBegin;
  lbfgs->watchdog = 0;
  lbfgs->needP    = PETSC_TRUE;
  if (lbfgs->allocated) {
    if (destructive) {
      PetscCall(VecDestroy(&lbfgs->work));
      PetscCall(PetscFree4(lbfgs->stp, lbfgs->yts, lbfgs->yty, lbfgs->sts));
      PetscCall(VecDestroyVecs(lmvm->m, &lbfgs->P));
      switch (lbfgs->scale_type) {
      case MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL:
        PetscCall(MatLMVMReset(lbfgs->D, PETSC_TRUE));
        break;
      default:
        break;
      }
      lbfgs->allocated = PETSC_FALSE;
    } else {
      switch (lbfgs->scale_type) {
      case MAT_LMVM_SYMBROYDEN_SCALE_SCALAR:
        lbfgs->sigma = lbfgs->delta;
        break;
      case MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL:
        PetscCall(MatLMVMReset(lbfgs->D, PETSC_FALSE));
        dbase = (Mat_LMVM *)lbfgs->D->data;
        dctx  = (Mat_DiagBrdn *)dbase->ctx;
        PetscCall(VecSet(dctx->invD, lbfgs->delta));
        break;
      case MAT_LMVM_SYMBROYDEN_SCALE_NONE:
        lbfgs->sigma = 1.0;
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

static PetscErrorCode MatAllocate_LMVMBFGS(Mat B, Vec X, Vec F)
{
  Mat_LMVM    *lmvm  = (Mat_LMVM *)B->data;
  Mat_SymBrdn *lbfgs = (Mat_SymBrdn *)lmvm->ctx;

  PetscFunctionBegin;
  PetscCall(MatAllocate_LMVM(B, X, F));
  if (!lbfgs->allocated) {
    PetscCall(VecDuplicate(X, &lbfgs->work));
    PetscCall(PetscMalloc4(lmvm->m, &lbfgs->stp, lmvm->m, &lbfgs->yts, lmvm->m, &lbfgs->yty, lmvm->m, &lbfgs->sts));
    if (lmvm->m > 0) PetscCall(VecDuplicateVecs(X, lmvm->m, &lbfgs->P));
    switch (lbfgs->scale_type) {
    case MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL:
      PetscCall(MatLMVMAllocate(lbfgs->D, X, F));
      break;
    default:
      break;
    }
    lbfgs->allocated = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatDestroy_LMVMBFGS(Mat B)
{
  Mat_LMVM    *lmvm  = (Mat_LMVM *)B->data;
  Mat_SymBrdn *lbfgs = (Mat_SymBrdn *)lmvm->ctx;

  PetscFunctionBegin;
  if (lbfgs->allocated) {
    PetscCall(VecDestroy(&lbfgs->work));
    PetscCall(PetscFree4(lbfgs->stp, lbfgs->yts, lbfgs->yty, lbfgs->sts));
    PetscCall(VecDestroyVecs(lmvm->m, &lbfgs->P));
    lbfgs->allocated = PETSC_FALSE;
  }
  PetscCall(MatDestroy(&lbfgs->D));
  PetscCall(PetscFree(lmvm->ctx));
  PetscCall(MatDestroy_LMVM(B));
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatSetUp_LMVMBFGS(Mat B)
{
  Mat_LMVM    *lmvm  = (Mat_LMVM *)B->data;
  Mat_SymBrdn *lbfgs = (Mat_SymBrdn *)lmvm->ctx;
  PetscInt     n, N;

  PetscFunctionBegin;
  PetscCall(MatSetUp_LMVM(B));
  lbfgs->max_seq_rejects = lmvm->m / 2;
  if (!lbfgs->allocated) {
    PetscCall(VecDuplicate(lmvm->Xprev, &lbfgs->work));
    PetscCall(PetscMalloc4(lmvm->m, &lbfgs->stp, lmvm->m, &lbfgs->yts, lmvm->m, &lbfgs->yty, lmvm->m, &lbfgs->sts));
    if (lmvm->m > 0) PetscCall(VecDuplicateVecs(lmvm->Xprev, lmvm->m, &lbfgs->P));
    switch (lbfgs->scale_type) {
    case MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL:
      PetscCall(MatGetLocalSize(B, &n, &n));
      PetscCall(MatGetSize(B, &N, &N));
      PetscCall(MatSetSizes(lbfgs->D, n, n, N, N));
      PetscCall(MatSetUp(lbfgs->D));
      break;
    default:
      break;
    }
    lbfgs->allocated = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatSetFromOptions_LMVMBFGS(Mat B, PetscOptionItems *PetscOptionsObject)
{
  PetscFunctionBegin;
  PetscCall(MatSetFromOptions_LMVM(B, PetscOptionsObject));
  PetscOptionsHeadBegin(PetscOptionsObject, "L-BFGS method for approximating SPD Jacobian actions (MATLMVMBFGS)");
  PetscCall(MatSetFromOptions_LMVMSymBrdn_Private(B, PetscOptionsObject));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode MatCreate_LMVMBFGS(Mat B)
{
  Mat_LMVM    *lmvm;
  Mat_SymBrdn *lbfgs;

  PetscFunctionBegin;
  PetscCall(MatCreate_LMVMSymBrdn(B));
  PetscCall(PetscObjectChangeTypeName((PetscObject)B, MATLMVMBFGS));
  B->ops->setup          = MatSetUp_LMVMBFGS;
  B->ops->destroy        = MatDestroy_LMVMBFGS;
  B->ops->setfromoptions = MatSetFromOptions_LMVMBFGS;
  B->ops->solve          = MatSolve_LMVMBFGS;

  lmvm                = (Mat_LMVM *)B->data;
  lmvm->ops->allocate = MatAllocate_LMVMBFGS;
  lmvm->ops->reset    = MatReset_LMVMBFGS;
  lmvm->ops->update   = MatUpdate_LMVMBFGS;
  lmvm->ops->mult     = MatMult_LMVMBFGS;
  lmvm->ops->copy     = MatCopy_LMVMBFGS;

  lbfgs        = (Mat_SymBrdn *)lmvm->ctx;
  lbfgs->needQ = PETSC_FALSE;
  lbfgs->phi   = 0.0;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/*@
   MatCreateLMVMBFGS - Creates a limited-memory Broyden-Fletcher-Goldfarb-Shano (BFGS)
   matrix used for approximating Jacobians. L-BFGS is symmetric positive-definite by
   construction, and is commonly used to approximate Hessians in optimization
   problems.

   The provided local and global sizes must match the solution and function vectors
   used with `MatLMVMUpdate()` and `MatSolve()`. The resulting L-BFGS matrix will have
   storage vectors allocated with `VecCreateSeq()` in serial and `VecCreateMPI()` in
   parallel. To use the L-BFGS matrix with other vector types, the matrix must be
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

.seealso: [](chapter_ksp), `MatCreate()`, `MATLMVM`, `MATLMVMBFGS`, `MatCreateLMVMDFP()`, `MatCreateLMVMSR1()`,
          `MatCreateLMVMBrdn()`, `MatCreateLMVMBadBrdn()`, `MatCreateLMVMSymBrdn()`
@*/
PetscErrorCode MatCreateLMVMBFGS(MPI_Comm comm, PetscInt n, PetscInt N, Mat *B)
{
  PetscFunctionBegin;
  PetscCall(KSPInitializePackage());
  PetscCall(MatCreate(comm, B));
  PetscCall(MatSetSizes(*B, n, n, N, N));
  PetscCall(MatSetType(*B, MATLMVMBFGS));
  PetscCall(MatSetUp(*B));
  PetscFunctionReturn(0);
}
