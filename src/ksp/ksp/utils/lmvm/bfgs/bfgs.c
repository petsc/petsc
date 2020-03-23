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
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_SymBrdn       *lbfgs = (Mat_SymBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  PetscInt          i;
  PetscReal         *alpha, beta;
  PetscScalar       stf, ytx;
  
  PetscFunctionBegin;
  VecCheckSameSize(F, 2, dX, 3);
  VecCheckMatCompatible(B, dX, 3, F, 2);
  
  /* Copy the function into the work vector for the first loop */
  ierr = VecCopy(F, lbfgs->work);CHKERRQ(ierr);
  
  /* Start the first loop */
  ierr = PetscMalloc1(lmvm->k+1, &alpha);CHKERRQ(ierr);
  for (i = lmvm->k; i >= 0; --i) {
    ierr = VecDot(lmvm->S[i], lbfgs->work, &stf);CHKERRQ(ierr);
    alpha[i] = PetscRealPart(stf)/lbfgs->yts[i];
    ierr = VecAXPY(lbfgs->work, -alpha[i], lmvm->Y[i]);CHKERRQ(ierr);
  }
  
  /* Invert the initial Jacobian onto the work vector (or apply scaling) */
  ierr = MatSymBrdnApplyJ0Inv(B, lbfgs->work, dX);CHKERRQ(ierr);
  
  /* Start the second loop */
  for (i = 0; i <= lmvm->k; ++i) {
    ierr = VecDot(lmvm->Y[i], dX, &ytx);CHKERRQ(ierr);
    beta = PetscRealPart(ytx)/lbfgs->yts[i];
    ierr = VecAXPY(dX, alpha[i]-beta, lmvm->S[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(alpha);CHKERRQ(ierr);
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
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_SymBrdn       *lbfgs = (Mat_SymBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  PetscInt          i, j;
  PetscScalar       sjtpi, yjtsi, ytx, stz, stp;
  
  PetscFunctionBegin;
  VecCheckSameSize(X, 2, Z, 3);
  VecCheckMatCompatible(B, X, 2, Z, 3);
  
  if (lbfgs->needP) {
    /* Pre-compute (P[i] = B_i * S[i]) */
    for (i = 0; i <= lmvm->k; ++i) {
      ierr = MatSymBrdnApplyJ0Fwd(B, lmvm->S[i], lbfgs->P[i]);CHKERRQ(ierr);
      for (j = 0; j <= i-1; ++j) {
        /* Compute the necessary dot products */
        ierr = VecDotBegin(lmvm->S[j], lbfgs->P[i], &sjtpi);CHKERRQ(ierr);
        ierr = VecDotBegin(lmvm->Y[j], lmvm->S[i], &yjtsi);CHKERRQ(ierr);
        ierr = VecDotEnd(lmvm->S[j], lbfgs->P[i], &sjtpi);CHKERRQ(ierr);
        ierr = VecDotEnd(lmvm->Y[j], lmvm->S[i], &yjtsi);CHKERRQ(ierr);
        /* Compute the pure BFGS component of the forward product */
        ierr = VecAXPBYPCZ(lbfgs->P[i], -PetscRealPart(sjtpi)/lbfgs->stp[j], PetscRealPart(yjtsi)/lbfgs->yts[j], 1.0, lbfgs->P[j], lmvm->Y[j]);CHKERRQ(ierr);
      }
      ierr = VecDot(lmvm->S[i], lbfgs->P[i], &stp);CHKERRQ(ierr);
      lbfgs->stp[i] = PetscRealPart(stp);
    }
    lbfgs->needP = PETSC_FALSE;
  }
  
  /* Start the outer loop (i) for the recursive formula */
  ierr = MatSymBrdnApplyJ0Fwd(B, X, Z);CHKERRQ(ierr);
  for (i = 0; i <= lmvm->k; ++i) {
    /* Get all the dot products we need */
    ierr = VecDotBegin(lmvm->S[i], Z, &stz);CHKERRQ(ierr);
    ierr = VecDotBegin(lmvm->Y[i], X, &ytx);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->S[i], Z, &stz);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->Y[i], X, &ytx);CHKERRQ(ierr);
    /* Update Z_{i+1} = B_{i+1} * X */
    ierr = VecAXPBYPCZ(Z, -PetscRealPart(stz)/lbfgs->stp[i], PetscRealPart(ytx)/lbfgs->yts[i], 1.0, lbfgs->P[i], lmvm->Y[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatUpdate_LMVMBFGS(Mat B, Vec X, Vec F)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_SymBrdn       *lbfgs = (Mat_SymBrdn*)lmvm->ctx;
  Mat_LMVM          *dbase;
  Mat_DiagBrdn      *dctx;
  PetscErrorCode    ierr;
  PetscInt          old_k, i;
  PetscReal         curvtol;
  PetscScalar       curvature, ytytmp, ststmp;

  PetscFunctionBegin;
  if (!lmvm->m) PetscFunctionReturn(0);
  if (lmvm->prev_set) {
    /* Compute the new (S = X - Xprev) and (Y = F - Fprev) vectors */
    ierr = VecAYPX(lmvm->Xprev, -1.0, X);CHKERRQ(ierr);
    ierr = VecAYPX(lmvm->Fprev, -1.0, F);CHKERRQ(ierr);
    /* Test if the updates can be accepted */
    ierr = VecDotBegin(lmvm->Xprev, lmvm->Fprev, &curvature);CHKERRQ(ierr);
    ierr = VecDotBegin(lmvm->Xprev, lmvm->Xprev, &ststmp);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->Xprev, lmvm->Fprev, &curvature);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->Xprev, lmvm->Xprev, &ststmp);CHKERRQ(ierr);
    if (PetscRealPart(ststmp) < lmvm->eps) {
      curvtol = 0.0;
    } else {
      curvtol = lmvm->eps * PetscRealPart(ststmp);
    }
    if (PetscRealPart(curvature) > curvtol) {
      /* Update is good, accept it */
      lbfgs->watchdog = 0;
      lbfgs->needP = PETSC_TRUE;
      old_k = lmvm->k;
      ierr = MatUpdateKernel_LMVM(B, lmvm->Xprev, lmvm->Fprev);CHKERRQ(ierr);
      /* If we hit the memory limit, shift the yts, yty and sts arrays */
      if (old_k == lmvm->k) {
        for (i = 0; i <= lmvm->k-1; ++i) {
          lbfgs->yts[i] = lbfgs->yts[i+1];
          lbfgs->yty[i] = lbfgs->yty[i+1];
          lbfgs->sts[i] = lbfgs->sts[i+1];
        }
      }
      /* Update history of useful scalars */
      ierr = VecDot(lmvm->Y[lmvm->k], lmvm->Y[lmvm->k], &ytytmp);CHKERRQ(ierr);
      lbfgs->yts[lmvm->k] = PetscRealPart(curvature);
      lbfgs->yty[lmvm->k] = PetscRealPart(ytytmp);
      lbfgs->sts[lmvm->k] = PetscRealPart(ststmp);
      /* Compute the scalar scale if necessary */
      if (lbfgs->scale_type == MAT_LMVM_SYMBROYDEN_SCALE_SCALAR) {
        ierr = MatSymBrdnComputeJ0Scalar(B);CHKERRQ(ierr);
      }
    } else {
      /* Update is bad, skip it */
      ++lmvm->nrejects;
      ++lbfgs->watchdog;
    }
  } else {
    switch (lbfgs->scale_type) {
    case MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL:
      dbase = (Mat_LMVM*)lbfgs->D->data;
      dctx = (Mat_DiagBrdn*)dbase->ctx;
      ierr = VecSet(dctx->invD, lbfgs->delta);CHKERRQ(ierr);
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
  if (lbfgs->scale_type == MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL) {
    ierr = MatLMVMUpdate(lbfgs->D, X, F);CHKERRQ(ierr);
  }
  
  if (lbfgs->watchdog > lbfgs->max_seq_rejects) {
    ierr = MatLMVMReset(B, PETSC_FALSE);CHKERRQ(ierr);
    if (lbfgs->scale_type == MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL) {
      ierr = MatLMVMReset(lbfgs->D, PETSC_FALSE);CHKERRQ(ierr);
    }
  }

  /* Save the solution and function to be used in the next update */
  ierr = VecCopy(X, lmvm->Xprev);CHKERRQ(ierr);
  ierr = VecCopy(F, lmvm->Fprev);CHKERRQ(ierr);
  lmvm->prev_set = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatCopy_LMVMBFGS(Mat B, Mat M, MatStructure str)
{
  Mat_LMVM          *bdata = (Mat_LMVM*)B->data;
  Mat_SymBrdn       *bctx = (Mat_SymBrdn*)bdata->ctx;
  Mat_LMVM          *mdata = (Mat_LMVM*)M->data;
  Mat_SymBrdn       *mctx = (Mat_SymBrdn*)mdata->ctx;
  PetscErrorCode    ierr;
  PetscInt          i;

  PetscFunctionBegin;
  mctx->needP = bctx->needP;
  for (i=0; i<=bdata->k; ++i) {
    mctx->stp[i] = bctx->stp[i];
    mctx->yts[i] = bctx->yts[i];
    ierr = VecCopy(bctx->P[i], mctx->P[i]);CHKERRQ(ierr);
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
    ierr = MatCopy(bctx->D, mctx->D, SAME_NONZERO_PATTERN);CHKERRQ(ierr);
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
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_SymBrdn       *lbfgs = (Mat_SymBrdn*)lmvm->ctx;
  Mat_LMVM          *dbase;
  Mat_DiagBrdn      *dctx;
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  lbfgs->watchdog = 0;
  lbfgs->needP = PETSC_TRUE;
  if (lbfgs->allocated) {
    if (destructive) {
      ierr = VecDestroy(&lbfgs->work);CHKERRQ(ierr);
      ierr = PetscFree4(lbfgs->stp, lbfgs->yts, lbfgs->yty, lbfgs->sts);CHKERRQ(ierr);
      ierr = VecDestroyVecs(lmvm->m, &lbfgs->P);CHKERRQ(ierr);
      switch (lbfgs->scale_type) {
      case MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL:
        ierr = MatLMVMReset(lbfgs->D, PETSC_TRUE);CHKERRQ(ierr);
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
        ierr = MatLMVMReset(lbfgs->D, PETSC_FALSE);CHKERRQ(ierr);
        dbase = (Mat_LMVM*)lbfgs->D->data;
        dctx = (Mat_DiagBrdn*)dbase->ctx;
        ierr = VecSet(dctx->invD, lbfgs->delta);CHKERRQ(ierr);
        break;
      case MAT_LMVM_SYMBROYDEN_SCALE_NONE:
        lbfgs->sigma = 1.0;
        break;
      default:
        break;
      }
    }
  }
  ierr = MatReset_LMVM(B, destructive);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatAllocate_LMVMBFGS(Mat B, Vec X, Vec F)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_SymBrdn       *lbfgs = (Mat_SymBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  ierr = MatAllocate_LMVM(B, X, F);CHKERRQ(ierr);
  if (!lbfgs->allocated) {
    ierr = VecDuplicate(X, &lbfgs->work);CHKERRQ(ierr);
    ierr = PetscMalloc4(lmvm->m, &lbfgs->stp, lmvm->m, &lbfgs->yts, lmvm->m, &lbfgs->yty, lmvm->m, &lbfgs->sts);CHKERRQ(ierr);
    if (lmvm->m > 0) {
      ierr = VecDuplicateVecs(X, lmvm->m, &lbfgs->P);CHKERRQ(ierr);
    }
    switch (lbfgs->scale_type) {
    case MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL:
      ierr = MatLMVMAllocate(lbfgs->D, X, F);CHKERRQ(ierr);
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
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_SymBrdn       *lbfgs = (Mat_SymBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (lbfgs->allocated) {
    ierr = VecDestroy(&lbfgs->work);CHKERRQ(ierr);
    ierr = PetscFree4(lbfgs->stp, lbfgs->yts, lbfgs->yty, lbfgs->sts);CHKERRQ(ierr);
    ierr = VecDestroyVecs(lmvm->m, &lbfgs->P);CHKERRQ(ierr);
    lbfgs->allocated = PETSC_FALSE;
  }
  ierr = MatDestroy(&lbfgs->D);CHKERRQ(ierr);
  ierr = PetscFree(lmvm->ctx);CHKERRQ(ierr);
  ierr = MatDestroy_LMVM(B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatSetUp_LMVMBFGS(Mat B)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_SymBrdn       *lbfgs = (Mat_SymBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  PetscInt          n, N;
  
  PetscFunctionBegin;
  ierr = MatSetUp_LMVM(B);CHKERRQ(ierr);
  lbfgs->max_seq_rejects = lmvm->m/2;
  if (!lbfgs->allocated) {
    ierr = VecDuplicate(lmvm->Xprev, &lbfgs->work);CHKERRQ(ierr);
    ierr = PetscMalloc4(lmvm->m, &lbfgs->stp, lmvm->m, &lbfgs->yts, lmvm->m, &lbfgs->yty, lmvm->m, &lbfgs->sts);CHKERRQ(ierr);
    if (lmvm->m > 0) {
      ierr = VecDuplicateVecs(lmvm->Xprev, lmvm->m, &lbfgs->P);CHKERRQ(ierr);
    }
    switch (lbfgs->scale_type) {
    case MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL:
      ierr = MatGetLocalSize(B, &n, &n);CHKERRQ(ierr);
      ierr = MatGetSize(B, &N, &N);CHKERRQ(ierr);
      ierr = MatSetSizes(lbfgs->D, n, n, N, N);CHKERRQ(ierr);
      ierr = MatSetUp(lbfgs->D);CHKERRQ(ierr);
      break;
    default:
      break;
    }
    lbfgs->allocated = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatSetFromOptions_LMVMBFGS(PetscOptionItems *PetscOptionsObject, Mat B)
{
  PetscErrorCode               ierr;

  PetscFunctionBegin;
  ierr = MatSetFromOptions_LMVM(PetscOptionsObject, B);CHKERRQ(ierr);
  ierr = PetscOptionsHead(PetscOptionsObject,"L-BFGS method for approximating SPD Jacobian actions (MATLMVMBFGS)");CHKERRQ(ierr);
  ierr = MatSetFromOptions_LMVMSymBrdn_Private(PetscOptionsObject, B);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode MatCreate_LMVMBFGS(Mat B)
{
  Mat_LMVM          *lmvm;
  Mat_SymBrdn       *lbfgs;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatCreate_LMVMSymBrdn(B);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B, MATLMVMBFGS);CHKERRQ(ierr);
  B->ops->setup = MatSetUp_LMVMBFGS;
  B->ops->destroy = MatDestroy_LMVMBFGS;
  B->ops->setfromoptions = MatSetFromOptions_LMVMBFGS;
  B->ops->solve = MatSolve_LMVMBFGS;

  lmvm = (Mat_LMVM*)B->data;
  lmvm->ops->allocate = MatAllocate_LMVMBFGS;
  lmvm->ops->reset = MatReset_LMVMBFGS;
  lmvm->ops->update = MatUpdate_LMVMBFGS;
  lmvm->ops->mult = MatMult_LMVMBFGS;
  lmvm->ops->copy = MatCopy_LMVMBFGS;

  lbfgs = (Mat_SymBrdn*)lmvm->ctx;
  lbfgs->needQ           = PETSC_FALSE;
  lbfgs->phi             = 0.0;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/*@
   MatCreateLMVMBFGS - Creates a limited-memory Broyden-Fletcher-Goldfarb-Shano (BFGS)
   matrix used for approximating Jacobians. L-BFGS is symmetric positive-definite by 
   construction, and is commonly used to approximate Hessians in optimization 
   problems.
   
   The provided local and global sizes must match the solution and function vectors 
   used with MatLMVMUpdate() and MatSolve(). The resulting L-BFGS matrix will have 
   storage vectors allocated with VecCreateSeq() in serial and VecCreateMPI() in 
   parallel. To use the L-BFGS matrix with other vector types, the matrix must be 
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
+   -mat_lmvm_num_vecs - maximum number of correction vectors (i.e.: updates) stored
.   -mat_lmvm_scale_type - (developer) type of scaling applied to J0 (none, scalar, diagonal)
.   -mat_lmvm_theta - (developer) convex ratio between BFGS and DFP components of the diagonal J0 scaling
.   -mat_lmvm_rho - (developer) update limiter for the J0 scaling
.   -mat_lmvm_alpha - (developer) coefficient factor for the quadratic subproblem in J0 scaling
.   -mat_lmvm_beta - (developer) exponential factor for the diagonal J0 scaling
-   -mat_lmvm_sigma_hist - (developer) number of past updates to use in J0 scaling

   Level: intermediate

.seealso: MatCreate(), MATLMVM, MATLMVMBFGS, MatCreateLMVMDFP(), MatCreateLMVMSR1(), 
          MatCreateLMVMBrdn(), MatCreateLMVMBadBrdn(), MatCreateLMVMSymBrdn()
@*/
PetscErrorCode MatCreateLMVMBFGS(MPI_Comm comm, PetscInt n, PetscInt N, Mat *B)
{
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  ierr = MatCreate(comm, B);CHKERRQ(ierr);
  ierr = MatSetSizes(*B, n, n, N, N);CHKERRQ(ierr);
  ierr = MatSetType(*B, MATLMVMBFGS);CHKERRQ(ierr);
  ierr = MatSetUp(*B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
