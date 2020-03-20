#include <../src/ksp/ksp/utils/lmvm/brdn/brdn.h> /*I "petscksp.h" I*/

/*------------------------------------------------------------*/

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
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_Brdn          *lbb = (Mat_Brdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  PetscInt          i, j;
  PetscScalar       yjtyi, ytf;
  
  PetscFunctionBegin;
  VecCheckSameSize(F, 2, dX, 3);
  VecCheckMatCompatible(B, dX, 3, F, 2);
  
  if (lbb->needQ) {
    /* Pre-compute (Q[i] = (B_i)^{-1} * Y[i]) */
    for (i = 0; i <= lmvm->k; ++i) {
      ierr = MatLMVMApplyJ0Inv(B, lmvm->Y[i], lbb->Q[i]);CHKERRQ(ierr);
      for (j = 0; j <= i-1; ++j) {
        ierr = VecDot(lmvm->Y[j], lmvm->Y[i], &yjtyi);CHKERRQ(ierr);
        ierr = VecAXPBYPCZ(lbb->Q[i], PetscRealPart(yjtyi)/lbb->yty[j], -PetscRealPart(yjtyi)/lbb->yty[j], 1.0, lmvm->S[j], lbb->Q[j]);CHKERRQ(ierr);
      }
    }
    lbb->needQ = PETSC_FALSE;
  }
  
  ierr = MatLMVMApplyJ0Inv(B, F, dX);CHKERRQ(ierr);
  for (i = 0; i <= lmvm->k; ++i) {
    ierr = VecDot(lmvm->Y[i], F, &ytf);CHKERRQ(ierr);
    ierr = VecAXPBYPCZ(dX, PetscRealPart(ytf)/lbb->yty[i], -PetscRealPart(ytf)/lbb->yty[i], 1.0, lmvm->S[i], lbb->Q[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

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
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_Brdn          *lbb = (Mat_Brdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  PetscInt          i, j;
  PetscScalar       yjtsi, ytx;
  
  PetscFunctionBegin;
  VecCheckSameSize(X, 2, Z, 3);
  VecCheckMatCompatible(B, X, 2, Z, 3);
  
  if (lbb->needP) {
    /* Pre-compute (P[i] = (B_i) * S[i]) */
    for (i = 0; i <= lmvm->k; ++i) {
      ierr = MatLMVMApplyJ0Fwd(B, lmvm->S[i], lbb->P[i]);CHKERRQ(ierr);
      for (j = 0; j <= i-1; ++j) {
        ierr = VecDot(lmvm->Y[j], lmvm->S[i], &yjtsi);CHKERRQ(ierr);
        ierr = VecAXPBYPCZ(lbb->P[i], PetscRealPart(yjtsi)/lbb->yts[j], -PetscRealPart(yjtsi)/lbb->yts[j], 1.0, lmvm->Y[j], lbb->P[j]);CHKERRQ(ierr);
      }
    }
    lbb->needP = PETSC_FALSE;
  }
  
  ierr = MatLMVMApplyJ0Fwd(B, X, Z);CHKERRQ(ierr);
  for (i = 0; i <= lmvm->k; ++i) {
    ierr = VecDot(lmvm->Y[i], X, &ytx);CHKERRQ(ierr);
    ierr = VecAXPBYPCZ(Z, PetscRealPart(ytx)/lbb->yts[i], -PetscRealPart(ytx)/lbb->yts[i], 1.0, lmvm->Y[i], lbb->P[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatUpdate_LMVMBadBrdn(Mat B, Vec X, Vec F)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_Brdn          *lbb = (Mat_Brdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  PetscInt          old_k, i;
  PetscScalar       yty, yts;

  PetscFunctionBegin;
  if (!lmvm->m) PetscFunctionReturn(0);
  if (lmvm->prev_set) {
    /* Compute the new (S = X - Xprev) and (Y = F - Fprev) vectors */
    ierr = VecAYPX(lmvm->Xprev, -1.0, X);CHKERRQ(ierr);
    ierr = VecAYPX(lmvm->Fprev, -1.0, F);CHKERRQ(ierr);
    /* Accept the update */
    lbb->needP = lbb->needQ = PETSC_TRUE;
    old_k = lmvm->k;
    ierr = MatUpdateKernel_LMVM(B, lmvm->Xprev, lmvm->Fprev);CHKERRQ(ierr);
    /* If we hit the memory limit, shift the yty and yts arrays */
    if (old_k == lmvm->k) {
      for (i = 0; i <= lmvm->k-1; ++i) {
        lbb->yty[i] = lbb->yty[i+1];
        lbb->yts[i] = lbb->yts[i+1];
      }
    }
    /* Accumulate the latest yTy and yTs dot products */
    ierr = VecDotBegin(lmvm->Y[lmvm->k], lmvm->Y[lmvm->k], &yty);CHKERRQ(ierr);
    ierr = VecDotBegin(lmvm->Y[lmvm->k], lmvm->S[lmvm->k], &yts);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->Y[lmvm->k], lmvm->Y[lmvm->k], &yty);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->Y[lmvm->k], lmvm->S[lmvm->k], &yts);CHKERRQ(ierr);
    lbb->yty[lmvm->k] = PetscRealPart(yty);
    lbb->yts[lmvm->k] = PetscRealPart(yts);
  }
  /* Save the solution and function to be used in the next update */
  ierr = VecCopy(X, lmvm->Xprev);CHKERRQ(ierr);
  ierr = VecCopy(F, lmvm->Fprev);CHKERRQ(ierr);
  lmvm->prev_set = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatCopy_LMVMBadBrdn(Mat B, Mat M, MatStructure str)
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
    mctx->yty[i] = bctx->yty[i];
    mctx->yts[i] = bctx->yts[i];
    ierr = VecCopy(bctx->P[i], mctx->P[i]);CHKERRQ(ierr);
    ierr = VecCopy(bctx->Q[i], mctx->Q[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatReset_LMVMBadBrdn(Mat B, PetscBool destructive)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_Brdn          *lbb = (Mat_Brdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  lbb->needP = lbb->needQ = PETSC_TRUE;
  if (destructive && lbb->allocated) {
    ierr = PetscFree2(lbb->yty, lbb->yts);CHKERRQ(ierr);
    ierr = VecDestroyVecs(lmvm->m, &lbb->P);CHKERRQ(ierr);
    ierr = VecDestroyVecs(lmvm->m, &lbb->Q);CHKERRQ(ierr);
    lbb->allocated = PETSC_FALSE;
  }
  ierr = MatReset_LMVM(B, destructive);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatAllocate_LMVMBadBrdn(Mat B, Vec X, Vec F)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_Brdn          *lbb = (Mat_Brdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  ierr = MatAllocate_LMVM(B, X, F);CHKERRQ(ierr);
  if (!lbb->allocated) {
    ierr = PetscMalloc2(lmvm->m, &lbb->yty, lmvm->m, &lbb->yts);CHKERRQ(ierr);
    if (lmvm->m > 0) {
      ierr = VecDuplicateVecs(X, lmvm->m, &lbb->P);CHKERRQ(ierr);
      ierr = VecDuplicateVecs(X, lmvm->m, &lbb->Q);CHKERRQ(ierr);
    }
    lbb->allocated = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatDestroy_LMVMBadBrdn(Mat B)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_Brdn          *lbb = (Mat_Brdn*)lmvm->ctx;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (lbb->allocated) {
    ierr = PetscFree2(lbb->yty, lbb->yts);CHKERRQ(ierr);
    ierr = VecDestroyVecs(lmvm->m, &lbb->P);CHKERRQ(ierr);
    ierr = VecDestroyVecs(lmvm->m, &lbb->Q);CHKERRQ(ierr);
    lbb->allocated = PETSC_FALSE;
  }
  ierr = PetscFree(lmvm->ctx);CHKERRQ(ierr);
  ierr = MatDestroy_LMVM(B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatSetUp_LMVMBadBrdn(Mat B)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_Brdn          *lbb = (Mat_Brdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  ierr = MatSetUp_LMVM(B);CHKERRQ(ierr);
  if (!lbb->allocated) {
    ierr = PetscMalloc2(lmvm->m, &lbb->yty, lmvm->m, &lbb->yts);CHKERRQ(ierr);
    if (lmvm->m > 0) {
      ierr = VecDuplicateVecs(lmvm->Xprev, lmvm->m, &lbb->P);CHKERRQ(ierr);
      ierr = VecDuplicateVecs(lmvm->Xprev, lmvm->m, &lbb->Q);CHKERRQ(ierr);
    }
    lbb->allocated = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode MatCreate_LMVMBadBrdn(Mat B)
{
  Mat_LMVM          *lmvm;
  Mat_Brdn          *lbb;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatCreate_LMVM(B);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B, MATLMVMBADBRDN);CHKERRQ(ierr);
  B->ops->setup = MatSetUp_LMVMBadBrdn;
  B->ops->destroy = MatDestroy_LMVMBadBrdn;
  B->ops->solve = MatSolve_LMVMBadBrdn;

  lmvm = (Mat_LMVM*)B->data;
  lmvm->square = PETSC_TRUE;
  lmvm->ops->allocate = MatAllocate_LMVMBadBrdn;
  lmvm->ops->reset = MatReset_LMVMBadBrdn;
  lmvm->ops->mult = MatMult_LMVMBadBrdn;
  lmvm->ops->update = MatUpdate_LMVMBadBrdn;
  lmvm->ops->copy = MatCopy_LMVMBadBrdn;

  ierr = PetscNewLog(B, &lbb);CHKERRQ(ierr);
  lmvm->ctx = (void*)lbb;
  lbb->allocated = PETSC_FALSE;
  lbb->needP = lbb->needQ = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/*@
   MatCreateLMVMBadBrdn - Creates a limited-memory modified (aka "bad") Broyden-type 
   approximation matrix used for a Jacobian. L-BadBrdn is not guaranteed to be 
   symmetric or positive-definite.
   
   The provided local and global sizes must match the solution and function vectors 
   used with MatLMVMUpdate() and MatSolve(). The resulting L-BadBrdn matrix will have 
   storage vectors allocated with VecCreateSeq() in serial and VecCreateMPI() in 
   parallel. To use the L-BadBrdn matrix with other vector types, the matrix must be 
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

.seealso: MatCreate(), MATLMVM, MATLMVMBADBRDN, MatCreateLMVMDFP(), MatCreateLMVMSR1(), 
          MatCreateLMVMBFGS(), MatCreateLMVMBrdn(), MatCreateLMVMSymBrdn()
@*/
PetscErrorCode MatCreateLMVMBadBrdn(MPI_Comm comm, PetscInt n, PetscInt N, Mat *B)
{
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  ierr = MatCreate(comm, B);CHKERRQ(ierr);
  ierr = MatSetSizes(*B, n, n, N, N);CHKERRQ(ierr);
  ierr = MatSetType(*B, MATLMVMBADBRDN);CHKERRQ(ierr);
  ierr = MatSetUp(*B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
