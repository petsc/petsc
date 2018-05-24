#include <../src/ksp/ksp/utils/lmvm/lmvm.h> /*I "petscksp.h" I*/

/*
  Limited-memory Davidon-Fletcher-Powell method for approximating both 
  the forward product and inverse application of a Jacobian.
 */

typedef struct {
  Vec *P;
  Vec work;
  PetscBool allocated;
} Mat_LDFP;

/*------------------------------------------------------------*/

/*
  The solution method (approximate inverse Jacobian application) is 
  matrix-vector product version of the recursive formula given in 
  Equation (6.15) of Nocedal and Wright "Numerical Optimization" 2nd 
  edition, pg 139.

  dX <- J0^{-1} * F

  for i = 0,1,2,...,k
    P[i] <- J0^{-1} & Y[i]

    for j=0,1,2,...,(i-1)
      gamma = (S[j]^T Y[i]) / (Y[j]^T S[j])
      zeta = (Y[j]^T P[i]) / (Y[j]^T P[j])
      P[i] <- P[i] + (gamma * S[j]) - (zeta * P[j])
    end

    gamma = (S[i]^T F) / (Y[i]^T S[i])
    zeta = (Y[i]^T dX) / (Y[i]^T P[i])
    dX <- dX + (gamma * S[i]) - (zeta * P[i])
  end
*/
static PetscErrorCode MatSolve_LMVMDFP(Mat B, Vec F, Vec dX)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_LDFP          *ldfp = (Mat_LDFP*)lmvm->ctx;
  PetscErrorCode    ierr;
  PetscInt          i, j;
  PetscReal         yts[lmvm->k+1], ytp[lmvm->k+1], ytx, stf, yjtpi, sjtyi;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(F, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(dX, VEC_CLASSID, 3);
  VecCheckSameSize(F, 2, dX, 3);
  VecCheckMatCompatible(B, dX, 3, F, 2);
  
  /* Start the outer loop (i) for the recursive formula */
  ierr = MatLMVMApplyJ0Inv(B, F, dX);CHKERRQ(ierr);
  for (i = 0; i <= lmvm->k; ++i) {
    /* First compute P[i] = (B^{-1})_i * y_i using an inner loop (j) 
       NOTE: This is essentially the same recipe used for dX, but we don't 
             recurse because reuse P[i] from previous outer iterations. */
    ierr = MatLMVMApplyJ0Inv(B, lmvm->Y[i], ldfp->P[i]);
    for (j = 0; j <= i-1; ++j) {
       ierr = VecDotBegin(lmvm->Y[j], ldfp->P[i], &yjtpi);CHKERRQ(ierr);
       ierr = VecDotBegin(lmvm->S[j], lmvm->Y[i], &sjtyi);CHKERRQ(ierr);
       ierr = VecDotEnd(lmvm->Y[j], ldfp->P[i], &yjtpi);CHKERRQ(ierr);
       ierr = VecDotEnd(lmvm->S[j], lmvm->Y[i], &sjtyi);CHKERRQ(ierr);
       ierr = VecAXPBYPCZ(ldfp->P[i], -yjtpi/ytp[j], sjtyi/yts[j], 1.0, ldfp->P[j], lmvm->S[j]);CHKERRQ(ierr);
    }
    /* Get all the dot products we need 
       NOTE: yTs and yTp are stored so that we can re-use them when computing 
             P[i] at the next outer iteration */
    ierr = VecDotBegin(lmvm->Y[i], lmvm->S[i], &yts[i]);CHKERRQ(ierr);
    ierr = VecDotBegin(lmvm->Y[i], ldfp->P[i], &ytp[i]);CHKERRQ(ierr);
    ierr = VecDotBegin(lmvm->Y[i], dX, &ytx);CHKERRQ(ierr);
    ierr = VecDotBegin(lmvm->S[i], F, &stf);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->Y[i], lmvm->S[i], &yts[i]);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->Y[i], ldfp->P[i], &ytp[i]);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->Y[i], dX, &ytx);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->S[i], F, &stf);CHKERRQ(ierr);
    /* Update dX_{i+1} = (B^{-1})_{i+1} * f */
    ierr = VecAXPBYPCZ(dX, -ytx/ytp[i], stf/yts[i], 1.0, ldfp->P[i], lmvm->S[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/*
  The forward product for the approximate Jacobian is the matrix-free 
  implementation of the recursive formula given in Equation 6.13 of 
  Nocedal and Wright "Numerical Optimization" 2nd edition, pg 139.
  
  Note that this forward product has a two-loop form similar to the 
  BFGS two-loop formulation for the inverse Jacobian application. 
  However, the S and Y vectors have interchanged roles.

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
static PetscErrorCode MatMult_LMVMDFP(Mat B, Vec X, Vec Z)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_LDFP          *ldfp = (Mat_LDFP*)lmvm->ctx;
  PetscErrorCode    ierr;
  PetscInt          i;
  PetscReal         alpha[lmvm->k+1], rho[lmvm->k+1];
  PetscReal         beta, yts, ytx, stz;
  
  PetscFunctionBegin;
  /* Copy the function into the work vector for the first loop */
  ierr = VecCopy(X, ldfp->work);CHKERRQ(ierr);
  
  /* Start the first loop */
  for (i = lmvm->k; i >= 0; --i) {
    ierr = VecDotBegin(lmvm->Y[i], lmvm->S[i], &yts);CHKERRQ(ierr);
    ierr = VecDotBegin(lmvm->Y[i], ldfp->work, &ytx);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->Y[i], lmvm->S[i], &yts);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->Y[i], ldfp->work, &ytx);CHKERRQ(ierr);
    rho[i] = 1.0/yts;
    alpha[i] = rho[i] * ytx;
    ierr = VecAXPY(ldfp->work, -alpha[i], lmvm->S[i]);CHKERRQ(ierr);
  }
  
  /* Apply the forward product with initial Jacobian */
  ierr = MatLMVMApplyJ0Fwd(B, ldfp->work, Z);CHKERRQ(ierr);
  
  /* Start the second loop */
  for (i = 0; i <= lmvm->k; ++i) {
    ierr = VecDot(lmvm->S[i], Z, &stz);CHKERRQ(ierr);
    beta = rho[i] * stz;
    ierr = VecAXPY(Z, alpha[i]-beta, lmvm->Y[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatUpdate_LMVMDFP(Mat B, Vec X, Vec F)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  PetscErrorCode    ierr;
  PetscReal         curvature;

  PetscFunctionBegin;
  if (lmvm->m == 0) PetscFunctionReturn(0);
  if (lmvm->prev_set) {
    /* Compute the new (S = X - Xprev) and (Y = F - Fprev) vectors */
    ierr = VecAXPBY(lmvm->Xprev, 1.0, -1.0, X);CHKERRQ(ierr);
    ierr = VecAXPBY(lmvm->Fprev, 1.0, -1.0, F);CHKERRQ(ierr);
    /* Test if the updates can be accepted */
    ierr = VecDot(lmvm->Xprev, lmvm->Fprev, &curvature);CHKERRQ(ierr);
    if (curvature > -lmvm->eps) {
      /* Update is good, accept it */
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

static PetscErrorCode MatReset_LMVMDFP(Mat B, PetscBool destructive)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_LDFP          *ldfp = (Mat_LDFP*)lmvm->ctx;
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  if (destructive && ldfp->allocated && lmvm->m > 0) {
    ierr = VecDestroy(&ldfp->work);CHKERRQ(ierr);
    ierr = VecDestroyVecs(lmvm->m, &ldfp->P);CHKERRQ(ierr);
    ldfp->allocated = PETSC_FALSE;
  }
  ierr = MatReset_LMVM(B, destructive);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatAllocate_LMVMDFP(Mat B, Vec X, Vec F)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_LDFP          *ldfp = (Mat_LDFP*)lmvm->ctx;
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  ierr = MatAllocate_LMVM(B, X, F);CHKERRQ(ierr);
  if (!ldfp->allocated && lmvm->m > 0) {
    ierr = VecDuplicate(X, &ldfp->work);CHKERRQ(ierr);
    ierr = VecDuplicateVecs(X, lmvm->m, &ldfp->P);CHKERRQ(ierr);
    ldfp->allocated = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatDestroy_LMVMDFP(Mat B)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_LDFP          *ldfp = (Mat_LDFP*)lmvm->ctx;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (ldfp->allocated && lmvm->m > 0) {
    ierr = VecDestroy(&ldfp->work);CHKERRQ(ierr);
    ierr = VecDestroyVecs(lmvm->m, &ldfp->P);CHKERRQ(ierr);
    ldfp->allocated = PETSC_FALSE;
  }
  ierr = PetscFree(lmvm->ctx);CHKERRQ(ierr);
  ierr = MatDestroy_LMVM(B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatSetUp_LMVMDFP(Mat B)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_LDFP          *ldfp = (Mat_LDFP*)lmvm->ctx;
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  ierr = MatSetUp_LMVM(B);CHKERRQ(ierr);
  if (!ldfp->allocated && lmvm->m > 0) {
    ierr = VecDuplicate(lmvm->Xprev, &ldfp->work);CHKERRQ(ierr);
    ierr = VecDuplicateVecs(lmvm->Xprev, lmvm->m, &ldfp->P);CHKERRQ(ierr);
    ldfp->allocated = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode MatCreate_LMVMDFP(Mat B)
{
  Mat_LDFP          *ldfp;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatCreate_LMVM(B);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B, MATLMVMDFP);CHKERRQ(ierr);
  ierr = MatSetOption(B, MAT_SPD, PETSC_TRUE);CHKERRQ(ierr);
  B->ops->solve = MatSolve_LMVMDFP;
  B->ops->setup = MatSetUp_LMVMDFP;
  B->ops->destroy = MatDestroy_LMVMDFP;

  Mat_LMVM *lmvm = (Mat_LMVM*)B->data;
  lmvm->square = PETSC_TRUE;
  lmvm->ops->allocate = MatAllocate_LMVMDFP;
  lmvm->ops->reset = MatReset_LMVMDFP;
  lmvm->ops->update = MatUpdate_LMVMDFP;
  lmvm->ops->mult = MatMult_LMVMDFP;

  ierr = PetscNewLog(B, &ldfp);CHKERRQ(ierr);
  lmvm->ctx = (void*)ldfp;
  ldfp->allocated = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/*@
   MatCreateLMVMDFP - Creates a limited-memory Davidon-Fletcher-Powell (DFP) matrix 
   used for approximating Jacobians. L-DFP is symmetric positive-definite by 
   construction, and is the dual of L-BFGS where Y and S vectors swap roles.
   
   The provided local and global sizes must match the solution and function vectors 
   used with MatLMVMUpdate() and MatSolve(). The resulting L-DFP matrix will have 
   storage vectors allocated with VecCreateSeq() in serial and VecCreateMPI() in 
   parallel. To use the L-DFP matrix with other vector types, the matrix must be 
   created using MatCreate() and MatSetType(), followed by MatLMVMAllocate(). 
   This ensures that the internal storage and work vectors are duplicated from the 
   correct type of vector.

   Collective on MPI_Comm

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

.seealso: MatCreate(), MATLMVM, MATLMVMDFP, MatCreateLMVMBFGS(), MatCreateLMVMSR1(), 
           MatCreateLMVMBrdn(), MatCreateLMVMBadBrdn(), MatCreateLMVMSymBrdn()
@*/
PetscErrorCode MatCreateLMVMDFP(MPI_Comm comm, PetscInt n, PetscInt N, Mat *B)
{
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  ierr = MatCreate(comm, B);CHKERRQ(ierr);
  ierr = MatSetSizes(*B, n, n, N, N);CHKERRQ(ierr);
  ierr = MatSetType(*B, MATLMVMDFP);CHKERRQ(ierr);
  ierr = MatSetUp(*B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}