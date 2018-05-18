#include <../src/ksp/ksp/utils/lmvm/lmvm.h> /*I "petscksp.h" I*/

/*
  Limited-memory Broyden-Fletcher-Goldfarb-Shano method for approximating the 
  inverse of a Jacobian.
  
  BFGS is symmetric positive-definite by construction.
*/

typedef struct {
  Vec *P;
  Vec work;
  PetscBool allocatedP;
} Mat_LBFGS;

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
  Mat_LBFGS         *lbfgs = (Mat_LBFGS*)lmvm->ctx;
  PetscErrorCode    ierr;
  PetscInt          i;
  PetscReal         alpha[lmvm->k+1], rho[lmvm->k+1];
  PetscReal         beta, yts, stf, ytx, yts_k, yty;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(F, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(dX, VEC_CLASSID, 3);
  VecCheckSameSize(F, 2, dX, 3);
  VecCheckMatCompatible(B, dX, 3, F, 2);
  
  /* Copy the function into the work vector for the first loop */
  ierr = VecCopy(F, lbfgs->work);CHKERRQ(ierr);
  
  /* Start the first loop */
  for (i = lmvm->k; i >= 0; --i) {
    ierr = VecDotBegin(lmvm->Y[i], lmvm->S[i], &yts);CHKERRQ(ierr);
    ierr = VecDotBegin(lmvm->S[i], lbfgs->work, &stf);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->Y[i], lmvm->S[i], &yts);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->S[i], lbfgs->work, &stf);CHKERRQ(ierr);
    if (i == lmvm->k) yts_k = yts; /* save this for later in case we need it for J0 */
    rho[i] = 1.0/yts;
    alpha[i] = rho[i] * stf;
    ierr = VecAXPY(lbfgs->work, -alpha[i], lmvm->Y[i]);CHKERRQ(ierr);
  }
  
  /* Invert the initial Jacobian onto Q (or apply scaling) */
  ierr = MatLMVMApplyJ0Inv(B, lbfgs->work, dX);CHKERRQ(ierr);
  
  if ((lmvm->k >= 0) && (!lmvm->user_scale) && (!lmvm->user_pc) && (!lmvm->user_ksp) && (!lmvm->J0)) {
    /* Since there is no J0 definition, finish the dot products then apply the gamma scaling */
    ierr = VecDot(lmvm->Y[lmvm->k], lmvm->Y[lmvm->k], &yty);CHKERRQ(ierr);
    ierr = VecScale(dX, yts_k/yty);CHKERRQ(ierr);
  }
  
  /* Start the second loop */
  for (i = 0; i <= lmvm->k; ++i) {
    ierr = VecDot(lmvm->Y[i], dX, &ytx);CHKERRQ(ierr);
    beta = rho[i] * ytx;
    ierr = VecAXPY(dX, alpha[i]-beta, lmvm->S[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/*
  The forward product for the approximate Jacobian is the matrix-free 
  implementation of Equation (6.19) in Nocedal and Wright "Numerical 
  Optimization" 2nd Edition, pg 140.
  
  Note that this forward product has the same structure as the 
  inverse Jacobian application in the DFP formulation, except with S 
  and Y exchanging roles.

  Z <- J0 * X

  for i = 0,1,2,...,k
    P[i] <- J0 & S[i]

    for j=0,1,2,...,(i-1)
      gamma = (Y[j]^T P[i]) / (Y[j]^T S[j])
      zeta = (S[j]^T P[i]) / (S[j]^T P[j])
      P[i] <- P[i] + (gamma * Y[j]) - (zeta * P[j])
    end

    gamma = (Y[i]^T dX) / (Y[i]^T S[i])
    zeta = (S[i]^T dX) / (S[i]^T P[i])
    Z <- Z + (gamma * Y[i]) - (zeta * P[i])
  end
*/
PetscErrorCode MatMult_LMVMBFGS(Mat B, Vec X, Vec Z)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_LBFGS         *lbfgs = (Mat_LBFGS*)lmvm->ctx;
  PetscErrorCode    ierr;
  PetscInt          i, j;
  PetscReal         yts[lmvm->k+1], stp[lmvm->k+1], ytz, stz, yjtpi, sjtpi;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(X, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(Z, VEC_CLASSID, 3);
  VecCheckSameSize(X, 2, Z, 3);
  VecCheckMatCompatible(B, X, 3, Z, 2);
  
  /* Start the outer loop (i) for the recursive formula */
  ierr = MatLMVMApplyJ0Fwd(B, X, Z);CHKERRQ(ierr);
  for (i = 0; i <= lmvm->k; ++i) {
    /* First compute P[i] = B_i * s_i using an inner loop (j) 
       NOTE: This is essentially the same recipe used for dX, but we don't 
             recurse because reuse P[i] from previous outer iterations. */
    ierr = MatLMVMApplyJ0Fwd(B, lmvm->S[i], lbfgs->P[i]);
    for (j = 0; j <= i-1; ++j) {
       ierr = VecDotBegin(lmvm->S[j], lbfgs->P[i], &sjtpi);CHKERRQ(ierr);
       ierr = VecDotBegin(lmvm->Y[j], lbfgs->P[i], &yjtpi);CHKERRQ(ierr);
       ierr = VecDotEnd(lmvm->S[j], lbfgs->P[i], &sjtpi);CHKERRQ(ierr);
       ierr = VecDotEnd(lmvm->Y[j], lbfgs->P[i], &yjtpi);CHKERRQ(ierr);
       ierr = VecAXPBYPCZ(lbfgs->P[i], -sjtpi/stp[j], yjtpi/yts[j], 1.0, lbfgs->P[j], lmvm->Y[j]);CHKERRQ(ierr);
    }
    /* Get all the dot products we need 
       NOTE: yTs and sTp are stored so that we can re-use them when computing 
             P[i] at the next outer iteration */
    ierr = VecDotBegin(lmvm->Y[i], lmvm->S[i], &yts[i]);CHKERRQ(ierr);
    ierr = VecDotBegin(lmvm->S[i], lbfgs->P[i], &stp[i]);CHKERRQ(ierr);
    ierr = VecDotBegin(lmvm->S[i], Z, &stz);CHKERRQ(ierr);
    ierr = VecDotBegin(lmvm->Y[i], Z, &ytz);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->Y[i], lmvm->S[i], &yts[i]);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->S[i], lbfgs->P[i], &stp[i]);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->S[i], Z, &stz);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->Y[i], Z, &ytz);CHKERRQ(ierr);
    /* Update Z_{i+1} = B_{i+1} * X */
    ierr = VecAXPBYPCZ(Z, -stz/stp[i], ytz/yts[i], 1.0, lbfgs->P[i], lmvm->Y[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PETSC_INTERN PetscErrorCode MatReset_LMVMBFGS(Mat B, PetscBool destructive)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_LBFGS         *lbfgs = (Mat_LBFGS*)lmvm->ctx;
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  if (destructive && lbfgs->allocatedP && lmvm->m > 0) {
    ierr = VecDestroy(&lbfgs->work);CHKERRQ(ierr);
    ierr = VecDestroyVecs(lmvm->m, &lbfgs->P);CHKERRQ(ierr);
    lbfgs->allocatedP = PETSC_FALSE;
  }
  ierr = MatReset_LMVM(B, destructive);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PETSC_INTERN PetscErrorCode MatAllocate_LMVMBFGS(Mat B, Vec X, Vec F)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_LBFGS         *lbfgs = (Mat_LBFGS*)lmvm->ctx;
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  ierr = MatAllocate_LMVM(B, X, F);CHKERRQ(ierr);
  if (!lbfgs->allocatedP && lmvm->m > 0) {
    ierr = VecDuplicate(X, &lbfgs->work);CHKERRQ(ierr);
    ierr = VecDuplicateVecs(X, lmvm->m, &lbfgs->P);CHKERRQ(ierr);
    lbfgs->allocatedP = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PETSC_INTERN PetscErrorCode MatDestroy_LMVMBFGS(Mat B)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_LBFGS         *lbfgs = (Mat_LBFGS*)lmvm->ctx;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (lbfgs->allocatedP && lmvm->m > 0) {
    ierr = VecDestroy(&lbfgs->work);CHKERRQ(ierr);
    ierr = VecDestroyVecs(lmvm->m, &lbfgs->P);CHKERRQ(ierr);
    lbfgs->allocatedP = PETSC_FALSE;
  }
  ierr = PetscFree(lmvm->ctx);CHKERRQ(ierr);
  ierr = MatDestroy_LMVM(B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PETSC_INTERN PetscErrorCode MatSetUp_LMVMBFGS(Mat B)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_LBFGS         *lbfgs = (Mat_LBFGS*)lmvm->ctx;
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  ierr = MatSetUp_LMVM(B);CHKERRQ(ierr);
  if (!lbfgs->allocatedP && lmvm->m > 0) {
    ierr = VecDuplicate(lmvm->Xprev, &lbfgs->work);CHKERRQ(ierr);
    ierr = VecDuplicateVecs(lmvm->Xprev, lmvm->m, &lbfgs->P);CHKERRQ(ierr);
    lbfgs->allocatedP = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode MatCreate_LMVMBFGS(Mat B)
{
  Mat_LBFGS         *lbfgs;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatCreate_LMVM(B);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B, MATLMVMBFGS);CHKERRQ(ierr);
  ierr = MatSetOption(B, MAT_SPD, PETSC_TRUE);CHKERRQ(ierr);
  B->ops->mult = MatMult_LMVMBFGS;
  B->ops->solve = MatSolve_LMVMBFGS;
  B->ops->setup = MatSetUp_LMVMBFGS;
  B->ops->destroy = MatDestroy_LMVMBFGS;
  
  Mat_LMVM *lmvm = (Mat_LMVM*)B->data;
  lmvm->square = PETSC_TRUE;
  lmvm->ops->allocate = MatAllocate_LMVMBFGS;
  lmvm->ops->reset = MatReset_LMVMBFGS;

  ierr = PetscNewLog(B, &lbfgs);CHKERRQ(ierr);
  lmvm->ctx = (void*)lbfgs;
  lbfgs->allocatedP = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/*@
   MatCreateLMVMBFGS - Creates a limited-memory Broyden-Fletcher-Goldfarb-Shano (BFGS)
   matrix used for approximating Jacobians. L-BFGS is symmetric positive-definite by 
   construction, and is commonly used to approximate Hessians in optimization 
   problems. This implementation only supports the MatSolve() operation, which is 
   an application of the approximate inverse of the Jacobian. 
   
   The provided local and global sizes must match the solution and function vectors 
   used with MatLMVMUpdate() and MatSolve(). The resulting L-BFGS matrix will have 
   storage vectors allocated with VecCreateSeq() in serial and VecCreateMPI() in 
   parallel. To use the L-BFGS matrix with other vector types, the matrix must be 
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