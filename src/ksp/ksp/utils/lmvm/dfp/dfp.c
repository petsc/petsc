#include <../src/ksp/ksp/utils/lmvm/lmvm.h> /*I "petscksp.h" I*/

/*
  Limited-memory Davidon-Fletcher-Powell method for approximating the 
  inverse of a Jacobian.
  
  L-DFP is symmetric positive-definite by construction.
  
  The solution method (approximate inverse Jacobian application) is 
  matrix-vector product version of the recursive formula given in 
  Equation (6.15) of Nocedal and Wright "Numerical Optimization" 2nd 
  edition, pg 139.
  
  dX <- J0^{-1} * F
  
  for i = 0,1,2,...,k
    P[i] <- J0^{-1} & Y[i]
    
    for j=0,1,2,...,(i-1)
      gamma = (S[j]^T P[i]) / (Y[j]^T S[j])
      zeta = (Y[j]^T P[i]) / (Y[j]^T P[j])
      P[i] <- P[i] + (gamma * S[j]) - (zeta * P[j])
    end
    
    gamma = (S[i]^T dX) / (Y[i]^T S[i])
    zeta = (Y[i]^T dX) / (Y[i]^T P[i])
    dX <- dX + (gamma * S[i]) - (zeta * P[i])
  end
 */

typedef struct {
  Vec *P;
  PetscBool allocatedP;
} Mat_LDFP;

PetscErrorCode MatSolve_LMVMDFP(Mat B, Vec F, Vec dX)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_LDFP          *ldfp = (Mat_LDFP*)lmvm->ctx;
  PetscErrorCode    ierr;
  PetscInt          i, j;
  PetscReal         yts[lmvm->k+1], ytp[lmvm->k+1], ytx, stx, yjtpi, sjtpi;
  
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
       ierr = VecDotBegin(lmvm->S[j], ldfp->P[i], &sjtpi);CHKERRQ(ierr);
       ierr = VecDotEnd(lmvm->Y[j], ldfp->P[i], &yjtpi);CHKERRQ(ierr);
       ierr = VecDotEnd(lmvm->S[j], ldfp->P[i], &sjtpi);CHKERRQ(ierr);
       ierr = VecAXPBYPCZ(ldfp->P[i], -yjtpi/ytp[j], sjtpi/yts[j], 1.0, ldfp->P[j], lmvm->S[j]);CHKERRQ(ierr);
    }
    /* Get all the dot products we need 
       NOTE: yTs and yTp are stored so that we can re-use them when computing 
             P[i] at the next outer iteration */
    ierr = VecDotBegin(lmvm->Y[i], lmvm->S[i], &yts[i]);CHKERRQ(ierr);
    ierr = VecDotBegin(lmvm->Y[i], ldfp->P[i], &ytp[i]);CHKERRQ(ierr);
    ierr = VecDotBegin(lmvm->Y[i], dX, &ytx);CHKERRQ(ierr);
    ierr = VecDotBegin(lmvm->S[i], dX, &stx);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->Y[i], lmvm->S[i], &yts[i]);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->Y[i], ldfp->P[i], &ytp[i]);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->Y[i], dX, &ytx);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->S[i], dX, &stx);CHKERRQ(ierr);
    /* Update dX_{i+1} = (B^{-1})_{i+1} * f */
    ierr = VecAXPBYPCZ(dX, -ytx/ytp[i], stx/yts[i], 1.0, ldfp->P[i], lmvm->S[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PETSC_INTERN PetscErrorCode MatReset_LMVMDFP(Mat B, PetscBool destructive)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_LDFP          *ldfp = (Mat_LDFP*)lmvm->ctx;
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  if (destructive && ldfp->allocatedP && lmvm->m > 0) {
    ierr = VecDestroyVecs(lmvm->m, &ldfp->P);CHKERRQ(ierr);
    ldfp->allocatedP = PETSC_FALSE;
  }
  ierr = MatReset_LMVM(B, destructive);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PETSC_INTERN PetscErrorCode MatAllocate_LMVMDFP(Mat B, Vec X, Vec F)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_LDFP          *ldfp = (Mat_LDFP*)lmvm->ctx;
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  ierr = MatAllocate_LMVM(B, X, F);CHKERRQ(ierr);
  if (!ldfp->allocatedP && lmvm->m > 0) {
    ierr = VecDuplicateVecs(X, lmvm->m, &ldfp->P);CHKERRQ(ierr);
    ldfp->allocatedP = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PETSC_INTERN PetscErrorCode MatDestroy_LMVMDFP(Mat B)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_LDFP          *ldfp = (Mat_LDFP*)lmvm->ctx;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (ldfp->allocatedP && lmvm->m > 0) {
    ierr = VecDestroyVecs(lmvm->m, &ldfp->P);CHKERRQ(ierr);
    ldfp->allocatedP = PETSC_FALSE;
  }
  ierr = PetscFree(lmvm->ctx);CHKERRQ(ierr);
  ierr = MatDestroy_LMVM(B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PETSC_INTERN PetscErrorCode MatSetUp_LMVMDFP(Mat B)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_LDFP          *ldfp = (Mat_LDFP*)lmvm->ctx;
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  ierr = MatSetUp_LMVM(B);CHKERRQ(ierr);
  if (!ldfp->allocatedP && lmvm->m > 0) {
    ierr = VecDuplicateVecs(lmvm->Xprev, lmvm->m, &ldfp->P);CHKERRQ(ierr);
    ldfp->allocatedP = PETSC_TRUE;
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

  ierr = PetscNewLog(B, &ldfp);CHKERRQ(ierr);
  lmvm->ctx = (void*)ldfp;
  ldfp->allocatedP = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/*@
   MatCreateLMVMDFP - Creates a limited-memory Davidon-Fletcher-Powell (DFP) matrix 
   used for approximating Jacobians. L-DFP is symmetric positive-definite by 
   construction, and is the dual of L-BFGS where Y and S vectors swap roles. This 
   implementation only supports the MatSolve() operation, which is an application 
   of the approximate inverse of the Jacobian. 
   
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