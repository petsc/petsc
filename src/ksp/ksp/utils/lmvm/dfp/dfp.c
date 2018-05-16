#include <../src/ksp/ksp/utils/lmvm/lmvm.h> /*I "petscksp.h" I*/

/*
  Limited-memory Davidon-Fletcher-Powell method for approximating the 
  inverse of a Jacobian.
  
  L-DFP is symmetric positive-definite by construction.
  
  The solution method (approximate inverse Jacobian application) is matrix-vector 
  product version of the recursive formula.
  
  Q <- F
  
  for i = k,k-1,k-2,...,0
    rho[i] = 1 / (S[i]^T Y[i])
    alpha[i] = rho[i] * (Y[i]^T Q)
    Q <- Q - (alpha[i] * S[i])
  end
  
  if J0^{-1} exists
    R <- J0^{01} * Q
  elif J0 exists or user_ksp
    R <- inv(J0) * Q via KSP
  elif user_scale
    if diag_scale exists
      R <- VecPointwiseMult(Q, diag_scale)
    else
      R <- scale * Q
    end
  else
    R <- Q
  end
  
  for i = 0,1,2,...,k
    beta = rho[i] * (S[i]^T R)
    R <- R + ((alpha[i] - beta) * Y[i])
  end
  
  dX <- R
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
  PetscReal         yts[lmvm->k+1], ytx[lmvm->k+1], ytf, stf, ytp, stp;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(F, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(dX, VEC_CLASSID, 3);
  VecCheckSameSize(F, 2, dX, 3);
  VecCheckMatCompatible(B, dX, 3, F, 2);
  
  ierr = MatLMVMApplyJ0Inv(B, F, dX);CHKERRQ(ierr);
  
  for (i = 0; i <= lmvm->k; ++i) {
    ierr = MatLMVMApplyJ0Inv(B, lmvm->Y[i], ldfp->P[i]);
    
    for (j = 0; j <= i-1; ++j) {
       ierr = VecDotBegin(lmvm->Y[j], ldfp->P[i], &ytp);CHKERRQ(ierr);
       ierr = VecDotBegin(lmvm->S[j], ldfp->P[i], &stp);CHKERRQ(ierr);
       
       ierr = VecDotEnd(lmvm->Y[j], ldfp->P[i], &ytp);CHKERRQ(ierr);
       ierr = VecDotEnd(lmvm->S[j], ldfp->P[i], &stp);CHKERRQ(ierr);
       
       ierr = VecAXPBYPCZ(ldfp->P[i], -ytp/ytx[j], stp/yts[j], 1.0, ldfp->P[j], lmvm->S[j]);CHKERRQ(ierr);
    }
    
    ierr = VecDotBegin(lmvm->Y[i], lmvm->S[i], &yts[i]);CHKERRQ(ierr);
    ierr = VecDotBegin(lmvm->Y[i], ldfp->P[i], &ytx[i]);CHKERRQ(ierr);
    ierr = VecDotBegin(lmvm->Y[i], dX, &ytf);CHKERRQ(ierr);
    ierr = VecDotBegin(lmvm->S[i], dX, &stf);CHKERRQ(ierr);
    
    ierr = VecDotEnd(lmvm->Y[i], lmvm->S[i], &yts[i]);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->Y[i], ldfp->P[i], &ytx[i]);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->Y[i], dX, &ytf);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->S[i], dX, &stf);CHKERRQ(ierr);
    
    ierr = VecAXPBYPCZ(dX, -ytf/ytx[i], stf/yts[i], 1.0, ldfp->P[i], lmvm->S[i]);CHKERRQ(ierr);
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