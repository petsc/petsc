#include <../src/ksp/ksp/utils/lmvm/lmvm.h> /*I "petscksp.h" I*/

/*
  Limited-memory "good" Broyden's method for approximating the inverse of 
  a Jacobian.
*/

typedef struct {
  Vec *P;
  PetscBool allocated;
} Mat_BadBrdn;

/*------------------------------------------------------------*/

/*
  The solution method is the matrix-free implementation of the inverse Hessian in
  Equation 6 on page 312 of Griewank "Broyden Updating, The Good and The Bad!" 
  (http://www.emis.ams.org/journals/DMJDMV/vol-ismp/45_griewank-andreas-broyden.pdf).
  
  dX <- J0^{-1} * F
  
  for i=0,1,2,...,k
    P[i] <- J0^{-1} * Y[i]
    
    for j=0,1,2,...,(i-1)
      tau = (Y[j]^T Y[i]) / (Y[j]^T Y[j])
      P[i] <- P[i] + (tau * (S[j] - P[j]))
    end
    
    tau = (Y[i]^T F) / (Y[i]^T Y[i])
    dX <- dX + (tau * (S[i] - P[i]))
  end
 */

PetscErrorCode MatSolve_LMVMBadBrdn(Mat B, Vec F, Vec dX)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_BadBrdn       *lbb = (Mat_BadBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  PetscInt          i, j;
  PetscReal         yty[lmvm->k+1], ytf;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  PetscValidHeaderSpecific(F, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(dX, VEC_CLASSID, 3);
  VecCheckSameSize(F, 2, dX, 3);
  VecCheckMatCompatible(B, dX, 3, F, 2);
  
  ierr = MatLMVMApplyJ0Inv(B, F, dX);CHKERRQ(ierr);
  for (i = 0; i <= lmvm->k-1; ++i) {
    ierr = MatLMVMApplyJ0Inv(B, lmvm->Y[i], lbb->P[i]);CHKERRQ(ierr);
    for (j = 0; j <= i-1; ++j) {
      ierr = VecDot(lmvm->Y[j], lmvm->Y[i], &ytf);CHKERRQ(ierr);
      ierr = VecAXPBYPCZ(lbb->P[i], ytf/yty[j], -ytf/yty[j], 1.0, lmvm->S[j], lbb->P[j]);CHKERRQ(ierr);
    }
    ierr = VecDotBegin(lmvm->Y[i], F, &ytf);CHKERRQ(ierr);
    ierr = VecDotBegin(lmvm->Y[i], lmvm->Y[i], &yty[i]);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->Y[i], F, &ytf);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->Y[i], lmvm->Y[i], &yty[i]);CHKERRQ(ierr);
    ierr = VecAXPBYPCZ(dX, ytf/yty[i], -ytf/yty[i], 1.0, lmvm->S[i], lbb->P[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/*
  The forward product is the matrix-free implementation of the direct update in 
  Equation 6 on page 302 of Griewank "Broyden Updating, The Good and The Bad!"
  (http://www.emis.ams.org/journals/DMJDMV/vol-ismp/45_griewank-andreas-broyden.pdf).
  
  Z <- J0 * X
  
  for i=0,1,2,...,k
    P[i] <- J0 * S[i]
    
    for j=0,1,2,...,(i-1)
      tau = (Y[j]^T S[i]) / (Y[j]^T S[j])
      P[i] <- P[i] + (tau * (Y[j] - P[j]))
    end
    
    tau = (Y[i]^T X) / (Y[i]^T S[i])
    dX <- dX + (tau * (Y[i] - P[i]))
  end
 */

PetscErrorCode MatMult_LMVMBadBrdn(Mat B, Vec X, Vec Z)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_BadBrdn       *lbb = (Mat_BadBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  PetscInt          i, j;
  PetscReal         yts[lmvm->k+1], ytx;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  PetscValidHeaderSpecific(X, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(Z, VEC_CLASSID, 3);
  VecCheckSameSize(X, 2, Z, 3);
  VecCheckMatCompatible(B, X, 2, Z, 3);
  
  ierr = MatLMVMApplyJ0Fwd(B, X, Z);CHKERRQ(ierr);
  for (i = 0; i <= lmvm->k-1; ++i) {
    ierr = MatLMVMApplyJ0Fwd(B, lmvm->S[i], lbb->P[i]);CHKERRQ(ierr);
    for (j = 0; j <= i-1; ++j) {
      ierr = VecDot(lmvm->Y[j], lmvm->S[i], &ytx);CHKERRQ(ierr);
      ierr = VecAXPBYPCZ(lbb->P[i], ytx/yts[j], -ytx/yts[j], 1.0, lmvm->Y[j], lbb->P[j]);CHKERRQ(ierr);
    }
    ierr = VecDotBegin(lmvm->Y[i], X, &ytx);CHKERRQ(ierr);
    ierr = VecDotBegin(lmvm->Y[i], lmvm->S[i], &yts[i]);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->Y[i], X, &ytx);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->Y[i], lmvm->S[i], &yts[i]);CHKERRQ(ierr);
    ierr = VecAXPBYPCZ(Z, ytx/yts[i], -ytx/yts[i], 1.0, lmvm->Y[i], lbb->P[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PETSC_INTERN PetscErrorCode MatReset_LMVMBadBrdn(Mat B, PetscBool destructive)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_BadBrdn       *lbb = (Mat_BadBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  if (destructive && lbb->allocated && lmvm->m > 0) {
    ierr = VecDestroyVecs(lmvm->m, &lbb->P);CHKERRQ(ierr);
    lbb->allocated = PETSC_FALSE;
  }
  ierr = MatReset_LMVM(B, destructive);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PETSC_INTERN PetscErrorCode MatAllocate_LMVMBadBrdn(Mat B, Vec X, Vec F)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_BadBrdn          *lbb = (Mat_BadBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  ierr = MatAllocate_LMVM(B, X, F);CHKERRQ(ierr);
  if (!lbb->allocated && lmvm->m > 0) {
    ierr = VecDuplicateVecs(X, lmvm->m, &lbb->P);CHKERRQ(ierr);
    lbb->allocated = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PETSC_INTERN PetscErrorCode MatDestroy_LMVMBadBrdn(Mat B)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_BadBrdn       *lbb = (Mat_BadBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (lbb->allocated && lmvm->m > 0) {
    ierr = VecDestroyVecs(lmvm->m, &lbb->P);CHKERRQ(ierr);
    lbb->allocated = PETSC_FALSE;
  }
  ierr = PetscFree(lmvm->ctx);CHKERRQ(ierr);
  ierr = MatDestroy_LMVM(B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PETSC_INTERN PetscErrorCode MatSetUp_LMVMBadBrdn(Mat B)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_BadBrdn       *lbb = (Mat_BadBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  ierr = MatSetUp_LMVM(B);CHKERRQ(ierr);
  if (!lbb->allocated && lmvm->m > 0) {
    ierr = VecDuplicateVecs(lmvm->Xprev, lmvm->m, &lbb->P);CHKERRQ(ierr);
    lbb->allocated = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode MatCreate_LMVMBadBrdn(Mat B)
{
  Mat_BadBrdn       *lbb;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatCreate_LMVM(B);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B, MATLMVMBRDN);CHKERRQ(ierr);
  B->ops->mult = MatMult_LMVMBadBrdn;
  B->ops->solve = MatSolve_LMVMBadBrdn;
  B->ops->setup = MatSetUp_LMVMBadBrdn;
  B->ops->destroy = MatDestroy_LMVMBadBrdn;

  Mat_LMVM *lmvm = (Mat_LMVM*)B->data;
  lmvm->square = PETSC_TRUE;
  lmvm->ops->allocate = MatAllocate_LMVMBadBrdn;
  lmvm->ops->reset = MatReset_LMVMBadBrdn;

  ierr = PetscNewLog(B, &lbb);CHKERRQ(ierr);
  lmvm->ctx = (void*)lbb;
  lbb->allocated = PETSC_FALSE;
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