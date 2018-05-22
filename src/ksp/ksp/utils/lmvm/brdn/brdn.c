#include <../src/ksp/ksp/utils/lmvm/lmvm.h> /*I "petscksp.h" I*/

/*
  Limited-memory "good" Broyden's method for approximating the inverse of 
  a Jacobian.
*/

typedef struct {
  Vec *P;
  PetscBool allocated;
} Mat_Brdn;

/*------------------------------------------------------------*/

/*
  The solution method is the matrix-free implementation of the inverse Hessian 
  representation in page 312 of Griewank "Broyden Updating, The Good and The Bad!" 
  (http://www.emis.ams.org/journals/DMJDMV/vol-ismp/45_griewank-andreas-broyden.pdf).
  
  dX <- J0^{-1} * F
  
  for i=0,1,2,...,k
    P[i] <- J0^{-1} * Y[i]
    
    for j=0,1,2,...,(i-1)
      tau = (S[j]^T P[i]) / (S[j]^T P[j])
      P[i] <- P[i] + (tau * (S[j] - P[j]))
    end
    
    tau = (S[i]^T dX) / (S[i]^T P[i])
    dX <- dX + (tau * (S[i] - P[i]))
  end
 */

PetscErrorCode MatSolve_LMVMBrdn(Mat B, Vec F, Vec dX)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_Brdn          *lbrdn = (Mat_Brdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  PetscInt          i, j;
  PetscReal         stp[lmvm->k+1], stx;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  PetscValidHeaderSpecific(F, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(dX, VEC_CLASSID, 3);
  VecCheckSameSize(F, 2, dX, 3);
  VecCheckMatCompatible(B, dX, 3, F, 2);
  
  ierr = MatLMVMApplyJ0Inv(B, F, dX);CHKERRQ(ierr);
  for (i = 0; i <= lmvm->k-1; ++i) {
    ierr = MatLMVMApplyJ0Inv(B, lmvm->Y[i], lbrdn->P[i]);CHKERRQ(ierr);
    for (j = 0; j <= i-1; ++j) {
      ierr = VecDot(lmvm->S[j], lbrdn->P[i], &stx);CHKERRQ(ierr);
      ierr = VecAXPBYPCZ(lbrdn->P[i], stx/stp[j], -stx/stp[j], 1.0, lmvm->S[j], lbrdn->P[j]);CHKERRQ(ierr);
    }
    ierr = VecDotBegin(lmvm->S[i], dX, &stx);CHKERRQ(ierr);
    ierr = VecDotBegin(lmvm->S[i], lbrdn->P[i], &stp[i]);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->S[i], dX, &stx);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->S[i], lbrdn->P[i], &stp[i]);CHKERRQ(ierr);
    ierr = VecAXPBYPCZ(dX, stx/stp[i], -stx/stp[i], 1.0, lmvm->S[i], lbrdn->P[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/*
  The forward product is the matrix-free implementation of Equation 2 in 
  page 302 of Griewank "Broyden Updating, The Good and The Bad!"
  (http://www.emis.ams.org/journals/DMJDMV/vol-ismp/45_griewank-andreas-broyden.pdf).
  
  Z <- J0 * X
  
  for i=0,1,2,...,k
    P[i] <- J0 * S[i]
    
    for j=0,1,2,...,(i-1)
      tau = (S[j]^T S[i]) / (S[j]^T S[j])
      P[i] <- P[i] + (tau * (Y[j] - P[j]))
    end
    
    tau = (S[i]^T X) / (S[i]^T S[i])
    dX <- dX + (tau * (Y[i] - P[i]))
  end
 */

PetscErrorCode MatMult_LMVMBrdn(Mat B, Vec X, Vec Z)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_Brdn          *lbrdn = (Mat_Brdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  PetscInt          i, j;
  PetscReal         sts[lmvm->k+1], stx;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  PetscValidHeaderSpecific(X, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(Z, VEC_CLASSID, 3);
  VecCheckSameSize(X, 2, Z, 3);
  VecCheckMatCompatible(B, X, 2, Z, 3);
  
  ierr = MatLMVMApplyJ0Fwd(B, X, Z);CHKERRQ(ierr);
  for (i = 0; i <= lmvm->k-1; ++i) {
    ierr = MatLMVMApplyJ0Fwd(B, lmvm->S[i], lbrdn->P[i]);CHKERRQ(ierr);
    for (j = 0; j <= i-1; ++j) {
      ierr = VecDot(lmvm->S[j], lmvm->S[i], &stx);CHKERRQ(ierr);
      ierr = VecAXPBYPCZ(lbrdn->P[i], stx/sts[j], -stx/sts[j], 1.0, lmvm->Y[j], lbrdn->P[j]);CHKERRQ(ierr);
    }
    ierr = VecDotBegin(lmvm->S[i], X, &stx);CHKERRQ(ierr);
    ierr = VecDotBegin(lmvm->S[i], lbrdn->P[i], &sts[i]);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->S[i], X, &stx);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->S[i], lbrdn->P[i], &sts[i]);CHKERRQ(ierr);
    ierr = VecAXPBYPCZ(Z, stx/sts[i], -stx/sts[i], 1.0, lmvm->Y[i], lbrdn->P[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PETSC_INTERN PetscErrorCode MatReset_LMVMBrdn(Mat B, PetscBool destructive)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_Brdn          *lbrdn = (Mat_Brdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  if (destructive && lbrdn->allocated && lmvm->m > 0) {
    ierr = VecDestroyVecs(lmvm->m, &lbrdn->P);CHKERRQ(ierr);
    lbrdn->allocated = PETSC_FALSE;
  }
  ierr = MatReset_LMVM(B, destructive);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PETSC_INTERN PetscErrorCode MatAllocate_LMVMBrdn(Mat B, Vec X, Vec F)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_Brdn          *lbrdn = (Mat_Brdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  ierr = MatAllocate_LMVM(B, X, F);CHKERRQ(ierr);
  if (!lbrdn->allocated && lmvm->m > 0) {
    ierr = VecDuplicateVecs(X, lmvm->m, &lbrdn->P);CHKERRQ(ierr);
    lbrdn->allocated = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PETSC_INTERN PetscErrorCode MatDestroy_LMVMBrdn(Mat B)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_Brdn          *lbrdn = (Mat_Brdn*)lmvm->ctx;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (lbrdn->allocated && lmvm->m > 0) {
    ierr = VecDestroyVecs(lmvm->m, &lbrdn->P);CHKERRQ(ierr);
    lbrdn->allocated = PETSC_FALSE;
  }
  ierr = PetscFree(lmvm->ctx);CHKERRQ(ierr);
  ierr = MatDestroy_LMVM(B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PETSC_INTERN PetscErrorCode MatSetUp_LMVMBrdn(Mat B)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_Brdn          *lbrdn = (Mat_Brdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  ierr = MatSetUp_LMVM(B);CHKERRQ(ierr);
  if (!lbrdn->allocated && lmvm->m > 0) {
    ierr = VecDuplicateVecs(lmvm->Xprev, lmvm->m, &lbrdn->P);CHKERRQ(ierr);
    lbrdn->allocated = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode MatCreate_LMVMBrdn(Mat B)
{
  Mat_Brdn          *lbrdn;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatCreate_LMVM(B);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B, MATLMVMBRDN);CHKERRQ(ierr);
  B->ops->mult = MatMult_LMVMBrdn;
  B->ops->solve = MatSolve_LMVMBrdn;
  B->ops->setup = MatSetUp_LMVMBrdn;
  B->ops->destroy = MatDestroy_LMVMBrdn;

  Mat_LMVM *lmvm = (Mat_LMVM*)B->data;
  lmvm->square = PETSC_TRUE;
  lmvm->ops->allocate = MatAllocate_LMVMBrdn;
  lmvm->ops->reset = MatReset_LMVMBrdn;

  ierr = PetscNewLog(B, &lbrdn);CHKERRQ(ierr);
  lmvm->ctx = (void*)lbrdn;
  lbrdn->allocated = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/*@
   MatCreateLMVMBrdn - Creates a limited-memory "good" Broyden-type approximation
   matrix used for a Jacobian. L-Brdn is not guaranteed to be symmetric or 
   positive-definite.
   
   The provided local and global sizes must match the solution and function vectors 
   used with MatLMVMUpdate() and MatSolve(). The resulting L-Brdn matrix will have 
   storage vectors allocated with VecCreateSeq() in serial and VecCreateMPI() in 
   parallel. To use the L-Brdn matrix with other vector types, the matrix must be 
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

.seealso: MatCreate(), MATLMVM, MATLMVMBRDN, MatCreateLMVMDFP(), MatCreateLMVMSR1(), 
         MatCreateLMVMBFGS(), MatCreateLMVMBadBrdn(), MatCreateLMVMSymBrdn()
@*/
PetscErrorCode MatCreateLMVMBrdn(MPI_Comm comm, PetscInt n, PetscInt N, Mat *B)
{
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  ierr = MatCreate(comm, B);CHKERRQ(ierr);
  ierr = MatSetSizes(*B, n, n, N, N);CHKERRQ(ierr);
  ierr = MatSetType(*B, MATLMVMBRDN);CHKERRQ(ierr);
  ierr = MatSetUp(*B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}