#include <../src/ksp/ksp/utils/lmvm/lmvm.h> /*I "petscksp.h" I*/

/*
  Limited-memory Symmetric-Rank-1 approximation matrix for a Jacobian.
  
  L-SR1 is symmetric by construction, but is not guaranteed to be 
  positive-definite.
  
  The solution method is adapted from Algorithm 8 of Erway and Marcia 
  "On Solving Large-Scale Limited-Memory Quasi-Newton Equations" 
  (https://arxiv.org/abs/1510.06378).
  
  Q <- 0 (zero)
  
  for i = 0,1,2,...,k
    if J0^{-1} exists
      P[i] <- J0^{01} * Y[i]
    elif J0 exists or user_ksp
      P[i] <- inv(J0) * Y[i] via KSP
    elif user_scale
      if diag_scale exists
        P[i] <- VecPointwiseMult(Y[i], diag_scale)
      else
        P[i] <- scale * Y[i]
      end
    else
      P[i] <- Y[i]
    end
    P[i] <- S[i] - P[i]
    for j = 0,1,2,...,i-1
      P[i] <- P[i] - ((P[j]^T Y[i]) / (P[j]^T Y[j])) * P[j]
    end
    Q <- Q + ((P[i]^T F) / (P[i]^T Y[i])) * P[i]
  end
  
  if J0^{-1} exists
    R <- J0^{01} * F
  elif J0 exists or user_ksp
    R <- inv(J0) * F via KSP
  elif user_scale
    if diag_scale exists
      R <- VecPointwiseMult(F, diag_scale)
    else
      R <- scale * F
    end
  else
    R <- F
  end
  
  dX <- R + Q
 */

typedef struct {
  Vec *P;
  PetscBool allocatedP;
} Mat_LSR1;

PetscErrorCode MatSolve_LMVMSR1(Mat B, Vec F, Vec dX)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_LSR1          *lsr1 = (Mat_LSR1*)B->data;
  PetscErrorCode    ierr;
  PetscInt          i, j;
  PetscReal         pjTyi, pjTyj, piTf, piTyi;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(F, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(dX, VEC_CLASSID, 3);
  VecCheckSameSize(F, 2, dX, 3);
  VecCheckMatCompatible(B, dX, 3, F, 2);
  
  /* Start outer loop */
  ierr = VecSet(lmvm->Q, 0.0);CHKERRQ(ierr);
  for (i = 0; i <= lmvm->k; ++i) {
    /* Invert the initial Jacobian onto Y[i] (or apply scaling) */
    ierr = MatLMVMApplyJ0Inv(B, lmvm->Y[i], lsr1->P[i]);CHKERRQ(ierr);
    /* Start the P[i] computation which involves an inner loop */
    ierr = VecAXPBY(lsr1->P[i], 1.0, -1.0, lmvm->S[i]);CHKERRQ(ierr);
    for (j = 0; j <= i-1; ++j) {
      ierr = VecDotBegin(lsr1->P[j], lmvm->Y[i], &pjTyi);CHKERRQ(ierr);
      ierr = VecDotBegin(lsr1->P[j], lmvm->Y[j], &pjTyj);CHKERRQ(ierr);
      ierr = VecDotEnd(lsr1->P[j], lmvm->Y[i], &pjTyi);CHKERRQ(ierr);
      ierr = VecDotEnd(lsr1->P[j], lmvm->Y[j], &pjTyj);CHKERRQ(ierr);
      ierr = VecAXPY(lsr1->P[i], -(pjTyi/pjTyj), lsr1->P[j]);CHKERRQ(ierr);
    }
    /* Accumulate the summation term */
    ierr = VecDotBegin(lsr1->P[i], F, &piTf);CHKERRQ(ierr);
    ierr = VecDotBegin(lsr1->P[i], lmvm->Y[i], &piTyi);CHKERRQ(ierr);
    ierr = VecDotEnd(lsr1->P[i], F, &piTf);CHKERRQ(ierr);
    ierr = VecDotEnd(lsr1->P[i], lmvm->Y[i], &piTyi);CHKERRQ(ierr);
    ierr = VecAXPY(lmvm->Q, (piTf/piTyi), lsr1->P[i]);CHKERRQ(ierr);
  }
  
  /* Invert the initial Jacobian onto F (or apply scaling) */
  ierr = MatLMVMApplyJ0Inv(B, F, lmvm->R);CHKERRQ(ierr);
  
  /* Now we have all the components to compute the solution */
  ierr = VecWAXPY(dX, 1.0, lmvm->R, lmvm->Q);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PETSC_INTERN PetscErrorCode MatReset_LMVMSR1(Mat B, PetscBool destructive)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_LSR1          *lsr1 = (Mat_LSR1*)B->data;
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  ierr = MatReset_LMVM(B, destructive);CHKERRQ(ierr);
  if (lsr1->allocatedP) {
    ierr = VecDestroyVecs(lmvm->m, &lsr1->P);CHKERRQ(ierr);
    lsr1->allocatedP = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PETSC_INTERN PetscErrorCode MatAllocate_LMVMSR1(Mat B, Vec X, Vec F)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_LSR1          *lsr1 = (Mat_LSR1*)B->data;
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  ierr = MatAllocate_LMVM(B, X, F);CHKERRQ(ierr);
  if (!lsr1->allocatedP) {
    ierr = VecDuplicateVecs(X, lmvm->m, &lsr1->P);CHKERRQ(ierr);
    lsr1->allocatedP = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PETSC_INTERN PetscErrorCode MatDestroy_LMVMSR1(Mat B)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_LSR1          *lsr1 = (Mat_LSR1*)B->data;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (lsr1->allocatedP) {
    ierr = VecDestroyVecs(lmvm->m, &lsr1->P);CHKERRQ(ierr);
  }
  ierr = MatDestroy_LMVM(B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PETSC_INTERN PetscErrorCode MatSetUp_LMVMSR1(Mat B)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_LSR1          *lsr1 = (Mat_LSR1*)B->data;
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  ierr = MatSetUp_LMVM(B);CHKERRQ(ierr);
  if (!lsr1->allocatedP) {
    ierr = VecDuplicateVecs(lmvm->Xprev, lmvm->m, &lsr1->P);CHKERRQ(ierr);
    lsr1->allocatedP = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode MatCreate_LMVMSR1(Mat B)
{
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatCreate_LMVM(B);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B, MATLMVMSR1);CHKERRQ(ierr);
  ierr = MatSetOption(B, MAT_SYMMETRIC, PETSC_TRUE);CHKERRQ(ierr);
  B->ops->solve = MatSolve_LMVMSR1;
  B->ops->setup = MatSetUp_LMVMSR1;
  B->ops->destroy = MatDestroy_LMVMSR1;
  Mat_LMVM *lmvm = (Mat_LMVM*)B->data;
  lmvm->square = PETSC_TRUE;
  lmvm->ops->allocate = MatAllocate_LMVMSR1;
  lmvm->ops->reset = MatReset_LMVMSR1;
  Mat_LSR1 *lsr1 = (Mat_LSR1*)B->data;
  lsr1->allocatedP = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/*@
   MatCreateLMVMSR1 - Creates a limited-memory Symmetric-Rank-1 approximation
   matrix used for a Jacobian. L-SR1 is symmetric by construction, but is not 
   guaranteed to be positive-definite. This implementation only supports the 
   MatSolve() operation, which is an application of the approximate inverse of 
   the Jacobian. 
   
   The provided local and global sizes must match the solution and function vectors 
   used with MatLMVMUpdate() and MatSolve(). The resulting L-SR1 matrix will have 
   storage vectors allocated with VecCreateSeq() in serial and VecCreateMPI() in 
   parallel. To use the L-SR1 matrix with other vector types, the matrix must be 
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

.seealso: MatCreate(), MATLMVM, MATLMVMSR1, MatCreateLMVMBFGS(), MatCreateLMVMDFP(), 
          MatCreateLMVMBrdn(), MatCreateLMVMBadBrdn(), MatCreateLMVMSymBrdn()
@*/
PetscErrorCode MatCreateLMVMSR1(MPI_Comm comm, PetscInt n, PetscInt N, Mat *B)
{
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  ierr = MatCreate(comm, B);CHKERRQ(ierr);
  ierr = MatSetSizes(*B, n, n, N, N);CHKERRQ(ierr);
  ierr = MatSetType(*B, MATLMVMSR1);CHKERRQ(ierr);
  ierr = MatSetUp(*B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}