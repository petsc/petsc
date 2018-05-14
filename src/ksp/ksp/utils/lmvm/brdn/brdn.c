#include <../src/ksp/ksp/utils/lmvm/lmvm.h> /*I "petscksp.h" I*/

/*
  Limited-memory "good" Broyden's method for approximating the inverse of 
  a Jacobian.
  
  Broyden's method is not guaranteed to be symmetric or positive definite.
  
  The solution method is adapted from Algorithm 7.3.1 on page 126 of Kelly 
  "Iterative Methods for Linear and Nonlinear Equations" 
  (https://doi.org/10.1137/1.9781611970944).
  
  Q <- F
  
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
  
  for i=0,1,2,...,k-1
    tau = (S[i]^T R) / (S[i]^T S[i])
    R <- R + (tau * S[i+1])
  end
  
  tau = (S[k]^T R) / (S[k]^T S[k])
  dX <- -(1 / (1 - tau)) * R
 */

PetscErrorCode MatSolve_LMVMBrdn(Mat B, Vec F, Vec dX)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  PetscErrorCode    ierr;
  PetscInt          i;
  PetscReal         str, sts;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  PetscValidHeaderSpecific(F, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(dX, VEC_CLASSID, 3);
  VecCheckSameSize(F, 2, dX, 3);
  VecCheckMatCompatible(B, dX, 3, F, 2);
  
  /* Invert the initial Jacobian onto q (or apply scaling) */
  ierr = MatLMVMApplyJ0Inv(B, F, dX);CHKERRQ(ierr);
  
  /* Start the interior loop */
  for (i = 0; i <= lmvm->k-1; ++i) {
    ierr = VecDotBegin(lmvm->S[i], lmvm->S[i], &sts);CHKERRQ(ierr);
    ierr = VecDotBegin(lmvm->S[i], dX, &str);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->S[i], lmvm->S[i], &sts);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->S[i], dX, &str);CHKERRQ(ierr);
    ierr = VecAXPY(dX, str/sts, lmvm->S[i+1]);CHKERRQ(ierr);
  }
  if (lmvm->k >= 0) {
    ierr = VecDotBegin(lmvm->S[lmvm->k], lmvm->S[lmvm->k], &sts);CHKERRQ(ierr);
    ierr = VecDotBegin(lmvm->S[lmvm->k], dX, &str);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->S[lmvm->k], lmvm->S[lmvm->k], &sts);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->S[lmvm->k], dX, &str);CHKERRQ(ierr);
    ierr = VecScale(dX, (1.0/(1.0 - str/sts)));CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode MatCreate_LMVMBrdn(Mat B)
{
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatCreate_LMVM(B);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B, MATLMVMBRDN);CHKERRQ(ierr);
  B->ops->solve = MatSolve_LMVMBrdn;
  Mat_LMVM *lmvm = (Mat_LMVM*)B->data;
  lmvm->square = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/*@
   MatCreateLMVMBrdn - Creates a limited-memory "good" Broyden-type approximation
   matrix used for a Jacobian. L-Broyden is not guaranteed to be symmetric or 
   positive-definite. This implementation only supports the MatSolve() operation, 
   which is an application of the approximate inverse of the Jacobian. 
   
   The provided local and global sizes must match the solution and function vectors 
   used with MatLMVMUpdate() and MatSolve(). The resulting L-Broyden matrix will have 
   storage vectors allocated with VecCreateSeq() in serial and VecCreateMPI() in 
   parallel. To use the L-Broyden matrix with other vector types, the matrix must be 
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