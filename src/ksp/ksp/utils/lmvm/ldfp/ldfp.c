#include <../src/ksp/ksp/utils/lmvm/lmvm.h> /*I "petscksp.h" I*/

/*
  Limited-memory Davidon-Fletcher-Powell method for approximating the 
  inverse of a Jacobian.
  
  L-DFP is symmetric positive-definite by construction.
  
  The solution method (approximate inverse Jacobian application) is the dual 
  of the L-BFGS two-loop, where the vectors Y[i] and S[i] swap roles.
  
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

PetscErrorCode MatSolve_LDFP(Mat B, Vec F, Vec dX)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  PetscErrorCode    ierr;
  PetscInt          i;
  PetscReal         alpha[lmvm->k+1], rho[lmvm->k+1];
  PetscReal         beta, sty, ytq, str;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(F, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(dX, VEC_CLASSID, 3);
  VecCheckSameSize(F, 2, dX, 3);
  VecCheckMatCompatible(B, dX, 3, F, 2);

  
  /* Copy the function into the work vector for the first loop */
  ierr = VecCopy(F, lmvm->Q);CHKERRQ(ierr);
  
  /* Start the first loop */
  for (i = lmvm->k; i >= 0; --i) {
    ierr = VecDotBegin(lmvm->S[i], lmvm->Y[i], &sty);CHKERRQ(ierr);
    ierr = VecDotBegin(lmvm->Y[i], lmvm->Q, &ytq);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->S[i], lmvm->Y[i], &sty);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->Y[i], lmvm->Q, &ytq);CHKERRQ(ierr);
    rho[i] = 1.0/sty;
    alpha[i] = rho[i] * ytq;
    ierr = VecAXPY(lmvm->Q, -alpha[i], lmvm->S[i]);CHKERRQ(ierr);
  }
  
  /* Invert the initial Jacobian onto Q (or apply scaling) */
  ierr = MatLMVMApplyJ0Inv(B, lmvm->Q, lmvm->R);CHKERRQ(ierr);
  
  /* Start the second loop */
  for (i = 0; i <= lmvm->k; ++i) {
    ierr = VecDot(lmvm->S[i], lmvm->R, &str);CHKERRQ(ierr);
    beta = rho[i] * str;
    ierr = VecAXPY(lmvm->R, alpha[i]-beta, lmvm->Y[i]);CHKERRQ(ierr);
  }
  
  /* R contains the approximate inverse of Jacobian applied to the function,
     so just save it into the output vector */
  ierr = VecCopy(lmvm->R, dX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode MatCreate_LDFP(Mat B)
{
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatCreate_LMVM(B);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B, MATLDFP);CHKERRQ(ierr);
  ierr = MatSetOption(B, MAT_SPD, PETSC_TRUE);CHKERRQ(ierr);
  B->ops->solve = MatSolve_LDFP;
  Mat_LMVM *lmvm = (Mat_LMVM*)B->data;
  lmvm->square = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/*@
   MatCreateLDFP - Creates a limited-memory Davidon-Fletcher-Powell (DFP) matrix 
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

.seealso: MatCreateLBFGS(), MatCreateLSR1(), MatCreateLBRDN(), MatCreateLMBRDN(), MatCreateLSBRDN()
@*/
PetscErrorCode MatCreateLDFP(MPI_Comm comm, PetscInt n, PetscInt N, Mat *B)
{
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  ierr = MatCreate(comm, B);CHKERRQ(ierr);
  ierr = MatSetSizes(*B, n, n, N, N);CHKERRQ(ierr);
  ierr = MatSetType(*B, MATLDFP);CHKERRQ(ierr);
  ierr = MatSetUp(*B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}