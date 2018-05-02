#include <../src/ksp/ksp/utils/lmvm/lmvm.h> /*I "petscksp.h" I*/

/*
  Limited-memory Symmetric Broyden method for approximating the inverse 
  of a Jacobian.
  
  L-SBroyden is a convex combination of L-BFGS and L-DFP such that 
  SBroyden = (1-phi)*BFGS + phi*DFP. The combination factor phi is typically 
  restricted to the range [0, 1], where the resulting approximation is 
  guaranteed to be symmetric positive-definite. However, phi can be set to 
  values outside of this range, which produces other known quasi-Newton methods 
  such as L-SR1.
  
  Q <- F
  
  for i = k,k-1,k-2,...,0
    rho[i] = 1 / (Y[i]^T S[i])
    alpha_bfgs[i] = rho[i] * (S[i]^T Q)
    alpha_dfp[i] = rho[i] * (Y[i]^T Q)
    Q <- Q - (1-phi)*(alpha_bfgs[i] * Y[i]) - phi*(alpha_dfp[i] * S[i])
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
    beta_bfgs = rho[i] * (Y[i]^T R)
    beta_dfp = rho[i] * (S[i]^T R)
    R <- R + (1-phi)*((alpha_bfgs[i] - beta_bfgs) * S[i]) + phi*((alpha_dfp[i] - beta_dfp) * Y[i])
  end
  
  dX <- R
 */

PetscErrorCode MatSolve_LSBRDN(Mat B, Vec F, Vec dX)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  PetscErrorCode    ierr;
  PetscInt          i;
  PetscReal         alpha_bfgs[lmvm->k+1], alpha_dfp[lmvm->k+1], rho[lmvm->k+1];
  PetscReal         bfgs = 1.0 - lmvm->phi, dfp = lmvm->phi;
  PetscReal         beta_bfgs, beta_dfp, yts, stq, ytq, ytr, str;
  
  PetscFunctionBegin;
  if (bfgs == 1.0) {
    ierr = MatSolve_LBFGS(B, F, dX);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  } 
  if (dfp == 1.0) {
    ierr = MatSolve_LDFP(B, F, dX);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  PetscValidHeaderSpecific(F, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(dX, VEC_CLASSID, 3);
  VecCheckSameSize(F, 2, dX, 3);
  VecCheckMatCompatible(B, dX, 3, F, 2);
  
  /* Copy the function into the work vector for the first loop */
  ierr = VecCopy(F, lmvm->Q);CHKERRQ(ierr);
  
  /* Start the first loop */
  for (i = lmvm->k; i >= 0; --i) {
    ierr = VecDotBegin(lmvm->Y[i], lmvm->S[i], &yts);CHKERRQ(ierr);
    ierr = VecDotBegin(lmvm->S[i], lmvm->Q, &stq);CHKERRQ(ierr);
    ierr = VecDotBegin(lmvm->Y[i], lmvm->Q, &ytq);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->Y[i], lmvm->S[i], &yts);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->S[i], lmvm->Q, &stq);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->Y[i], lmvm->Q, &ytq);CHKERRQ(ierr);
    rho[i] = 1.0/yts;
    alpha_bfgs[i] = rho[i] * stq;
    alpha_dfp[i] = rho[i] * ytq;
    ierr = VecAXPBYPCZ(lmvm->Q, -alpha_bfgs[i]*bfgs, -alpha_dfp[i]*dfp, 1.0, lmvm->Y[i], lmvm->S[i]);CHKERRQ(ierr);
  }
  
  /* Invert the initial Jacobian onto Q (or apply scaling) */
  ierr = MatLMVMApplyJ0Inv(B, lmvm->Q, lmvm->R);CHKERRQ(ierr);
  
  /* Start the second loop */
  for (i = 0; i <= lmvm->k; ++i) {
    ierr = VecDotBegin(lmvm->Y[i], lmvm->R, &ytr);CHKERRQ(ierr);
    ierr = VecDotBegin(lmvm->S[i], lmvm->R, &str);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->Y[i], lmvm->R, &ytr);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->S[i], lmvm->R, &str);CHKERRQ(ierr);
    beta_bfgs = rho[i] * ytr;
    beta_dfp = rho[i] * str;
    ierr = VecAXPBYPCZ(lmvm->R, (alpha_bfgs[i]-beta_bfgs)*bfgs, (alpha_dfp[i] - beta_dfp)*dfp, 1.0, lmvm->S[i], lmvm->Y[i]);CHKERRQ(ierr);
  }
  
  /* R contains the approximate inverse of Jacobian applied to the function,
     so just save it into the output vector */
  ierr = VecCopy(lmvm->R, dX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode MatCreate_LSBRDN(Mat B)
{
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatCreate_LMVM(B);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B, MATLSBRDN);CHKERRQ(ierr);
  ierr = MatSetOption(B, MAT_SPD, PETSC_TRUE);CHKERRQ(ierr);
  B->ops->solve = MatSolve_LSBRDN;
  Mat_LMVM *lmvm = (Mat_LMVM*)B->data;
  lmvm->square = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/*@
   MatCreateLSBRDN - Creates a limited-memory Symmetric Broyden-type matrix used 
   for approximating Jacobians. L-SBroyden is a convex combination of L-DFP and 
   L-BFGS such that SBroyden = (1-phi)*BFGS + phi*DFP. The combination factor 
   phi is typically restricted to the range [0, 1], where the L-SBroyden matrix 
   is guaranteed to be symmetric positive-definite. However, other variants where 
   phi lies outside of this range is possible, and produces some known LMVM 
   methods such as L-SR1. This implementation of L-SBroyden only supports the 
   MatSolve() operation, which is an application of the approximate inverse of 
   the Jacobian. 
   
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

.seealso: MatCreateLDFP(), MatCreateLSR1(), MatCreateLBFGS(), MatCreateLBRDN(), MatCreateLMBRDN()
@*/
PetscErrorCode MatCreateLSBRDN(MPI_Comm comm, PetscInt n, PetscInt N, Mat *B)
{
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  ierr = MatCreate(comm, B);CHKERRQ(ierr);
  ierr = MatSetSizes(*B, n, n, N, N);CHKERRQ(ierr);
  ierr = MatSetType(*B, MATLSBRDN);CHKERRQ(ierr);
  ierr = MatSetUp(*B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}