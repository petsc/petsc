#include <../src/ksp/ksp/utils/lmvm/lmvm.h> /*I "petscksp.h" I*/

/*
  Limited-memory Broyden-Fletcher-Goldfarb-Shano method for approximating the 
  inverse of a Jacobian.
  
  BFGS is symmetric positive-definite by construction.
  
  The solution method (approximate inverse Jacobian application) is adapted 
  from Algorithm 7.4 on page 178 of Nocedal and Wright "Numerical Optimization" 
  2nd edition (https://doi.org/10.1007/978-0-387-40065-5). The initial inverse 
  Jacobian application falls back onto the gamma scaling recommended in equation 
  (7.20) if the user has not provided any estimation of the initial Jacobian or 
  its inverse.
  
  Fwork <- F
  
  for i = k,k-1,k-2,...,0
    rho[i] = 1 / (Y[i]^T S[i])
    alpha[i] = rho[i] * (S[i]^T Fwork)
    Fwork <- Fwork - (alpha[i] * Y[i])
  end
  
  dX <- J0^{-1} * Fwork
  
  for i = 0,1,2,...,k
    beta = rho[i] * (Y[i]^T dX)
    dX <- dX + ((alpha[i] - beta) * S[i])
  end
 */

PetscErrorCode MatSolve_LMVMBFGS(Mat B, Vec F, Vec dX)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
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
  ierr = VecCopy(F, lmvm->Fwork);CHKERRQ(ierr);
  
  /* Start the first loop */
  for (i = lmvm->k; i >= 0; --i) {
    ierr = VecDotBegin(lmvm->Y[i], lmvm->S[i], &yts);CHKERRQ(ierr);
    ierr = VecDotBegin(lmvm->S[i], lmvm->Fwork, &stf);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->Y[i], lmvm->S[i], &yts);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->S[i], lmvm->Fwork, &stf);CHKERRQ(ierr);
    if (i == lmvm->k) yts_k = yts; /* save this for later in case we need it for J0 */
    rho[i] = 1.0/yts;
    alpha[i] = rho[i] * stf;
    ierr = VecAXPY(lmvm->Fwork, -alpha[i], lmvm->Y[i]);CHKERRQ(ierr);
  }
  
  /* Invert the initial Jacobian onto Q (or apply scaling) */
  ierr = MatLMVMApplyJ0Inv(B, lmvm->Fwork, dX);CHKERRQ(ierr);
  
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

PetscErrorCode MatCreate_LMVMBFGS(Mat B)
{
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatCreate_LMVM(B);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B, MATLMVMBFGS);CHKERRQ(ierr);
  ierr = MatSetOption(B, MAT_SPD, PETSC_TRUE);CHKERRQ(ierr);
  B->ops->solve = MatSolve_LMVMBFGS;
  Mat_LMVM *lmvm = (Mat_LMVM*)B->data;
  lmvm->square = PETSC_TRUE;
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