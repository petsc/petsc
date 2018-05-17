#include <../src/ksp/ksp/utils/lmvm/lmvm.h> /*I "petscksp.h" I*/

/*
  Limited-memory Symmetric Broyden method for approximating the inverse 
  of a Jacobian.
  
  L-SymBroyden is a convex combination of L-BFGS and L-DFP such that 
  SymBroyden = (1-phi)*BFGS + phi*DFP. The combination factor phi is restricted 
  to the range [0, 1] where the resulting approximation is guaranteed to be 
  symmetric positive-definite.
  
  The solution method below is the BFGS two-loop where the second loop (inside out) 
  has the DFP solution embedded in it. At the end, the BFGS and DFP components are 
  combined using the convex ratio. For the stand-alone algorithms for BFGS and DFP, 
  see the MATLMVMBFGS and MATLMVMDFP implementations.
  
  if (phi == 1.0)
    MatSolve_LMVMBFGS(F, dX)
  elif (phi == 0.0)
    MatSolve_LMVMDFP(F, dX)
  end
  
  Fwork <- F

  for i = k,k-1,k-2,...,0
    rho[i] = 1 / (Y[i]^T S[i])
    alpha[i] = rho[i] * (S[i]^T Fwork)
    Fwork <- Fwork - (alpha[i] * Y[i])
  end

  Xwork <- J0^{-1} * Fwork
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
    
    beta = rho[i] * (Y[i]^T Xwork)
    Xwork <- Xwork + ((alpha[i] - beta) * S[i])
  end
  
  dX <- ((1 - phi) * dX) + (phi * Xwork)
 */

typedef struct {
  Vec *P;
  PetscBool allocatedP;
  PetscReal phi;
} Mat_SymBrdn;

PetscErrorCode MatSolve_LMVMSymBrdn(Mat B, Vec F, Vec dX)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_SymBrdn       *lsb = (Mat_SymBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  PetscInt          i, j;
  PetscReal         alpha[lmvm->k+1], rho[lmvm->k+1], yts[lmvm->k+1], ytp[lmvm->k+1];
  PetscReal         beta, stf, stx, sjtpi, ytxwork, ytx, yjtpi;
  PetscReal         bfgs = 1.0 - lsb->phi, dfp = lsb->phi;
  
  PetscFunctionBegin;
  if (bfgs == 1.0) {
    ierr = MatSolve_LMVMBFGS(B, F, dX);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  } 
  if (dfp == 1.0) {
    ierr = MatSolve_LMVMDFP(B, F, dX);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  PetscValidHeaderSpecific(F, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(dX, VEC_CLASSID, 3);
  VecCheckSameSize(F, 2, dX, 3);
  VecCheckMatCompatible(B, dX, 3, F, 2);
  
  /* Copy the function into the work vector for the first loop */
  ierr = VecCopy(F, lmvm->Fwork);CHKERRQ(ierr);

  /* Start the first loop (outside in) for BFGS, but store some useful dot products */
  for (i = lmvm->k; i >= 0; --i) {
    ierr = VecDotBegin(lmvm->Y[i], lmvm->S[i], &yts[i]);CHKERRQ(ierr);
    ierr = VecDotBegin(lmvm->S[i], lmvm->Fwork, &stf);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->Y[i], lmvm->S[i], &yts[i]);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->S[i], lmvm->Fwork, &stf);CHKERRQ(ierr);
    rho[i] = 1.0/yts[i];
    alpha[i] = rho[i] * stf;
    ierr = VecAXPY(lmvm->Fwork, -alpha[i], lmvm->Y[i]);CHKERRQ(ierr);
  }

  /* Apply the initial Jacobian inversion for BFGS only */
  ierr = MatLMVMApplyJ0Inv(B, lmvm->Fwork, lmvm->Xwork);CHKERRQ(ierr);
  
  /* Initialize the DFP part of the update with the initial Jacobian */
  ierr = MatLMVMApplyJ0Inv(B, F, dX);CHKERRQ(ierr);
  
  /* Start the outer loop (i) for the recursive formula */
  for (i = 0; i <= lmvm->k; ++i) {
    /* First compute P[i] = (B^{-1})_i * y_i using an inner loop (j) 
       NOTE: This is essentially the same recipe used for dX, but we don't 
             recurse because reuse P[i] from previous outer iterations. */
    ierr = MatLMVMApplyJ0Inv(B, lmvm->Y[i], lsb->P[i]);
    for (j = 0; j <= i-1; ++j) {
       ierr = VecDotBegin(lmvm->Y[j], lsb->P[i], &yjtpi);CHKERRQ(ierr);
       ierr = VecDotBegin(lmvm->S[j], lsb->P[i], &sjtpi);CHKERRQ(ierr);
       ierr = VecDotEnd(lmvm->Y[j], lsb->P[i], &yjtpi);CHKERRQ(ierr);
       ierr = VecDotEnd(lmvm->S[j], lsb->P[i], &sjtpi);CHKERRQ(ierr);
       ierr = VecAXPBYPCZ(lsb->P[i], -yjtpi/ytp[j], sjtpi/yts[j], 1.0, lsb->P[j], lmvm->S[j]);CHKERRQ(ierr);
    }
    /* Get all the dot products we need 
       NOTE: yTs and yTp are stored so that we can re-use them when computing 
             P[i] at the next outer iteration */
    ierr = VecDotBegin(lmvm->Y[i], lsb->P[i], &ytp[i]);CHKERRQ(ierr);
    ierr = VecDotBegin(lmvm->Y[i], lmvm->Xwork, &ytxwork);CHKERRQ(ierr);
    ierr = VecDotBegin(lmvm->Y[i], dX, &ytx);CHKERRQ(ierr);
    ierr = VecDotBegin(lmvm->S[i], dX, &stx);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->Y[i], lsb->P[i], &ytp[i]);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->Y[i], lmvm->Xwork, &ytxwork);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->Y[i], dX, &ytx);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->S[i], dX, &stx);CHKERRQ(ierr);
    /* Compute the DFP part of the update */
    ierr = VecAXPBYPCZ(dX, -ytx/ytp[i], stx/yts[i], 1.0, lsb->P[i], lmvm->S[i]);CHKERRQ(ierr);
    /* Now compute the BFGS part of it */
    beta = rho[i] * ytxwork;
    ierr = VecAXPY(lmvm->Xwork, alpha[i]-beta, lmvm->S[i]);CHKERRQ(ierr);
  }
  
  /* Assemble the final restricted-class Broyden solution */
  ierr = VecAXPBY(dX, bfgs, dfp, lmvm->Xwork);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PETSC_INTERN PetscErrorCode MatReset_LMVMSymBrdn(Mat B, PetscBool destructive)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_SymBrdn       *lsb = (Mat_SymBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  if (destructive && lsb->allocatedP && lmvm->m > 0) {
    ierr = VecDestroyVecs(lmvm->m, &lsb->P);CHKERRQ(ierr);
    lsb->allocatedP = PETSC_FALSE;
  }
  ierr = MatReset_LMVM(B, destructive);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PETSC_INTERN PetscErrorCode MatAllocate_LMVMSymBrdn(Mat B, Vec X, Vec F)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_SymBrdn       *lsb = (Mat_SymBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  ierr = MatAllocate_LMVM(B, X, F);CHKERRQ(ierr);
  if (!lsb->allocatedP && lmvm->m > 0) {
    ierr = VecDuplicateVecs(X, lmvm->m, &lsb->P);CHKERRQ(ierr);
    lsb->allocatedP = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PETSC_INTERN PetscErrorCode MatDestroy_LMVMSymBrdn(Mat B)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_SymBrdn       *lsb = (Mat_SymBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (lsb->allocatedP && lmvm->m > 0) {
    ierr = VecDestroyVecs(lmvm->m, &lsb->P);CHKERRQ(ierr);
    lsb->allocatedP = PETSC_FALSE;
  }
  ierr = PetscFree(lmvm->ctx);CHKERRQ(ierr);
  ierr = MatDestroy_LMVM(B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PETSC_INTERN PetscErrorCode MatSetUp_LMVMSymBrdn(Mat B)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_SymBrdn       *lsb = (Mat_SymBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  ierr = MatSetUp_LMVM(B);CHKERRQ(ierr);
  if (!lsb->allocatedP && lmvm->m > 0) {
    ierr = VecDuplicateVecs(lmvm->Xprev, lmvm->m, &lsb->P);CHKERRQ(ierr);
    lsb->allocatedP = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PETSC_INTERN PetscErrorCode MatSetFromOptions_LMVMSymBrdn(PetscOptionItems *PetscOptionsObject, Mat B)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_SymBrdn       *lsb = (Mat_SymBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatSetFromOptions_LMVM(PetscOptionsObject, B);CHKERRQ(ierr);
  ierr = PetscOptionsHead(PetscOptionsObject,"Limited-memory Variable Metric matrix for approximating Jacobians");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_lmvm_phi","(developer) convex ratio between BFGS and DFP components in the Broyden update","",lsb->phi,&lsb->phi,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  if ((lsb->phi < 0.0) || (lsb->phi > 1.0)) SETERRQ(PetsObjectComm((PetscObject)B), PETSC_ERR_ARG_OUTOFRANGE, "convex ratio cannot be outside the range of [0, 1]");
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode MatCreate_LMVMSymBrdn(Mat B)
{
  Mat_SymBrdn       *lsb;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatCreate_LMVM(B);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B, MATLMVMSYMBRDN);CHKERRQ(ierr);
  ierr = MatSetOption(B, MAT_SPD, PETSC_TRUE);CHKERRQ(ierr);
  
  B->ops->solve = MatSolve_LMVMSymBrdn;
  B->ops->setfromoptions = MatSetFromOptions_LMVMSymBrdn;
  B->ops->setup = MatSetUp_LMVMSymBrdn;
  B->ops->destroy = MatDestroy_LMVMSymBrdn;
  
  Mat_LMVM *lmvm = (Mat_LMVM*)B->data;
  lmvm->square = PETSC_TRUE;
  lmvm->ops->allocate = MatAllocate_LMVMSymBrdn;
  lmvm->ops->reset = MatReset_LMVMSymBrdn;
  
  ierr = PetscNewLog(B, &lsb);CHKERRQ(ierr);
  lmvm->ctx = (void*)lsb;
  lsb->allocatedP = PETSC_FALSE;
  lsb->phi = 0.125;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/*@
   MatCreateLMVMSymBrdn - Creates a limited-memory Symmetric Broyden-type matrix used 
   for approximating Jacobians. L-SymBrdn is a convex combination of L-DFP and 
   L-BFGS such that SymBrdn = (1-phi)*BFGS + phi*DFP. The combination factor 
   phi is typically restricted to the range [0, 1], where the L-SymBrdn matrix 
   is guaranteed to be symmetric positive-definite. This implementation of L-SymBrdn 
   only supports the MatSolve() operation, which is an application of the approximate 
   inverse of the Jacobian. 
   
   The provided local and global sizes must match the solution and function vectors 
   used with MatLMVMUpdate() and MatSolve(). The resulting L-SymBrdn matrix will have 
   storage vectors allocated with VecCreateSeq() in serial and VecCreateMPI() in 
   parallel. To use the L-SymBrdn matrix with other vector types, the matrix must be 
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
.   -mat_lmvm_phi - (developer) convex ratio between BFGS and DFP components of the inverse
.   -mat_lmvm_phi_diag - (developer) convex ratio between BFGS and DFP components of the diagonal

   Level: intermediate

.seealso: MatCreate(), MATLMVM, MATLMVMSYMBRDN, MatCreateLMVMDFP(), MatCreateLMVMSR1(), 
          MatCreateLMVMBFGS(), MatCreateLMVMBrdn(), MatCreateLMVMBadBrdn()
@*/
PetscErrorCode MatCreateLMVMSymBrdn(MPI_Comm comm, PetscInt n, PetscInt N, Mat *B)
{
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  ierr = MatCreate(comm, B);CHKERRQ(ierr);
  ierr = MatSetSizes(*B, n, n, N, N);CHKERRQ(ierr);
  ierr = MatSetType(*B, MATLMVMSYMBRDN);CHKERRQ(ierr);
  ierr = MatSetUp(*B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}