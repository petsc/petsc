#include <../src/ksp/ksp/utils/lmvm/lmvm.h> /*I "petscksp.h" I*/

/*
  Limited-memory Symmetric Broyden method for approximating both 
  the forward product and inverse application of a Jacobian.
*/

typedef struct {
  Vec *P;
  Vec work;
  PetscBool allocated;
  PetscReal phi;
} Mat_SymBrdn;

/*------------------------------------------------------------*/

/*
  The solution method below is the matrix-free implementation of 
  Equation 18 in Dennis and Wolkowicz "Sizing and Least Change Secant 
  Methods" (http://www.caam.rice.edu/caam/trs/90/TR90-05.pdf).
  
  dX <- J0^{-1} * F
  
  for i=0,1,2,...,k
    P[i] <- J0^{-1} * Y[i]

    for j=0,1,2,...,i-1
      rho = 1.0 / (Y[j]^T S[j])
      alpha = rho * (S[j]^T Y[i])
      zeta = 1.0 / (Y[j]^T P[j])
      gamma = zeta * (Y[j]^T P[i])
      
      P[i] <- P[i] - (gamma * P[j]) + (alpha * Y[j])
      W <- (rho * S[j]) - (zeta * P[j])
      P[i] <- P[i] + ((1 - phi) * (Y[j]^T P[j]) * (W^T F) * W)
    end
    
    rho = 1.0 / (Y[i]^T S[i])
    alpha = rho * (S[i]^T F)
    zeta = 1.0 / (Y[i]^T P[i])
    gamma = zeta * (Y[i]^T dX)
    
    dX <- dX - (gamma * P[i]) + (alpha * Y[i])
    W <- (rho * S[i]) - (zeta * P[i])
    dX <- dX + ((1 - phi) * (Y[i]^T P[i]) * (W^T F) * W)
  end
*/
static PetscErrorCode MatSolve_LMVMSymBrdn(Mat B, Vec F, Vec dX)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_SymBrdn       *lsb = (Mat_SymBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  PetscInt          i, j;
  PetscReal         yts[lmvm->k+1], ytp[lmvm->k+1];
  PetscReal         ytx, stf, wtf, yjtpi, sjtyi, wtyi; 
  
  PetscFunctionBegin;
  /* Efficient shortcuts for pure BFGS and pure DFP configurations */
  if (lsb->phi == 1.0) {
    ierr = MatSolve_LMVMDFP(B, F, dX);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  if (lsb->phi == 0.0) {
    ierr = MatSolve_LMVMBFGS(B, F, dX);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  
  PetscValidHeaderSpecific(F, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(dX, VEC_CLASSID, 3);
  VecCheckSameSize(F, 2, dX, 3);
  VecCheckMatCompatible(B, dX, 3, F, 2);

  /* Start the outer iterations for ((B^{-1}) * dX) */
  ierr = MatLMVMApplyJ0Inv(B, F, dX);CHKERRQ(ierr);
  for (i = 0; i <= lmvm->k; ++i) {
    /* Start the inner iterations for ((B^{-1})_i * y_i) */
    ierr = MatLMVMApplyJ0Inv(B, lmvm->Y[i], lsb->P[i]);CHKERRQ(ierr);
    for (j = 0; j <= i-1; ++j) {
      /* Compute the necessary dot products -- re-use yTs and yTp from outer iterations */
      ierr = VecDotBegin(lmvm->Y[j], lsb->P[i], &yjtpi);CHKERRQ(ierr);
      ierr = VecDotBegin(lmvm->S[j], lmvm->Y[i], &sjtyi);CHKERRQ(ierr);
      ierr = VecDotEnd(lmvm->Y[j], lsb->P[i], &yjtpi);CHKERRQ(ierr);
      ierr = VecDotEnd(lmvm->S[j], lmvm->Y[i], &sjtyi);CHKERRQ(ierr);
      /* Compute the pure DFP component */
      ierr = VecAXPBYPCZ(lsb->P[i], -yjtpi/ytp[j], sjtyi/yts[j], 1.0, lsb->P[j], lmvm->S[j]);CHKERRQ(ierr);
      /* Tack on the convexly scaled extras */
      ierr = VecAXPBYPCZ(lsb->work, 1.0/yts[j], -1.0/ytp[j], 0.0, lmvm->S[j], lsb->P[j]);CHKERRQ(ierr);
      ierr = VecDot(lsb->work, lmvm->Y[i], &wtyi);CHKERRQ(ierr);
      ierr = VecAXPY(lsb->P[i], (1-lsb->phi)*ytp[j]*wtyi, lsb->work);CHKERRQ(ierr);
    }
    /* Compute the necessary dot products -- store yTs and yTp for inner iterations later */
    ierr = VecDotBegin(lmvm->Y[i], lsb->P[i], &ytp[i]);CHKERRQ(ierr);
    ierr = VecDotBegin(lmvm->Y[i], lmvm->S[i], &yts[i]);CHKERRQ(ierr);
    ierr = VecDotBegin(lmvm->Y[i], dX, &ytx);CHKERRQ(ierr);
    ierr = VecDotBegin(lmvm->S[i], F, &stf);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->Y[i], lsb->P[i], &ytp[i]);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->Y[i], lmvm->S[i], &yts[i]);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->Y[i], dX, &ytx);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->S[i], F, &stf);CHKERRQ(ierr);
    /* Compute the pure DFP component */
    ierr = VecAXPBYPCZ(dX, -ytx/ytp[i], stf/yts[i], 1.0, lsb->P[i], lmvm->S[i]);CHKERRQ(ierr);
    /* Tack on the convexly scaled extras */
    ierr = VecAXPBYPCZ(lsb->work, 1.0/yts[i], -1.0/ytp[i], 0.0, lmvm->S[i], lsb->P[j]);CHKERRQ(ierr);
    ierr = VecDot(lsb->work, F, &wtf);CHKERRQ(ierr);
    ierr = VecAXPY(dX, (1-lsb->phi)*ytp[i]*wtf, lsb->work);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/*
  The forward-product below is the matrix-free implementation of 
  Equation 16 in Dennis and Wolkowicz "Sizing and Least Change Secant 
  Methods" (http://www.caam.rice.edu/caam/trs/90/TR90-05.pdf).
  
  Z <- J0 * X
  
  for i=0,1,2,...,k
    P[i] <- J0 * S[i]

    for j=0,1,2,...,i-1
      rho = 1.0 / (Y[j]^T S[j])
      alpha = rho * (Y[j]^T S[i])
      zeta = 1.0 / (S[j]^T P[j])
      gamma = zeta * (S[j]^T P[i])
      
      P[i] <- P[i] - (gamma * P[j]) + (alpha * S[j])
      W <- (rho * Y[j]) - (zeta * P[j])
      P[i] <- P[i] + ((1 - phi) * (S[j]^T P[j]) * (W^T F) * W)
    end
    
    rho = 1.0 / (Y[i]^T S[i])
    alpha = rho * (Y[i]^T F)
    zeta = 1.0 / (S[i]^T P[i])
    gamma = zeta * (S[i]^T dX)
    
    dX <- dX - (gamma * P[i]) + (alpha * S[i])
    W <- (rho * Y[i]) - (zeta * P[i])
    dX <- dX + ((1 - phi) * (S[i]^T P[i]) * (W^T F) * W)
  end
*/
static PetscErrorCode MatMult_LMVMSymBrdn(Mat B, Vec X, Vec Z)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_SymBrdn       *lsb = (Mat_SymBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  PetscInt          i, j;
  PetscReal         yts[lmvm->k+1], stp[lmvm->k+1];
  PetscReal         stz, ytx, wtx, sjtpi, yjtsi, wtsi;
  
  
  PetscFunctionBegin;
  /* Efficient shortcuts for pure BFGS and pure DFP configurations */
  if (lsb->phi == 1.0) {
    ierr = MatMult_LMVMBFGS(B, X, Z);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  } 
  if (lsb->phi == 0.0) {
    ierr = MatMult_LMVMDFP(B, X, Z);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  /* Start the outer iterations for (B * X) */
  ierr = MatLMVMApplyJ0Fwd(B, X, Z);CHKERRQ(ierr);
  for (i = 0; i <= lmvm->k; ++i) {
    /* Start the inner iterations for (B_i * s_i) */
    ierr = MatLMVMApplyJ0Fwd(B, lmvm->S[i], lsb->P[i]);CHKERRQ(ierr);
    for (j = 0; j <= i-1; ++j) {
      /* Compute the necessary dot products -- re-use yTs and sTp from outer iterations */
      ierr = VecDotBegin(lmvm->S[j], lsb->P[i], &sjtpi);CHKERRQ(ierr);
      ierr = VecDotBegin(lmvm->Y[j], lmvm->S[i], &yjtsi);CHKERRQ(ierr);
      ierr = VecDotEnd(lmvm->S[j], lsb->P[i], &sjtpi);CHKERRQ(ierr);
      ierr = VecDotEnd(lmvm->Y[j], lmvm->S[i], &yjtsi);CHKERRQ(ierr);
      /* Compute the pure BFGS component */
      ierr = VecAXPBYPCZ(lsb->P[i], -sjtpi/stp[j], yjtsi/yts[j], 1.0, lsb->P[j], lmvm->Y[j]);CHKERRQ(ierr);
      /* Tack on the convexly scaled extras */
      ierr = VecAXPBYPCZ(lsb->work, 1.0/yts[j], -1.0/stp[j], 0.0, lmvm->Y[j], lsb->P[j]);CHKERRQ(ierr);
      ierr = VecDot(lsb->work, lmvm->S[i], &wtsi);CHKERRQ(ierr);
      ierr = VecAXPY(lsb->P[i], (1-lsb->phi)*stp[j]*wtsi, lsb->work);CHKERRQ(ierr);
    }
    /* Compute the necessary dot products -- store yTs and sTp for inner iterations later */
    ierr = VecDotBegin(lmvm->S[i], lsb->P[i], &stp[i]);CHKERRQ(ierr);
    ierr = VecDotBegin(lmvm->Y[i], lmvm->S[i], &yts[i]);CHKERRQ(ierr);
    ierr = VecDotBegin(lmvm->S[i], Z, &stz);CHKERRQ(ierr);
    ierr = VecDotBegin(lmvm->Y[i], X, &ytx);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->S[i], lsb->P[i], &stp[i]);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->Y[i], lmvm->S[i], &yts[i]);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->S[i], Z, &stz);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->Y[i], X, &ytx);CHKERRQ(ierr);
    /* Compute the pure BFGS component */
    ierr = VecAXPBYPCZ(Z, -stz/stp[i], ytx/yts[i], 1.0, lsb->P[i], lmvm->Y[i]);CHKERRQ(ierr);
    /* Tack on the convexly scaled extras */
    ierr = VecAXPBYPCZ(lsb->work, 1.0/yts[i], -1.0/stp[i], 0.0, lmvm->Y[i], lsb->P[j]);CHKERRQ(ierr);
    ierr = VecDot(lsb->work, X, &wtx);CHKERRQ(ierr);
    ierr = VecAXPY(Z, (1-lsb->phi)*stp[i]*wtx, lsb->work);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatUpdate_LMVMSymBrdn(Mat B, Vec X, Vec F)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  PetscErrorCode    ierr;
  PetscReal         curvature;

  PetscFunctionBegin;
  if (lmvm->m == 0) PetscFunctionReturn(0);
  if (lmvm->prev_set) {
    /* Compute the new (S = X - Xprev) and (Y = F - Fprev) vectors */
    ierr = VecAXPBY(lmvm->Xprev, 1.0, -1.0, X);CHKERRQ(ierr);
    ierr = VecAXPBY(lmvm->Fprev, 1.0, -1.0, F);CHKERRQ(ierr);
    /* Test if the updates can be accepted */
    ierr = VecDot(lmvm->Xprev, lmvm->Fprev, &curvature);CHKERRQ(ierr);
    if (curvature > -lmvm->eps) {
      /* Update is good, accept it */
      ierr = MatUpdateKernel_LMVM(B, lmvm->Xprev, lmvm->Fprev);CHKERRQ(ierr);
    } else {
      /* Update is bad, skip it */
      ++lmvm->nrejects;
    }
  }

  /* Save the solution and function to be used in the next update */
  ierr = VecCopy(X, lmvm->Xprev);CHKERRQ(ierr);
  ierr = VecCopy(F, lmvm->Fprev);CHKERRQ(ierr);
  lmvm->prev_set = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatReset_LMVMSymBrdn(Mat B, PetscBool destructive)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_SymBrdn       *lsb = (Mat_SymBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  if (destructive && lsb->allocated && lmvm->m > 0) {
    ierr = VecDestroy(&lsb->work);CHKERRQ(ierr);
    ierr = VecDestroyVecs(lmvm->m, &lsb->P);CHKERRQ(ierr);
    lsb->allocated = PETSC_FALSE;
  }
  ierr = MatReset_LMVM(B, destructive);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatAllocate_LMVMSymBrdn(Mat B, Vec X, Vec F)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_SymBrdn       *lsb = (Mat_SymBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  ierr = MatAllocate_LMVM(B, X, F);CHKERRQ(ierr);
  if (!lsb->allocated && lmvm->m > 0) {
    ierr = VecDuplicate(X, &lsb->work);CHKERRQ(ierr);
    ierr = VecDuplicateVecs(X, lmvm->m, &lsb->P);CHKERRQ(ierr);
    lsb->allocated = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatDestroy_LMVMSymBrdn(Mat B)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_SymBrdn       *lsb = (Mat_SymBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (lsb->allocated && lmvm->m > 0) {
    ierr = VecDestroy(&lsb->work);CHKERRQ(ierr);
    ierr = VecDestroyVecs(lmvm->m, &lsb->P);CHKERRQ(ierr);
    lsb->allocated = PETSC_FALSE;
  }
  ierr = PetscFree(lmvm->ctx);CHKERRQ(ierr);
  ierr = MatDestroy_LMVM(B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatSetUp_LMVMSymBrdn(Mat B)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_SymBrdn       *lsb = (Mat_SymBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  ierr = MatSetUp_LMVM(B);CHKERRQ(ierr);
  if (!lsb->allocated && lmvm->m > 0) {
    ierr = VecDuplicate(lmvm->Xprev, &lsb->work);CHKERRQ(ierr);
    ierr = VecDuplicateVecs(lmvm->Xprev, lmvm->m, &lsb->P);CHKERRQ(ierr);
    lsb->allocated = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatSetFromOptions_LMVMSymBrdn(PetscOptionItems *PetscOptionsObject, Mat B)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_SymBrdn       *lsb = (Mat_SymBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatSetFromOptions_LMVM(PetscOptionsObject, B);CHKERRQ(ierr);
  ierr = PetscOptionsHead(PetscOptionsObject,"Limited-memory Variable Metric matrix for approximating Jacobians");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_lmvm_phi","(developer) convex ratio between BFGS and DFP components in the Broyden update","",lsb->phi,&lsb->phi,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  if ((lsb->phi < 0.0) || (lsb->phi > 1.0)) SETERRQ(PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_OUTOFRANGE, "convex ratio cannot be outside the range of [0, 1]");
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
  lmvm->ops->update = MatUpdate_LMVMSymBrdn;
  lmvm->ops->mult = MatMult_LMVMSymBrdn;
  
  ierr = PetscNewLog(B, &lsb);CHKERRQ(ierr);
  lmvm->ctx = (void*)lsb;
  lsb->allocated = PETSC_FALSE;
  lsb->phi = 0.125;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/*@
   MatCreateLMVMSymBrdn - Creates a limited-memory Symmetric Broyden-type matrix used 
   for approximating Jacobians. L-SymBrdn is a convex combination of L-DFP and 
   L-BFGS such that SymBrdn = phi*BFGS + (1-phi)*DFP. The combination factor 
   phi is typically restricted to the range [0, 1], where the L-SymBrdn matrix 
   is guaranteed to be symmetric positive-definite. Note that the convex relationship 
   is inverted for the inverse Hessian approximation, where 
   SymBrdn^{-1} = (1-phi)*BFGS^{-1} + phi*DFP^{-1}.
   
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