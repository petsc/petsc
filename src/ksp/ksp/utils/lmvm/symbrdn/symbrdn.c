#include <../src/ksp/ksp/utils/lmvm/lmvm.h> /*I "petscksp.h" I*/

/*
  Limited-memory Symmetric Broyden method for approximating the inverse 
  of a Jacobian.
  
  L-SymBroyden is a convex combination of L-BFGS and L-DFP such that 
  SymBroyden = (1-phi)*BFGS + phi*DFP. The combination factor phi is restricted 
  to the range [0, 1] where the resulting approximation is guaranteed to be 
  symmetric positive-definite.
*/


typedef struct {
  Vec *P;
  Vec work;
  Vec *Q;
  Vec V, W;
  PetscBool allocated;
  PetscReal phi;
} Mat_SymBrdn;

/*------------------------------------------------------------*/

/*
  The solution method below is the matrix-free implementation of Equation 3 in 
  Erway and Marcia "On Solving Large-Scale Limited-Memory Quasi-Newton Equations" 
  (https://arxiv.org/pdf/1510.06378.pdf). The required forward product is computed 
  using an embedded matrix-free implementation of Equation 2 that re-uses as much 
  information as possible from the outer iterations.
  
  dX <- J0^{-1} * F
  
  for i=0,1,2,...,k
    P[i] <- J0^{-1} * Y[i]
    Q[i] <- J0 * S[i]
    
    for j=0,1,2,...,i-1
      rho = 1.0 / (Y[j]^T S[j])
      alpha = rho * (S[j]^T Y[i])
      zeta = 1.0 / (Y[j]^T P[j])
      gamma = zeta * (Y[j]^T P[i])
      
      P[i] <- P[i] - (gamma * P[j]) + (alpha * Y[j])
      V <- (rho * S[j]) - (zeta * P[j])
      psi = ((1-phi)*(Y[j]^T S[j])^2) / (1-phi)*(Y[j]^T S[j])^2 - phi*(Y[j]^T P[j])*(S[j]^T Q[j]))
      P[i] <- P[i] + (psi * (Y[j]^T P[j]) * (V^T F) * V)
      
      sigma = rho * (Y[j]^T S[i])
      nu = 1.0 / (S[j]^T Q[j])
      beta = nu * (S[j]^T Q[i])
      
      Q[i] <- Q[i] - (beta * Q[j]) + (sigma * S[j])
      W <- (rho * Y[j]) - (nu * Q[j])
      Q[i] <- Q[i] + (phi * (S[j]^T Q[j]) * (W^T S[i]) * W)
    end
    
    rho = 1.0 / (Y[i]^T S[i])
    alpha = rho * (S[i]^T F)
    zeta = 1.0 / (Y[i]^T P[i])
    gamma = zeta * (Y[i]^T dX)
    
    dX <- dX - (gamma * P[i]) + (alpha * Y[i])
    V <- (rho * S[i]) - (zeta * P[i])
    psi = ((1-phi)*(Y[i]^T S[i])^2) / (1-phi)*(Y[i]^T S[i])^2 - phi*(Y[i]^T P[i])*(S[i]^T Q[i]))
    dX <- dX + (psi * (Y[i]^T P[i]) * (V^T F) * V)
  end
*/
PetscErrorCode MatSolve_LMVMSymBrdn(Mat B, Vec F, Vec dX)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_SymBrdn       *lsb = (Mat_SymBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  PetscInt          i, j;
  PetscReal         yts[lmvm->k+1], ytp[lmvm->k+1], stq[lmvm->k+1];
  PetscReal         stf, ytx, vtf, yjtpi, sjtyi, vtyi, psi, sjtqi, yjtsi, wtsi; 
  
  PetscFunctionBegin;
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

  /* Start the outer loop (i) for the recursive formula */
  ierr = MatLMVMApplyJ0Inv(B, F, dX);CHKERRQ(ierr);
  for (i = 0; i <= lmvm->k; ++i) {
    /* First compute P[i] = (B^{-1})_i * y_i using an inner loop (j) 
       NOTE: This is essentially the same recipe used for dX, but we don't 
             recurse because reuse P[i] from previous outer iterations. */
    ierr = MatLMVMApplyJ0Inv(B, lmvm->Y[i], lsb->P[i]);
    ierr = MatLMVMApplyJ0Fwd(B, lmvm->S[i], lsb->Q[i]);
    for (j = 0; j <= i-1; ++j) {
      /* This is the pure DFP component of restricted Broyden */
      ierr = VecDotBegin(lmvm->Y[j], lsb->P[i], &yjtpi);CHKERRQ(ierr);
      ierr = VecDotBegin(lmvm->S[j], lmvm->Y[i], &sjtyi);CHKERRQ(ierr);
      ierr = VecDotEnd(lmvm->Y[j], lsb->P[i], &yjtpi);CHKERRQ(ierr);
      ierr = VecDotEnd(lmvm->S[j], lmvm->Y[i], &sjtyi);CHKERRQ(ierr);
      ierr = VecAXPBYPCZ(lsb->P[i], -yjtpi/ytp[j], sjtyi/yts[j], 1.0, lsb->P[j], lmvm->S[j]);CHKERRQ(ierr);
      /* Now we tack on the extra stuff scaled with the convex factor psi */
      ierr = VecAXPBYPCZ(lsb->V, 1.0/yts[j], -1.0/ytp[j], 0.0, lmvm->S[j], lsb->P[j]);CHKERRQ(ierr);
      ierr = VecDot(lsb->V, lmvm->Y[i], &vtyi);CHKERRQ(ierr);
      psi = ((1-lsb->phi)*yts[j]*yts[j]) / (((1-lsb->phi)*yts[j]*yts[j]) + (lsb->phi*ytp[j]*stq[j]));
      ierr = VecAXPY(lsb->P[i], psi*ytp[j]*vtyi, lsb->V);CHKERRQ(ierr);
      /* We need to also compute B_i * s_i for the next outer iteration's convex factor */
      ierr = VecDotBegin(lmvm->S[j], lsb->Q[i], &sjtqi);CHKERRQ(ierr);
      ierr = VecDotBegin(lmvm->Y[j], lmvm->S[i], &yjtsi);CHKERRQ(ierr);
      ierr = VecDotEnd(lmvm->S[j], lsb->Q[i], &sjtqi);CHKERRQ(ierr);
      ierr = VecDotEnd(lmvm->Y[j], lmvm->S[i], &yjtsi);CHKERRQ(ierr);
      ierr = VecAXPBYPCZ(lsb->Q[i], -sjtqi/stq[j], yjtsi/yts[j], 1.0, lsb->Q[j], lmvm->Y[i]);CHKERRQ(ierr);
      ierr = VecAXPBYPCZ(lsb->W, 1.0/yts[j], -1.0/stq[j], 0.0, lmvm->Y[j], lsb->Q[j]);CHKERRQ(ierr);
      ierr = VecDot(lsb->W, lmvm->S[i], &wtsi);CHKERRQ(ierr);
      ierr = VecAXPY(lsb->Q[i], lsb->phi*stq[j]*wtsi, lsb->W);CHKERRQ(ierr);
    }
    /* Get all the dot products we need 
       NOTE: yTs and yTp are stored so that we can re-use them when computing 
             P[i] and Q[i] at the next outer iteration */
    ierr = VecDotBegin(lmvm->Y[i], lmvm->S[i], &yts[i]);CHKERRQ(ierr);
    ierr = VecDotBegin(lmvm->Y[i], lsb->P[i], &ytp[i]);CHKERRQ(ierr);
    ierr = VecDotBegin(lmvm->Y[i], dX, &ytx);CHKERRQ(ierr);
    ierr = VecDotBegin(lmvm->S[i], F, &stf);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->Y[i], lmvm->S[i], &yts[i]);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->Y[i], lsb->P[i], &ytp[i]);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->Y[i], dX, &ytx);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->S[i], F, &stf);CHKERRQ(ierr);
    /* This is the pure DFP component of restricted Broyden */
    ierr = VecAXPBYPCZ(dX, -ytx/ytp[i], stf/yts[i], 1.0, lsb->P[i], lmvm->S[i]);CHKERRQ(ierr);
    /* Now we takc on the extra stuff with the scaled convex factor psi */
    ierr = VecAXPBYPCZ(lsb->V, 1.0/yts[i], -1.0/ytp[i], 0.0, lmvm->S[i], lsb->P[i]);CHKERRQ(ierr);
    ierr = VecDotBegin(lmvm->S[i], lsb->Q[i], &stq[i]);CHKERRQ(ierr);
    ierr = VecDotBegin(lsb->V, F, &vtf);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->S[i], lsb->Q[i], &stq[i]);CHKERRQ(ierr);
    ierr = VecDotEnd(lsb->V, F, &vtf);CHKERRQ(ierr);
    psi = ((1-lsb->phi)*yts[i]*yts[i]) / (((1-lsb->phi)*yts[i]*yts[i]) + (lsb->phi*ytp[i]*stq[i]));
    ierr = VecAXPY(dX, psi*ytp[i]*vtf, lsb->V);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/*
  Symmetric/Restricted Broyden is defined as the convex combination of 
  BFGS and DFP such that S-Brdn = (1-phi)*BFGS + phi*DFP. The implementation 
  of the matrix-free product here combines the DFP double-loop with the BFGS 
  nested-loop, where the BFGS component is embedded into the 2nd loop (inside-out) 
  for the DFP product.
  
  if (phi == 0.0)
    MatMult_LMVMBFGS(X, Z)
  elif (phi == 1.0)
    MatSolve_LMVMDFP(X, Z)
  end

  V <- X

  for i = k,k-1,k-2,...,0
    rho[i] = 1 / (Y[i]^T S[i])
    alpha[i] = rho[i] * (Y[i]^T V)
    V <- V - (alpha[i] * S[i])
  end

  W <- J0 * V
  Z <- J0 * X

  for i = 0,1,2,...,k
    P[i] <- J0 & S[i]

    for j=0,1,2,...,(i-1)
      zeta = (S[j]^T P[i]) / (S[j]^T P[j])
      gamma = (Y[j]^T S[i]) / (Y[j]^T S[j])
      P[i] <- P[i] - (zeta * P[j]) + (gamma * Y[j])
    end

    zeta = (S[i]^T Z) / (Y[i]^T P[i])
    gamma = (Y[i]^T X) / (Y[i]^T S[i])
    dX <- dX - (zeta * P[i]) + (gamma * Y[i])

    beta = rho[i] * (S[i]^T W)
    W <- W + ((alpha[i] - beta) * Y[i])
  end

  Z <- ((1 - phi) * Z) + (phi * W)
*/
PetscErrorCode MatMult_LMVMSymBrdn(Mat B, Vec X, Vec Z)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_SymBrdn       *lsb = (Mat_SymBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  PetscInt          i, j;
  PetscReal         yts[lmvm->k+1], stp[lmvm->k+1], alpha[lmvm->k+1], rho[lmvm->k+1];
  PetscReal         stz, ytx, sjtpi, yjtsi, beta;
  
  
  PetscFunctionBegin;
  if (lsb->phi == 1.0) {
    ierr = MatMult_LMVMDFP(B, X, Z);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  } 
  if (lsb->phi == 0.0) {
    ierr = MatMult_LMVMBFGS(B, X, Z);CHKERRQ(ierr);
  }
  
  PetscValidHeaderSpecific(X, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(Z, VEC_CLASSID, 3);
  VecCheckSameSize(X, 2, Z, 3);
  VecCheckMatCompatible(B, X, 2, Z, 3);

  /* Copy the function into the work vector for the first loop */
  ierr = VecCopy(X, lsb->V);CHKERRQ(ierr);

  /* Start the first loop for the DFP component only */
  for (i = lmvm->k; i >= 0; --i) {
    /* Compute all the dot products we need
       NOTE: yTs is stored for re-use in the second iteration (inside-out) */
    ierr = VecDotBegin(lmvm->Y[i], lmvm->S[i], &yts[i]);CHKERRQ(ierr);
    ierr = VecDotBegin(lmvm->Y[i], lsb->V, &ytx);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->Y[i], lmvm->S[i], &yts[i]);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->Y[i], lsb->V, &ytx);CHKERRQ(ierr);
    rho[i] = 1.0/yts[i];
    alpha[i] = rho[i] * ytx;
    ierr = VecAXPY(lsb->V, -alpha[i], lmvm->S[i]);CHKERRQ(ierr);
  }

  /* Apply the forward product with initial Jacobian */
  ierr = MatLMVMApplyJ0Fwd(B, X, Z);CHKERRQ(ierr);
  if (lmvm->k >= 0) {
    ierr = MatLMVMApplyJ0Fwd(B, lsb->V, lsb->W);CHKERRQ(ierr);
  }

  /* Start the second loop for both DFP and BFGS components*/
  for (i = 0; i <= lmvm->k; ++i) {
    /* First compute P[i] = B_i * s_i using an inner loop (j) 
       NOTE: This is essentially the same recipe used for dX in BFGS, but we don't 
             recurse because reuse P[i] from previous outer iterations. */
    ierr = MatLMVMApplyJ0Fwd(B, lmvm->S[i], lsb->P[i]);
    for (j = 0; j <= i-1; ++j) {
       ierr = VecDotBegin(lmvm->S[j], lsb->P[i], &sjtpi);CHKERRQ(ierr);
       ierr = VecDotBegin(lmvm->Y[j], lmvm->S[i], &yjtsi);CHKERRQ(ierr);
       ierr = VecDotEnd(lmvm->S[j], lsb->P[i], &sjtpi);CHKERRQ(ierr);
       ierr = VecDotEnd(lmvm->Y[j], lmvm->S[i], &yjtsi);CHKERRQ(ierr);
       ierr = VecAXPBYPCZ(lsb->P[i], -sjtpi/stp[j], yjtsi/yts[j], 1.0, lsb->P[j], lmvm->Y[j]);CHKERRQ(ierr);
    }
    /* Get all the dot products we need 
       NOTE: sTp is stored so that we can re-use it when computing P[i] at the next outer iteration */
    ierr = VecDotBegin(lmvm->S[i], lsb->P[i], &stp[i]);CHKERRQ(ierr);
    ierr = VecDotBegin(lmvm->S[i], Z, &stz);CHKERRQ(ierr);
    ierr = VecDotBegin(lmvm->Y[i], X, &ytx);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->S[i], lsb->P[i], &stp[i]);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->S[i], Z, &stz);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->Y[i], X, &ytx);CHKERRQ(ierr);
    /* Update the BFGS component */
    ierr = VecAXPBYPCZ(Z, -stz/stp[i], ytx/yts[i], 1.0, lsb->P[i], lmvm->Y[i]);CHKERRQ(ierr);
    /* Update the DFP component */
    beta = rho[i] * stz;
    ierr = VecAXPY(lsb->W, alpha[i]-beta, lmvm->Y[i]);CHKERRQ(ierr);
  }
  
  /* Compute convex combination of BFGS and DFP */
  ierr = VecAXPBY(Z, lsb->phi, (1.0 - lsb->phi), lsb->W);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PETSC_INTERN PetscErrorCode MatReset_LMVMSymBrdn(Mat B, PetscBool destructive)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_SymBrdn       *lsb = (Mat_SymBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  if (destructive && lsb->allocated && lmvm->m > 0) {
    ierr = VecDestroy(&lsb->work);CHKERRQ(ierr);
    ierr = VecDestroy(&lsb->V);CHKERRQ(ierr);
    ierr = VecDestroy(&lsb->W);CHKERRQ(ierr);
    ierr = VecDestroyVecs(lmvm->m, &lsb->P);CHKERRQ(ierr);
    ierr = VecDestroyVecs(lmvm->m, &lsb->Q);CHKERRQ(ierr);
    lsb->allocated = PETSC_FALSE;
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
  if (!lsb->allocated && lmvm->m > 0) {
    ierr = VecDuplicate(X, &lsb->work);CHKERRQ(ierr);
    ierr = VecDuplicate(X, &lsb->V);CHKERRQ(ierr);
    ierr = VecDuplicate(F, &lsb->W);CHKERRQ(ierr);
    ierr = VecDuplicateVecs(X, lmvm->m, &lsb->P);CHKERRQ(ierr);
    ierr = VecDuplicateVecs(X, lmvm->m, &lsb->Q);CHKERRQ(ierr);
    lsb->allocated = PETSC_TRUE;
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
  if (lsb->allocated && lmvm->m > 0) {
    ierr = VecDestroy(&lsb->work);CHKERRQ(ierr);
    ierr = VecDestroy(&lsb->V);CHKERRQ(ierr);
    ierr = VecDestroy(&lsb->W);CHKERRQ(ierr);
    ierr = VecDestroyVecs(lmvm->m, &lsb->P);CHKERRQ(ierr);
    ierr = VecDestroyVecs(lmvm->m, &lsb->Q);CHKERRQ(ierr);
    lsb->allocated = PETSC_FALSE;
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
  if (!lsb->allocated && lmvm->m > 0) {
    ierr = VecDuplicate(lmvm->Xprev, &lsb->work);CHKERRQ(ierr);
    ierr = VecDuplicate(lmvm->Xprev, &lsb->V);CHKERRQ(ierr);
    ierr = VecDuplicate(lmvm->Fprev, &lsb->W);CHKERRQ(ierr);
    ierr = VecDuplicateVecs(lmvm->Xprev, lmvm->m, &lsb->P);CHKERRQ(ierr);
    ierr = VecDuplicateVecs(lmvm->Xprev, lmvm->m, &lsb->Q);CHKERRQ(ierr);
    lsb->allocated = PETSC_TRUE;
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