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

typedef struct {
  PetscBool updateD, allocatedD;
  Vec D, YYT, SST, WWT, W, work;
  PetscReal phi, phi_diag;
} Mat_SymBrdn;

PetscErrorCode MatSolve_LMVMSymBrdn(Mat B, Vec F, Vec dX)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_SymBrdn       *lsb = (Mat_SymBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  PetscInt          i;
  PetscReal         alpha_bfgs[lmvm->k+1], alpha_dfp[lmvm->k+1], rho[lmvm->k+1];
  PetscReal         bfgs = 1.0 - lsb->phi, dfp = lsb->phi;
  PetscReal         beta_bfgs, beta_dfp, yts, stf, ytf, ytx, stx;
  
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
  
  /* Start the first loop */
  for (i = lmvm->k; i >= 0; --i) {
    ierr = VecDotBegin(lmvm->Y[i], lmvm->S[i], &yts);CHKERRQ(ierr);
    ierr = VecDotBegin(lmvm->S[i], lmvm->Fwork, &stf);CHKERRQ(ierr);
    ierr = VecDotBegin(lmvm->Y[i], lmvm->Fwork, &ytf);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->Y[i], lmvm->S[i], &yts);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->S[i], lmvm->Fwork, &stf);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->Y[i], lmvm->Fwork, &ytf);CHKERRQ(ierr);
    rho[i] = 1.0/yts;
    alpha_bfgs[i] = rho[i] * stf;
    alpha_dfp[i] = rho[i] * ytf;
    ierr = VecAXPBYPCZ(lmvm->Fwork, -alpha_bfgs[i]*bfgs, -alpha_dfp[i]*dfp, 1.0, lmvm->Y[i], lmvm->S[i]);CHKERRQ(ierr);
  }
  
  /* Invert the initial Jacobian onto Q (or apply scaling) */
  ierr = MatLMVMApplyJ0Inv(B, lmvm->Fwork, lmvm->Xwork);CHKERRQ(ierr);
  
  /* Start the second loop */
  for (i = 0; i <= lmvm->k; ++i) {
    ierr = VecDotBegin(lmvm->Y[i], lmvm->Xwork, &ytx);CHKERRQ(ierr);
    ierr = VecDotBegin(lmvm->S[i], lmvm->Xwork, &stx);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->Y[i], lmvm->Xwork, &ytx);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->S[i], lmvm->Xwork, &stx);CHKERRQ(ierr);
    beta_bfgs = rho[i] * ytx;
    beta_dfp = rho[i] * stx;
    ierr = VecAXPBYPCZ(lmvm->Xwork, (alpha_bfgs[i]-beta_bfgs)*bfgs, (alpha_dfp[i]-beta_dfp)*dfp, 1.0, lmvm->S[i], lmvm->Y[i]);CHKERRQ(ierr);
  }
  
  /* R contains the approximate inverse of Jacobian applied to the function,
     so just save it into the output vector */
  ierr = VecCopy(lmvm->Xwork, dX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode MatUpdate_LMVMSymBrdn(Mat B, Vec X, Vec F)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_SymBrdn       *lsb = (Mat_SymBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  PetscBool         isAssembled, hasDiag;
  Mat               Amat, Pmat;
  PetscReal         sTDs, yTs;

  PetscFunctionBegin;
  ierr = MatUpdate_LMVM(B, X, F);CHKERRQ(ierr);
  if (!lsb->updateD) PetscFunctionReturn(0);
  
  if (lmvm->k < 0) {
    if (lmvm->user_pc || lmvm->user_ksp || lmvm->J0) {
      if (lmvm->user_pc) {
        ierr = PCGetOperators(lmvm->J0pc, &Amat, &Pmat);CHKERRQ(ierr);
      } else if (lmvm->user_ksp) {
        ierr = KSPGetOperators(lmvm->J0ksp, &Amat, &Pmat);CHKERRQ(ierr);
      } else {
        Amat = lmvm->J0;
      }
      ierr = MatAssembled(Amat, &isAssembled);CHKERRQ(ierr);
      ierr = MatHasOperation(Amat, MATOP_GET_DIAGONAL, &hasDiag);CHKERRQ(ierr);
      if (isAssembled && hasDiag) {
        ierr = MatGetDiagonal(Amat, lsb->D);CHKERRQ(ierr);
      } else {
        ierr = VecSet(lsb->D, 1.0);CHKERRQ(ierr);
      }
    } else if (lmvm->user_scale) {
      if (lmvm->diag_scale) {
        ierr = VecCopy(lmvm->diag_scale, lsb->D);CHKERRQ(ierr);
        ierr = VecReciprocal(lsb->D);CHKERRQ(ierr);
      } else {
        ierr = VecSet(lsb->D, 1.0/lmvm->scale);CHKERRQ(ierr);
      }
    } else {
      ierr = VecSet(lsb->D, 1.0);CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
  }
  
  /* put together W = y/(yTs) - Ds/sTDs */
  ierr = VecPointwiseMult(lsb->work, lsb->D, lmvm->S[lmvm->k]);CHKERRQ(ierr);
  ierr = VecDotBegin(lmvm->S[lmvm->k], lsb->work, &sTDs);CHKERRQ(ierr);
  ierr = VecDotBegin(lmvm->Y[lmvm->k], lmvm->S[lmvm->k], &yTs);CHKERRQ(ierr);
  ierr = VecDotEnd(lmvm->S[lmvm->k], lsb->work, &sTDs);CHKERRQ(ierr);
  ierr = VecDotEnd(lmvm->Y[lmvm->k], lmvm->S[lmvm->k], &yTs);CHKERRQ(ierr);
  ierr = VecAXPBYPCZ(lsb->W, 1.0/yTs, 1.0/sTDs, 0.0, lmvm->Y[lmvm->k], lsb->work);CHKERRQ(ierr);
  
  /* compute the intermediate matrix diagonals */
  ierr = VecPointwiseMult(lsb->SST, lmvm->S[lmvm->k], lmvm->S[lmvm->k]);CHKERRQ(ierr);
  ierr = VecPointwiseMult(lsb->YYT, lmvm->Y[lmvm->k], lmvm->Y[lmvm->k]);CHKERRQ(ierr);
  ierr = VecPointwiseMult(lsb->WWT, lsb->W, lsb->W);CHKERRQ(ierr);
  
  /* update the quasi-Newton diagonal */
  ierr = VecPointwiseMult(lsb->work, lsb->SST, lsb->D);CHKERRQ(ierr);
  ierr = VecPointwiseMult(lsb->work, lsb->D, lsb->work);CHKERRQ(ierr);
  ierr = VecAXPY(lsb->D, -1.0/sTDs, lsb->work);CHKERRQ(ierr);
  ierr = VecAXPY(lsb->D, 1.0/yTs, lsb->YYT);CHKERRQ(ierr);
  ierr = VecAXPY(lsb->D, lsb->phi_diag*sTDs, lsb->WWT);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PETSC_INTERN PetscErrorCode MatReset_LMVMSymBrdn(Mat B, PetscBool destructive)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_SymBrdn       *lsb = (Mat_SymBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  if (destructive && lsb->allocatedD) {
    ierr = VecDestroy(&lsb->D);CHKERRQ(ierr);
    ierr = VecDestroy(&lsb->YYT);CHKERRQ(ierr);
    ierr = VecDestroy(&lsb->SST);CHKERRQ(ierr);
    ierr = VecDestroy(&lsb->WWT);CHKERRQ(ierr);
    ierr = VecDestroy(&lsb->work);CHKERRQ(ierr);
    lsb->allocatedD = PETSC_FALSE;
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
  if (!lsb->allocatedD && lsb->updateD) {
    ierr = VecDuplicate(X, &lsb->D);CHKERRQ(ierr);
    ierr = VecDuplicate(X, &lsb->YYT);CHKERRQ(ierr);
    ierr = VecDuplicate(X, &lsb->SST);CHKERRQ(ierr);
    ierr = VecDuplicate(X, &lsb->WWT);CHKERRQ(ierr);
    ierr = VecDuplicate(X, &lsb->work);CHKERRQ(ierr);
    lsb->allocatedD = PETSC_TRUE;
    ierr = VecSet(lsb->D, 1.0);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PETSC_INTERN PetscErrorCode MatGetDiagonal_LMVMSymBrdn(Mat B, Vec D)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_SymBrdn       *lsb = (Mat_SymBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (!lsb->updateD) SETERRQ(PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_INCOMP, "Approximate Hessian diagonal not enabled in options.");
  if (!lsb->allocatedD) SETERRQ(PetscObjectComm((PetscObject)B), PETSC_ERR_ORDER, "LMVM matrix must be allocated first");
  VecCheckSameSize(D, 2, lsb->D, 3);
  ierr = VecCopy(lsb->D, D);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PETSC_INTERN PetscErrorCode MatDestroy_LMVMSymBrdn(Mat B)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_SymBrdn       *lsb = (Mat_SymBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (lsb->allocatedD) {
    ierr = VecDestroy(&lsb->D);CHKERRQ(ierr);
    ierr = VecDestroy(&lsb->YYT);CHKERRQ(ierr);
    ierr = VecDestroy(&lsb->SST);CHKERRQ(ierr);
    ierr = VecDestroy(&lsb->WWT);CHKERRQ(ierr);
    ierr = VecDestroy(&lsb->work);CHKERRQ(ierr);
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
  if (!lsb->allocatedD && lsb->updateD) {
    ierr = VecDuplicate(lmvm->Xprev, &lsb->D);CHKERRQ(ierr);
    ierr = VecDuplicate(lmvm->Xprev, &lsb->YYT);CHKERRQ(ierr);
    ierr = VecDuplicate(lmvm->Xprev, &lsb->SST);CHKERRQ(ierr);
    ierr = VecDuplicate(lmvm->Xprev, &lsb->WWT);CHKERRQ(ierr);
    ierr = VecDuplicate(lmvm->Xprev, &lsb->work);CHKERRQ(ierr);
    lsb->allocatedD = PETSC_TRUE;
    ierr = VecSet(lsb->D, 1.0);CHKERRQ(ierr);
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
  ierr = PetscOptionsReal("-mat_lmvm_phi_diag","(developer) convex ratio between BFGS and DFP components in the approximate diagonal","",lsb->phi_diag,&lsb->phi_diag,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-mat_lmvm_update_diag","(developer) turn on the Broyden approximation of the Hessian diagonal","",lsb->updateD,&lsb->updateD,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
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
  B->ops->getdiagonal = MatGetDiagonal_LMVMSymBrdn;
  
  Mat_LMVM *lmvm = (Mat_LMVM*)B->data;
  lmvm->square = PETSC_TRUE;
  lmvm->ops->update = MatUpdate_LMVMSymBrdn;
  lmvm->ops->allocate = MatAllocate_LMVMSymBrdn;
  lmvm->ops->reset = MatReset_LMVMSymBrdn;
  
  ierr = PetscNewLog(B, &lsb);CHKERRQ(ierr);
  lmvm->ctx = (void*)lsb;
  lsb->allocatedD = PETSC_FALSE;
  lsb->updateD = PETSC_FALSE;
  lsb->phi = 0.125;
  lsb->phi_diag = 0.125;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/*@
   MatCreateLMVMSymBrdn - Creates a limited-memory Symmetric Broyden-type matrix used 
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