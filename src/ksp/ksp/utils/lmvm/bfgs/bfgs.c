#include <../src/ksp/ksp/utils/lmvm/symbrdn/symbrdn.h> /*I "petscksp.h" I*/

/*
  Limited-memory Broyden-Fletcher-Goldfarb-Shano method for approximating both 
  the forward product and inverse application of a Jacobian.
*/

/*------------------------------------------------------------*/

/*
  The solution method (approximate inverse Jacobian application) is adapted 
   from Algorithm 7.4 on page 178 of Nocedal and Wright "Numerical Optimization" 
   2nd edition (https://doi.org/10.1007/978-0-387-40065-5). The initial inverse 
   Jacobian application falls back onto the gamma scaling recommended in equation 
   (7.20) if the user has not provided any estimation of the initial Jacobian or 
   its inverse.

   work <- F

   for i = k,k-1,k-2,...,0
     rho[i] = 1 / (Y[i]^T S[i])
     alpha[i] = rho[i] * (S[i]^T work)
     Fwork <- work - (alpha[i] * Y[i])
   end

   dX <- J0^{-1} * work

   for i = 0,1,2,...,k
     beta = rho[i] * (Y[i]^T dX)
     dX <- dX + ((alpha[i] - beta) * S[i])
   end
*/
PetscErrorCode MatSolve_LMVMBFGS(Mat B, Vec F, Vec dX)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_SymBrdn       *lbfgs = (Mat_SymBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  PetscInt          i;
  PetscReal         alpha[lmvm->k+1], rho[lmvm->k+1];
  PetscReal         beta, stf, ytx;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(F, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(dX, VEC_CLASSID, 3);
  VecCheckSameSize(F, 2, dX, 3);
  VecCheckMatCompatible(B, dX, 3, F, 2);
  
  /* Copy the function into the work vector for the first loop */
  ierr = VecCopy(F, lbfgs->work);CHKERRQ(ierr);
  
  /* Start the first loop */
  for (i = lmvm->k; i >= 0; --i) {
    ierr = VecDot(lmvm->S[i], lbfgs->work, &stf);CHKERRQ(ierr);
    rho[i] = 1.0/lbfgs->yts[i];
    alpha[i] = rho[i] * stf;
    ierr = VecAXPY(lbfgs->work, -alpha[i], lmvm->Y[i]);CHKERRQ(ierr);
  }
  
  /* Invert the initial Jacobian onto the work vector (or apply scaling) */
  ierr = MatSymBrdnApplyJ0Inv(B, lbfgs->work, dX);CHKERRQ(ierr);
  
  /* Start the second loop */
  for (i = 0; i <= lmvm->k; ++i) {
    ierr = VecDot(lmvm->Y[i], dX, &ytx);CHKERRQ(ierr);
    beta = rho[i] * ytx;
    ierr = VecAXPY(dX, alpha[i]-beta, lmvm->S[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/*
  The forward product for the approximate Jacobian is the matrix-free 
  implementation of Equation (6.19) in Nocedal and Wright "Numerical 
  Optimization" 2nd Edition, pg 140.
  
  Note that this forward product has the same structure as the 
  inverse Jacobian application in the DFP formulation, except with S 
  and Y exchanging roles.

  Z <- J0 * X

  for i = 0,1,2,...,k
    P[i] <- J0 * S[i]
    for j = 0,1,2,...,(i-1)
      gamma = (Y[j]^T S[i]) / (Y[j]^T S[j])
      zeta = (S[j]^ P[i]) / (S[j]^T P[j])
      P[i] <- P[i] - (zeta * P[j]) + (gamma * Y[j])
    end
    gamma = (Y[i]^T X) / (Y[i]^T S[i])
    zeta = (S[i]^T Z) / (S[i]^T P[i])
    Z <- Z - (zeta * P[i]) + (gamma * Y[i])
  end
*/
PetscErrorCode MatMult_LMVMBFGS(Mat B, Vec X, Vec Z)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_SymBrdn       *lbfgs = (Mat_SymBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  PetscInt          i;
  PetscReal         ytx, stz;
  
  PetscFunctionBegin;
  /* Start the outer loop (i) for the recursive formula */
  ierr = MatSymBrdnApplyJ0Fwd(B, X, Z);CHKERRQ(ierr);
  for (i = 0; i <= lmvm->k; ++i) {
    /* Get all the dot products we need */
    ierr = VecDotBegin(lmvm->S[i], Z, &stz);CHKERRQ(ierr);
    ierr = VecDotBegin(lmvm->Y[i], X, &ytx);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->S[i], Z, &stz);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->Y[i], X, &ytx);CHKERRQ(ierr);
    /* Update Z_{i+1} = B_{i+1} * X */
    ierr = VecAXPBYPCZ(Z, -stz/lbfgs->stp[i], ytx/lbfgs->yts[i], 1.0, lbfgs->P[i], lmvm->Y[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatUpdate_LMVMBFGS(Mat B, Vec X, Vec F)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_SymBrdn       *lbfgs = (Mat_SymBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  PetscInt          old_k, i, j;
  PetscReal         curvature, ytytmp, rhotol;
  PetscReal         sjtpi, yjtsi;
  Vec               Ptmp;

  PetscFunctionBegin;
  if (lmvm->m == 0) PetscFunctionReturn(0);
  if (lmvm->prev_set) {
    /* Compute the new (S = X - Xprev) and (Y = F - Fprev) vectors */
    ierr = VecAXPBY(lmvm->Xprev, 1.0, -1.0, X);CHKERRQ(ierr);
    ierr = VecAXPBY(lmvm->Fprev, 1.0, -1.0, F);CHKERRQ(ierr);
    /* Test if the updates can be accepted */
    ierr = VecDot(lmvm->Xprev, lmvm->Fprev, &curvature);CHKERRQ(ierr);
    ierr = VecDot(lmvm->Fprev, lmvm->Fprev, &ytytmp);CHKERRQ(ierr);
    rhotol = lmvm->eps * ytytmp;
    if (curvature > rhotol) {
      /* Update is good, accept it */
      old_k = lmvm->k;
      ierr = MatUpdateKernel_LMVM(B, lmvm->Xprev, lmvm->Fprev);CHKERRQ(ierr);
      /* If we hit the memory limit, shift the P and Q vectors */
      if (old_k == lmvm->k) {
        Ptmp = lbfgs->P[0];
        for (i = 0; i <= lmvm->k-1; ++i) {
          lbfgs->P[i] = lbfgs->P[i+1];
          lbfgs->stp[i] = lbfgs->stp[i+1];
          lbfgs->yts[i] = lbfgs->yts[i+1];
          lbfgs->yty[i] = lbfgs->yty[i+1];
          lbfgs->sts[i] = lbfgs->sts[i+1];
        }
        lbfgs->P[lmvm->k] = Ptmp;
      }
      /* Update history of useful scalars */
      lbfgs->yty[lmvm->k] = ytytmp;
      lbfgs->yts[lmvm->k] = curvature;
      ierr = VecDot(lmvm->S[lmvm->k], lmvm->S[lmvm->k], &lbfgs->sts[lmvm->k]);CHKERRQ(ierr);
      /* Update the scaling */
      switch (lbfgs->scale_type) {
      case SYMBRDN_SCALE_SCALAR:
        ierr = MatSymBrdnComputeJ0Scalar(B);CHKERRQ(ierr);
        break;
      case SYMBRDN_SCALE_DIAG:
        ierr = MatSymBrdnComputeJ0Diag(B);CHKERRQ(ierr);
        break;
      case SYMBRDN_SCALE_NONE:
      default:
        break;
      }
      /* Pre-compute (P[i] = B_i * S[i]) */
      for (i = 0; i <= lmvm->k; ++i) {
        ierr = MatSymBrdnApplyJ0Fwd(B, lmvm->S[i], lbfgs->P[i]);CHKERRQ(ierr);
        for (j = 0; j <= i-1; ++j) {
          /* Compute the necessary dot products */
          ierr = VecDotBegin(lmvm->S[j], lbfgs->P[i], &sjtpi);CHKERRQ(ierr);
          ierr = VecDotBegin(lmvm->Y[j], lmvm->S[i], &yjtsi);CHKERRQ(ierr);
          ierr = VecDotEnd(lmvm->S[j], lbfgs->P[i], &sjtpi);CHKERRQ(ierr);
          ierr = VecDotEnd(lmvm->Y[j], lmvm->S[i], &yjtsi);CHKERRQ(ierr);
          /* Compute the pure BFGS component of the forward product */
          ierr = VecAXPBYPCZ(lbfgs->P[i], -sjtpi/lbfgs->stp[j], yjtsi/lbfgs->yts[j], 1.0, lbfgs->P[j], lmvm->Y[j]);CHKERRQ(ierr);
        }
        ierr = VecDot(lmvm->S[i], lbfgs->P[i], &lbfgs->stp[i]);CHKERRQ(ierr);
      }
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

static PetscErrorCode MatCopy_LMVMBFGS(Mat B, Mat M, MatStructure str)
{
  Mat_LMVM          *bdata = (Mat_LMVM*)B->data;
  Mat_SymBrdn       *bctx = (Mat_SymBrdn*)bdata->ctx;
  Mat_LMVM          *mdata = (Mat_LMVM*)M->data;
  Mat_SymBrdn       *mctx = (Mat_SymBrdn*)mdata->ctx;
  PetscErrorCode    ierr;
  PetscInt          i;

  PetscFunctionBegin;
  for (i=0; i<=bdata->k; ++i) {
    mctx->stp[i] = bctx->stp[i];
    mctx->yts[i] = bctx->yts[i];
    ierr = VecCopy(bctx->P[i], mctx->P[i]);CHKERRQ(ierr);
  }
  mctx->scale_type = bctx->scale_type;
  mctx->alpha = bctx->alpha;
  mctx->beta = bctx->beta;
  mctx->rho = bctx->rho;
  mctx->sigma_hist = bctx->sigma_hist;
  switch (bctx->scale_type) {
  case SYMBRDN_SCALE_SCALAR:
    mctx->sigma = bctx->sigma;
    break;
  case SYMBRDN_SCALE_DIAG:
    ierr = VecCopy(bctx->invD, mctx->invD);CHKERRQ(ierr);
    break;
  case SYMBRDN_SCALE_NONE:
  default:
    break;
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatReset_LMVMBFGS(Mat B, PetscBool destructive)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_SymBrdn       *lbfgs = (Mat_SymBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  if (lbfgs->allocated) {
    if (destructive) {
      ierr = VecDestroy(&lbfgs->work);CHKERRQ(ierr);
      ierr = PetscFree4(lbfgs->stp, lbfgs->yts, lbfgs->yty, lbfgs->sts);CHKERRQ(ierr);
      if (lmvm->m > 0) {
        ierr = VecDestroyVecs(lmvm->m, &lbfgs->P);CHKERRQ(ierr); 
      }
      switch (lbfgs->scale_type) {
      case SYMBRDN_SCALE_DIAG:
        ierr = VecDestroy(&lbfgs->invDnew);CHKERRQ(ierr);
        ierr = VecDestroy(&lbfgs->invD);CHKERRQ(ierr);
        ierr = VecDestroy(&lbfgs->BFGS);CHKERRQ(ierr);
        ierr = VecDestroy(&lbfgs->DFP);CHKERRQ(ierr);
        ierr = VecDestroy(&lbfgs->U);CHKERRQ(ierr);
        ierr = VecDestroy(&lbfgs->V);CHKERRQ(ierr);
        ierr = VecDestroy(&lbfgs->W);CHKERRQ(ierr);
        break;
      default:
        break;
      }
      lbfgs->allocated = PETSC_FALSE;
    } else {
      switch (lbfgs->scale_type) {
      case SYMBRDN_SCALE_SCALAR:
        lbfgs->sigma = 1.0;
        break;
      case SYMBRDN_SCALE_DIAG:
        ierr = VecSet(lbfgs->invD, 1.0);CHKERRQ(ierr);
        break;
      case SYMBRDN_SCALE_NONE:
      default:
        break;
      }
    }
  }
  ierr = MatReset_LMVM(B, destructive);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatAllocate_LMVMBFGS(Mat B, Vec X, Vec F)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_SymBrdn       *lbfgs = (Mat_SymBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  ierr = MatAllocate_LMVM(B, X, F);CHKERRQ(ierr);
  if (!lbfgs->allocated) {
    ierr = VecDuplicate(X, &lbfgs->work);CHKERRQ(ierr);
    ierr = PetscMalloc4(lmvm->m, &lbfgs->stp, lmvm->m, &lbfgs->yts, lmvm->m, &lbfgs->yty, lmvm->m, &lbfgs->sts);CHKERRQ(ierr);
    if (lmvm->m > 0) {
      ierr = VecDuplicateVecs(X, lmvm->m, &lbfgs->P);CHKERRQ(ierr);
    }
    switch (lbfgs->scale_type) {
    case SYMBRDN_SCALE_DIAG:
      ierr = VecDuplicate(X, &lbfgs->invDnew);CHKERRQ(ierr);
      ierr = VecDuplicate(X, &lbfgs->invD);CHKERRQ(ierr);
      ierr = VecDuplicate(X, &lbfgs->BFGS);CHKERRQ(ierr);
      ierr = VecDuplicate(X, &lbfgs->DFP);CHKERRQ(ierr);
      ierr = VecDuplicate(X, &lbfgs->U);CHKERRQ(ierr);
      ierr = VecDuplicate(X, &lbfgs->V);CHKERRQ(ierr);
      ierr = VecDuplicate(X, &lbfgs->W);CHKERRQ(ierr);
      break;
    default:
      break;
    }
    lbfgs->allocated = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatDestroy_LMVMBFGS(Mat B)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_SymBrdn       *lbfgs = (Mat_SymBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (lbfgs->allocated) {
    ierr = VecDestroy(&lbfgs->work);CHKERRQ(ierr);
    ierr = PetscFree4(lbfgs->stp, lbfgs->yts, lbfgs->yty, lbfgs->sts);CHKERRQ(ierr);
    if (lmvm->m > 0) {
      ierr = VecDestroyVecs(lmvm->m, &lbfgs->P);CHKERRQ(ierr); 
    }
    switch (lbfgs->scale_type) {
    case SYMBRDN_SCALE_DIAG:
      ierr = VecDestroy(&lbfgs->invDnew);CHKERRQ(ierr);
      ierr = VecDestroy(&lbfgs->invD);CHKERRQ(ierr);
      ierr = VecDestroy(&lbfgs->BFGS);CHKERRQ(ierr);
      ierr = VecDestroy(&lbfgs->DFP);CHKERRQ(ierr);
      ierr = VecDestroy(&lbfgs->U);CHKERRQ(ierr);
      ierr = VecDestroy(&lbfgs->V);CHKERRQ(ierr);
      ierr = VecDestroy(&lbfgs->W);CHKERRQ(ierr);
      break;
    default:
      break;
    }
    lbfgs->allocated = PETSC_FALSE;
  }
  ierr = PetscFree(lmvm->ctx);CHKERRQ(ierr);
  ierr = MatDestroy_LMVM(B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatSetUp_LMVMBFGS(Mat B)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_SymBrdn       *lbfgs = (Mat_SymBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  ierr = MatSetUp_LMVM(B);CHKERRQ(ierr);
  if (!lbfgs->allocated) {
    ierr = VecDuplicate(lmvm->Xprev, &lbfgs->work);CHKERRQ(ierr);
    ierr = PetscMalloc4(lmvm->m, &lbfgs->stp, lmvm->m, &lbfgs->yts, lmvm->m, &lbfgs->yty, lmvm->m, &lbfgs->sts);CHKERRQ(ierr);
    if (lmvm->m > 0) {
      ierr = VecDuplicateVecs(lmvm->Xprev, lmvm->m, &lbfgs->P);CHKERRQ(ierr);
    }
    switch (lbfgs->scale_type) {
    case SYMBRDN_SCALE_DIAG:
      ierr = VecDuplicate(lmvm->Xprev, &lbfgs->invDnew);CHKERRQ(ierr);
      ierr = VecDuplicate(lmvm->Xprev, &lbfgs->invD);CHKERRQ(ierr);
      ierr = VecDuplicate(lmvm->Xprev, &lbfgs->BFGS);CHKERRQ(ierr);
      ierr = VecDuplicate(lmvm->Xprev, &lbfgs->DFP);CHKERRQ(ierr);
      ierr = VecDuplicate(lmvm->Xprev, &lbfgs->U);CHKERRQ(ierr);
      ierr = VecDuplicate(lmvm->Xprev, &lbfgs->V);CHKERRQ(ierr);
      ierr = VecDuplicate(lmvm->Xprev, &lbfgs->W);CHKERRQ(ierr);
      break;
    default:
      break;
    }
    lbfgs->allocated = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatSetFromOptions_LMVMBFGS(PetscOptionItems *PetscOptionsObject, Mat B)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_SymBrdn       *lsb = (Mat_SymBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatSetFromOptions_LMVM(PetscOptionsObject, B);CHKERRQ(ierr);
  ierr = PetscOptionsHead(PetscOptionsObject,"Restricted Broyden method for approximating SPD Jacobian actions (MATLMVMSYMBRDN)");CHKERRQ(ierr);
  ierr = PetscOptionsEList("-mat_lmvm_scale_type", "(developer) scaling type applied to J0", "", Scale_Table, SYMBRDN_SCALE_SIZE, Scale_Table[lsb->scale_type], &lsb->scale_type,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_lmvm_rho","(developer) update limiter in the J0 scaling","",lsb->rho,&lsb->rho,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_lmvm_alpha","(developer) convex ratio in the J0 scaling","",lsb->alpha,&lsb->alpha,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_lmvm_beta","(developer) exponential factor in the diagonal J0 scaling","",lsb->alpha,&lsb->alpha,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mat_lmvm_sigma_hist","(developer) number of past updates to use in the default J0 scalar","",lsb->sigma_hist,&lsb->sigma_hist,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  if ((lsb->alpha < 0.0) || (lsb->alpha > 1.0)) SETERRQ(PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_OUTOFRANGE, "convex ratio in the J0 scaling cannot be outside the range of [0, 1]");
  if ((lsb->rho < 0.0) || (lsb->rho > 1.0)) SETERRQ(PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_OUTOFRANGE, "update limiter in the J0 scaling cannot be outside the range of [0, 1]");
  if (lsb->sigma_hist < 0) SETERRQ(PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_OUTOFRANGE, "J0 scaling history length cannot be negative");
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode MatCreate_LMVMBFGS(Mat B)
{
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatCreate_LMVMSymBrdn(B);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B, MATLMVMBFGS);CHKERRQ(ierr);
  B->ops->solve = MatSolve_LMVMBFGS;
  B->ops->setup = MatSetUp_LMVMBFGS;
  B->ops->destroy = MatDestroy_LMVMBFGS;
  B->ops->setfromoptions = MatSetFromOptions_LMVMBFGS;
  
  Mat_LMVM *lmvm = (Mat_LMVM*)B->data;
  lmvm->ops->allocate = MatAllocate_LMVMBFGS;
  lmvm->ops->reset = MatReset_LMVMBFGS;
  lmvm->ops->update = MatUpdate_LMVMBFGS;
  lmvm->ops->mult = MatMult_LMVMBFGS;
  lmvm->ops->copy = MatCopy_LMVMBFGS;
  
  Mat_SymBrdn *lbfgs = (Mat_SymBrdn*)lmvm->ctx;
  lbfgs->phi = 0.0;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/*@
   MatCreateLMVMBFGS - Creates a limited-memory Broyden-Fletcher-Goldfarb-Shano (BFGS)
   matrix used for approximating Jacobians. L-BFGS is symmetric positive-definite by 
   construction, and is commonly used to approximate Hessians in optimization 
   problems.
   
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