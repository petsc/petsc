#include <../src/ksp/ksp/utils/lmvm/lmvm.h> /*I "petscksp.h" I*/

/*
  Zero-memory Symmetric Broyden method for explicitly approximating 
  the diagonal of a Jacobian.
*/

typedef struct {
  Vec work, Hnew, H, Bs, y2, BFGS, DFP;
  PetscBool allocated;
  PetscReal phi, alpha, beta, rho;
  PetscReal *yty, *yts, *sts;
} Mat_DiagBrdn;

/*------------------------------------------------------------*/

static PetscErrorCode MatSolve_LMVMDiagBrdn(Mat B, Vec F, Vec dX)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_DiagBrdn      *ldb = (Mat_DiagBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(F, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(dX, VEC_CLASSID, 3);
  VecCheckSameSize(F, 2, dX, 3);
  if (!lmvm->allocated) {
    ierr = MatLMVMAllocate(B, dX, F);CHKERRQ(ierr);
  } else {
    VecCheckMatCompatible(B, dX, 3, F, 2);
  }
  ierr = VecPointwiseMult(dX, F, ldb->H);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatMult_LMVMDiagBrdn(Mat B, Vec X, Vec Z)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_DiagBrdn      *ldb = (Mat_DiagBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(X, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(Z, VEC_CLASSID, 3);
  VecCheckSameSize(X, 2, Z, 3);
  if (!lmvm->allocated) {
    ierr = MatLMVMAllocate(B, X, Z);CHKERRQ(ierr);
  } else {
    VecCheckMatCompatible(B, X, 2, Z, 3);
  }
  ierr = VecPointwiseDivide(Z, X, ldb->H);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/*
  The diagonal Hessian update is derived from Equation 2 in 
  Erway and Marcia "On Solving Large-scale Limited-Memory 
  Quasi-Newton Equations" (https://arxiv.org/pdf/1510.06378.pdf).
  In this "zero"-memory implementation, the matrix-matrix products 
  are replaced by pointwise multiplications between their diagonal 
  vectors. Unlike limited-memory methods, the incoming updates 
  are directly applied to the diagonal instead of being stored 
  for later use.
*/
static PetscErrorCode MatUpdate_LMVMDiagBrdn(Mat B, Vec X, Vec F)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_DiagBrdn      *ldb = (Mat_DiagBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  PetscInt          i, old_k;
  PetscReal         yts, stBs;
  PetscReal         yy_sum, ss_sum, ys_sum;
  PetscReal         ytHy, ytBs, sBtBs, stHy, yHtHy, yHtBs;
  PetscReal         denom, sigma;

  PetscFunctionBegin;
  if (lmvm->prev_set) {
    /* Compute the new (S = X - Xprev) and (Y = F - Fprev) vectors */
    ierr = VecAXPBY(lmvm->Xprev, 1.0, -1.0, X);CHKERRQ(ierr);
    ierr = VecAXPBY(lmvm->Fprev, 1.0, -1.0, F);CHKERRQ(ierr);
    /* Test if the updates can be accepted */
    ierr = VecDot(lmvm->Fprev, lmvm->Xprev, &yts);CHKERRQ(ierr);
    if (yts > -lmvm->eps) {
      /* Update is good, accept it */
      ++lmvm->nupdates;
      /*  Set BFGS and DFP components to the forward diagonal initially */
      ierr = VecCopy(ldb->H, ldb->BFGS);CHKERRQ(ierr);
      ierr = VecReciprocal(ldb->BFGS);CHKERRQ(ierr);
      ierr = VecCopy(ldb->BFGS, ldb->DFP);CHKERRQ(ierr);
      /*  Compute y*y^T (Hadamard product) */
      ierr = VecPointwiseMult(ldb->y2, lmvm->Fprev, lmvm->Fprev);CHKERRQ(ierr);
      /*  Compute B*s and s^T Bs */
      ierr = MatMult(B, lmvm->Xprev, ldb->Bs);CHKERRQ(ierr);
      ierr = VecDot(ldb->Bs, lmvm->Xprev, &stBs);CHKERRQ(ierr);
      /*  Safeguard rhotemp and sDs */
      if (yts >= 0.0) {
        yts = lmvm->eps;
      }
      if (0.0 == stBs) {
        stBs = lmvm->eps;
      }
      /*  BFGS portion of the update */
      if (1.0 != ldb->phi) {
        ierr = VecPointwiseMult(ldb->work, ldb->Bs, ldb->Bs);CHKERRQ(ierr);
        ierr = VecAXPBYPCZ(ldb->BFGS, 1.0/yts, -1.0/stBs, 1.0, ldb->y2, ldb->work);CHKERRQ(ierr);
      }
      /*  DFP portion of the update */
      if (0.0 != ldb->phi) {
        ierr = VecPointwiseMult(ldb->work, ldb->Bs, lmvm->Fprev);CHKERRQ(ierr);
        ierr = VecAXPBYPCZ(ldb->DFP, 1.0/yts + stBs/(yts*yts), -2.0/yts, 1.0, ldb->y2, ldb->work);CHKERRQ(ierr);
      }
      /* Combine the components into the Broyden update */
      if (0.0 == ldb->phi) {
        ierr = VecCopy(ldb->BFGS, ldb->Hnew);CHKERRQ(ierr);
      } else if (1.0 == ldb->phi) {
        ierr = VecCopy(ldb->DFP, ldb->Hnew);CHKERRQ(ierr);
      } else {
        ierr = VecAXPBYPCZ(ldb->Hnew, 1.0-ldb->phi, ldb->phi, 0.0, ldb->BFGS, ldb->DFP);
      }
      /*  Obtain inverse and ensure positive definite */
      ierr = VecReciprocal(ldb->Hnew);CHKERRQ(ierr);
      ierr = VecAbs(ldb->Hnew);CHKERRQ(ierr);
      /* Compute the diagonal scaling */
      if (lmvm->m == 0) {
        /* If history is zero, just use identity scaling */
        sigma = 1.0;
      } else {
        /* Compute the diagonal scaling */
        old_k = lmvm->k;
        ierr = MatUpdateKernel_LMVM(B, lmvm->Xprev, lmvm->Fprev);CHKERRQ(ierr);
        if (old_k == lmvm->k) {
          for (i = 0; i <= lmvm->k-1; ++i) {
            ldb->yty[i] = ldb->yty[i+1];
            ldb->yts[i] = ldb->yts[i+1];
            ldb->sts[i] = ldb->sts[i+1];
          }
        }
        /*  Save information for special cases of scalar rescaling */
        ierr = VecDot(lmvm->Xprev, lmvm->Xprev, &ldb->sts[lmvm->k]);CHKERRQ(ierr);
        ierr = VecDot(lmvm->Fprev, lmvm->Fprev, &ldb->yty[lmvm->k]);CHKERRQ(ierr);
        ldb->yts[lmvm->k] = yts;

        if (ldb->beta == 0.5) {
          /*  Compute summations for scalar scaling */
          yy_sum = 0;       /*  No safeguard required */
          ys_sum = 0;       /*  No safeguard required */
          ss_sum = 0;       /*  No safeguard required */
          for (i = 0; i <= lmvm->k; ++i) {
            ierr = VecPointwiseMult(ldb->work, ldb->Hnew, lmvm->Y[i]);CHKERRQ(ierr);
            ierr = VecDot(lmvm->Y[i], ldb->work, &ytHy);CHKERRQ(ierr);
            yy_sum += ytHy;
            ierr = VecPointwiseDivide(ldb->work, lmvm->S[i], ldb->Hnew);CHKERRQ(ierr);
            ierr = VecDot(lmvm->S[i], ldb->work, &stBs);CHKERRQ(ierr);
            ss_sum += stBs;
            ys_sum += ldb->yts[i];
          }
        } else if (ldb->beta == 0.0) {
          /*  Compute summations for scalar scaling */
          yy_sum = 0;       /*  No safeguard required */
          ys_sum = 0;       /*  No safeguard required */
          ss_sum = 0;       /*  No safeguard required */
          for (i = 0; i <= lmvm->k; ++i) {
            ierr = VecPointwiseDivide(ldb->work, lmvm->S[i], ldb->Hnew);CHKERRQ(ierr);
            ierr = VecDot(lmvm->Y[i], ldb->work, &ytBs);CHKERRQ(ierr);
            ys_sum += ytBs;
            ierr = VecDot(ldb->work, ldb->work, &sBtBs);CHKERRQ(ierr);
            ss_sum += sBtBs;
            yy_sum += ldb->yty[i];
          }
        } else if (ldb->beta == 1.0) {
          /*  Compute summations for scalar scaling */
          yy_sum = 0; /*  No safeguard required */
          ys_sum = 0; /*  No safeguard required */
          ss_sum = 0; /*  No safeguard required */
          for (i = 0; i <= lmvm->k; ++i) {
            ierr = VecPointwiseMult(ldb->work, ldb->Hnew, lmvm->Y[i]);CHKERRQ(ierr);
            ierr = VecDot(lmvm->S[i], ldb->work, &stHy);CHKERRQ(ierr);
            ys_sum += stHy;
            ierr = VecDot(ldb->work, ldb->work, &yHtHy);CHKERRQ(ierr);
            yy_sum += yHtHy;
            ss_sum += ldb->sts[i];
          }
        } else {
          /*  Compute summations for scalar scaling */
          yy_sum = 0; /*  No safeguard required */
          ys_sum = 0; /*  No safeguard required */
          ss_sum = 0; /*  No safeguard required */
          ierr = VecCopy(ldb->Hnew, ldb->Bs);CHKERRQ(ierr);
          ierr = VecPow(ldb->Bs, ldb->beta);CHKERRQ(ierr);
          ierr = VecPointwiseDivide(ldb->y2, ldb->Bs, ldb->Hnew);CHKERRQ(ierr);
          for (i = 0; i <= lmvm->k; ++i) {
            ierr = VecPointwiseMult(ldb->Bs, ldb->Bs, lmvm->Y[i]);CHKERRQ(ierr);
            ierr = VecPointwiseMult(ldb->y2, ldb->y2, lmvm->S[i]);CHKERRQ(ierr);
            ierr = VecDot(ldb->Bs, ldb->Bs, &yHtHy);CHKERRQ(ierr);
            ierr = VecDot(ldb->Bs, ldb->y2, &yHtBs);CHKERRQ(ierr);
            ierr = VecDot(ldb->y2, ldb->y2, &sBtBs);CHKERRQ(ierr);
            yy_sum += yHtHy;
            ys_sum += yHtBs;
            ss_sum += sBtBs;
          }
        }

        if (ldb->alpha == 0.0) {
          /*  Safeguard ys_sum  */
          if (0.0 == ys_sum) {
            ys_sum = lmvm->eps;
          }
          sigma = ss_sum / ys_sum;
        } else if (ldb->alpha == 1.0) {
          /*  Safeguard yy_sum  */
          if (0.0 == yy_sum) {
            ys_sum = lmvm->eps;
          }
          sigma = ys_sum / yy_sum;
        } else {
          denom = 2*ldb->alpha*yy_sum;
          /*  Safeguard denom */
          if (denom == 0.0) {
            denom = lmvm->eps;
          }
          sigma = ((2*ldb->alpha-1)*ys_sum + PetscSqrtReal((2*ldb->alpha-1)*(2*ldb->alpha-1)*ys_sum*ys_sum - 4*ldb->alpha*(ldb->alpha-1)*yy_sum*ss_sum)) / denom;
        }
      }
      ierr = VecScale(ldb->Hnew, sigma);CHKERRQ(ierr);
      /* Use the rho limiter to combine the new diagonal with the old one */
      if (ldb->rho == 1.0) {
        ierr = VecCopy(ldb->Hnew, ldb->H);CHKERRQ(ierr);
      } else if (ldb->rho) {
        ierr = VecAXPBY(ldb->H, ldb->rho, 1.0 - ldb->rho, ldb->Hnew);CHKERRQ(ierr);
      }
    } else {
      /* Update is bad, skip it */
      ++lmvm->nrejects;
    }
  } else {
    ierr = VecSet(ldb->H, 1.0/lmvm->J0scalar);CHKERRQ(ierr);
  }
  /* Save the solution and function to be used in the next update */
  ierr = VecCopy(X, lmvm->Xprev);CHKERRQ(ierr);
  ierr = VecCopy(F, lmvm->Fprev);CHKERRQ(ierr);
  lmvm->prev_set = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatCopy_LMVMDiagBrdn(Mat B, Mat M, MatStructure str)
{
  Mat_LMVM          *bdata = (Mat_LMVM*)B->data;
  Mat_DiagBrdn      *bctx = (Mat_DiagBrdn*)bdata->ctx;
  Mat_LMVM          *mdata = (Mat_LMVM*)M->data;
  Mat_DiagBrdn      *mctx = (Mat_DiagBrdn*)mdata->ctx;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  mctx->phi = bctx->phi;
  mctx->alpha = bctx->alpha;
  ierr = VecCopy(bctx->H, mctx->H);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatReset_LMVMDiagBrdn(Mat B, PetscBool destructive)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_DiagBrdn      *ldb = (Mat_DiagBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  if (destructive && ldb->allocated) {
    ierr = PetscFree3(ldb->yty, ldb->yts, ldb->sts);CHKERRQ(ierr);
    ierr = VecDestroy(&ldb->work);CHKERRQ(ierr);
    ierr = VecDestroy(&ldb->Hnew);CHKERRQ(ierr);
    ierr = VecDestroy(&ldb->H);CHKERRQ(ierr);
    ierr = VecDestroy(&ldb->Bs);CHKERRQ(ierr);
    ierr = VecDestroy(&ldb->y2);CHKERRQ(ierr);
    ierr = VecDestroy(&ldb->BFGS);CHKERRQ(ierr);
    ierr = VecDestroy(&ldb->DFP);CHKERRQ(ierr);
    ldb->allocated = PETSC_FALSE;
  } else {
    ierr = VecSet(ldb->H, 1.0/lmvm->J0scalar);CHKERRQ(ierr);
  }
  ierr = MatReset_LMVM(B, destructive);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatAllocate_LMVMDiagBrdn(Mat B, Vec X, Vec F)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_DiagBrdn      *ldb = (Mat_DiagBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  ierr = MatAllocate_LMVM(B, X, F);CHKERRQ(ierr);
  if (!ldb->allocated) {
    ierr = PetscMalloc3(lmvm->m, &ldb->yty, lmvm->m, &ldb->yts, lmvm->m, &ldb->sts);CHKERRQ(ierr);
    ierr = VecDuplicate(X, &ldb->work);CHKERRQ(ierr);
    ierr = VecDuplicate(X, &ldb->Hnew);CHKERRQ(ierr);
    ierr = VecDuplicate(X, &ldb->H);CHKERRQ(ierr);
    ierr = VecDuplicate(X, &ldb->Bs);CHKERRQ(ierr);
    ierr = VecDuplicate(X, &ldb->y2);CHKERRQ(ierr);
    ierr = VecDuplicate(X, &ldb->BFGS);CHKERRQ(ierr);
    ierr = VecDuplicate(X, &ldb->DFP);CHKERRQ(ierr);
    ldb->allocated = PETSC_TRUE;
  }
  ierr = VecSet(ldb->H, 1.0/lmvm->J0scalar);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatDestroy_LMVMDiagBrdn(Mat B)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_DiagBrdn      *ldb = (Mat_DiagBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (ldb->allocated) {
    ierr = PetscFree3(ldb->yty, ldb->yts, ldb->sts);CHKERRQ(ierr);
    ierr = VecDestroy(&ldb->work);CHKERRQ(ierr);
    ierr = VecDestroy(&ldb->Hnew);CHKERRQ(ierr);
    ierr = VecDestroy(&ldb->H);CHKERRQ(ierr);
    ierr = VecDestroy(&ldb->Bs);CHKERRQ(ierr);
    ierr = VecDestroy(&ldb->y2);CHKERRQ(ierr);
    ierr = VecDestroy(&ldb->BFGS);CHKERRQ(ierr);
    ierr = VecDestroy(&ldb->DFP);CHKERRQ(ierr);
    ldb->allocated = PETSC_FALSE;
  }
  ierr = PetscFree(lmvm->ctx);CHKERRQ(ierr);
  ierr = MatDestroy_LMVM(B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatSetUp_LMVMDiagBrdn(Mat B)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_DiagBrdn      *ldb = (Mat_DiagBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  ierr = MatSetUp_LMVM(B);CHKERRQ(ierr);
  if (!ldb->allocated) {
    ierr = PetscMalloc3(lmvm->m, &ldb->yty, lmvm->m, &ldb->yts, lmvm->m, &ldb->sts);CHKERRQ(ierr);
    ierr = VecDuplicate(lmvm->Xprev, &ldb->work);CHKERRQ(ierr);
    ierr = VecDuplicate(lmvm->Xprev, &ldb->Hnew);CHKERRQ(ierr);
    ierr = VecDuplicate(lmvm->Xprev, &ldb->H);CHKERRQ(ierr);
    ierr = VecDuplicate(lmvm->Xprev, &ldb->Bs);CHKERRQ(ierr);
    ierr = VecDuplicate(lmvm->Xprev, &ldb->y2);CHKERRQ(ierr);
    ierr = VecDuplicate(lmvm->Xprev, &ldb->BFGS);CHKERRQ(ierr);
    ierr = VecDuplicate(lmvm->Xprev, &ldb->DFP);CHKERRQ(ierr);
    ldb->allocated = PETSC_TRUE;
  }
  ierr = VecSet(ldb->H, 1.0/lmvm->J0scalar);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatSetFromOptions_LMVMDiagBrdn(PetscOptionItems *PetscOptionsObject, Mat B)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_DiagBrdn      *ldb = (Mat_DiagBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatSetFromOptions_LMVM(PetscOptionsObject, B);CHKERRQ(ierr);
  ierr = PetscOptionsHead(PetscOptionsObject,"Restricted Broyden for approximating the explicit diagonal of an SPD Jacobian (MATLMVMDIAGBRDN)");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_lmvm_phi","(developer) convex ratio between BFGS and DFP components of the diagonal update","",ldb->phi,&ldb->phi,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_lmvm_beta","(developer) exponential factor in diagonal re-scaling","",ldb->beta,&ldb->beta,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_lmvm_alpha","(developer) convex ratio between BFGS and DFP components in the diagonal re-scaling","",ldb->alpha,&ldb->alpha,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_lmvm_rho","(developer) limiter term in combining the new diagonal with the old one","",ldb->rho,&ldb->rho,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  if ((ldb->phi < 0.0) || (ldb->phi > 1.0)) SETERRQ(PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_OUTOFRANGE, "convex ratio for the update formula cannot be outside the range of [0, 1]");
  if ((ldb->alpha < 0.0) || (ldb->alpha > 1.0)) SETERRQ(PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_OUTOFRANGE, "convex ratio for the scaling cannot be outside the range of [0, 1]");
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode MatCreate_LMVMDiagBrdn(Mat B)
{
  Mat_DiagBrdn       *ldb;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatCreate_LMVM(B);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B, MATLMVMDIAGBRDN);CHKERRQ(ierr);
  ierr = MatSetOption(B, MAT_SPD, PETSC_TRUE);CHKERRQ(ierr);
  
  B->ops->solve = MatSolve_LMVMDiagBrdn;
  B->ops->setup = MatSetUp_LMVMDiagBrdn;
  B->ops->destroy = MatDestroy_LMVMDiagBrdn;
  B->ops->setfromoptions = MatSetFromOptions_LMVMDiagBrdn;
  
  Mat_LMVM *lmvm = (Mat_LMVM*)B->data;
  lmvm->m = 1;
  lmvm->square = PETSC_TRUE;
  lmvm->ops->update = MatUpdate_LMVMDiagBrdn;
  lmvm->ops->allocate = MatAllocate_LMVMDiagBrdn;
  lmvm->ops->reset = MatReset_LMVMDiagBrdn;
  lmvm->ops->mult = MatMult_LMVMDiagBrdn;
  lmvm->ops->copy = MatCopy_LMVMDiagBrdn;
  
  ierr = PetscNewLog(B, &ldb);CHKERRQ(ierr);
  lmvm->ctx = (void*)ldb;
  ldb->allocated = PETSC_FALSE;
  ldb->phi = 0.125;
  ldb->beta = 0.5;
  ldb->alpha = 1.0;
  ldb->rho = 1.0;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/*@
   MatCreateLMVMDiagBrdn - Creates a zero-memory symmetric Broyden approximation 
   for the diagonal of a Jacobian. This matrix does not store any LMVM update vectors, 
   and instead uses the full-memory symmetric Broyden formula to update a vector that 
   respresents the diagonal of a Jacobian.

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
.   -mat_lmvm_phi - (developer) convex ratio between BFGS and DFP components of the diagonal update
.   -mat_lmvm_rho - (developer) convex ratio between old diagonal and new diagonal
.   -mat_lmvm_alpha - (developer) convex ratio between BFGS and DFP components in the diagonal re-scaling
.   -mat_lmvm_beta - (developer) exponential factor in the diagonal re-scaling
.   -mat_lmvm_sigma_hist - (developer) number of past updates to use when re-scaling the diagonal

   Level: intermediate

.seealso: MatCreate(), MATLMVM, MATLMVMSYMBRDN, MatCreateLMVMDFP(), MatCreateLMVMSR1(), 
          MatCreateLMVMBFGS(), MatCreateLMVMBrdn(), MatCreateLMVMBadBrdn()
@*/
PetscErrorCode MatCreateLMVMDiagBrdn(MPI_Comm comm, PetscInt n, PetscInt N, Mat *B)
{
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  ierr = MatCreate(comm, B);CHKERRQ(ierr);
  ierr = MatSetSizes(*B, n, n, N, N);CHKERRQ(ierr);
  ierr = MatSetType(*B, MATLMVMDIAGBRDN);CHKERRQ(ierr);
  ierr = MatSetUp(*B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}