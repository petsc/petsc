#include <../src/ksp/ksp/utils/lmvm/lmvm.h> /*I "petscksp.h" I*/

/*
  Zero-memory Symmetric Broyden method for explicitly approximating 
  the diagonal of a Jacobian.
*/

typedef struct {
  Vec Bnew, B, y2, Bs2, w2, work;
  PetscBool allocated;
  PetscReal *ytHpy, *ytHps, *stHps;
  PetscReal phi, alpha, beta, rho;
  PetscInt sigma_hist;
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
  ierr = VecPointwiseDivide(dX, F, ldb->B);CHKERRQ(ierr);
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
  ierr = VecPointwiseMult(Z, X, ldb->B);CHKERRQ(ierr);
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
  PetscReal         yts, stBs, sigma, a, b, c, sig1, sig2;

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
      /* Compute the diagonal scaling */
      if (ldb->sigma_hist == 0) {
        /* If history is zero, just use identity scaling */
        sigma = 1.0;
      } else {
        /* Shift arrays if we hit the history limit */
        old_k = lmvm->k;
        lmvm->k = PetscMin(lmvm->k+1, ldb->sigma_hist-1);
        if ((old_k == lmvm->k) && (ldb->sigma_hist > 0)) {
          for (i = 0; i <= lmvm->k-1; ++i) {
            ldb->ytHpy[i] = ldb->ytHpy[i+1];
            ldb->ytHps[i] = ldb->ytHps[i+1];
            ldb->stHps[i] = ldb->stHps[i+1];
          }
        }
        /* Compute y^T H^{2*beta} y */
        ierr = VecCopy(ldb->B, ldb->work);CHKERRQ(ierr);
        ierr = VecReciprocal(ldb->work);CHKERRQ(ierr);
        ierr = VecPow(ldb->work, 2.0*ldb->beta);CHKERRQ(ierr);
        ierr = VecPointwiseMult(ldb->work, ldb->work, lmvm->Fprev);CHKERRQ(ierr);
        ierr = VecDot(lmvm->Fprev, ldb->work, &ldb->ytHpy[lmvm->k]);CHKERRQ(ierr);
        /* Compute y^T H^{2*beta - 1} s */
        ierr = VecCopy(ldb->B, ldb->work);CHKERRQ(ierr);
        ierr = VecReciprocal(ldb->work);CHKERRQ(ierr);
        ierr = VecPow(ldb->work, 2.0*ldb->beta - 1.0);CHKERRQ(ierr);
        ierr = VecPointwiseMult(ldb->work, ldb->work, lmvm->Xprev);CHKERRQ(ierr);
        ierr = VecDot(lmvm->Fprev, ldb->work, &ldb->ytHps[lmvm->k]);CHKERRQ(ierr);
        /* Compute s^T H^{2*beta - 2} s */
        ierr = VecCopy(ldb->B, ldb->work);CHKERRQ(ierr);
        ierr = VecReciprocal(ldb->work);CHKERRQ(ierr);
        ierr = VecPow(ldb->work, 2.0*ldb->beta - 2.0);CHKERRQ(ierr);
        ierr = VecPointwiseMult(ldb->work, ldb->work, lmvm->Xprev);CHKERRQ(ierr);
        ierr = VecDot(lmvm->Xprev, ldb->work, &ldb->stHps[lmvm->k]);CHKERRQ(ierr);
        /* Compute the diagonal scaling */
        sigma = 0.0;
        if (ldb->alpha == 1.0) {
          for (i = 0; i <= lmvm->k; ++i) {
            sigma += ldb->ytHps[i]/ldb->ytHpy[i];
          }
        } else if (ldb->alpha == 0.5) {
          for (i = 0; i <= lmvm->k; ++i) {
            sigma += ldb->stHps[i]/ldb->ytHpy[i];
          }
          sigma = PetscSqrtReal(sigma);
        } else if (ldb->alpha == 0.0) {
          for (i = 0; i <= lmvm->k; ++i) {
            sigma += ldb->stHps[i]/ldb->ytHps[i];
          }
        } else {
          /* compute coefficients of the quadratic */
          a = b = c = 0.0; 
          for (i = 0; i <= lmvm->k; ++i) {
            a += ldb->ytHpy[i];
            b += ldb->ytHps[i];
            c += ldb->stHps[i];
          }
          a *= ldb->alpha;
          b *= -(2.0*ldb->alpha - 1.0);
          c *= ldb->alpha - 1.0;
          /* use quadratic formula to find roots */
          sig1 = (-b + PetscSqrtReal(b*b - 4.0*a*c))/(2.0*a);
          sig2 = (-b - PetscSqrtReal(b*b - 4.0*a*c))/(2.0*a);
          /* accept the positive root as the scalar */
          if (sig1 > 0.0) {
            sigma = sig1;
          } else if (sig2 > 0.0) {
            sigma = sig2;
          } else {
            SETERRQ(PetscObjectComm((PetscObject)B), PETSC_ERR_CONV_FAILED, "Cannot find positive scalar");
          }
        }
      }
      /* Compute Bs and stBs */
      ierr = MatMult(B, lmvm->Xprev, ldb->Bs2);CHKERRQ(ierr);
      ierr = VecDot(lmvm->Xprev, ldb->Bs2, &stBs);CHKERRQ(ierr);
      /* Compute w = y/yts - Bs/stBs  and then w*w^T (Hadamard product)*/
      ierr = VecAXPBYPCZ(ldb->w2, 1.0/yts, -1.0/stBs, 0.0, lmvm->Fprev, ldb->Bs2);CHKERRQ(ierr);
      ierr = VecPow(ldb->w2, 2.0);CHKERRQ(ierr);
      /* Compute (Bs)*(Bs)^T diagonal (Hadamard product) */
      ierr = VecPow(ldb->Bs2, 2.0);CHKERRQ(ierr);
      /* Compute y*y^T diagonal (Hadamard product) */
      ierr = VecPointwiseMult(ldb->y2, lmvm->Fprev, lmvm->Fprev);CHKERRQ(ierr);
      /* Assemble the inverse diagonal starting with the pure BFGS component */
      ierr = VecCopy(ldb->B, ldb->Bnew);CHKERRQ(ierr);
      ierr = VecAXPBYPCZ(ldb->Bnew, 1.0/yts, -1.0/stBs, 1.0, ldb->y2, ldb->Bs2);CHKERRQ(ierr);
      /* Add the convexly scaled DFP component */
      ierr = VecAXPY(ldb->Bnew, ldb->phi*stBs, ldb->w2);CHKERRQ(ierr);
      /* Combine the new diagonal with the old one using the scaling */
      if (ldb->rho == 1.0) {
        ierr = VecCopy(ldb->Bnew, ldb->B);CHKERRQ(ierr);
        ierr = VecScale(ldb->B, 1.0/sigma);CHKERRQ(ierr);
      } else {
        ierr = VecReciprocal(ldb->B);CHKERRQ(ierr);
        ierr = VecReciprocal(ldb->Bnew);CHKERRQ(ierr);
        ierr = VecAXPBY(ldb->B, ldb->rho*sigma, 1.0 - ldb->rho, ldb->Bnew);CHKERRQ(ierr);
        ierr = VecReciprocal(ldb->B);CHKERRQ(ierr);
      }
    } else {
      /* Update is bad, skip it */
      ++lmvm->nrejects;
    }
  } else {
    ierr = VecSet(ldb->B, 1.0);CHKERRQ(ierr);
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
  PetscInt          i;

  PetscFunctionBegin;
  mctx->phi = bctx->phi;
  mctx->alpha = bctx->alpha;
  mctx->beta = bctx->beta;
  mctx->rho = bctx->rho;
  mctx->sigma_hist = bctx->sigma_hist;
  for (i = 0; i <= bdata->k; ++i) {
    mctx->ytHpy[i] = bctx->ytHpy[i];
    mctx->ytHps[i] = bctx->ytHps[i];
    mctx->stHps[i] = bctx->stHps[i];
  }
  ierr = VecCopy(bctx->B, mctx->B);CHKERRQ(ierr);
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
    ierr = PetscFree3(ldb->ytHpy, ldb->ytHps, ldb->stHps);CHKERRQ(ierr);
    ierr = VecDestroy(&ldb->Bnew);CHKERRQ(ierr);
    ierr = VecDestroy(&ldb->B);CHKERRQ(ierr);
    ierr = VecDestroy(&ldb->y2);CHKERRQ(ierr);
    ierr = VecDestroy(&ldb->Bs2);CHKERRQ(ierr);
    ierr = VecDestroy(&ldb->w2);CHKERRQ(ierr);
    ierr = VecDestroy(&ldb->work);CHKERRQ(ierr);
    ldb->allocated = PETSC_FALSE;
  } else {
    ierr = VecSet(ldb->B, 1.0);CHKERRQ(ierr);
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
  lmvm->m = 0;
  ierr = MatAllocate_LMVM(B, X, F);CHKERRQ(ierr);
  if (!ldb->allocated) {
    ierr = PetscMalloc3(ldb->sigma_hist+1, &ldb->ytHpy, ldb->sigma_hist+1, &ldb->ytHps, ldb->sigma_hist+1, &ldb->stHps);CHKERRQ(ierr);
    ierr = VecDuplicate(X, &ldb->Bnew);CHKERRQ(ierr);
    ierr = VecDuplicate(X, &ldb->B);CHKERRQ(ierr);
    ierr = VecDuplicate(X, &ldb->y2);CHKERRQ(ierr);
    ierr = VecDuplicate(X, &ldb->Bs2);CHKERRQ(ierr);
    ierr = VecDuplicate(X, &ldb->w2);CHKERRQ(ierr);
    ierr = VecDuplicate(X, &ldb->work);CHKERRQ(ierr);
    ldb->allocated = PETSC_TRUE;
  }
  ierr = VecSet(ldb->B, 1.0);CHKERRQ(ierr);
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
    ierr = PetscFree3(ldb->ytHpy, ldb->ytHps, ldb->stHps);CHKERRQ(ierr);
    ierr = VecDestroy(&ldb->Bnew);CHKERRQ(ierr);
    ierr = VecDestroy(&ldb->B);CHKERRQ(ierr);
    ierr = VecDestroy(&ldb->y2);CHKERRQ(ierr);
    ierr = VecDestroy(&ldb->Bs2);CHKERRQ(ierr);
    ierr = VecDestroy(&ldb->w2);CHKERRQ(ierr);
    ierr = VecDestroy(&ldb->work);CHKERRQ(ierr);
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
  lmvm->m = 0;
  ierr = MatSetUp_LMVM(B);CHKERRQ(ierr);
  if (!ldb->allocated) {
    ierr = PetscMalloc3(ldb->sigma_hist+1, &ldb->ytHpy, ldb->sigma_hist+1, &ldb->ytHps, ldb->sigma_hist+1, &ldb->stHps);CHKERRQ(ierr);
    ierr = VecDuplicate(lmvm->Xprev, &ldb->Bnew);CHKERRQ(ierr);
    ierr = VecDuplicate(lmvm->Xprev, &ldb->B);CHKERRQ(ierr);
    ierr = VecDuplicate(lmvm->Xprev, &ldb->y2);CHKERRQ(ierr);
    ierr = VecDuplicate(lmvm->Xprev, &ldb->Bs2);CHKERRQ(ierr);
    ierr = VecDuplicate(lmvm->Xprev, &ldb->w2);CHKERRQ(ierr);
    ierr = VecDuplicate(lmvm->Xprev, &ldb->work);CHKERRQ(ierr);
    ldb->allocated = PETSC_TRUE;
  }
  ierr = VecSet(ldb->B, 1.0);CHKERRQ(ierr);
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
  ierr = PetscOptionsReal("-mat_lmvm_rho","(developer) convex ratio between old diagonal and new diagonal","",ldb->rho,&ldb->rho,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_lmvm_alpha","(developer) convex ratio between BFGS and DFP components in the diagonal re-scaling","",ldb->alpha,&ldb->alpha,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_lmvm_beta","(developer) exponential factor in the diagonal re-scaling","",ldb->beta,&ldb->beta,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mat_lmvm_sigma_hist","(developer) number of past updates to use when re-scaling the diagonal","",ldb->sigma_hist,&ldb->sigma_hist,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  if ((ldb->phi < 0.0) || (ldb->phi > 1.0)) SETERRQ(PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_OUTOFRANGE, "convex ratio for the update formula cannot be outside the range of [0, 1]");
  if ((ldb->alpha < 0.0) || (ldb->alpha > 1.0)) SETERRQ(PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_OUTOFRANGE, "convex ratio for the scaling cannot be outside the range of [0, 1]");
  if ((ldb->rho < 0.0) || (ldb->rho > 1.0)) SETERRQ(PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_OUTOFRANGE, "convex ratio for the diagonal combination cannot be outside the range of [0, 1]");
  if (ldb->beta < 0.0) SETERRQ(PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_OUTOFRANGE, "exponential factor cannot be negative");
  if (ldb->sigma_hist < 0) SETERRQ(PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_OUTOFRANGE, "diagonal scaling history length cannot be negative");
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
  lmvm->m = 0;
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
  ldb->alpha = 1.0;
  ldb->beta = 0.5;
  ldb->rho = 1.0;
  ldb->sigma_hist = 1;
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