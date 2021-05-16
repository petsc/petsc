#include <../src/ksp/ksp/utils/lmvm/symbrdn/symbrdn.h> /*I "petscksp.h" I*/
#include <../src/ksp/ksp/utils/lmvm/diagbrdn/diagbrdn.h>

/*------------------------------------------------------------*/

static PetscErrorCode MatSolve_LMVMSymBadBrdn(Mat B, Vec F, Vec dX)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_SymBrdn       *lsb = (Mat_SymBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  PetscInt          i, j;
  PetscScalar       yjtqi, sjtyi, wtyi, ytx, stf, wtf, ytq;

  PetscFunctionBegin;
  /* Efficient shortcuts for pure BFGS and pure DFP configurations */
  if (lsb->phi == 0.0) {
    ierr = MatSolve_LMVMBFGS(B, F, dX);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  if (lsb->phi == 1.0) {
    ierr = MatSolve_LMVMDFP(B, F, dX);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  VecCheckSameSize(F, 2, dX, 3);
  VecCheckMatCompatible(B, dX, 3, F, 2);

  if (lsb->needQ) {
    /* Start the loop for (Q[k] = (B_k)^{-1} * Y[k]) */
    for (i = 0; i <= lmvm->k; ++i) {
      ierr = MatSymBrdnApplyJ0Inv(B, lmvm->Y[i], lsb->Q[i]);CHKERRQ(ierr);
      for (j = 0; j <= i-1; ++j) {
        /* Compute the necessary dot products */
        ierr = VecDotBegin(lmvm->Y[j], lsb->Q[i], &yjtqi);CHKERRQ(ierr);
        ierr = VecDotBegin(lmvm->S[j], lmvm->Y[i], &sjtyi);CHKERRQ(ierr);
        ierr = VecDotEnd(lmvm->Y[j], lsb->Q[i], &yjtqi);CHKERRQ(ierr);
        ierr = VecDotEnd(lmvm->S[j], lmvm->Y[i], &sjtyi);CHKERRQ(ierr);
        /* Compute the pure DFP component of the inverse application*/
        ierr = VecAXPBYPCZ(lsb->Q[i], -PetscRealPart(yjtqi)/lsb->ytq[j], PetscRealPart(sjtyi)/lsb->yts[j], 1.0, lsb->Q[j], lmvm->S[j]);CHKERRQ(ierr);
        /* Tack on the convexly scaled extras to the inverse application*/
        if (lsb->psi[j] > 0.0) {
          ierr = VecAXPBYPCZ(lsb->work, 1.0/lsb->yts[j], -1.0/lsb->ytq[j], 0.0, lmvm->S[j], lsb->Q[j]);CHKERRQ(ierr);
          ierr = VecDot(lsb->work, lmvm->Y[i], &wtyi);CHKERRQ(ierr);
          ierr = VecAXPY(lsb->Q[i], lsb->phi*lsb->ytq[j]*PetscRealPart(wtyi), lsb->work);CHKERRQ(ierr);
        }
      }
      ierr = VecDot(lmvm->Y[i], lsb->Q[i], &ytq);CHKERRQ(ierr);
      lsb->ytq[i] = PetscRealPart(ytq);
    }
    lsb->needQ = PETSC_FALSE;
  }

  /* Start the outer iterations for ((B^{-1}) * dX) */
  ierr = MatSymBrdnApplyJ0Inv(B, F, dX);CHKERRQ(ierr);
  for (i = 0; i <= lmvm->k; ++i) {
    /* Compute the necessary dot products -- store yTs and yTp for inner iterations later */
    ierr = VecDotBegin(lmvm->Y[i], dX, &ytx);CHKERRQ(ierr);
    ierr = VecDotBegin(lmvm->S[i], F, &stf);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->Y[i], dX, &ytx);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->S[i], F, &stf);CHKERRQ(ierr);
    /* Compute the pure DFP component */
    ierr = VecAXPBYPCZ(dX, -PetscRealPart(ytx)/lsb->ytq[i], PetscRealPart(stf)/lsb->yts[i], 1.0, lsb->Q[i], lmvm->S[i]);CHKERRQ(ierr);
    /* Tack on the convexly scaled extras */
    ierr = VecAXPBYPCZ(lsb->work, 1.0/lsb->yts[i], -1.0/lsb->ytq[i], 0.0, lmvm->S[i], lsb->Q[i]);CHKERRQ(ierr);
    ierr = VecDot(lsb->work, F, &wtf);CHKERRQ(ierr);
    ierr = VecAXPY(dX, lsb->phi*lsb->ytq[i]*PetscRealPart(wtf), lsb->work);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatMult_LMVMSymBadBrdn(Mat B, Vec X, Vec Z)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_SymBrdn       *lsb = (Mat_SymBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  PetscInt          i, j;
  PetscReal         numer;
  PetscScalar       sjtpi, sjtyi, yjtsi, yjtqi, wtsi, wtyi, stz, ytx, ytq, wtx, stp;

  PetscFunctionBegin;
  /* Efficient shortcuts for pure BFGS and pure DFP configurations */
  if (lsb->phi == 0.0) {
    ierr = MatMult_LMVMBFGS(B, X, Z);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  if (lsb->phi == 1.0) {
    ierr = MatMult_LMVMDFP(B, X, Z);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  VecCheckSameSize(X, 2, Z, 3);
  VecCheckMatCompatible(B, X, 2, Z, 3);

  if (lsb->needQ) {
    /* Start the loop for (Q[k] = (B_k)^{-1} * Y[k]) */
    for (i = 0; i <= lmvm->k; ++i) {
      ierr = MatSymBrdnApplyJ0Inv(B, lmvm->Y[i], lsb->Q[i]);CHKERRQ(ierr);
      for (j = 0; j <= i-1; ++j) {
        /* Compute the necessary dot products */
        ierr = VecDotBegin(lmvm->Y[j], lsb->Q[i], &yjtqi);CHKERRQ(ierr);
        ierr = VecDotBegin(lmvm->S[j], lmvm->Y[i], &sjtyi);CHKERRQ(ierr);
        ierr = VecDotEnd(lmvm->Y[j], lsb->Q[i], &yjtqi);CHKERRQ(ierr);
        ierr = VecDotEnd(lmvm->S[j], lmvm->Y[i], &sjtyi);CHKERRQ(ierr);
        /* Compute the pure DFP component of the inverse application*/
        ierr = VecAXPBYPCZ(lsb->Q[i], -PetscRealPart(yjtqi)/lsb->ytq[j], PetscRealPart(sjtyi)/lsb->yts[j], 1.0, lsb->Q[j], lmvm->S[j]);CHKERRQ(ierr);
        /* Tack on the convexly scaled extras to the inverse application*/
        if (lsb->psi[j] > 0.0) {
          ierr = VecAXPBYPCZ(lsb->work, 1.0/lsb->yts[j], -1.0/lsb->ytq[j], 0.0, lmvm->S[j], lsb->Q[j]);CHKERRQ(ierr);
          ierr = VecDot(lsb->work, lmvm->Y[i], &wtyi);CHKERRQ(ierr);
          ierr = VecAXPY(lsb->Q[i], lsb->phi*lsb->ytq[j]*PetscRealPart(wtyi), lsb->work);CHKERRQ(ierr);
        }
      }
      ierr = VecDot(lmvm->Y[i], lsb->Q[i], &ytq);CHKERRQ(ierr);
      lsb->ytq[i] = PetscRealPart(ytq);
    }
    lsb->needQ = PETSC_FALSE;
  }
  if (lsb->needP) {
    /* Start the loop for (P[k] = (B_k) * S[k]) */
    for (i = 0; i <= lmvm->k; ++i) {
      ierr = MatSymBrdnApplyJ0Fwd(B, lmvm->S[i], lsb->P[i]);CHKERRQ(ierr);
      for (j = 0; j <= i-1; ++j) {
        /* Compute the necessary dot products */
        ierr = VecDotBegin(lmvm->S[j], lsb->P[i], &sjtpi);CHKERRQ(ierr);
        ierr = VecDotBegin(lmvm->Y[j], lmvm->S[i], &yjtsi);CHKERRQ(ierr);
        ierr = VecDotEnd(lmvm->S[j], lsb->P[i], &sjtpi);CHKERRQ(ierr);
        ierr = VecDotEnd(lmvm->Y[j], lmvm->S[i], &yjtsi);CHKERRQ(ierr);
        /* Compute the pure BFGS component of the forward product */
        ierr = VecAXPBYPCZ(lsb->P[i], -PetscRealPart(sjtpi)/lsb->stp[j], PetscRealPart(yjtsi)/lsb->yts[j], 1.0, lsb->P[j], lmvm->Y[j]);CHKERRQ(ierr);
        /* Tack on the convexly scaled extras to the forward product */
        if (lsb->phi > 0.0) {
          ierr = VecAXPBYPCZ(lsb->work, 1.0/lsb->yts[j], -1.0/lsb->stp[j], 0.0, lmvm->Y[j], lsb->P[j]);CHKERRQ(ierr);
          ierr = VecDot(lsb->work, lmvm->S[i], &wtsi);CHKERRQ(ierr);
          ierr = VecAXPY(lsb->P[i], lsb->psi[j]*lsb->stp[j]*PetscRealPart(wtsi), lsb->work);CHKERRQ(ierr);
        }
      }
      ierr = VecDot(lmvm->S[i], lsb->P[i], &stp);CHKERRQ(ierr);
      lsb->stp[i] = PetscRealPart(stp);
      if (lsb->phi == 1.0) {
        lsb->psi[i] = 0.0;
      } else if (lsb->phi == 0.0) {
        lsb->psi[i] = 1.0;
      } else {
        numer = (1.0 - lsb->phi)*lsb->yts[i]*lsb->yts[i];
        lsb->psi[i] = numer / (numer + (lsb->phi*lsb->ytq[i]*lsb->stp[i]));
      }
    }
    lsb->needP = PETSC_FALSE;
  }

  /* Start the outer iterations for (B * X) */
  ierr = MatSymBrdnApplyJ0Fwd(B, X, Z);CHKERRQ(ierr);
  for (i = 0; i <= lmvm->k; ++i) {
    /* Compute the necessary dot products */
    ierr = VecDotBegin(lmvm->S[i], Z, &stz);CHKERRQ(ierr);
    ierr = VecDotBegin(lmvm->Y[i], X, &ytx);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->S[i], Z, &stz);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->Y[i], X, &ytx);CHKERRQ(ierr);
    /* Compute the pure BFGS component */
    ierr = VecAXPBYPCZ(Z, -PetscRealPart(stz)/lsb->stp[i], PetscRealPart(ytx)/lsb->yts[i], 1.0, lsb->P[i], lmvm->Y[i]);CHKERRQ(ierr);
    /* Tack on the convexly scaled extras */
    ierr = VecAXPBYPCZ(lsb->work, 1.0/lsb->yts[i], -1.0/lsb->stp[i], 0.0, lmvm->Y[i], lsb->P[i]);CHKERRQ(ierr);
    ierr = VecDot(lsb->work, X, &wtx);CHKERRQ(ierr);
    ierr = VecAXPY(Z, lsb->psi[i]*lsb->stp[i]*PetscRealPart(wtx), lsb->work);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatSetFromOptions_LMVMSymBadBrdn(PetscOptionItems *PetscOptionsObject, Mat B)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_SymBrdn       *lsb = (Mat_SymBrdn*)lmvm->ctx;
  Mat_LMVM          *dbase;
  Mat_DiagBrdn      *dctx;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatSetFromOptions_LMVMSymBrdn(PetscOptionsObject, B);CHKERRQ(ierr);
  if (lsb->scale_type == MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL) {
    dbase = (Mat_LMVM*)lsb->D->data;
    dctx = (Mat_DiagBrdn*)dbase->ctx;
    dctx->forward = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode MatCreate_LMVMSymBadBrdn(Mat B)
{
  Mat_LMVM          *lmvm;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatCreate_LMVMSymBrdn(B);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B, MATLMVMSYMBADBROYDEN);CHKERRQ(ierr);
  B->ops->setfromoptions = MatSetFromOptions_LMVMSymBadBrdn;
  B->ops->solve = MatSolve_LMVMSymBadBrdn;

  lmvm = (Mat_LMVM*)B->data;
  lmvm->ops->mult = MatMult_LMVMSymBadBrdn;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/*@
   MatCreateLMVMSymBadBroyden - Creates a limited-memory Symmetric "Bad" Broyden-type matrix used
   for approximating Jacobians. L-SymBadBrdn is a convex combination of L-DFP and
   L-BFGS such that `^{-1} = (1 - phi)*BFGS^{-1} + phi*DFP^{-1}. The combination factor
   phi is restricted to the range [0, 1], where the L-SymBadBrdn matrix is guaranteed
   to be symmetric positive-definite. Note that this combination is on the inverses and not
   on the forwards. For forward convex combinations, use the L-SymBrdn matrix.

   The provided local and global sizes must match the solution and function vectors
   used with MatLMVMUpdate() and MatSolve(). The resulting L-SymBrdn matrix will have
   storage vectors allocated with VecCreateSeq() in serial and VecCreateMPI() in
   parallel. To use the L-SymBrdn matrix with other vector types, the matrix must be
   created using MatCreate() and MatSetType(), followed by MatLMVMAllocate().
   This ensures that the internal storage and work vectors are duplicated from the
   correct type of vector.

   Collective

   Input Parameters:
+  comm - MPI communicator, set to PETSC_COMM_SELF
.  n - number of local rows for storage vectors
-  N - global size of the storage vectors

   Output Parameter:
.  B - the matrix

   It is recommended that one use the MatCreate(), MatSetType() and/or MatSetFromOptions()
   paradigm instead of this routine directly.

   Options Database Keys:
+   -mat_lmvm_num_vecs - maximum number of correction vectors (i.e.: updates) stored
.   -mat_lmvm_phi - (developer) convex ratio between BFGS and DFP components of the update
.   -mat_lmvm_scale_type - (developer) type of scaling applied to J0 (none, scalar, diagonal)
.   -mat_lmvm_theta - (developer) convex ratio between BFGS and DFP components of the diagonal J0 scaling
.   -mat_lmvm_rho - (developer) update limiter for the J0 scaling
.   -mat_lmvm_alpha - (developer) coefficient factor for the quadratic subproblem in J0 scaling
.   -mat_lmvm_beta - (developer) exponential factor for the diagonal J0 scaling
-   -mat_lmvm_sigma_hist - (developer) number of past updates to use in J0 scaling

   Level: intermediate

.seealso: MatCreate(), MATLMVM, MATLMVMSYMBROYDEN, MatCreateLMVMDFP(), MatCreateLMVMSR1(),
          MatCreateLMVMBFGS(), MatCreateLMVMBrdn(), MatCreateLMVMBadBrdn()
@*/
PetscErrorCode MatCreateLMVMSymBadBroyden(MPI_Comm comm, PetscInt n, PetscInt N, Mat *B)
{
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatCreate(comm, B);CHKERRQ(ierr);
  ierr = MatSetSizes(*B, n, n, N, N);CHKERRQ(ierr);
  ierr = MatSetType(*B, MATLMVMSYMBADBROYDEN);CHKERRQ(ierr);
  ierr = MatSetUp(*B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
