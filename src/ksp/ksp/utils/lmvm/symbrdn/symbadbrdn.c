#include <../src/ksp/ksp/utils/lmvm/symbrdn/symbrdn.h> /*I "petscksp.h" I*/
#include <../src/ksp/ksp/utils/lmvm/diagbrdn/diagbrdn.h>

/*------------------------------------------------------------*/

static PetscErrorCode MatSolve_LMVMSymBadBrdn(Mat B, Vec F, Vec dX)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_SymBrdn       *lsb = (Mat_SymBrdn*)lmvm->ctx;
  PetscInt          i, j;
  PetscScalar       yjtqi, sjtyi, wtyi, ytx, stf, wtf, ytq;

  PetscFunctionBegin;
  /* Efficient shortcuts for pure BFGS and pure DFP configurations */
  if (lsb->phi == 0.0) {
    PetscCall(MatSolve_LMVMBFGS(B, F, dX));
    PetscFunctionReturn(0);
  }
  if (lsb->phi == 1.0) {
    PetscCall(MatSolve_LMVMDFP(B, F, dX));
    PetscFunctionReturn(0);
  }

  VecCheckSameSize(F, 2, dX, 3);
  VecCheckMatCompatible(B, dX, 3, F, 2);

  if (lsb->needQ) {
    /* Start the loop for (Q[k] = (B_k)^{-1} * Y[k]) */
    for (i = 0; i <= lmvm->k; ++i) {
      PetscCall(MatSymBrdnApplyJ0Inv(B, lmvm->Y[i], lsb->Q[i]));
      for (j = 0; j <= i-1; ++j) {
        /* Compute the necessary dot products */
        PetscCall(VecDotBegin(lmvm->Y[j], lsb->Q[i], &yjtqi));
        PetscCall(VecDotBegin(lmvm->S[j], lmvm->Y[i], &sjtyi));
        PetscCall(VecDotEnd(lmvm->Y[j], lsb->Q[i], &yjtqi));
        PetscCall(VecDotEnd(lmvm->S[j], lmvm->Y[i], &sjtyi));
        /* Compute the pure DFP component of the inverse application*/
        PetscCall(VecAXPBYPCZ(lsb->Q[i], -PetscRealPart(yjtqi)/lsb->ytq[j], PetscRealPart(sjtyi)/lsb->yts[j], 1.0, lsb->Q[j], lmvm->S[j]));
        /* Tack on the convexly scaled extras to the inverse application*/
        if (lsb->psi[j] > 0.0) {
          PetscCall(VecAXPBYPCZ(lsb->work, 1.0/lsb->yts[j], -1.0/lsb->ytq[j], 0.0, lmvm->S[j], lsb->Q[j]));
          PetscCall(VecDot(lsb->work, lmvm->Y[i], &wtyi));
          PetscCall(VecAXPY(lsb->Q[i], lsb->phi*lsb->ytq[j]*PetscRealPart(wtyi), lsb->work));
        }
      }
      PetscCall(VecDot(lmvm->Y[i], lsb->Q[i], &ytq));
      lsb->ytq[i] = PetscRealPart(ytq);
    }
    lsb->needQ = PETSC_FALSE;
  }

  /* Start the outer iterations for ((B^{-1}) * dX) */
  PetscCall(MatSymBrdnApplyJ0Inv(B, F, dX));
  for (i = 0; i <= lmvm->k; ++i) {
    /* Compute the necessary dot products -- store yTs and yTp for inner iterations later */
    PetscCall(VecDotBegin(lmvm->Y[i], dX, &ytx));
    PetscCall(VecDotBegin(lmvm->S[i], F, &stf));
    PetscCall(VecDotEnd(lmvm->Y[i], dX, &ytx));
    PetscCall(VecDotEnd(lmvm->S[i], F, &stf));
    /* Compute the pure DFP component */
    PetscCall(VecAXPBYPCZ(dX, -PetscRealPart(ytx)/lsb->ytq[i], PetscRealPart(stf)/lsb->yts[i], 1.0, lsb->Q[i], lmvm->S[i]));
    /* Tack on the convexly scaled extras */
    PetscCall(VecAXPBYPCZ(lsb->work, 1.0/lsb->yts[i], -1.0/lsb->ytq[i], 0.0, lmvm->S[i], lsb->Q[i]));
    PetscCall(VecDot(lsb->work, F, &wtf));
    PetscCall(VecAXPY(dX, lsb->phi*lsb->ytq[i]*PetscRealPart(wtf), lsb->work));
  }

  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatMult_LMVMSymBadBrdn(Mat B, Vec X, Vec Z)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_SymBrdn       *lsb = (Mat_SymBrdn*)lmvm->ctx;
  PetscInt          i, j;
  PetscReal         numer;
  PetscScalar       sjtpi, sjtyi, yjtsi, yjtqi, wtsi, wtyi, stz, ytx, ytq, wtx, stp;

  PetscFunctionBegin;
  /* Efficient shortcuts for pure BFGS and pure DFP configurations */
  if (lsb->phi == 0.0) {
    PetscCall(MatMult_LMVMBFGS(B, X, Z));
    PetscFunctionReturn(0);
  }
  if (lsb->phi == 1.0) {
    PetscCall(MatMult_LMVMDFP(B, X, Z));
    PetscFunctionReturn(0);
  }

  VecCheckSameSize(X, 2, Z, 3);
  VecCheckMatCompatible(B, X, 2, Z, 3);

  if (lsb->needQ) {
    /* Start the loop for (Q[k] = (B_k)^{-1} * Y[k]) */
    for (i = 0; i <= lmvm->k; ++i) {
      PetscCall(MatSymBrdnApplyJ0Inv(B, lmvm->Y[i], lsb->Q[i]));
      for (j = 0; j <= i-1; ++j) {
        /* Compute the necessary dot products */
        PetscCall(VecDotBegin(lmvm->Y[j], lsb->Q[i], &yjtqi));
        PetscCall(VecDotBegin(lmvm->S[j], lmvm->Y[i], &sjtyi));
        PetscCall(VecDotEnd(lmvm->Y[j], lsb->Q[i], &yjtqi));
        PetscCall(VecDotEnd(lmvm->S[j], lmvm->Y[i], &sjtyi));
        /* Compute the pure DFP component of the inverse application*/
        PetscCall(VecAXPBYPCZ(lsb->Q[i], -PetscRealPart(yjtqi)/lsb->ytq[j], PetscRealPart(sjtyi)/lsb->yts[j], 1.0, lsb->Q[j], lmvm->S[j]));
        /* Tack on the convexly scaled extras to the inverse application*/
        if (lsb->psi[j] > 0.0) {
          PetscCall(VecAXPBYPCZ(lsb->work, 1.0/lsb->yts[j], -1.0/lsb->ytq[j], 0.0, lmvm->S[j], lsb->Q[j]));
          PetscCall(VecDot(lsb->work, lmvm->Y[i], &wtyi));
          PetscCall(VecAXPY(lsb->Q[i], lsb->phi*lsb->ytq[j]*PetscRealPart(wtyi), lsb->work));
        }
      }
      PetscCall(VecDot(lmvm->Y[i], lsb->Q[i], &ytq));
      lsb->ytq[i] = PetscRealPart(ytq);
    }
    lsb->needQ = PETSC_FALSE;
  }
  if (lsb->needP) {
    /* Start the loop for (P[k] = (B_k) * S[k]) */
    for (i = 0; i <= lmvm->k; ++i) {
      PetscCall(MatSymBrdnApplyJ0Fwd(B, lmvm->S[i], lsb->P[i]));
      for (j = 0; j <= i-1; ++j) {
        /* Compute the necessary dot products */
        PetscCall(VecDotBegin(lmvm->S[j], lsb->P[i], &sjtpi));
        PetscCall(VecDotBegin(lmvm->Y[j], lmvm->S[i], &yjtsi));
        PetscCall(VecDotEnd(lmvm->S[j], lsb->P[i], &sjtpi));
        PetscCall(VecDotEnd(lmvm->Y[j], lmvm->S[i], &yjtsi));
        /* Compute the pure BFGS component of the forward product */
        PetscCall(VecAXPBYPCZ(lsb->P[i], -PetscRealPart(sjtpi)/lsb->stp[j], PetscRealPart(yjtsi)/lsb->yts[j], 1.0, lsb->P[j], lmvm->Y[j]));
        /* Tack on the convexly scaled extras to the forward product */
        if (lsb->phi > 0.0) {
          PetscCall(VecAXPBYPCZ(lsb->work, 1.0/lsb->yts[j], -1.0/lsb->stp[j], 0.0, lmvm->Y[j], lsb->P[j]));
          PetscCall(VecDot(lsb->work, lmvm->S[i], &wtsi));
          PetscCall(VecAXPY(lsb->P[i], lsb->psi[j]*lsb->stp[j]*PetscRealPart(wtsi), lsb->work));
        }
      }
      PetscCall(VecDot(lmvm->S[i], lsb->P[i], &stp));
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
  PetscCall(MatSymBrdnApplyJ0Fwd(B, X, Z));
  for (i = 0; i <= lmvm->k; ++i) {
    /* Compute the necessary dot products */
    PetscCall(VecDotBegin(lmvm->S[i], Z, &stz));
    PetscCall(VecDotBegin(lmvm->Y[i], X, &ytx));
    PetscCall(VecDotEnd(lmvm->S[i], Z, &stz));
    PetscCall(VecDotEnd(lmvm->Y[i], X, &ytx));
    /* Compute the pure BFGS component */
    PetscCall(VecAXPBYPCZ(Z, -PetscRealPart(stz)/lsb->stp[i], PetscRealPart(ytx)/lsb->yts[i], 1.0, lsb->P[i], lmvm->Y[i]));
    /* Tack on the convexly scaled extras */
    PetscCall(VecAXPBYPCZ(lsb->work, 1.0/lsb->yts[i], -1.0/lsb->stp[i], 0.0, lmvm->Y[i], lsb->P[i]));
    PetscCall(VecDot(lsb->work, X, &wtx));
    PetscCall(VecAXPY(Z, lsb->psi[i]*lsb->stp[i]*PetscRealPart(wtx), lsb->work));
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

  PetscFunctionBegin;
  PetscCall(MatSetFromOptions_LMVMSymBrdn(PetscOptionsObject, B));
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

  PetscFunctionBegin;
  PetscCall(MatCreate_LMVMSymBrdn(B));
  PetscCall(PetscObjectChangeTypeName((PetscObject)B, MATLMVMSYMBADBROYDEN));
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

.seealso: `MatCreate()`, `MATLMVM`, `MATLMVMSYMBROYDEN`, `MatCreateLMVMDFP()`, `MatCreateLMVMSR1()`,
          `MatCreateLMVMBFGS()`, `MatCreateLMVMBrdn()`, `MatCreateLMVMBadBrdn()`
@*/
PetscErrorCode MatCreateLMVMSymBadBroyden(MPI_Comm comm, PetscInt n, PetscInt N, Mat *B)
{
  PetscFunctionBegin;
  PetscCall(MatCreate(comm, B));
  PetscCall(MatSetSizes(*B, n, n, N, N));
  PetscCall(MatSetType(*B, MATLMVMSYMBADBROYDEN));
  PetscCall(MatSetUp(*B));
  PetscFunctionReturn(0);
}
