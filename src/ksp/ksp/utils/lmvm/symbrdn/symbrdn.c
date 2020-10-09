#include <../src/ksp/ksp/utils/lmvm/symbrdn/symbrdn.h> /*I "petscksp.h" I*/
#include <../src/ksp/ksp/utils/lmvm/diagbrdn/diagbrdn.h>

const char *const MatLMVMSymBroydenScaleTypes[] = {"NONE","SCALAR","DIAGONAL","USER","MatLMVMSymBrdnScaleType","MAT_LMVM_SYMBROYDEN_SCALING_",NULL};

/*------------------------------------------------------------*/

/*
  The solution method below is the matrix-free implementation of
  Equation 8.6a in Dennis and More "Quasi-Newton Methods, Motivation
  and Theory" (https://epubs.siam.org/doi/abs/10.1137/1019005).

  Q[i] = (B_i)^{-1}*S[i] terms are computed ahead of time whenever
  the matrix is updated with a new (S[i], Y[i]) pair. This allows
  repeated calls of MatSolve without incurring redundant computation.

  dX <- J0^{-1} * F

  for i=0,1,2,...,k
    # Q[i] = (B_i)^T{-1} Y[i]

    rho = 1.0 / (Y[i]^T S[i])
    alpha = rho * (S[i]^T F)
    zeta = 1.0 / (Y[i]^T Q[i])
    gamma = zeta * (Y[i]^T dX)

    dX <- dX - (gamma * Q[i]) + (alpha * Y[i])
    W <- (rho * S[i]) - (zeta * Q[i])
    dX <- dX + (psi[i] * (Y[i]^T Q[i]) * (W^T F) * W)
  end
*/
static PetscErrorCode MatSolve_LMVMSymBrdn(Mat B, Vec F, Vec dX)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_SymBrdn       *lsb = (Mat_SymBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  PetscInt          i, j;
  PetscReal         numer;
  PetscScalar       sjtpi, yjtsi, wtsi, yjtqi, sjtyi, wtyi, ytx, stf, wtf, stp, ytq;

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
          ierr = VecAXPY(lsb->P[i], lsb->phi*lsb->stp[j]*PetscRealPart(wtsi), lsb->work);CHKERRQ(ierr);
        }
      }
      ierr = VecDot(lmvm->S[i], lsb->P[i], &stp);CHKERRQ(ierr);
      lsb->stp[i] = PetscRealPart(stp);
    }
    lsb->needP = PETSC_FALSE;
  }
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
          ierr = VecAXPY(lsb->Q[i], lsb->psi[j]*lsb->ytq[j]*PetscRealPart(wtyi), lsb->work);CHKERRQ(ierr);
        }
      }
      ierr = VecDot(lmvm->Y[i], lsb->Q[i], &ytq);CHKERRQ(ierr);
      lsb->ytq[i] = PetscRealPart(ytq);
      if (lsb->phi == 1.0) {
        lsb->psi[i] = 0.0;
      } else if (lsb->phi == 0.0) {
        lsb->psi[i] = 1.0;
      } else {
        numer = (1.0 - lsb->phi)*lsb->yts[i]*lsb->yts[i];
        lsb->psi[i] = numer / (numer + (lsb->phi*lsb->ytq[i]*lsb->stp[i]));
      }
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
    ierr = VecAXPY(dX, lsb->psi[i]*lsb->ytq[i]*PetscRealPart(wtf), lsb->work);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/*
  The forward-product below is the matrix-free implementation of
  Equation 16 in Dennis and Wolkowicz "Sizing and Least Change Secant
  Methods" (http://www.caam.rice.edu/caam/trs/90/TR90-05.pdf).

  P[i] = (B_i)*S[i] terms are computed ahead of time whenever
  the matrix is updated with a new (S[i], Y[i]) pair. This allows
  repeated calls of MatMult inside KSP solvers without unnecessarily
  recomputing P[i] terms in expensive nested-loops.

  Z <- J0 * X

  for i=0,1,2,...,k
    # P[i] = (B_k) * S[i]

    rho = 1.0 / (Y[i]^T S[i])
    alpha = rho * (Y[i]^T F)
    zeta = 1.0 / (S[i]^T P[i])
    gamma = zeta * (S[i]^T dX)

    dX <- dX - (gamma * P[i]) + (alpha * S[i])
    W <- (rho * Y[i]) - (zeta * P[i])
    dX <- dX + (phi * (S[i]^T P[i]) * (W^T F) * W)
  end
*/
static PetscErrorCode MatMult_LMVMSymBrdn(Mat B, Vec X, Vec Z)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_SymBrdn       *lsb = (Mat_SymBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  PetscInt          i, j;
  PetscScalar         sjtpi, yjtsi, wtsi, stz, ytx, wtx, stp;


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
          ierr = VecAXPY(lsb->P[i], lsb->phi*lsb->stp[j]*PetscRealPart(wtsi), lsb->work);CHKERRQ(ierr);
        }
      }
      ierr = VecDot(lmvm->S[i], lsb->P[i], &stp);CHKERRQ(ierr);
      lsb->stp[i] = PetscRealPart(stp);
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
    ierr = VecAXPY(Z, lsb->phi*lsb->stp[i]*PetscRealPart(wtx), lsb->work);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatUpdate_LMVMSymBrdn(Mat B, Vec X, Vec F)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_SymBrdn       *lsb = (Mat_SymBrdn*)lmvm->ctx;
  Mat_LMVM          *dbase;
  Mat_DiagBrdn      *dctx;
  PetscErrorCode    ierr;
  PetscInt          old_k, i;
  PetscReal         curvtol;
  PetscScalar       curvature, ytytmp, ststmp;

  PetscFunctionBegin;
  if (!lmvm->m) PetscFunctionReturn(0);
  if (lmvm->prev_set) {
    /* Compute the new (S = X - Xprev) and (Y = F - Fprev) vectors */
    ierr = VecAYPX(lmvm->Xprev, -1.0, X);CHKERRQ(ierr);
    ierr = VecAYPX(lmvm->Fprev, -1.0, F);CHKERRQ(ierr);
    /* Test if the updates can be accepted */
    ierr = VecDotBegin(lmvm->Xprev, lmvm->Fprev, &curvature);CHKERRQ(ierr);
    ierr = VecDotBegin(lmvm->Xprev, lmvm->Xprev, &ststmp);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->Xprev, lmvm->Fprev, &curvature);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->Xprev, lmvm->Xprev, &ststmp);CHKERRQ(ierr);
    if (PetscRealPart(ststmp) < lmvm->eps) {
      curvtol = 0.0;
    } else {
      curvtol = lmvm->eps * PetscRealPart(ststmp);
    }
    if (PetscRealPart(curvature) > curvtol) {
      /* Update is good, accept it */
      lsb->watchdog = 0;
      lsb->needP = lsb->needQ = PETSC_TRUE;
      old_k = lmvm->k;
      ierr = MatUpdateKernel_LMVM(B, lmvm->Xprev, lmvm->Fprev);CHKERRQ(ierr);
      /* If we hit the memory limit, shift the yts, yty and sts arrays */
      if (old_k == lmvm->k) {
        for (i = 0; i <= lmvm->k-1; ++i) {
          lsb->yts[i] = lsb->yts[i+1];
          lsb->yty[i] = lsb->yty[i+1];
          lsb->sts[i] = lsb->sts[i+1];
        }
      }
      /* Update history of useful scalars */
      ierr = VecDot(lmvm->Y[lmvm->k], lmvm->Y[lmvm->k], &ytytmp);CHKERRQ(ierr);
      lsb->yts[lmvm->k] = PetscRealPart(curvature);
      lsb->yty[lmvm->k] = PetscRealPart(ytytmp);
      lsb->sts[lmvm->k] = PetscRealPart(ststmp);
      /* Compute the scalar scale if necessary */
      if (lsb->scale_type == MAT_LMVM_SYMBROYDEN_SCALE_SCALAR) {
        ierr = MatSymBrdnComputeJ0Scalar(B);CHKERRQ(ierr);
      }
    } else {
      /* Update is bad, skip it */
      ++lmvm->nrejects;
      ++lsb->watchdog;
    }
  } else {
    switch (lsb->scale_type) {
    case MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL:
      dbase = (Mat_LMVM*)lsb->D->data;
      dctx = (Mat_DiagBrdn*)dbase->ctx;
      ierr = VecSet(dctx->invD, lsb->delta);CHKERRQ(ierr);
      break;
    case MAT_LMVM_SYMBROYDEN_SCALE_SCALAR:
      lsb->sigma = lsb->delta;
      break;
    case MAT_LMVM_SYMBROYDEN_SCALE_NONE:
      lsb->sigma = 1.0;
      break;
    default:
      break;
    }
  }

  /* Update the scaling */
  if (lsb->scale_type == MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL) {
    ierr = MatLMVMUpdate(lsb->D, X, F);CHKERRQ(ierr);
  }

  if (lsb->watchdog > lsb->max_seq_rejects) {
    ierr = MatLMVMReset(B, PETSC_FALSE);CHKERRQ(ierr);
    if (lsb->scale_type == MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL) {
      ierr = MatLMVMReset(lsb->D, PETSC_FALSE);CHKERRQ(ierr);
    }
  }

  /* Save the solution and function to be used in the next update */
  ierr = VecCopy(X, lmvm->Xprev);CHKERRQ(ierr);
  ierr = VecCopy(F, lmvm->Fprev);CHKERRQ(ierr);
  lmvm->prev_set = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatCopy_LMVMSymBrdn(Mat B, Mat M, MatStructure str)
{
  Mat_LMVM          *bdata = (Mat_LMVM*)B->data;
  Mat_SymBrdn       *blsb = (Mat_SymBrdn*)bdata->ctx;
  Mat_LMVM          *mdata = (Mat_LMVM*)M->data;
  Mat_SymBrdn       *mlsb = (Mat_SymBrdn*)mdata->ctx;
  PetscErrorCode    ierr;
  PetscInt          i;

  PetscFunctionBegin;
  mlsb->phi = blsb->phi;
  mlsb->needP = blsb->needP;
  mlsb->needQ = blsb->needQ;
  for (i=0; i<=bdata->k; ++i) {
    mlsb->stp[i] = blsb->stp[i];
    mlsb->ytq[i] = blsb->ytq[i];
    mlsb->yts[i] = blsb->yts[i];
    mlsb->psi[i] = blsb->psi[i];
    ierr = VecCopy(blsb->P[i], mlsb->P[i]);CHKERRQ(ierr);
    ierr = VecCopy(blsb->Q[i], mlsb->Q[i]);CHKERRQ(ierr);
  }
  mlsb->scale_type      = blsb->scale_type;
  mlsb->alpha           = blsb->alpha;
  mlsb->beta            = blsb->beta;
  mlsb->rho             = blsb->rho;
  mlsb->delta           = blsb->delta;
  mlsb->sigma_hist      = blsb->sigma_hist;
  mlsb->watchdog        = blsb->watchdog;
  mlsb->max_seq_rejects = blsb->max_seq_rejects;
  switch (blsb->scale_type) {
  case MAT_LMVM_SYMBROYDEN_SCALE_SCALAR:
    mlsb->sigma = blsb->sigma;
    break;
  case MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL:
    ierr = MatCopy(blsb->D, mlsb->D, SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    break;
  case MAT_LMVM_SYMBROYDEN_SCALE_NONE:
    mlsb->sigma = 1.0;
    break;
  default:
    break;
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatReset_LMVMSymBrdn(Mat B, PetscBool destructive)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_SymBrdn       *lsb = (Mat_SymBrdn*)lmvm->ctx;
  Mat_LMVM          *dbase;
  Mat_DiagBrdn      *dctx;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  lsb->watchdog = 0;
  lsb->needP = lsb->needQ = PETSC_TRUE;
  if (lsb->allocated) {
    if (destructive) {
      ierr = VecDestroy(&lsb->work);CHKERRQ(ierr);
      ierr = PetscFree5(lsb->stp, lsb->ytq, lsb->yts, lsb->yty, lsb->sts);CHKERRQ(ierr);
      ierr = PetscFree(lsb->psi);CHKERRQ(ierr);
      ierr = VecDestroyVecs(lmvm->m, &lsb->P);CHKERRQ(ierr);
      ierr = VecDestroyVecs(lmvm->m, &lsb->Q);CHKERRQ(ierr);
      switch (lsb->scale_type) {
      case MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL:
        ierr = MatLMVMReset(lsb->D, PETSC_TRUE);CHKERRQ(ierr);
        break;
      default:
        break;
      }
      lsb->allocated = PETSC_FALSE;
    } else {
      ierr = PetscMemzero(lsb->psi, lmvm->m);CHKERRQ(ierr);
      switch (lsb->scale_type) {
      case MAT_LMVM_SYMBROYDEN_SCALE_SCALAR:
        lsb->sigma = lsb->delta;
        break;
      case MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL:
        ierr = MatLMVMReset(lsb->D, PETSC_FALSE);CHKERRQ(ierr);
        dbase = (Mat_LMVM*)lsb->D->data;
        dctx = (Mat_DiagBrdn*)dbase->ctx;
        ierr = VecSet(dctx->invD, lsb->delta);CHKERRQ(ierr);
        break;
      case MAT_LMVM_SYMBROYDEN_SCALE_NONE:
        lsb->sigma = 1.0;
        break;
      default:
        break;
      }
    }
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
  if (!lsb->allocated) {
    ierr = VecDuplicate(X, &lsb->work);CHKERRQ(ierr);
    if (lmvm->m > 0) {
      ierr = PetscMalloc5(lmvm->m,&lsb->stp,lmvm->m,&lsb->ytq,lmvm->m,&lsb->yts,lmvm->m,&lsb->yty,lmvm->m,&lsb->sts);CHKERRQ(ierr);
      ierr = PetscCalloc1(lmvm->m,&lsb->psi);CHKERRQ(ierr);
      ierr = VecDuplicateVecs(X, lmvm->m, &lsb->P);CHKERRQ(ierr);
      ierr = VecDuplicateVecs(X, lmvm->m, &lsb->Q);CHKERRQ(ierr);
    }
    switch (lsb->scale_type) {
    case MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL:
      ierr = MatLMVMAllocate(lsb->D, X, F);CHKERRQ(ierr);
      break;
    default:
      break;
    }
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
  if (lsb->allocated) {
    ierr = VecDestroy(&lsb->work);CHKERRQ(ierr);
    ierr = PetscFree5(lsb->stp, lsb->ytq, lsb->yts, lsb->yty, lsb->sts);CHKERRQ(ierr);
    ierr = PetscFree(lsb->psi);CHKERRQ(ierr);
    ierr = VecDestroyVecs(lmvm->m, &lsb->P);CHKERRQ(ierr);
    ierr = VecDestroyVecs(lmvm->m, &lsb->Q);CHKERRQ(ierr);
    lsb->allocated = PETSC_FALSE;
  }
  ierr = MatDestroy(&lsb->D);CHKERRQ(ierr);
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
  PetscInt          n, N;

  PetscFunctionBegin;
  ierr = MatSetUp_LMVM(B);CHKERRQ(ierr);
  if (!lsb->allocated) {
    ierr = VecDuplicate(lmvm->Xprev, &lsb->work);CHKERRQ(ierr);
    if (lmvm->m > 0) {
      ierr = PetscMalloc5(lmvm->m,&lsb->stp,lmvm->m,&lsb->ytq,lmvm->m,&lsb->yts,lmvm->m,&lsb->yty,lmvm->m,&lsb->sts);CHKERRQ(ierr);
      ierr = PetscCalloc1(lmvm->m,&lsb->psi);CHKERRQ(ierr);
      ierr = VecDuplicateVecs(lmvm->Xprev, lmvm->m, &lsb->P);CHKERRQ(ierr);
      ierr = VecDuplicateVecs(lmvm->Xprev, lmvm->m, &lsb->Q);CHKERRQ(ierr);
    }
    switch (lsb->scale_type) {
    case MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL:
      ierr = MatGetLocalSize(B, &n, &n);CHKERRQ(ierr);
      ierr = MatGetSize(B, &N, &N);CHKERRQ(ierr);
      ierr = MatSetSizes(lsb->D, n, n, N, N);CHKERRQ(ierr);
      ierr = MatSetUp(lsb->D);CHKERRQ(ierr);
      break;
    default:
      break;
    }
    lsb->allocated = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode MatView_LMVMSymBrdn(Mat B, PetscViewer pv)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_SymBrdn       *lsb = (Mat_SymBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  PetscBool         isascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)pv,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(pv,"Scale type: %s\n",MatLMVMSymBroydenScaleTypes[lsb->scale_type]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(pv,"Scale history: %d\n",lsb->sigma_hist);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(pv,"Scale params: alpha=%g, beta=%g, rho=%g\n",(double)lsb->alpha, (double)lsb->beta, (double)lsb->rho);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(pv,"Convex factors: phi=%g, theta=%g\n",(double)lsb->phi, (double)lsb->theta);CHKERRQ(ierr);
  }
  ierr = MatView_LMVM(B, pv);CHKERRQ(ierr);
  if (lsb->scale_type == MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL) {
    ierr = MatView(lsb->D, pv);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode MatSetFromOptions_LMVMSymBrdn(PetscOptionItems *PetscOptionsObject, Mat B)
{
  Mat_LMVM                     *lmvm = (Mat_LMVM*)B->data;
  Mat_SymBrdn                  *lsb = (Mat_SymBrdn*)lmvm->ctx;
  PetscErrorCode               ierr;

  PetscFunctionBegin;
  ierr = MatSetFromOptions_LMVM(PetscOptionsObject, B);CHKERRQ(ierr);
  ierr = PetscOptionsHead(PetscOptionsObject,"Restricted/Symmetric Broyden method for approximating SPD Jacobian actions (MATLMVMSYMBRDN)");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_lmvm_phi","(developer) convex ratio between BFGS and DFP components of the update","",lsb->phi,&lsb->phi,NULL);CHKERRQ(ierr);
  if ((lsb->phi < 0.0) || (lsb->phi > 1.0)) SETERRQ(PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_OUTOFRANGE, "convex ratio for the update formula cannot be outside the range of [0, 1]");
  ierr = MatSetFromOptions_LMVMSymBrdn_Private(PetscOptionsObject, B);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetFromOptions_LMVMSymBrdn_Private(PetscOptionItems *PetscOptionsObject, Mat B)
{
  Mat_LMVM                     *lmvm = (Mat_LMVM*)B->data;
  Mat_SymBrdn                  *lsb = (Mat_SymBrdn*)lmvm->ctx;
  Mat_LMVM                     *dbase;
  Mat_DiagBrdn                 *dctx;
  MatLMVMSymBroydenScaleType   stype = lsb->scale_type;
  PetscBool                    flg;
  PetscErrorCode               ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsReal("-mat_lmvm_beta","(developer) exponential factor in the diagonal J0 scaling","",lsb->beta,&lsb->beta,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_lmvm_theta","(developer) convex ratio between BFGS and DFP components of the diagonal J0 scaling","",lsb->theta,&lsb->theta,NULL);CHKERRQ(ierr);
  if ((lsb->theta < 0.0) || (lsb->theta > 1.0)) SETERRQ(PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_OUTOFRANGE, "convex ratio for the diagonal J0 scale cannot be outside the range of [0, 1]");
  ierr = PetscOptionsReal("-mat_lmvm_rho","(developer) update limiter in the J0 scaling","",lsb->rho,&lsb->rho,NULL);CHKERRQ(ierr);
  if ((lsb->rho < 0.0) || (lsb->rho > 1.0)) SETERRQ(PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_OUTOFRANGE, "update limiter in the J0 scaling cannot be outside the range of [0, 1]");
  ierr = PetscOptionsReal("-mat_lmvm_alpha","(developer) convex ratio in the J0 scaling","",lsb->alpha,&lsb->alpha,NULL);CHKERRQ(ierr);
  if ((lsb->alpha < 0.0) || (lsb->alpha > 1.0)) SETERRQ(PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_OUTOFRANGE, "convex ratio in the J0 scaling cannot be outside the range of [0, 1]");
  ierr = PetscOptionsBoundedInt("-mat_lmvm_sigma_hist","(developer) number of past updates to use in the default J0 scalar","",lsb->sigma_hist,&lsb->sigma_hist,NULL,1);CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-mat_lmvm_scale_type", "(developer) scaling type applied to J0","MatLMVMSymBrdnScaleType",MatLMVMSymBroydenScaleTypes,(PetscEnum)stype,(PetscEnum*)&stype,&flg);CHKERRQ(ierr);
  if (flg) ierr = MatLMVMSymBroydenSetScaleType(B, stype);CHKERRQ(ierr);
  if (lsb->scale_type == MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL) {
    ierr = MatSetFromOptions(lsb->D);CHKERRQ(ierr);
    dbase = (Mat_LMVM*)lsb->D->data;
    dctx = (Mat_DiagBrdn*)dbase->ctx;
    dctx->delta_min  = lsb->delta_min;
    dctx->delta_max  = lsb->delta_max;
    dctx->theta      = lsb->theta;
    dctx->rho        = lsb->rho;
    dctx->alpha      = lsb->alpha;
    dctx->beta       = lsb->beta;
    dctx->sigma_hist = lsb->sigma_hist;
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode MatCreate_LMVMSymBrdn(Mat B)
{
  Mat_LMVM          *lmvm;
  Mat_SymBrdn       *lsb;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatCreate_LMVM(B);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B, MATLMVMSYMBROYDEN);CHKERRQ(ierr);
  ierr = MatSetOption(B, MAT_SPD, PETSC_TRUE);CHKERRQ(ierr);
  B->ops->view = MatView_LMVMSymBrdn;
  B->ops->setfromoptions = MatSetFromOptions_LMVMSymBrdn;
  B->ops->setup = MatSetUp_LMVMSymBrdn;
  B->ops->destroy = MatDestroy_LMVMSymBrdn;
  B->ops->solve = MatSolve_LMVMSymBrdn;

  lmvm = (Mat_LMVM*)B->data;
  lmvm->square = PETSC_TRUE;
  lmvm->ops->allocate = MatAllocate_LMVMSymBrdn;
  lmvm->ops->reset = MatReset_LMVMSymBrdn;
  lmvm->ops->update = MatUpdate_LMVMSymBrdn;
  lmvm->ops->mult = MatMult_LMVMSymBrdn;
  lmvm->ops->copy = MatCopy_LMVMSymBrdn;

  ierr = PetscNewLog(B, &lsb);CHKERRQ(ierr);
  lmvm->ctx = (void*)lsb;
  lsb->allocated       = PETSC_FALSE;
  lsb->needP           = lsb->needQ = PETSC_TRUE;
  lsb->phi             = 0.125;
  lsb->theta           = 0.125;
  lsb->alpha           = 1.0;
  lsb->rho             = 1.0;
  lsb->beta            = 0.5;
  lsb->sigma           = 1.0;
  lsb->delta           = 1.0;
  lsb->delta_min       = 1e-7;
  lsb->delta_max       = 100.0;
  lsb->sigma_hist      = 1;
  lsb->scale_type      = MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL;
  lsb->watchdog        = 0;
  lsb->max_seq_rejects = lmvm->m/2;

  ierr = MatCreate(PetscObjectComm((PetscObject)B), &lsb->D);CHKERRQ(ierr);
  ierr = MatSetType(lsb->D, MATLMVMDIAGBROYDEN);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(lsb->D, "J0_");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/*@
   MatLMVMSymBroydenSetDelta - Sets the starting value for the diagonal scaling vector computed
   in the SymBrdn approximations (also works for BFGS and DFP).

   Input Parameters:
+  B - LMVM matrix
-  delta - initial value for diagonal scaling

   Level: intermediate
@*/

PetscErrorCode MatLMVMSymBroydenSetDelta(Mat B, PetscScalar delta)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_SymBrdn       *lsb = (Mat_SymBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  PetscBool         is_bfgs, is_dfp, is_symbrdn, is_symbadbrdn;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)B, MATLMVMBFGS, &is_bfgs);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)B, MATLMVMDFP, &is_dfp);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)B, MATLMVMSYMBROYDEN, &is_symbrdn);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)B, MATLMVMSYMBADBROYDEN, &is_symbadbrdn);CHKERRQ(ierr);
  if (!is_bfgs && !is_dfp && !is_symbrdn && !is_symbadbrdn) SETERRQ(PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_INCOMP, "diagonal scaling is only available for DFP, BFGS and SymBrdn matrices");
  lsb->delta = PetscAbsReal(PetscRealPart(delta));
  lsb->delta = PetscMin(lsb->delta, lsb->delta_max);CHKERRQ(ierr);
  lsb->delta = PetscMax(lsb->delta, lsb->delta_min);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/*@
    MatLMVMSymBroydenSetScaleType - Sets the scale type for symmetric Broyden-type updates.

    Input Parameters:
+   snes - the iterative context
-   rtype - restart type

    Options Database:
.   -mat_lmvm_scale_type <none,scalar,diagonal> - set the scaling type

    Level: intermediate

    MatLMVMSymBrdnScaleTypes:
+   MAT_LMVM_SYMBROYDEN_SCALE_NONE - initial Hessian is the identity matrix
.   MAT_LMVM_SYMBROYDEN_SCALE_SCALAR - use the Shanno scalar as the initial Hessian
-   MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL - use a diagonalized BFGS update as the initial Hessian

.seealso: MATLMVMSYMBROYDEN, MatCreateLMVMSymBroyden()
@*/
PetscErrorCode MatLMVMSymBroydenSetScaleType(Mat B, MatLMVMSymBroydenScaleType stype)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_SymBrdn       *lsb = (Mat_SymBrdn*)lmvm->ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B,MAT_CLASSID,1);
  lsb->scale_type = stype;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/*@
   MatCreateLMVMSymBroyden - Creates a limited-memory Symmetric Broyden-type matrix used
   for approximating Jacobians. L-SymBrdn is a convex combination of L-DFP and
   L-BFGS such that SymBrdn = (1 - phi)*BFGS + phi*DFP. The combination factor
   phi is restricted to the range [0, 1], where the L-SymBrdn matrix is guaranteed
   to be symmetric positive-definite.

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
PetscErrorCode MatCreateLMVMSymBroyden(MPI_Comm comm, PetscInt n, PetscInt N, Mat *B)
{
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatCreate(comm, B);CHKERRQ(ierr);
  ierr = MatSetSizes(*B, n, n, N, N);CHKERRQ(ierr);
  ierr = MatSetType(*B, MATLMVMSYMBROYDEN);CHKERRQ(ierr);
  ierr = MatSetUp(*B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode MatSymBrdnApplyJ0Fwd(Mat B, Vec X, Vec Z)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_SymBrdn       *lsb = (Mat_SymBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (lmvm->J0 || lmvm->user_pc || lmvm->user_ksp || lmvm->user_scale) {
    lsb->scale_type = MAT_LMVM_SYMBROYDEN_SCALE_USER;
    ierr = MatLMVMApplyJ0Fwd(B, X, Z);CHKERRQ(ierr);
  } else {
    switch (lsb->scale_type) {
    case MAT_LMVM_SYMBROYDEN_SCALE_SCALAR:
      ierr = VecCopy(X, Z);CHKERRQ(ierr);
      ierr = VecScale(Z, 1.0/lsb->sigma);CHKERRQ(ierr);
      break;
    case MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL:
      ierr = MatMult(lsb->D, X, Z);CHKERRQ(ierr);
      break;
    case MAT_LMVM_SYMBROYDEN_SCALE_NONE:
    default:
      ierr = VecCopy(X, Z);CHKERRQ(ierr);
      break;
    }
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode MatSymBrdnApplyJ0Inv(Mat B, Vec F, Vec dX)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_SymBrdn       *lsb = (Mat_SymBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (lmvm->J0 || lmvm->user_pc || lmvm->user_ksp || lmvm->user_scale) {
    lsb->scale_type = MAT_LMVM_SYMBROYDEN_SCALE_USER;
    ierr = MatLMVMApplyJ0Inv(B, F, dX);CHKERRQ(ierr);
  } else {
    switch (lsb->scale_type) {
    case MAT_LMVM_SYMBROYDEN_SCALE_SCALAR:
      ierr = VecCopy(F, dX);CHKERRQ(ierr);
      ierr = VecScale(dX, lsb->sigma);CHKERRQ(ierr);
      break;
    case MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL:
      ierr = MatSolve(lsb->D, F, dX);CHKERRQ(ierr);
      break;
    case MAT_LMVM_SYMBROYDEN_SCALE_NONE:
    default:
      ierr = VecCopy(F, dX);CHKERRQ(ierr);
      break;
    }
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode MatSymBrdnComputeJ0Scalar(Mat B)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_SymBrdn       *lsb = (Mat_SymBrdn*)lmvm->ctx;
  PetscInt          i, start;
  PetscReal         a, b, c, sig1, sig2, signew;

  PetscFunctionBegin;
  if (lsb->sigma_hist == 0) {
    signew = 1.0;
  } else {
    start = PetscMax(0, lmvm->k-lsb->sigma_hist+1);
    signew = 0.0;
    if (lsb->alpha == 1.0) {
      for (i = start; i <= lmvm->k; ++i) {
        signew += lsb->yts[i]/lsb->yty[i];
      }
    } else if (lsb->alpha == 0.5) {
      for (i = start; i <= lmvm->k; ++i) {
        signew += lsb->sts[i]/lsb->yty[i];
      }
      signew = PetscSqrtReal(signew);
    } else if (lsb->alpha == 0.0) {
      for (i = start; i <= lmvm->k; ++i) {
        signew += lsb->sts[i]/lsb->yts[i];
      }
    } else {
      /* compute coefficients of the quadratic */
      a = b = c = 0.0;
      for (i = start; i <= lmvm->k; ++i) {
        a += lsb->yty[i];
        b += lsb->yts[i];
        c += lsb->sts[i];
      }
      a *= lsb->alpha;
      b *= -(2.0*lsb->alpha - 1.0);
      c *= lsb->alpha - 1.0;
      /* use quadratic formula to find roots */
      sig1 = (-b + PetscSqrtReal(b*b - 4.0*a*c))/(2.0*a);
      sig2 = (-b - PetscSqrtReal(b*b - 4.0*a*c))/(2.0*a);
      /* accept the positive root as the scalar */
      if (sig1 > 0.0) {
        signew = sig1;
      } else if (sig2 > 0.0) {
        signew = sig2;
      } else {
        SETERRQ(PetscObjectComm((PetscObject)B), PETSC_ERR_CONV_FAILED, "Cannot find positive scalar");
      }
    }
  }
  lsb->sigma = lsb->rho*signew + (1.0 - lsb->rho)*lsb->sigma;
  PetscFunctionReturn(0);
}
