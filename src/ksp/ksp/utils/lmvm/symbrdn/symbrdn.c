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
  PetscInt          i, j;
  PetscReal         numer;
  PetscScalar       sjtpi, yjtsi, wtsi, yjtqi, sjtyi, wtyi, ytx, stf, wtf, stp, ytq;

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
          PetscCall(VecAXPY(lsb->P[i], lsb->phi*lsb->stp[j]*PetscRealPart(wtsi), lsb->work));
        }
      }
      PetscCall(VecDot(lmvm->S[i], lsb->P[i], &stp));
      lsb->stp[i] = PetscRealPart(stp);
    }
    lsb->needP = PETSC_FALSE;
  }
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
          PetscCall(VecAXPY(lsb->Q[i], lsb->psi[j]*lsb->ytq[j]*PetscRealPart(wtyi), lsb->work));
        }
      }
      PetscCall(VecDot(lmvm->Y[i], lsb->Q[i], &ytq));
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
    PetscCall(VecAXPY(dX, lsb->psi[i]*lsb->ytq[i]*PetscRealPart(wtf), lsb->work));
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
  PetscInt          i, j;
  PetscScalar         sjtpi, yjtsi, wtsi, stz, ytx, wtx, stp;

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
          PetscCall(VecAXPY(lsb->P[i], lsb->phi*lsb->stp[j]*PetscRealPart(wtsi), lsb->work));
        }
      }
      PetscCall(VecDot(lmvm->S[i], lsb->P[i], &stp));
      lsb->stp[i] = PetscRealPart(stp);
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
    PetscCall(VecAXPY(Z, lsb->phi*lsb->stp[i]*PetscRealPart(wtx), lsb->work));
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
  PetscInt          old_k, i;
  PetscReal         curvtol;
  PetscScalar       curvature, ytytmp, ststmp;

  PetscFunctionBegin;
  if (!lmvm->m) PetscFunctionReturn(0);
  if (lmvm->prev_set) {
    /* Compute the new (S = X - Xprev) and (Y = F - Fprev) vectors */
    PetscCall(VecAYPX(lmvm->Xprev, -1.0, X));
    PetscCall(VecAYPX(lmvm->Fprev, -1.0, F));
    /* Test if the updates can be accepted */
    PetscCall(VecDotBegin(lmvm->Xprev, lmvm->Fprev, &curvature));
    PetscCall(VecDotBegin(lmvm->Xprev, lmvm->Xprev, &ststmp));
    PetscCall(VecDotEnd(lmvm->Xprev, lmvm->Fprev, &curvature));
    PetscCall(VecDotEnd(lmvm->Xprev, lmvm->Xprev, &ststmp));
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
      PetscCall(MatUpdateKernel_LMVM(B, lmvm->Xprev, lmvm->Fprev));
      /* If we hit the memory limit, shift the yts, yty and sts arrays */
      if (old_k == lmvm->k) {
        for (i = 0; i <= lmvm->k-1; ++i) {
          lsb->yts[i] = lsb->yts[i+1];
          lsb->yty[i] = lsb->yty[i+1];
          lsb->sts[i] = lsb->sts[i+1];
        }
      }
      /* Update history of useful scalars */
      PetscCall(VecDot(lmvm->Y[lmvm->k], lmvm->Y[lmvm->k], &ytytmp));
      lsb->yts[lmvm->k] = PetscRealPart(curvature);
      lsb->yty[lmvm->k] = PetscRealPart(ytytmp);
      lsb->sts[lmvm->k] = PetscRealPart(ststmp);
      /* Compute the scalar scale if necessary */
      if (lsb->scale_type == MAT_LMVM_SYMBROYDEN_SCALE_SCALAR) {
        PetscCall(MatSymBrdnComputeJ0Scalar(B));
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
      PetscCall(VecSet(dctx->invD, lsb->delta));
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
    PetscCall(MatLMVMUpdate(lsb->D, X, F));
  }

  if (lsb->watchdog > lsb->max_seq_rejects) {
    PetscCall(MatLMVMReset(B, PETSC_FALSE));
    if (lsb->scale_type == MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL) {
      PetscCall(MatLMVMReset(lsb->D, PETSC_FALSE));
    }
  }

  /* Save the solution and function to be used in the next update */
  PetscCall(VecCopy(X, lmvm->Xprev));
  PetscCall(VecCopy(F, lmvm->Fprev));
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
    PetscCall(VecCopy(blsb->P[i], mlsb->P[i]));
    PetscCall(VecCopy(blsb->Q[i], mlsb->Q[i]));
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
    PetscCall(MatCopy(blsb->D, mlsb->D, SAME_NONZERO_PATTERN));
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

  PetscFunctionBegin;
  lsb->watchdog = 0;
  lsb->needP = lsb->needQ = PETSC_TRUE;
  if (lsb->allocated) {
    if (destructive) {
      PetscCall(VecDestroy(&lsb->work));
      PetscCall(PetscFree5(lsb->stp, lsb->ytq, lsb->yts, lsb->yty, lsb->sts));
      PetscCall(PetscFree(lsb->psi));
      PetscCall(VecDestroyVecs(lmvm->m, &lsb->P));
      PetscCall(VecDestroyVecs(lmvm->m, &lsb->Q));
      switch (lsb->scale_type) {
      case MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL:
        PetscCall(MatLMVMReset(lsb->D, PETSC_TRUE));
        break;
      default:
        break;
      }
      lsb->allocated = PETSC_FALSE;
    } else {
      PetscCall(PetscMemzero(lsb->psi, lmvm->m));
      switch (lsb->scale_type) {
      case MAT_LMVM_SYMBROYDEN_SCALE_SCALAR:
        lsb->sigma = lsb->delta;
        break;
      case MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL:
        PetscCall(MatLMVMReset(lsb->D, PETSC_FALSE));
        dbase = (Mat_LMVM*)lsb->D->data;
        dctx = (Mat_DiagBrdn*)dbase->ctx;
        PetscCall(VecSet(dctx->invD, lsb->delta));
        break;
      case MAT_LMVM_SYMBROYDEN_SCALE_NONE:
        lsb->sigma = 1.0;
        break;
      default:
        break;
      }
    }
  }
  PetscCall(MatReset_LMVM(B, destructive));
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatAllocate_LMVMSymBrdn(Mat B, Vec X, Vec F)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_SymBrdn       *lsb = (Mat_SymBrdn*)lmvm->ctx;

  PetscFunctionBegin;
  PetscCall(MatAllocate_LMVM(B, X, F));
  if (!lsb->allocated) {
    PetscCall(VecDuplicate(X, &lsb->work));
    if (lmvm->m > 0) {
      PetscCall(PetscMalloc5(lmvm->m,&lsb->stp,lmvm->m,&lsb->ytq,lmvm->m,&lsb->yts,lmvm->m,&lsb->yty,lmvm->m,&lsb->sts));
      PetscCall(PetscCalloc1(lmvm->m,&lsb->psi));
      PetscCall(VecDuplicateVecs(X, lmvm->m, &lsb->P));
      PetscCall(VecDuplicateVecs(X, lmvm->m, &lsb->Q));
    }
    switch (lsb->scale_type) {
    case MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL:
      PetscCall(MatLMVMAllocate(lsb->D, X, F));
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

  PetscFunctionBegin;
  if (lsb->allocated) {
    PetscCall(VecDestroy(&lsb->work));
    PetscCall(PetscFree5(lsb->stp, lsb->ytq, lsb->yts, lsb->yty, lsb->sts));
    PetscCall(PetscFree(lsb->psi));
    PetscCall(VecDestroyVecs(lmvm->m, &lsb->P));
    PetscCall(VecDestroyVecs(lmvm->m, &lsb->Q));
    lsb->allocated = PETSC_FALSE;
  }
  PetscCall(MatDestroy(&lsb->D));
  PetscCall(PetscFree(lmvm->ctx));
  PetscCall(MatDestroy_LMVM(B));
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatSetUp_LMVMSymBrdn(Mat B)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_SymBrdn       *lsb = (Mat_SymBrdn*)lmvm->ctx;
  PetscInt          n, N;

  PetscFunctionBegin;
  PetscCall(MatSetUp_LMVM(B));
  if (!lsb->allocated) {
    PetscCall(VecDuplicate(lmvm->Xprev, &lsb->work));
    if (lmvm->m > 0) {
      PetscCall(PetscMalloc5(lmvm->m,&lsb->stp,lmvm->m,&lsb->ytq,lmvm->m,&lsb->yts,lmvm->m,&lsb->yty,lmvm->m,&lsb->sts));
      PetscCall(PetscCalloc1(lmvm->m,&lsb->psi));
      PetscCall(VecDuplicateVecs(lmvm->Xprev, lmvm->m, &lsb->P));
      PetscCall(VecDuplicateVecs(lmvm->Xprev, lmvm->m, &lsb->Q));
    }
    switch (lsb->scale_type) {
    case MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL:
      PetscCall(MatGetLocalSize(B, &n, &n));
      PetscCall(MatGetSize(B, &N, &N));
      PetscCall(MatSetSizes(lsb->D, n, n, N, N));
      PetscCall(MatSetUp(lsb->D));
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
  PetscBool         isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)pv,PETSCVIEWERASCII,&isascii));
  if (isascii) {
    PetscCall(PetscViewerASCIIPrintf(pv,"Scale type: %s\n",MatLMVMSymBroydenScaleTypes[lsb->scale_type]));
    PetscCall(PetscViewerASCIIPrintf(pv,"Scale history: %d\n",lsb->sigma_hist));
    PetscCall(PetscViewerASCIIPrintf(pv,"Scale params: alpha=%g, beta=%g, rho=%g\n",(double)lsb->alpha, (double)lsb->beta, (double)lsb->rho));
    PetscCall(PetscViewerASCIIPrintf(pv,"Convex factors: phi=%g, theta=%g\n",(double)lsb->phi, (double)lsb->theta));
  }
  PetscCall(MatView_LMVM(B, pv));
  if (lsb->scale_type == MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL) {
    PetscCall(MatView(lsb->D, pv));
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode MatSetFromOptions_LMVMSymBrdn(PetscOptionItems *PetscOptionsObject, Mat B)
{
  Mat_LMVM                     *lmvm = (Mat_LMVM*)B->data;
  Mat_SymBrdn                  *lsb = (Mat_SymBrdn*)lmvm->ctx;

  PetscFunctionBegin;
  PetscCall(MatSetFromOptions_LMVM(PetscOptionsObject, B));
  PetscOptionsHeadBegin(PetscOptionsObject,"Restricted/Symmetric Broyden method for approximating SPD Jacobian actions (MATLMVMSYMBRDN)");
  PetscCall(PetscOptionsReal("-mat_lmvm_phi","(developer) convex ratio between BFGS and DFP components of the update","",lsb->phi,&lsb->phi,NULL));
  PetscCheck(!(lsb->phi < 0.0) && !(lsb->phi > 1.0),PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_OUTOFRANGE, "convex ratio for the update formula cannot be outside the range of [0, 1]");
  PetscCall(MatSetFromOptions_LMVMSymBrdn_Private(PetscOptionsObject, B));
  PetscOptionsHeadEnd();
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

  PetscFunctionBegin;
  PetscCall(PetscOptionsReal("-mat_lmvm_beta","(developer) exponential factor in the diagonal J0 scaling","",lsb->beta,&lsb->beta,NULL));
  PetscCall(PetscOptionsReal("-mat_lmvm_theta","(developer) convex ratio between BFGS and DFP components of the diagonal J0 scaling","",lsb->theta,&lsb->theta,NULL));
  PetscCheck(!(lsb->theta < 0.0) && !(lsb->theta > 1.0),PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_OUTOFRANGE, "convex ratio for the diagonal J0 scale cannot be outside the range of [0, 1]");
  PetscCall(PetscOptionsReal("-mat_lmvm_rho","(developer) update limiter in the J0 scaling","",lsb->rho,&lsb->rho,NULL));
  PetscCheck(!(lsb->rho < 0.0) && !(lsb->rho > 1.0),PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_OUTOFRANGE, "update limiter in the J0 scaling cannot be outside the range of [0, 1]");
  PetscCall(PetscOptionsReal("-mat_lmvm_alpha","(developer) convex ratio in the J0 scaling","",lsb->alpha,&lsb->alpha,NULL));
  PetscCheck(!(lsb->alpha < 0.0) && !(lsb->alpha > 1.0),PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_OUTOFRANGE, "convex ratio in the J0 scaling cannot be outside the range of [0, 1]");
  PetscCall(PetscOptionsBoundedInt("-mat_lmvm_sigma_hist","(developer) number of past updates to use in the default J0 scalar","",lsb->sigma_hist,&lsb->sigma_hist,NULL,1));
  PetscCall(PetscOptionsEnum("-mat_lmvm_scale_type", "(developer) scaling type applied to J0","MatLMVMSymBrdnScaleType",MatLMVMSymBroydenScaleTypes,(PetscEnum)stype,(PetscEnum*)&stype,&flg));
  if (flg) PetscCall(MatLMVMSymBroydenSetScaleType(B, stype));
  if (lsb->scale_type == MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL) {
    PetscCall(MatSetFromOptions(lsb->D));
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

  PetscFunctionBegin;
  PetscCall(MatCreate_LMVM(B));
  PetscCall(PetscObjectChangeTypeName((PetscObject)B, MATLMVMSYMBROYDEN));
  PetscCall(MatSetOption(B, MAT_SPD, PETSC_TRUE));
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

  PetscCall(PetscNewLog(B, &lsb));
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

  PetscCall(MatCreate(PetscObjectComm((PetscObject)B), &lsb->D));
  PetscCall(MatSetType(lsb->D, MATLMVMDIAGBROYDEN));
  PetscCall(MatSetOptionsPrefix(lsb->D, "J0_"));
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
  PetscBool         is_bfgs, is_dfp, is_symbrdn, is_symbadbrdn;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)B, MATLMVMBFGS, &is_bfgs));
  PetscCall(PetscObjectTypeCompare((PetscObject)B, MATLMVMDFP, &is_dfp));
  PetscCall(PetscObjectTypeCompare((PetscObject)B, MATLMVMSYMBROYDEN, &is_symbrdn));
  PetscCall(PetscObjectTypeCompare((PetscObject)B, MATLMVMSYMBADBROYDEN, &is_symbadbrdn));
  PetscCheckFalse(!is_bfgs && !is_dfp && !is_symbrdn && !is_symbadbrdn,PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_INCOMP, "diagonal scaling is only available for DFP, BFGS and SymBrdn matrices");
  lsb->delta = PetscAbsReal(PetscRealPart(delta));
  lsb->delta = PetscMin(lsb->delta, lsb->delta_max);
  lsb->delta = PetscMax(lsb->delta, lsb->delta_min);
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
  PetscFunctionBegin;
  PetscCall(MatCreate(comm, B));
  PetscCall(MatSetSizes(*B, n, n, N, N));
  PetscCall(MatSetType(*B, MATLMVMSYMBROYDEN));
  PetscCall(MatSetUp(*B));
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode MatSymBrdnApplyJ0Fwd(Mat B, Vec X, Vec Z)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_SymBrdn       *lsb = (Mat_SymBrdn*)lmvm->ctx;

  PetscFunctionBegin;
  if (lmvm->J0 || lmvm->user_pc || lmvm->user_ksp || lmvm->user_scale) {
    lsb->scale_type = MAT_LMVM_SYMBROYDEN_SCALE_USER;
    PetscCall(MatLMVMApplyJ0Fwd(B, X, Z));
  } else {
    switch (lsb->scale_type) {
    case MAT_LMVM_SYMBROYDEN_SCALE_SCALAR:
      PetscCall(VecCopy(X, Z));
      PetscCall(VecScale(Z, 1.0/lsb->sigma));
      break;
    case MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL:
      PetscCall(MatMult(lsb->D, X, Z));
      break;
    case MAT_LMVM_SYMBROYDEN_SCALE_NONE:
    default:
      PetscCall(VecCopy(X, Z));
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

  PetscFunctionBegin;
  if (lmvm->J0 || lmvm->user_pc || lmvm->user_ksp || lmvm->user_scale) {
    lsb->scale_type = MAT_LMVM_SYMBROYDEN_SCALE_USER;
    PetscCall(MatLMVMApplyJ0Inv(B, F, dX));
  } else {
    switch (lsb->scale_type) {
    case MAT_LMVM_SYMBROYDEN_SCALE_SCALAR:
      PetscCall(VecCopy(F, dX));
      PetscCall(VecScale(dX, lsb->sigma));
      break;
    case MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL:
      PetscCall(MatSolve(lsb->D, F, dX));
      break;
    case MAT_LMVM_SYMBROYDEN_SCALE_NONE:
    default:
      PetscCall(VecCopy(F, dX));
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
