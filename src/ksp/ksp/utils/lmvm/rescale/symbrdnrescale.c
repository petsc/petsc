#include <petscdevice.h>
#include "symbrdnrescale.h"

PetscLogEvent SBRDN_Rescale;

const char *const MatLMVMSymBroydenScaleTypes[] = {"none", "scalar", "diagonal", "user", "decide", "MatLMVMSymBroydenScaleType", "MAT_LMVM_SYMBROYDEN_SCALING_", NULL};

static PetscErrorCode SymBroydenRescaleUpdateScalar(Mat B, SymBroydenRescale ldb)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  PetscReal a, b, c, signew;
  PetscReal sigma_inv, sigma;
  PetscInt  oldest, next;

  PetscFunctionBegin;
  next   = ldb->k;
  oldest = PetscMax(0, ldb->k - ldb->sigma_hist);
  PetscCall(MatNorm(lmvm->J0, NORM_INFINITY, &sigma_inv));
  sigma = 1.0 / sigma_inv;
  if (ldb->sigma_hist == 0) {
    signew = 1.0;
  } else {
    signew = 0.0;
    if (ldb->alpha == 1.0) {
      for (PetscInt i = 0; i < next - oldest; ++i) signew += ldb->yts[i] / ldb->yty[i];
    } else if (ldb->alpha == 0.5) {
      for (PetscInt i = 0; i < next - oldest; ++i) signew += ldb->sts[i] / ldb->yty[i];
      signew = PetscSqrtReal(signew);
    } else if (ldb->alpha == 0.0) {
      for (PetscInt i = 0; i < next - oldest; ++i) signew += ldb->sts[i] / ldb->yts[i];
    } else {
      /* compute coefficients of the quadratic */
      a = b = c = 0.0;
      for (PetscInt i = 0; i < next - oldest; ++i) {
        a += ldb->yty[i];
        b += ldb->yts[i];
        c += ldb->sts[i];
      }
      a *= ldb->alpha;
      b *= -(2.0 * ldb->alpha - 1.0);
      c *= ldb->alpha - 1.0;
      /* use quadratic formula to find roots */
      PetscReal sqrtdisc = PetscSqrtReal(b * b - 4 * a * c);
      if (b >= 0.0) {
        if (a >= 0.0) {
          signew = (2 * c) / (-b - sqrtdisc);
        } else {
          signew = (-b - sqrtdisc) / (2 * a);
        }
      } else {
        if (a >= 0.0) {
          signew = (-b + sqrtdisc) / (2 * a);
        } else {
          signew = (2 * c) / (-b + sqrtdisc);
        }
      }
      PetscCheck(signew > 0.0, PetscObjectComm((PetscObject)B), PETSC_ERR_CONV_FAILED, "Cannot find positive scalar");
    }
  }
  sigma = ldb->rho * signew + (1.0 - ldb->rho) * sigma;
  PetscCall(MatLMVMSetJ0Scale(B, 1.0 / sigma));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DiagonalUpdate(SymBroydenRescale ldb, Vec D, Vec s, Vec y, Vec V, Vec W, Vec BFGS, Vec DFP, PetscReal theta, PetscReal yts)
{
  PetscFunctionBegin;
  /*  V = |y o y| */
  PetscCall(VecPointwiseMult(V, y, y));
  if (PetscDefined(USE_COMPLEX)) PetscCall(VecAbs(V));

  /*  W = D o s */
  PetscReal stDs;
  PetscCall(VecPointwiseMult(W, D, s));
  PetscCall(VecDotRealPart(W, s, &stDs));

  PetscCall(VecAXPY(D, 1.0 / yts, ldb->V));

  /*  Safeguard stDs */
  stDs = PetscMax(stDs, ldb->tol);

  if (theta != 1.0) {
    /*  BFGS portion of the update */

    /*  U = |(D o s) o (D o s)| */
    PetscCall(VecPointwiseMult(BFGS, W, W));
    if (PetscDefined(USE_COMPLEX)) PetscCall(VecAbs(BFGS));

    /*  Assemble */
    PetscCall(VecScale(BFGS, -1.0 / stDs));
  }

  if (theta != 0.0) {
    /*  DFP portion of the update */
    /*  U = Real(conj(y) o D o s) */
    PetscCall(VecCopy(y, DFP));
    PetscCall(VecConjugate(DFP));
    PetscCall(VecPointwiseMult(DFP, DFP, W));
    if (PetscDefined(USE_COMPLEX)) {
      PetscCall(VecCopy(DFP, W));
      PetscCall(VecConjugate(W));
      PetscCall(VecAXPY(DFP, 1.0, W));
    } else {
      PetscCall(VecScale(DFP, 2.0));
    }

    /*  Assemble */
    PetscCall(VecAXPBY(DFP, stDs / yts, -1.0, V));
  }

  if (theta == 0.0) {
    PetscCall(VecAXPY(D, 1.0, BFGS));
  } else if (theta == 1.0) {
    PetscCall(VecAXPY(D, 1.0 / yts, DFP));
  } else {
    /*  Broyden update Dkp1 = Dk + (1-theta)*P + theta*Q + y_i^2/yts*/
    PetscCall(VecAXPBYPCZ(D, 1.0 - theta, theta / yts, 1.0, BFGS, DFP));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SymBroydenRescaleUpdateDiagonal(Mat B, SymBroydenRescale ldb)
{
  Mat_LMVM   *lmvm = (Mat_LMVM *)B->data;
  PetscInt    oldest, next;
  Vec         invD, s_last, y_last;
  LMBasis     S, Y;
  PetscScalar yts;
  PetscReal   sigma;

  PetscFunctionBegin;
  next   = ldb->k;
  oldest = PetscMax(0, ldb->k - ldb->sigma_hist);
  PetscCall(MatLMVMProductsGetDiagonalValue(B, LMBASIS_Y, LMBASIS_S, next - 1, &yts));
  PetscCall(MatLMVMGetUpdatedBasis(B, LMBASIS_S, &S, NULL, NULL));
  PetscCall(MatLMVMGetUpdatedBasis(B, LMBASIS_Y, &Y, NULL, NULL));
  PetscCall(LMBasisGetVecRead(S, next - 1, &s_last));
  PetscCall(LMBasisGetVecRead(Y, next - 1, &y_last));
  PetscCall(MatLMVMGetJ0InvDiag(B, &invD));
  if (ldb->forward) {
    /* We are doing diagonal scaling of the forward Hessian B */
    /*  BFGS = DFP = inv(D); */
    PetscCall(VecCopy(invD, ldb->invDnew));
    PetscCall(VecReciprocal(ldb->invDnew));
    PetscCall(DiagonalUpdate(ldb, ldb->invDnew, s_last, y_last, ldb->V, ldb->W, ldb->BFGS, ldb->DFP, ldb->theta, PetscRealPart(yts)));
    /*  Obtain inverse and ensure positive definite */
    PetscCall(VecReciprocal(ldb->invDnew));
  } else {
    /* Inverse Hessian update instead. */
    PetscCall(VecCopy(invD, ldb->invDnew));
    PetscCall(DiagonalUpdate(ldb, ldb->invDnew, y_last, s_last, ldb->V, ldb->W, ldb->DFP, ldb->BFGS, 1.0 - ldb->theta, PetscRealPart(yts)));
  }
  PetscCall(VecAbs(ldb->invDnew));
  PetscCall(LMBasisRestoreVecRead(Y, next - 1, &y_last));
  PetscCall(LMBasisRestoreVecRead(S, next - 1, &s_last));

  if (ldb->sigma_hist > 0) {
    // We are computing the scaling factor sigma that minimizes
    //
    // Sum_i || sigma^(alpha) (D^(-beta) o y_i) - sigma^(alpha-1) (D^(1-beta) o s_i) ||_2^2
    //                        `-------.-------'                   `--------.-------'
    //                               v_i                                  w_i
    //
    // To do this we first have to compute the sums of the dot product terms
    //
    // yy_sum = Sum_i v_i^T v_i,
    // ys_sum = Sum_i v_i^T w_i, and
    // ss_sum = Sum_i w_i^T w_i.
    //
    // These appear in the quadratic equation for the optimality condition for sigma,
    //
    // [alpha yy_sum] sigma^2 - [(2 alpha - 1) ys_sum] * sigma + [(alpha - 1) * ss_sum] = 0
    //
    // which we solve for sigma.

    PetscReal yy_sum = 0; /*  No safeguard required */
    PetscReal ys_sum = 0; /*  No safeguard required */
    PetscReal ss_sum = 0; /*  No safeguard required */
    PetscInt  start  = PetscMax(oldest, lmvm->k - ldb->sigma_hist);

    Vec D_minus_beta             = NULL;
    Vec D_minus_beta_squared     = NULL;
    Vec D_one_minus_beta         = NULL;
    Vec D_one_minus_beta_squared = NULL;
    if (ldb->beta == 0.5) {
      D_minus_beta_squared = ldb->invDnew; // (D^(-0.5))^2 = D^-1

      PetscCall(VecCopy(ldb->invDnew, ldb->U));
      PetscCall(VecReciprocal(ldb->U));
      D_one_minus_beta_squared = ldb->U; // (D^(1-0.5))^2 = D
    } else if (ldb->beta == 0.0) {
      PetscCall(VecCopy(ldb->invDnew, ldb->U));
      PetscCall(VecReciprocal(ldb->U));
      D_one_minus_beta = ldb->U; // D^1
    } else if (ldb->beta == 1.0) {
      D_minus_beta = ldb->invDnew; // D^-1
    } else {
      PetscCall(VecCopy(ldb->invDnew, ldb->DFP));
      PetscCall(VecPow(ldb->DFP, ldb->beta));
      D_minus_beta = ldb->DFP;

      PetscCall(VecCopy(ldb->invDnew, ldb->BFGS));
      PetscCall(VecPow(ldb->BFGS, ldb->beta - 1));
      D_one_minus_beta = ldb->BFGS;
    }
    for (PetscInt i = start - oldest; i < next - oldest; ++i) {
      Vec s_i, y_i;

      PetscCall(LMBasisGetVecRead(S, oldest + i, &s_i));
      PetscCall(LMBasisGetVecRead(Y, oldest + i, &y_i));
      if (ldb->beta == 0.5) {
        PetscReal ytDinvy, stDs;

        PetscCall(VecPointwiseMult(ldb->V, y_i, D_minus_beta_squared));
        PetscCall(VecPointwiseMult(ldb->W, s_i, D_one_minus_beta_squared));
        PetscCall(VecDotRealPart(ldb->W, s_i, &stDs));
        PetscCall(VecDotRealPart(ldb->V, y_i, &ytDinvy));
        ss_sum += stDs;        // ||s||_{D^(2*(1-beta))}^2
        ys_sum += ldb->yts[i]; // s^T D^(1 - 2*beta) y
        yy_sum += ytDinvy;     // ||y||_{D^(-2*beta)}^2
      } else if (ldb->beta == 0.0) {
        PetscScalar ytDs_scalar;
        PetscReal   stDsr;

        PetscCall(VecPointwiseMult(ldb->W, s_i, D_one_minus_beta));
        PetscCall(VecDotNorm2(y_i, ldb->W, &ytDs_scalar, &stDsr));
        ss_sum += stDsr;                      // ||s||_{D^(2*(1-beta))}^2
        ys_sum += PetscRealPart(ytDs_scalar); // s^T D^(1 - 2*beta) y
        yy_sum += ldb->yty[i];                // ||y||_{D^(-2*beta)}^2
      } else if (ldb->beta == 1.0) {
        PetscScalar ytDs_scalar;
        PetscReal   ytDyr;

        PetscCall(VecPointwiseMult(ldb->V, y_i, D_minus_beta));
        PetscCall(VecDotNorm2(s_i, ldb->V, &ytDs_scalar, &ytDyr));
        ss_sum += ldb->sts[i];                // ||s||_{D^(2*(1-beta))}^2
        ys_sum += PetscRealPart(ytDs_scalar); // s^T D^(1 - 2*beta) y
        yy_sum += ytDyr;                      // ||y||_{D^(-2*beta)}^2
      } else {
        PetscScalar ytDs_scalar;
        PetscReal   ytDyr, stDs;

        PetscCall(VecPointwiseMult(ldb->V, y_i, D_minus_beta));
        PetscCall(VecPointwiseMult(ldb->W, s_i, D_one_minus_beta));
        PetscCall(VecDotNorm2(ldb->W, ldb->V, &ytDs_scalar, &ytDyr));
        PetscCall(VecDotRealPart(ldb->W, ldb->W, &stDs));
        ss_sum += stDs;                       // ||s||_{D^(2*(1-beta))}^2
        ys_sum += PetscRealPart(ytDs_scalar); // s^T D^(1 - 2*beta) y
        yy_sum += ytDyr;                      // ||y||_{D^(-2*beta)}^2
      }
      PetscCall(LMBasisRestoreVecRead(Y, oldest + i, &y_i));
      PetscCall(LMBasisRestoreVecRead(S, oldest + i, &s_i));
    }

    if (ldb->alpha == 0.0) {
      /*  Safeguard ys_sum  */
      ys_sum = PetscMax(ldb->tol, ys_sum);

      sigma = ss_sum / ys_sum;
    } else if (1.0 == ldb->alpha) {
      /* yy_sum is never 0; if it were, we'd be at the minimum */
      sigma = ys_sum / yy_sum;
    } else {
      PetscReal a         = ldb->alpha * yy_sum;
      PetscReal b         = -(2.0 * ldb->alpha - 1.0) * ys_sum;
      PetscReal c         = (ldb->alpha - 1.0) * ss_sum;
      PetscReal sqrt_disc = PetscSqrtReal(b * b - 4 * a * c);

      // numerically stable computation of positive root
      if (b >= 0.0) {
        if (a >= 0) {
          PetscReal denom = PetscMax(-b - sqrt_disc, ldb->tol);

          sigma = (2 * c) / denom;
        } else {
          PetscReal denom = PetscMax(2 * a, ldb->tol);

          sigma = (-b - sqrt_disc) / denom;
        }
      } else {
        if (a >= 0) {
          PetscReal denom = PetscMax(2 * a, ldb->tol);

          sigma = (-b + sqrt_disc) / denom;
        } else {
          PetscReal denom = PetscMax(-b + sqrt_disc, ldb->tol);

          sigma = (2 * c) / denom;
        }
      }
    }
  } else {
    sigma = 1.0;
  }
  /*  If Q has small values, then Q^(r_beta - 1)
      can have very large values.  Hence, ys_sum
      and ss_sum can be infinity.  In this case,
      sigma can either be not-a-number or infinity. */
  if (PetscIsNormalReal(sigma)) PetscCall(VecScale(ldb->invDnew, sigma));
  /* Combine the old diagonal and the new diagonal using a convex limiter */
  if (ldb->rho == 1.0) PetscCall(VecCopy(ldb->invDnew, invD));
  else if (ldb->rho) PetscCall(VecAXPBY(invD, 1.0 - ldb->rho, ldb->rho, ldb->invDnew));
  PetscCall(MatLMVMRestoreJ0InvDiag(B, &invD));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SymBroydenRescaleUpdateJ0(Mat B, SymBroydenRescale ldb)
{
  PetscFunctionBegin;
  if (ldb->scale_type == MAT_LMVM_SYMBROYDEN_SCALE_SCALAR) PetscCall(SymBroydenRescaleUpdateScalar(B, ldb));
  else if (ldb->scale_type == MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL) PetscCall(SymBroydenRescaleUpdateDiagonal(B, ldb));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode SymBroydenRescaleUpdate(Mat B, SymBroydenRescale ldb)
{
  PetscInt oldest, next;

  PetscFunctionBegin;
  PetscCall(SymBroydenRescaleSetUp(B, ldb));
  if (ldb->scale_type == MAT_LMVM_SYMBROYDEN_SCALE_NONE || ldb->scale_type == MAT_LMVM_SYMBROYDEN_SCALE_USER) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscLogEventBegin(SBRDN_Rescale, NULL, NULL, NULL, NULL));
  PetscCall(MatLMVMGetRange(B, &oldest, &next));
  if (next > ldb->k) {
    PetscInt new_oldest = PetscMax(0, next - ldb->sigma_hist);
    PetscInt ldb_oldest = PetscMax(0, ldb->k - ldb->sigma_hist);

    if (new_oldest > ldb_oldest) {
      for (PetscInt i = new_oldest; i < ldb->k; i++) {
        ldb->yty[i - new_oldest] = ldb->yty[i - ldb_oldest];
        ldb->yts[i - new_oldest] = ldb->yts[i - ldb_oldest];
        ldb->sts[i - new_oldest] = ldb->sts[i - ldb_oldest];
      }
    }
    for (PetscInt i = PetscMax(new_oldest, ldb->k); i < next; i++) {
      PetscScalar yty, sts, yts;

      PetscCall(MatLMVMProductsGetDiagonalValue(B, LMBASIS_Y, LMBASIS_Y, i, &yty));
      PetscCall(MatLMVMProductsGetDiagonalValue(B, LMBASIS_Y, LMBASIS_S, i, &yts));
      PetscCall(MatLMVMProductsGetDiagonalValue(B, LMBASIS_S, LMBASIS_S, i, &sts));
      ldb->yty[i - new_oldest] = PetscRealPart(yty);
      ldb->yts[i - new_oldest] = PetscRealPart(yts);
      ldb->sts[i - new_oldest] = PetscRealPart(sts);
    }
    ldb->k = next;
  }
  PetscCall(SymBroydenRescaleUpdateJ0(B, ldb));
  PetscCall(PetscLogEventEnd(SBRDN_Rescale, NULL, NULL, NULL, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode SymBroydenRescaleSetDelta(Mat B, SymBroydenRescale ldb, PetscReal delta)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  PetscBool same;

  PetscFunctionBegin;
  same       = (delta == ldb->delta) ? PETSC_TRUE : PETSC_FALSE;
  ldb->delta = delta;
  ldb->delta = PetscMin(ldb->delta, ldb->delta_max);
  ldb->delta = PetscMax(ldb->delta, ldb->delta_min);
  // if there have been no updates yet, propagate delta to J0
  if (!same && !lmvm->prev_set && (ldb->scale_type == MAT_LMVM_SYMBROYDEN_SCALE_SCALAR || ldb->scale_type == MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL)) {
    ldb->initialized = PETSC_FALSE;
    B->preallocated  = PETSC_FALSE; // make sure SyBroydenInitializeJ0() is called in the next MatCheckPreallocated()
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode SymBroydenRescaleCopy(SymBroydenRescale bctx, SymBroydenRescale mctx)
{
  PetscFunctionBegin;
  mctx->scale_type = bctx->scale_type;
  mctx->theta      = bctx->theta;
  mctx->alpha      = bctx->alpha;
  mctx->beta       = bctx->beta;
  mctx->rho        = bctx->rho;
  mctx->delta      = bctx->delta;
  mctx->delta_min  = bctx->delta_min;
  mctx->delta_max  = bctx->delta_max;
  mctx->tol        = bctx->tol;
  mctx->sigma_hist = bctx->sigma_hist;
  mctx->forward    = bctx->forward;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode SymBroydenRescaleSetDiagonalMode(SymBroydenRescale ldb, PetscBool forward)
{
  PetscFunctionBegin;
  ldb->forward = forward;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode SymBroydenRescaleGetType(SymBroydenRescale ldb, MatLMVMSymBroydenScaleType *stype)
{
  PetscFunctionBegin;
  *stype = ldb->scale_type;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode SymBroydenRescaleSetType(SymBroydenRescale ldb, MatLMVMSymBroydenScaleType stype)
{
  PetscFunctionBegin;
  ldb->scale_type = stype;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode SymBroydenRescaleSetFromOptions(Mat B, SymBroydenRescale ldb, PetscOptionItems PetscOptionsObject)
{
  MatLMVMSymBroydenScaleType stype = ldb->scale_type;
  PetscBool                  flg;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "Restricted Broyden method for updating diagonal Jacobian approximation (MATLMVMDIAGBRDN)");
  PetscCall(PetscOptionsEnum("-mat_lmvm_scale_type", "(developer) scaling type applied to J0", "MatLMVMSymBroydenScaleType", MatLMVMSymBroydenScaleTypes, (PetscEnum)stype, (PetscEnum *)&stype, &flg));
  PetscCall(PetscOptionsReal("-mat_lmvm_theta", "(developer) convex ratio between BFGS and DFP components of the diagonal J0 scaling", "", ldb->theta, &ldb->theta, NULL));
  PetscCall(PetscOptionsReal("-mat_lmvm_rho", "(developer) update limiter in the J0 scaling", "", ldb->rho, &ldb->rho, NULL));
  PetscCall(PetscOptionsReal("-mat_lmvm_tol", "(developer) tolerance for bounding rescaling denominator", "", ldb->tol, &ldb->tol, NULL));
  PetscCall(PetscOptionsRangeReal("-mat_lmvm_alpha", "(developer) convex ratio in the J0 scaling", "", ldb->alpha, &ldb->alpha, NULL, 0.0, 1.0));
  PetscCall(PetscOptionsBool("-mat_lmvm_forward", "Forward -> Update diagonal scaling for B. Else -> diagonal scaling for H.", "", ldb->forward, &ldb->forward, NULL));
  PetscCall(PetscOptionsReal("-mat_lmvm_beta", "(developer) exponential factor in the diagonal J0 scaling", "", ldb->beta, &ldb->beta, NULL));
  PetscCall(PetscOptionsBoundedInt("-mat_lmvm_sigma_hist", "(developer) number of past updates to use in the default J0 scalar", "", ldb->sigma_hist, &ldb->sigma_hist, NULL, 0));
  PetscOptionsHeadEnd();
  PetscCheck(!(ldb->theta < 0.0) && !(ldb->theta > 1.0), PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_OUTOFRANGE, "convex ratio for the diagonal J0 scale cannot be outside the range of [0, 1]");
  PetscCheck(!(ldb->alpha < 0.0) && !(ldb->alpha > 1.0), PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_OUTOFRANGE, "convex ratio in the J0 scaling cannot be outside the range of [0, 1]");
  PetscCheck(!(ldb->rho < 0.0) && !(ldb->rho > 1.0), PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_OUTOFRANGE, "convex update limiter in the J0 scaling cannot be outside the range of [0, 1]");
  PetscCheck(ldb->sigma_hist >= 0, PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_OUTOFRANGE, "J0 scaling history length cannot be negative");
  if (flg) PetscCall(SymBroydenRescaleSetType(ldb, stype));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SymBroydenRescaleAllocate(Mat B, SymBroydenRescale ldb)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;

  PetscFunctionBegin;
  if (!ldb->allocated) {
    PetscCall(PetscMalloc3(ldb->sigma_hist, &ldb->yty, ldb->sigma_hist, &ldb->yts, ldb->sigma_hist, &ldb->sts));
    PetscCall(VecDuplicate(lmvm->Xprev, &ldb->invDnew));
    PetscCall(VecDuplicate(lmvm->Xprev, &ldb->BFGS));
    PetscCall(VecDuplicate(lmvm->Xprev, &ldb->DFP));
    PetscCall(VecDuplicate(lmvm->Xprev, &ldb->U));
    PetscCall(VecDuplicate(lmvm->Xprev, &ldb->V));
    PetscCall(VecDuplicate(lmvm->Xprev, &ldb->W));
    ldb->allocated = PETSC_TRUE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode SymBroydenRescaleSetUp(Mat B, SymBroydenRescale ldb)
{
  PetscFunctionBegin;
  PetscCall(SymBroydenRescaleAllocate(B, ldb));
  if (ldb->scale_type == MAT_LMVM_SYMBROYDEN_SCALE_DECIDE) {
    Mat       J0;
    PetscBool is_constant_or_diagonal;

    PetscCall(MatLMVMGetJ0(B, &J0));
    PetscCall(PetscObjectTypeCompareAny((PetscObject)J0, &is_constant_or_diagonal, MATCONSTANTDIAGONAL, MATDIAGONAL, ""));
    ldb->scale_type = is_constant_or_diagonal ? MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL : MAT_LMVM_SYMBROYDEN_SCALE_NONE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode SymBroydenRescaleInitializeJ0(Mat B, SymBroydenRescale ldb)
{
  PetscFunctionBegin;
  if (!ldb->initialized) {
    PetscCall(SymBroydenRescaleSetUp(B, ldb));
    switch (ldb->scale_type) {
    case MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL: {
      Vec invD;

      PetscCall(MatLMVMGetJ0InvDiag(B, &invD));
      PetscCall(VecSet(invD, ldb->delta));
      PetscCall(MatLMVMRestoreJ0InvDiag(B, &invD));
      break;
    }
    case MAT_LMVM_SYMBROYDEN_SCALE_SCALAR:
      PetscCall(MatLMVMSetJ0Scale(B, 1.0 / ldb->delta));
      break;
    default:
      break;
    }
    ldb->initialized = PETSC_TRUE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode SymBroydenRescaleView(SymBroydenRescale ldb, PetscViewer pv)
{
  PetscFunctionBegin;
  PetscBool isascii;
  PetscCall(PetscObjectTypeCompare((PetscObject)pv, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscCall(PetscViewerASCIIPrintf(pv, "Rescale type: %s\n", MatLMVMSymBroydenScaleTypes[ldb->scale_type]));
    if (ldb->scale_type == MAT_LMVM_SYMBROYDEN_SCALE_SCALAR || ldb->scale_type == MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL) {
      PetscCall(PetscViewerASCIIPrintf(pv, "Rescale history: %" PetscInt_FMT "\n", ldb->sigma_hist));
      PetscCall(PetscViewerASCIIPrintf(pv, "Rescale params: alpha=%g, beta=%g, rho=%g\n", (double)ldb->alpha, (double)ldb->beta, (double)ldb->rho));
    }
    if (ldb->scale_type == MAT_LMVM_SYMBROYDEN_SCALE_DIAGONAL) PetscCall(PetscViewerASCIIPrintf(pv, "Rescale convex factor: theta=%g\n", (double)ldb->theta));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode SymBroydenRescaleReset(Mat B, SymBroydenRescale ldb, MatLMVMResetMode mode)
{
  PetscFunctionBegin;
  ldb->k = 0;
  if (MatLMVMResetClearsBases(mode)) {
    if (ldb->allocated) {
      PetscCall(PetscFree3(ldb->yty, ldb->yts, ldb->sts));
      PetscCall(VecDestroy(&ldb->invDnew));
      PetscCall(VecDestroy(&ldb->BFGS));
      PetscCall(VecDestroy(&ldb->DFP));
      PetscCall(VecDestroy(&ldb->U));
      PetscCall(VecDestroy(&ldb->V));
      PetscCall(VecDestroy(&ldb->W));
      ldb->allocated = PETSC_FALSE;
    }
  }
  if (B && ldb->initialized && !MatLMVMResetClearsAll(mode)) PetscCall(SymBroydenRescaleInitializeJ0(B, ldb)); // eagerly reset J0 if we are rescaling
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode SymBroydenRescaleDestroy(SymBroydenRescale *ldb)
{
  PetscFunctionBegin;
  PetscCall(SymBroydenRescaleReset(NULL, *ldb, MAT_LMVM_RESET_ALL));
  PetscCall(PetscFree(*ldb));
  *ldb = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode SymBroydenRescaleCreate(SymBroydenRescale *ldb)
{
  PetscFunctionBegin;
  PetscCall(PetscNew(ldb));
  (*ldb)->scale_type = MAT_LMVM_SYMBROYDEN_SCALE_DECIDE;
  (*ldb)->theta      = 0.0;
  (*ldb)->alpha      = 1.0;
  (*ldb)->rho        = 1.0;
  (*ldb)->forward    = PETSC_TRUE;
  (*ldb)->beta       = 0.5;
  (*ldb)->delta      = 1.0;
  (*ldb)->delta_min  = 1e-7;
  (*ldb)->delta_max  = 100.0;
  (*ldb)->tol        = 1e-8;
  (*ldb)->sigma_hist = 1;
  (*ldb)->allocated  = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}
