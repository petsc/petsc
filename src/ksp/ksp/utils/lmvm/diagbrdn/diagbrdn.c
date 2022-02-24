#include <../src/ksp/ksp/utils/lmvm/diagbrdn/diagbrdn.h> /*I "petscksp.h" I*/

/*------------------------------------------------------------*/

static PetscErrorCode MatSolve_DiagBrdn(Mat B, Vec F, Vec dX)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_DiagBrdn      *ldb = (Mat_DiagBrdn*)lmvm->ctx;

  PetscFunctionBegin;
  VecCheckSameSize(F, 2, dX, 3);
  VecCheckMatCompatible(B, dX, 3, F, 2);
  CHKERRQ(VecPointwiseMult(dX, ldb->invD, F));
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatMult_DiagBrdn(Mat B, Vec X, Vec Z)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_DiagBrdn      *ldb = (Mat_DiagBrdn*)lmvm->ctx;

  PetscFunctionBegin;
  VecCheckSameSize(X, 2, Z, 3);
  VecCheckMatCompatible(B, X, 2, Z, 3);
  CHKERRQ(VecPointwiseDivide(Z, X, ldb->invD));
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatUpdate_DiagBrdn(Mat B, Vec X, Vec F)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_DiagBrdn      *ldb = (Mat_DiagBrdn*)lmvm->ctx;
  PetscInt          old_k, i, start;
  PetscScalar       yty, ststmp, curvature, ytDy, stDs, ytDs;
  PetscReal         curvtol, sigma, yy_sum, ss_sum, ys_sum, denom;

  PetscFunctionBegin;
  if (!lmvm->m) PetscFunctionReturn(0);
  if (lmvm->prev_set) {
    /* Compute the new (S = X - Xprev) and (Y = F - Fprev) vectors */
    CHKERRQ(VecAYPX(lmvm->Xprev, -1.0, X));
    CHKERRQ(VecAYPX(lmvm->Fprev, -1.0, F));
    /* Compute tolerance for accepting the update */
    CHKERRQ(VecDotBegin(lmvm->Xprev, lmvm->Fprev, &curvature));
    CHKERRQ(VecDotBegin(lmvm->Xprev, lmvm->Xprev, &ststmp));
    CHKERRQ(VecDotEnd(lmvm->Xprev, lmvm->Fprev, &curvature));
    CHKERRQ(VecDotEnd(lmvm->Xprev, lmvm->Xprev, &ststmp));
    if (PetscRealPart(ststmp) < lmvm->eps) {
      curvtol = 0.0;
    } else {
      curvtol = lmvm->eps * PetscRealPart(ststmp);
    }
    /* Test the curvature for the update */
    if (PetscRealPart(curvature) > curvtol) {
      /* Update is good so we accept it */
      old_k = lmvm->k;
      CHKERRQ(MatUpdateKernel_LMVM(B, lmvm->Xprev, lmvm->Fprev));
      /* If we hit the memory limit, shift the yty and yts arrays */
      if (old_k == lmvm->k) {
        for (i = 0; i <= lmvm->k-1; ++i) {
          ldb->yty[i] = ldb->yty[i+1];
          ldb->yts[i] = ldb->yts[i+1];
          ldb->sts[i] = ldb->sts[i+1];
        }
      }
      /* Accept dot products into the history */
      CHKERRQ(VecDot(lmvm->Y[lmvm->k], lmvm->Y[lmvm->k], &yty));
      ldb->yty[lmvm->k] = PetscRealPart(yty);
      ldb->yts[lmvm->k] = PetscRealPart(curvature);
      ldb->sts[lmvm->k] = PetscRealPart(ststmp);
      if (ldb->forward) {
        /* We are doing diagonal scaling of the forward Hessian B */
        /*  BFGS = DFP = inv(D); */
        CHKERRQ(VecCopy(ldb->invD, ldb->invDnew));
        CHKERRQ(VecReciprocal(ldb->invDnew));

        /*  V = y*y */
        CHKERRQ(VecPointwiseMult(ldb->V, lmvm->Y[lmvm->k], lmvm->Y[lmvm->k]));

        /*  W = inv(D)*s */
        CHKERRQ(VecPointwiseMult(ldb->W, ldb->invDnew, lmvm->S[lmvm->k]));
        CHKERRQ(VecDot(ldb->W, lmvm->S[lmvm->k], &stDs));

        /*  Safeguard stDs */
        stDs = PetscMax(PetscRealPart(stDs), ldb->tol);

        if (1.0 != ldb->theta) {
          /*  BFGS portion of the update */
          /*  U = (inv(D)*s)*(inv(D)*s) */
          CHKERRQ(VecPointwiseMult(ldb->U, ldb->W, ldb->W));

          /*  Assemble */
          CHKERRQ(VecAXPBY(ldb->BFGS, -1.0/stDs, 0.0, ldb->U));
        }
        if (0.0 != ldb->theta) {
          /*  DFP portion of the update */
          /*  U = inv(D)*s*y */
          CHKERRQ(VecPointwiseMult(ldb->U, ldb->W, lmvm->Y[lmvm->k]));

          /*  Assemble */
          CHKERRQ(VecAXPBY(ldb->DFP, stDs/ldb->yts[lmvm->k], 0.0, ldb->V));
          CHKERRQ(VecAXPY(ldb->DFP, -2.0, ldb->U));
        }

        if (0.0 == ldb->theta) {
          CHKERRQ(VecAXPY(ldb->invDnew, 1.0, ldb->BFGS));
        } else if (1.0 == ldb->theta) {
          CHKERRQ(VecAXPY(ldb->invDnew, 1.0/ldb->yts[lmvm->k], ldb->DFP));
        } else {
          /*  Broyden update Dkp1 = Dk + (1-theta)*P + theta*Q + y_i^2/yts*/
          CHKERRQ(VecAXPBYPCZ(ldb->invDnew, 1.0-ldb->theta, (ldb->theta)/ldb->yts[lmvm->k], 1.0, ldb->BFGS, ldb->DFP));
        }

        CHKERRQ(VecAXPY(ldb->invDnew, 1.0/ldb->yts[lmvm->k], ldb->V));
        /*  Obtain inverse and ensure positive definite */
        CHKERRQ(VecReciprocal(ldb->invDnew));
        CHKERRQ(VecAbs(ldb->invDnew));

      } else {
        /* Inverse Hessian update instead. */
        CHKERRQ(VecCopy(ldb->invD, ldb->invDnew));

        /*  V = s*s */
        CHKERRQ(VecPointwiseMult(ldb->V, lmvm->S[lmvm->k], lmvm->S[lmvm->k]));

        /*  W = D*y */
        CHKERRQ(VecPointwiseMult(ldb->W, ldb->invDnew, lmvm->Y[lmvm->k]));
        CHKERRQ(VecDot(ldb->W, lmvm->Y[lmvm->k], &ytDy));

        /*  Safeguard ytDy */
        ytDy = PetscMax(PetscRealPart(ytDy), ldb->tol);

        if (1.0 != ldb->theta) {
          /*  BFGS portion of the update */
          /*  U = s*Dy */
          CHKERRQ(VecPointwiseMult(ldb->U, ldb->W, lmvm->S[lmvm->k]));

          /*  Assemble */
          CHKERRQ(VecAXPBY(ldb->BFGS, ytDy/ldb->yts[lmvm->k], 0.0, ldb->V));
          CHKERRQ(VecAXPY(ldb->BFGS, -2.0, ldb->U));
        }
        if (0.0 != ldb->theta) {
          /*  DFP portion of the update */

          /*  U = (inv(D)*y)*(inv(D)*y) */
          CHKERRQ(VecPointwiseMult(ldb->U, ldb->W, ldb->W));

          /*  Assemble */
          CHKERRQ(VecAXPBY(ldb->DFP, -1.0/ytDy, 0.0, ldb->U));
        }

        if (0.0 == ldb->theta) {
          CHKERRQ(VecAXPY(ldb->invDnew, 1.0/ldb->yts[lmvm->k], ldb->BFGS));
        } else if (1.0 == ldb->theta) {
          CHKERRQ(VecAXPY(ldb->invDnew, 1.0, ldb->DFP));
        } else {
          /*  Broyden update U=(1-theta)*P + theta*Q */
          CHKERRQ(VecAXPBYPCZ(ldb->invDnew, (1.0-ldb->theta)/ldb->yts[lmvm->k], ldb->theta, 1.0, ldb->BFGS, ldb->DFP));
        }
        CHKERRQ(VecAXPY(ldb->invDnew, 1.0/ldb->yts[lmvm->k], ldb->V));
        /*  Ensure positive definite */
        CHKERRQ(VecAbs(ldb->invDnew));
      }
      if (ldb->sigma_hist > 0) {
        /*  Start with re-scaling on the newly computed diagonal */
        if (0.5 == ldb->beta) {
          if (1 == PetscMin(lmvm->nupdates, ldb->sigma_hist)) {
            CHKERRQ(VecPointwiseMult(ldb->V, lmvm->Y[0], ldb->invDnew));
            CHKERRQ(VecPointwiseDivide(ldb->W, lmvm->S[0], ldb->invDnew));

            CHKERRQ(VecDotBegin(ldb->V, lmvm->Y[0], &ytDy));
            CHKERRQ(VecDotBegin(ldb->W, lmvm->S[0], &stDs));
            CHKERRQ(VecDotEnd(ldb->V, lmvm->Y[0], &ytDy));
            CHKERRQ(VecDotEnd(ldb->W, lmvm->S[0], &stDs));

            ss_sum = PetscRealPart(stDs);
            yy_sum = PetscRealPart(ytDy);
            ys_sum = ldb->yts[0];
          } else {
            CHKERRQ(VecCopy(ldb->invDnew, ldb->U));
            CHKERRQ(VecReciprocal(ldb->U));

            /*  Compute summations for scalar scaling */
            yy_sum = 0;       /*  No safeguard required */
            ys_sum = 0;       /*  No safeguard required */
            ss_sum = 0;       /*  No safeguard required */
            start = PetscMax(0, lmvm->k-ldb->sigma_hist+1);
            for (i = start; i < PetscMin(lmvm->nupdates, ldb->sigma_hist); ++i) {
              CHKERRQ(VecPointwiseMult(ldb->V, lmvm->Y[i], ldb->U));
              CHKERRQ(VecPointwiseMult(ldb->W, lmvm->S[i], ldb->U));

              CHKERRQ(VecDotBegin(ldb->W, lmvm->S[i], &stDs));
              CHKERRQ(VecDotBegin(ldb->V, lmvm->Y[i], &ytDy));
              CHKERRQ(VecDotEnd(ldb->W, lmvm->S[i], &stDs));
              CHKERRQ(VecDotEnd(ldb->V, lmvm->Y[i], &ytDy));

              ss_sum += PetscRealPart(stDs);
              ys_sum += ldb->yts[i];
              yy_sum += PetscRealPart(ytDy);
            }
          }
        } else if (0.0 == ldb->beta) {
          if (1 == PetscMin(lmvm->nupdates, ldb->sigma_hist)) {
            /*  Compute summations for scalar scaling */
            CHKERRQ(VecPointwiseDivide(ldb->W, lmvm->S[0], ldb->invDnew));

            CHKERRQ(VecDotBegin(ldb->W, lmvm->Y[0], &ytDs));
            CHKERRQ(VecDotBegin(ldb->W, ldb->W, &stDs));
            CHKERRQ(VecDotEnd(ldb->W, lmvm->Y[0], &ytDs));
            CHKERRQ(VecDotEnd(ldb->W, ldb->W, &stDs));

            ys_sum = PetscRealPart(ytDs);
            ss_sum = PetscRealPart(stDs);
            yy_sum = ldb->yty[0];
          } else {
            CHKERRQ(VecCopy(ldb->invDnew, ldb->U));
            CHKERRQ(VecReciprocal(ldb->U));

            /*  Compute summations for scalar scaling */
            yy_sum = 0;       /*  No safeguard required */
            ys_sum = 0;       /*  No safeguard required */
            ss_sum = 0;       /*  No safeguard required */
            start = PetscMax(0, lmvm->k-ldb->sigma_hist+1);
            for (i = start; i < PetscMin(lmvm->nupdates, ldb->sigma_hist); ++i) {
              CHKERRQ(VecPointwiseMult(ldb->W, lmvm->S[i], ldb->U));

              CHKERRQ(VecDotBegin(ldb->W, lmvm->Y[i], &ytDs));
              CHKERRQ(VecDotBegin(ldb->W, ldb->W, &stDs));
              CHKERRQ(VecDotEnd(ldb->W, lmvm->Y[i], &ytDs));
              CHKERRQ(VecDotEnd(ldb->W, ldb->W, &stDs));

              ss_sum += PetscRealPart(stDs);
              ys_sum += PetscRealPart(ytDs);
              yy_sum += ldb->yty[i];
            }
          }
        } else if (1.0 == ldb->beta) {
          /*  Compute summations for scalar scaling */
          yy_sum = 0; /*  No safeguard required */
          ys_sum = 0; /*  No safeguard required */
          ss_sum = 0; /*  No safeguard required */
          start = PetscMax(0, lmvm->k-ldb->sigma_hist+1);
          for (i = start; i < PetscMin(lmvm->nupdates, ldb->sigma_hist); ++i) {
            CHKERRQ(VecPointwiseMult(ldb->V, lmvm->Y[i], ldb->invDnew));

            CHKERRQ(VecDotBegin(ldb->V, lmvm->S[i], &ytDs));
            CHKERRQ(VecDotBegin(ldb->V, ldb->V, &ytDy));
            CHKERRQ(VecDotEnd(ldb->V, lmvm->S[i], &ytDs));
            CHKERRQ(VecDotEnd(ldb->V, ldb->V, &ytDy));

            yy_sum += PetscRealPart(ytDy);
            ys_sum += PetscRealPart(ytDs);
            ss_sum += ldb->sts[i];
          }
        } else {
          CHKERRQ(VecCopy(ldb->invDnew, ldb->U));
          CHKERRQ(VecPow(ldb->U, ldb->beta-1));

          /*  Compute summations for scalar scaling */
          yy_sum = 0; /*  No safeguard required */
          ys_sum = 0; /*  No safeguard required */
          ss_sum = 0; /*  No safeguard required */
          start = PetscMax(0, lmvm->k-ldb->sigma_hist+1);
          for (i = start; i < PetscMin(lmvm->nupdates, ldb->sigma_hist); ++i) {
            CHKERRQ(VecPointwiseMult(ldb->V, ldb->invDnew, lmvm->Y[i]));
            CHKERRQ(VecPointwiseMult(ldb->W, ldb->U, lmvm->S[i]));

            CHKERRQ(VecDotBegin(ldb->V, ldb->W, &ytDs));
            CHKERRQ(VecDotBegin(ldb->V, ldb->V, &ytDy));
            CHKERRQ(VecDotBegin(ldb->W, ldb->W, &stDs));
            CHKERRQ(VecDotEnd(ldb->V, ldb->W, &ytDs));
            CHKERRQ(VecDotEnd(ldb->V, ldb->V, &ytDy));
            CHKERRQ(VecDotEnd(ldb->W, ldb->W, &stDs));

            yy_sum += PetscRealPart(ytDy);
            ys_sum += PetscRealPart(ytDs);
            ss_sum += PetscRealPart(stDs);
          }
        }

        if (0.0 == ldb->alpha) {
          /*  Safeguard ys_sum  */
          ys_sum = PetscMax(ldb->tol, ys_sum);

          sigma = ss_sum / ys_sum;
        } else if (1.0 == ldb->alpha) {
          /* yy_sum is never 0; if it were, we'd be at the minimum */
          sigma = ys_sum / yy_sum;
        } else {
          denom = 2.0*ldb->alpha*yy_sum;

          /*  Safeguard denom */
          denom = PetscMax(ldb->tol, denom);

          sigma = ((2.0*ldb->alpha-1)*ys_sum + PetscSqrtReal((2.0*ldb->alpha-1)*(2.0*ldb->alpha-1)*ys_sum*ys_sum - 4.0*ldb->alpha*(ldb->alpha-1)*yy_sum*ss_sum)) / denom;
        }
      } else {
        sigma = 1.0;
      }
      /*  If Q has small values, then Q^(r_beta - 1)
       can have very large values.  Hence, ys_sum
       and ss_sum can be infinity.  In this case,
       sigma can either be not-a-number or infinity. */

      if (PetscIsInfOrNanScalar(sigma)) {
        /*  sigma is not-a-number; skip rescaling */
      } else if (0.0 == sigma) {
        /*  sigma is zero; this is a bad case; skip rescaling */
      } else {
        /*  sigma is positive */
        CHKERRQ(VecScale(ldb->invDnew, sigma));
      }

      /* Combine the old diagonal and the new diagonal using a convex limiter */
      if (1.0 == ldb->rho) {
        CHKERRQ(VecCopy(ldb->invDnew, ldb->invD));
      } else if (ldb->rho) {
        CHKERRQ(VecAXPBY(ldb->invD, 1.0-ldb->rho, ldb->rho, ldb->invDnew));
      }
    } else {
      CHKERRQ(MatLMVMReset(B, PETSC_FALSE));
    }
    /* End DiagBrdn update */

  }
  /* Save the solution and function to be used in the next update */
  CHKERRQ(VecCopy(X, lmvm->Xprev));
  CHKERRQ(VecCopy(F, lmvm->Fprev));
  lmvm->prev_set = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatCopy_DiagBrdn(Mat B, Mat M, MatStructure str)
{
  Mat_LMVM          *bdata = (Mat_LMVM*)B->data;
  Mat_DiagBrdn      *bctx = (Mat_DiagBrdn*)bdata->ctx;
  Mat_LMVM          *mdata = (Mat_LMVM*)M->data;
  Mat_DiagBrdn      *mctx = (Mat_DiagBrdn*)mdata->ctx;
  PetscInt          i;

  PetscFunctionBegin;
  mctx->theta = bctx->theta;
  mctx->alpha = bctx->alpha;
  mctx->beta = bctx->beta;
  mctx->rho = bctx->rho;
  mctx->delta = bctx->delta;
  mctx->delta_min = bctx->delta_min;
  mctx->delta_max = bctx->delta_max;
  mctx->tol = bctx->tol;
  mctx->sigma = bctx->sigma;
  mctx->sigma_hist = bctx->sigma_hist;
  mctx->forward = bctx->forward;
  CHKERRQ(VecCopy(bctx->invD, mctx->invD));
  for (i=0; i<=bdata->k; ++i) {
    mctx->yty[i] = bctx->yty[i];
    mctx->yts[i] = bctx->yts[i];
    mctx->sts[i] = bctx->sts[i];
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatView_DiagBrdn(Mat B, PetscViewer pv)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_DiagBrdn      *ldb = (Mat_DiagBrdn*)lmvm->ctx;
  PetscBool         isascii;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)pv,PETSCVIEWERASCII,&isascii));
  if (isascii) {
    CHKERRQ(PetscViewerASCIIPrintf(pv,"Scale history: %d\n",ldb->sigma_hist));
    CHKERRQ(PetscViewerASCIIPrintf(pv,"Scale params: alpha=%g, beta=%g, rho=%g\n",(double)ldb->alpha, (double)ldb->beta, (double)ldb->rho));
    CHKERRQ(PetscViewerASCIIPrintf(pv,"Convex factor: theta=%g\n", (double)ldb->theta));
  }
  CHKERRQ(MatView_LMVM(B, pv));
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatSetFromOptions_DiagBrdn(PetscOptionItems *PetscOptionsObject, Mat B)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_DiagBrdn       *ldb = (Mat_DiagBrdn*)lmvm->ctx;

  PetscFunctionBegin;
  CHKERRQ(MatSetFromOptions_LMVM(PetscOptionsObject, B));
  CHKERRQ(PetscOptionsHead(PetscOptionsObject,"Restricted Broyden method for approximating SPD Jacobian actions (MATLMVMDIAGBRDN)"));
  CHKERRQ(PetscOptionsReal("-mat_lmvm_theta","(developer) convex ratio between BFGS and DFP components of the diagonal J0 scaling","",ldb->theta,&ldb->theta,NULL));
  CHKERRQ(PetscOptionsReal("-mat_lmvm_rho","(developer) update limiter in the J0 scaling","",ldb->rho,&ldb->rho,NULL));
  CHKERRQ(PetscOptionsReal("-mat_lmvm_tol","(developer) tolerance for bounding rescaling denominator","",ldb->tol,&ldb->tol,NULL));
  CHKERRQ(PetscOptionsReal("-mat_lmvm_alpha","(developer) convex ratio in the J0 scaling","",ldb->alpha,&ldb->alpha,NULL));
  CHKERRQ(PetscOptionsBool("-mat_lmvm_forward","Forward -> Update diagonal scaling for B. Else -> diagonal scaling for H.","",ldb->forward,&ldb->forward,NULL));
  CHKERRQ(PetscOptionsReal("-mat_lmvm_beta","(developer) exponential factor in the diagonal J0 scaling","",ldb->beta,&ldb->beta,NULL));
  CHKERRQ(PetscOptionsInt("-mat_lmvm_sigma_hist","(developer) number of past updates to use in the default J0 scalar","",ldb->sigma_hist,&ldb->sigma_hist,NULL));
  CHKERRQ(PetscOptionsTail());
  PetscCheckFalse((ldb->theta < 0.0) || (ldb->theta > 1.0),PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_OUTOFRANGE, "convex ratio for the diagonal J0 scale cannot be outside the range of [0, 1]");
  PetscCheckFalse((ldb->alpha < 0.0) || (ldb->alpha > 1.0),PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_OUTOFRANGE, "convex ratio in the J0 scaling cannot be outside the range of [0, 1]");
  PetscCheckFalse((ldb->rho < 0.0) || (ldb->rho > 1.0),PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_OUTOFRANGE, "convex update limiter in the J0 scaling cannot be outside the range of [0, 1]");
  PetscCheckFalse(ldb->sigma_hist < 0,PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_OUTOFRANGE, "J0 scaling history length cannot be negative");
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatReset_DiagBrdn(Mat B, PetscBool destructive)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_DiagBrdn      *ldb = (Mat_DiagBrdn*)lmvm->ctx;

  PetscFunctionBegin;
  CHKERRQ(VecSet(ldb->invD, ldb->delta));
  if (destructive && ldb->allocated) {
    CHKERRQ(PetscFree3(ldb->yty, ldb->yts, ldb->sts));
    CHKERRQ(VecDestroy(&ldb->invDnew));
    CHKERRQ(VecDestroy(&ldb->invD));
    CHKERRQ(VecDestroy(&ldb->BFGS));
    CHKERRQ(VecDestroy(&ldb->DFP));
    CHKERRQ(VecDestroy(&ldb->U));
    CHKERRQ(VecDestroy(&ldb->V));
    CHKERRQ(VecDestroy(&ldb->W));
    ldb->allocated = PETSC_FALSE;
  }
  CHKERRQ(MatReset_LMVM(B, destructive));
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatAllocate_DiagBrdn(Mat B, Vec X, Vec F)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_DiagBrdn      *ldb = (Mat_DiagBrdn*)lmvm->ctx;

  PetscFunctionBegin;
  CHKERRQ(MatAllocate_LMVM(B, X, F));
  if (!ldb->allocated) {
    CHKERRQ(PetscMalloc3(lmvm->m, &ldb->yty, lmvm->m, &ldb->yts, lmvm->m, &ldb->sts));
    CHKERRQ(VecDuplicate(lmvm->Xprev, &ldb->invDnew));
    CHKERRQ(VecDuplicate(lmvm->Xprev, &ldb->invD));
    CHKERRQ(VecDuplicate(lmvm->Xprev, &ldb->BFGS));
    CHKERRQ(VecDuplicate(lmvm->Xprev, &ldb->DFP));
    CHKERRQ(VecDuplicate(lmvm->Xprev, &ldb->U));
    CHKERRQ(VecDuplicate(lmvm->Xprev, &ldb->V));
    CHKERRQ(VecDuplicate(lmvm->Xprev, &ldb->W));
    ldb->allocated = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatDestroy_DiagBrdn(Mat B)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_DiagBrdn      *ldb = (Mat_DiagBrdn*)lmvm->ctx;

  PetscFunctionBegin;
  if (ldb->allocated) {
    CHKERRQ(PetscFree3(ldb->yty, ldb->yts, ldb->sts));
    CHKERRQ(VecDestroy(&ldb->invDnew));
    CHKERRQ(VecDestroy(&ldb->invD));
    CHKERRQ(VecDestroy(&ldb->BFGS));
    CHKERRQ(VecDestroy(&ldb->DFP));
    CHKERRQ(VecDestroy(&ldb->U));
    CHKERRQ(VecDestroy(&ldb->V));
    CHKERRQ(VecDestroy(&ldb->W));
    ldb->allocated = PETSC_FALSE;
  }
  CHKERRQ(PetscFree(lmvm->ctx));
  CHKERRQ(MatDestroy_LMVM(B));
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatSetUp_DiagBrdn(Mat B)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_DiagBrdn      *ldb = (Mat_DiagBrdn*)lmvm->ctx;

  PetscFunctionBegin;
  CHKERRQ(MatSetUp_LMVM(B));
  if (!ldb->allocated) {
    CHKERRQ(PetscMalloc3(lmvm->m, &ldb->yty, lmvm->m, &ldb->yts, lmvm->m, &ldb->sts));
    CHKERRQ(VecDuplicate(lmvm->Xprev, &ldb->invDnew));
    CHKERRQ(VecDuplicate(lmvm->Xprev, &ldb->invD));
    CHKERRQ(VecDuplicate(lmvm->Xprev, &ldb->BFGS));
    CHKERRQ(VecDuplicate(lmvm->Xprev, &ldb->DFP));
    CHKERRQ(VecDuplicate(lmvm->Xprev, &ldb->U));
    CHKERRQ(VecDuplicate(lmvm->Xprev, &ldb->V));
    CHKERRQ(VecDuplicate(lmvm->Xprev, &ldb->W));
    ldb->allocated = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode MatCreate_LMVMDiagBrdn(Mat B)
{
  Mat_LMVM          *lmvm;
  Mat_DiagBrdn      *ldb;

  PetscFunctionBegin;
  CHKERRQ(MatCreate_LMVM(B));
  CHKERRQ(PetscObjectChangeTypeName((PetscObject)B, MATLMVMDIAGBROYDEN));
  B->ops->setup = MatSetUp_DiagBrdn;
  B->ops->setfromoptions = MatSetFromOptions_DiagBrdn;
  B->ops->destroy = MatDestroy_DiagBrdn;
  B->ops->solve = MatSolve_DiagBrdn;
  B->ops->view = MatView_DiagBrdn;

  lmvm = (Mat_LMVM*)B->data;
  lmvm->square = PETSC_TRUE;
  lmvm->m = 1;
  lmvm->ops->allocate = MatAllocate_DiagBrdn;
  lmvm->ops->reset = MatReset_DiagBrdn;
  lmvm->ops->mult = MatMult_DiagBrdn;
  lmvm->ops->update = MatUpdate_DiagBrdn;
  lmvm->ops->copy = MatCopy_DiagBrdn;

  CHKERRQ(PetscNewLog(B, &ldb));
  lmvm->ctx = (void*)ldb;
  ldb->theta = 0.0;
  ldb->alpha = 1.0;
  ldb->rho = 1.0;
  ldb->forward = PETSC_TRUE;
  ldb->beta = 0.5;
  ldb->sigma = 1.0;
  ldb->delta = 1.0;
  ldb->delta_min = 1e-7;
  ldb->delta_max = 100.0;
  ldb->tol = 1e-8;
  ldb->sigma_hist = 1;
  ldb->allocated = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/*@
   MatCreateLMVMDiagBroyden - DiagBrdn creates a symmetric Broyden-type diagonal matrix used
   for approximating Hessians. It consists of a convex combination of DFP and BFGS
   diagonal approximation schemes, such that DiagBrdn = (1-theta)*BFGS + theta*DFP.
   To preserve symmetric positive-definiteness, we restrict theta to be in [0, 1].
   We also ensure positive definiteness by taking the VecAbs() of the final vector.

   There are two ways of approximating the diagonal: using the forward (B) update
   schemes for BFGS and DFP and then taking the inverse, or directly working with
   the inverse (H) update schemes for the BFGS and DFP updates, derived using the
   Sherman-Morrison-Woodbury formula. We have implemented both, controlled by a
   parameter below.

   In order to use the DiagBrdn matrix with other vector types, i.e. doing MatMults
   and MatSolves, the matrix must first be created using MatCreate() and MatSetType(),
   followed by MatLMVMAllocate(). Then it will be available for updating
   (via MatLMVMUpdate) in one's favored solver implementation.
   This allows for MPI compatibility.

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
+   -mat_lmvm_theta - (developer) convex ratio between BFGS and DFP components of the diagonal J0 scaling
.   -mat_lmvm_rho - (developer) update limiter for the J0 scaling
.   -mat_lmvm_alpha - (developer) coefficient factor for the quadratic subproblem in J0 scaling
.   -mat_lmvm_beta - (developer) exponential factor for the diagonal J0 scaling
.   -mat_lmvm_sigma_hist - (developer) number of past updates to use in J0 scaling.
.   -mat_lmvm_tol - (developer) tolerance for bounding the denominator of the rescaling away from 0.
-   -mat_lmvm_forward - (developer) whether or not to use the forward or backward Broyden update to the diagonal

   Level: intermediate

.seealso: MatCreate(), MATLMVM, MATLMVMDIAGBRDN, MatCreateLMVMDFP(), MatCreateLMVMSR1(),
          MatCreateLMVMBFGS(), MatCreateLMVMBrdn(), MatCreateLMVMSymBrdn()
@*/
PetscErrorCode MatCreateLMVMDiagBroyden(MPI_Comm comm, PetscInt n, PetscInt N, Mat *B)
{
  PetscFunctionBegin;
  CHKERRQ(MatCreate(comm, B));
  CHKERRQ(MatSetSizes(*B, n, n, N, N));
  CHKERRQ(MatSetType(*B, MATLMVMDIAGBROYDEN));
  CHKERRQ(MatSetUp(*B));
  PetscFunctionReturn(0);
}
