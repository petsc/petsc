#include <../src/tao/unconstrained/impls/ntr/ntrimpl.h>

#include <petscksp.h>

#define NTR_INIT_CONSTANT         0
#define NTR_INIT_DIRECTION        1
#define NTR_INIT_INTERPOLATION    2
#define NTR_INIT_TYPES            3

#define NTR_UPDATE_REDUCTION      0
#define NTR_UPDATE_INTERPOLATION  1
#define NTR_UPDATE_TYPES          2

static const char *NTR_INIT[64] = {"constant","direction","interpolation"};

static const char *NTR_UPDATE[64] = {"reduction","interpolation"};

/*
   TaoSolve_NTR - Implements Newton's Method with a trust region approach
   for solving unconstrained minimization problems.

   The basic algorithm is taken from MINPACK-2 (dstrn).

   TaoSolve_NTR computes a local minimizer of a twice differentiable function
   f by applying a trust region variant of Newton's method.  At each stage
   of the algorithm, we use the prconditioned conjugate gradient method to
   determine an approximate minimizer of the quadratic equation

        q(s) = <s, Hs + g>

   subject to the trust region constraint

        || s ||_M <= radius,

   where radius is the trust region radius and M is a symmetric positive
   definite matrix (the preconditioner).  Here g is the gradient and H
   is the Hessian matrix.

   Note:  TaoSolve_NTR MUST use the iterative solver KSPNASH, KSPSTCG,
          or KSPGLTR.  Thus, we set KSPNASH, KSPSTCG, or KSPGLTR in this
          routine regardless of what the user may have previously specified.
*/
static PetscErrorCode TaoSolve_NTR(Tao tao)
{
  TAO_NTR            *tr = (TAO_NTR *)tao->data;
  KSPType            ksp_type;
  PetscBool          is_nash,is_stcg,is_gltr,is_bfgs,is_jacobi,is_symmetric,sym_set;
  KSPConvergedReason ksp_reason;
  PC                 pc;
  PetscReal          fmin, ftrial, prered, actred, kappa, sigma, beta;
  PetscReal          tau, tau_1, tau_2, tau_max, tau_min, max_radius;
  PetscReal          f, gnorm;

  PetscReal          norm_d;
  PetscInt           bfgsUpdates = 0;
  PetscInt           needH;

  PetscInt           i_max = 5;
  PetscInt           j_max = 1;
  PetscInt           i, j, N, n, its;

  PetscFunctionBegin;
  if (tao->XL || tao->XU || tao->ops->computebounds) {
    PetscCall(PetscInfo(tao,"WARNING: Variable bounds have been set but will be ignored by ntr algorithm\n"));
  }

  PetscCall(KSPGetType(tao->ksp,&ksp_type));
  PetscCall(PetscStrcmp(ksp_type,KSPNASH,&is_nash));
  PetscCall(PetscStrcmp(ksp_type,KSPSTCG,&is_stcg));
  PetscCall(PetscStrcmp(ksp_type,KSPGLTR,&is_gltr));
  PetscCheck(is_nash || is_stcg || is_gltr,PetscObjectComm((PetscObject)tao),PETSC_ERR_SUP,"TAO_NTR requires nash, stcg, or gltr for the KSP");

  /* Initialize the radius and modify if it is too large or small */
  tao->trust = tao->trust0;
  tao->trust = PetscMax(tao->trust, tr->min_radius);
  tao->trust = PetscMin(tao->trust, tr->max_radius);

  /* Allocate the vectors needed for the BFGS approximation */
  PetscCall(KSPGetPC(tao->ksp, &pc));
  PetscCall(PetscObjectTypeCompare((PetscObject)pc, PCLMVM, &is_bfgs));
  PetscCall(PetscObjectTypeCompare((PetscObject)pc, PCJACOBI, &is_jacobi));
  if (is_bfgs) {
    tr->bfgs_pre = pc;
    PetscCall(PCLMVMGetMatLMVM(tr->bfgs_pre, &tr->M));
    PetscCall(VecGetLocalSize(tao->solution, &n));
    PetscCall(VecGetSize(tao->solution, &N));
    PetscCall(MatSetSizes(tr->M, n, n, N, N));
    PetscCall(MatLMVMAllocate(tr->M, tao->solution, tao->gradient));
    PetscCall(MatIsSymmetricKnown(tr->M, &sym_set, &is_symmetric));
    PetscCheck(sym_set && is_symmetric,PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_INCOMP, "LMVM matrix in the LMVM preconditioner must be symmetric.");
  } else if (is_jacobi) {
    PetscCall(PCJacobiSetUseAbs(pc,PETSC_TRUE));
  }

  /* Check convergence criteria */
  PetscCall(TaoComputeObjectiveAndGradient(tao, tao->solution, &f, tao->gradient));
  PetscCall(TaoGradientNorm(tao, tao->gradient,NORM_2,&gnorm));
  PetscCheck(!PetscIsInfOrNanReal(f) && !PetscIsInfOrNanReal(gnorm),PetscObjectComm((PetscObject)tao),PETSC_ERR_USER,"User provided compute function generated Inf or NaN");
  needH = 1;

  tao->reason = TAO_CONTINUE_ITERATING;
  PetscCall(TaoLogConvergenceHistory(tao,f,gnorm,0.0,tao->ksp_its));
  PetscCall(TaoMonitor(tao,tao->niter,f,gnorm,0.0,1.0));
  PetscCall((*tao->ops->convergencetest)(tao,tao->cnvP));
  if (tao->reason != TAO_CONTINUE_ITERATING) PetscFunctionReturn(0);

  /*  Initialize trust-region radius */
  switch (tr->init_type) {
  case NTR_INIT_CONSTANT:
    /*  Use the initial radius specified */
    break;

  case NTR_INIT_INTERPOLATION:
    /*  Use the initial radius specified */
    max_radius = 0.0;

    for (j = 0; j < j_max; ++j) {
      fmin = f;
      sigma = 0.0;

      if (needH) {
        PetscCall(TaoComputeHessian(tao,tao->solution,tao->hessian,tao->hessian_pre));
        needH = 0;
      }

      for (i = 0; i < i_max; ++i) {

        PetscCall(VecCopy(tao->solution, tr->W));
        PetscCall(VecAXPY(tr->W, -tao->trust/gnorm, tao->gradient));
        PetscCall(TaoComputeObjective(tao, tr->W, &ftrial));

        if (PetscIsInfOrNanReal(ftrial)) {
          tau = tr->gamma1_i;
        }
        else {
          if (ftrial < fmin) {
            fmin = ftrial;
            sigma = -tao->trust / gnorm;
          }

          PetscCall(MatMult(tao->hessian, tao->gradient, tao->stepdirection));
          PetscCall(VecDot(tao->gradient, tao->stepdirection, &prered));

          prered = tao->trust * (gnorm - 0.5 * tao->trust * prered / (gnorm * gnorm));
          actred = f - ftrial;
          if ((PetscAbsScalar(actred) <= tr->epsilon) &&
              (PetscAbsScalar(prered) <= tr->epsilon)) {
            kappa = 1.0;
          }
          else {
            kappa = actred / prered;
          }

          tau_1 = tr->theta_i * gnorm * tao->trust / (tr->theta_i * gnorm * tao->trust + (1.0 - tr->theta_i) * prered - actred);
          tau_2 = tr->theta_i * gnorm * tao->trust / (tr->theta_i * gnorm * tao->trust - (1.0 + tr->theta_i) * prered + actred);
          tau_min = PetscMin(tau_1, tau_2);
          tau_max = PetscMax(tau_1, tau_2);

          if (PetscAbsScalar(kappa - (PetscReal)1.0) <= tr->mu1_i) {
            /*  Great agreement */
            max_radius = PetscMax(max_radius, tao->trust);

            if (tau_max < 1.0) {
              tau = tr->gamma3_i;
            }
            else if (tau_max > tr->gamma4_i) {
              tau = tr->gamma4_i;
            }
            else {
              tau = tau_max;
            }
          }
          else if (PetscAbsScalar(kappa - (PetscReal)1.0) <= tr->mu2_i) {
            /*  Good agreement */
            max_radius = PetscMax(max_radius, tao->trust);

            if (tau_max < tr->gamma2_i) {
              tau = tr->gamma2_i;
            }
            else if (tau_max > tr->gamma3_i) {
              tau = tr->gamma3_i;
            }
            else {
              tau = tau_max;
            }
          }
          else {
            /*  Not good agreement */
            if (tau_min > 1.0) {
              tau = tr->gamma2_i;
            }
            else if (tau_max < tr->gamma1_i) {
              tau = tr->gamma1_i;
            }
            else if ((tau_min < tr->gamma1_i) && (tau_max >= 1.0)) {
              tau = tr->gamma1_i;
            }
            else if ((tau_1 >= tr->gamma1_i) && (tau_1 < 1.0) &&
                     ((tau_2 < tr->gamma1_i) || (tau_2 >= 1.0))) {
              tau = tau_1;
            }
            else if ((tau_2 >= tr->gamma1_i) && (tau_2 < 1.0) &&
                     ((tau_1 < tr->gamma1_i) || (tau_2 >= 1.0))) {
              tau = tau_2;
            }
            else {
              tau = tau_max;
            }
          }
        }
        tao->trust = tau * tao->trust;
      }

      if (fmin < f) {
        f = fmin;
        PetscCall(VecAXPY(tao->solution, sigma, tao->gradient));
        PetscCall(TaoComputeGradient(tao,tao->solution, tao->gradient));

        PetscCall(TaoGradientNorm(tao, tao->gradient,NORM_2,&gnorm));
        PetscCheck(!PetscIsInfOrNanReal(f) && !PetscIsInfOrNanReal(gnorm),PetscObjectComm((PetscObject)tao),PETSC_ERR_USER, "User provided compute function generated Inf or NaN");
        needH = 1;

        PetscCall(TaoLogConvergenceHistory(tao,f,gnorm,0.0,tao->ksp_its));
        PetscCall(TaoMonitor(tao,tao->niter,f,gnorm,0.0,1.0));
        PetscCall((*tao->ops->convergencetest)(tao,tao->cnvP));
        if (tao->reason != TAO_CONTINUE_ITERATING) {
          PetscFunctionReturn(0);
        }
      }
    }
    tao->trust = PetscMax(tao->trust, max_radius);

    /*  Modify the radius if it is too large or small */
    tao->trust = PetscMax(tao->trust, tr->min_radius);
    tao->trust = PetscMin(tao->trust, tr->max_radius);
    break;

  default:
    /*  Norm of the first direction will initialize radius */
    tao->trust = 0.0;
    break;
  }

  /* Have not converged; continue with Newton method */
  while (tao->reason == TAO_CONTINUE_ITERATING) {
    /* Call general purpose update function */
    if (tao->ops->update) {
      PetscCall((*tao->ops->update)(tao, tao->niter, tao->user_update));
    }
    ++tao->niter;
    tao->ksp_its=0;
    /* Compute the Hessian */
    if (needH) {
      PetscCall(TaoComputeHessian(tao,tao->solution,tao->hessian,tao->hessian_pre));
      needH = 0;
    }

    if (tr->bfgs_pre) {
      /* Update the limited memory preconditioner */
      PetscCall(MatLMVMUpdate(tr->M, tao->solution, tao->gradient));
      ++bfgsUpdates;
    }

    while (tao->reason == TAO_CONTINUE_ITERATING) {
      PetscCall(KSPSetOperators(tao->ksp, tao->hessian, tao->hessian_pre));

      /* Solve the trust region subproblem */
      PetscCall(KSPCGSetRadius(tao->ksp,tao->trust));
      PetscCall(KSPSolve(tao->ksp, tao->gradient, tao->stepdirection));
      PetscCall(KSPGetIterationNumber(tao->ksp,&its));
      tao->ksp_its+=its;
      tao->ksp_tot_its+=its;
      PetscCall(KSPCGGetNormD(tao->ksp, &norm_d));

      if (0.0 == tao->trust) {
        /* Radius was uninitialized; use the norm of the direction */
        if (norm_d > 0.0) {
          tao->trust = norm_d;

          /* Modify the radius if it is too large or small */
          tao->trust = PetscMax(tao->trust, tr->min_radius);
          tao->trust = PetscMin(tao->trust, tr->max_radius);
        }
        else {
          /* The direction was bad; set radius to default value and re-solve
             the trust-region subproblem to get a direction */
          tao->trust = tao->trust0;

          /* Modify the radius if it is too large or small */
          tao->trust = PetscMax(tao->trust, tr->min_radius);
          tao->trust = PetscMin(tao->trust, tr->max_radius);

          PetscCall(KSPCGSetRadius(tao->ksp,tao->trust));
          PetscCall(KSPSolve(tao->ksp, tao->gradient, tao->stepdirection));
          PetscCall(KSPGetIterationNumber(tao->ksp,&its));
          tao->ksp_its+=its;
          tao->ksp_tot_its+=its;
          PetscCall(KSPCGGetNormD(tao->ksp, &norm_d));

          PetscCheck(norm_d != 0.0,PetscObjectComm((PetscObject)tao),PETSC_ERR_PLIB, "Initial direction zero");
        }
      }
      PetscCall(VecScale(tao->stepdirection, -1.0));
      PetscCall(KSPGetConvergedReason(tao->ksp, &ksp_reason));
      if ((KSP_DIVERGED_INDEFINITE_PC == ksp_reason) && (tr->bfgs_pre)) {
        /* Preconditioner is numerically indefinite; reset the
           approximate if using BFGS preconditioning. */
        PetscCall(MatLMVMReset(tr->M, PETSC_FALSE));
        PetscCall(MatLMVMUpdate(tr->M, tao->solution, tao->gradient));
        bfgsUpdates = 1;
      }

      if (NTR_UPDATE_REDUCTION == tr->update_type) {
        /* Get predicted reduction */
        PetscCall(KSPCGGetObjFcn(tao->ksp,&prered));
        if (prered >= 0.0) {
          /* The predicted reduction has the wrong sign.  This cannot
             happen in infinite precision arithmetic.  Step should
             be rejected! */
          tao->trust = tr->alpha1 * PetscMin(tao->trust, norm_d);
        }
        else {
          /* Compute trial step and function value */
          PetscCall(VecCopy(tao->solution,tr->W));
          PetscCall(VecAXPY(tr->W, 1.0, tao->stepdirection));
          PetscCall(TaoComputeObjective(tao, tr->W, &ftrial));

          if (PetscIsInfOrNanReal(ftrial)) {
            tao->trust = tr->alpha1 * PetscMin(tao->trust, norm_d);
          } else {
            /* Compute and actual reduction */
            actred = f - ftrial;
            prered = -prered;
            if ((PetscAbsScalar(actred) <= tr->epsilon) &&
                (PetscAbsScalar(prered) <= tr->epsilon)) {
              kappa = 1.0;
            }
            else {
              kappa = actred / prered;
            }

            /* Accept or reject the step and update radius */
            if (kappa < tr->eta1) {
              /* Reject the step */
              tao->trust = tr->alpha1 * PetscMin(tao->trust, norm_d);
            }
            else {
              /* Accept the step */
              if (kappa < tr->eta2) {
                /* Marginal bad step */
                tao->trust = tr->alpha2 * PetscMin(tao->trust, norm_d);
              }
              else if (kappa < tr->eta3) {
                /* Reasonable step */
                tao->trust = tr->alpha3 * tao->trust;
              }
              else if (kappa < tr->eta4) {
                /* Good step */
                tao->trust = PetscMax(tr->alpha4 * norm_d, tao->trust);
              }
              else {
                /* Very good step */
                tao->trust = PetscMax(tr->alpha5 * norm_d, tao->trust);
              }
              break;
            }
          }
        }
      }
      else {
        /* Get predicted reduction */
        PetscCall(KSPCGGetObjFcn(tao->ksp,&prered));
        if (prered >= 0.0) {
          /* The predicted reduction has the wrong sign.  This cannot
             happen in infinite precision arithmetic.  Step should
             be rejected! */
          tao->trust = tr->gamma1 * PetscMin(tao->trust, norm_d);
        }
        else {
          PetscCall(VecCopy(tao->solution, tr->W));
          PetscCall(VecAXPY(tr->W, 1.0, tao->stepdirection));
          PetscCall(TaoComputeObjective(tao, tr->W, &ftrial));
          if (PetscIsInfOrNanReal(ftrial)) {
            tao->trust = tr->gamma1 * PetscMin(tao->trust, norm_d);
          }
          else {
            PetscCall(VecDot(tao->gradient, tao->stepdirection, &beta));
            actred = f - ftrial;
            prered = -prered;
            if ((PetscAbsScalar(actred) <= tr->epsilon) &&
                (PetscAbsScalar(prered) <= tr->epsilon)) {
              kappa = 1.0;
            }
            else {
              kappa = actred / prered;
            }

            tau_1 = tr->theta * beta / (tr->theta * beta - (1.0 - tr->theta) * prered + actred);
            tau_2 = tr->theta * beta / (tr->theta * beta + (1.0 + tr->theta) * prered - actred);
            tau_min = PetscMin(tau_1, tau_2);
            tau_max = PetscMax(tau_1, tau_2);

            if (kappa >= 1.0 - tr->mu1) {
              /* Great agreement; accept step and update radius */
              if (tau_max < 1.0) {
                tao->trust = PetscMax(tao->trust, tr->gamma3 * norm_d);
              }
              else if (tau_max > tr->gamma4) {
                tao->trust = PetscMax(tao->trust, tr->gamma4 * norm_d);
              }
              else {
                tao->trust = PetscMax(tao->trust, tau_max * norm_d);
              }
              break;
            }
            else if (kappa >= 1.0 - tr->mu2) {
              /* Good agreement */

              if (tau_max < tr->gamma2) {
                tao->trust = tr->gamma2 * PetscMin(tao->trust, norm_d);
              }
              else if (tau_max > tr->gamma3) {
                tao->trust = PetscMax(tao->trust, tr->gamma3 * norm_d);
              }
              else if (tau_max < 1.0) {
                tao->trust = tau_max * PetscMin(tao->trust, norm_d);
              }
              else {
                tao->trust = PetscMax(tao->trust, tau_max * norm_d);
              }
              break;
            }
            else {
              /* Not good agreement */
              if (tau_min > 1.0) {
                tao->trust = tr->gamma2 * PetscMin(tao->trust, norm_d);
              }
              else if (tau_max < tr->gamma1) {
                tao->trust = tr->gamma1 * PetscMin(tao->trust, norm_d);
              }
              else if ((tau_min < tr->gamma1) && (tau_max >= 1.0)) {
                tao->trust = tr->gamma1 * PetscMin(tao->trust, norm_d);
              }
              else if ((tau_1 >= tr->gamma1) && (tau_1 < 1.0) &&
                       ((tau_2 < tr->gamma1) || (tau_2 >= 1.0))) {
                tao->trust = tau_1 * PetscMin(tao->trust, norm_d);
              }
              else if ((tau_2 >= tr->gamma1) && (tau_2 < 1.0) &&
                       ((tau_1 < tr->gamma1) || (tau_2 >= 1.0))) {
                tao->trust = tau_2 * PetscMin(tao->trust, norm_d);
              }
              else {
                tao->trust = tau_max * PetscMin(tao->trust, norm_d);
              }
            }
          }
        }
      }

      /* The step computed was not good and the radius was decreased.
         Monitor the radius to terminate. */
      PetscCall(TaoLogConvergenceHistory(tao,f,gnorm,0.0,tao->ksp_its));
      PetscCall(TaoMonitor(tao,tao->niter,f,gnorm,0.0,tao->trust));
      PetscCall((*tao->ops->convergencetest)(tao,tao->cnvP));
    }

    /* The radius may have been increased; modify if it is too large */
    tao->trust = PetscMin(tao->trust, tr->max_radius);

    if (tao->reason == TAO_CONTINUE_ITERATING) {
      PetscCall(VecCopy(tr->W, tao->solution));
      f = ftrial;
      PetscCall(TaoComputeGradient(tao, tao->solution, tao->gradient));
      PetscCall(TaoGradientNorm(tao, tao->gradient,NORM_2,&gnorm));
      PetscCheck(!PetscIsInfOrNanReal(f) && !PetscIsInfOrNanReal(gnorm),PetscObjectComm((PetscObject)tao),PETSC_ERR_USER, "User provided compute function generated Inf or NaN");
      needH = 1;
      PetscCall(TaoLogConvergenceHistory(tao,f,gnorm,0.0,tao->ksp_its));
      PetscCall(TaoMonitor(tao,tao->niter,f,gnorm,0.0,tao->trust));
      PetscCall((*tao->ops->convergencetest)(tao,tao->cnvP));
    }
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
static PetscErrorCode TaoSetUp_NTR(Tao tao)
{
  TAO_NTR *tr = (TAO_NTR *)tao->data;

  PetscFunctionBegin;
  if (!tao->gradient) PetscCall(VecDuplicate(tao->solution, &tao->gradient));
  if (!tao->stepdirection) PetscCall(VecDuplicate(tao->solution, &tao->stepdirection));
  if (!tr->W) PetscCall(VecDuplicate(tao->solution, &tr->W));

  tr->bfgs_pre = NULL;
  tr->M = NULL;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
static PetscErrorCode TaoDestroy_NTR(Tao tao)
{
  TAO_NTR        *tr = (TAO_NTR *)tao->data;

  PetscFunctionBegin;
  if (tao->setupcalled) {
    PetscCall(VecDestroy(&tr->W));
  }
  PetscCall(PetscFree(tao->data));
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
static PetscErrorCode TaoSetFromOptions_NTR(PetscOptionItems *PetscOptionsObject,Tao tao)
{
  TAO_NTR        *tr = (TAO_NTR *)tao->data;

  PetscFunctionBegin;
  PetscCall(PetscOptionsHead(PetscOptionsObject,"Newton trust region method for unconstrained optimization"));
  PetscCall(PetscOptionsEList("-tao_ntr_init_type", "tao->trust initialization type", "", NTR_INIT, NTR_INIT_TYPES, NTR_INIT[tr->init_type], &tr->init_type,NULL));
  PetscCall(PetscOptionsEList("-tao_ntr_update_type", "radius update type", "", NTR_UPDATE, NTR_UPDATE_TYPES, NTR_UPDATE[tr->update_type], &tr->update_type,NULL));
  PetscCall(PetscOptionsReal("-tao_ntr_eta1", "step is unsuccessful if actual reduction < eta1 * predicted reduction", "", tr->eta1, &tr->eta1,NULL));
  PetscCall(PetscOptionsReal("-tao_ntr_eta2", "", "", tr->eta2, &tr->eta2,NULL));
  PetscCall(PetscOptionsReal("-tao_ntr_eta3", "", "", tr->eta3, &tr->eta3,NULL));
  PetscCall(PetscOptionsReal("-tao_ntr_eta4", "", "", tr->eta4, &tr->eta4,NULL));
  PetscCall(PetscOptionsReal("-tao_ntr_alpha1", "", "", tr->alpha1, &tr->alpha1,NULL));
  PetscCall(PetscOptionsReal("-tao_ntr_alpha2", "", "", tr->alpha2, &tr->alpha2,NULL));
  PetscCall(PetscOptionsReal("-tao_ntr_alpha3", "", "", tr->alpha3, &tr->alpha3,NULL));
  PetscCall(PetscOptionsReal("-tao_ntr_alpha4", "", "", tr->alpha4, &tr->alpha4,NULL));
  PetscCall(PetscOptionsReal("-tao_ntr_alpha5", "", "", tr->alpha5, &tr->alpha5,NULL));
  PetscCall(PetscOptionsReal("-tao_ntr_mu1", "", "", tr->mu1, &tr->mu1,NULL));
  PetscCall(PetscOptionsReal("-tao_ntr_mu2", "", "", tr->mu2, &tr->mu2,NULL));
  PetscCall(PetscOptionsReal("-tao_ntr_gamma1", "", "", tr->gamma1, &tr->gamma1,NULL));
  PetscCall(PetscOptionsReal("-tao_ntr_gamma2", "", "", tr->gamma2, &tr->gamma2,NULL));
  PetscCall(PetscOptionsReal("-tao_ntr_gamma3", "", "", tr->gamma3, &tr->gamma3,NULL));
  PetscCall(PetscOptionsReal("-tao_ntr_gamma4", "", "", tr->gamma4, &tr->gamma4,NULL));
  PetscCall(PetscOptionsReal("-tao_ntr_theta", "", "", tr->theta, &tr->theta,NULL));
  PetscCall(PetscOptionsReal("-tao_ntr_mu1_i", "", "", tr->mu1_i, &tr->mu1_i,NULL));
  PetscCall(PetscOptionsReal("-tao_ntr_mu2_i", "", "", tr->mu2_i, &tr->mu2_i,NULL));
  PetscCall(PetscOptionsReal("-tao_ntr_gamma1_i", "", "", tr->gamma1_i, &tr->gamma1_i,NULL));
  PetscCall(PetscOptionsReal("-tao_ntr_gamma2_i", "", "", tr->gamma2_i, &tr->gamma2_i,NULL));
  PetscCall(PetscOptionsReal("-tao_ntr_gamma3_i", "", "", tr->gamma3_i, &tr->gamma3_i,NULL));
  PetscCall(PetscOptionsReal("-tao_ntr_gamma4_i", "", "", tr->gamma4_i, &tr->gamma4_i,NULL));
  PetscCall(PetscOptionsReal("-tao_ntr_theta_i", "", "", tr->theta_i, &tr->theta_i,NULL));
  PetscCall(PetscOptionsReal("-tao_ntr_min_radius", "lower bound on initial trust-region radius", "", tr->min_radius, &tr->min_radius,NULL));
  PetscCall(PetscOptionsReal("-tao_ntr_max_radius", "upper bound on trust-region radius", "", tr->max_radius, &tr->max_radius,NULL));
  PetscCall(PetscOptionsReal("-tao_ntr_epsilon", "tolerance used when computing actual and predicted reduction", "", tr->epsilon, &tr->epsilon,NULL));
  PetscCall(PetscOptionsTail());
  PetscCall(KSPSetFromOptions(tao->ksp));
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
/*MC
  TAONTR - Newton's method with trust region for unconstrained minimization.
  At each iteration, the Newton trust region method solves the system.
  NTR expects a KSP solver with a trust region radius.
            min_d  .5 dT Hk d + gkT d,  s.t.   ||d|| < Delta_k

  Options Database Keys:
+ -tao_ntr_init_type - "constant","direction","interpolation"
. -tao_ntr_update_type - "reduction","interpolation"
. -tao_ntr_min_radius - lower bound on trust region radius
. -tao_ntr_max_radius - upper bound on trust region radius
. -tao_ntr_epsilon - tolerance for accepting actual / predicted reduction
. -tao_ntr_mu1_i - mu1 interpolation init factor
. -tao_ntr_mu2_i - mu2 interpolation init factor
. -tao_ntr_gamma1_i - gamma1 interpolation init factor
. -tao_ntr_gamma2_i - gamma2 interpolation init factor
. -tao_ntr_gamma3_i - gamma3 interpolation init factor
. -tao_ntr_gamma4_i - gamma4 interpolation init factor
. -tao_ntr_theta_i - theta1 interpolation init factor
. -tao_ntr_eta1 - eta1 reduction update factor
. -tao_ntr_eta2 - eta2 reduction update factor
. -tao_ntr_eta3 - eta3 reduction update factor
. -tao_ntr_eta4 - eta4 reduction update factor
. -tao_ntr_alpha1 - alpha1 reduction update factor
. -tao_ntr_alpha2 - alpha2 reduction update factor
. -tao_ntr_alpha3 - alpha3 reduction update factor
. -tao_ntr_alpha4 - alpha4 reduction update factor
. -tao_ntr_alpha4 - alpha4 reduction update factor
. -tao_ntr_mu1 - mu1 interpolation update
. -tao_ntr_mu2 - mu2 interpolation update
. -tao_ntr_gamma1 - gamma1 interpolcation update
. -tao_ntr_gamma2 - gamma2 interpolcation update
. -tao_ntr_gamma3 - gamma3 interpolcation update
. -tao_ntr_gamma4 - gamma4 interpolation update
- -tao_ntr_theta - theta interpolation update

  Level: beginner
M*/

PETSC_EXTERN PetscErrorCode TaoCreate_NTR(Tao tao)
{
  TAO_NTR *tr;

  PetscFunctionBegin;

  PetscCall(PetscNewLog(tao,&tr));

  tao->ops->setup = TaoSetUp_NTR;
  tao->ops->solve = TaoSolve_NTR;
  tao->ops->setfromoptions = TaoSetFromOptions_NTR;
  tao->ops->destroy = TaoDestroy_NTR;

  /* Override default settings (unless already changed) */
  if (!tao->max_it_changed) tao->max_it = 50;
  if (!tao->trust0_changed) tao->trust0 = 100.0;
  tao->data = (void*)tr;

  /*  Standard trust region update parameters */
  tr->eta1 = 1.0e-4;
  tr->eta2 = 0.25;
  tr->eta3 = 0.50;
  tr->eta4 = 0.90;

  tr->alpha1 = 0.25;
  tr->alpha2 = 0.50;
  tr->alpha3 = 1.00;
  tr->alpha4 = 2.00;
  tr->alpha5 = 4.00;

  /*  Interpolation trust region update parameters */
  tr->mu1 = 0.10;
  tr->mu2 = 0.50;

  tr->gamma1 = 0.25;
  tr->gamma2 = 0.50;
  tr->gamma3 = 2.00;
  tr->gamma4 = 4.00;

  tr->theta = 0.05;

  /*  Interpolation parameters for initialization */
  tr->mu1_i = 0.35;
  tr->mu2_i = 0.50;

  tr->gamma1_i = 0.0625;
  tr->gamma2_i = 0.50;
  tr->gamma3_i = 2.00;
  tr->gamma4_i = 5.00;

  tr->theta_i = 0.25;

  tr->min_radius = 1.0e-10;
  tr->max_radius = 1.0e10;
  tr->epsilon    = 1.0e-6;

  tr->init_type       = NTR_INIT_INTERPOLATION;
  tr->update_type     = NTR_UPDATE_REDUCTION;

  /* Set linear solver to default for trust region */
  PetscCall(KSPCreate(((PetscObject)tao)->comm,&tao->ksp));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)tao->ksp,(PetscObject)tao,1));
  PetscCall(KSPSetOptionsPrefix(tao->ksp,tao->hdr.prefix));
  PetscCall(KSPAppendOptionsPrefix(tao->ksp,"tao_ntr_"));
  PetscCall(KSPSetType(tao->ksp,KSPSTCG));
  PetscFunctionReturn(0);
}
