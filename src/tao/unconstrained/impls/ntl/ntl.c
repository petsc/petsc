#include <../src/tao/unconstrained/impls/ntl/ntlimpl.h>

#include <petscksp.h>

#define NTL_INIT_CONSTANT         0
#define NTL_INIT_DIRECTION        1
#define NTL_INIT_INTERPOLATION    2
#define NTL_INIT_TYPES            3

#define NTL_UPDATE_REDUCTION      0
#define NTL_UPDATE_INTERPOLATION  1
#define NTL_UPDATE_TYPES          2

static const char *NTL_INIT[64] = {"constant","direction","interpolation"};

static const char *NTL_UPDATE[64] = {"reduction","interpolation"};

/* Implements Newton's Method with a trust-region, line-search approach for
   solving unconstrained minimization problems.  A More'-Thuente line search
   is used to guarantee that the bfgs preconditioner remains positive
   definite. */

#define NTL_NEWTON              0
#define NTL_BFGS                1
#define NTL_SCALED_GRADIENT     2
#define NTL_GRADIENT            3

static PetscErrorCode TaoSolve_NTL(Tao tao)
{
  TAO_NTL                      *tl = (TAO_NTL *)tao->data;
  KSPType                      ksp_type;
  PetscBool                    is_nash,is_stcg,is_gltr,is_bfgs,is_jacobi,is_symmetric,sym_set;
  KSPConvergedReason           ksp_reason;
  PC                           pc;
  TaoLineSearchConvergedReason ls_reason;

  PetscReal                    fmin, ftrial, prered, actred, kappa, sigma;
  PetscReal                    tau, tau_1, tau_2, tau_max, tau_min, max_radius;
  PetscReal                    f, fold, gdx, gnorm;
  PetscReal                    step = 1.0;

  PetscReal                    norm_d = 0.0;
  PetscInt                     stepType;
  PetscInt                     its;

  PetscInt                     bfgsUpdates = 0;
  PetscInt                     needH;

  PetscInt                     i_max = 5;
  PetscInt                     j_max = 1;
  PetscInt                     i, j, n, N;

  PetscInt                     tr_reject;

  PetscFunctionBegin;
  if (tao->XL || tao->XU || tao->ops->computebounds) {
    PetscCall(PetscInfo(tao,"WARNING: Variable bounds have been set but will be ignored by ntl algorithm\n"));
  }

  PetscCall(KSPGetType(tao->ksp,&ksp_type));
  PetscCall(PetscStrcmp(ksp_type,KSPNASH,&is_nash));
  PetscCall(PetscStrcmp(ksp_type,KSPSTCG,&is_stcg));
  PetscCall(PetscStrcmp(ksp_type,KSPGLTR,&is_gltr));
  PetscCheck(is_nash || is_stcg || is_gltr,PetscObjectComm((PetscObject)tao),PETSC_ERR_SUP,"TAO_NTR requires nash, stcg, or gltr for the KSP");

  /* Initialize the radius and modify if it is too large or small */
  tao->trust = tao->trust0;
  tao->trust = PetscMax(tao->trust, tl->min_radius);
  tao->trust = PetscMin(tao->trust, tl->max_radius);

  /* Allocate the vectors needed for the BFGS approximation */
  PetscCall(KSPGetPC(tao->ksp, &pc));
  PetscCall(PetscObjectTypeCompare((PetscObject)pc, PCLMVM, &is_bfgs));
  PetscCall(PetscObjectTypeCompare((PetscObject)pc, PCJACOBI, &is_jacobi));
  if (is_bfgs) {
    tl->bfgs_pre = pc;
    PetscCall(PCLMVMGetMatLMVM(tl->bfgs_pre, &tl->M));
    PetscCall(VecGetLocalSize(tao->solution, &n));
    PetscCall(VecGetSize(tao->solution, &N));
    PetscCall(MatSetSizes(tl->M, n, n, N, N));
    PetscCall(MatLMVMAllocate(tl->M, tao->solution, tao->gradient));
    PetscCall(MatIsSymmetricKnown(tl->M, &sym_set, &is_symmetric));
    PetscCheck(sym_set && is_symmetric,PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_INCOMP, "LMVM matrix in the LMVM preconditioner must be symmetric.");
  } else if (is_jacobi) {
    PetscCall(PCJacobiSetUseAbs(pc,PETSC_TRUE));
  }

  /* Check convergence criteria */
  PetscCall(TaoComputeObjectiveAndGradient(tao, tao->solution, &f, tao->gradient));
  PetscCall(VecNorm(tao->gradient, NORM_2, &gnorm));
  PetscCheck(!PetscIsInfOrNanReal(f) && !PetscIsInfOrNanReal(gnorm),PetscObjectComm((PetscObject)tao),PETSC_ERR_USER, "User provided compute function generated Inf or NaN");
  needH = 1;

  tao->reason = TAO_CONTINUE_ITERATING;
  PetscCall(TaoLogConvergenceHistory(tao,f,gnorm,0.0,tao->ksp_its));
  PetscCall(TaoMonitor(tao,tao->niter,f,gnorm,0.0,step));
  PetscCall((*tao->ops->convergencetest)(tao,tao->cnvP));
  if (tao->reason != TAO_CONTINUE_ITERATING) PetscFunctionReturn(0);

  /* Initialize trust-region radius */
  switch(tl->init_type) {
  case NTL_INIT_CONSTANT:
    /* Use the initial radius specified */
    break;

  case NTL_INIT_INTERPOLATION:
    /* Use the initial radius specified */
    max_radius = 0.0;

    for (j = 0; j < j_max; ++j) {
      fmin = f;
      sigma = 0.0;

      if (needH) {
        PetscCall(TaoComputeHessian(tao,tao->solution,tao->hessian,tao->hessian_pre));
        needH = 0;
      }

      for (i = 0; i < i_max; ++i) {
        PetscCall(VecCopy(tao->solution, tl->W));
        PetscCall(VecAXPY(tl->W, -tao->trust/gnorm, tao->gradient));

        PetscCall(TaoComputeObjective(tao, tl->W, &ftrial));
        if (PetscIsInfOrNanReal(ftrial)) {
          tau = tl->gamma1_i;
        } else {
          if (ftrial < fmin) {
            fmin = ftrial;
            sigma = -tao->trust / gnorm;
          }

          PetscCall(MatMult(tao->hessian, tao->gradient, tao->stepdirection));
          PetscCall(VecDot(tao->gradient, tao->stepdirection, &prered));

          prered = tao->trust * (gnorm - 0.5 * tao->trust * prered / (gnorm * gnorm));
          actred = f - ftrial;
          if ((PetscAbsScalar(actred) <= tl->epsilon) && (PetscAbsScalar(prered) <= tl->epsilon)) {
            kappa = 1.0;
          } else {
            kappa = actred / prered;
          }

          tau_1 = tl->theta_i * gnorm * tao->trust / (tl->theta_i * gnorm * tao->trust + (1.0 - tl->theta_i) * prered - actred);
          tau_2 = tl->theta_i * gnorm * tao->trust / (tl->theta_i * gnorm * tao->trust - (1.0 + tl->theta_i) * prered + actred);
          tau_min = PetscMin(tau_1, tau_2);
          tau_max = PetscMax(tau_1, tau_2);

          if (PetscAbsScalar(kappa - (PetscReal)1.0) <= tl->mu1_i) {
            /* Great agreement */
            max_radius = PetscMax(max_radius, tao->trust);

            if (tau_max < 1.0) {
              tau = tl->gamma3_i;
            } else if (tau_max > tl->gamma4_i) {
              tau = tl->gamma4_i;
            } else if (tau_1 >= 1.0 && tau_1 <= tl->gamma4_i && tau_2 < 1.0) {
              tau = tau_1;
            } else if (tau_2 >= 1.0 && tau_2 <= tl->gamma4_i && tau_1 < 1.0) {
              tau = tau_2;
            } else {
              tau = tau_max;
            }
          } else if (PetscAbsScalar(kappa - (PetscReal)1.0) <= tl->mu2_i) {
            /* Good agreement */
            max_radius = PetscMax(max_radius, tao->trust);

            if (tau_max < tl->gamma2_i) {
              tau = tl->gamma2_i;
            } else if (tau_max > tl->gamma3_i) {
              tau = tl->gamma3_i;
            } else {
              tau = tau_max;
            }
          } else {
            /* Not good agreement */
            if (tau_min > 1.0) {
              tau = tl->gamma2_i;
            } else if (tau_max < tl->gamma1_i) {
              tau = tl->gamma1_i;
            } else if ((tau_min < tl->gamma1_i) && (tau_max >= 1.0)) {
              tau = tl->gamma1_i;
            } else if ((tau_1 >= tl->gamma1_i) && (tau_1 < 1.0) &&  ((tau_2 < tl->gamma1_i) || (tau_2 >= 1.0))) {
              tau = tau_1;
            } else if ((tau_2 >= tl->gamma1_i) && (tau_2 < 1.0) &&  ((tau_1 < tl->gamma1_i) || (tau_2 >= 1.0))) {
              tau = tau_2;
            } else {
              tau = tau_max;
            }
          }
        }
        tao->trust = tau * tao->trust;
      }

      if (fmin < f) {
        f = fmin;
        PetscCall(VecAXPY(tao->solution, sigma, tao->gradient));
        PetscCall(TaoComputeGradient(tao, tao->solution, tao->gradient));

        PetscCall(VecNorm(tao->gradient, NORM_2, &gnorm));
        PetscCheck(!PetscIsInfOrNanReal(f) && !PetscIsInfOrNanReal(gnorm),PetscObjectComm((PetscObject)tao),PETSC_ERR_USER, "User provided compute function generated Inf or NaN");
        needH = 1;

        PetscCall(TaoLogConvergenceHistory(tao,f,gnorm,0.0,tao->ksp_its));
        PetscCall(TaoMonitor(tao,tao->niter,f,gnorm,0.0,step));
        PetscCall((*tao->ops->convergencetest)(tao,tao->cnvP));
        if (tao->reason != TAO_CONTINUE_ITERATING) PetscFunctionReturn(0);
      }
    }
    tao->trust = PetscMax(tao->trust, max_radius);

    /* Modify the radius if it is too large or small */
    tao->trust = PetscMax(tao->trust, tl->min_radius);
    tao->trust = PetscMin(tao->trust, tl->max_radius);
    break;

  default:
    /* Norm of the first direction will initialize radius */
    tao->trust = 0.0;
    break;
  }

  /* Set counter for gradient/reset steps */
  tl->ntrust = 0;
  tl->newt = 0;
  tl->bfgs = 0;
  tl->grad = 0;

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
    }

    if (tl->bfgs_pre) {
      /* Update the limited memory preconditioner */
      PetscCall(MatLMVMUpdate(tl->M,tao->solution, tao->gradient));
      ++bfgsUpdates;
    }
    PetscCall(KSPSetOperators(tao->ksp, tao->hessian, tao->hessian_pre));
    /* Solve the Newton system of equations */
    PetscCall(KSPCGSetRadius(tao->ksp,tl->max_radius));
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
        tao->trust = PetscMax(tao->trust, tl->min_radius);
        tao->trust = PetscMin(tao->trust, tl->max_radius);
      } else {
        /* The direction was bad; set radius to default value and re-solve
           the trust-region subproblem to get a direction */
        tao->trust = tao->trust0;

        /* Modify the radius if it is too large or small */
        tao->trust = PetscMax(tao->trust, tl->min_radius);
        tao->trust = PetscMin(tao->trust, tl->max_radius);

        PetscCall(KSPCGSetRadius(tao->ksp,tl->max_radius));
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
    if ((KSP_DIVERGED_INDEFINITE_PC == ksp_reason) && (tl->bfgs_pre)) {
      /* Preconditioner is numerically indefinite; reset the
         approximate if using BFGS preconditioning. */
      PetscCall(MatLMVMReset(tl->M, PETSC_FALSE));
      PetscCall(MatLMVMUpdate(tl->M, tao->solution, tao->gradient));
      bfgsUpdates = 1;
    }

    /* Check trust-region reduction conditions */
    tr_reject = 0;
    if (NTL_UPDATE_REDUCTION == tl->update_type) {
      /* Get predicted reduction */
      PetscCall(KSPCGGetObjFcn(tao->ksp,&prered));
      if (prered >= 0.0) {
        /* The predicted reduction has the wrong sign.  This cannot
           happen in infinite precision arithmetic.  Step should
           be rejected! */
        tao->trust = tl->alpha1 * PetscMin(tao->trust, norm_d);
        tr_reject = 1;
      } else {
        /* Compute trial step and function value */
        PetscCall(VecCopy(tao->solution, tl->W));
        PetscCall(VecAXPY(tl->W, 1.0, tao->stepdirection));
        PetscCall(TaoComputeObjective(tao, tl->W, &ftrial));

        if (PetscIsInfOrNanReal(ftrial)) {
          tao->trust = tl->alpha1 * PetscMin(tao->trust, norm_d);
          tr_reject = 1;
        } else {
          /* Compute and actual reduction */
          actred = f - ftrial;
          prered = -prered;
          if ((PetscAbsScalar(actred) <= tl->epsilon) &&
              (PetscAbsScalar(prered) <= tl->epsilon)) {
            kappa = 1.0;
          } else {
            kappa = actred / prered;
          }

          /* Accept of reject the step and update radius */
          if (kappa < tl->eta1) {
            /* Reject the step */
            tao->trust = tl->alpha1 * PetscMin(tao->trust, norm_d);
            tr_reject = 1;
          } else {
            /* Accept the step */
            if (kappa < tl->eta2) {
              /* Marginal bad step */
              tao->trust = tl->alpha2 * PetscMin(tao->trust, norm_d);
            } else if (kappa < tl->eta3) {
              /* Reasonable step */
              tao->trust = tl->alpha3 * tao->trust;
            } else if (kappa < tl->eta4) {
              /* Good step */
              tao->trust = PetscMax(tl->alpha4 * norm_d, tao->trust);
            } else {
              /* Very good step */
              tao->trust = PetscMax(tl->alpha5 * norm_d, tao->trust);
            }
          }
        }
      }
    } else {
      /* Get predicted reduction */
      PetscCall(KSPCGGetObjFcn(tao->ksp,&prered));
      if (prered >= 0.0) {
        /* The predicted reduction has the wrong sign.  This cannot
           happen in infinite precision arithmetic.  Step should
           be rejected! */
        tao->trust = tl->gamma1 * PetscMin(tao->trust, norm_d);
        tr_reject = 1;
      } else {
        PetscCall(VecCopy(tao->solution, tl->W));
        PetscCall(VecAXPY(tl->W, 1.0, tao->stepdirection));
        PetscCall(TaoComputeObjective(tao, tl->W, &ftrial));
        if (PetscIsInfOrNanReal(ftrial)) {
          tao->trust = tl->gamma1 * PetscMin(tao->trust, norm_d);
          tr_reject = 1;
        } else {
          PetscCall(VecDot(tao->gradient, tao->stepdirection, &gdx));

          actred = f - ftrial;
          prered = -prered;
          if ((PetscAbsScalar(actred) <= tl->epsilon) &&
              (PetscAbsScalar(prered) <= tl->epsilon)) {
            kappa = 1.0;
          } else {
            kappa = actred / prered;
          }

          tau_1 = tl->theta * gdx / (tl->theta * gdx - (1.0 - tl->theta) * prered + actred);
          tau_2 = tl->theta * gdx / (tl->theta * gdx + (1.0 + tl->theta) * prered - actred);
          tau_min = PetscMin(tau_1, tau_2);
          tau_max = PetscMax(tau_1, tau_2);

          if (kappa >= 1.0 - tl->mu1) {
            /* Great agreement; accept step and update radius */
            if (tau_max < 1.0) {
              tao->trust = PetscMax(tao->trust, tl->gamma3 * norm_d);
            } else if (tau_max > tl->gamma4) {
              tao->trust = PetscMax(tao->trust, tl->gamma4 * norm_d);
            } else {
              tao->trust = PetscMax(tao->trust, tau_max * norm_d);
            }
          } else if (kappa >= 1.0 - tl->mu2) {
            /* Good agreement */

            if (tau_max < tl->gamma2) {
              tao->trust = tl->gamma2 * PetscMin(tao->trust, norm_d);
            } else if (tau_max > tl->gamma3) {
              tao->trust = PetscMax(tao->trust, tl->gamma3 * norm_d);
            } else if (tau_max < 1.0) {
              tao->trust = tau_max * PetscMin(tao->trust, norm_d);
            } else {
              tao->trust = PetscMax(tao->trust, tau_max * norm_d);
            }
          } else {
            /* Not good agreement */
            if (tau_min > 1.0) {
              tao->trust = tl->gamma2 * PetscMin(tao->trust, norm_d);
            } else if (tau_max < tl->gamma1) {
              tao->trust = tl->gamma1 * PetscMin(tao->trust, norm_d);
            } else if ((tau_min < tl->gamma1) && (tau_max >= 1.0)) {
              tao->trust = tl->gamma1 * PetscMin(tao->trust, norm_d);
            } else if ((tau_1 >= tl->gamma1) && (tau_1 < 1.0) && ((tau_2 < tl->gamma1) || (tau_2 >= 1.0))) {
              tao->trust = tau_1 * PetscMin(tao->trust, norm_d);
            } else if ((tau_2 >= tl->gamma1) && (tau_2 < 1.0) && ((tau_1 < tl->gamma1) || (tau_2 >= 1.0))) {
              tao->trust = tau_2 * PetscMin(tao->trust, norm_d);
            } else {
              tao->trust = tau_max * PetscMin(tao->trust, norm_d);
            }
            tr_reject = 1;
          }
        }
      }
    }

    if (tr_reject) {
      /* The trust-region constraints rejected the step.  Apply a linesearch.
         Check for descent direction. */
      PetscCall(VecDot(tao->stepdirection, tao->gradient, &gdx));
      if ((gdx >= 0.0) || PetscIsInfOrNanReal(gdx)) {
        /* Newton step is not descent or direction produced Inf or NaN */

        if (!tl->bfgs_pre) {
          /* We don't have the bfgs matrix around and updated
             Must use gradient direction in this case */
          PetscCall(VecCopy(tao->gradient, tao->stepdirection));
          PetscCall(VecScale(tao->stepdirection, -1.0));
          ++tl->grad;
          stepType = NTL_GRADIENT;
        } else {
          /* Attempt to use the BFGS direction */
          PetscCall(MatSolve(tl->M, tao->gradient, tao->stepdirection));
          PetscCall(VecScale(tao->stepdirection, -1.0));

          /* Check for success (descent direction) */
          PetscCall(VecDot(tao->stepdirection, tao->gradient, &gdx));
          if ((gdx >= 0) || PetscIsInfOrNanReal(gdx)) {
            /* BFGS direction is not descent or direction produced not a number
               We can assert bfgsUpdates > 1 in this case because
               the first solve produces the scaled gradient direction,
               which is guaranteed to be descent */

            /* Use steepest descent direction (scaled) */
            PetscCall(MatLMVMReset(tl->M, PETSC_FALSE));
            PetscCall(MatLMVMUpdate(tl->M, tao->solution, tao->gradient));
            PetscCall(MatSolve(tl->M, tao->gradient, tao->stepdirection));
            PetscCall(VecScale(tao->stepdirection, -1.0));

            bfgsUpdates = 1;
            ++tl->grad;
            stepType = NTL_GRADIENT;
          } else {
            PetscCall(MatLMVMGetUpdateCount(tl->M, &bfgsUpdates));
            if (1 == bfgsUpdates) {
              /* The first BFGS direction is always the scaled gradient */
              ++tl->grad;
              stepType = NTL_GRADIENT;
            } else {
              ++tl->bfgs;
              stepType = NTL_BFGS;
            }
          }
        }
      } else {
        /* Computed Newton step is descent */
        ++tl->newt;
        stepType = NTL_NEWTON;
      }

      /* Perform the linesearch */
      fold = f;
      PetscCall(VecCopy(tao->solution, tl->Xold));
      PetscCall(VecCopy(tao->gradient, tl->Gold));

      step = 1.0;
      PetscCall(TaoLineSearchApply(tao->linesearch, tao->solution, &f, tao->gradient, tao->stepdirection, &step, &ls_reason));
      PetscCall(TaoAddLineSearchCounts(tao));

      while (ls_reason != TAOLINESEARCH_SUCCESS && ls_reason != TAOLINESEARCH_SUCCESS_USER && stepType != NTL_GRADIENT) {      /* Linesearch failed */
        /* Linesearch failed */
        f = fold;
        PetscCall(VecCopy(tl->Xold, tao->solution));
        PetscCall(VecCopy(tl->Gold, tao->gradient));

        switch(stepType) {
        case NTL_NEWTON:
          /* Failed to obtain acceptable iterate with Newton step */

          if (tl->bfgs_pre) {
            /* We don't have the bfgs matrix around and being updated
               Must use gradient direction in this case */
            PetscCall(VecCopy(tao->gradient, tao->stepdirection));
            ++tl->grad;
            stepType = NTL_GRADIENT;
          } else {
            /* Attempt to use the BFGS direction */
            PetscCall(MatSolve(tl->M, tao->gradient, tao->stepdirection));

            /* Check for success (descent direction) */
            PetscCall(VecDot(tao->stepdirection, tao->gradient, &gdx));
            if ((gdx <= 0) || PetscIsInfOrNanReal(gdx)) {
              /* BFGS direction is not descent or direction produced
                 not a number.  We can assert bfgsUpdates > 1 in this case
                 Use steepest descent direction (scaled) */
              PetscCall(MatLMVMReset(tl->M, PETSC_FALSE));
              PetscCall(MatLMVMUpdate(tl->M, tao->solution, tao->gradient));
              PetscCall(MatSolve(tl->M, tao->gradient, tao->stepdirection));

              bfgsUpdates = 1;
              ++tl->grad;
              stepType = NTL_GRADIENT;
            } else {
              PetscCall(MatLMVMGetUpdateCount(tl->M, &bfgsUpdates));
              if (1 == bfgsUpdates) {
                /* The first BFGS direction is always the scaled gradient */
                ++tl->grad;
                stepType = NTL_GRADIENT;
              } else {
                ++tl->bfgs;
                stepType = NTL_BFGS;
              }
            }
          }
          break;

        case NTL_BFGS:
          /* Can only enter if pc_type == NTL_PC_BFGS
             Failed to obtain acceptable iterate with BFGS step
             Attempt to use the scaled gradient direction */
          PetscCall(MatLMVMReset(tl->M, PETSC_FALSE));
          PetscCall(MatLMVMUpdate(tl->M, tao->solution, tao->gradient));
          PetscCall(MatSolve(tl->M, tao->gradient, tao->stepdirection));

          bfgsUpdates = 1;
          ++tl->grad;
          stepType = NTL_GRADIENT;
          break;
        }
        PetscCall(VecScale(tao->stepdirection, -1.0));

        /* This may be incorrect; linesearch has values for stepmax and stepmin
           that should be reset. */
        step = 1.0;
        PetscCall(TaoLineSearchApply(tao->linesearch, tao->solution, &f, tao->gradient, tao->stepdirection, &step, &ls_reason));
        PetscCall(TaoAddLineSearchCounts(tao));
      }

      if (ls_reason != TAOLINESEARCH_SUCCESS && ls_reason != TAOLINESEARCH_SUCCESS_USER) {
        /* Failed to find an improving point */
        f = fold;
        PetscCall(VecCopy(tl->Xold, tao->solution));
        PetscCall(VecCopy(tl->Gold, tao->gradient));
        tao->trust = 0.0;
        step = 0.0;
        tao->reason = TAO_DIVERGED_LS_FAILURE;
        break;
      } else if (stepType == NTL_NEWTON) {
        if (step < tl->nu1) {
          /* Very bad step taken; reduce radius */
          tao->trust = tl->omega1 * PetscMin(norm_d, tao->trust);
        } else if (step < tl->nu2) {
          /* Reasonably bad step taken; reduce radius */
          tao->trust = tl->omega2 * PetscMin(norm_d, tao->trust);
        } else if (step < tl->nu3) {
          /* Reasonable step was taken; leave radius alone */
          if (tl->omega3 < 1.0) {
            tao->trust = tl->omega3 * PetscMin(norm_d, tao->trust);
          } else if (tl->omega3 > 1.0) {
            tao->trust = PetscMax(tl->omega3 * norm_d, tao->trust);
          }
        } else if (step < tl->nu4) {
          /* Full step taken; increase the radius */
          tao->trust = PetscMax(tl->omega4 * norm_d, tao->trust);
        } else {
          /* More than full step taken; increase the radius */
          tao->trust = PetscMax(tl->omega5 * norm_d, tao->trust);
        }
      } else {
        /* Newton step was not good; reduce the radius */
        tao->trust = tl->omega1 * PetscMin(norm_d, tao->trust);
      }
    } else {
      /* Trust-region step is accepted */
      PetscCall(VecCopy(tl->W, tao->solution));
      f = ftrial;
      PetscCall(TaoComputeGradient(tao, tao->solution, tao->gradient));
      ++tl->ntrust;
    }

    /* The radius may have been increased; modify if it is too large */
    tao->trust = PetscMin(tao->trust, tl->max_radius);

    /* Check for converged */
    PetscCall(VecNorm(tao->gradient, NORM_2, &gnorm));
    PetscCheck(!PetscIsInfOrNanReal(f) && !PetscIsInfOrNanReal(gnorm),PetscObjectComm((PetscObject)tao),PETSC_ERR_USER,"User provided compute function generated Not-a-Number");
    needH = 1;

    PetscCall(TaoLogConvergenceHistory(tao,f,gnorm,0.0,tao->ksp_its));
    PetscCall(TaoMonitor(tao,tao->niter,f,gnorm,0.0,step));
    PetscCall((*tao->ops->convergencetest)(tao,tao->cnvP));
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------- */
static PetscErrorCode TaoSetUp_NTL(Tao tao)
{
  TAO_NTL        *tl = (TAO_NTL *)tao->data;

  PetscFunctionBegin;
  if (!tao->gradient) PetscCall(VecDuplicate(tao->solution, &tao->gradient));
  if (!tao->stepdirection) PetscCall(VecDuplicate(tao->solution, &tao->stepdirection));
  if (!tl->W) PetscCall(VecDuplicate(tao->solution, &tl->W));
  if (!tl->Xold) PetscCall(VecDuplicate(tao->solution, &tl->Xold));
  if (!tl->Gold) PetscCall(VecDuplicate(tao->solution, &tl->Gold));
  tl->bfgs_pre = NULL;
  tl->M = NULL;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
static PetscErrorCode TaoDestroy_NTL(Tao tao)
{
  TAO_NTL        *tl = (TAO_NTL *)tao->data;

  PetscFunctionBegin;
  if (tao->setupcalled) {
    PetscCall(VecDestroy(&tl->W));
    PetscCall(VecDestroy(&tl->Xold));
    PetscCall(VecDestroy(&tl->Gold));
  }
  PetscCall(KSPDestroy(&tao->ksp));
  PetscCall(PetscFree(tao->data));
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
static PetscErrorCode TaoSetFromOptions_NTL(PetscOptionItems *PetscOptionsObject,Tao tao)
{
  TAO_NTL        *tl = (TAO_NTL *)tao->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject,"Newton trust region with line search method for unconstrained optimization");
  PetscCall(PetscOptionsEList("-tao_ntl_init_type", "radius initialization type", "", NTL_INIT, NTL_INIT_TYPES, NTL_INIT[tl->init_type], &tl->init_type,NULL));
  PetscCall(PetscOptionsEList("-tao_ntl_update_type", "radius update type", "", NTL_UPDATE, NTL_UPDATE_TYPES, NTL_UPDATE[tl->update_type], &tl->update_type,NULL));
  PetscCall(PetscOptionsReal("-tao_ntl_eta1", "poor steplength; reduce radius", "", tl->eta1, &tl->eta1,NULL));
  PetscCall(PetscOptionsReal("-tao_ntl_eta2", "reasonable steplength; leave radius alone", "", tl->eta2, &tl->eta2,NULL));
  PetscCall(PetscOptionsReal("-tao_ntl_eta3", "good steplength; increase radius", "", tl->eta3, &tl->eta3,NULL));
  PetscCall(PetscOptionsReal("-tao_ntl_eta4", "excellent steplength; greatly increase radius", "", tl->eta4, &tl->eta4,NULL));
  PetscCall(PetscOptionsReal("-tao_ntl_alpha1", "", "", tl->alpha1, &tl->alpha1,NULL));
  PetscCall(PetscOptionsReal("-tao_ntl_alpha2", "", "", tl->alpha2, &tl->alpha2,NULL));
  PetscCall(PetscOptionsReal("-tao_ntl_alpha3", "", "", tl->alpha3, &tl->alpha3,NULL));
  PetscCall(PetscOptionsReal("-tao_ntl_alpha4", "", "", tl->alpha4, &tl->alpha4,NULL));
  PetscCall(PetscOptionsReal("-tao_ntl_alpha5", "", "", tl->alpha5, &tl->alpha5,NULL));
  PetscCall(PetscOptionsReal("-tao_ntl_nu1", "poor steplength; reduce radius", "", tl->nu1, &tl->nu1,NULL));
  PetscCall(PetscOptionsReal("-tao_ntl_nu2", "reasonable steplength; leave radius alone", "", tl->nu2, &tl->nu2,NULL));
  PetscCall(PetscOptionsReal("-tao_ntl_nu3", "good steplength; increase radius", "", tl->nu3, &tl->nu3,NULL));
  PetscCall(PetscOptionsReal("-tao_ntl_nu4", "excellent steplength; greatly increase radius", "", tl->nu4, &tl->nu4,NULL));
  PetscCall(PetscOptionsReal("-tao_ntl_omega1", "", "", tl->omega1, &tl->omega1,NULL));
  PetscCall(PetscOptionsReal("-tao_ntl_omega2", "", "", tl->omega2, &tl->omega2,NULL));
  PetscCall(PetscOptionsReal("-tao_ntl_omega3", "", "", tl->omega3, &tl->omega3,NULL));
  PetscCall(PetscOptionsReal("-tao_ntl_omega4", "", "", tl->omega4, &tl->omega4,NULL));
  PetscCall(PetscOptionsReal("-tao_ntl_omega5", "", "", tl->omega5, &tl->omega5,NULL));
  PetscCall(PetscOptionsReal("-tao_ntl_mu1_i", "", "", tl->mu1_i, &tl->mu1_i,NULL));
  PetscCall(PetscOptionsReal("-tao_ntl_mu2_i", "", "", tl->mu2_i, &tl->mu2_i,NULL));
  PetscCall(PetscOptionsReal("-tao_ntl_gamma1_i", "", "", tl->gamma1_i, &tl->gamma1_i,NULL));
  PetscCall(PetscOptionsReal("-tao_ntl_gamma2_i", "", "", tl->gamma2_i, &tl->gamma2_i,NULL));
  PetscCall(PetscOptionsReal("-tao_ntl_gamma3_i", "", "", tl->gamma3_i, &tl->gamma3_i,NULL));
  PetscCall(PetscOptionsReal("-tao_ntl_gamma4_i", "", "", tl->gamma4_i, &tl->gamma4_i,NULL));
  PetscCall(PetscOptionsReal("-tao_ntl_theta_i", "", "", tl->theta_i, &tl->theta_i,NULL));
  PetscCall(PetscOptionsReal("-tao_ntl_mu1", "", "", tl->mu1, &tl->mu1,NULL));
  PetscCall(PetscOptionsReal("-tao_ntl_mu2", "", "", tl->mu2, &tl->mu2,NULL));
  PetscCall(PetscOptionsReal("-tao_ntl_gamma1", "", "", tl->gamma1, &tl->gamma1,NULL));
  PetscCall(PetscOptionsReal("-tao_ntl_gamma2", "", "", tl->gamma2, &tl->gamma2,NULL));
  PetscCall(PetscOptionsReal("-tao_ntl_gamma3", "", "", tl->gamma3, &tl->gamma3,NULL));
  PetscCall(PetscOptionsReal("-tao_ntl_gamma4", "", "", tl->gamma4, &tl->gamma4,NULL));
  PetscCall(PetscOptionsReal("-tao_ntl_theta", "", "", tl->theta, &tl->theta,NULL));
  PetscCall(PetscOptionsReal("-tao_ntl_min_radius", "lower bound on initial radius", "", tl->min_radius, &tl->min_radius,NULL));
  PetscCall(PetscOptionsReal("-tao_ntl_max_radius", "upper bound on radius", "", tl->max_radius, &tl->max_radius,NULL));
  PetscCall(PetscOptionsReal("-tao_ntl_epsilon", "tolerance used when computing actual and predicted reduction", "", tl->epsilon, &tl->epsilon,NULL));
  PetscOptionsHeadEnd();
  PetscCall(TaoLineSearchSetFromOptions(tao->linesearch));
  PetscCall(KSPSetFromOptions(tao->ksp));
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
static PetscErrorCode TaoView_NTL(Tao tao, PetscViewer viewer)
{
  TAO_NTL        *tl = (TAO_NTL *)tao->data;
  PetscBool      isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii));
  if (isascii) {
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Trust-region steps: %" PetscInt_FMT "\n", tl->ntrust));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Newton search steps: %" PetscInt_FMT "\n", tl->newt));
    PetscCall(PetscViewerASCIIPrintf(viewer, "BFGS search steps: %" PetscInt_FMT "\n", tl->bfgs));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Gradient search steps: %" PetscInt_FMT "\n", tl->grad));
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------- */
/*MC
  TAONTL - Newton's method with trust region globalization and line search fallback.
  At each iteration, the Newton trust region method solves the system for d
  and performs a line search in the d direction:

            min_d  .5 dT Hk d + gkT d,  s.t.   ||d|| < Delta_k

  Options Database Keys:
+ -tao_ntl_init_type - "constant","direction","interpolation"
. -tao_ntl_update_type - "reduction","interpolation"
. -tao_ntl_min_radius - lower bound on trust region radius
. -tao_ntl_max_radius - upper bound on trust region radius
. -tao_ntl_epsilon - tolerance for accepting actual / predicted reduction
. -tao_ntl_mu1_i - mu1 interpolation init factor
. -tao_ntl_mu2_i - mu2 interpolation init factor
. -tao_ntl_gamma1_i - gamma1 interpolation init factor
. -tao_ntl_gamma2_i - gamma2 interpolation init factor
. -tao_ntl_gamma3_i - gamma3 interpolation init factor
. -tao_ntl_gamma4_i - gamma4 interpolation init factor
. -tao_ntl_theta_i - theta1 interpolation init factor
. -tao_ntl_eta1 - eta1 reduction update factor
. -tao_ntl_eta2 - eta2 reduction update factor
. -tao_ntl_eta3 - eta3 reduction update factor
. -tao_ntl_eta4 - eta4 reduction update factor
. -tao_ntl_alpha1 - alpha1 reduction update factor
. -tao_ntl_alpha2 - alpha2 reduction update factor
. -tao_ntl_alpha3 - alpha3 reduction update factor
. -tao_ntl_alpha4 - alpha4 reduction update factor
. -tao_ntl_alpha4 - alpha4 reduction update factor
. -tao_ntl_mu1 - mu1 interpolation update
. -tao_ntl_mu2 - mu2 interpolation update
. -tao_ntl_gamma1 - gamma1 interpolcation update
. -tao_ntl_gamma2 - gamma2 interpolcation update
. -tao_ntl_gamma3 - gamma3 interpolcation update
. -tao_ntl_gamma4 - gamma4 interpolation update
- -tao_ntl_theta - theta1 interpolation update

  Level: beginner
M*/
PETSC_EXTERN PetscErrorCode TaoCreate_NTL(Tao tao)
{
  TAO_NTL        *tl;
  const char     *morethuente_type = TAOLINESEARCHMT;

  PetscFunctionBegin;
  PetscCall(PetscNewLog(tao,&tl));
  tao->ops->setup = TaoSetUp_NTL;
  tao->ops->solve = TaoSolve_NTL;
  tao->ops->view = TaoView_NTL;
  tao->ops->setfromoptions = TaoSetFromOptions_NTL;
  tao->ops->destroy = TaoDestroy_NTL;

  /* Override default settings (unless already changed) */
  if (!tao->max_it_changed) tao->max_it = 50;
  if (!tao->trust0_changed) tao->trust0 = 100.0;

  tao->data = (void*)tl;

  /* Default values for trust-region radius update based on steplength */
  tl->nu1 = 0.25;
  tl->nu2 = 0.50;
  tl->nu3 = 1.00;
  tl->nu4 = 1.25;

  tl->omega1 = 0.25;
  tl->omega2 = 0.50;
  tl->omega3 = 1.00;
  tl->omega4 = 2.00;
  tl->omega5 = 4.00;

  /* Default values for trust-region radius update based on reduction */
  tl->eta1 = 1.0e-4;
  tl->eta2 = 0.25;
  tl->eta3 = 0.50;
  tl->eta4 = 0.90;

  tl->alpha1 = 0.25;
  tl->alpha2 = 0.50;
  tl->alpha3 = 1.00;
  tl->alpha4 = 2.00;
  tl->alpha5 = 4.00;

  /* Default values for trust-region radius update based on interpolation */
  tl->mu1 = 0.10;
  tl->mu2 = 0.50;

  tl->gamma1 = 0.25;
  tl->gamma2 = 0.50;
  tl->gamma3 = 2.00;
  tl->gamma4 = 4.00;

  tl->theta = 0.05;

  /* Default values for trust region initialization based on interpolation */
  tl->mu1_i = 0.35;
  tl->mu2_i = 0.50;

  tl->gamma1_i = 0.0625;
  tl->gamma2_i = 0.5;
  tl->gamma3_i = 2.0;
  tl->gamma4_i = 5.0;

  tl->theta_i = 0.25;

  /* Remaining parameters */
  tl->min_radius = 1.0e-10;
  tl->max_radius = 1.0e10;
  tl->epsilon = 1.0e-6;

  tl->init_type       = NTL_INIT_INTERPOLATION;
  tl->update_type     = NTL_UPDATE_REDUCTION;

  PetscCall(TaoLineSearchCreate(((PetscObject)tao)->comm, &tao->linesearch));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)tao->linesearch,(PetscObject)tao,1));
  PetscCall(TaoLineSearchSetType(tao->linesearch, morethuente_type));
  PetscCall(TaoLineSearchUseTaoRoutines(tao->linesearch, tao));
  PetscCall(TaoLineSearchSetOptionsPrefix(tao->linesearch,tao->hdr.prefix));
  PetscCall(KSPCreate(((PetscObject)tao)->comm,&tao->ksp));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)tao->ksp,(PetscObject)tao,1));
  PetscCall(KSPSetOptionsPrefix(tao->ksp,tao->hdr.prefix));
  PetscCall(KSPAppendOptionsPrefix(tao->ksp,"tao_ntl_"));
  PetscCall(KSPSetType(tao->ksp,KSPSTCG));
  PetscFunctionReturn(0);
}
