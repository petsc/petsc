#include <../src/tao/bound/impls/bnk/bnk.h>
#include <petscksp.h>

/*
 Implements Newton's Method with a line search approach for solving
 unconstrained minimization problems.  A More'-Thuente line search
 is used to guarantee that the bfgs preconditioner remains positive
 definite.

 The method can shift the Hessian matrix.  The shifting procedure is
 adapted from the PATH algorithm for solving complementarity
 problems.

 The linear system solve should be done with a conjugate gradient
 method, although any method can be used.
*/

static PetscErrorCode TaoSolve_BNLS(Tao tao)
{
  PetscErrorCode               ierr;
  TAO_BNK                      *bnk = (TAO_BNK *)tao->data;
  TaoLineSearchConvergedReason ls_reason;

  PetscReal                    prered, actred, kappa;
  PetscReal                    tau_1, tau_2, tau_max, tau_min;
  PetscReal                    f_full, fold, gdx;
  PetscReal                    step = 1.0;
  PetscReal                    delta;
  PetscReal                    norm_d = 0.0, e_min;

  PetscInt                     stepType;
  PetscInt                     bfgsUpdates = 0;
  PetscInt                     needH = 1;
  
  PetscFunctionBegin;
  if (tao->XL || tao->XU || tao->ops->computebounds) {
    ierr = PetscPrintf(((PetscObject)tao)->comm,"WARNING: Variable bounds have been set but will be ignored by nls algorithm\n");CHKERRQ(ierr);
  }

  /* Check convergence criteria */
  ierr = TaoComputeObjectiveAndGradient(tao, tao->solution, &bnk->f, tao->gradient);CHKERRQ(ierr);
  ierr = TaoGradientNorm(tao, tao->gradient,NORM_2,&bnk->gnorm);CHKERRQ(ierr);
  if (PetscIsInfOrNanReal(bnk->f) || PetscIsInfOrNanReal(bnk->gnorm)) SETERRQ(PETSC_COMM_SELF,1, "User provided compute function generated Inf or NaN");

  tao->reason = TAO_CONTINUE_ITERATING;
  ierr = TaoLogConvergenceHistory(tao,bnk->f,bnk->gnorm,0.0,tao->ksp_its);CHKERRQ(ierr);
  ierr = TaoMonitor(tao,tao->niter,bnk->f,bnk->gnorm,0.0,step);CHKERRQ(ierr);
  ierr = (*tao->ops->convergencetest)(tao,tao->cnvP);CHKERRQ(ierr);
  if (tao->reason != TAO_CONTINUE_ITERATING) PetscFunctionReturn(0);
  
  /* Initialize the preconditioner and trust radius */
  ierr = TaoBNKInitialize(tao);CHKERRQ(ierr);

  /* Have not converged; continue with Newton method */
  while (tao->reason == TAO_CONTINUE_ITERATING) {
    ++tao->niter;
    tao->ksp_its=0;
    
    /* Use the common BNK kernel to compute the step */
    ierr = TaoBNKComputeStep(tao, &stepType);CHKERRQ(ierr);

    /* Perform the linesearch */
    fold = bnk->f;
    ierr = VecCopy(tao->solution, bnk->Xold);CHKERRQ(ierr);
    ierr = VecCopy(tao->gradient, bnk->Gold);CHKERRQ(ierr);

    ierr = TaoLineSearchApply(tao->linesearch, tao->solution, &bnk->f, tao->gradient, bnk->D, &step, &ls_reason);CHKERRQ(ierr);
    ierr = TaoAddLineSearchCounts(tao);CHKERRQ(ierr);

    while (ls_reason != TAOLINESEARCH_SUCCESS && ls_reason != TAOLINESEARCH_SUCCESS_USER && stepType != BNK_GRADIENT) {
      /* Linesearch failed, revert solution */
      bnk->f = fold;
      ierr = VecCopy(bnk->Xold, tao->solution);CHKERRQ(ierr);
      ierr = VecCopy(bnk->Gold, tao->gradient);CHKERRQ(ierr);

      switch(stepType) {
      case BNK_NEWTON:
        /* Failed to obtain acceptable iterate with Newton 1step
           Update the perturbation for next time */
        if (bnk->pert <= 0.0) {
          /* Initialize the perturbation */
          bnk->pert = PetscMin(bnk->imax, PetscMax(bnk->imin, bnk->imfac * bnk->gnorm));
          if (bnk->is_gltr) {
            ierr = KSPCGGLTRGetMinEig(tao->ksp,&e_min);CHKERRQ(ierr);
            bnk->pert = PetscMax(bnk->pert, -e_min);
          }
        } else {
          /* Increase the perturbation */
          bnk->pert = PetscMin(bnk->pmax, PetscMax(bnk->pgfac * bnk->pert, bnk->pmgfac * bnk->gnorm));
        }

        if (BNK_PC_BFGS != bnk->pc_type) {
          /* We don't have the bfgs matrix around and being updated
             Must use gradient direction in this case */
          ierr = VecCopy(tao->gradient, bnk->D);CHKERRQ(ierr);
          ++bnk->grad;
          stepType = BNK_GRADIENT;
        } else {
          /* Attempt to use the BFGS direction */
          ierr = MatLMVMSolve(bnk->M, tao->gradient, bnk->D);CHKERRQ(ierr);
          /* Check for success (descent direction) */
          ierr = VecDot(tao->solution, bnk->D, &gdx);CHKERRQ(ierr);
          if ((gdx <= 0) || PetscIsInfOrNanReal(gdx)) {
            /* BFGS direction is not descent or direction produced not a number
               We can assert bfgsUpdates > 1 in this case
               Use steepest descent direction (scaled) */

            if (bnk->f != 0.0) {
              delta = 2.0 * PetscAbsScalar(bnk->f) / (bnk->gnorm*bnk->gnorm);
            } else {
              delta = 2.0 / (bnk->gnorm*bnk->gnorm);
            }
            ierr = MatLMVMSetDelta(bnk->M, delta);CHKERRQ(ierr);
            ierr = MatLMVMReset(bnk->M);CHKERRQ(ierr);
            ierr = MatLMVMUpdate(bnk->M, tao->solution, tao->gradient);CHKERRQ(ierr);
            ierr = MatLMVMSolve(bnk->M, tao->gradient, bnk->D);CHKERRQ(ierr);

            bfgsUpdates = 1;
            ++bnk->sgrad;
            stepType = BNK_SCALED_GRADIENT;
          } else {
            ierr = MatLMVMGetUpdates(bnk->M, &bfgsUpdates);CHKERRQ(ierr);
            if (1 == bfgsUpdates) {
              /* The first BFGS direction is always the scaled gradient */
              ++bnk->sgrad;
              stepType = BNK_SCALED_GRADIENT;
            } else {
              ++bnk->bfgs;
              stepType = BNK_BFGS;
            }
          }
        }
        break;

      case BNK_BFGS:
        /* Can only enter if pc_type == BNK_PC_BFGS
           Failed to obtain acceptable iterate with BFGS step
           Attempt to use the scaled gradient direction */

        if (bnk->f != 0.0) {
          delta = 2.0 * PetscAbsScalar(bnk->f) / (bnk->gnorm*bnk->gnorm);
        } else {
          delta = 2.0 / (bnk->gnorm*bnk->gnorm);
        }
        ierr = MatLMVMSetDelta(bnk->M, delta);CHKERRQ(ierr);
        ierr = MatLMVMReset(bnk->M);CHKERRQ(ierr);
        ierr = MatLMVMUpdate(bnk->M, tao->solution, tao->gradient);CHKERRQ(ierr);
        ierr = MatLMVMSolve(bnk->M, tao->gradient, bnk->D);CHKERRQ(ierr);

        bfgsUpdates = 1;
        ++bnk->sgrad;
        stepType = BNK_SCALED_GRADIENT;
        break;

      case BNK_SCALED_GRADIENT:
        /* Can only enter if pc_type == BNK_PC_BFGS
           The scaled gradient step did not produce a new iterate;
           attemp to use the gradient direction.
           Need to make sure we are not using a different diagonal scaling */

        ierr = MatLMVMSetScale(bnk->M,0);CHKERRQ(ierr);
        ierr = MatLMVMSetDelta(bnk->M,1.0);CHKERRQ(ierr);
        ierr = MatLMVMReset(bnk->M);CHKERRQ(ierr);
        ierr = MatLMVMUpdate(bnk->M, tao->solution, tao->gradient);CHKERRQ(ierr);
        ierr = MatLMVMSolve(bnk->M, tao->gradient, bnk->D);CHKERRQ(ierr);

        bfgsUpdates = 1;
        ++bnk->grad;
        stepType = BNK_GRADIENT;
        break;
      }
      ierr = VecScale(bnk->D, -1.0);CHKERRQ(ierr);

      ierr = TaoLineSearchApply(tao->linesearch, tao->solution, &bnk->f, tao->gradient, bnk->D, &step, &ls_reason);CHKERRQ(ierr);
      ierr = TaoLineSearchGetFullStepObjective(tao->linesearch, &f_full);CHKERRQ(ierr);
      ierr = TaoAddLineSearchCounts(tao);CHKERRQ(ierr);
    }

    if (ls_reason != TAOLINESEARCH_SUCCESS && ls_reason != TAOLINESEARCH_SUCCESS_USER) {
      /* Failed to find an improving point */
      bnk->f = fold;
      ierr = VecCopy(bnk->Xold, tao->solution);CHKERRQ(ierr);
      ierr = VecCopy(bnk->Gold, tao->gradient);CHKERRQ(ierr);
      step = 0.0;
      tao->reason = TAO_DIVERGED_LS_FAILURE;
      break;
    }
    
    /* Update trust region radius */
    if (bnk->is_nash || bnk->is_stcg || bnk->is_gltr) {
      switch(bnk->update_type) {
      case BNK_UPDATE_STEP:
        if (stepType == BNK_NEWTON) {
          if (step < bnk->nu1) {
            /* Very bad step taken; reduce radius */
            tao->trust = bnk->omega1 * PetscMin(norm_d, tao->trust);
          } else if (step < bnk->nu2) {
            /* Reasonably bad step taken; reduce radius */
            tao->trust = bnk->omega2 * PetscMin(norm_d, tao->trust);
          } else if (step < bnk->nu3) {
            /*  Reasonable step was taken; leave radius alone */
            if (bnk->omega3 < 1.0) {
              tao->trust = bnk->omega3 * PetscMin(norm_d, tao->trust);
            } else if (bnk->omega3 > 1.0) {
              tao->trust = PetscMax(bnk->omega3 * norm_d, tao->trust);
            }
          } else if (step < bnk->nu4) {
            /*  Full step taken; increase the radius */
            tao->trust = PetscMax(bnk->omega4 * norm_d, tao->trust);
          } else {
            /*  More than full step taken; increase the radius */
            tao->trust = PetscMax(bnk->omega5 * norm_d, tao->trust);
          }
        } else {
          /*  Newton step was not good; reduce the radius */
          tao->trust = bnk->omega1 * PetscMin(norm_d, tao->trust);
        }
        break;

      case BNK_UPDATE_REDUCTION:
        if (stepType == BNK_NEWTON) {
          /*  Get predicted reduction */
          ierr = KSPCGGetObjFcn(tao->ksp,&prered);CHKERRQ(ierr);
          if (prered >= 0.0) {
            /*  The predicted reduction has the wrong sign.  This cannot */
            /*  happen in infinite precision arithmetic.  Step should */
            /*  be rejected! */
            tao->trust = bnk->alpha1 * PetscMin(tao->trust, norm_d);
          } else {
            if (PetscIsInfOrNanReal(f_full)) {
              tao->trust = bnk->alpha1 * PetscMin(tao->trust, norm_d);
            } else {
              /*  Compute and actual reduction */
              actred = fold - f_full;
              prered = -prered;
              if ((PetscAbsScalar(actred) <= bnk->epsilon) &&
                  (PetscAbsScalar(prered) <= bnk->epsilon)) {
                kappa = 1.0;
              } else {
                kappa = actred / prered;
              }

              /*  Accept of reject the step and update radius */
              if (kappa < bnk->eta1) {
                /*  Very bad step */
                tao->trust = bnk->alpha1 * PetscMin(tao->trust, norm_d);
              } else if (kappa < bnk->eta2) {
                /*  Marginal bad step */
                tao->trust = bnk->alpha2 * PetscMin(tao->trust, norm_d);
              } else if (kappa < bnk->eta3) {
                /*  Reasonable step */
                if (bnk->alpha3 < 1.0) {
                  tao->trust = bnk->alpha3 * PetscMin(norm_d, tao->trust);
                } else if (bnk->alpha3 > 1.0) {
                  tao->trust = PetscMax(bnk->alpha3 * norm_d, tao->trust);
                }
              } else if (kappa < bnk->eta4) {
                /*  Good step */
                tao->trust = PetscMax(bnk->alpha4 * norm_d, tao->trust);
              } else {
                /*  Very good step */
                tao->trust = PetscMax(bnk->alpha5 * norm_d, tao->trust);
              }
            }
          }
        } else {
          /*  Newton step was not good; reduce the radius */
          tao->trust = bnk->alpha1 * PetscMin(norm_d, tao->trust);
        }
        break;

      default:
        if (stepType == BNK_NEWTON) {
          ierr = KSPCGGetObjFcn(tao->ksp,&prered);CHKERRQ(ierr);
          if (prered >= 0.0) {
            /*  The predicted reduction has the wrong sign.  This cannot */
            /*  happen in infinite precision arithmetic.  Step should */
            /*  be rejected! */
            tao->trust = bnk->gamma1 * PetscMin(tao->trust, norm_d);
          } else {
            if (PetscIsInfOrNanReal(f_full)) {
              tao->trust = bnk->gamma1 * PetscMin(tao->trust, norm_d);
            } else {
              actred = fold - f_full;
              prered = -prered;
              if ((PetscAbsScalar(actred) <= bnk->epsilon) && (PetscAbsScalar(prered) <= bnk->epsilon)) {
                kappa = 1.0;
              } else {
                kappa = actred / prered;
              }

              tau_1 = bnk->theta * gdx / (bnk->theta * gdx - (1.0 - bnk->theta) * prered + actred);
              tau_2 = bnk->theta * gdx / (bnk->theta * gdx + (1.0 + bnk->theta) * prered - actred);
              tau_min = PetscMin(tau_1, tau_2);
              tau_max = PetscMax(tau_1, tau_2);

              if (kappa >= 1.0 - bnk->mu1) {
                /*  Great agreement */
                if (tau_max < 1.0) {
                  tao->trust = PetscMax(tao->trust, bnk->gamma3 * norm_d);
                } else if (tau_max > bnk->gamma4) {
                  tao->trust = PetscMax(tao->trust, bnk->gamma4 * norm_d);
                } else {
                  tao->trust = PetscMax(tao->trust, tau_max * norm_d);
                }
              } else if (kappa >= 1.0 - bnk->mu2) {
                /*  Good agreement */

                if (tau_max < bnk->gamma2) {
                  tao->trust = bnk->gamma2 * PetscMin(tao->trust, norm_d);
                } else if (tau_max > bnk->gamma3) {
                  tao->trust = PetscMax(tao->trust, bnk->gamma3 * norm_d);
                } else if (tau_max < 1.0) {
                  tao->trust = tau_max * PetscMin(tao->trust, norm_d);
                } else {
                  tao->trust = PetscMax(tao->trust, tau_max * norm_d);
                }
              } else {
                /*  Not good agreement */
                if (tau_min > 1.0) {
                  tao->trust = bnk->gamma2 * PetscMin(tao->trust, norm_d);
                } else if (tau_max < bnk->gamma1) {
                  tao->trust = bnk->gamma1 * PetscMin(tao->trust, norm_d);
                } else if ((tau_min < bnk->gamma1) && (tau_max >= 1.0)) {
                  tao->trust = bnk->gamma1 * PetscMin(tao->trust, norm_d);
                } else if ((tau_1 >= bnk->gamma1) && (tau_1 < 1.0) && ((tau_2 < bnk->gamma1) || (tau_2 >= 1.0))) {
                  tao->trust = tau_1 * PetscMin(tao->trust, norm_d);
                } else if ((tau_2 >= bnk->gamma1) && (tau_2 < 1.0) && ((tau_1 < bnk->gamma1) || (tau_2 >= 1.0))) {
                  tao->trust = tau_2 * PetscMin(tao->trust, norm_d);
                } else {
                  tao->trust = tau_max * PetscMin(tao->trust, norm_d);
                }
              }
            }
          }
        } else {
          /*  Newton step was not good; reduce the radius */
          tao->trust = bnk->gamma1 * PetscMin(norm_d, tao->trust);
        }
        break;
      }

      /*  The radius may have been increased; modify if it is too large */
      tao->trust = PetscMin(tao->trust, bnk->max_radius);
    }

    /*  Check for termination */
    ierr = TaoGradientNorm(tao, tao->gradient,NORM_2,&bnk->gnorm);CHKERRQ(ierr);
    if (PetscIsInfOrNanReal(bnk->f) || PetscIsInfOrNanReal(bnk->gnorm)) SETERRQ(PETSC_COMM_SELF,1,"User provided compute function generated Not-a-Number");
    needH = 1;
    ierr = TaoLogConvergenceHistory(tao,bnk->f,bnk->gnorm,0.0,tao->ksp_its);CHKERRQ(ierr);
    ierr = TaoMonitor(tao,tao->niter,bnk->f,bnk->gnorm,0.0,step);CHKERRQ(ierr);
    ierr = (*tao->ops->convergencetest)(tao,tao->cnvP);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode TaoCreate_BNLS(Tao tao)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = TaoCreate_BNK(tao);CHKERRQ(ierr);
  tao->ops->solve=TaoSolve_BNLS;
  PetscFunctionReturn(0);
}