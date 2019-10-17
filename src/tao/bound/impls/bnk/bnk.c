#include <petsctaolinesearch.h>
#include <../src/tao/bound/impls/bnk/bnk.h>
#include <petscksp.h>

static const char *BNK_INIT[64] = {"constant", "direction", "interpolation"};
static const char *BNK_UPDATE[64] = {"step", "reduction", "interpolation"};
static const char *BNK_AS[64] = {"none", "bertsekas"};

/*------------------------------------------------------------*/

/* Routine for initializing the KSP solver, the BFGS preconditioner, and the initial trust radius estimation */

PetscErrorCode TaoBNKInitialize(Tao tao, PetscInt initType, PetscBool *needH)
{
  PetscErrorCode               ierr;
  TAO_BNK                      *bnk = (TAO_BNK *)tao->data;
  PC                           pc;

  PetscReal                    f_min, ftrial, prered, actred, kappa, sigma, resnorm;
  PetscReal                    tau, tau_1, tau_2, tau_max, tau_min, max_radius;
  PetscBool                    is_bfgs, is_jacobi, is_symmetric, sym_set;
  PetscInt                     n, N, nDiff;
  PetscInt                     i_max = 5;
  PetscInt                     j_max = 1;
  PetscInt                     i, j;

  PetscFunctionBegin;
  /* Project the current point onto the feasible set */
  ierr = TaoComputeVariableBounds(tao);CHKERRQ(ierr);
  ierr = TaoSetVariableBounds(bnk->bncg, tao->XL, tao->XU);CHKERRQ(ierr);
  if (tao->bounded) {
    ierr = TaoLineSearchSetVariableBounds(tao->linesearch,tao->XL,tao->XU);CHKERRQ(ierr);
  }

  /* Project the initial point onto the feasible region */
  ierr = TaoBoundSolution(tao->solution, tao->XL,tao->XU, 0.0, &nDiff, tao->solution);CHKERRQ(ierr);

  /* Check convergence criteria */
  ierr = TaoComputeObjectiveAndGradient(tao, tao->solution, &bnk->f, bnk->unprojected_gradient);CHKERRQ(ierr);
  ierr = TaoBNKEstimateActiveSet(tao, bnk->as_type);CHKERRQ(ierr);
  ierr = VecCopy(bnk->unprojected_gradient, tao->gradient);CHKERRQ(ierr);
  ierr = VecISSet(tao->gradient, bnk->active_idx, 0.0);CHKERRQ(ierr);
  ierr = TaoGradientNorm(tao, tao->gradient, NORM_2, &bnk->gnorm);CHKERRQ(ierr);

  /* Test the initial point for convergence */
  ierr = VecFischer(tao->solution, bnk->unprojected_gradient, tao->XL, tao->XU, bnk->W);CHKERRQ(ierr);
  ierr = VecNorm(bnk->W, NORM_2, &resnorm);CHKERRQ(ierr);
  if (PetscIsInfOrNanReal(bnk->f) || PetscIsInfOrNanReal(resnorm)) SETERRQ(PetscObjectComm((PetscObject)tao),PETSC_ERR_USER, "User provided compute function generated Inf or NaN");
  ierr = TaoLogConvergenceHistory(tao,bnk->f,resnorm,0.0,tao->ksp_its);CHKERRQ(ierr);
  ierr = TaoMonitor(tao,tao->niter,bnk->f,resnorm,0.0,1.0);CHKERRQ(ierr);
  ierr = (*tao->ops->convergencetest)(tao,tao->cnvP);CHKERRQ(ierr);
  if (tao->reason != TAO_CONTINUE_ITERATING) PetscFunctionReturn(0);
  
  /* Reset KSP stopping reason counters */
  bnk->ksp_atol = 0;
  bnk->ksp_rtol = 0;
  bnk->ksp_dtol = 0;
  bnk->ksp_ctol = 0;
  bnk->ksp_negc = 0;
  bnk->ksp_iter = 0;
  bnk->ksp_othr = 0;
  
  /* Reset accepted step type counters */
  bnk->tot_cg_its = 0;
  bnk->newt = 0;
  bnk->bfgs = 0;
  bnk->sgrad = 0;
  bnk->grad = 0;
  
  /* Initialize the Hessian perturbation */
  bnk->pert = bnk->sval;
  
  /* Reset initial steplength to zero (this helps BNCG reset its direction internally) */
  ierr = VecSet(tao->stepdirection, 0.0);CHKERRQ(ierr);

  /* Allocate the vectors needed for the BFGS approximation */
  ierr = KSPGetPC(tao->ksp, &pc);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)pc, PCLMVM, &is_bfgs);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)pc, PCJACOBI, &is_jacobi);CHKERRQ(ierr);
  if (is_bfgs) {
    bnk->bfgs_pre = pc;
    ierr = PCLMVMGetMatLMVM(bnk->bfgs_pre, &bnk->M);CHKERRQ(ierr);
    ierr = VecGetLocalSize(tao->solution, &n);CHKERRQ(ierr);
    ierr = VecGetSize(tao->solution, &N);CHKERRQ(ierr);
    ierr = MatSetSizes(bnk->M, n, n, N, N);CHKERRQ(ierr);
    ierr = MatLMVMAllocate(bnk->M, tao->solution, bnk->unprojected_gradient);CHKERRQ(ierr);
    ierr = MatIsSymmetricKnown(bnk->M, &sym_set, &is_symmetric);CHKERRQ(ierr);
    if (!sym_set || !is_symmetric) SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_INCOMP, "LMVM matrix in the LMVM preconditioner must be symmetric.");
  } else if (is_jacobi) {
    ierr = PCJacobiSetUseAbs(pc,PETSC_TRUE);CHKERRQ(ierr);
  }
  
  /* Prepare the min/max vectors for safeguarding diagonal scales */
  ierr = VecSet(bnk->Diag_min, bnk->dmin);CHKERRQ(ierr);
  ierr = VecSet(bnk->Diag_max, bnk->dmax);CHKERRQ(ierr);

  /* Initialize trust-region radius.  The initialization is only performed
     when we are using Nash, Steihaug-Toint or the Generalized Lanczos method. */
  *needH = PETSC_TRUE;
  if (bnk->is_nash || bnk->is_stcg || bnk->is_gltr) {
    switch(initType) {
    case BNK_INIT_CONSTANT:
      /* Use the initial radius specified */
      tao->trust = tao->trust0;
      break;

    case BNK_INIT_INTERPOLATION:
      /* Use interpolation based on the initial Hessian */
      max_radius = 0.0;
      tao->trust = tao->trust0;
      for (j = 0; j < j_max; ++j) {
        f_min = bnk->f;
        sigma = 0.0;

        if (*needH) {
          /* Compute the Hessian at the new step, and extract the inactive subsystem */
          ierr = bnk->computehessian(tao);CHKERRQ(ierr);
          ierr = TaoBNKEstimateActiveSet(tao, BNK_AS_NONE);CHKERRQ(ierr);
          ierr = MatDestroy(&bnk->H_inactive);CHKERRQ(ierr);
          if (bnk->active_idx) {
            ierr = MatCreateSubMatrix(tao->hessian, bnk->inactive_idx, bnk->inactive_idx, MAT_INITIAL_MATRIX, &bnk->H_inactive);CHKERRQ(ierr);
          } else {
            ierr = MatDuplicate(tao->hessian, MAT_COPY_VALUES, &bnk->H_inactive);CHKERRQ(ierr);
          }
          *needH = PETSC_FALSE;
        }

        for (i = 0; i < i_max; ++i) {
          /* Take a steepest descent step and snap it to bounds */
          ierr = VecCopy(tao->solution, bnk->Xold);CHKERRQ(ierr);
          ierr = VecAXPY(tao->solution, -tao->trust/bnk->gnorm, tao->gradient);CHKERRQ(ierr);
          ierr = TaoBoundSolution(tao->solution, tao->XL,tao->XU, 0.0, &nDiff, tao->solution);CHKERRQ(ierr);
          /* Compute the step we actually accepted */
          ierr = VecCopy(tao->solution, bnk->W);CHKERRQ(ierr);
          ierr = VecAXPY(bnk->W, -1.0, bnk->Xold);CHKERRQ(ierr);
          /* Compute the objective at the trial */
          ierr = TaoComputeObjective(tao, tao->solution, &ftrial);CHKERRQ(ierr);
          if (PetscIsInfOrNanReal(bnk->f)) SETERRQ(PetscObjectComm((PetscObject)tao),PETSC_ERR_USER, "User provided compute function generated Inf or NaN");
          ierr = VecCopy(bnk->Xold, tao->solution);CHKERRQ(ierr);
          if (PetscIsInfOrNanReal(ftrial)) {
            tau = bnk->gamma1_i;
          } else {
            if (ftrial < f_min) {
              f_min = ftrial;
              sigma = -tao->trust / bnk->gnorm;
            }
            
            /* Compute the predicted and actual reduction */
            if (bnk->active_idx) {
              ierr = VecGetSubVector(bnk->W, bnk->inactive_idx, &bnk->X_inactive);CHKERRQ(ierr);
              ierr = VecGetSubVector(bnk->Xwork, bnk->inactive_idx, &bnk->inactive_work);CHKERRQ(ierr);
            } else {
              bnk->X_inactive = bnk->W;
              bnk->inactive_work = bnk->Xwork;
            }
            ierr = MatMult(bnk->H_inactive, bnk->X_inactive, bnk->inactive_work);CHKERRQ(ierr);
            ierr = VecDot(bnk->X_inactive, bnk->inactive_work, &prered);CHKERRQ(ierr);
            if (bnk->active_idx) {
              ierr = VecRestoreSubVector(bnk->W, bnk->inactive_idx, &bnk->X_inactive);CHKERRQ(ierr);
              ierr = VecRestoreSubVector(bnk->Xwork, bnk->inactive_idx, &bnk->inactive_work);CHKERRQ(ierr);
            } 
            prered = tao->trust * (bnk->gnorm - 0.5 * tao->trust * prered / (bnk->gnorm * bnk->gnorm));
            actred = bnk->f - ftrial;
            if ((PetscAbsScalar(actred) <= bnk->epsilon) && (PetscAbsScalar(prered) <= bnk->epsilon)) {
              kappa = 1.0;
            } else {
              kappa = actred / prered;
            }

            tau_1 = bnk->theta_i * bnk->gnorm * tao->trust / (bnk->theta_i * bnk->gnorm * tao->trust + (1.0 - bnk->theta_i) * prered - actred);
            tau_2 = bnk->theta_i * bnk->gnorm * tao->trust / (bnk->theta_i * bnk->gnorm * tao->trust - (1.0 + bnk->theta_i) * prered + actred);
            tau_min = PetscMin(tau_1, tau_2);
            tau_max = PetscMax(tau_1, tau_2);

            if (PetscAbsScalar(kappa - (PetscReal)1.0) <= bnk->mu1_i) {
              /*  Great agreement */
              max_radius = PetscMax(max_radius, tao->trust);

              if (tau_max < 1.0) {
                tau = bnk->gamma3_i;
              } else if (tau_max > bnk->gamma4_i) {
                tau = bnk->gamma4_i;
              } else {
                tau = tau_max;
              }
            } else if (PetscAbsScalar(kappa - (PetscReal)1.0) <= bnk->mu2_i) {
              /*  Good agreement */
              max_radius = PetscMax(max_radius, tao->trust);

              if (tau_max < bnk->gamma2_i) {
                tau = bnk->gamma2_i;
              } else if (tau_max > bnk->gamma3_i) {
                tau = bnk->gamma3_i;
              } else {
                tau = tau_max;
              }
            } else {
              /*  Not good agreement */
              if (tau_min > 1.0) {
                tau = bnk->gamma2_i;
              } else if (tau_max < bnk->gamma1_i) {
                tau = bnk->gamma1_i;
              } else if ((tau_min < bnk->gamma1_i) && (tau_max >= 1.0)) {
                tau = bnk->gamma1_i;
              } else if ((tau_1 >= bnk->gamma1_i) && (tau_1 < 1.0) && ((tau_2 < bnk->gamma1_i) || (tau_2 >= 1.0))) {
                tau = tau_1;
              } else if ((tau_2 >= bnk->gamma1_i) && (tau_2 < 1.0) && ((tau_1 < bnk->gamma1_i) || (tau_2 >= 1.0))) {
                tau = tau_2;
              } else {
                tau = tau_max;
              }
            }
          }
          tao->trust = tau * tao->trust;
        }

        if (f_min < bnk->f) {
          /* We accidentally found a solution better than the initial, so accept it */
          bnk->f = f_min;
          ierr = VecCopy(tao->solution, bnk->Xold);CHKERRQ(ierr);
          ierr = VecAXPY(tao->solution,sigma,tao->gradient);CHKERRQ(ierr);
          ierr = TaoBoundSolution(tao->solution, tao->XL,tao->XU, 0.0, &nDiff, tao->solution);CHKERRQ(ierr);
          ierr = VecCopy(tao->solution, tao->stepdirection);CHKERRQ(ierr);
          ierr = VecAXPY(tao->stepdirection, -1.0, bnk->Xold);CHKERRQ(ierr);
          ierr = TaoComputeGradient(tao,tao->solution,bnk->unprojected_gradient);CHKERRQ(ierr);
          ierr = TaoBNKEstimateActiveSet(tao, bnk->as_type);CHKERRQ(ierr);
          ierr = VecCopy(bnk->unprojected_gradient, tao->gradient);CHKERRQ(ierr);
          ierr = VecISSet(tao->gradient, bnk->active_idx, 0.0);CHKERRQ(ierr);
          /* Compute gradient at the new iterate and flip switch to compute the Hessian later */
          ierr = TaoGradientNorm(tao, tao->gradient, NORM_2, &bnk->gnorm);CHKERRQ(ierr);
          *needH = PETSC_TRUE;
          /* Test the new step for convergence */
          ierr = VecFischer(tao->solution, bnk->unprojected_gradient, tao->XL, tao->XU, bnk->W);CHKERRQ(ierr);
          ierr = VecNorm(bnk->W, NORM_2, &resnorm);CHKERRQ(ierr);
          if (PetscIsInfOrNanReal(resnorm)) SETERRQ(PetscObjectComm((PetscObject)tao),PETSC_ERR_USER, "User provided compute function generated Inf or NaN");
          ierr = TaoLogConvergenceHistory(tao,bnk->f,resnorm,0.0,tao->ksp_its);CHKERRQ(ierr);
          ierr = TaoMonitor(tao,tao->niter,bnk->f,resnorm,0.0,1.0);CHKERRQ(ierr);
          ierr = (*tao->ops->convergencetest)(tao,tao->cnvP);CHKERRQ(ierr);
          if (tao->reason != TAO_CONTINUE_ITERATING) PetscFunctionReturn(0);
          /* active BNCG recycling early because we have a stepdirection computed */
          ierr = TaoBNCGSetRecycleFlag(bnk->bncg, PETSC_TRUE);CHKERRQ(ierr);
        }
      }
      tao->trust = PetscMax(tao->trust, max_radius);
      
      /* Ensure that the trust radius is within the limits */
      tao->trust = PetscMax(tao->trust, bnk->min_radius);
      tao->trust = PetscMin(tao->trust, bnk->max_radius);
      break;

    default:
      /* Norm of the first direction will initialize radius */
      tao->trust = 0.0;
      break;
    }
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/* Routine for computing the exact Hessian and preparing the preconditioner at the new iterate */

PetscErrorCode TaoBNKComputeHessian(Tao tao)
{
  PetscErrorCode               ierr;
  TAO_BNK                      *bnk = (TAO_BNK *)tao->data;

  PetscFunctionBegin;
  /* Compute the Hessian */
  ierr = TaoComputeHessian(tao,tao->solution,tao->hessian,tao->hessian_pre);CHKERRQ(ierr);
  /* Add a correction to the BFGS preconditioner */
  if (bnk->M) {
    ierr = MatLMVMUpdate(bnk->M, tao->solution, bnk->unprojected_gradient);CHKERRQ(ierr);
  }
  /* Prepare the reduced sub-matrices for the inactive set */
  if (bnk->Hpre_inactive) {
    ierr = MatDestroy(&bnk->Hpre_inactive);CHKERRQ(ierr);
  }
  if (bnk->H_inactive) {
    ierr = MatDestroy(&bnk->H_inactive);CHKERRQ(ierr);
  }
  if (bnk->active_idx) {
    ierr = MatCreateSubMatrix(tao->hessian, bnk->inactive_idx, bnk->inactive_idx, MAT_INITIAL_MATRIX, &bnk->H_inactive);CHKERRQ(ierr);
    if (tao->hessian == tao->hessian_pre) {
      ierr = PetscObjectReference((PetscObject)bnk->H_inactive);CHKERRQ(ierr);
      bnk->Hpre_inactive = bnk->H_inactive;
    } else {
      ierr = MatCreateSubMatrix(tao->hessian_pre, bnk->inactive_idx, bnk->inactive_idx, MAT_INITIAL_MATRIX, &bnk->Hpre_inactive);CHKERRQ(ierr);
    }
    if (bnk->bfgs_pre) {
      ierr = PCLMVMSetIS(bnk->bfgs_pre, bnk->inactive_idx);CHKERRQ(ierr);
    }
  } else {
    ierr = MatDuplicate(tao->hessian, MAT_COPY_VALUES, &bnk->H_inactive);CHKERRQ(ierr);
    if (tao->hessian == tao->hessian_pre) {
      ierr = PetscObjectReference((PetscObject)bnk->H_inactive);CHKERRQ(ierr);
      bnk->Hpre_inactive = bnk->H_inactive;
    } else {
      ierr = MatDuplicate(tao->hessian_pre, MAT_COPY_VALUES, &bnk->Hpre_inactive);CHKERRQ(ierr);
    }
    if (bnk->bfgs_pre) {
      ierr = PCLMVMClearIS(bnk->bfgs_pre);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/* Routine for estimating the active set */

PetscErrorCode TaoBNKEstimateActiveSet(Tao tao, PetscInt asType) 
{
  PetscErrorCode               ierr;
  TAO_BNK                      *bnk = (TAO_BNK *)tao->data;
  PetscBool                    hessComputed, diagExists;

  PetscFunctionBegin;
  switch (asType) {
  case BNK_AS_NONE:
    ierr = ISDestroy(&bnk->inactive_idx);CHKERRQ(ierr);
    ierr = VecWhichInactive(tao->XL, tao->solution, bnk->unprojected_gradient, tao->XU, PETSC_TRUE, &bnk->inactive_idx);CHKERRQ(ierr);
    ierr = ISDestroy(&bnk->active_idx);CHKERRQ(ierr);
    ierr = ISComplementVec(bnk->inactive_idx, tao->solution, &bnk->active_idx);CHKERRQ(ierr);
    break;

  case BNK_AS_BERTSEKAS:
    /* Compute the trial step vector with which we will estimate the active set at the next iteration */
    if (bnk->M) {
      /* If the BFGS preconditioner matrix is available, we will construct a trial step with it */
      ierr = MatSolve(bnk->M, bnk->unprojected_gradient, bnk->W);CHKERRQ(ierr);
    } else {
      hessComputed = diagExists = PETSC_FALSE;
      if (tao->hessian) {
        ierr = MatAssembled(tao->hessian, &hessComputed);CHKERRQ(ierr);
      }
      if (hessComputed) {
        ierr = MatHasOperation(tao->hessian, MATOP_GET_DIAGONAL, &diagExists);CHKERRQ(ierr);
      }
      if (diagExists) {
        /* BFGS preconditioner doesn't exist so let's invert the absolute diagonal of the Hessian instead onto the gradient */
        ierr = MatGetDiagonal(tao->hessian, bnk->Xwork);CHKERRQ(ierr);
        ierr = VecAbs(bnk->Xwork);CHKERRQ(ierr);
        ierr = VecMedian(bnk->Diag_min, bnk->Xwork, bnk->Diag_max, bnk->Xwork);CHKERRQ(ierr);
        ierr = VecReciprocal(bnk->Xwork);CHKERRQ(ierr);CHKERRQ(ierr);
        ierr = VecPointwiseMult(bnk->W, bnk->Xwork, bnk->unprojected_gradient);CHKERRQ(ierr);
      } else {
        /* If the Hessian or its diagonal does not exist, we will simply use gradient step */
        ierr = VecCopy(bnk->unprojected_gradient, bnk->W);CHKERRQ(ierr);
      }
    }
    ierr = VecScale(bnk->W, -1.0);CHKERRQ(ierr);
    ierr = TaoEstimateActiveBounds(tao->solution, tao->XL, tao->XU, bnk->unprojected_gradient, bnk->W, bnk->Xwork, bnk->as_step, &bnk->as_tol, 
                                   &bnk->active_lower, &bnk->active_upper, &bnk->active_fixed, &bnk->active_idx, &bnk->inactive_idx);CHKERRQ(ierr);
    break;
    
  default:
    break;
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/* Routine for bounding the step direction */

PetscErrorCode TaoBNKBoundStep(Tao tao, PetscInt asType, Vec step) 
{
  PetscErrorCode               ierr;
  TAO_BNK                      *bnk = (TAO_BNK *)tao->data;
  
  PetscFunctionBegin;
  switch (asType) {
  case BNK_AS_NONE:
    ierr = VecISSet(step, bnk->active_idx, 0.0);CHKERRQ(ierr);
    break;

  case BNK_AS_BERTSEKAS:
    ierr = TaoBoundStep(tao->solution, tao->XL, tao->XU, bnk->active_lower, bnk->active_upper, bnk->active_fixed, 1.0, step);CHKERRQ(ierr);
    break;

  default:
    break;
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/* Routine for taking a finite number of BNCG iterations to 
   accelerate Newton convergence.
   
   In practice, this approach simply trades off Hessian evaluations 
   for more gradient evaluations.
*/

PetscErrorCode TaoBNKTakeCGSteps(Tao tao, PetscBool *terminate)
{
  TAO_BNK                      *bnk = (TAO_BNK *)tao->data;
  PetscErrorCode               ierr;
  
  PetscFunctionBegin;
  *terminate = PETSC_FALSE;
  if (bnk->max_cg_its > 0) {
    /* Copy the current function value (important vectors are already shared) */
    bnk->bncg_ctx->f = bnk->f;
    /* Take some small finite number of BNCG iterations */
    ierr = TaoSolve(bnk->bncg);CHKERRQ(ierr);
    /* Add the number of gradient and function evaluations to the total */
    tao->nfuncs += bnk->bncg->nfuncs;
    tao->nfuncgrads += bnk->bncg->nfuncgrads;
    tao->ngrads += bnk->bncg->ngrads;
    tao->nhess += bnk->bncg->nhess;
    bnk->tot_cg_its += bnk->bncg->niter;
    /* Extract the BNCG function value out and save it into BNK */
    bnk->f = bnk->bncg_ctx->f;
    if (bnk->bncg->reason == TAO_CONVERGED_GATOL || bnk->bncg->reason == TAO_CONVERGED_GRTOL || bnk->bncg->reason == TAO_CONVERGED_GTTOL || bnk->bncg->reason == TAO_CONVERGED_MINF) {
      *terminate = PETSC_TRUE;
    } else {
      ierr = TaoBNKEstimateActiveSet(tao, bnk->as_type);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/* Routine for computing the Newton step. */

PetscErrorCode TaoBNKComputeStep(Tao tao, PetscBool shift, KSPConvergedReason *ksp_reason, PetscInt *step_type)
{
  PetscErrorCode               ierr;
  TAO_BNK                      *bnk = (TAO_BNK *)tao->data;
  PetscInt                     bfgsUpdates = 0;
  PetscInt                     kspits;
  PetscBool                    is_lmvm;
  
  PetscFunctionBegin;
  /* If there are no inactive variables left, save some computation and return an adjusted zero step
     that has (l-x) and (u-x) for lower and upper bounded variables. */
  if (!bnk->inactive_idx) {
    ierr = VecSet(tao->stepdirection, 0.0);CHKERRQ(ierr);
    ierr = TaoBNKBoundStep(tao, bnk->as_type, tao->stepdirection);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  
  /* Shift the reduced Hessian matrix */
  if ((shift) && (bnk->pert > 0)) {
    ierr = PetscObjectTypeCompare((PetscObject)tao->hessian, MATLMVM, &is_lmvm);CHKERRQ(ierr);
    if (is_lmvm) {
      ierr = MatShift(tao->hessian, bnk->pert);CHKERRQ(ierr);
    } else {
      ierr = MatShift(bnk->H_inactive, bnk->pert);CHKERRQ(ierr);
      if (bnk->H_inactive != bnk->Hpre_inactive) {
        ierr = MatShift(bnk->Hpre_inactive, bnk->pert);CHKERRQ(ierr);
      }
    }
  }
  
  /* Solve the Newton system of equations */
  tao->ksp_its = 0;
  ierr = VecSet(tao->stepdirection, 0.0);CHKERRQ(ierr);
  ierr = KSPReset(tao->ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(tao->ksp,bnk->H_inactive,bnk->Hpre_inactive);CHKERRQ(ierr);
  ierr = VecCopy(bnk->unprojected_gradient, bnk->Gwork);CHKERRQ(ierr);
  if (bnk->active_idx) {
    ierr = VecGetSubVector(bnk->Gwork, bnk->inactive_idx, &bnk->G_inactive);CHKERRQ(ierr);
    ierr = VecGetSubVector(tao->stepdirection, bnk->inactive_idx, &bnk->X_inactive);CHKERRQ(ierr);
  } else {
    bnk->G_inactive = bnk->unprojected_gradient;
    bnk->X_inactive = tao->stepdirection;
  }
  if (bnk->is_nash || bnk->is_stcg || bnk->is_gltr) {
    ierr = KSPCGSetRadius(tao->ksp,tao->trust);CHKERRQ(ierr);
    ierr = KSPSolve(tao->ksp, bnk->G_inactive, bnk->X_inactive);CHKERRQ(ierr);
    ierr = KSPGetIterationNumber(tao->ksp,&kspits);CHKERRQ(ierr);
    tao->ksp_its+=kspits;
    tao->ksp_tot_its+=kspits;
    ierr = KSPCGGetNormD(tao->ksp,&bnk->dnorm);CHKERRQ(ierr);

    if (0.0 == tao->trust) {
      /* Radius was uninitialized; use the norm of the direction */
      if (bnk->dnorm > 0.0) {
        tao->trust = bnk->dnorm;

        /* Modify the radius if it is too large or small */
        tao->trust = PetscMax(tao->trust, bnk->min_radius);
        tao->trust = PetscMin(tao->trust, bnk->max_radius);
      } else {
        /* The direction was bad; set radius to default value and re-solve
           the trust-region subproblem to get a direction */
        tao->trust = tao->trust0;

        /* Modify the radius if it is too large or small */
        tao->trust = PetscMax(tao->trust, bnk->min_radius);
        tao->trust = PetscMin(tao->trust, bnk->max_radius);

        ierr = KSPCGSetRadius(tao->ksp,tao->trust);CHKERRQ(ierr);
        ierr = KSPSolve(tao->ksp, bnk->G_inactive, bnk->X_inactive);CHKERRQ(ierr);
        ierr = KSPGetIterationNumber(tao->ksp,&kspits);CHKERRQ(ierr);
        tao->ksp_its+=kspits;
        tao->ksp_tot_its+=kspits;
        ierr = KSPCGGetNormD(tao->ksp,&bnk->dnorm);CHKERRQ(ierr);

        if (bnk->dnorm == 0.0) SETERRQ(PetscObjectComm((PetscObject)tao),PETSC_ERR_PLIB, "Initial direction zero");
      }
    }
  } else {
    ierr = KSPSolve(tao->ksp, bnk->G_inactive, bnk->X_inactive);CHKERRQ(ierr);
    ierr = KSPGetIterationNumber(tao->ksp, &kspits);CHKERRQ(ierr);
    tao->ksp_its += kspits;
    tao->ksp_tot_its+=kspits;
  }
  /* Restore sub vectors back */
  if (bnk->active_idx) {
    ierr = VecRestoreSubVector(bnk->Gwork, bnk->inactive_idx, &bnk->G_inactive);CHKERRQ(ierr);
    ierr = VecRestoreSubVector(tao->stepdirection, bnk->inactive_idx, &bnk->X_inactive);CHKERRQ(ierr);
  }
  /* Make sure the safeguarded fall-back step is zero for actively bounded variables */
  ierr = VecScale(tao->stepdirection, -1.0);CHKERRQ(ierr);
  ierr = TaoBNKBoundStep(tao, bnk->as_type, tao->stepdirection);CHKERRQ(ierr);
  
  /* Record convergence reasons */
  ierr = KSPGetConvergedReason(tao->ksp, ksp_reason);CHKERRQ(ierr);
  if (KSP_CONVERGED_ATOL == *ksp_reason) {
    ++bnk->ksp_atol;
  } else if (KSP_CONVERGED_RTOL == *ksp_reason) {
    ++bnk->ksp_rtol;
  } else if (KSP_CONVERGED_CG_CONSTRAINED == *ksp_reason) {
    ++bnk->ksp_ctol;
  } else if (KSP_CONVERGED_CG_NEG_CURVE == *ksp_reason) {
    ++bnk->ksp_negc;
  } else if (KSP_DIVERGED_DTOL == *ksp_reason) {
    ++bnk->ksp_dtol;
  } else if (KSP_DIVERGED_ITS == *ksp_reason) {
    ++bnk->ksp_iter;
  } else {
    ++bnk->ksp_othr;
  }
  
  /* Make sure the BFGS preconditioner is healthy */
  if (bnk->M) {
    ierr = MatLMVMGetUpdateCount(bnk->M, &bfgsUpdates);CHKERRQ(ierr);
    if ((KSP_DIVERGED_INDEFINITE_PC == *ksp_reason) && (bfgsUpdates > 0)) {
      /* Preconditioner is numerically indefinite; reset the approximation. */
      ierr = MatLMVMReset(bnk->M, PETSC_FALSE);CHKERRQ(ierr);
      ierr = MatLMVMUpdate(bnk->M, tao->solution, bnk->unprojected_gradient);CHKERRQ(ierr);
    }
  }
  *step_type = BNK_NEWTON;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/* Routine for recomputing the predicted reduction for a given step vector */

PetscErrorCode TaoBNKRecomputePred(Tao tao, Vec S, PetscReal *prered)
{
  PetscErrorCode               ierr;
  TAO_BNK                      *bnk = (TAO_BNK *)tao->data;
  
  PetscFunctionBegin;
  /* Extract subvectors associated with the inactive set */
  if (bnk->active_idx){
    ierr = VecGetSubVector(tao->stepdirection, bnk->inactive_idx, &bnk->X_inactive);CHKERRQ(ierr);
    ierr = VecGetSubVector(bnk->Xwork, bnk->inactive_idx, &bnk->inactive_work);CHKERRQ(ierr);
    ierr = VecGetSubVector(bnk->Gwork, bnk->inactive_idx, &bnk->G_inactive);CHKERRQ(ierr);
  } else {
    bnk->X_inactive = tao->stepdirection;
    bnk->inactive_work = bnk->Xwork;
    bnk->G_inactive = bnk->Gwork;
  }
  /* Recompute the predicted decrease based on the quadratic model */
  ierr = MatMult(bnk->H_inactive, bnk->X_inactive, bnk->inactive_work);CHKERRQ(ierr);
  ierr = VecAYPX(bnk->inactive_work, -0.5, bnk->G_inactive);CHKERRQ(ierr);
  ierr = VecDot(bnk->inactive_work, bnk->X_inactive, prered);CHKERRQ(ierr);
  /* Restore the sub vectors */
  if (bnk->active_idx){
    ierr = VecRestoreSubVector(tao->stepdirection, bnk->inactive_idx, &bnk->X_inactive);CHKERRQ(ierr);
    ierr = VecRestoreSubVector(bnk->Xwork, bnk->inactive_idx, &bnk->inactive_work);CHKERRQ(ierr);
    ierr = VecRestoreSubVector(bnk->Gwork, bnk->inactive_idx, &bnk->G_inactive);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/* Routine for ensuring that the Newton step is a descent direction.

   The step direction falls back onto BFGS, scaled gradient and gradient steps 
   in the event that the Newton step fails the test.
*/

PetscErrorCode TaoBNKSafeguardStep(Tao tao, KSPConvergedReason ksp_reason, PetscInt *stepType)
{
  PetscErrorCode               ierr;
  TAO_BNK                      *bnk = (TAO_BNK *)tao->data;
  
  PetscReal                    gdx, e_min;
  PetscInt                     bfgsUpdates;
  
  PetscFunctionBegin;
  switch (*stepType) {
  case BNK_NEWTON:
    ierr = VecDot(tao->stepdirection, tao->gradient, &gdx);CHKERRQ(ierr);
    if ((gdx >= 0.0) || PetscIsInfOrNanReal(gdx)) {
      /* Newton step is not descent or direction produced Inf or NaN
        Update the perturbation for next time */
      if (bnk->pert <= 0.0) {
        /* Initialize the perturbation */
        bnk->pert = PetscMin(bnk->imax, PetscMax(bnk->imin, bnk->imfac * bnk->gnorm));
        if (bnk->is_gltr) {
          ierr = KSPGLTRGetMinEig(tao->ksp,&e_min);CHKERRQ(ierr);
          bnk->pert = PetscMax(bnk->pert, -e_min);
        }
      } else {
        /* Increase the perturbation */
        bnk->pert = PetscMin(bnk->pmax, PetscMax(bnk->pgfac * bnk->pert, bnk->pmgfac * bnk->gnorm));
      }

      if (!bnk->M) {
        /* We don't have the bfgs matrix around and updated
          Must use gradient direction in this case */
        ierr = VecCopy(tao->gradient, tao->stepdirection);CHKERRQ(ierr);
        *stepType = BNK_GRADIENT;
      } else {
        /* Attempt to use the BFGS direction */
        ierr = MatSolve(bnk->M, bnk->unprojected_gradient, tao->stepdirection);CHKERRQ(ierr);

        /* Check for success (descent direction) 
          NOTE: Negative gdx here means not a descent direction because 
          the fall-back step is missing a negative sign. */
        ierr = VecDot(tao->gradient, tao->stepdirection, &gdx);CHKERRQ(ierr);
        if ((gdx <= 0.0) || PetscIsInfOrNanReal(gdx)) {
          /* BFGS direction is not descent or direction produced not a number
            We can assert bfgsUpdates > 1 in this case because
            the first solve produces the scaled gradient direction,
            which is guaranteed to be descent */

          /* Use steepest descent direction (scaled) */
          ierr = MatLMVMReset(bnk->M, PETSC_FALSE);CHKERRQ(ierr);
          ierr = MatLMVMUpdate(bnk->M, tao->solution, bnk->unprojected_gradient);CHKERRQ(ierr);
          ierr = MatSolve(bnk->M, bnk->unprojected_gradient, tao->stepdirection);CHKERRQ(ierr);

          *stepType = BNK_SCALED_GRADIENT;
        } else {
          ierr = MatLMVMGetUpdateCount(bnk->M, &bfgsUpdates);CHKERRQ(ierr);
          if (1 == bfgsUpdates) {
            /* The first BFGS direction is always the scaled gradient */
            *stepType = BNK_SCALED_GRADIENT;
          } else {
            *stepType = BNK_BFGS;
          }
        }
      }
      /* Make sure the safeguarded fall-back step is zero for actively bounded variables */
      ierr = VecScale(tao->stepdirection, -1.0);CHKERRQ(ierr);
      ierr = TaoBNKBoundStep(tao, bnk->as_type, tao->stepdirection);CHKERRQ(ierr);
    } else {
      /* Computed Newton step is descent */
      switch (ksp_reason) {
      case KSP_DIVERGED_NANORINF:
      case KSP_DIVERGED_BREAKDOWN:
      case KSP_DIVERGED_INDEFINITE_MAT:
      case KSP_DIVERGED_INDEFINITE_PC:
      case KSP_CONVERGED_CG_NEG_CURVE:
        /* Matrix or preconditioner is indefinite; increase perturbation */
        if (bnk->pert <= 0.0) {
          /* Initialize the perturbation */
          bnk->pert = PetscMin(bnk->imax, PetscMax(bnk->imin, bnk->imfac * bnk->gnorm));
          if (bnk->is_gltr) {
            ierr = KSPGLTRGetMinEig(tao->ksp, &e_min);CHKERRQ(ierr);
            bnk->pert = PetscMax(bnk->pert, -e_min);
          }
        } else {
          /* Increase the perturbation */
          bnk->pert = PetscMin(bnk->pmax, PetscMax(bnk->pgfac * bnk->pert, bnk->pmgfac * bnk->gnorm));
        }
        break;

      default:
        /* Newton step computation is good; decrease perturbation */
        bnk->pert = PetscMin(bnk->psfac * bnk->pert, bnk->pmsfac * bnk->gnorm);
        if (bnk->pert < bnk->pmin) {
          bnk->pert = 0.0;
        }
        break;
      }
      *stepType = BNK_NEWTON;
    }
    break;
  
  case BNK_BFGS:
    /* Check for success (descent direction) */
    ierr = VecDot(tao->stepdirection, tao->gradient, &gdx);CHKERRQ(ierr);
    if (gdx >= 0 || PetscIsInfOrNanReal(gdx)) {
      /* Step is not descent or solve was not successful
         Use steepest descent direction (scaled) */
      ierr = MatLMVMReset(bnk->M, PETSC_FALSE);CHKERRQ(ierr);
      ierr = MatLMVMUpdate(bnk->M, tao->solution, bnk->unprojected_gradient);CHKERRQ(ierr);
      ierr = MatSolve(bnk->M, tao->gradient, tao->stepdirection);CHKERRQ(ierr);
      ierr = VecScale(tao->stepdirection,-1.0);CHKERRQ(ierr);
      ierr = TaoBNKBoundStep(tao, bnk->as_type, tao->stepdirection);CHKERRQ(ierr);
      *stepType = BNK_SCALED_GRADIENT;
    } else {
      *stepType = BNK_BFGS;
    }
    break;
  
  case BNK_SCALED_GRADIENT:
    break;
    
  default:
    break;
  }
  
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/* Routine for performing a bound-projected More-Thuente line search.

  Includes fallbacks to BFGS, scaled gradient, and unscaled gradient steps if the 
  Newton step does not produce a valid step length.
*/

PetscErrorCode TaoBNKPerformLineSearch(Tao tao, PetscInt *stepType, PetscReal *steplen, TaoLineSearchConvergedReason *reason)
{
  TAO_BNK        *bnk = (TAO_BNK *)tao->data;
  PetscErrorCode ierr;
  TaoLineSearchConvergedReason ls_reason;
  
  PetscReal      e_min, gdx;
  PetscInt       bfgsUpdates;
  
  PetscFunctionBegin;
  /* Perform the linesearch */
  ierr = TaoLineSearchApply(tao->linesearch, tao->solution, &bnk->f, bnk->unprojected_gradient, tao->stepdirection, steplen, &ls_reason);CHKERRQ(ierr);
  ierr = TaoAddLineSearchCounts(tao);CHKERRQ(ierr);

  while (ls_reason != TAOLINESEARCH_SUCCESS && ls_reason != TAOLINESEARCH_SUCCESS_USER && *stepType != BNK_SCALED_GRADIENT && *stepType != BNK_GRADIENT) {
    /* Linesearch failed, revert solution */
    bnk->f = bnk->fold;
    ierr = VecCopy(bnk->Xold, tao->solution);CHKERRQ(ierr);
    ierr = VecCopy(bnk->unprojected_gradient_old, bnk->unprojected_gradient);CHKERRQ(ierr);

    switch(*stepType) {
    case BNK_NEWTON:
      /* Failed to obtain acceptable iterate with Newton step
         Update the perturbation for next time */
      if (bnk->pert <= 0.0) {
        /* Initialize the perturbation */
        bnk->pert = PetscMin(bnk->imax, PetscMax(bnk->imin, bnk->imfac * bnk->gnorm));
        if (bnk->is_gltr) {
          ierr = KSPGLTRGetMinEig(tao->ksp,&e_min);CHKERRQ(ierr);
          bnk->pert = PetscMax(bnk->pert, -e_min);
        }
      } else {
        /* Increase the perturbation */
        bnk->pert = PetscMin(bnk->pmax, PetscMax(bnk->pgfac * bnk->pert, bnk->pmgfac * bnk->gnorm));
      }

      if (!bnk->M) {
        /* We don't have the bfgs matrix around and being updated
           Must use gradient direction in this case */
        ierr = VecCopy(bnk->unprojected_gradient, tao->stepdirection);CHKERRQ(ierr);
        *stepType = BNK_GRADIENT;
      } else {
        /* Attempt to use the BFGS direction */
        ierr = MatSolve(bnk->M, bnk->unprojected_gradient, tao->stepdirection);CHKERRQ(ierr);
        /* Check for success (descent direction) 
           NOTE: Negative gdx means not a descent direction because the step here is missing a negative sign. */
        ierr = VecDot(tao->gradient, tao->stepdirection, &gdx);CHKERRQ(ierr);
        if ((gdx <= 0.0) || PetscIsInfOrNanReal(gdx)) {
          /* BFGS direction is not descent or direction produced not a number
             We can assert bfgsUpdates > 1 in this case
             Use steepest descent direction (scaled) */
          ierr = MatLMVMReset(bnk->M, PETSC_FALSE);CHKERRQ(ierr);
          ierr = MatLMVMUpdate(bnk->M, tao->solution, bnk->unprojected_gradient);CHKERRQ(ierr);
          ierr = MatSolve(bnk->M, bnk->unprojected_gradient, tao->stepdirection);CHKERRQ(ierr);

          bfgsUpdates = 1;
          *stepType = BNK_SCALED_GRADIENT;
        } else {
          ierr = MatLMVMGetUpdateCount(bnk->M, &bfgsUpdates);CHKERRQ(ierr);
          if (1 == bfgsUpdates) {
            /* The first BFGS direction is always the scaled gradient */
            *stepType = BNK_SCALED_GRADIENT;
          } else {
            *stepType = BNK_BFGS;
          }
        }
      }
      break;

    case BNK_BFGS:
      /* Can only enter if pc_type == BNK_PC_BFGS
         Failed to obtain acceptable iterate with BFGS step
         Attempt to use the scaled gradient direction */
      ierr = MatLMVMReset(bnk->M, PETSC_FALSE);CHKERRQ(ierr);
      ierr = MatLMVMUpdate(bnk->M, tao->solution, bnk->unprojected_gradient);CHKERRQ(ierr);
      ierr = MatSolve(bnk->M, bnk->unprojected_gradient, tao->stepdirection);CHKERRQ(ierr);

      bfgsUpdates = 1;
      *stepType = BNK_SCALED_GRADIENT;
      break;
    }
    /* Make sure the safeguarded fall-back step is zero for actively bounded variables */
    ierr = VecScale(tao->stepdirection, -1.0);CHKERRQ(ierr);
    ierr = TaoBNKBoundStep(tao, bnk->as_type, tao->stepdirection);CHKERRQ(ierr);
    
    /* Perform one last line search with the fall-back step */
    ierr = TaoLineSearchApply(tao->linesearch, tao->solution, &bnk->f, bnk->unprojected_gradient, tao->stepdirection, steplen, &ls_reason);CHKERRQ(ierr);
    ierr = TaoAddLineSearchCounts(tao);CHKERRQ(ierr);
  }
  *reason = ls_reason;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/* Routine for updating the trust radius. 

  Function features three different update methods: 
  1) Line-search step length based
  2) Predicted decrease on the CG quadratic model
  3) Interpolation
*/

PetscErrorCode TaoBNKUpdateTrustRadius(Tao tao, PetscReal prered, PetscReal actred, PetscInt updateType, PetscInt stepType, PetscBool *accept)
{
  TAO_BNK        *bnk = (TAO_BNK *)tao->data;
  PetscErrorCode ierr;
  
  PetscReal      step, kappa;
  PetscReal      gdx, tau_1, tau_2, tau_min, tau_max;

  PetscFunctionBegin;
  /* Update trust region radius */
  *accept = PETSC_FALSE;
  switch(updateType) {
  case BNK_UPDATE_STEP:
    *accept = PETSC_TRUE; /* always accept here because line search succeeded */
    if (stepType == BNK_NEWTON) {
      ierr = TaoLineSearchGetStepLength(tao->linesearch, &step);CHKERRQ(ierr);
      if (step < bnk->nu1) {
        /* Very bad step taken; reduce radius */
        tao->trust = bnk->omega1 * PetscMin(bnk->dnorm, tao->trust);
      } else if (step < bnk->nu2) {
        /* Reasonably bad step taken; reduce radius */
        tao->trust = bnk->omega2 * PetscMin(bnk->dnorm, tao->trust);
      } else if (step < bnk->nu3) {
        /*  Reasonable step was taken; leave radius alone */
        if (bnk->omega3 < 1.0) {
          tao->trust = bnk->omega3 * PetscMin(bnk->dnorm, tao->trust);
        } else if (bnk->omega3 > 1.0) {
          tao->trust = PetscMax(bnk->omega3 * bnk->dnorm, tao->trust);
        }
      } else if (step < bnk->nu4) {
        /*  Full step taken; increase the radius */
        tao->trust = PetscMax(bnk->omega4 * bnk->dnorm, tao->trust);
      } else {
        /*  More than full step taken; increase the radius */
        tao->trust = PetscMax(bnk->omega5 * bnk->dnorm, tao->trust);
      }
    } else {
      /*  Newton step was not good; reduce the radius */
      tao->trust = bnk->omega1 * PetscMin(bnk->dnorm, tao->trust);
    }
    break;

  case BNK_UPDATE_REDUCTION:
    if (stepType == BNK_NEWTON) {
      if ((prered < 0.0) || PetscIsInfOrNanReal(prered)) {
        /* The predicted reduction has the wrong sign.  This cannot
           happen in infinite precision arithmetic.  Step should
           be rejected! */
        tao->trust = bnk->alpha1 * PetscMin(tao->trust, bnk->dnorm);
      } else {
        if (PetscIsInfOrNanReal(actred)) {
          tao->trust = bnk->alpha1 * PetscMin(tao->trust, bnk->dnorm);
        } else {
          if ((PetscAbsScalar(actred) <= PetscMax(1.0, PetscAbsScalar(bnk->f))*bnk->epsilon) && (PetscAbsScalar(prered) <= PetscMax(1.0, PetscAbsScalar(bnk->f))*bnk->epsilon)) {
            kappa = 1.0;
          } else {
            kappa = actred / prered;
          }
          /* Accept or reject the step and update radius */
          if (kappa < bnk->eta1) {
            /* Reject the step */
            tao->trust = bnk->alpha1 * PetscMin(tao->trust, bnk->dnorm);
          } else {
            /* Accept the step */
            *accept = PETSC_TRUE;
            /* Update the trust region radius only if the computed step is at the trust radius boundary */
            if (bnk->dnorm == tao->trust) {
              if (kappa < bnk->eta2) {
                /* Marginal bad step */
                tao->trust = bnk->alpha2 * tao->trust;
              } else if (kappa < bnk->eta3) {
                /* Reasonable step */
                tao->trust = bnk->alpha3 * tao->trust;
              } else if (kappa < bnk->eta4) {
                /* Good step */
                tao->trust = bnk->alpha4 * tao->trust;
              } else {
                /* Very good step */
                tao->trust = bnk->alpha5 * tao->trust;
              }
            }
          }
        }
      }
    } else {
      /*  Newton step was not good; reduce the radius */
      tao->trust = bnk->alpha1 * PetscMin(bnk->dnorm, tao->trust);
    }
    break;

  default:
    if (stepType == BNK_NEWTON) {
      if (prered < 0.0) {
        /*  The predicted reduction has the wrong sign.  This cannot */
        /*  happen in infinite precision arithmetic.  Step should */
        /*  be rejected! */
        tao->trust = bnk->gamma1 * PetscMin(tao->trust, bnk->dnorm);
      } else {
        if (PetscIsInfOrNanReal(actred)) {
          tao->trust = bnk->gamma1 * PetscMin(tao->trust, bnk->dnorm);
        } else {
          if ((PetscAbsScalar(actred) <= bnk->epsilon) && (PetscAbsScalar(prered) <= bnk->epsilon)) {
            kappa = 1.0;
          } else {
            kappa = actred / prered;
          }
          
          ierr = VecDot(tao->gradient, tao->stepdirection, &gdx);CHKERRQ(ierr);
          tau_1 = bnk->theta * gdx / (bnk->theta * gdx - (1.0 - bnk->theta) * prered + actred);
          tau_2 = bnk->theta * gdx / (bnk->theta * gdx + (1.0 + bnk->theta) * prered - actred);
          tau_min = PetscMin(tau_1, tau_2);
          tau_max = PetscMax(tau_1, tau_2);

          if (kappa >= 1.0 - bnk->mu1) {
            /*  Great agreement */
            *accept = PETSC_TRUE;
            if (tau_max < 1.0) {
              tao->trust = PetscMax(tao->trust, bnk->gamma3 * bnk->dnorm);
            } else if (tau_max > bnk->gamma4) {
              tao->trust = PetscMax(tao->trust, bnk->gamma4 * bnk->dnorm);
            } else {
              tao->trust = PetscMax(tao->trust, tau_max * bnk->dnorm);
            }
          } else if (kappa >= 1.0 - bnk->mu2) {
            /*  Good agreement */
            *accept = PETSC_TRUE;
            if (tau_max < bnk->gamma2) {
              tao->trust = bnk->gamma2 * PetscMin(tao->trust, bnk->dnorm);
            } else if (tau_max > bnk->gamma3) {
              tao->trust = PetscMax(tao->trust, bnk->gamma3 * bnk->dnorm);
            } else if (tau_max < 1.0) {
              tao->trust = tau_max * PetscMin(tao->trust, bnk->dnorm);
            } else {
              tao->trust = PetscMax(tao->trust, tau_max * bnk->dnorm);
            }
          } else {
            /*  Not good agreement */
            if (tau_min > 1.0) {
              tao->trust = bnk->gamma2 * PetscMin(tao->trust, bnk->dnorm);
            } else if (tau_max < bnk->gamma1) {
              tao->trust = bnk->gamma1 * PetscMin(tao->trust, bnk->dnorm);
            } else if ((tau_min < bnk->gamma1) && (tau_max >= 1.0)) {
              tao->trust = bnk->gamma1 * PetscMin(tao->trust, bnk->dnorm);
            } else if ((tau_1 >= bnk->gamma1) && (tau_1 < 1.0) && ((tau_2 < bnk->gamma1) || (tau_2 >= 1.0))) {
              tao->trust = tau_1 * PetscMin(tao->trust, bnk->dnorm);
            } else if ((tau_2 >= bnk->gamma1) && (tau_2 < 1.0) && ((tau_1 < bnk->gamma1) || (tau_2 >= 1.0))) {
              tao->trust = tau_2 * PetscMin(tao->trust, bnk->dnorm);
            } else {
              tao->trust = tau_max * PetscMin(tao->trust, bnk->dnorm);
            }
          }
        }
      }
    } else {
      /*  Newton step was not good; reduce the radius */
      tao->trust = bnk->gamma1 * PetscMin(bnk->dnorm, tao->trust);
    }
    break;
  }
  /* Make sure the radius does not violate min and max settings */
  tao->trust = PetscMin(tao->trust, bnk->max_radius);
  tao->trust = PetscMax(tao->trust, bnk->min_radius);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------- */

PetscErrorCode TaoBNKAddStepCounts(Tao tao, PetscInt stepType)
{
  TAO_BNK        *bnk = (TAO_BNK *)tao->data;
  
  PetscFunctionBegin;
  switch (stepType) {
  case BNK_NEWTON:
    ++bnk->newt;
    break;
  case BNK_BFGS:
    ++bnk->bfgs;
    break;
  case BNK_SCALED_GRADIENT:
    ++bnk->sgrad;
    break;
  case BNK_GRADIENT:
    ++bnk->grad;
    break;
  default:
    break;
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------- */

PetscErrorCode TaoSetUp_BNK(Tao tao)
{
  TAO_BNK        *bnk = (TAO_BNK *)tao->data;
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  if (!tao->gradient) {
    ierr = VecDuplicate(tao->solution,&tao->gradient);CHKERRQ(ierr);
  }
  if (!tao->stepdirection) {
    ierr = VecDuplicate(tao->solution,&tao->stepdirection);CHKERRQ(ierr);
  }
  if (!bnk->W) {
    ierr = VecDuplicate(tao->solution,&bnk->W);CHKERRQ(ierr);
  }
  if (!bnk->Xold) {
    ierr = VecDuplicate(tao->solution,&bnk->Xold);CHKERRQ(ierr);
  }
  if (!bnk->Gold) {
    ierr = VecDuplicate(tao->solution,&bnk->Gold);CHKERRQ(ierr);
  }
  if (!bnk->Xwork) {
    ierr = VecDuplicate(tao->solution,&bnk->Xwork);CHKERRQ(ierr);
  }
  if (!bnk->Gwork) {
    ierr = VecDuplicate(tao->solution,&bnk->Gwork);CHKERRQ(ierr);
  }
  if (!bnk->unprojected_gradient) {
    ierr = VecDuplicate(tao->solution,&bnk->unprojected_gradient);CHKERRQ(ierr);
  }
  if (!bnk->unprojected_gradient_old) {
    ierr = VecDuplicate(tao->solution,&bnk->unprojected_gradient_old);CHKERRQ(ierr);
  }
  if (!bnk->Diag_min) {
    ierr = VecDuplicate(tao->solution,&bnk->Diag_min);CHKERRQ(ierr);
  }
  if (!bnk->Diag_max) {
    ierr = VecDuplicate(tao->solution,&bnk->Diag_max);CHKERRQ(ierr);
  }
  if (bnk->max_cg_its > 0) {
    /* Ensure that the important common vectors are shared between BNK and embedded BNCG */
    bnk->bncg_ctx = (TAO_BNCG *)bnk->bncg->data;
    ierr = PetscObjectReference((PetscObject)(bnk->unprojected_gradient_old));CHKERRQ(ierr);
    ierr = VecDestroy(&bnk->bncg_ctx->unprojected_gradient_old);CHKERRQ(ierr);
    bnk->bncg_ctx->unprojected_gradient_old = bnk->unprojected_gradient_old;
    ierr = PetscObjectReference((PetscObject)(bnk->unprojected_gradient));CHKERRQ(ierr);
    ierr = VecDestroy(&bnk->bncg_ctx->unprojected_gradient);CHKERRQ(ierr);
    bnk->bncg_ctx->unprojected_gradient = bnk->unprojected_gradient;
    ierr = PetscObjectReference((PetscObject)(bnk->Gold));CHKERRQ(ierr);
    ierr = VecDestroy(&bnk->bncg_ctx->G_old);CHKERRQ(ierr);
    bnk->bncg_ctx->G_old = bnk->Gold;
    ierr = PetscObjectReference((PetscObject)(tao->gradient));CHKERRQ(ierr);
    ierr = VecDestroy(&bnk->bncg->gradient);CHKERRQ(ierr);
    bnk->bncg->gradient = tao->gradient;
    ierr = PetscObjectReference((PetscObject)(tao->stepdirection));CHKERRQ(ierr);
    ierr = VecDestroy(&bnk->bncg->stepdirection);CHKERRQ(ierr);
    bnk->bncg->stepdirection = tao->stepdirection;
    ierr = TaoSetInitialVector(bnk->bncg, tao->solution);CHKERRQ(ierr);
    /* Copy over some settings from BNK into BNCG */
    ierr = TaoSetMaximumIterations(bnk->bncg, bnk->max_cg_its);CHKERRQ(ierr);
    ierr = TaoSetTolerances(bnk->bncg, tao->gatol, tao->grtol, tao->gttol);CHKERRQ(ierr);
    ierr = TaoSetFunctionLowerBound(bnk->bncg, tao->fmin);CHKERRQ(ierr);
    ierr = TaoSetConvergenceTest(bnk->bncg, tao->ops->convergencetest, tao->cnvP);CHKERRQ(ierr);
    ierr = TaoSetObjectiveRoutine(bnk->bncg, tao->ops->computeobjective, tao->user_objP);CHKERRQ(ierr);
    ierr = TaoSetGradientRoutine(bnk->bncg, tao->ops->computegradient, tao->user_gradP);CHKERRQ(ierr);
    ierr = TaoSetObjectiveAndGradientRoutine(bnk->bncg, tao->ops->computeobjectiveandgradient, tao->user_objgradP);CHKERRQ(ierr);
    ierr = PetscObjectCopyFortranFunctionPointers((PetscObject)tao, (PetscObject)(bnk->bncg));CHKERRQ(ierr);
    for (i=0; i<tao->numbermonitors; ++i) {
      ierr = TaoSetMonitor(bnk->bncg, tao->monitor[i], tao->monitorcontext[i], tao->monitordestroy[i]);CHKERRQ(ierr);
      ierr = PetscObjectReference((PetscObject)(tao->monitorcontext[i]));CHKERRQ(ierr);
    }
  }
  bnk->X_inactive = 0;
  bnk->G_inactive = 0;
  bnk->inactive_work = 0;
  bnk->active_work = 0;
  bnk->inactive_idx = 0;
  bnk->active_idx = 0;
  bnk->active_lower = 0;
  bnk->active_upper = 0;
  bnk->active_fixed = 0;
  bnk->M = 0;
  bnk->H_inactive = 0;
  bnk->Hpre_inactive = 0;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode TaoDestroy_BNK(Tao tao)
{
  TAO_BNK        *bnk = (TAO_BNK *)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (tao->setupcalled) {
    ierr = VecDestroy(&bnk->W);CHKERRQ(ierr);
    ierr = VecDestroy(&bnk->Xold);CHKERRQ(ierr);
    ierr = VecDestroy(&bnk->Gold);CHKERRQ(ierr);
    ierr = VecDestroy(&bnk->Xwork);CHKERRQ(ierr);
    ierr = VecDestroy(&bnk->Gwork);CHKERRQ(ierr);
    ierr = VecDestroy(&bnk->unprojected_gradient);CHKERRQ(ierr);
    ierr = VecDestroy(&bnk->unprojected_gradient_old);CHKERRQ(ierr);
    ierr = VecDestroy(&bnk->Diag_min);CHKERRQ(ierr);
    ierr = VecDestroy(&bnk->Diag_max);CHKERRQ(ierr);
  }
  ierr = ISDestroy(&bnk->active_lower);CHKERRQ(ierr);
  ierr = ISDestroy(&bnk->active_upper);CHKERRQ(ierr);
  ierr = ISDestroy(&bnk->active_fixed);CHKERRQ(ierr);
  ierr = ISDestroy(&bnk->active_idx);CHKERRQ(ierr);
  ierr = ISDestroy(&bnk->inactive_idx);CHKERRQ(ierr);
  ierr = MatDestroy(&bnk->Hpre_inactive);CHKERRQ(ierr);
  ierr = MatDestroy(&bnk->H_inactive);CHKERRQ(ierr);
  ierr = TaoDestroy(&bnk->bncg);CHKERRQ(ierr);
  ierr = PetscFree(tao->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode TaoSetFromOptions_BNK(PetscOptionItems *PetscOptionsObject,Tao tao)
{
  TAO_BNK        *bnk = (TAO_BNK *)tao->data;
  PetscErrorCode ierr;
  KSPType        ksp_type;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"Newton-Krylov method for bound constrained optimization");CHKERRQ(ierr);
  ierr = PetscOptionsEList("-tao_bnk_init_type", "radius initialization type", "", BNK_INIT, BNK_INIT_TYPES, BNK_INIT[bnk->init_type], &bnk->init_type, 0);CHKERRQ(ierr);
  ierr = PetscOptionsEList("-tao_bnk_update_type", "radius update type", "", BNK_UPDATE, BNK_UPDATE_TYPES, BNK_UPDATE[bnk->update_type], &bnk->update_type, 0);CHKERRQ(ierr);
  ierr = PetscOptionsEList("-tao_bnk_as_type", "active set estimation method", "", BNK_AS, BNK_AS_TYPES, BNK_AS[bnk->as_type], &bnk->as_type, 0);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bnk_sval", "(developer) Hessian perturbation starting value", "", bnk->sval, &bnk->sval,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bnk_imin", "(developer) minimum initial Hessian perturbation", "", bnk->imin, &bnk->imin,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bnk_imax", "(developer) maximum initial Hessian perturbation", "", bnk->imax, &bnk->imax,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bnk_imfac", "(developer) initial merit factor for Hessian perturbation", "", bnk->imfac, &bnk->imfac,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bnk_pmin", "(developer) minimum Hessian perturbation", "", bnk->pmin, &bnk->pmin,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bnk_pmax", "(developer) maximum Hessian perturbation", "", bnk->pmax, &bnk->pmax,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bnk_pgfac", "(developer) Hessian perturbation growth factor", "", bnk->pgfac, &bnk->pgfac,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bnk_psfac", "(developer) Hessian perturbation shrink factor", "", bnk->psfac, &bnk->psfac,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bnk_pmgfac", "(developer) merit growth factor for Hessian perturbation", "", bnk->pmgfac, &bnk->pmgfac,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bnk_pmsfac", "(developer) merit shrink factor for Hessian perturbation", "", bnk->pmsfac, &bnk->pmsfac,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bnk_eta1", "(developer) threshold for rejecting step (-tao_bnk_update_type reduction)", "", bnk->eta1, &bnk->eta1,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bnk_eta2", "(developer) threshold for accepting marginal step (-tao_bnk_update_type reduction)", "", bnk->eta2, &bnk->eta2,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bnk_eta3", "(developer) threshold for accepting reasonable step (-tao_bnk_update_type reduction)", "", bnk->eta3, &bnk->eta3,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bnk_eta4", "(developer) threshold for accepting good step (-tao_bnk_update_type reduction)", "", bnk->eta4, &bnk->eta4,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bnk_alpha1", "(developer) radius reduction factor for rejected step (-tao_bnk_update_type reduction)", "", bnk->alpha1, &bnk->alpha1,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bnk_alpha2", "(developer) radius reduction factor for marginally accepted bad step (-tao_bnk_update_type reduction)", "", bnk->alpha2, &bnk->alpha2,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bnk_alpha3", "(developer) radius increase factor for reasonable accepted step (-tao_bnk_update_type reduction)", "", bnk->alpha3, &bnk->alpha3,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bnk_alpha4", "(developer) radius increase factor for good accepted step (-tao_bnk_update_type reduction)", "", bnk->alpha4, &bnk->alpha4,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bnk_alpha5", "(developer) radius increase factor for very good accepted step (-tao_bnk_update_type reduction)", "", bnk->alpha5, &bnk->alpha5,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bnk_nu1", "(developer) threshold for small line-search step length (-tao_bnk_update_type step)", "", bnk->nu1, &bnk->nu1,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bnk_nu2", "(developer) threshold for reasonable line-search step length (-tao_bnk_update_type step)", "", bnk->nu2, &bnk->nu2,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bnk_nu3", "(developer) threshold for large line-search step length (-tao_bnk_update_type step)", "", bnk->nu3, &bnk->nu3,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bnk_nu4", "(developer) threshold for very large line-search step length (-tao_bnk_update_type step)", "", bnk->nu4, &bnk->nu4,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bnk_omega1", "(developer) radius reduction factor for very small line-search step length (-tao_bnk_update_type step)", "", bnk->omega1, &bnk->omega1,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bnk_omega2", "(developer) radius reduction factor for small line-search step length (-tao_bnk_update_type step)", "", bnk->omega2, &bnk->omega2,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bnk_omega3", "(developer) radius factor for decent line-search step length (-tao_bnk_update_type step)", "", bnk->omega3, &bnk->omega3,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bnk_omega4", "(developer) radius increase factor for large line-search step length (-tao_bnk_update_type step)", "", bnk->omega4, &bnk->omega4,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bnk_omega5", "(developer) radius increase factor for very large line-search step length (-tao_bnk_update_type step)", "", bnk->omega5, &bnk->omega5,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bnk_mu1_i", "(developer) threshold for accepting very good step (-tao_bnk_init_type interpolation)", "", bnk->mu1_i, &bnk->mu1_i,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bnk_mu2_i", "(developer) threshold for accepting good step (-tao_bnk_init_type interpolation)", "", bnk->mu2_i, &bnk->mu2_i,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bnk_gamma1_i", "(developer) radius reduction factor for rejected very bad step (-tao_bnk_init_type interpolation)", "", bnk->gamma1_i, &bnk->gamma1_i,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bnk_gamma2_i", "(developer) radius reduction factor for rejected bad step (-tao_bnk_init_type interpolation)", "", bnk->gamma2_i, &bnk->gamma2_i,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bnk_gamma3_i", "(developer) radius increase factor for accepted good step (-tao_bnk_init_type interpolation)", "", bnk->gamma3_i, &bnk->gamma3_i,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bnk_gamma4_i", "(developer) radius increase factor for accepted very good step (-tao_bnk_init_type interpolation)", "", bnk->gamma4_i, &bnk->gamma4_i,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bnk_theta_i", "(developer) trust region interpolation factor (-tao_bnk_init_type interpolation)", "", bnk->theta_i, &bnk->theta_i,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bnk_mu1", "(developer) threshold for accepting very good step (-tao_bnk_update_type interpolation)", "", bnk->mu1, &bnk->mu1,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bnk_mu2", "(developer) threshold for accepting good step (-tao_bnk_update_type interpolation)", "", bnk->mu2, &bnk->mu2,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bnk_gamma1", "(developer) radius reduction factor for rejected very bad step (-tao_bnk_update_type interpolation)", "", bnk->gamma1, &bnk->gamma1,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bnk_gamma2", "(developer) radius reduction factor for rejected bad step (-tao_bnk_update_type interpolation)", "", bnk->gamma2, &bnk->gamma2,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bnk_gamma3", "(developer) radius increase factor for accepted good step (-tao_bnk_update_type interpolation)", "", bnk->gamma3, &bnk->gamma3,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bnk_gamma4", "(developer) radius increase factor for accepted very good step (-tao_bnk_update_type interpolation)", "", bnk->gamma4, &bnk->gamma4,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bnk_theta", "(developer) trust region interpolation factor (-tao_bnk_update_type interpolation)", "", bnk->theta, &bnk->theta,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bnk_min_radius", "(developer) lower bound on initial radius", "", bnk->min_radius, &bnk->min_radius,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bnk_max_radius", "(developer) upper bound on radius", "", bnk->max_radius, &bnk->max_radius,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bnk_epsilon", "(developer) tolerance used when computing actual and predicted reduction", "", bnk->epsilon, &bnk->epsilon,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bnk_as_tol", "(developer) initial tolerance used when estimating actively bounded variables", "", bnk->as_tol, &bnk->as_tol,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bnk_as_step", "(developer) step length used when estimating actively bounded variables", "", bnk->as_step, &bnk->as_step,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-tao_bnk_max_cg_its", "number of BNCG iterations to take for each Newton step", "", bnk->max_cg_its, &bnk->max_cg_its,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  ierr = TaoSetFromOptions(bnk->bncg);CHKERRQ(ierr);
  ierr = TaoLineSearchSetFromOptions(tao->linesearch);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(tao->ksp);CHKERRQ(ierr);
  ierr = KSPGetType(tao->ksp,&ksp_type);CHKERRQ(ierr);
  ierr = PetscStrcmp(ksp_type,KSPNASH,&bnk->is_nash);CHKERRQ(ierr);
  ierr = PetscStrcmp(ksp_type,KSPSTCG,&bnk->is_stcg);CHKERRQ(ierr);
  ierr = PetscStrcmp(ksp_type,KSPGLTR,&bnk->is_gltr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode TaoView_BNK(Tao tao, PetscViewer viewer)
{
  TAO_BNK        *bnk = (TAO_BNK *)tao->data;
  PetscInt       nrejects;
  PetscBool      isascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    if (bnk->M) {
      ierr = MatLMVMGetRejectCount(bnk->M,&nrejects);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer, "Rejected BFGS updates: %D\n",nrejects);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer, "CG steps: %D\n", bnk->tot_cg_its);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "Newton steps: %D\n", bnk->newt);CHKERRQ(ierr);
    if (bnk->M) {
      ierr = PetscViewerASCIIPrintf(viewer, "BFGS steps: %D\n", bnk->bfgs);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer, "Scaled gradient steps: %D\n", bnk->sgrad);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "Gradient steps: %D\n", bnk->grad);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "KSP termination reasons:\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "  atol: %D\n", bnk->ksp_atol);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "  rtol: %D\n", bnk->ksp_rtol);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "  ctol: %D\n", bnk->ksp_ctol);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "  negc: %D\n", bnk->ksp_negc);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "  dtol: %D\n", bnk->ksp_dtol);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "  iter: %D\n", bnk->ksp_iter);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "  othr: %D\n", bnk->ksp_othr);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------- */

/*MC
  TAOBNK - Shared base-type for Bounded Newton-Krylov type algorithms.
  At each iteration, the BNK methods solve the symmetric
  system of equations to obtain the step diretion dk:
              Hk dk = -gk
  for free variables only. The step can be globalized either through 
  trust-region methods, or a line search, or a heuristic mixture of both.

    Options Database Keys:
+ -max_cg_its - maximum number of bounded conjugate-gradient iterations taken in each Newton loop
. -init_type - trust radius initialization method ("constant", "direction", "interpolation")
. -update_type - trust radius update method ("step", "direction", "interpolation")
. -as_type - active-set estimation method ("none", "bertsekas")
. -as_tol - (developer) initial tolerance used in estimating bounded active variables (-as_type bertsekas)
. -as_step - (developer) trial step length used in estimating bounded active variables (-as_type bertsekas)
. -sval - (developer) Hessian perturbation starting value
. -imin - (developer) minimum initial Hessian perturbation
. -imax - (developer) maximum initial Hessian perturbation
. -pmin - (developer) minimum Hessian perturbation
. -pmax - (developer) aximum Hessian perturbation
. -pgfac - (developer) Hessian perturbation growth factor
. -psfac - (developer) Hessian perturbation shrink factor
. -imfac - (developer) initial merit factor for Hessian perturbation
. -pmgfac - (developer) merit growth factor for Hessian perturbation
. -pmsfac - (developer) merit shrink factor for Hessian perturbation
. -eta1 - (developer) threshold for rejecting step (-update_type reduction)
. -eta2 - (developer) threshold for accepting marginal step (-update_type reduction)
. -eta3 - (developer) threshold for accepting reasonable step (-update_type reduction)
. -eta4 - (developer) threshold for accepting good step (-update_type reduction)
. -alpha1 - (developer) radius reduction factor for rejected step (-update_type reduction)
. -alpha2 - (developer) radius reduction factor for marginally accepted bad step (-update_type reduction)
. -alpha3 - (developer) radius increase factor for reasonable accepted step (-update_type reduction)
. -alpha4 - (developer) radius increase factor for good accepted step (-update_type reduction)
. -alpha5 - (developer) radius increase factor for very good accepted step (-update_type reduction)
. -epsilon - (developer) tolerance for small pred/actual ratios that trigger automatic step acceptance (-update_type reduction)
. -mu1 - (developer) threshold for accepting very good step (-update_type interpolation)
. -mu2 - (developer) threshold for accepting good step (-update_type interpolation)
. -gamma1 - (developer) radius reduction factor for rejected very bad step (-update_type interpolation)
. -gamma2 - (developer) radius reduction factor for rejected bad step (-update_type interpolation)
. -gamma3 - (developer) radius increase factor for accepted good step (-update_type interpolation)
. -gamma4 - (developer) radius increase factor for accepted very good step (-update_type interpolation)
. -theta - (developer) trust region interpolation factor (-update_type interpolation)
. -nu1 - (developer) threshold for small line-search step length (-update_type step)
. -nu2 - (developer) threshold for reasonable line-search step length (-update_type step)
. -nu3 - (developer) threshold for large line-search step length (-update_type step)
. -nu4 - (developer) threshold for very large line-search step length (-update_type step)
. -omega1 - (developer) radius reduction factor for very small line-search step length (-update_type step)
. -omega2 - (developer) radius reduction factor for small line-search step length (-update_type step)
. -omega3 - (developer) radius factor for decent line-search step length (-update_type step)
. -omega4 - (developer) radius increase factor for large line-search step length (-update_type step)
. -omega5 - (developer) radius increase factor for very large line-search step length (-update_type step)
. -mu1_i -  (developer) threshold for accepting very good step (-init_type interpolation)
. -mu2_i -  (developer) threshold for accepting good step (-init_type interpolation)
. -gamma1_i - (developer) radius reduction factor for rejected very bad step (-init_type interpolation)
. -gamma2_i - (developer) radius reduction factor for rejected bad step (-init_type interpolation)
. -gamma3_i - (developer) radius increase factor for accepted good step (-init_type interpolation)
. -gamma4_i - (developer) radius increase factor for accepted very good step (-init_type interpolation)
- -theta_i - (developer) trust region interpolation factor (-init_type interpolation)

  Level: beginner
M*/

PetscErrorCode TaoCreate_BNK(Tao tao)
{
  TAO_BNK        *bnk;
  const char     *morethuente_type = TAOLINESEARCHMT;
  PetscErrorCode ierr;
  PC             pc;

  PetscFunctionBegin;
  ierr = PetscNewLog(tao,&bnk);CHKERRQ(ierr);

  tao->ops->setup = TaoSetUp_BNK;
  tao->ops->view = TaoView_BNK;
  tao->ops->setfromoptions = TaoSetFromOptions_BNK;
  tao->ops->destroy = TaoDestroy_BNK;

  /*  Override default settings (unless already changed) */
  if (!tao->max_it_changed) tao->max_it = 50;
  if (!tao->trust0_changed) tao->trust0 = 100.0;

  tao->data = (void*)bnk;
  
  /*  Hessian shifting parameters */
  bnk->computehessian = TaoBNKComputeHessian;
  bnk->computestep = TaoBNKComputeStep;
  
  bnk->sval   = 0.0;
  bnk->imin   = 1.0e-4;
  bnk->imax   = 1.0e+2;
  bnk->imfac  = 1.0e-1;

  bnk->pmin   = 1.0e-12;
  bnk->pmax   = 1.0e+2;
  bnk->pgfac  = 1.0e+1;
  bnk->psfac  = 4.0e-1;
  bnk->pmgfac = 1.0e-1;
  bnk->pmsfac = 1.0e-1;

  /*  Default values for trust-region radius update based on steplength */
  bnk->nu1 = 0.25;
  bnk->nu2 = 0.50;
  bnk->nu3 = 1.00;
  bnk->nu4 = 1.25;

  bnk->omega1 = 0.25;
  bnk->omega2 = 0.50;
  bnk->omega3 = 1.00;
  bnk->omega4 = 2.00;
  bnk->omega5 = 4.00;

  /*  Default values for trust-region radius update based on reduction */
  bnk->eta1 = 1.0e-4;
  bnk->eta2 = 0.25;
  bnk->eta3 = 0.50;
  bnk->eta4 = 0.90;

  bnk->alpha1 = 0.25;
  bnk->alpha2 = 0.50;
  bnk->alpha3 = 1.00;
  bnk->alpha4 = 2.00;
  bnk->alpha5 = 4.00;

  /*  Default values for trust-region radius update based on interpolation */
  bnk->mu1 = 0.10;
  bnk->mu2 = 0.50;

  bnk->gamma1 = 0.25;
  bnk->gamma2 = 0.50;
  bnk->gamma3 = 2.00;
  bnk->gamma4 = 4.00;

  bnk->theta = 0.05;

  /*  Default values for trust region initialization based on interpolation */
  bnk->mu1_i = 0.35;
  bnk->mu2_i = 0.50;

  bnk->gamma1_i = 0.0625;
  bnk->gamma2_i = 0.5;
  bnk->gamma3_i = 2.0;
  bnk->gamma4_i = 5.0;

  bnk->theta_i = 0.25;

  /*  Remaining parameters */
  bnk->max_cg_its = 0;
  bnk->min_radius = 1.0e-10;
  bnk->max_radius = 1.0e10;
  bnk->epsilon = PetscPowReal(PETSC_MACHINE_EPSILON, 2.0/3.0);
  bnk->as_tol = 1.0e-3;
  bnk->as_step = 1.0e-3;
  bnk->dmin = 1.0e-6;
  bnk->dmax = 1.0e6;
  
  bnk->M               = 0;
  bnk->bfgs_pre        = 0;
  bnk->init_type       = BNK_INIT_INTERPOLATION;
  bnk->update_type     = BNK_UPDATE_REDUCTION;
  bnk->as_type         = BNK_AS_BERTSEKAS;
  
  /* Create the embedded BNCG solver */
  ierr = TaoCreate(PetscObjectComm((PetscObject)tao), &bnk->bncg);CHKERRQ(ierr);
  ierr = PetscObjectIncrementTabLevel((PetscObject)bnk->bncg, (PetscObject)tao, 1);CHKERRQ(ierr);
  ierr = TaoSetOptionsPrefix(bnk->bncg, "tao_bnk_");CHKERRQ(ierr);
  ierr = TaoSetType(bnk->bncg, TAOBNCG);CHKERRQ(ierr);

  /* Create the line search */
  ierr = TaoLineSearchCreate(((PetscObject)tao)->comm,&tao->linesearch);CHKERRQ(ierr);
  ierr = PetscObjectIncrementTabLevel((PetscObject)tao->linesearch, (PetscObject)tao, 1);CHKERRQ(ierr);
  ierr = TaoLineSearchSetOptionsPrefix(tao->linesearch,tao->hdr.prefix);CHKERRQ(ierr);
  ierr = TaoLineSearchSetType(tao->linesearch,morethuente_type);CHKERRQ(ierr);
  ierr = TaoLineSearchUseTaoRoutines(tao->linesearch,tao);CHKERRQ(ierr);

  /*  Set linear solver to default for symmetric matrices */
  ierr = KSPCreate(((PetscObject)tao)->comm,&tao->ksp);CHKERRQ(ierr);
  ierr = PetscObjectIncrementTabLevel((PetscObject)tao->ksp, (PetscObject)tao, 1);CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(tao->ksp,"tao_bnk_");CHKERRQ(ierr);
  ierr = KSPSetType(tao->ksp,KSPSTCG);CHKERRQ(ierr);
  ierr = KSPGetPC(tao->ksp, &pc);CHKERRQ(ierr);
  ierr = PCSetType(pc, PCLMVM);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
