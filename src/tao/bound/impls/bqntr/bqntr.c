#include <../src/tao/bound/impls/bnk/bnk.h>
#include <petscksp.h>

static const char *BNK_INIT[64] = {"constant", "direction"};
static const char *BNK_UPDATE[64] = {"step", "reduction", "interpolation"};
static const char *BNK_AS[64] = {"none", "bertsekas"};

/*
  Quasi-Newton Trust-Region CG algorithm.
*/

PETSC_INTERN PetscErrorCode TaoSolve_BQNTR(Tao tao)
{
  PetscErrorCode               ierr;
  TAO_BNK                      *bqntr = (TAO_BNK *)tao->data;
  KSPConvergedReason           ksp_reason;

  PetscReal                    oldTrust, prered, actred, steplen, resnorm;
  PetscBool                    stepAccepted;
  PetscInt                     stepType = BNK_NEWTON, nDiff;
  
  PetscFunctionBegin;
  /* Initialize the preconditioner, KSP solver and trust radius/line search */
  tao->reason = TAO_CONTINUE_ITERATING;
  /* Project the current point onto the feasible set */
  ierr = TaoComputeVariableBounds(tao);CHKERRQ(ierr);
  ierr = TaoSetVariableBounds(bqntr->bncg, tao->XL, tao->XU);CHKERRQ(ierr);
  if (tao->bounded) {
    ierr = TaoLineSearchSetVariableBounds(tao->linesearch,tao->XL,tao->XU);CHKERRQ(ierr);
  }

  /* Project the initial point onto the feasible region */
  ierr = TaoBoundSolution(tao->solution, tao->XL,tao->XU, 0.0, &nDiff, tao->solution);CHKERRQ(ierr);

  /* Check convergence criteria */
  ierr = TaoComputeObjectiveAndGradient(tao, tao->solution, &bqntr->f, bqntr->unprojected_gradient);CHKERRQ(ierr);
  ierr = TaoBNKEstimateActiveSet(tao, bqntr->as_type);CHKERRQ(ierr);
  ierr = VecCopy(bqntr->unprojected_gradient, tao->gradient);CHKERRQ(ierr);
  ierr = VecISSet(tao->gradient, bqntr->active_idx, 0.0);CHKERRQ(ierr);
  ierr = VecNorm(tao->gradient,NORM_2,&bqntr->gnorm);CHKERRQ(ierr);

  /* Test the initial point for convergence */
  ierr = VecFischer(tao->solution, bqntr->unprojected_gradient, tao->XL, tao->XU, bqntr->W);CHKERRQ(ierr);
  ierr = VecNorm(bqntr->W, NORM_2, &resnorm);CHKERRQ(ierr);
  if (PetscIsInfOrNanReal(bqntr->f) || PetscIsInfOrNanReal(resnorm)) SETERRQ(PETSC_COMM_SELF,1, "User provided compute function generated Inf or NaN");
  ierr = TaoLogConvergenceHistory(tao,bqntr->f,resnorm,0.0,tao->ksp_its);CHKERRQ(ierr);
  ierr = TaoMonitor(tao,tao->niter,bqntr->f,resnorm,0.0,1.0);CHKERRQ(ierr);
  ierr = (*tao->ops->convergencetest)(tao,tao->cnvP);CHKERRQ(ierr);
  if (tao->reason != TAO_CONTINUE_ITERATING) PetscFunctionReturn(0);

  /* Reset KSP stopping reason counters */
  bqntr->ksp_atol = 0;
  bqntr->ksp_rtol = 0;
  bqntr->ksp_dtol = 0;
  bqntr->ksp_ctol = 0;
  bqntr->ksp_negc = 0;
  bqntr->ksp_iter = 0;
  bqntr->ksp_othr = 0;

  /* Reset initial steplength to zero */
  ierr = VecSet(tao->stepdirection, 0.0);CHKERRQ(ierr);
  if (tao->reason != TAO_CONTINUE_ITERATING) PetscFunctionReturn(0);
  
  /* initialize trust radius */
  switch (bqntr->init_type) {
  case BNK_INIT_CONSTANT:
    tao->trust = tao->trust0;
    break;
  
  default:
    tao->trust = 0.0;
    break;
  }

  /* Have not converged; continue with Newton method */
  while (tao->reason == TAO_CONTINUE_ITERATING) {
    ++tao->niter;
    
    /* Update the LMVM matrix */
    ierr = MatLMVMUpdate(tao->hessian, tao->solution, bqntr->unprojected_gradient);CHKERRQ(ierr);
    
    /* Store current solution before it changes */
    bqntr->fold = bqntr->f;
    ierr = VecCopy(tao->solution, bqntr->Xold);CHKERRQ(ierr);
    ierr = VecCopy(tao->gradient, bqntr->Gold);CHKERRQ(ierr);
    ierr = VecCopy(bqntr->unprojected_gradient, bqntr->unprojected_gradient_old);CHKERRQ(ierr);
    
    /* Enter into trust region loops */
    stepAccepted = PETSC_FALSE;
    while (!stepAccepted && tao->reason == TAO_CONTINUE_ITERATING) {
      if (!bqntr->inactive_idx) {
        /* If there are no inactive variables left, save some computation and return an adjusted zero step
        that has (l-x) and (u-x) for lower and upper bounded variables. */
        ierr = VecSet(tao->stepdirection, 0.0);CHKERRQ(ierr);
        ierr = TaoBNKBoundStep(tao, bqntr->as_type, tao->stepdirection);CHKERRQ(ierr);
      } else {
        /* Prepare the reduced sub-matrices for the inactive set */
        ierr = MatDestroy(&bqntr->H_inactive);CHKERRQ(ierr);
        if (bqntr->active_idx) {
          ierr = MatCreateSubMatrixVirtual(tao->hessian, bqntr->inactive_idx, bqntr->inactive_idx, &bqntr->H_inactive);CHKERRQ(ierr);
        } else {
          ierr = PetscObjectReference((PetscObject)tao->hessian);CHKERRQ(ierr);
          bqntr->H_inactive = tao->hessian;
        }
        
        /* Solve the Newton system of equations */
        ierr = VecSet(tao->stepdirection, 0.0);CHKERRQ(ierr);
        ierr = KSPReset(tao->ksp);CHKERRQ(ierr);
        ierr = KSPSetOperators(tao->ksp,bqntr->H_inactive,bqntr->H_inactive);CHKERRQ(ierr);
        ierr = VecCopy(bqntr->unprojected_gradient, bqntr->Gwork);CHKERRQ(ierr);
        if (bqntr->active_idx) {
          ierr = VecGetSubVector(bqntr->Gwork, bqntr->inactive_idx, &bqntr->G_inactive);CHKERRQ(ierr);
          ierr = VecGetSubVector(tao->stepdirection, bqntr->inactive_idx, &bqntr->X_inactive);CHKERRQ(ierr);
        } else {
          bqntr->G_inactive = bqntr->unprojected_gradient;
          bqntr->X_inactive = tao->stepdirection;
        }
        if (bqntr->is_nash || bqntr->is_stcg || bqntr->is_gltr) {
          ierr = KSPCGSetRadius(tao->ksp,tao->trust);CHKERRQ(ierr);
          ierr = KSPSolve(tao->ksp, bqntr->G_inactive, bqntr->X_inactive);CHKERRQ(ierr);
          ierr = KSPGetIterationNumber(tao->ksp,&tao->ksp_its);CHKERRQ(ierr);
          tao->ksp_tot_its+=tao->ksp_its;
          ierr = KSPCGGetNormD(tao->ksp,&bqntr->dnorm);CHKERRQ(ierr);

          if (0.0 == tao->trust) {
            /* Radius was uninitialized; use the norm of the direction */
            if (bqntr->dnorm > 0.0) {
              tao->trust = bqntr->dnorm;

              /* Modify the radius if it is too large or small */
              tao->trust = PetscMax(tao->trust, bqntr->min_radius);
              tao->trust = PetscMin(tao->trust, bqntr->max_radius);
            } else {
              /* The direction was bad; set radius to default value and re-solve
                 the trust-region subproblem to get a direction */
              tao->trust = tao->trust0;

              /* Modify the radius if it is too large or small */
              tao->trust = PetscMax(tao->trust, bqntr->min_radius);
              tao->trust = PetscMin(tao->trust, bqntr->max_radius);

              ierr = KSPCGSetRadius(tao->ksp,tao->trust);CHKERRQ(ierr);
              ierr = KSPSolve(tao->ksp, bqntr->G_inactive, bqntr->X_inactive);CHKERRQ(ierr);
              ierr = KSPGetIterationNumber(tao->ksp,&tao->ksp_its);CHKERRQ(ierr);
              tao->ksp_tot_its+=tao->ksp_its;
              ierr = KSPCGGetNormD(tao->ksp,&bqntr->dnorm);CHKERRQ(ierr);

              if (bqntr->dnorm == 0.0) SETERRQ(PETSC_COMM_SELF,1, "Initial direction zero");
            }
          }
        } else {
          ierr = KSPSolve(tao->ksp, bqntr->G_inactive, bqntr->X_inactive);CHKERRQ(ierr);
          ierr = KSPGetIterationNumber(tao->ksp, &tao->ksp_its);CHKERRQ(ierr);
          tao->ksp_tot_its+=tao->ksp_its;
        }
        /* Restore sub vectors back */
        if (bqntr->active_idx) {
          ierr = VecRestoreSubVector(bqntr->Gwork, bqntr->inactive_idx, &bqntr->G_inactive);CHKERRQ(ierr);
          ierr = VecRestoreSubVector(tao->stepdirection, bqntr->inactive_idx, &bqntr->X_inactive);CHKERRQ(ierr);
        }
        /* Make sure the safeguarded fall-back step is zero for actively bounded variables */
        ierr = VecScale(tao->stepdirection, -1.0);CHKERRQ(ierr);
        ierr = TaoBNKBoundStep(tao, bqntr->as_type, tao->stepdirection);CHKERRQ(ierr);
        
        /* Record convergence reasons */
        ierr = KSPGetConvergedReason(tao->ksp, &ksp_reason);CHKERRQ(ierr);
        if (KSP_CONVERGED_ATOL == ksp_reason) {
          ++bqntr->ksp_atol;
        } else if (KSP_CONVERGED_RTOL == ksp_reason) {
          ++bqntr->ksp_rtol;
        } else if (KSP_CONVERGED_CG_CONSTRAINED == ksp_reason) {
          ++bqntr->ksp_ctol;
        } else if (KSP_CONVERGED_CG_NEG_CURVE == ksp_reason) {
          ++bqntr->ksp_negc;
        } else if (KSP_DIVERGED_DTOL == ksp_reason) {
          ++bqntr->ksp_dtol;
        } else if (KSP_DIVERGED_ITS == ksp_reason) {
          ++bqntr->ksp_iter;
        } else {
          ++bqntr->ksp_othr;
        }
        
        /* If the KSP solution failed, reset the LMVM matrix */
        if (ksp_reason < 0) {
          ierr = MatLMVMReset(tao->hessian, PETSC_FALSE);
          ierr = MatLMVMUpdate(tao->hessian, tao->solution, bqntr->unprojected_gradient);
        }
      }

      /* Temporarily accept the step and project it into the bounds */
      ierr = VecAXPY(tao->solution, 1.0, tao->stepdirection);CHKERRQ(ierr);
      ierr = TaoBoundSolution(tao->solution, tao->XL,tao->XU, 0.0, &nDiff, tao->solution);CHKERRQ(ierr);

      /* Check if the projection changed the step direction */
      if (nDiff > 0) {
        /* Projection changed the step, so we have to recompute the step and 
           the predicted reduction. Leave the trust radius unchanged. */
        ierr = VecCopy(tao->solution, tao->stepdirection);CHKERRQ(ierr);
        ierr = VecAXPY(tao->stepdirection, -1.0, bqntr->Xold);CHKERRQ(ierr);
        ierr = TaoBNKRecomputePred(tao, tao->stepdirection, &prered);CHKERRQ(ierr);
      } else {
        /* Step did not change, so we can just recover the pre-computed prediction */
        ierr = KSPCGGetObjFcn(tao->ksp, &prered);CHKERRQ(ierr);
      }
      prered = -prered;

      /* Compute the actual reduction and update the trust radius */
      ierr = TaoComputeObjective(tao, tao->solution, &bqntr->f);CHKERRQ(ierr);
      if (PetscIsInfOrNanReal(bqntr->f)) SETERRQ(PETSC_COMM_SELF,1, "User provided compute function generated Inf or NaN");
      actred = bqntr->fold - bqntr->f;
      oldTrust = tao->trust;
      ierr = TaoBNKUpdateTrustRadius(tao, prered, actred, bqntr->update_type, stepType, &stepAccepted);CHKERRQ(ierr);

      if (stepAccepted) {
        /* Step is good, evaluate the gradient and flip the need-Hessian switch */
        steplen = 1.0;
        ++bqntr->newt;
        ierr = TaoComputeGradient(tao, tao->solution, bqntr->unprojected_gradient);CHKERRQ(ierr);
        ierr = TaoBNKEstimateActiveSet(tao, bqntr->as_type);CHKERRQ(ierr);
        ierr = VecCopy(bqntr->unprojected_gradient, tao->gradient);CHKERRQ(ierr);
        ierr = VecISSet(tao->gradient, bqntr->active_idx, 0.0);CHKERRQ(ierr);
        ierr = VecNorm(tao->gradient, NORM_2, &bqntr->gnorm);CHKERRQ(ierr);
      } else {
        /* Step is bad, revert old solution and re-solve with new radius*/
        steplen = 0.0;
        bqntr->f = bqntr->fold;
        ierr = VecCopy(bqntr->Xold, tao->solution);CHKERRQ(ierr);
        ierr = VecCopy(bqntr->Gold, tao->gradient);CHKERRQ(ierr);
        ierr = VecCopy(bqntr->unprojected_gradient_old, bqntr->unprojected_gradient);CHKERRQ(ierr);
        if (oldTrust == tao->trust) {
          /* Can't change the radius anymore so just terminate */
          tao->reason = TAO_DIVERGED_TR_REDUCTION;
        }
      }

      /*  Check for termination */
      ierr = VecFischer(tao->solution, bqntr->unprojected_gradient, tao->XL, tao->XU, bqntr->W);CHKERRQ(ierr);
      ierr = VecNorm(bqntr->W, NORM_2, &resnorm);CHKERRQ(ierr);
      if (PetscIsInfOrNanReal(resnorm)) SETERRQ(PETSC_COMM_SELF,1, "User provided compute function generated Inf or NaN");
      ierr = TaoLogConvergenceHistory(tao, bqntr->f, resnorm, 0.0, tao->ksp_its);CHKERRQ(ierr);
      ierr = TaoMonitor(tao, tao->niter, bqntr->f, resnorm, 0.0, steplen);CHKERRQ(ierr);
      ierr = (*tao->ops->convergencetest)(tao, tao->cnvP);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PETSC_INTERN PetscErrorCode TaoSetFromOptions_BQNTR(PetscOptionItems *PetscOptionsObject,Tao tao)
{
  TAO_BNK        *bqntr = (TAO_BNK *)tao->data;
  PetscErrorCode ierr;
  PC             pc;
  PetscBool      is_lmvm, is_sym;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"Newton line search method for unconstrained optimization");CHKERRQ(ierr);
  ierr = PetscOptionsEList("-tao_bqntr_init_type", "radius initialization type", "", BNK_INIT, BNK_INIT_TYPES, BNK_INIT[bqntr->init_type], &bqntr->init_type, 0);CHKERRQ(ierr);
  ierr = PetscOptionsEList("-tao_bqntr_update_type", "radius update type", "", BNK_UPDATE, BNK_UPDATE_TYPES, BNK_UPDATE[bqntr->update_type], &bqntr->update_type, 0);CHKERRQ(ierr);
  ierr = PetscOptionsEList("-tao_bqntr_as_type", "active set estimation method", "", BNK_AS, BNK_AS_TYPES, BNK_AS[bqntr->as_type], &bqntr->as_type, 0);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bqntr_eta1", "(developer) threshold for rejecting step (-tao_bqntr_update_type reduction)", "", bqntr->eta1, &bqntr->eta1,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bqntr_eta2", "(developer) threshold for accepting marginal step (-tao_bqntr_update_type reduction)", "", bqntr->eta2, &bqntr->eta2,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bqntr_eta3", "(developer) threshold for accepting reasonable step (-tao_bqntr_update_type reduction)", "", bqntr->eta3, &bqntr->eta3,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bqntr_eta4", "(developer) threshold for accepting good step (-tao_bqntr_update_type reduction)", "", bqntr->eta4, &bqntr->eta4,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bqntr_alpha1", "(developer) radius reduction factor for rejected step (-tao_bqntr_update_type reduction)", "", bqntr->alpha1, &bqntr->alpha1,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bqntr_alpha2", "(developer) radius reduction factor for marginally accepted bad step (-tao_bqntr_update_type reduction)", "", bqntr->alpha2, &bqntr->alpha2,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bqntr_alpha3", "(developer) radius increase factor for reasonable accepted step (-tao_bqntr_update_type reduction)", "", bqntr->alpha3, &bqntr->alpha3,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bqntr_alpha4", "(developer) radius increase factor for good accepted step (-tao_bqntr_update_type reduction)", "", bqntr->alpha4, &bqntr->alpha4,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bqntr_alpha5", "(developer) radius increase factor for very good accepted step (-tao_bqntr_update_type reduction)", "", bqntr->alpha5, &bqntr->alpha5,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bqntr_mu1_i", "(developer) threshold for accepting very good step (-tao_bqntr_init_type interpolation)", "", bqntr->mu1_i, &bqntr->mu1_i,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bqntr_mu2_i", "(developer) threshold for accepting good step (-tao_bqntr_init_type interpolation)", "", bqntr->mu2_i, &bqntr->mu2_i,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bqntr_gamma1_i", "(developer) radius reduction factor for rejected very bad step (-tao_bqntr_init_type interpolation)", "", bqntr->gamma1_i, &bqntr->gamma1_i,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bqntr_gamma2_i", "(developer) radius reduction factor for rejected bad step (-tao_bqntr_init_type interpolation)", "", bqntr->gamma2_i, &bqntr->gamma2_i,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bqntr_gamma3_i", "(developer) radius increase factor for accepted good step (-tao_bqntr_init_type interpolation)", "", bqntr->gamma3_i, &bqntr->gamma3_i,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bqntr_gamma4_i", "(developer) radius increase factor for accepted very good step (-tao_bqntr_init_type interpolation)", "", bqntr->gamma4_i, &bqntr->gamma4_i,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bqntr_theta_i", "(developer) trust region interpolation factor (-tao_bqntr_init_type interpolation)", "", bqntr->theta_i, &bqntr->theta_i,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bqntr_mu1", "(developer) threshold for accepting very good step (-tao_bqntr_update_type interpolation)", "", bqntr->mu1, &bqntr->mu1,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bqntr_mu2", "(developer) threshold for accepting good step (-tao_bqntr_update_type interpolation)", "", bqntr->mu2, &bqntr->mu2,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bqntr_gamma1", "(developer) radius reduction factor for rejected very bad step (-tao_bqntr_update_type interpolation)", "", bqntr->gamma1, &bqntr->gamma1,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bqntr_gamma2", "(developer) radius reduction factor for rejected bad step (-tao_bqntr_update_type interpolation)", "", bqntr->gamma2, &bqntr->gamma2,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bqntr_gamma3", "(developer) radius increase factor for accepted good step (-tao_bqntr_update_type interpolation)", "", bqntr->gamma3, &bqntr->gamma3,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bqntr_gamma4", "(developer) radius increase factor for accepted very good step (-tao_bqntr_update_type interpolation)", "", bqntr->gamma4, &bqntr->gamma4,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bqntr_theta", "(developer) trust region interpolation factor (-tao_bqntr_update_type interpolation)", "", bqntr->theta, &bqntr->theta,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bqntr_min_radius", "(developer) lower bound on initial radius", "", bqntr->min_radius, &bqntr->min_radius,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bqntr_max_radius", "(developer) upper bound on radius", "", bqntr->max_radius, &bqntr->max_radius,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bqntr_epsilon", "(developer) tolerance used when computing actual and predicted reduction", "", bqntr->epsilon, &bqntr->epsilon,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bqntr_as_tol", "(developer) initial tolerance used when estimating actively bounded variables", "", bqntr->as_tol, &bqntr->as_tol,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bqntr_as_step", "(developer) step length used when estimating actively bounded variables", "", bqntr->as_step, &bqntr->as_step,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-tao_bqntr_max_cg_its", "number of BNCG iterations to take for each Newton step", "", bqntr->max_cg_its, &bqntr->max_cg_its,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  ierr = KSPSetFromOptions(tao->ksp);CHKERRQ(ierr);
  ierr = KSPGetPC(tao->ksp, &pc);CHKERRQ(ierr);
  ierr = PCSetType(pc, PCNONE);CHKERRQ(ierr);
  ierr = PetscObjectBaseTypeCompare((PetscObject)tao->hessian, MATLMVM, &is_lmvm);CHKERRQ(ierr);
  ierr = MatSetFromOptions(tao->hessian);CHKERRQ(ierr);
  if (!is_lmvm) SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_INCOMP, "Matrix must be an LMVM-type");
  ierr = MatGetOption(tao->hessian, MAT_SYMMETRIC, &is_sym);CHKERRQ(ierr);
  if (!is_sym) SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_INCOMP, "LMVM matrix must be symmetric");
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PETSC_INTERN PetscErrorCode TaoView_BQNTR(Tao tao, PetscViewer viewer)
{
  TAO_BNK        *bqntr = (TAO_BNK *)tao->data;
  PetscBool      isascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer, "KSP termination reasons:\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "  atol: %D\n", bqntr->ksp_atol);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "  rtol: %D\n", bqntr->ksp_rtol);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "  ctol: %D\n", bqntr->ksp_ctol);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "  negc: %D\n", bqntr->ksp_negc);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "  dtol: %D\n", bqntr->ksp_dtol);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "  iter: %D\n", bqntr->ksp_iter);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "  othr: %D\n", bqntr->ksp_othr);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PETSC_INTERN PetscErrorCode TaoSetUp_BQNTR(Tao tao)
{
  TAO_BNK        *bqntr = (TAO_BNK *)tao->data;
  PetscErrorCode ierr;
  PetscInt       n, N;

  PetscFunctionBegin;
  ierr = TaoSetUp_BNK(tao);CHKERRQ(ierr);
  if (!bqntr->is_nash && !bqntr->is_stcg && !bqntr->is_gltr) SETERRQ(PETSC_COMM_SELF,1,"Must use a trust-region CG method for KSP (KSPNASH, KSPSTCG, KSPGLTR)");
  /* Create matrix for the limited memory approximation */
  ierr = VecGetLocalSize(tao->solution,&n);CHKERRQ(ierr);
  ierr = VecGetSize(tao->solution,&N);CHKERRQ(ierr);
  ierr = MatSetSizes(tao->hessian, n, n, N, N);CHKERRQ(ierr);
  ierr = MatLMVMAllocate(tao->hessian,tao->solution,bqntr->unprojected_gradient);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode TaoDestroy_BQNTR(Tao tao)
{
  TAO_BNK        *bqntr = (TAO_BNK *)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (tao->setupcalled) {
    ierr = VecDestroy(&bqntr->W);CHKERRQ(ierr);
    ierr = VecDestroy(&bqntr->Xold);CHKERRQ(ierr);
    ierr = VecDestroy(&bqntr->Gold);CHKERRQ(ierr);
    ierr = VecDestroy(&bqntr->Xwork);CHKERRQ(ierr);
    ierr = VecDestroy(&bqntr->Gwork);CHKERRQ(ierr);
    ierr = VecDestroy(&bqntr->unprojected_gradient);CHKERRQ(ierr);
    ierr = VecDestroy(&bqntr->unprojected_gradient_old);CHKERRQ(ierr);
    ierr = VecDestroy(&bqntr->Diag_min);CHKERRQ(ierr);
    ierr = VecDestroy(&bqntr->Diag_max);CHKERRQ(ierr);
  }
  ierr = ISDestroy(&bqntr->active_lower);CHKERRQ(ierr);
  ierr = ISDestroy(&bqntr->active_upper);CHKERRQ(ierr);
  ierr = ISDestroy(&bqntr->active_fixed);CHKERRQ(ierr);
  ierr = ISDestroy(&bqntr->active_idx);CHKERRQ(ierr);
  ierr = ISDestroy(&bqntr->inactive_idx);CHKERRQ(ierr);
  ierr = MatDestroy(&bqntr->H_inactive);CHKERRQ(ierr);
  ierr = TaoDestroy(&bqntr->bncg);CHKERRQ(ierr);
  ierr = PetscFree(tao->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PETSC_EXTERN PetscErrorCode TaoCreate_BQNTR(Tao tao)
{
  TAO_BNK        *bqntr;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = TaoCreate_BNK(tao);CHKERRQ(ierr);
  tao->ops->solve=TaoSolve_BQNTR;
  tao->ops->setup=TaoSetUp_BQNTR;
  tao->ops->view=TaoView_BQNTR;
  tao->ops->setfromoptions=TaoSetFromOptions_BQNTR;
  tao->ops->destroy=TaoDestroy_BQNTR;
  
  bqntr = (TAO_BNK *)tao->data;
  bqntr->as_type = BNK_AS_BERTSEKAS;
  bqntr->init_type = BNK_INIT_DIRECTION;
  bqntr->update_type = BNK_UPDATE_REDUCTION; /* trust region updates based on predicted/actual reduction */
  
  ierr = MatDestroy(&tao->hessian);CHKERRQ(ierr);
  ierr = MatCreate(((PetscObject)tao)->comm, &tao->hessian);CHKERRQ(ierr);
  ierr = PetscObjectIncrementTabLevel((PetscObject)tao->hessian, (PetscObject)tao, 1);CHKERRQ(ierr);
  ierr = MatSetType(tao->hessian, MATLMVMSR1);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(tao->hessian, "tao_bqntr_");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
