#include <petsctaolinesearch.h>
#include <../src/tao/bound/impls/bnk/bnk.h>

#include <petscksp.h>

/* Routine for BFGS preconditioner */

PetscErrorCode MatLMVMSolveShell(PC pc, Vec b, Vec x)
{
  PetscErrorCode ierr;
  Mat            M;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(b,VEC_CLASSID,2);
  PetscValidHeaderSpecific(x,VEC_CLASSID,3);
  ierr = PCShellGetContext(pc,(void**)&M);CHKERRQ(ierr);
  ierr = MatLMVMSolveInactive(M, b, x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/* Routine for initializing the KSP solver, the BFGS preconditioner, and the initial trust radius estimation */

PetscErrorCode TaoBNKInitialize(Tao tao)
{
  PetscErrorCode               ierr;
  TAO_BNK                      *bnk = (TAO_BNK *)tao->data;
  KSPType                      ksp_type;
  PC                           pc;
  
  PetscReal                    fmin, ftrial, prered, actred, kappa, sigma, dnorm;
  PetscReal                    tau, tau_1, tau_2, tau_max, tau_min, max_radius;
  PetscReal                    delta, step = 1.0;
  
  PetscInt                     n,N,needH = 1;

  PetscInt                     i_max = 5;
  PetscInt                     j_max = 1;
  PetscInt                     i, j;
  
  PetscFunctionBegin;
  /* Number of times ksp stopped because of these reasons */
  bnk->ksp_atol = 0;
  bnk->ksp_rtol = 0;
  bnk->ksp_dtol = 0;
  bnk->ksp_ctol = 0;
  bnk->ksp_negc = 0;
  bnk->ksp_iter = 0;
  bnk->ksp_othr = 0;
  
  /* Initialize the Hessian perturbation */
  bnk->pert = bnk->sval;

  /* Initialize trust-region radius when using nash, stcg, or gltr
     Command automatically ignored for other methods
     Will be reset during the first iteration
  */
  ierr = KSPGetType(tao->ksp,&ksp_type);CHKERRQ(ierr);
  ierr = PetscStrcmp(ksp_type,KSPCGNASH,&bnk->is_nash);CHKERRQ(ierr);
  ierr = PetscStrcmp(ksp_type,KSPCGSTCG,&bnk->is_stcg);CHKERRQ(ierr);
  ierr = PetscStrcmp(ksp_type,KSPCGGLTR,&bnk->is_gltr);CHKERRQ(ierr);

  ierr = KSPCGSetRadius(tao->ksp,bnk->max_radius);CHKERRQ(ierr);

  if (bnk->is_nash || bnk->is_stcg || bnk->is_gltr) {
    if (tao->trust0 < 0.0) SETERRQ(PETSC_COMM_SELF,1,"Initial radius negative");
    tao->trust = tao->trust0;
    tao->trust = PetscMax(tao->trust, bnk->min_radius);
    tao->trust = PetscMin(tao->trust, bnk->max_radius);
  }

  /* Get vectors we will need */
  if (BNK_PC_BFGS == bnk->pc_type && !bnk->M) {
    ierr = VecGetLocalSize(tao->solution,&n);CHKERRQ(ierr);
    ierr = VecGetSize(tao->solution,&N);CHKERRQ(ierr);
    ierr = MatCreateLMVM(((PetscObject)tao)->comm,n,N,&bnk->M);CHKERRQ(ierr);
    ierr = MatLMVMAllocateVectors(bnk->M,tao->solution);CHKERRQ(ierr);
  }

  /* create vectors for the limited memory preconditioner */
  if ((BNK_PC_BFGS == bnk->pc_type) && (BFGS_SCALE_BFGS != bnk->bfgs_scale_type)) {
    if (!bnk->Diag) {
      ierr = VecDuplicate(tao->solution,&bnk->Diag);CHKERRQ(ierr);
    }
  }

  /* Modify the preconditioner to use the bfgs approximation */
  ierr = KSPGetPC(tao->ksp, &pc);CHKERRQ(ierr);
  switch(bnk->pc_type) {
  case BNK_PC_NONE:
    ierr = PCSetType(pc, PCNONE);CHKERRQ(ierr);
    ierr = PCSetFromOptions(pc);CHKERRQ(ierr);
    break;

  case BNK_PC_AHESS:
    ierr = PCSetType(pc, PCJACOBI);CHKERRQ(ierr);
    ierr = PCSetFromOptions(pc);CHKERRQ(ierr);
    ierr = PCJacobiSetUseAbs(pc,PETSC_TRUE);CHKERRQ(ierr);
    break;

  case BNK_PC_BFGS:
    ierr = PCSetType(pc, PCSHELL);CHKERRQ(ierr);
    ierr = PCSetFromOptions(pc);CHKERRQ(ierr);
    ierr = PCShellSetName(pc, "bfgs");CHKERRQ(ierr);
    ierr = PCShellSetContext(pc, bnk->M);CHKERRQ(ierr);
    ierr = PCShellSetApply(pc, MatLMVMSolveShell);CHKERRQ(ierr);
    break;

  default:
    /* Use the pc method set by pc_type */
    break;
  }

  /* Initialize trust-region radius.  The initialization is only performed
     when we are using Nash, Steihaug-Toint or the Generalized Lanczos method. */
  if (bnk->is_nash || bnk->is_stcg || bnk->is_gltr) {
    switch(bnk->init_type) {
    case BNK_INIT_CONSTANT:
      /* Use the initial radius specified */
      break;

    case BNK_INIT_INTERPOLATION:
      /* Use the initial radius specified */
      max_radius = 0.0;

      for (j = 0; j < j_max; ++j) {
        fmin = bnk->f;
        sigma = 0.0;

        if (needH) {
          ierr  = TaoComputeHessian(tao, tao->solution,tao->hessian,tao->hessian_pre);CHKERRQ(ierr);
          needH = 0;
        }

        for (i = 0; i < i_max; ++i) {
          ierr = VecCopy(tao->solution,bnk->W);CHKERRQ(ierr);
          ierr = VecAXPY(bnk->W,-tao->trust/bnk->gnorm,tao->gradient);CHKERRQ(ierr);
          ierr = VecMedian(tao->XL, bnk->W, tao->XU, bnk->W);CHKERRQ(ierr);
          ierr = VecNorm(bnk->W, NORM_2, &dnorm);CHKERRQ(ierr);
          if (dnorm != tao->trust/bnk->gnorm) tao->trust = dnorm;
          ierr = TaoComputeObjective(tao, bnk->W, &ftrial);CHKERRQ(ierr);
          if (PetscIsInfOrNanReal(ftrial)) {
            tau = bnk->gamma1_i;
          } else {
            if (ftrial < fmin) {
              fmin = ftrial;
              sigma = -tao->trust / bnk->gnorm;
            }

            ierr = MatMult(tao->hessian, tao->gradient, tao->stepdirection);CHKERRQ(ierr);
            ierr = VecDot(tao->gradient, tao->stepdirection, &prered);CHKERRQ(ierr);

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

            if (PetscAbsScalar(kappa - 1.0) <= bnk->mu1_i) {
              /* Great agreement */
              max_radius = PetscMax(max_radius, tao->trust);

              if (tau_max < 1.0) {
                tau = bnk->gamma3_i;
              } else if (tau_max > bnk->gamma4_i) {
                tau = bnk->gamma4_i;
              } else if (tau_1 >= 1.0 && tau_1 <= bnk->gamma4_i && tau_2 < 1.0) {
                tau = tau_1;
              } else if (tau_2 >= 1.0 && tau_2 <= bnk->gamma4_i && tau_1 < 1.0) {
                tau = tau_2;
              } else {
                tau = tau_max;
              }
            } else if (PetscAbsScalar(kappa - 1.0) <= bnk->mu2_i) {
              /* Good agreement */
              max_radius = PetscMax(max_radius, tao->trust);

              if (tau_max < bnk->gamma2_i) {
                tau = bnk->gamma2_i;
              } else if (tau_max > bnk->gamma3_i) {
                tau = bnk->gamma3_i;
              } else {
                tau = tau_max;
              }
            } else {
              /* Not good agreement */
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

        if (fmin < bnk->f) {
          bnk->f = fmin;
          ierr = VecAXPY(tao->solution,sigma,tao->gradient);CHKERRQ(ierr);
          ierr = VecMedian(tao->XL, tao->solution, tao->XU, tao->solution);CHKERRQ(ierr);
          ierr = TaoComputeGradient(tao,tao->solution,bnk->unprojected_gradient);CHKERRQ(ierr);
          ierr = VecBoundGradientProjection(bnk->unprojected_gradient,tao->solution,tao->XL,tao->XU,tao->gradient);CHKERRQ(ierr);

          ierr = TaoGradientNorm(tao, tao->gradient,NORM_2,&bnk->gnorm);CHKERRQ(ierr);
          if (PetscIsInfOrNanReal(bnk->gnorm)) SETERRQ(PETSC_COMM_SELF,1, "User provided compute gradient generated Inf or NaN");
          needH = 1;

          ierr = TaoLogConvergenceHistory(tao,bnk->f,bnk->gnorm,0.0,tao->ksp_its);CHKERRQ(ierr);
          ierr = TaoMonitor(tao,tao->niter,bnk->f,bnk->gnorm,0.0,step);CHKERRQ(ierr);
          ierr = (*tao->ops->convergencetest)(tao,tao->cnvP);CHKERRQ(ierr);
          if (tao->reason != TAO_CONTINUE_ITERATING) PetscFunctionReturn(0);
        }
      }
      tao->trust = PetscMax(tao->trust, max_radius);

      /* Modify the radius if it is too large or small */
      tao->trust = PetscMax(tao->trust, bnk->min_radius);
      tao->trust = PetscMin(tao->trust, bnk->max_radius);
      break;

    default:
      /* Norm of the first direction will initialize radius */
      tao->trust = 0.0;
      break;
    }
  }

  /* Set initial scaling for the BFGS preconditioner
     This step is done after computing the initial trust-region radius
     since the function value may have decreased */
  if (BNK_PC_BFGS == bnk->pc_type) {
    if (bnk->f != 0.0) {
      delta = 2.0 * PetscAbsScalar(bnk->f) / (bnk->gnorm*bnk->gnorm);
    } else {
      delta = 2.0 / (bnk->gnorm*bnk->gnorm);
    }
    ierr = MatLMVMSetDelta(bnk->M,delta);CHKERRQ(ierr);
  }

  /* Set counter for gradient/reset steps*/
  bnk->newt = 0;
  bnk->bfgs = 0;
  bnk->sgrad = 0;
  bnk->grad = 0;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/* Routine for computing the Newton step.

  If the safeguard is enabled, the Newton step is verified to be a 
  descent direction, with fallbacks onto BFGS, scaled gradient, and unscaled 
  gradient steps if/when necessary.
  
  The function reports back on which type of step has ultimately been stored 
  under tao->stepdirection.
*/

PetscErrorCode TaoBNKComputeStep(Tao tao, PetscBool safeguard, PetscInt *stepType)
{
  PetscErrorCode               ierr;
  TAO_BNK                      *bnk = (TAO_BNK *)tao->data;
  KSPConvergedReason           ksp_reason;
  
  PetscReal                    gdx, delta, e_min;

  PetscInt                     bfgsUpdates = 0;
  PetscInt                     kspits;
  
  PetscFunctionBegin;
  /* Shift the Hessian matrix */
  if (bnk->pert > 0) {
    ierr = MatShift(tao->hessian, bnk->pert);CHKERRQ(ierr);
    if (tao->hessian != tao->hessian_pre) {
      ierr = MatShift(tao->hessian_pre, bnk->pert);CHKERRQ(ierr);
    }
  }
  
  /* Determine the inactive set */
  ierr = ISDestroy(&bnk->inactive_idx);CHKERRQ(ierr);
  ierr = VecWhichInactive(tao->XL,tao->solution,bnk->unprojected_gradient,tao->XU,PETSC_TRUE,&bnk->inactive_idx);CHKERRQ(ierr);
  
  /* Prepare masked matrices for the inactive set */
  ierr = MatLMVMSetInactive(bnk->M, bnk->inactive_idx);CHKERRQ(ierr);
  ierr = VecSet(tao->stepdirection, 0.0);CHKERRQ(ierr);
  ierr = TaoMatGetSubMat(tao->hessian, bnk->inactive_idx, bnk->Xwork, TAO_SUBSET_MASK, &bnk->H_inactive);CHKERRQ(ierr);
  if (tao->hessian == tao->hessian_pre) {
    bnk->Hpre_inactive = bnk->H_inactive;
  } else {
    ierr = TaoMatGetSubMat(tao->hessian_pre, bnk->inactive_idx, bnk->Xwork, TAO_SUBSET_MASK, &bnk->Hpre_inactive);CHKERRQ(ierr);
  }
  
  /* Solve the Newton system of equations */
  ierr = KSPSetOperators(tao->ksp,bnk->H_inactive,bnk->Hpre_inactive);CHKERRQ(ierr);
  if (bnk->is_nash || bnk->is_stcg || bnk->is_gltr) {
    ierr = KSPCGSetRadius(tao->ksp,tao->trust);CHKERRQ(ierr);
    ierr = KSPSolve(tao->ksp, tao->gradient, tao->stepdirection);CHKERRQ(ierr);
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
        ierr = KSPSolve(tao->ksp, tao->gradient, tao->stepdirection);CHKERRQ(ierr);
        ierr = KSPGetIterationNumber(tao->ksp,&kspits);CHKERRQ(ierr);
        tao->ksp_its+=kspits;
        tao->ksp_tot_its+=kspits;
        ierr = KSPCGGetNormD(tao->ksp,&bnk->dnorm);CHKERRQ(ierr);

        if (bnk->dnorm == 0.0) SETERRQ(PETSC_COMM_SELF,1, "Initial direction zero");
      }
    }
  } else {
    ierr = KSPSolve(tao->ksp, tao->gradient, tao->stepdirection);CHKERRQ(ierr);
    ierr = KSPGetIterationNumber(tao->ksp, &kspits);CHKERRQ(ierr);
    tao->ksp_its += kspits;
    tao->ksp_tot_its+=kspits;
  }
  ierr = VecScale(tao->stepdirection, -1.0);CHKERRQ(ierr);
  ierr = KSPGetConvergedReason(tao->ksp, &ksp_reason);CHKERRQ(ierr);
  
  /* Destroy masked matrices */
  if (bnk->H_inactive != bnk->Hpre_inactive) { 
    ierr = MatDestroy(&bnk->Hpre_inactive);CHKERRQ(ierr); 
  }
  ierr = MatDestroy(&bnk->H_inactive);CHKERRQ(ierr);
  
  /* Make sure the BFGS preconditioner is healthy */
  if (bnk->pc_type == BNK_PC_BFGS) {
    ierr = MatLMVMGetUpdates(bnk->M, &bfgsUpdates);CHKERRQ(ierr);
    if ((KSP_DIVERGED_INDEFINITE_PC == ksp_reason) && (bfgsUpdates > 1)) {
      /* Preconditioner is numerically indefinite; reset the approximation. */
      if (bnk->f != 0.0) {
        delta = 2.0 * PetscAbsScalar(bnk->f) / (bnk->gnorm*bnk->gnorm);
      } else {
        delta = 2.0 / (bnk->gnorm*bnk->gnorm);
      }
      ierr = MatLMVMSetDelta(bnk->M,delta);CHKERRQ(ierr);
      ierr = MatLMVMReset(bnk->M);CHKERRQ(ierr);
      ierr = MatLMVMUpdate(bnk->M, tao->solution, bnk->unprojected_gradient);CHKERRQ(ierr);
      bfgsUpdates = 1;
    }
  }

  if (KSP_CONVERGED_ATOL == ksp_reason) {
    ++bnk->ksp_atol;
  } else if (KSP_CONVERGED_RTOL == ksp_reason) {
    ++bnk->ksp_rtol;
  } else if (KSP_CONVERGED_CG_CONSTRAINED == ksp_reason) {
    ++bnk->ksp_ctol;
  } else if (KSP_CONVERGED_CG_NEG_CURVE == ksp_reason) {
    ++bnk->ksp_negc;
  } else if (KSP_DIVERGED_DTOL == ksp_reason) {
    ++bnk->ksp_dtol;
  } else if (KSP_DIVERGED_ITS == ksp_reason) {
    ++bnk->ksp_iter;
  } else {
    ++bnk->ksp_othr;
  }

  /* Check for success (descent direction) */
  if (safeguard) {
    ierr = VecDot(tao->stepdirection, tao->gradient, &gdx);CHKERRQ(ierr);
    if ((gdx >= 0.0) || PetscIsInfOrNanReal(gdx)) {
      /* Newton step is not descent or direction produced Inf or NaN
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
        /* We don't have the bfgs matrix around and updated
           Must use gradient direction in this case */
        ierr = VecCopy(tao->gradient, tao->stepdirection);CHKERRQ(ierr);
        ierr = VecScale(tao->stepdirection, -1.0);CHKERRQ(ierr);
        ++bnk->grad;
        *stepType = BNK_GRADIENT;
      } else {
        /* Attempt to use the BFGS direction */
        ierr = MatLMVMSolve(bnk->M, bnk->unprojected_gradient, tao->stepdirection);CHKERRQ(ierr);
        ierr = VecBoundGradientProjection(tao->stepdirection,tao->solution,tao->XL,tao->XU,tao->stepdirection);CHKERRQ(ierr);
        ierr = VecScale(tao->stepdirection, -1.0);CHKERRQ(ierr);

        /* Check for success (descent direction) */
        ierr = VecDot(tao->gradient, tao->stepdirection, &gdx);CHKERRQ(ierr);
        if ((gdx >= 0) || PetscIsInfOrNanReal(gdx)) {
          /* BFGS direction is not descent or direction produced not a number
             We can assert bfgsUpdates > 1 in this case because
             the first solve produces the scaled gradient direction,
             which is guaranteed to be descent */

          /* Use steepest descent direction (scaled) */

          if (bnk->f != 0.0) {
            delta = 2.0 * PetscAbsScalar(bnk->f) / (bnk->gnorm*bnk->gnorm);
          } else {
            delta = 2.0 / (bnk->gnorm*bnk->gnorm);
          }
          ierr = MatLMVMSetDelta(bnk->M, delta);CHKERRQ(ierr);
          ierr = MatLMVMReset(bnk->M);CHKERRQ(ierr);
          ierr = MatLMVMUpdate(bnk->M, tao->solution, bnk->unprojected_gradient);CHKERRQ(ierr);
          ierr = MatLMVMSolve(bnk->M, bnk->unprojected_gradient, tao->stepdirection);CHKERRQ(ierr);
          ierr = VecBoundGradientProjection(tao->stepdirection,tao->solution,tao->XL,tao->XU,tao->stepdirection);CHKERRQ(ierr);
          ierr = VecScale(tao->stepdirection, -1.0);CHKERRQ(ierr);

          bfgsUpdates = 1;
          ++bnk->sgrad;
          *stepType = BNK_SCALED_GRADIENT;
        } else {
          if (1 == bfgsUpdates) {
            /* The first BFGS direction is always the scaled gradient */
            ++bnk->sgrad;
            *stepType = BNK_SCALED_GRADIENT;
          } else {
            ++bnk->bfgs;
            *stepType = BNK_BFGS;
          }
        }
      }
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
            ierr = KSPCGGLTRGetMinEig(tao->ksp, &e_min);CHKERRQ(ierr);
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

      ++bnk->newt;
      *stepType = BNK_NEWTON;
    }
  } else {
    ++bnk->newt;
    bnk->pert = 0.0;
    *stepType = BNK_NEWTON;
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/* Routine for performing a bound-projected More-Thuente line search.

  Includes fallbacks to BFGS, scaled gradient, and unscaled gradient steps if the 
  Newton step does not produce a valid step length.
*/

PetscErrorCode TaoBNKPerformLineSearch(Tao tao, PetscInt stepType, PetscReal *steplen, TaoLineSearchConvergedReason *reason)
{
  TAO_BNK        *bnk = (TAO_BNK *)tao->data;
  PetscErrorCode ierr;
  TaoLineSearchConvergedReason ls_reason;
  
  PetscReal      e_min, gdx, delta;
  PetscInt       bfgsUpdates;
  
  PetscFunctionBegin;
  /* Perform the linesearch */
  ierr = TaoLineSearchApply(tao->linesearch, tao->solution, &bnk->f, bnk->unprojected_gradient, tao->stepdirection, steplen, &ls_reason);CHKERRQ(ierr);
  ierr = VecBoundGradientProjection(bnk->unprojected_gradient,tao->solution,tao->XL,tao->XU,tao->gradient);CHKERRQ(ierr);
  ierr = TaoAddLineSearchCounts(tao);CHKERRQ(ierr);

  while (ls_reason != TAOLINESEARCH_SUCCESS && ls_reason != TAOLINESEARCH_SUCCESS_USER && stepType != BNK_GRADIENT) {
    /* Linesearch failed, revert solution */
    bnk->f = bnk->fold;
    ierr = VecCopy(bnk->Xold, tao->solution);CHKERRQ(ierr);
    ierr = VecCopy(bnk->Gold, tao->gradient);CHKERRQ(ierr);
    ierr = VecCopy(bnk->unprojected_gradient_old, bnk->unprojected_gradient);CHKERRQ(ierr);

    switch(stepType) {
    case BNK_NEWTON:
      /* Failed to obtain acceptable iterate with Newton 1step
         Update the perturbation for next time */
      --bnk->newt;
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
        ierr = VecCopy(tao->gradient, tao->stepdirection);CHKERRQ(ierr);
        ++bnk->grad;
        stepType = BNK_GRADIENT;
      } else {
        /* Attempt to use the BFGS direction */
        ierr = MatLMVMSolve(bnk->M, bnk->unprojected_gradient, tao->stepdirection);CHKERRQ(ierr);
        ierr = VecBoundGradientProjection(tao->stepdirection,tao->solution,tao->XL,tao->XU,tao->stepdirection);CHKERRQ(ierr);
        /* Check for success (descent direction) */
        ierr = VecDot(tao->gradient, tao->stepdirection, &gdx);CHKERRQ(ierr);
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
          ierr = MatLMVMUpdate(bnk->M, tao->solution, bnk->unprojected_gradient);CHKERRQ(ierr);
          ierr = MatLMVMSolve(bnk->M, bnk->unprojected_gradient, tao->stepdirection);CHKERRQ(ierr);
          ierr = VecBoundGradientProjection(tao->stepdirection,tao->solution,tao->XL,tao->XU,tao->stepdirection);CHKERRQ(ierr);

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
      ierr = MatLMVMUpdate(bnk->M, tao->solution, bnk->unprojected_gradient);CHKERRQ(ierr);
      ierr = MatLMVMSolve(bnk->M, bnk->unprojected_gradient, tao->stepdirection);CHKERRQ(ierr);
      ierr = VecBoundGradientProjection(tao->stepdirection,tao->solution,tao->XL,tao->XU,tao->stepdirection);CHKERRQ(ierr);

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
      ierr = MatLMVMUpdate(bnk->M, tao->solution, bnk->unprojected_gradient);CHKERRQ(ierr);
      ierr = MatLMVMSolve(bnk->M, bnk->unprojected_gradient, tao->stepdirection);CHKERRQ(ierr);
      ierr = VecBoundGradientProjection(tao->stepdirection,tao->solution,tao->XL,tao->XU,tao->stepdirection);CHKERRQ(ierr);

      bfgsUpdates = 1;
      ++bnk->grad;
      stepType = BNK_GRADIENT;
      break;
    }
    ierr = VecScale(tao->stepdirection, -1.0);CHKERRQ(ierr);

    ierr = TaoLineSearchApply(tao->linesearch, tao->solution, &bnk->f, bnk->unprojected_gradient, tao->stepdirection, steplen, &ls_reason);CHKERRQ(ierr);
    ierr = VecBoundGradientProjection(bnk->unprojected_gradient,tao->solution,tao->XL,tao->XU,tao->gradient);CHKERRQ(ierr);
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

PetscErrorCode TaoBNKUpdateTrustRadius(Tao tao, PetscReal prered, PetscReal actred, PetscInt stepType, PetscBool *accept)
{
  TAO_BNK        *bnk = (TAO_BNK *)tao->data;
  PetscErrorCode ierr;
  
  PetscReal      step, kappa;
  PetscReal      gdx, tau_1, tau_2, tau_min, tau_max;

  PetscFunctionBegin;
  /* Update trust region radius */
  *accept = PETSC_FALSE;
  switch(bnk->update_type) {
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
      if (prered < 0.0) {
        /* The predicted reduction has the wrong sign.  This cannot
           happen in infinite precision arithmetic.  Step should
           be rejected! */
        tao->trust = bnk->alpha1 * PetscMin(tao->trust, bnk->dnorm);
      }
      else {
        if (PetscIsInfOrNanReal(actred)) {
          tao->trust = bnk->alpha1 * PetscMin(tao->trust, bnk->dnorm);
        } else {
          if ((PetscAbsScalar(actred) <= bnk->epsilon) &&
              (PetscAbsScalar(prered) <= bnk->epsilon)) {
            kappa = 1.0;
          }
          else {
            kappa = actred / prered;
          }

          /* Accept or reject the step and update radius */
          if (kappa < bnk->eta1) {
            /* Reject the step */
            tao->trust = bnk->alpha1 * PetscMin(tao->trust, bnk->dnorm);
          }
          else {
            /* Accept the step */
            if (kappa < bnk->eta2) {
              /* Marginal bad step */
              tao->trust = bnk->alpha2 * PetscMin(tao->trust, bnk->dnorm);
            }
            else if (kappa < bnk->eta3) {
              /* Reasonable step */
              tao->trust = bnk->alpha3 * tao->trust;
            }
            else if (kappa < bnk->eta4) {
              /* Good step */
              tao->trust = PetscMax(bnk->alpha4 * bnk->dnorm, tao->trust);
            }
            else {
              /* Very good step */
              tao->trust = PetscMax(bnk->alpha5 * bnk->dnorm, tao->trust);
            }
            *accept = PETSC_TRUE;
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
    /*  The radius may have been increased; modify if it is too large */
    tao->trust = PetscMin(tao->trust, bnk->max_radius);
  }
  tao->trust = PetscMax(tao->trust, bnk->min_radius);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------- */

static PetscErrorCode TaoSetUp_BNK(Tao tao)
{
  TAO_BNK        *bnk = (TAO_BNK *)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!tao->gradient) {ierr = VecDuplicate(tao->solution,&tao->gradient);CHKERRQ(ierr);}
  if (!tao->stepdirection) {ierr = VecDuplicate(tao->solution,&tao->stepdirection);CHKERRQ(ierr);}
  if (!bnk->W) {ierr = VecDuplicate(tao->solution,&bnk->W);CHKERRQ(ierr);}
  if (!bnk->Xold) {ierr = VecDuplicate(tao->solution,&bnk->Xold);CHKERRQ(ierr);}
  if (!bnk->Gold) {ierr = VecDuplicate(tao->solution,&bnk->Gold);CHKERRQ(ierr);}
  if (!bnk->Xwork) {ierr = VecDuplicate(tao->solution,&bnk->Xwork);CHKERRQ(ierr);}
  if (!bnk->Gwork) {ierr = VecDuplicate(tao->solution,&bnk->Gwork);CHKERRQ(ierr);}
  if (!bnk->unprojected_gradient) {ierr = VecDuplicate(tao->solution,&bnk->unprojected_gradient);CHKERRQ(ierr);}
  if (!bnk->unprojected_gradient_old) {ierr = VecDuplicate(tao->solution,&bnk->unprojected_gradient_old);CHKERRQ(ierr);}
  bnk->Diag = 0;
  bnk->M = 0;
  bnk->H_inactive = 0;
  bnk->Hpre_inactive = 0;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode TaoDestroy_BNK(Tao tao)
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
  }
  ierr = VecDestroy(&bnk->Diag);CHKERRQ(ierr);
  ierr = MatDestroy(&bnk->M);CHKERRQ(ierr);
  ierr = PetscFree(tao->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode TaoSetFromOptions_BNK(PetscOptionItems *PetscOptionsObject,Tao tao)
{
  TAO_BNK        *bnk = (TAO_BNK *)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"Newton line search method for unconstrained optimization");CHKERRQ(ierr);
  ierr = PetscOptionsEList("-tao_BNK_pc_type", "pc type", "", BNK_PC, BNK_PC_TYPES, BNK_PC[bnk->pc_type], &bnk->pc_type, 0);CHKERRQ(ierr);
  ierr = PetscOptionsEList("-tao_BNK_bfgs_scale_type", "bfgs scale type", "", BFGS_SCALE, BFGS_SCALE_TYPES, BFGS_SCALE[bnk->bfgs_scale_type], &bnk->bfgs_scale_type, 0);CHKERRQ(ierr);
  ierr = PetscOptionsEList("-tao_BNK_init_type", "radius initialization type", "", BNK_INIT, BNK_INIT_TYPES, BNK_INIT[bnk->init_type], &bnk->init_type, 0);CHKERRQ(ierr);
  ierr = PetscOptionsEList("-tao_BNK_update_type", "radius update type", "", BNK_UPDATE, BNK_UPDATE_TYPES, BNK_UPDATE[bnk->update_type], &bnk->update_type, 0);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_BNK_sval", "perturbation starting value", "", bnk->sval, &bnk->sval,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_BNK_imin", "minimum initial perturbation", "", bnk->imin, &bnk->imin,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_BNK_imax", "maximum initial perturbation", "", bnk->imax, &bnk->imax,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_BNK_imfac", "initial merit factor", "", bnk->imfac, &bnk->imfac,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_BNK_pmin", "minimum perturbation", "", bnk->pmin, &bnk->pmin,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_BNK_pmax", "maximum perturbation", "", bnk->pmax, &bnk->pmax,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_BNK_pgfac", "growth factor", "", bnk->pgfac, &bnk->pgfac,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_BNK_psfac", "shrink factor", "", bnk->psfac, &bnk->psfac,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_BNK_pmgfac", "merit growth factor", "", bnk->pmgfac, &bnk->pmgfac,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_BNK_pmsfac", "merit shrink factor", "", bnk->pmsfac, &bnk->pmsfac,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_BNK_eta1", "poor steplength; reduce radius", "", bnk->eta1, &bnk->eta1,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_BNK_eta2", "reasonable steplength; leave radius alone", "", bnk->eta2, &bnk->eta2,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_BNK_eta3", "good steplength; increase radius", "", bnk->eta3, &bnk->eta3,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_BNK_eta4", "excellent steplength; greatly increase radius", "", bnk->eta4, &bnk->eta4,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_BNK_alpha1", "", "", bnk->alpha1, &bnk->alpha1,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_BNK_alpha2", "", "", bnk->alpha2, &bnk->alpha2,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_BNK_alpha3", "", "", bnk->alpha3, &bnk->alpha3,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_BNK_alpha4", "", "", bnk->alpha4, &bnk->alpha4,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_BNK_alpha5", "", "", bnk->alpha5, &bnk->alpha5,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_BNK_nu1", "poor steplength; reduce radius", "", bnk->nu1, &bnk->nu1,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_BNK_nu2", "reasonable steplength; leave radius alone", "", bnk->nu2, &bnk->nu2,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_BNK_nu3", "good steplength; increase radius", "", bnk->nu3, &bnk->nu3,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_BNK_nu4", "excellent steplength; greatly increase radius", "", bnk->nu4, &bnk->nu4,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_BNK_omega1", "", "", bnk->omega1, &bnk->omega1,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_BNK_omega2", "", "", bnk->omega2, &bnk->omega2,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_BNK_omega3", "", "", bnk->omega3, &bnk->omega3,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_BNK_omega4", "", "", bnk->omega4, &bnk->omega4,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_BNK_omega5", "", "", bnk->omega5, &bnk->omega5,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_BNK_mu1_i", "", "", bnk->mu1_i, &bnk->mu1_i,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_BNK_mu2_i", "", "", bnk->mu2_i, &bnk->mu2_i,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_BNK_gamma1_i", "", "", bnk->gamma1_i, &bnk->gamma1_i,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_BNK_gamma2_i", "", "", bnk->gamma2_i, &bnk->gamma2_i,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_BNK_gamma3_i", "", "", bnk->gamma3_i, &bnk->gamma3_i,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_BNK_gamma4_i", "", "", bnk->gamma4_i, &bnk->gamma4_i,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_BNK_theta_i", "", "", bnk->theta_i, &bnk->theta_i,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_BNK_mu1", "", "", bnk->mu1, &bnk->mu1,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_BNK_mu2", "", "", bnk->mu2, &bnk->mu2,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_BNK_gamma1", "", "", bnk->gamma1, &bnk->gamma1,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_BNK_gamma2", "", "", bnk->gamma2, &bnk->gamma2,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_BNK_gamma3", "", "", bnk->gamma3, &bnk->gamma3,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_BNK_gamma4", "", "", bnk->gamma4, &bnk->gamma4,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_BNK_theta", "", "", bnk->theta, &bnk->theta,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_BNK_min_radius", "lower bound on initial radius", "", bnk->min_radius, &bnk->min_radius,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_BNK_max_radius", "upper bound on radius", "", bnk->max_radius, &bnk->max_radius,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_BNK_epsilon", "tolerance used when computing actual and predicted reduction", "", bnk->epsilon, &bnk->epsilon,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  ierr = TaoLineSearchSetFromOptions(tao->linesearch);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(tao->ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode TaoView_BNK(Tao tao, PetscViewer viewer)
{
  TAO_BNK        *bnk = (TAO_BNK *)tao->data;
  PetscInt       nrejects;
  PetscBool      isascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    if (BNK_PC_BFGS == bnk->pc_type && bnk->M) {
      ierr = MatLMVMGetRejects(bnk->M,&nrejects);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer, "Rejected matrix updates: %D\n",nrejects);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer, "Newton steps: %D\n", bnk->newt);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "BFGS steps: %D\n", bnk->bfgs);CHKERRQ(ierr);
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
+ -tao_BNK_pc_type - "none","ahess","bfgs","petsc"
. -tao_BNK_bfgs_scale_type - "ahess","phess","bfgs"
. -tao_BNK_init_type - "constant","direction","interpolation"
. -tao_BNK_update_type - "step","direction","interpolation"
. -tao_BNK_sval - perturbation starting value
. -tao_BNK_imin - minimum initial perturbation
. -tao_BNK_imax - maximum initial perturbation
. -tao_BNK_pmin - minimum perturbation
. -tao_BNK_pmax - maximum perturbation
. -tao_BNK_pgfac - growth factor
. -tao_BNK_psfac - shrink factor
. -tao_BNK_imfac - initial merit factor
. -tao_BNK_pmgfac - merit growth factor
. -tao_BNK_pmsfac - merit shrink factor
. -tao_BNK_eta1 - poor steplength; reduce radius
. -tao_BNK_eta2 - reasonable steplength; leave radius
. -tao_BNK_eta3 - good steplength; increase readius
. -tao_BNK_eta4 - excellent steplength; greatly increase radius
. -tao_BNK_alpha1 - alpha1 reduction
. -tao_BNK_alpha2 - alpha2 reduction
. -tao_BNK_alpha3 - alpha3 reduction
. -tao_BNK_alpha4 - alpha4 reduction
. -tao_BNK_alpha - alpha5 reduction
. -tao_BNK_mu1 - mu1 interpolation update
. -tao_BNK_mu2 - mu2 interpolation update
. -tao_BNK_gamma1 - gamma1 interpolation update
. -tao_BNK_gamma2 - gamma2 interpolation update
. -tao_BNK_gamma3 - gamma3 interpolation update
. -tao_BNK_gamma4 - gamma4 interpolation update
. -tao_BNK_theta - theta interpolation update
. -tao_BNK_omega1 - omega1 step update
. -tao_BNK_omega2 - omega2 step update
. -tao_BNK_omega3 - omega3 step update
. -tao_BNK_omega4 - omega4 step update
. -tao_BNK_omega5 - omega5 step update
. -tao_BNK_mu1_i -  mu1 interpolation init factor
. -tao_BNK_mu2_i -  mu2 interpolation init factor
. -tao_BNK_gamma1_i -  gamma1 interpolation init factor
. -tao_BNK_gamma2_i -  gamma2 interpolation init factor
. -tao_BNK_gamma3_i -  gamma3 interpolation init factor
. -tao_BNK_gamma4_i -  gamma4 interpolation init factor
- -tao_BNK_theta_i -  theta interpolation init factor

  Level: beginner
M*/

PetscErrorCode TaoCreate_BNK(Tao tao)
{
  TAO_BNK        *bnk;
  const char     *morethuente_type = TAOLINESEARCHMT;
  PetscErrorCode ierr;

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
  bnk->min_radius = 1.0e-10;
  bnk->max_radius = 1.0e10;
  bnk->epsilon = 1.0e-6;

  bnk->pc_type         = BNK_PC_BFGS;
  bnk->bfgs_scale_type = BFGS_SCALE_PHESS;
  bnk->init_type       = BNK_INIT_INTERPOLATION;
  bnk->update_type     = BNK_UPDATE_INTERPOLATION;

  ierr = TaoLineSearchCreate(((PetscObject)tao)->comm,&tao->linesearch);CHKERRQ(ierr);
  ierr = PetscObjectIncrementTabLevel((PetscObject)tao->linesearch, (PetscObject)tao, 1);CHKERRQ(ierr);
  ierr = TaoLineSearchSetType(tao->linesearch,morethuente_type);CHKERRQ(ierr);
  ierr = TaoLineSearchUseTaoRoutines(tao->linesearch,tao);CHKERRQ(ierr);
  ierr = TaoLineSearchSetOptionsPrefix(tao->linesearch,tao->hdr.prefix);CHKERRQ(ierr);

  /*  Set linear solver to default for symmetric matrices */
  ierr = KSPCreate(((PetscObject)tao)->comm,&tao->ksp);CHKERRQ(ierr);
  ierr = PetscObjectIncrementTabLevel((PetscObject)tao->ksp, (PetscObject)tao, 1);CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(tao->ksp,tao->hdr.prefix);CHKERRQ(ierr);
  ierr = KSPSetType(tao->ksp,KSPCGSTCG);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
