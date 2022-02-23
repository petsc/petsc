#include <petsctaolinesearch.h>
#include <../src/tao/unconstrained/impls/nls/nlsimpl.h>

#include <petscksp.h>

#define NLS_INIT_CONSTANT         0
#define NLS_INIT_DIRECTION        1
#define NLS_INIT_INTERPOLATION    2
#define NLS_INIT_TYPES            3

#define NLS_UPDATE_STEP           0
#define NLS_UPDATE_REDUCTION      1
#define NLS_UPDATE_INTERPOLATION  2
#define NLS_UPDATE_TYPES          3

static const char *NLS_INIT[64] = {"constant", "direction", "interpolation"};

static const char *NLS_UPDATE[64] = {"step", "reduction", "interpolation"};

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

#define NLS_NEWTON              0
#define NLS_BFGS                1
#define NLS_GRADIENT            2

static PetscErrorCode TaoSolve_NLS(Tao tao)
{
  PetscErrorCode               ierr;
  TAO_NLS                      *nlsP = (TAO_NLS *)tao->data;
  KSPType                      ksp_type;
  PetscBool                    is_nash,is_stcg,is_gltr,is_bfgs,is_jacobi,is_symmetric,sym_set;
  KSPConvergedReason           ksp_reason;
  PC                           pc;
  TaoLineSearchConvergedReason ls_reason;

  PetscReal                    fmin, ftrial, f_full, prered, actred, kappa, sigma;
  PetscReal                    tau, tau_1, tau_2, tau_max, tau_min, max_radius;
  PetscReal                    f, fold, gdx, gnorm, pert;
  PetscReal                    step = 1.0;
  PetscReal                    norm_d = 0.0, e_min;

  PetscInt                     stepType;
  PetscInt                     bfgsUpdates = 0;
  PetscInt                     n,N,kspits;
  PetscInt                     needH = 1;

  PetscInt                     i_max = 5;
  PetscInt                     j_max = 1;
  PetscInt                     i, j;

  PetscFunctionBegin;
  if (tao->XL || tao->XU || tao->ops->computebounds) {
    ierr = PetscInfo(tao,"WARNING: Variable bounds have been set but will be ignored by nls algorithm\n");CHKERRQ(ierr);
  }

  /* Initialized variables */
  pert = nlsP->sval;

  /* Number of times ksp stopped because of these reasons */
  nlsP->ksp_atol = 0;
  nlsP->ksp_rtol = 0;
  nlsP->ksp_dtol = 0;
  nlsP->ksp_ctol = 0;
  nlsP->ksp_negc = 0;
  nlsP->ksp_iter = 0;
  nlsP->ksp_othr = 0;

  /* Initialize trust-region radius when using nash, stcg, or gltr
     Command automatically ignored for other methods
     Will be reset during the first iteration
  */
  ierr = KSPGetType(tao->ksp,&ksp_type);CHKERRQ(ierr);
  ierr = PetscStrcmp(ksp_type,KSPNASH,&is_nash);CHKERRQ(ierr);
  ierr = PetscStrcmp(ksp_type,KSPSTCG,&is_stcg);CHKERRQ(ierr);
  ierr = PetscStrcmp(ksp_type,KSPGLTR,&is_gltr);CHKERRQ(ierr);

  ierr = KSPCGSetRadius(tao->ksp,nlsP->max_radius);CHKERRQ(ierr);

  if (is_nash || is_stcg || is_gltr) {
    PetscCheckFalse(tao->trust0 < 0.0,PetscObjectComm((PetscObject)tao),PETSC_ERR_ARG_OUTOFRANGE,"Initial radius negative");
    tao->trust = tao->trust0;
    tao->trust = PetscMax(tao->trust, nlsP->min_radius);
    tao->trust = PetscMin(tao->trust, nlsP->max_radius);
  }

  /* Check convergence criteria */
  ierr = TaoComputeObjectiveAndGradient(tao, tao->solution, &f, tao->gradient);CHKERRQ(ierr);
  ierr = TaoGradientNorm(tao, tao->gradient,NORM_2,&gnorm);CHKERRQ(ierr);
  PetscCheckFalse(PetscIsInfOrNanReal(f) || PetscIsInfOrNanReal(gnorm),PetscObjectComm((PetscObject)tao),PETSC_ERR_USER, "User provided compute function generated Inf or NaN");

  tao->reason = TAO_CONTINUE_ITERATING;
  ierr = TaoLogConvergenceHistory(tao,f,gnorm,0.0,tao->ksp_its);CHKERRQ(ierr);
  ierr = TaoMonitor(tao,tao->niter,f,gnorm,0.0,step);CHKERRQ(ierr);
  ierr = (*tao->ops->convergencetest)(tao,tao->cnvP);CHKERRQ(ierr);
  if (tao->reason != TAO_CONTINUE_ITERATING) PetscFunctionReturn(0);

  /* Allocate the vectors needed for the BFGS approximation */
  ierr = KSPGetPC(tao->ksp, &pc);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)pc, PCLMVM, &is_bfgs);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)pc, PCJACOBI, &is_jacobi);CHKERRQ(ierr);
  if (is_bfgs) {
    nlsP->bfgs_pre = pc;
    ierr = PCLMVMGetMatLMVM(nlsP->bfgs_pre, &nlsP->M);CHKERRQ(ierr);
    ierr = VecGetLocalSize(tao->solution, &n);CHKERRQ(ierr);
    ierr = VecGetSize(tao->solution, &N);CHKERRQ(ierr);
    ierr = MatSetSizes(nlsP->M, n, n, N, N);CHKERRQ(ierr);
    ierr = MatLMVMAllocate(nlsP->M, tao->solution, tao->gradient);CHKERRQ(ierr);
    ierr = MatIsSymmetricKnown(nlsP->M, &sym_set, &is_symmetric);CHKERRQ(ierr);
    PetscCheckFalse(!sym_set || !is_symmetric,PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_INCOMP, "LMVM matrix in the LMVM preconditioner must be symmetric.");
  } else if (is_jacobi) {
    ierr = PCJacobiSetUseAbs(pc,PETSC_TRUE);CHKERRQ(ierr);
  }

  /* Initialize trust-region radius.  The initialization is only performed
     when we are using Nash, Steihaug-Toint or the Generalized Lanczos method. */
  if (is_nash || is_stcg || is_gltr) {
    switch (nlsP->init_type) {
    case NLS_INIT_CONSTANT:
      /* Use the initial radius specified */
      break;

    case NLS_INIT_INTERPOLATION:
      /* Use the initial radius specified */
      max_radius = 0.0;

      for (j = 0; j < j_max; ++j) {
        fmin = f;
        sigma = 0.0;

        if (needH) {
          ierr  = TaoComputeHessian(tao, tao->solution,tao->hessian,tao->hessian_pre);CHKERRQ(ierr);
          needH = 0;
        }

        for (i = 0; i < i_max; ++i) {
          ierr = VecCopy(tao->solution,nlsP->W);CHKERRQ(ierr);
          ierr = VecAXPY(nlsP->W,-tao->trust/gnorm,tao->gradient);CHKERRQ(ierr);
          ierr = TaoComputeObjective(tao, nlsP->W, &ftrial);CHKERRQ(ierr);
          if (PetscIsInfOrNanReal(ftrial)) {
            tau = nlsP->gamma1_i;
          } else {
            if (ftrial < fmin) {
              fmin = ftrial;
              sigma = -tao->trust / gnorm;
            }

            ierr = MatMult(tao->hessian, tao->gradient, nlsP->D);CHKERRQ(ierr);
            ierr = VecDot(tao->gradient, nlsP->D, &prered);CHKERRQ(ierr);

            prered = tao->trust * (gnorm - 0.5 * tao->trust * prered / (gnorm * gnorm));
            actred = f - ftrial;
            if ((PetscAbsScalar(actred) <= nlsP->epsilon) && (PetscAbsScalar(prered) <= nlsP->epsilon)) {
              kappa = 1.0;
            } else {
              kappa = actred / prered;
            }

            tau_1 = nlsP->theta_i * gnorm * tao->trust / (nlsP->theta_i * gnorm * tao->trust + (1.0 - nlsP->theta_i) * prered - actred);
            tau_2 = nlsP->theta_i * gnorm * tao->trust / (nlsP->theta_i * gnorm * tao->trust - (1.0 + nlsP->theta_i) * prered + actred);
            tau_min = PetscMin(tau_1, tau_2);
            tau_max = PetscMax(tau_1, tau_2);

            if (PetscAbsScalar(kappa - (PetscReal)1.0) <= nlsP->mu1_i) {
              /* Great agreement */
              max_radius = PetscMax(max_radius, tao->trust);

              if (tau_max < 1.0) {
                tau = nlsP->gamma3_i;
              } else if (tau_max > nlsP->gamma4_i) {
                tau = nlsP->gamma4_i;
              } else if (tau_1 >= 1.0 && tau_1 <= nlsP->gamma4_i && tau_2 < 1.0) {
                tau = tau_1;
              } else if (tau_2 >= 1.0 && tau_2 <= nlsP->gamma4_i && tau_1 < 1.0) {
                tau = tau_2;
              } else {
                tau = tau_max;
              }
            } else if (PetscAbsScalar(kappa - (PetscReal)1.0) <= nlsP->mu2_i) {
              /* Good agreement */
              max_radius = PetscMax(max_radius, tao->trust);

              if (tau_max < nlsP->gamma2_i) {
                tau = nlsP->gamma2_i;
              } else if (tau_max > nlsP->gamma3_i) {
                tau = nlsP->gamma3_i;
              } else {
                tau = tau_max;
              }
            } else {
              /* Not good agreement */
              if (tau_min > 1.0) {
                tau = nlsP->gamma2_i;
              } else if (tau_max < nlsP->gamma1_i) {
                tau = nlsP->gamma1_i;
              } else if ((tau_min < nlsP->gamma1_i) && (tau_max >= 1.0)) {
                tau = nlsP->gamma1_i;
              } else if ((tau_1 >= nlsP->gamma1_i) && (tau_1 < 1.0) && ((tau_2 < nlsP->gamma1_i) || (tau_2 >= 1.0))) {
                tau = tau_1;
              } else if ((tau_2 >= nlsP->gamma1_i) && (tau_2 < 1.0) && ((tau_1 < nlsP->gamma1_i) || (tau_2 >= 1.0))) {
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
          ierr = VecAXPY(tao->solution,sigma,tao->gradient);CHKERRQ(ierr);
          ierr = TaoComputeGradient(tao,tao->solution,tao->gradient);CHKERRQ(ierr);

          ierr = TaoGradientNorm(tao, tao->gradient,NORM_2,&gnorm);CHKERRQ(ierr);
          PetscCheckFalse(PetscIsInfOrNanReal(gnorm),PetscObjectComm((PetscObject)tao),PETSC_ERR_USER, "User provided compute gradient generated Inf or NaN");
          needH = 1;

          ierr = TaoLogConvergenceHistory(tao,f,gnorm,0.0,tao->ksp_its);CHKERRQ(ierr);
          ierr = TaoMonitor(tao,tao->niter,f,gnorm,0.0,step);CHKERRQ(ierr);
          ierr = (*tao->ops->convergencetest)(tao,tao->cnvP);CHKERRQ(ierr);
          if (tao->reason != TAO_CONTINUE_ITERATING) PetscFunctionReturn(0);
        }
      }
      tao->trust = PetscMax(tao->trust, max_radius);

      /* Modify the radius if it is too large or small */
      tao->trust = PetscMax(tao->trust, nlsP->min_radius);
      tao->trust = PetscMin(tao->trust, nlsP->max_radius);
      break;

    default:
      /* Norm of the first direction will initialize radius */
      tao->trust = 0.0;
      break;
    }
  }

  /* Set counter for gradient/reset steps*/
  nlsP->newt = 0;
  nlsP->bfgs = 0;
  nlsP->grad = 0;

  /* Have not converged; continue with Newton method */
  while (tao->reason == TAO_CONTINUE_ITERATING) {
    /* Call general purpose update function */
    if (tao->ops->update) {
      ierr = (*tao->ops->update)(tao, tao->niter, tao->user_update);CHKERRQ(ierr);
    }
    ++tao->niter;
    tao->ksp_its = 0;

    /* Compute the Hessian */
    if (needH) {
      ierr = TaoComputeHessian(tao,tao->solution,tao->hessian,tao->hessian_pre);CHKERRQ(ierr);
    }

    /* Shift the Hessian matrix */
    if (pert > 0) {
      ierr = MatShift(tao->hessian, pert);CHKERRQ(ierr);
      if (tao->hessian != tao->hessian_pre) {
        ierr = MatShift(tao->hessian_pre, pert);CHKERRQ(ierr);
      }
    }

    if (nlsP->bfgs_pre) {
      ierr = MatLMVMUpdate(nlsP->M, tao->solution, tao->gradient);CHKERRQ(ierr);
      ++bfgsUpdates;
    }

    /* Solve the Newton system of equations */
    ierr = KSPSetOperators(tao->ksp,tao->hessian,tao->hessian_pre);CHKERRQ(ierr);
    if (is_nash || is_stcg || is_gltr) {
      ierr = KSPCGSetRadius(tao->ksp,nlsP->max_radius);CHKERRQ(ierr);
      ierr = KSPSolve(tao->ksp, tao->gradient, nlsP->D);CHKERRQ(ierr);
      ierr = KSPGetIterationNumber(tao->ksp,&kspits);CHKERRQ(ierr);
      tao->ksp_its += kspits;
      tao->ksp_tot_its += kspits;
      ierr = KSPCGGetNormD(tao->ksp,&norm_d);CHKERRQ(ierr);

      if (0.0 == tao->trust) {
        /* Radius was uninitialized; use the norm of the direction */
        if (norm_d > 0.0) {
          tao->trust = norm_d;

          /* Modify the radius if it is too large or small */
          tao->trust = PetscMax(tao->trust, nlsP->min_radius);
          tao->trust = PetscMin(tao->trust, nlsP->max_radius);
        } else {
          /* The direction was bad; set radius to default value and re-solve
             the trust-region subproblem to get a direction */
          tao->trust = tao->trust0;

          /* Modify the radius if it is too large or small */
          tao->trust = PetscMax(tao->trust, nlsP->min_radius);
          tao->trust = PetscMin(tao->trust, nlsP->max_radius);

          ierr = KSPCGSetRadius(tao->ksp,nlsP->max_radius);CHKERRQ(ierr);
          ierr = KSPSolve(tao->ksp, tao->gradient, nlsP->D);CHKERRQ(ierr);
          ierr = KSPGetIterationNumber(tao->ksp,&kspits);CHKERRQ(ierr);
          tao->ksp_its += kspits;
          tao->ksp_tot_its += kspits;
          ierr = KSPCGGetNormD(tao->ksp,&norm_d);CHKERRQ(ierr);

          PetscCheckFalse(norm_d == 0.0,PETSC_COMM_SELF,PETSC_ERR_PLIB, "Initial direction zero");
        }
      }
    } else {
      ierr = KSPSolve(tao->ksp, tao->gradient, nlsP->D);CHKERRQ(ierr);
      ierr = KSPGetIterationNumber(tao->ksp, &kspits);CHKERRQ(ierr);
      tao->ksp_its += kspits;
      tao->ksp_tot_its += kspits;
    }
    ierr = VecScale(nlsP->D, -1.0);CHKERRQ(ierr);
    ierr = KSPGetConvergedReason(tao->ksp, &ksp_reason);CHKERRQ(ierr);
    if ((KSP_DIVERGED_INDEFINITE_PC == ksp_reason) &&  (nlsP->bfgs_pre)) {
      /* Preconditioner is numerically indefinite; reset the
         approximate if using BFGS preconditioning. */
      ierr = MatLMVMReset(nlsP->M, PETSC_FALSE);CHKERRQ(ierr);
      ierr = MatLMVMUpdate(nlsP->M, tao->solution, tao->gradient);CHKERRQ(ierr);
      bfgsUpdates = 1;
    }

    if (KSP_CONVERGED_ATOL == ksp_reason) {
      ++nlsP->ksp_atol;
    } else if (KSP_CONVERGED_RTOL == ksp_reason) {
      ++nlsP->ksp_rtol;
    } else if (KSP_CONVERGED_CG_CONSTRAINED == ksp_reason) {
      ++nlsP->ksp_ctol;
    } else if (KSP_CONVERGED_CG_NEG_CURVE == ksp_reason) {
      ++nlsP->ksp_negc;
    } else if (KSP_DIVERGED_DTOL == ksp_reason) {
      ++nlsP->ksp_dtol;
    } else if (KSP_DIVERGED_ITS == ksp_reason) {
      ++nlsP->ksp_iter;
    } else {
      ++nlsP->ksp_othr;
    }

    /* Check for success (descent direction) */
    ierr = VecDot(nlsP->D, tao->gradient, &gdx);CHKERRQ(ierr);
    if ((gdx >= 0.0) || PetscIsInfOrNanReal(gdx)) {
      /* Newton step is not descent or direction produced Inf or NaN
         Update the perturbation for next time */
      if (pert <= 0.0) {
        /* Initialize the perturbation */
        pert = PetscMin(nlsP->imax, PetscMax(nlsP->imin, nlsP->imfac * gnorm));
        if (is_gltr) {
          ierr = KSPGLTRGetMinEig(tao->ksp,&e_min);CHKERRQ(ierr);
          pert = PetscMax(pert, -e_min);
        }
      } else {
        /* Increase the perturbation */
        pert = PetscMin(nlsP->pmax, PetscMax(nlsP->pgfac * pert, nlsP->pmgfac * gnorm));
      }

      if (!nlsP->bfgs_pre) {
        /* We don't have the bfgs matrix around and updated
           Must use gradient direction in this case */
        ierr = VecCopy(tao->gradient, nlsP->D);CHKERRQ(ierr);
        ierr = VecScale(nlsP->D, -1.0);CHKERRQ(ierr);
        ++nlsP->grad;
        stepType = NLS_GRADIENT;
      } else {
        /* Attempt to use the BFGS direction */
        ierr = MatSolve(nlsP->M, tao->gradient, nlsP->D);CHKERRQ(ierr);
        ierr = VecScale(nlsP->D, -1.0);CHKERRQ(ierr);

        /* Check for success (descent direction) */
        ierr = VecDot(tao->gradient, nlsP->D, &gdx);CHKERRQ(ierr);
        if ((gdx >= 0) || PetscIsInfOrNanReal(gdx)) {
          /* BFGS direction is not descent or direction produced not a number
             We can assert bfgsUpdates > 1 in this case because
             the first solve produces the scaled gradient direction,
             which is guaranteed to be descent */

          /* Use steepest descent direction (scaled) */
          ierr = MatLMVMReset(nlsP->M, PETSC_FALSE);CHKERRQ(ierr);
          ierr = MatLMVMUpdate(nlsP->M, tao->solution, tao->gradient);CHKERRQ(ierr);
          ierr = MatSolve(nlsP->M, tao->gradient, nlsP->D);CHKERRQ(ierr);
          ierr = VecScale(nlsP->D, -1.0);CHKERRQ(ierr);

          bfgsUpdates = 1;
          ++nlsP->grad;
          stepType = NLS_GRADIENT;
        } else {
          ierr = MatLMVMGetUpdateCount(nlsP->M, &bfgsUpdates);CHKERRQ(ierr);
          if (1 == bfgsUpdates) {
            /* The first BFGS direction is always the scaled gradient */
            ++nlsP->grad;
            stepType = NLS_GRADIENT;
          } else {
            ++nlsP->bfgs;
            stepType = NLS_BFGS;
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
        if (pert <= 0.0) {
          /* Initialize the perturbation */
          pert = PetscMin(nlsP->imax, PetscMax(nlsP->imin, nlsP->imfac * gnorm));
          if (is_gltr) {
            ierr = KSPGLTRGetMinEig(tao->ksp, &e_min);CHKERRQ(ierr);
            pert = PetscMax(pert, -e_min);
          }
        } else {
          /* Increase the perturbation */
          pert = PetscMin(nlsP->pmax, PetscMax(nlsP->pgfac * pert, nlsP->pmgfac * gnorm));
        }
        break;

      default:
        /* Newton step computation is good; decrease perturbation */
        pert = PetscMin(nlsP->psfac * pert, nlsP->pmsfac * gnorm);
        if (pert < nlsP->pmin) {
          pert = 0.0;
        }
        break;
      }

      ++nlsP->newt;
      stepType = NLS_NEWTON;
    }

    /* Perform the linesearch */
    fold = f;
    ierr = VecCopy(tao->solution, nlsP->Xold);CHKERRQ(ierr);
    ierr = VecCopy(tao->gradient, nlsP->Gold);CHKERRQ(ierr);

    ierr = TaoLineSearchApply(tao->linesearch, tao->solution, &f, tao->gradient, nlsP->D, &step, &ls_reason);CHKERRQ(ierr);
    ierr = TaoAddLineSearchCounts(tao);CHKERRQ(ierr);

    while (ls_reason != TAOLINESEARCH_SUCCESS && ls_reason != TAOLINESEARCH_SUCCESS_USER && stepType != NLS_GRADIENT) {      /* Linesearch failed */
      f = fold;
      ierr = VecCopy(nlsP->Xold, tao->solution);CHKERRQ(ierr);
      ierr = VecCopy(nlsP->Gold, tao->gradient);CHKERRQ(ierr);

      switch(stepType) {
      case NLS_NEWTON:
        /* Failed to obtain acceptable iterate with Newton 1step
           Update the perturbation for next time */
        if (pert <= 0.0) {
          /* Initialize the perturbation */
          pert = PetscMin(nlsP->imax, PetscMax(nlsP->imin, nlsP->imfac * gnorm));
          if (is_gltr) {
            ierr = KSPGLTRGetMinEig(tao->ksp,&e_min);CHKERRQ(ierr);
            pert = PetscMax(pert, -e_min);
          }
        } else {
          /* Increase the perturbation */
          pert = PetscMin(nlsP->pmax, PetscMax(nlsP->pgfac * pert, nlsP->pmgfac * gnorm));
        }

        if (!nlsP->bfgs_pre) {
          /* We don't have the bfgs matrix around and being updated
             Must use gradient direction in this case */
          ierr = VecCopy(tao->gradient, nlsP->D);CHKERRQ(ierr);
          ++nlsP->grad;
          stepType = NLS_GRADIENT;
        } else {
          /* Attempt to use the BFGS direction */
          ierr = MatSolve(nlsP->M, tao->gradient, nlsP->D);CHKERRQ(ierr);
          /* Check for success (descent direction) */
          ierr = VecDot(tao->solution, nlsP->D, &gdx);CHKERRQ(ierr);
          if ((gdx <= 0) || PetscIsInfOrNanReal(gdx)) {
            /* BFGS direction is not descent or direction produced not a number
               We can assert bfgsUpdates > 1 in this case
               Use steepest descent direction (scaled) */
            ierr = MatLMVMReset(nlsP->M, PETSC_FALSE);CHKERRQ(ierr);
            ierr = MatLMVMUpdate(nlsP->M, tao->solution, tao->gradient);CHKERRQ(ierr);
            ierr = MatSolve(nlsP->M, tao->gradient, nlsP->D);CHKERRQ(ierr);

            bfgsUpdates = 1;
            ++nlsP->grad;
            stepType = NLS_GRADIENT;
          } else {
            if (1 == bfgsUpdates) {
              /* The first BFGS direction is always the scaled gradient */
              ++nlsP->grad;
              stepType = NLS_GRADIENT;
            } else {
              ++nlsP->bfgs;
              stepType = NLS_BFGS;
            }
          }
        }
        break;

      case NLS_BFGS:
        /* Can only enter if pc_type == NLS_PC_BFGS
           Failed to obtain acceptable iterate with BFGS step
           Attempt to use the scaled gradient direction */
        ierr = MatLMVMReset(nlsP->M, PETSC_FALSE);CHKERRQ(ierr);
        ierr = MatLMVMUpdate(nlsP->M, tao->solution, tao->gradient);CHKERRQ(ierr);
        ierr = MatSolve(nlsP->M, tao->gradient, nlsP->D);CHKERRQ(ierr);

        bfgsUpdates = 1;
        ++nlsP->grad;
        stepType = NLS_GRADIENT;
        break;
      }
      ierr = VecScale(nlsP->D, -1.0);CHKERRQ(ierr);

      ierr = TaoLineSearchApply(tao->linesearch, tao->solution, &f, tao->gradient, nlsP->D, &step, &ls_reason);CHKERRQ(ierr);
      ierr = TaoLineSearchGetFullStepObjective(tao->linesearch, &f_full);CHKERRQ(ierr);
      ierr = TaoAddLineSearchCounts(tao);CHKERRQ(ierr);
    }

    if (ls_reason != TAOLINESEARCH_SUCCESS && ls_reason != TAOLINESEARCH_SUCCESS_USER) {
      /* Failed to find an improving point */
      f = fold;
      ierr = VecCopy(nlsP->Xold, tao->solution);CHKERRQ(ierr);
      ierr = VecCopy(nlsP->Gold, tao->gradient);CHKERRQ(ierr);
      step = 0.0;
      tao->reason = TAO_DIVERGED_LS_FAILURE;
      break;
    }

    /* Update trust region radius */
    if (is_nash || is_stcg || is_gltr) {
      switch(nlsP->update_type) {
      case NLS_UPDATE_STEP:
        if (stepType == NLS_NEWTON) {
          if (step < nlsP->nu1) {
            /* Very bad step taken; reduce radius */
            tao->trust = nlsP->omega1 * PetscMin(norm_d, tao->trust);
          } else if (step < nlsP->nu2) {
            /* Reasonably bad step taken; reduce radius */
            tao->trust = nlsP->omega2 * PetscMin(norm_d, tao->trust);
          } else if (step < nlsP->nu3) {
            /*  Reasonable step was taken; leave radius alone */
            if (nlsP->omega3 < 1.0) {
              tao->trust = nlsP->omega3 * PetscMin(norm_d, tao->trust);
            } else if (nlsP->omega3 > 1.0) {
              tao->trust = PetscMax(nlsP->omega3 * norm_d, tao->trust);
            }
          } else if (step < nlsP->nu4) {
            /*  Full step taken; increase the radius */
            tao->trust = PetscMax(nlsP->omega4 * norm_d, tao->trust);
          } else {
            /*  More than full step taken; increase the radius */
            tao->trust = PetscMax(nlsP->omega5 * norm_d, tao->trust);
          }
        } else {
          /*  Newton step was not good; reduce the radius */
          tao->trust = nlsP->omega1 * PetscMin(norm_d, tao->trust);
        }
        break;

      case NLS_UPDATE_REDUCTION:
        if (stepType == NLS_NEWTON) {
          /*  Get predicted reduction */

          ierr = KSPCGGetObjFcn(tao->ksp,&prered);CHKERRQ(ierr);
          if (prered >= 0.0) {
            /*  The predicted reduction has the wrong sign.  This cannot */
            /*  happen in infinite precision arithmetic.  Step should */
            /*  be rejected! */
            tao->trust = nlsP->alpha1 * PetscMin(tao->trust, norm_d);
          } else {
            if (PetscIsInfOrNanReal(f_full)) {
              tao->trust = nlsP->alpha1 * PetscMin(tao->trust, norm_d);
            } else {
              /*  Compute and actual reduction */
              actred = fold - f_full;
              prered = -prered;
              if ((PetscAbsScalar(actred) <= nlsP->epsilon) &&
                  (PetscAbsScalar(prered) <= nlsP->epsilon)) {
                kappa = 1.0;
              } else {
                kappa = actred / prered;
              }

              /*  Accept of reject the step and update radius */
              if (kappa < nlsP->eta1) {
                /*  Very bad step */
                tao->trust = nlsP->alpha1 * PetscMin(tao->trust, norm_d);
              } else if (kappa < nlsP->eta2) {
                /*  Marginal bad step */
                tao->trust = nlsP->alpha2 * PetscMin(tao->trust, norm_d);
              } else if (kappa < nlsP->eta3) {
                /*  Reasonable step */
                if (nlsP->alpha3 < 1.0) {
                  tao->trust = nlsP->alpha3 * PetscMin(norm_d, tao->trust);
                } else if (nlsP->alpha3 > 1.0) {
                  tao->trust = PetscMax(nlsP->alpha3 * norm_d, tao->trust);
                }
              } else if (kappa < nlsP->eta4) {
                /*  Good step */
                tao->trust = PetscMax(nlsP->alpha4 * norm_d, tao->trust);
              } else {
                /*  Very good step */
                tao->trust = PetscMax(nlsP->alpha5 * norm_d, tao->trust);
              }
            }
          }
        } else {
          /*  Newton step was not good; reduce the radius */
          tao->trust = nlsP->alpha1 * PetscMin(norm_d, tao->trust);
        }
        break;

      default:
        if (stepType == NLS_NEWTON) {
          ierr = KSPCGGetObjFcn(tao->ksp,&prered);CHKERRQ(ierr);
          if (prered >= 0.0) {
            /*  The predicted reduction has the wrong sign.  This cannot */
            /*  happen in infinite precision arithmetic.  Step should */
            /*  be rejected! */
            tao->trust = nlsP->gamma1 * PetscMin(tao->trust, norm_d);
          } else {
            if (PetscIsInfOrNanReal(f_full)) {
              tao->trust = nlsP->gamma1 * PetscMin(tao->trust, norm_d);
            } else {
              actred = fold - f_full;
              prered = -prered;
              if ((PetscAbsScalar(actred) <= nlsP->epsilon) && (PetscAbsScalar(prered) <= nlsP->epsilon)) {
                kappa = 1.0;
              } else {
                kappa = actred / prered;
              }

              tau_1 = nlsP->theta * gdx / (nlsP->theta * gdx - (1.0 - nlsP->theta) * prered + actred);
              tau_2 = nlsP->theta * gdx / (nlsP->theta * gdx + (1.0 + nlsP->theta) * prered - actred);
              tau_min = PetscMin(tau_1, tau_2);
              tau_max = PetscMax(tau_1, tau_2);

              if (kappa >= 1.0 - nlsP->mu1) {
                /*  Great agreement */
                if (tau_max < 1.0) {
                  tao->trust = PetscMax(tao->trust, nlsP->gamma3 * norm_d);
                } else if (tau_max > nlsP->gamma4) {
                  tao->trust = PetscMax(tao->trust, nlsP->gamma4 * norm_d);
                } else {
                  tao->trust = PetscMax(tao->trust, tau_max * norm_d);
                }
              } else if (kappa >= 1.0 - nlsP->mu2) {
                /*  Good agreement */

                if (tau_max < nlsP->gamma2) {
                  tao->trust = nlsP->gamma2 * PetscMin(tao->trust, norm_d);
                } else if (tau_max > nlsP->gamma3) {
                  tao->trust = PetscMax(tao->trust, nlsP->gamma3 * norm_d);
                } else if (tau_max < 1.0) {
                  tao->trust = tau_max * PetscMin(tao->trust, norm_d);
                } else {
                  tao->trust = PetscMax(tao->trust, tau_max * norm_d);
                }
              } else {
                /*  Not good agreement */
                if (tau_min > 1.0) {
                  tao->trust = nlsP->gamma2 * PetscMin(tao->trust, norm_d);
                } else if (tau_max < nlsP->gamma1) {
                  tao->trust = nlsP->gamma1 * PetscMin(tao->trust, norm_d);
                } else if ((tau_min < nlsP->gamma1) && (tau_max >= 1.0)) {
                  tao->trust = nlsP->gamma1 * PetscMin(tao->trust, norm_d);
                } else if ((tau_1 >= nlsP->gamma1) && (tau_1 < 1.0) && ((tau_2 < nlsP->gamma1) || (tau_2 >= 1.0))) {
                  tao->trust = tau_1 * PetscMin(tao->trust, norm_d);
                } else if ((tau_2 >= nlsP->gamma1) && (tau_2 < 1.0) && ((tau_1 < nlsP->gamma1) || (tau_2 >= 1.0))) {
                  tao->trust = tau_2 * PetscMin(tao->trust, norm_d);
                } else {
                  tao->trust = tau_max * PetscMin(tao->trust, norm_d);
                }
              }
            }
          }
        } else {
          /*  Newton step was not good; reduce the radius */
          tao->trust = nlsP->gamma1 * PetscMin(norm_d, tao->trust);
        }
        break;
      }

      /*  The radius may have been increased; modify if it is too large */
      tao->trust = PetscMin(tao->trust, nlsP->max_radius);
    }

    /*  Check for termination */
    ierr = TaoGradientNorm(tao, tao->gradient,NORM_2,&gnorm);CHKERRQ(ierr);
    PetscCheckFalse(PetscIsInfOrNanReal(f) || PetscIsInfOrNanReal(gnorm),PetscObjectComm((PetscObject)tao),PETSC_ERR_USER,"User provided compute function generated Not-a-Number");
    needH = 1;
    ierr = TaoLogConvergenceHistory(tao,f,gnorm,0.0,tao->ksp_its);CHKERRQ(ierr);
    ierr = TaoMonitor(tao,tao->niter,f,gnorm,0.0,step);CHKERRQ(ierr);
    ierr = (*tao->ops->convergencetest)(tao,tao->cnvP);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------- */
static PetscErrorCode TaoSetUp_NLS(Tao tao)
{
  TAO_NLS        *nlsP = (TAO_NLS *)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!tao->gradient) {ierr = VecDuplicate(tao->solution,&tao->gradient);CHKERRQ(ierr);}
  if (!tao->stepdirection) {ierr = VecDuplicate(tao->solution,&tao->stepdirection);CHKERRQ(ierr);}
  if (!nlsP->W) {ierr = VecDuplicate(tao->solution,&nlsP->W);CHKERRQ(ierr);}
  if (!nlsP->D) {ierr = VecDuplicate(tao->solution,&nlsP->D);CHKERRQ(ierr);}
  if (!nlsP->Xold) {ierr = VecDuplicate(tao->solution,&nlsP->Xold);CHKERRQ(ierr);}
  if (!nlsP->Gold) {ierr = VecDuplicate(tao->solution,&nlsP->Gold);CHKERRQ(ierr);}
  nlsP->M = NULL;
  nlsP->bfgs_pre = NULL;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
static PetscErrorCode TaoDestroy_NLS(Tao tao)
{
  TAO_NLS        *nlsP = (TAO_NLS *)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (tao->setupcalled) {
    ierr = VecDestroy(&nlsP->D);CHKERRQ(ierr);
    ierr = VecDestroy(&nlsP->W);CHKERRQ(ierr);
    ierr = VecDestroy(&nlsP->Xold);CHKERRQ(ierr);
    ierr = VecDestroy(&nlsP->Gold);CHKERRQ(ierr);
  }
  ierr = PetscFree(tao->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
static PetscErrorCode TaoSetFromOptions_NLS(PetscOptionItems *PetscOptionsObject,Tao tao)
{
  TAO_NLS        *nlsP = (TAO_NLS *)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"Newton line search method for unconstrained optimization");CHKERRQ(ierr);
  ierr = PetscOptionsEList("-tao_nls_init_type", "radius initialization type", "", NLS_INIT, NLS_INIT_TYPES, NLS_INIT[nlsP->init_type], &nlsP->init_type, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEList("-tao_nls_update_type", "radius update type", "", NLS_UPDATE, NLS_UPDATE_TYPES, NLS_UPDATE[nlsP->update_type], &nlsP->update_type, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_sval", "perturbation starting value", "", nlsP->sval, &nlsP->sval,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_imin", "minimum initial perturbation", "", nlsP->imin, &nlsP->imin,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_imax", "maximum initial perturbation", "", nlsP->imax, &nlsP->imax,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_imfac", "initial merit factor", "", nlsP->imfac, &nlsP->imfac,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_pmin", "minimum perturbation", "", nlsP->pmin, &nlsP->pmin,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_pmax", "maximum perturbation", "", nlsP->pmax, &nlsP->pmax,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_pgfac", "growth factor", "", nlsP->pgfac, &nlsP->pgfac,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_psfac", "shrink factor", "", nlsP->psfac, &nlsP->psfac,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_pmgfac", "merit growth factor", "", nlsP->pmgfac, &nlsP->pmgfac,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_pmsfac", "merit shrink factor", "", nlsP->pmsfac, &nlsP->pmsfac,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_eta1", "poor steplength; reduce radius", "", nlsP->eta1, &nlsP->eta1,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_eta2", "reasonable steplength; leave radius alone", "", nlsP->eta2, &nlsP->eta2,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_eta3", "good steplength; increase radius", "", nlsP->eta3, &nlsP->eta3,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_eta4", "excellent steplength; greatly increase radius", "", nlsP->eta4, &nlsP->eta4,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_alpha1", "", "", nlsP->alpha1, &nlsP->alpha1,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_alpha2", "", "", nlsP->alpha2, &nlsP->alpha2,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_alpha3", "", "", nlsP->alpha3, &nlsP->alpha3,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_alpha4", "", "", nlsP->alpha4, &nlsP->alpha4,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_alpha5", "", "", nlsP->alpha5, &nlsP->alpha5,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_nu1", "poor steplength; reduce radius", "", nlsP->nu1, &nlsP->nu1,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_nu2", "reasonable steplength; leave radius alone", "", nlsP->nu2, &nlsP->nu2,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_nu3", "good steplength; increase radius", "", nlsP->nu3, &nlsP->nu3,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_nu4", "excellent steplength; greatly increase radius", "", nlsP->nu4, &nlsP->nu4,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_omega1", "", "", nlsP->omega1, &nlsP->omega1,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_omega2", "", "", nlsP->omega2, &nlsP->omega2,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_omega3", "", "", nlsP->omega3, &nlsP->omega3,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_omega4", "", "", nlsP->omega4, &nlsP->omega4,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_omega5", "", "", nlsP->omega5, &nlsP->omega5,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_mu1_i", "", "", nlsP->mu1_i, &nlsP->mu1_i,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_mu2_i", "", "", nlsP->mu2_i, &nlsP->mu2_i,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_gamma1_i", "", "", nlsP->gamma1_i, &nlsP->gamma1_i,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_gamma2_i", "", "", nlsP->gamma2_i, &nlsP->gamma2_i,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_gamma3_i", "", "", nlsP->gamma3_i, &nlsP->gamma3_i,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_gamma4_i", "", "", nlsP->gamma4_i, &nlsP->gamma4_i,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_theta_i", "", "", nlsP->theta_i, &nlsP->theta_i,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_mu1", "", "", nlsP->mu1, &nlsP->mu1,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_mu2", "", "", nlsP->mu2, &nlsP->mu2,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_gamma1", "", "", nlsP->gamma1, &nlsP->gamma1,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_gamma2", "", "", nlsP->gamma2, &nlsP->gamma2,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_gamma3", "", "", nlsP->gamma3, &nlsP->gamma3,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_gamma4", "", "", nlsP->gamma4, &nlsP->gamma4,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_theta", "", "", nlsP->theta, &nlsP->theta,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_min_radius", "lower bound on initial radius", "", nlsP->min_radius, &nlsP->min_radius,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_max_radius", "upper bound on radius", "", nlsP->max_radius, &nlsP->max_radius,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_epsilon", "tolerance used when computing actual and predicted reduction", "", nlsP->epsilon, &nlsP->epsilon,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  ierr = TaoLineSearchSetFromOptions(tao->linesearch);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(tao->ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
static PetscErrorCode TaoView_NLS(Tao tao, PetscViewer viewer)
{
  TAO_NLS        *nlsP = (TAO_NLS *)tao->data;
  PetscBool      isascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "Newton steps: %D\n", nlsP->newt);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "BFGS steps: %D\n", nlsP->bfgs);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "Gradient steps: %D\n", nlsP->grad);CHKERRQ(ierr);

    ierr = PetscViewerASCIIPrintf(viewer, "nls ksp atol: %D\n", nlsP->ksp_atol);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "nls ksp rtol: %D\n", nlsP->ksp_rtol);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "nls ksp ctol: %D\n", nlsP->ksp_ctol);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "nls ksp negc: %D\n", nlsP->ksp_negc);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "nls ksp dtol: %D\n", nlsP->ksp_dtol);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "nls ksp iter: %D\n", nlsP->ksp_iter);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "nls ksp othr: %D\n", nlsP->ksp_othr);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------- */
/*MC
  TAONLS - Newton's method with linesearch for unconstrained minimization.
  At each iteration, the Newton line search method solves the symmetric
  system of equations to obtain the step diretion dk:
              Hk dk = -gk
  a More-Thuente line search is applied on the direction dk to approximately
  solve
              min_t f(xk + t d_k)

    Options Database Keys:
+ -tao_nls_init_type - "constant","direction","interpolation"
. -tao_nls_update_type - "step","direction","interpolation"
. -tao_nls_sval - perturbation starting value
. -tao_nls_imin - minimum initial perturbation
. -tao_nls_imax - maximum initial perturbation
. -tao_nls_pmin - minimum perturbation
. -tao_nls_pmax - maximum perturbation
. -tao_nls_pgfac - growth factor
. -tao_nls_psfac - shrink factor
. -tao_nls_imfac - initial merit factor
. -tao_nls_pmgfac - merit growth factor
. -tao_nls_pmsfac - merit shrink factor
. -tao_nls_eta1 - poor steplength; reduce radius
. -tao_nls_eta2 - reasonable steplength; leave radius
. -tao_nls_eta3 - good steplength; increase readius
. -tao_nls_eta4 - excellent steplength; greatly increase radius
. -tao_nls_alpha1 - alpha1 reduction
. -tao_nls_alpha2 - alpha2 reduction
. -tao_nls_alpha3 - alpha3 reduction
. -tao_nls_alpha4 - alpha4 reduction
. -tao_nls_alpha - alpha5 reduction
. -tao_nls_mu1 - mu1 interpolation update
. -tao_nls_mu2 - mu2 interpolation update
. -tao_nls_gamma1 - gamma1 interpolation update
. -tao_nls_gamma2 - gamma2 interpolation update
. -tao_nls_gamma3 - gamma3 interpolation update
. -tao_nls_gamma4 - gamma4 interpolation update
. -tao_nls_theta - theta interpolation update
. -tao_nls_omega1 - omega1 step update
. -tao_nls_omega2 - omega2 step update
. -tao_nls_omega3 - omega3 step update
. -tao_nls_omega4 - omega4 step update
. -tao_nls_omega5 - omega5 step update
. -tao_nls_mu1_i -  mu1 interpolation init factor
. -tao_nls_mu2_i -  mu2 interpolation init factor
. -tao_nls_gamma1_i -  gamma1 interpolation init factor
. -tao_nls_gamma2_i -  gamma2 interpolation init factor
. -tao_nls_gamma3_i -  gamma3 interpolation init factor
. -tao_nls_gamma4_i -  gamma4 interpolation init factor
- -tao_nls_theta_i -  theta interpolation init factor

  Level: beginner
M*/

PETSC_EXTERN PetscErrorCode TaoCreate_NLS(Tao tao)
{
  TAO_NLS        *nlsP;
  const char     *morethuente_type = TAOLINESEARCHMT;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(tao,&nlsP);CHKERRQ(ierr);

  tao->ops->setup = TaoSetUp_NLS;
  tao->ops->solve = TaoSolve_NLS;
  tao->ops->view = TaoView_NLS;
  tao->ops->setfromoptions = TaoSetFromOptions_NLS;
  tao->ops->destroy = TaoDestroy_NLS;

  /* Override default settings (unless already changed) */
  if (!tao->max_it_changed) tao->max_it = 50;
  if (!tao->trust0_changed) tao->trust0 = 100.0;

  tao->data = (void*)nlsP;

  nlsP->sval   = 0.0;
  nlsP->imin   = 1.0e-4;
  nlsP->imax   = 1.0e+2;
  nlsP->imfac  = 1.0e-1;

  nlsP->pmin   = 1.0e-12;
  nlsP->pmax   = 1.0e+2;
  nlsP->pgfac  = 1.0e+1;
  nlsP->psfac  = 4.0e-1;
  nlsP->pmgfac = 1.0e-1;
  nlsP->pmsfac = 1.0e-1;

  /*  Default values for trust-region radius update based on steplength */
  nlsP->nu1 = 0.25;
  nlsP->nu2 = 0.50;
  nlsP->nu3 = 1.00;
  nlsP->nu4 = 1.25;

  nlsP->omega1 = 0.25;
  nlsP->omega2 = 0.50;
  nlsP->omega3 = 1.00;
  nlsP->omega4 = 2.00;
  nlsP->omega5 = 4.00;

  /*  Default values for trust-region radius update based on reduction */
  nlsP->eta1 = 1.0e-4;
  nlsP->eta2 = 0.25;
  nlsP->eta3 = 0.50;
  nlsP->eta4 = 0.90;

  nlsP->alpha1 = 0.25;
  nlsP->alpha2 = 0.50;
  nlsP->alpha3 = 1.00;
  nlsP->alpha4 = 2.00;
  nlsP->alpha5 = 4.00;

  /*  Default values for trust-region radius update based on interpolation */
  nlsP->mu1 = 0.10;
  nlsP->mu2 = 0.50;

  nlsP->gamma1 = 0.25;
  nlsP->gamma2 = 0.50;
  nlsP->gamma3 = 2.00;
  nlsP->gamma4 = 4.00;

  nlsP->theta = 0.05;

  /*  Default values for trust region initialization based on interpolation */
  nlsP->mu1_i = 0.35;
  nlsP->mu2_i = 0.50;

  nlsP->gamma1_i = 0.0625;
  nlsP->gamma2_i = 0.5;
  nlsP->gamma3_i = 2.0;
  nlsP->gamma4_i = 5.0;

  nlsP->theta_i = 0.25;

  /*  Remaining parameters */
  nlsP->min_radius = 1.0e-10;
  nlsP->max_radius = 1.0e10;
  nlsP->epsilon = 1.0e-6;

  nlsP->init_type       = NLS_INIT_INTERPOLATION;
  nlsP->update_type     = NLS_UPDATE_STEP;

  ierr = TaoLineSearchCreate(((PetscObject)tao)->comm,&tao->linesearch);CHKERRQ(ierr);
  ierr = PetscObjectIncrementTabLevel((PetscObject)tao->linesearch,(PetscObject)tao,1);CHKERRQ(ierr);
  ierr = TaoLineSearchSetType(tao->linesearch,morethuente_type);CHKERRQ(ierr);
  ierr = TaoLineSearchUseTaoRoutines(tao->linesearch,tao);CHKERRQ(ierr);
  ierr = TaoLineSearchSetOptionsPrefix(tao->linesearch,tao->hdr.prefix);CHKERRQ(ierr);

  /*  Set linear solver to default for symmetric matrices */
  ierr = KSPCreate(((PetscObject)tao)->comm,&tao->ksp);CHKERRQ(ierr);
  ierr = PetscObjectIncrementTabLevel((PetscObject)tao->ksp,(PetscObject)tao,1);CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(tao->ksp,tao->hdr.prefix);CHKERRQ(ierr);
  ierr = KSPAppendOptionsPrefix(tao->ksp,"tao_nls_");CHKERRQ(ierr);
  ierr = KSPSetType(tao->ksp,KSPSTCG);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
