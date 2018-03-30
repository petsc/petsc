#include <../src/tao/bound/impls/bnk/bnk.h>
#include <petscksp.h>

/*
 Implements Newton's Method with a line search approach for solving
 bound constrained minimization problems. A projected More'-Thuente line 
 search is used to guarantee that the bfgs preconditioner remains positive
 definite.

 The method can shift the Hessian matrix. The shifting procedure is
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

  PetscReal                    f_full, gdx, prered, actred;
  PetscReal                    step = 1.0;
  PetscReal                    delta;
  PetscReal                    e_min;
  
  PetscBool                    trustAccept;
  PetscInt                     stepType;
  PetscInt                     bfgsUpdates = 0;
  
  PetscFunctionBegin;
  /*   Project the current point onto the feasible set */
  ierr = TaoComputeVariableBounds(tao);CHKERRQ(ierr);
  ierr = TaoLineSearchSetVariableBounds(tao->linesearch,tao->XL,tao->XU);CHKERRQ(ierr);

  /* Project the initial point onto the feasible region */
  ierr = VecMedian(tao->XL,tao->solution,tao->XU,tao->solution);CHKERRQ(ierr);

  /* Check convergence criteria */
  ierr = TaoComputeObjectiveAndGradient(tao, tao->solution, &bnk->f, bnk->unprojected_gradient);CHKERRQ(ierr);
  ierr = VecBoundGradientProjection(bnk->unprojected_gradient,tao->solution,tao->XL,tao->XU,tao->gradient);CHKERRQ(ierr);
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
    
    /* Compute the Hessian */
    ierr = TaoComputeHessian(tao,tao->solution,tao->hessian,tao->hessian_pre);CHKERRQ(ierr);
    
    /* Update the BFGS preconditioner */
    if (BNK_PC_BFGS == bnk->pc_type) {
      if (BFGS_SCALE_PHESS == bnk->bfgs_scale_type) {
        /* Obtain diagonal for the bfgs preconditioner  */
        ierr = MatGetDiagonal(tao->hessian, bnk->Diag);CHKERRQ(ierr);
        ierr = VecAbs(bnk->Diag);CHKERRQ(ierr);
        ierr = VecReciprocal(bnk->Diag);CHKERRQ(ierr);
        ierr = MatLMVMSetScale(bnk->M,bnk->Diag);CHKERRQ(ierr);
      }
      /* Update the limited memory preconditioner and get existing # of updates */
      ierr = MatLMVMUpdate(bnk->M, tao->solution, bnk->unprojected_gradient);CHKERRQ(ierr);
      ierr = MatLMVMGetUpdates(bnk->M, &bfgsUpdates);CHKERRQ(ierr);
    }
    
    /* Use the common BNK kernel to compute the safeguarded Newton step */
    ierr = TaoBNKComputeStep(tao, PETSC_TRUE, &stepType);CHKERRQ(ierr);

    /* Store current solution before it changes */
    bnk->fold = bnk->f;
    ierr = VecCopy(tao->solution, bnk->Xold);CHKERRQ(ierr);
    ierr = VecCopy(tao->gradient, bnk->Gold);CHKERRQ(ierr);
    ierr = VecCopy(bnk->unprojected_gradient, bnk->unprojected_gradient_old);CHKERRQ(ierr);
    
    /* Perform the linesearch */
    ierr = TaoLineSearchApply(tao->linesearch, tao->solution, &bnk->f, bnk->unprojected_gradient, tao->stepdirection, &step, &ls_reason);CHKERRQ(ierr);
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

      ierr = TaoLineSearchApply(tao->linesearch, tao->solution, &bnk->f, bnk->unprojected_gradient, tao->stepdirection, &step, &ls_reason);CHKERRQ(ierr);
      ierr = VecBoundGradientProjection(bnk->unprojected_gradient,tao->solution,tao->XL,tao->XU,tao->gradient);CHKERRQ(ierr);
      ierr = TaoAddLineSearchCounts(tao);CHKERRQ(ierr);
    }

    if (ls_reason != TAOLINESEARCH_SUCCESS && ls_reason != TAOLINESEARCH_SUCCESS_USER) {
      /* Failed to find an improving point */
      bnk->f = bnk->fold;
      ierr = VecCopy(bnk->Xold, tao->solution);CHKERRQ(ierr);
      ierr = VecCopy(bnk->Gold, tao->gradient);CHKERRQ(ierr);
      ierr = VecCopy(bnk->unprojected_gradient_old, bnk->unprojected_gradient);CHKERRQ(ierr);
      step = 0.0;
      tao->reason = TAO_DIVERGED_LS_FAILURE;
      break;
    }
    
    /* update trust radius */
    ierr = TaoLineSearchGetFullStepObjective(tao->linesearch, &f_full);CHKERRQ(ierr);
    ierr = KSPCGGetObjFcn(tao->ksp, &prered);CHKERRQ(ierr);
    prered = -prered;
    actred = bnk->fold - f_full;
    ierr = TaoBNKUpdateTrustRadius(tao, prered, actred, stepType, &trustAccept);CHKERRQ(ierr);

    /*  Check for termination */
    ierr = TaoGradientNorm(tao, tao->gradient,NORM_2,&bnk->gnorm);CHKERRQ(ierr);
    if (PetscIsInfOrNanReal(bnk->f) || PetscIsInfOrNanReal(bnk->gnorm)) SETERRQ(PETSC_COMM_SELF,1,"User provided compute function generated Not-a-Number");
    ierr = TaoLogConvergenceHistory(tao,bnk->f,bnk->gnorm,0.0,tao->ksp_its);CHKERRQ(ierr);
    ierr = TaoMonitor(tao,tao->niter,bnk->f,bnk->gnorm,0.0,step);CHKERRQ(ierr);
    ierr = (*tao->ops->convergencetest)(tao,tao->cnvP);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode TaoCreate_BNLS(Tao tao)
{
  TAO_BNK        *bnk;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = TaoCreate_BNK(tao);CHKERRQ(ierr);
  tao->ops->solve = TaoSolve_BNLS;
  
  bnk = (TAO_BNK *)tao->data;
  bnk->update_type = BNK_UPDATE_STEP; /* trust region updates based on line search step length */
  PetscFunctionReturn(0);
}