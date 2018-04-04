#include <../src/tao/bound/impls/bnk/bnk.h>
#include <petscksp.h>

/*
 Implements Newton's Method with a trust region approach for solving
 bound constrained minimization problems. This version includes a 
 line search fall-back in the event of a trust region failure.

 The linear system solve has to be done with a conjugate gradient method.
*/

static PetscErrorCode TaoSolve_BNTL(Tao tao)
{
  PetscErrorCode               ierr;
  TAO_BNK                      *bnk = (TAO_BNK *)tao->data;
  TaoLineSearchConvergedReason ls_reason;

  PetscReal                    oldTrust, prered, actred, stepNorm, gdx, delta, steplen;
  PetscBool                    stepAccepted = PETSC_TRUE;
  PetscInt                     stepType, bfgsUpdates, updateType;
  
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
  ierr = TaoMonitor(tao,tao->niter,bnk->f,bnk->gnorm,0.0,tao->trust);CHKERRQ(ierr);
  ierr = (*tao->ops->convergencetest)(tao,tao->cnvP);CHKERRQ(ierr);
  if (tao->reason != TAO_CONTINUE_ITERATING) PetscFunctionReturn(0);
  
  /* Initialize the preconditioner and trust radius */
  ierr = TaoBNKInitialize(tao);CHKERRQ(ierr);

  /* Have not converged; continue with Newton method */
  while (tao->reason == TAO_CONTINUE_ITERATING) {
    
    if (stepAccepted) { 
      tao->niter++;
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
      }
    }
    
    /* Use the common BNK kernel to compute the raw Newton step */
    ierr = TaoBNKComputeStep(tao, PETSC_FALSE, &stepType);CHKERRQ(ierr);

    /* Store current solution before it changes */
    oldTrust = tao->trust;
    bnk->fold = bnk->f;
    ierr = VecCopy(tao->solution, bnk->Xold);CHKERRQ(ierr);
    ierr = VecCopy(tao->gradient, bnk->Gold);CHKERRQ(ierr);
    ierr = VecCopy(bnk->unprojected_gradient, bnk->unprojected_gradient_old);CHKERRQ(ierr);
    
    /* Temporarily accept the step and project it into the bounds */
    ierr = VecAXPY(tao->solution, 1.0, tao->stepdirection);CHKERRQ(ierr);
    ierr = VecMedian(tao->XL, tao->solution, tao->XU, tao->solution);CHKERRQ(ierr);
    
    /* Check if the projection changed the step direction */
    ierr = VecCopy(tao->solution, tao->stepdirection);CHKERRQ(ierr);
    ierr = VecAXPBY(tao->stepdirection, -1.0, 1.0, bnk->Xold);CHKERRQ(ierr);
    ierr = VecNorm(tao->stepdirection, NORM_2, &stepNorm);CHKERRQ(ierr);
    if (stepNorm != bnk->dnorm) {
      /* Projection changed the step, so we have to adjust trust radius and recompute predicted reduction */
      bnk->dnorm = stepNorm;
      tao->trust = bnk->dnorm;
      ierr = MatMult(tao->hessian, tao->stepdirection, bnk->Xwork);CHKERRQ(ierr);
      ierr = VecAYPX(bnk->Xwork, -0.5, tao->gradient);CHKERRQ(ierr);
      ierr = VecDot(bnk->Xwork, tao->stepdirection, &prered);
    } else {
      /* Step did not change, so we can just recover the pre-computed prediction */
      ierr = KSPCGGetObjFcn(tao->ksp, &prered);CHKERRQ(ierr);
    }
    prered = -prered;
    
    /* Compute the actual reduction and update the trust radius */
    ierr = TaoComputeObjective(tao, tao->solution, &bnk->f);CHKERRQ(ierr);
    actred = bnk->fold - bnk->f;
    ierr = TaoBNKUpdateTrustRadius(tao, prered, actred, stepType, &stepAccepted);CHKERRQ(ierr);
    
    if (stepAccepted) {
      /* Step is good, evaluate the gradient and the hessian */
      ierr = TaoComputeGradient(tao, tao->solution, bnk->unprojected_gradient);CHKERRQ(ierr);
      ierr = VecBoundGradientProjection(bnk->unprojected_gradient,tao->solution,tao->XL,tao->XU,tao->gradient);CHKERRQ(ierr);
    } else {
      /* Trust-region rejected the step. Revert the solution. */
      bnk->f = bnk->fold;
      ierr = VecCopy(bnk->Xold, tao->solution);CHKERRQ(ierr);
      
      /* Now check to make sure the Newton step is a descent direction... */
      ierr = VecDot(tao->stepdirection, tao->gradient, &gdx);CHKERRQ(ierr);
      if ((gdx >= 0.0) || PetscIsInfOrNanReal(gdx)) {
        /* Newton step is not descent or direction produced Inf or NaN */
        --bnk->newt;
        if (BNK_PC_BFGS != bnk->pc_type) {
          /* We don't have the BFGS matrix around and updated
             Must use gradient direction in this case */
          ierr = VecCopy(tao->gradient, tao->stepdirection);CHKERRQ(ierr);
          ierr = VecScale(tao->stepdirection, -1.0);CHKERRQ(ierr);
          ++bnk->grad;
          stepType = BNK_GRADIENT;
        } else {
          /* We have the BFGS matrix, so attempt to use the BFGS direction */
          ierr = MatLMVMSolve(bnk->M, tao->gradient, tao->stepdirection);CHKERRQ(ierr);
          ierr = VecScale(tao->stepdirection, -1.0);CHKERRQ(ierr);

          /* Check for success (descent direction) */
          ierr = VecDot(tao->stepdirection, tao->gradient, &gdx);CHKERRQ(ierr);
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
            ierr = MatLMVMUpdate(bnk->M, tao->solution, tao->gradient);CHKERRQ(ierr);
            ierr = MatLMVMSolve(bnk->M, tao->gradient, tao->stepdirection);CHKERRQ(ierr);
            ierr = VecScale(tao->stepdirection, -1.0);CHKERRQ(ierr);

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
      } 
      
      /* Trigger the line search */
      ierr = TaoBNKPerformLineSearch(tao, stepType, &steplen, &ls_reason);CHKERRQ(ierr);
      if (ls_reason != TAOLINESEARCH_SUCCESS && ls_reason != TAOLINESEARCH_SUCCESS_USER) {
        /* Line search failed, revert solution and terminate */
        bnk->f = bnk->fold;
        ierr = VecCopy(bnk->Xold, tao->solution);CHKERRQ(ierr);
        ierr = VecCopy(bnk->Gold, tao->gradient);CHKERRQ(ierr);
        ierr = VecCopy(bnk->unprojected_gradient_old, bnk->unprojected_gradient);CHKERRQ(ierr);
        tao->trust = 0.0;
        tao->reason = TAO_DIVERGED_LS_FAILURE;
      } else {
        /* Line search succeeded so we should update the trust radius based on the LS step length */
        updateType = bnk->update_type;
        bnk->update_type = BNK_UPDATE_STEP;
        ierr = TaoBNKUpdateTrustRadius(tao, prered, actred, stepType, &stepAccepted);CHKERRQ(ierr);
        bnk->update_type = updateType;
      }
    }

    /*  Check for termination */
    ierr = TaoGradientNorm(tao, tao->gradient, NORM_2, &bnk->gnorm);CHKERRQ(ierr);
    if (PetscIsInfOrNanReal(bnk->gnorm)) SETERRQ(PETSC_COMM_SELF,1,"User provided compute function generated Not-a-Number");
    ierr = TaoLogConvergenceHistory(tao,bnk->f,bnk->gnorm,0.0,tao->ksp_its);CHKERRQ(ierr);
    ierr = TaoMonitor(tao,tao->niter,bnk->f,bnk->gnorm,0.0,tao->trust);CHKERRQ(ierr);
    ierr = (*tao->ops->convergencetest)(tao,tao->cnvP);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PETSC_EXTERN PetscErrorCode TaoCreate_BNTL(Tao tao)
{
  TAO_BNK        *bnk;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = TaoCreate_BNK(tao);CHKERRQ(ierr);
  tao->ops->solve=TaoSolve_BNTL;
  
  bnk = (TAO_BNK *)tao->data;
  bnk->update_type = BNK_UPDATE_REDUCTION; /* trust region updates based on predicted/actual reduction */
  bnk->sval = 0.0; /* disable Hessian shifting */
  PetscFunctionReturn(0);
}