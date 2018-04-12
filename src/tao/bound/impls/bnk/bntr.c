#include <../src/tao/bound/impls/bnk/bnk.h>
#include <petscksp.h>

/*
 Implements Newton's Method with a trust region approach for solving
 bound constrained minimization problems.

 The linear system solve has to be done with a conjugate gradient method.
*/

static PetscErrorCode TaoSolve_BNTR(Tao tao)
{
  PetscErrorCode               ierr;
  TAO_BNK                      *bnk = (TAO_BNK *)tao->data;
  KSPConvergedReason           ksp_reason;

  PetscReal                    oldTrust, prered, actred, stepNorm, steplen;
  PetscBool                    stepAccepted = PETSC_TRUE;
  PetscInt                     stepType = BNK_NEWTON;
  
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
    
    /* Use the common BNK kernel to compute the Newton step (for inactive variables only) */
    ierr = TaoBNKComputeStep(tao, &ksp_reason);CHKERRQ(ierr);

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
    ierr = VecAXPY(tao->stepdirection, -1.0, bnk->Xold);CHKERRQ(ierr);
    ierr = VecNorm(tao->stepdirection, NORM_2, &stepNorm);CHKERRQ(ierr);
    if (stepNorm != bnk->dnorm) {
      /* Projection changed the step, so we have to recompute predicted reduction.
         However, we deliberately do not change the step norm and the trust radius 
         in order for the safeguard to more closely mimic a piece-wise linesearch 
         along the bounds. */
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
      steplen = 1.0;
      ++bnk->newt;
      ierr = TaoComputeGradient(tao, tao->solution, bnk->unprojected_gradient);CHKERRQ(ierr);
      ierr = VecBoundGradientProjection(bnk->unprojected_gradient,tao->solution,tao->XL,tao->XU,tao->gradient);CHKERRQ(ierr);
    } else {
      /* Step is bad, revert old solution and re-solve with new radius*/
      steplen = 0.0;
      bnk->f = bnk->fold;
      ierr = VecCopy(bnk->Xold, tao->solution);CHKERRQ(ierr);
      ierr = VecCopy(bnk->Gold, tao->gradient);CHKERRQ(ierr);
      ierr = VecCopy(bnk->unprojected_gradient_old, bnk->unprojected_gradient);CHKERRQ(ierr);
      if (oldTrust == tao->trust) {
        /* Can't change the radius anymore so just terminate */
        tao->reason = TAO_DIVERGED_TR_REDUCTION;
      }
    }

    /*  Check for termination */
    ierr = TaoGradientNorm(tao, tao->gradient, NORM_2, &bnk->gnorm);CHKERRQ(ierr);
    if (PetscIsInfOrNanReal(bnk->gnorm)) SETERRQ(PETSC_COMM_SELF,1,"User provided compute function generated Not-a-Number");
    ierr = TaoLogConvergenceHistory(tao,bnk->f,bnk->gnorm,0.0,tao->ksp_its);CHKERRQ(ierr);
    ierr = TaoMonitor(tao,tao->niter,bnk->f,bnk->gnorm,0.0,steplen);CHKERRQ(ierr);
    ierr = (*tao->ops->convergencetest)(tao,tao->cnvP);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PETSC_EXTERN PetscErrorCode TaoCreate_BNTR(Tao tao)
{
  TAO_BNK        *bnk;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = TaoCreate_BNK(tao);CHKERRQ(ierr);
  tao->ops->solve=TaoSolve_BNTR;
  
  bnk = (TAO_BNK *)tao->data;
  bnk->update_type = BNK_UPDATE_REDUCTION; /* trust region updates based on predicted/actual reduction */
  bnk->sval = 0.0; /* disable Hessian shifting */
  PetscFunctionReturn(0);
}