#include <../src/tao/bound/impls/bnk/bnk.h>
#include <petscksp.h>

/*
 Implements Newton's Method with a line search approach for solving
 bound constrained minimization problems. A projected More'-Thuente line 
 search is used to guarantee that the BFGS preconditioner remains positive
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

  PetscReal                    f_full, prered, actred;
  PetscReal                    steplen = 1.0;
  
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
  ierr = TaoMonitor(tao,tao->niter,bnk->f,bnk->gnorm,0.0,steplen);CHKERRQ(ierr);
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
    
    /* Use the common BNK kernel to compute the safeguarded Newton step (for inactive variables only) */
    ierr = TaoBNKComputeStep(tao, PETSC_TRUE, &stepType);CHKERRQ(ierr);

    /* Store current solution before it changes */
    bnk->fold = bnk->f;
    ierr = VecCopy(tao->solution, bnk->Xold);CHKERRQ(ierr);
    ierr = VecCopy(tao->gradient, bnk->Gold);CHKERRQ(ierr);
    ierr = VecCopy(bnk->unprojected_gradient, bnk->unprojected_gradient_old);CHKERRQ(ierr);
    
    /* Trigger the line search */
    ierr = TaoBNKPerformLineSearch(tao, stepType, &steplen, &ls_reason);CHKERRQ(ierr);

    if (ls_reason != TAOLINESEARCH_SUCCESS && ls_reason != TAOLINESEARCH_SUCCESS_USER) {
      /* Failed to find an improving point */
      bnk->f = bnk->fold;
      ierr = VecCopy(bnk->Xold, tao->solution);CHKERRQ(ierr);
      ierr = VecCopy(bnk->Gold, tao->gradient);CHKERRQ(ierr);
      ierr = VecCopy(bnk->unprojected_gradient_old, bnk->unprojected_gradient);CHKERRQ(ierr);
      steplen = 0.0;
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
    ierr = TaoMonitor(tao,tao->niter,bnk->f,bnk->gnorm,0.0,steplen);CHKERRQ(ierr);
    ierr = (*tao->ops->convergencetest)(tao,tao->cnvP);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

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