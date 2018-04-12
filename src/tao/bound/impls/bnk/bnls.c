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
  KSPConvergedReason           ksp_reason;
  TaoLineSearchConvergedReason ls_reason;

  PetscReal                    steplen = 1.0;
  PetscBool                    shift = PETSC_TRUE;
  PetscInt                     stepType;
  
  PetscFunctionBegin;
  /* Initialize the preconditioner, KSP solver and trust radius/line search */
  tao->reason = TAO_CONTINUE_ITERATING;
  ierr = TaoBNKInitialize(tao, BNK_INIT_CONSTANT);CHKERRQ(ierr);
  if (tao->reason != TAO_CONTINUE_ITERATING) PetscFunctionReturn(0);

  /* Have not converged; continue with Newton method */
  while (tao->reason == TAO_CONTINUE_ITERATING) {
    ++tao->niter;
    tao->ksp_its=0;
    
    /* Compute the hessian and update the BFGS preconditioner at the new iterate*/
    ierr = TaoBNKComputeHessian(tao);CHKERRQ(ierr);
    
    /* Use the common BNK kernel to compute the safeguarded Newton step (for inactive variables only) */
    tao->trust = bnk->max_radius;
    ierr = TaoBNKComputeStep(tao, shift, &ksp_reason);CHKERRQ(ierr);
    ierr = TaoBNKSafeguardStep(tao, ksp_reason, &stepType);CHKERRQ(ierr);

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
    } else {
      /* count the accepted step type */
      ierr = TaoBNKAddStepCounts(tao, stepType);CHKERRQ(ierr);
    }

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
  bnk->init_type = BNK_INIT_CONSTANT;
  bnk->update_type = BNK_UPDATE_STEP; /* trust region updates based on line search step length */
  PetscFunctionReturn(0);
}