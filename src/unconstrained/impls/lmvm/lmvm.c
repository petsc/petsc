#include "taolinesearch.h"
#include "src/matrix/lmvmmat.h"
#include "lmvm.h"

#define LMVM_BFGS                0
#define LMVM_SCALED_GRADIENT     1
#define LMVM_GRADIENT            2

#undef __FUNCT__  
#define __FUNCT__ "TaoSolverSolve_LMVM"
static PetscErrorCode TaoSolverSolve_LMVM(TaoSolver tao)
{

  TAO_LMVM *lmP = (TAO_LMVM *)tao->data;
  
  PetscReal f, fold, gdx, gnorm;
  PetscReal step = 1.0;

  PetscReal delta;

  PetscErrorCode ierr;
  PetscInt stepType;
  PetscInt iter = 0;
  TaoSolverTerminationReason reason = TAO_CONTINUE_ITERATING;
  TaoLineSearchTerminationReason ls_status = TAOLINESEARCH_CONTINUE_ITERATING;

  PetscFunctionBegin;

  if (tao->XL || tao->XU || tao->ops->computebounds) {
    ierr = PetscPrintf(((PetscObject)tao)->comm,"WARNING: Variable bounds have been set but will be ignored by lmvm algorithm\n"); CHKERRQ(ierr);
  }

  // Check convergence criteria
  ierr = TaoSolverComputeObjectiveAndGradient(tao, tao->solution, &f, tao->gradient); CHKERRQ(ierr);
  ierr = VecNorm(tao->gradient,NORM_2,&gnorm); CHKERRQ(ierr);
  if (TaoInfOrNaN(f) || TaoInfOrNaN(gnorm)) {
    SETERRQ(PETSC_COMM_SELF,1, "User provided compute function generated Inf or NaN");
  }

  ierr = TaoSolverMonitor(tao, iter, f, gnorm, 0.0, step, &reason); CHKERRQ(ierr);
  if (reason != TAO_CONTINUE_ITERATING) {
    PetscFunctionReturn(0);
  }

  // Set initial scaling for the function
  if (f != 0.0) {
    delta = 2.0 * PetscAbsScalar(f) / (gnorm*gnorm);
  }
  else {
    delta = 2.0 / (gnorm*gnorm);
  }
  ierr = MatLMVMSetDelta(lmP->M,delta); CHKERRQ(ierr);

  // Set counter for gradient/reset steps
  lmP->bfgs = 0;
  lmP->sgrad = 0;
  lmP->grad = 0;

  // Have not converged; continue with Newton method
  while (reason == TAO_CONTINUE_ITERATING) {
    // Compute direction
    ierr = MatLMVMUpdate(lmP->M,tao->solution,tao->gradient); CHKERRQ(ierr);
    ierr = MatLMVMSolve(lmP->M, tao->gradient, lmP->D); CHKERRQ(ierr);
    ++lmP->bfgs;

    // Check for success (descent direction)
    ierr = VecDot(lmP->D, tao->gradient, &gdx); CHKERRQ(ierr);
    if ((gdx <= 0.0) || TaoInfOrNaN(gdx)) {
      // Step is not descent or direction produced not a number
      // We can assert bfgsUpdates > 1 in this case because
      // the first solve produces the scaled gradient direction,
      // which is guaranteed to be descent
      //
      // Use steepest descent direction (scaled)
      ++lmP->grad;

      if (f != 0.0) {
        delta = 2.0 * PetscAbsScalar(f) / (gnorm*gnorm);
      }
      else {
        delta = 2.0 / (gnorm*gnorm);
      }
      ierr = MatLMVMSetDelta(lmP->M, delta); CHKERRQ(ierr);
      ierr = MatLMVMReset(lmP->M); CHKERRQ(ierr);
      ierr = MatLMVMUpdate(lmP->M, tao->solution, tao->gradient); CHKERRQ(ierr);
      ierr = MatLMVMSolve(lmP->M,tao->gradient, lmP->D); CHKERRQ(ierr);

      // On a reset, the direction cannot be not a number; it is a 
      // scaled gradient step.  No need to check for this condition.
      // info = D->Norm2(&dnorm); CHKERRQ(info);
      // if (TaoInfOrNaN(dnorm)) {
      //   SETERRQ(PETSC_COMM_SELF,1, "Direction generated Not-a-Number");
      // }

      lmP->bfgs = 1;
      ++lmP->sgrad;
      stepType = LMVM_SCALED_GRADIENT;
    }
    else {
      if (1 == lmP->bfgs) {
        // The first BFGS direction is always the scaled gradient
        ++lmP->sgrad;
        stepType = LMVM_SCALED_GRADIENT;
      }
      else {
        ++lmP->bfgs;
        stepType = LMVM_BFGS;
      }
    }
    ierr = VecScale(lmP->D, -1.0); CHKERRQ(ierr);
    
    // Perform the linesearch
    fold = f;
    ierr = VecCopy(tao->solution, lmP->Xold); CHKERRQ(ierr);
    ierr = VecCopy(tao->gradient, lmP->Gold); CHKERRQ(ierr);

    ierr = TaoLineSearchApply(tao->linesearch, tao->solution, &f, tao->gradient, lmP->D, &step,&ls_status); CHKERRQ(ierr);
    

    while (((int)ls_status < 0) && (stepType != LMVM_GRADIENT)) {
      // Linesearch failed
      // Reset factors and use scaled gradient step
      f = fold;
      ierr = VecCopy(lmP->Xold, tao->solution); CHKERRQ(ierr);
      ierr = VecCopy(lmP->Gold, tao->gradient); CHKERRQ(ierr);
        
      switch(stepType) {
      case LMVM_BFGS:
        // Failed to obtain acceptable iterate with BFGS step
        // Attempt to use the scaled gradient direction

        if (f != 0.0) {
          delta = 2.0 * PetscAbsScalar(f) / (gnorm*gnorm);
        }
        else {
          delta = 2.0 / (gnorm*gnorm);
        }
	ierr = MatLMVMSetDelta(lmP->M, delta); CHKERRQ(ierr);
	ierr = MatLMVMReset(lmP->M); CHKERRQ(ierr);
	ierr = MatLMVMUpdate(lmP->M, tao->solution, tao->gradient); CHKERRQ(ierr);
	ierr = MatLMVMSolve(lmP->M, tao->gradient, lmP->D); CHKERRQ(ierr);

        // On a reset, the direction cannot be not a number; it is a 
        // scaled gradient step.  No need to check for this condition.
        // info = D->Norm2(&dnorm); CHKERRQ(info);
        // if (TaoInfOrNaN(dnorm)) {
        //   SETERRQ(PETSC_COMM_SELF,1, "Direction generated Not-a-Number");
        // }
  
	lmP->bfgs = 1;
	++lmP->sgrad;
	stepType = LMVM_SCALED_GRADIENT;
	break;

      case LMVM_SCALED_GRADIENT:
        // The scaled gradient step did not produce a new iterate;
	// attempt to use the gradient direction.
	// Need to make sure we are not using a different diagonal scaling
	ierr = MatLMVMSetDelta(lmP->M, 1.0); CHKERRQ(ierr);
	ierr = MatLMVMReset(lmP->M); CHKERRQ(ierr);
	ierr = MatLMVMUpdate(lmP->M, tao->solution, tao->gradient); CHKERRQ(ierr);
	ierr = MatLMVMSolve(lmP->M, tao->gradient, lmP->D); CHKERRQ(ierr);

        lmP->bfgs = 1;
        ++lmP->grad;
        stepType = LMVM_GRADIENT;
        break;
      }
      ierr = VecScale(lmP->D, -1.0); CHKERRQ(ierr);
        
      // Perform the linesearch
      ierr = TaoLineSearchApply(tao->linesearch, tao->solution, &f, tao->gradient, lmP->D, &step, &ls_status); CHKERRQ(ierr);
      
    }

    if ((int)ls_status < 0) {
      // Failed to find an improving point
      f = fold;
      ierr = VecCopy(lmP->Xold, tao->solution); CHKERRQ(ierr);
      ierr = VecCopy(lmP->Gold, tao->gradient); CHKERRQ(ierr);
      step = 0.0;
    }
    // Check for termination
    ierr = VecNorm(tao->gradient, NORM_2, &gnorm); CHKERRQ(ierr);
    iter++;
    ierr = TaoSolverMonitor(tao,iter,f,gnorm,0.0,step,&reason); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
#undef __FUNCT__  
#define __FUNCT__ "TaoSolverSetUp_LMVM"
static PetscErrorCode TaoSolverSetUp_LMVM(TaoSolver tao)
{
  TAO_LMVM *lmP = (TAO_LMVM *)tao->data;
  PetscInt n,N;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Existence of tao->solution checked in TaoSolverSetUp() */
  if (!tao->gradient) {ierr = VecDuplicate(tao->solution,&tao->gradient); CHKERRQ(ierr);  }
  if (!tao->stepdirection) {ierr = VecDuplicate(tao->solution,&tao->stepdirection); CHKERRQ(ierr);  }
  if (!lmP->D) {ierr = VecDuplicate(tao->solution,&lmP->D); CHKERRQ(ierr);  }
  if (!lmP->Xold) {ierr = VecDuplicate(tao->solution,&lmP->Xold); CHKERRQ(ierr);  }
  if (!lmP->Gold) {ierr = VecDuplicate(tao->solution,&lmP->Gold); CHKERRQ(ierr);  }
  
  // Create matrix for the limited memory approximation
  ierr = VecGetLocalSize(tao->solution,&n); CHKERRQ(ierr);
  ierr = VecGetSize(tao->solution,&N); CHKERRQ(ierr);
  ierr = MatCreateLMVM(((PetscObject)tao)->comm,n,N,&lmP->M); CHKERRQ(ierr);
  ierr = MatLMVMAllocateVectors(lmP->M,tao->solution); CHKERRQ(ierr);
  

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "TaoSolverDestroy_LMVM"
static PetscErrorCode TaoSolverDestroy_LMVM(TaoSolver tao)
{

  TAO_LMVM *lmP = (TAO_LMVM *)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (tao->setupcalled) {
    ierr = VecDestroy(&lmP->Xold); CHKERRQ(ierr);
    ierr = VecDestroy(&lmP->Gold); CHKERRQ(ierr);
    ierr = VecDestroy(&lmP->D); CHKERRQ(ierr);
    ierr = MatDestroy(&lmP->M); CHKERRQ(ierr);
  }
  ierr = PetscFree(tao->data); CHKERRQ(ierr);
  tao->data = PETSC_NULL;

  PetscFunctionReturn(0); 
}

/*------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "TaoSolverSetFromOptions_LMVM"
static PetscErrorCode TaoSolverSetFromOptions_LMVM(TaoSolver tao)
{

  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("Limited-memory variable-metric method for unconstrained optimization"); CHKERRQ(ierr);
  ierr = TaoLineSearchSetFromOptions(tao->linesearch); CHKERRQ(ierr);
  ierr = PetscOptionsTail(); CHKERRQ(ierr);
  PetscFunctionReturn(0);

  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "TaoSolverView_LMVM"
static PetscErrorCode TaoSolverView_LMVM(TaoSolver tao, PetscViewer viewer)
{

    TAO_LMVM *lm = (TAO_LMVM *)tao->data;
    PetscBool isascii;
    PetscErrorCode ierr;


    PetscFunctionBegin;
    ierr = PetscTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii); CHKERRQ(ierr);
    if (isascii) {

        ierr = PetscViewerASCIIPushTab(viewer); CHKERRQ(ierr);
	ierr = PetscViewerASCIIPrintf(viewer, "BFGS steps: %d\n", lm->bfgs); CHKERRQ(ierr);
	ierr = PetscViewerASCIIPrintf(viewer, "Scaled gradient steps: %d\n", lm->sgrad); CHKERRQ(ierr);
	ierr = PetscViewerASCIIPrintf(viewer, "Gradient steps: %d\n", lm->grad); CHKERRQ(ierr);
        ierr = PetscViewerASCIIPopTab(viewer); CHKERRQ(ierr);
    } else {
      SETERRQ1(((PetscObject)tao)->comm,PETSC_ERR_SUP,"Viewer type %s not supported for TAO LMVM",((PetscObject)viewer)->type_name);
    }
    PetscFunctionReturn(0);
}

/* ---------------------------------------------------------- */

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "TaoSolverCreate_LMVM"
PetscErrorCode TaoSolverCreate_LMVM(TaoSolver tao)
{
    
  TAO_LMVM *lmP;
  const char *morethuente_type = TAOLINESEARCH_MT;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  tao->ops->setup = TaoSolverSetUp_LMVM;
  tao->ops->solve = TaoSolverSolve_LMVM;
  tao->ops->view = TaoSolverView_LMVM;
  tao->ops->setfromoptions = TaoSolverSetFromOptions_LMVM;
  tao->ops->destroy = TaoSolverDestroy_LMVM;

  ierr = PetscNewLog(tao,TAO_LMVM, &lmP); CHKERRQ(ierr);
  lmP->D = 0;
  lmP->M = 0;
  lmP->Xold = 0;
  lmP->Gold = 0;

  tao->data = (void*)lmP;
  tao->max_its = 2000;
  tao->max_funcs = 4000;
  tao->fatol = 1e-4;
  tao->frtol = 1e-4;

  ierr = TaoLineSearchCreate(((PetscObject)tao)->comm,&tao->linesearch); CHKERRQ(ierr);
  ierr = TaoLineSearchSetType(tao->linesearch,morethuente_type); CHKERRQ(ierr);
  ierr = TaoLineSearchUseTaoSolverRoutines(tao->linesearch,tao); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END
