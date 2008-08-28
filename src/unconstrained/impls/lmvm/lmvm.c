#include "taolinesearch.h"
#include "src/matrix/lmvmmat.h"
#include "lmvm.h"

#define LMM_BFGS                0
#define LMM_SCALED_GRADIENT     1
#define LMM_GRADIENT            2

#undef __FUNCT__  
#define __FUNCT__ "TaoSolverSolve_LMVM"
static PetscErrorCode TaoSolverSolve_LMVM(TaoSolver tao)
{

  TAO_LMVM *lmP = (TAO_LMVM *)tao->data;
  
  
  TaoSolverConvergedReason reason;
//  PetscTruth success;

  PetscScalar f, fold, gdx, gnorm;
  PetscScalar step = 1.0;

  PetscScalar delta;

  PetscErrorCode ierr;
  PetscInt stepType;
//  PetscInt iter = 0;
  TaoLineSearchTerminationReason ls_status = TAOLINESEARCH_CONTINUE_ITERATING;
//  PetscInt bfgsUpdates = 0;

  PetscFunctionBegin;

  // Check convergence criteria
  ierr = TaoSolverComputeObjectiveAndGradient(tao, tao->solution, &f, tao->gradient); CHKERRQ(ierr);
  ierr = VecNorm(tao->gradient,NORM_2,&gnorm); CHKERRQ(ierr);
  if (TaoInfOrNaN(f) || TaoInfOrNaN(gnorm)) {
    SETERRQ(1, "User provided compute function generated Inf or NaN");
  }

//  info = TaoMonitor(tao, iter, f, gnorm, 0.0, step, &reason); CHKERRQ(info);
//  if (reason != TAO_CONTINUE_ITERATING) {
//    TaoFunctionReturn(0);
//  }

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
    ierr = MatLMVMUpdate(lmP->M,lmP->X,lmP->G); CHKERRQ(ierr);
    ierr = MatSolve(lmP->M, lmP->G, lmP->D); CHKERRQ(ierr);
    ++lmP->bfgs;

    // Check for success (descent direction)
    ierr = VecDot(lmP->D, lmP->G, &gdx); CHKERRQ(ierr);
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
      ierr = MatLMVMUpdate(lmP->M, lmP->X, lmP->G); CHKERRQ(ierr);
      ierr = MatSolve(lmP->M,lmP->G, lmP->D); CHKERRQ(ierr);

      // On a reset, the direction cannot be not a number; it is a 
      // scaled gradient step.  No need to check for this condition.
      // info = D->Norm2(&dnorm); CHKERRQ(info);
      // if (TaoInfOrNaN(dnorm)) {
      //   SETERRQ(1, "Direction generated Not-a-Number");
      // }

      lmP->bfgs = 1;
      ++lmP->sgrad;
      stepType = LMM_SCALED_GRADIENT;
    }
    else {
      if (1 == lmP->bfgs) {
        // The first BFGS direction is always the scaled gradient
        ++lmP->sgrad;
        stepType = LMM_SCALED_GRADIENT;
      }
      else {
        ++lmP->bfgs;
        stepType = LMM_BFGS;
      }
    }
    ierr = VecScale(lmP->D, -1.0); CHKERRQ(ierr);
    
    // Perform the linesearch
    fold = f;
    ierr = VecCopy(lmP->X, lmP->Xold); CHKERRQ(ierr);
    ierr = VecCopy(lmP->G, lmP->Gold); CHKERRQ(ierr);

    step = 1.0;
    ierr = TaoLineSearchApply(tao->linesearch, lmP->X, &f, lmP->G, lmP->D, &step,&ls_status); CHKERRQ(ierr);
    
//    info = TaoLineSearchApply(tao, X, G, D, W, &f, &f_full, &step, &status); CHKERRQ(info);

    while (((int)ls_status < 0) && (stepType != LMM_GRADIENT)) {
      // Linesearch failed
      // Reset factors and use scaled gradient step
      f = fold;
      ierr = VecCopy(lmP->Xold, lmP->X); CHKERRQ(ierr);
      ierr = VecCopy(lmP->Gold, lmP->G); CHKERRQ(ierr);
        
      switch(stepType) {
      case LMM_BFGS:
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
	ierr = MatLMVMUpdate(lmP->M, lmP->X, lmP->G); CHKERRQ(ierr);
	ierr = MatSolve(lmP->M, lmP->G, lmP->D); CHKERRQ(ierr);

        // On a reset, the direction cannot be not a number; it is a 
        // scaled gradient step.  No need to check for this condition.
        // info = D->Norm2(&dnorm); CHKERRQ(info);
        // if (TaoInfOrNaN(dnorm)) {
        //   SETERRQ(1, "Direction generated Not-a-Number");
        // }
  
	lmP->bfgs = 1;
	++lmP->sgrad;
	stepType = LMM_SCALED_GRADIENT;
	break;

      case LMM_SCALED_GRADIENT:
        // The scaled gradient step did not produce a new iterate;
	// attempt to use the gradient direction.
	// Need to make sure we are not using a different diagonal scaling
	ierr = MatLMVMSetDelta(lmP->M, 1.0); CHKERRQ(ierr);
	ierr = MatLMVMReset(lmP->M); CHKERRQ(ierr);
	ierr = MatLMVMUpdate(lmP->M, lmP->X, lmP->G); CHKERRQ(ierr);
	ierr = MatSolve(lmP->M, lmP->G, lmP->D); CHKERRQ(ierr);

        lmP->bfgs = 1;
        ++lmP->grad;
        stepType = LMM_GRADIENT;
        break;
      }
      ierr = VecScale(lmP->D, -1.0); CHKERRQ(ierr);
        
      // Perform the linesearch
      ierr = TaoLineSearchApply(tao->linesearch, lmP->X, &f, lmP->G, lmP->D, &step, &ls_status); CHKERRQ(ierr);


    }

    if ((int)ls_status < 0) {
      // Failed to find an improving point
      f = fold;
      ierr = VecCopy(lmP->Xold, lmP->X); CHKERRQ(ierr);
      ierr = VecCopy(lmP->Gold, lmP->G); CHKERRQ(ierr);
      step = 0.0;
    }

    // Check for termination
    ierr = VecNorm(lmP->G, NORM_2, &gnorm); CHKERRQ(ierr);
    if (TaoInfOrNaN(f) || TaoInfOrNaN(gnorm)) {
      SETERRQ(1, "User provided compute function generated Inf or NaN");
    }
    //    info = TaoMonitor(tao, ++iter, f, gnorm, 0.0, step, &reason); CHKERRQ(info);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TaoSetUp_LMVM"
static PetscErrorCode TaoSolverSetUp_LMVM(TaoSolver tao)
{
  TAO_LMVM *lmP = (TAO_LMVM *)tao->data;
  PetscInt n,N;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Existence of tao->solution checked in TaoSolverSetUp() */
  if (!tao->gradient) {ierr = VecDuplicate(tao->solution,&tao->gradient); CHKERRQ(ierr);  }
  if (!tao->stepdirection) {ierr = VecDuplicate(tao->solution,&tao->stepdirection); CHKERRQ(ierr);  }
  if (!lmP->W) {ierr = VecDuplicate(tao->solution,&lmP->W); CHKERRQ(ierr);  }
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
  ierr = VecDestroy(tao->gradient); CHKERRQ(ierr);
  tao->gradient=PETSC_NULL;
  ierr = VecDestroy(tao->stepdirection); CHKERRQ(ierr);
  tao->stepdirection=PETSC_NULL;

  ierr = VecDestroy(lmP->W); CHKERRQ(ierr);
  ierr = VecDestroy(lmP->Xold); CHKERRQ(ierr);
  ierr = VecDestroy(lmP->Gold); CHKERRQ(ierr);
  ierr = MatDestroy(lmP->M); CHKERRQ(ierr);
  ierr = PetscFree(tao->data); CHKERRQ(ierr);

  ierr = TaoLineSearchDestroy(tao->linesearch); CHKERRQ(ierr);
  tao->linesearch = PETSC_NULL;

  PetscFunctionReturn(0); 

}

/*------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "TaoSolverSetFromOptions_LMVM"
static PetscErrorCode TaoSolverSetFromOptions_LMVM(TaoSolver tao)
{
    /*
  int info;

  TaoFunctionBegin;
  info = TaoOptionsHead("Limited-memory variable-metric method for unconstrained optimization"); CHKERRQ(info);
  info = TaoLineSearchSetFromOptions(tao); CHKERRQ(info);
  info = TaoOptionsTail(); CHKERRQ(info);
  TaoFunctionReturn(0);
    */
    return 0;
}

/*------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "TaoSolverView_LMVM"
static PetscErrorCode TaoSolverView_LMVM(TaoSolver tao, PetscViewer viewer)
{
    /*
  TAO_LMVM *lm = (TAO_LMVM *)solver;
  int info;

  TaoFunctionBegin;
  info = TaoPrintInt(tao, "  Rejected matrix updates: %d\n", lm->M->GetRejects()); CHKERRQ(info);
  info = TaoPrintInt(tao, "  BFGS steps: %d\n", lm->bfgs); CHKERRQ(info);
  info = TaoPrintInt(tao, "  Scaled gradient steps: %d\n", lm->sgrad); CHKERRQ(info);
  info = TaoPrintInt(tao, "  Gradient steps: %d\n", lm->grad); CHKERRQ(info);
  info = TaoLineSearchView(tao); CHKERRQ(info);
  TaoFunctionReturn(0);
    */
    return 0;
}

/* ---------------------------------------------------------- */

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "TaoSolverCreate_LMVM"
PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverCreate_LMVM(TaoSolver tao)
{
    
  TAO_LMVM *lmP;
  char lstype[256] = TAOLINESEARCH_MT;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  tao->ops->setup = TaoSolverSetUp_LMVM;
  tao->ops->solve = TaoSolverSolve_LMVM;
  tao->ops->view = TaoSolverView_LMVM;
  tao->ops->setfromoptions = TaoSolverSetFromOptions_LMVM;
  tao->ops->destroy = TaoSolverDestroy_LMVM;

  ierr = PetscNewLog(tao,TAO_LMVM, &lmP); CHKERRQ(ierr);
  tao->data = (void*)lmP;
  tao->max_its = 2000;
  tao->max_funcs = 4000;
  tao->fatol = 1e-4;
  tao->frtol = 1e-4;
  ierr = TaoLineSearchCreate(((PetscObject)tao)->comm,&tao->linesearch); CHKERRQ(ierr);

  /* Need to set to "more-thuente" */
  ierr = TaoLineSearchSetType(tao->linesearch,lstype); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END

// Todd: do not delete; they are needed for the component version
// of the code.

/* ---------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "TaoLMVMGetX0"
int TaoLMVMGetX0(TaoSolver tao, Vec x0)
{
    /*
  TAO_LMVM *lm;
  int info;

  TaoFunctionBegin;
  info=TaoGetSolverContext(tao, "tao_lmvm", (void **)&lm); CHKERRQ(info);
  if (lm && lm->M) {
    info=lm->M->GetX0(x0); CHKERRQ(info);
  }
  TaoFunctionReturn(0);
  */
      return -1;
}

/* ---------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "TaoInitializeLMVMmatrix"
int TaoInitializeLMVMmatrix(TaoSolver tao, Vec HV)
{
    /*
  TAO_LMVM *lm;
  int info;
  
  TaoFunctionBegin;
  info = TaoGetSolverContext(tao, "tao_lmvm", (void **)&lm); CHKERRQ(info);
  if (lm && lm->M) {
    info = lm->M->InitialApproximation(HV); CHKERRQ(info);
  }
  TaoFunctionReturn(0);
    */
    return -1;
}

