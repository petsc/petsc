/*$Id$*/
#include "taolinesearch.h"
#include "src/matrix/lmvmmat.h"
#include "blmvm.h"

/*------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "TaoSolverSolve_BLMVM"
static PetscErrorCode TaoSolverSolve_BLMVM(TaoSolver tao)
{
  PetscErrorCode ierr;
  TAO_BLMVM *blmP = (TAO_BLMVM *)tao->data;

  TaoSolverTerminationReason reason = TAO_CONTINUE_ITERATING;
  TaoLineSearchTerminationReason ls_status = TAOLINESEARCH_CONTINUE_ITERATING;

  Vec Xold,Gold;
  PetscReal f, fold, gdx, gnorm;
  PetscReal stepsize = 1.0,delta;

  PetscInt iter = 0;
  

  /* TODO Replace tao->gradient with local gradient, replace GP with tao->gradient */
  PetscFunctionBegin;
  ierr = VecDuplicate(tao->solution,&Xold); CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution,&Gold); CHKERRQ(ierr);
  
  // Project initial point onto bounds
  ierr = VecMedian(tao->XL,tao->solution,tao->XU,tao->solution); CHKERRQ(ierr);

  // Check convergence criteria
  ierr = TaoSolverComputeObjectiveAndGradient(tao, tao->solution,&f,tao->gradient); CHKERRQ(ierr);
  ierr = VecBoundGradientProjection(tao->gradient,tao->solution, tao->XL,tao->XU,blmP->GP); CHKERRQ(ierr);


  ierr = VecNorm(blmP->GP,NORM_2,&gnorm); CHKERRQ(ierr);
  if (TaoInfOrNaN(f) || TaoInfOrNaN(gnorm)) {
    SETERRQ(PETSC_COMM_SELF,1, "User provided compute function generated Inf pr NaN");
  }

  ierr = TaoSolverMonitor(tao, iter, f, gnorm, 0.0, stepsize, &reason); CHKERRQ(ierr);
  if (reason != TAO_CONTINUE_ITERATING) {
    PetscFunctionReturn(0);
  }

  // Set initial scaling for the function
  if (f != 0.0) {
    delta = 2.0*PetscAbsScalar(f) / (gnorm*gnorm);
  }
  else {
    delta = 2.0 / (gnorm*gnorm);
  }
  ierr = MatLMVMSetDelta(blmP->M,delta); CHKERRQ(ierr);

  // Set counter for gradient/reset steps
  blmP->grad = 0;
  blmP->reset = 0;

  // Have not converged; continue with Newton method
  while (reason == TAO_CONTINUE_ITERATING) {
    
    // Compute direction
    ierr = MatLMVMUpdate(blmP->M, tao->solution, blmP->GP); CHKERRQ(ierr);
    ierr = MatLMVMSolve(blmP->M, tao->gradient, tao->stepdirection); CHKERRQ(ierr);
    ierr = VecBoundGradientProjection(tao->stepdirection,tao->solution,tao->XL,tao->XU,blmP->GP); CHKERRQ(ierr);

    // Check for success (descent direction)
    ierr = VecDot(tao->gradient, blmP->GP, &gdx); CHKERRQ(ierr);
    if (gdx <= 0) {
      // Step is not descent or solve was not successful
      // Use steepest descent direction (scaled)
      ++blmP->grad;

      if (f != 0.0) {
	delta = 2.0*PetscAbsScalar(f) / (gnorm*gnorm);
      }
      else {
	delta = 2.0 / (gnorm*gnorm);
      }
      ierr = MatLMVMSetDelta(blmP->M,delta); CHKERRQ(ierr);
      ierr = MatLMVMReset(blmP->M); CHKERRQ(ierr);
      ierr = MatLMVMUpdate(blmP->M, tao->solution, tao->gradient); CHKERRQ(ierr);
      ierr = MatLMVMSolve(blmP->M,tao->gradient, tao->stepdirection); CHKERRQ(ierr);
    } 
    ierr = VecScale(tao->stepdirection,-1.0); CHKERRQ(ierr);

    // Perform the linesearch
    fold = f;
    ierr = VecCopy(tao->solution, Xold); CHKERRQ(ierr);
    ierr = VecCopy(tao->gradient, Gold); CHKERRQ(ierr);
    ierr = TaoLineSearchSetInitialStepLength(tao->linesearch,1.0); CHKERRQ(ierr);
    ierr = TaoLineSearchApply(tao->linesearch, tao->solution, &f, tao->gradient, tao->stepdirection, &stepsize, &ls_status); CHKERRQ(ierr);

    if (ls_status<0) {
      // Linesearch failed
      // Reset factors and use scaled (projected) gradient step
      ++blmP->reset;

      f = fold;
      ierr = VecCopy(Xold, tao->solution); CHKERRQ(ierr);
      ierr = VecCopy(Gold, tao->gradient); CHKERRQ(ierr);

      if (f != 0.0) {
	delta = 2.0* PetscAbsScalar(f) / (gnorm*gnorm);
      }
      else {
	delta = 2.0/ (gnorm*gnorm);
      }
      ierr = MatLMVMSetDelta(blmP->M,delta); CHKERRQ(ierr);
      ierr = MatLMVMReset(blmP->M); CHKERRQ(ierr);
      ierr = MatLMVMUpdate(blmP->M, tao->solution, tao->gradient); CHKERRQ(ierr);
      ierr = MatLMVMSolve(blmP->M, tao->gradient, tao->stepdirection); CHKERRQ(ierr);
      ierr = VecScale(tao->stepdirection, -1.0); CHKERRQ(ierr);

      // This may be incorrect; linesearch has values fo stepmax and stepmin
      // that should be reset.
      ierr = TaoLineSearchSetInitialStepLength(tao->linesearch,1.0);
      ierr = TaoLineSearchApply(tao->linesearch,tao->solution,&f, tao->gradient, tao->stepdirection,  &stepsize, &ls_status); CHKERRQ(ierr);

      if ((int) ls_status < 0) {
        // Linesearch failed
        // Probably stop here
      }
    }

    // Check for termination
    ierr = VecBoundGradientProjection(tao->gradient, tao->solution, tao->XL, tao->XU, blmP->GP); CHKERRQ(ierr);
    ierr = VecNorm(blmP->GP, NORM_2, &gnorm); CHKERRQ(ierr);


    if (TaoInfOrNaN(f) || TaoInfOrNaN(gnorm)) {
      SETERRQ(PETSC_COMM_SELF,1, "User provided compute function generated Not-a-Number");
    }
    iter++;
    ierr = TaoSolverMonitor(tao, iter, f, gnorm, 0.0, stepsize, &reason); CHKERRQ(ierr);
  }
  ierr = VecDestroy(Xold); CHKERRQ(ierr);
  ierr = VecDestroy(Gold); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TaoSolverSetup_BLMVM"
static PetscErrorCode TaoSolverSetup_BLMVM(TaoSolver tao)
{
  TAO_BLMVM *blmP = (TAO_BLMVM *)tao->data;
  PetscInt n,N;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Existence of tao->solution checked in TaoSolverSetup() */
  if (!tao->gradient) {
      ierr = VecDuplicate(tao->solution, &tao->gradient);
      CHKERRQ(ierr);
  }
  if (!tao->stepdirection) {
      ierr = VecDuplicate(tao->solution, &tao->stepdirection);
      CHKERRQ(ierr);
  }
  if (!blmP->GP) {
      ierr = VecDuplicate(tao->solution,&blmP->GP); CHKERRQ(ierr);
  }
  if (!tao->XL) {
      ierr = VecDuplicate(tao->solution,&tao->XL); CHKERRQ(ierr);
      ierr = VecSet(tao->XL,TAO_NINFINITY); CHKERRQ(ierr);
  }
  if (!tao->XU) {
      ierr = VecDuplicate(tao->solution,&tao->XU); CHKERRQ(ierr);
      ierr = VecSet(tao->XU,TAO_INFINITY); CHKERRQ(ierr);
  }
  ierr = TaoLineSearchSetVariableBounds(tao->linesearch,tao->XL,tao->XU); CHKERRQ(ierr);
  // Create matrix for the limited memory approximation
  ierr = VecGetLocalSize(tao->solution,&n); CHKERRQ(ierr);
  ierr = VecGetSize(tao->solution,&N); CHKERRQ(ierr);
  ierr = MatCreateLMVM(((PetscObject)tao)->comm,n,N,&blmP->M); CHKERRQ(ierr);
  ierr = MatLMVMAllocateVectors(blmP->M,tao->solution); CHKERRQ(ierr);

   PetscFunctionReturn(0);
}

/* ---------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "TaoSolverDestroy_BLMVM"
static PetscErrorCode TaoSolverDestroy_BLMVM(TaoSolver tao)
{
  TAO_BLMVM *blmP = (TAO_BLMVM *)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (blmP->M) {
    ierr = MatDestroy(blmP->M); CHKERRQ(ierr);
  }
  if (tao->data) {
    ierr = PetscFree(tao->data); CHKERRQ(ierr);
  }
  if (tao->linesearch) {
    ierr = TaoLineSearchDestroy(tao->linesearch); CHKERRQ(ierr);
  }
  
  tao->linesearch = PETSC_NULL;
  tao->data = PETSC_NULL;


  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "TaoSolverSetFromOptions_BLMVM"
static PetscErrorCode TaoSolverSetFromOptions_BLMVM(TaoSolver tao)
{

  PetscErrorCode ierr;

  PetscFunctionBegin;
  //  info = TaoOptionsHead("Limited-memory variable-metric method for bound constrained optimization"); CHKERRQ(info);
  ierr = TaoLineSearchSetFromOptions(tao->linesearch);CHKERRQ(ierr);
  //info = TaoOptionsTail();CHKERRQ(info);
  PetscFunctionReturn(0);

    return 0;
}


/*------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "TaoSolverView_BLMVM"
static int TaoSolverView_BLMVM(TaoSolver tao, PetscViewer viewer)
{

    
    PetscFunctionBegin;
    PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TaoSolverComputeDual_BLMVM" 
static PetscErrorCode TaoSolverComputeDual_BLMVM(TaoSolver tao, Vec DXL, Vec DXU)
{
  TAO_BLMVM *blm = (TAO_BLMVM *) tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAOSOLVER_CLASSID,1);
  PetscValidHeaderSpecific(DXL,VEC_CLASSID,2);
  PetscValidHeaderSpecific(DXU,VEC_CLASSID,3);

  if (!blm->GP || !tao->gradient) {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Dual variables don't exist yet or no longer exist.\n");
  }
  
  ierr = VecCopy(blm->GP,DXL); CHKERRQ(ierr);
  ierr = VecAXPY(DXL,-1.0,tao->gradient); CHKERRQ(ierr);
  ierr = VecSet(DXU,0.0); CHKERRQ(ierr);
  ierr = VecPointwiseMax(DXL,DXL,DXU); CHKERRQ(ierr);

  ierr = VecCopy(tao->gradient,DXU); CHKERRQ(ierr);
  ierr = VecAXPY(DXU,-1.0,blm->GP); CHKERRQ(ierr);
  ierr = VecAXPY(DXU,1.0,DXL); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------- */
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "TaoSolverCreate_BLMVM"
PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverCreate_BLMVM(TaoSolver tao)
{
  TAO_BLMVM *blmP;
  const char *morethuente_type = TAOLINESEARCH_MT;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  tao->ops->setup = TaoSolverSetup_BLMVM;
  tao->ops->solve = TaoSolverSolve_BLMVM;
  tao->ops->view = TaoSolverView_BLMVM;
  tao->ops->setfromoptions = TaoSolverSetFromOptions_BLMVM;
  tao->ops->destroy = TaoSolverDestroy_BLMVM;
  tao->ops->computedual = TaoSolverComputeDual_BLMVM;

  ierr = PetscNewLog(tao, TAO_BLMVM, &blmP); CHKERRQ(ierr);
  tao->data = (void*)blmP;
  tao->max_its = 2000;
  tao->max_funcs = 4000;
  tao->fatol = 1e-4;
  tao->frtol = 1e-4;

  ierr = TaoLineSearchCreate(((PetscObject)tao)->comm, &tao->linesearch); CHKERRQ(ierr);
  ierr = TaoLineSearchSetType(tao->linesearch, morethuente_type); CHKERRQ(ierr);
  ierr = TaoLineSearchUseTaoSolverRoutines(tao->linesearch,tao); CHKERRQ(ierr);
  PetscFunctionReturn(0);

}
EXTERN_C_END

