#include <../src/tao/bound/impls/pgd/pgd.h>

static PetscErrorCode TaoSolve_PGD(Tao tao)
{
  TAO_PGD                      *pg = (TAO_PGD *)tao->data;
  PetscErrorCode               ierr;
  TaoConvergedReason           reason = TAO_CONTINUE_ITERATING;
  TaoLineSearchConvergedReason ls_status = TAOLINESEARCH_CONTINUE_ITERATING;

  PetscFunctionBegin;
  /*   Project the current point onto the feasible set */
  ierr = TaoComputeVariableBounds(tao);CHKERRQ(ierr);
  ierr = TaoLineSearchSetVariableBounds(tao->linesearch,tao->XL,tao->XU);CHKERRQ(ierr);
  
  /* Project the initial point onto the feasible region */
  ierr = VecMedian(tao->XL,tao->solution,tao->XU,tao->solution);CHKERRQ(ierr);
  
  /* Compute the objective function and gradient */
  ierr = TaoComputeObjectiveAndGradient(tao,tao->solution,&pg->f,pg->unprojected_gradient);CHKERRQ(ierr);
  ierr = VecNorm(pg->unprojected_gradient,NORM_2,&pg->gnorm);CHKERRQ(ierr);
  if (PetscIsInfOrNanReal(pg->f) || PetscIsInfOrNanReal(pg->gnorm)) SETERRQ(PETSC_COMM_SELF,1, "User provided compute function generated Inf or NaN");
  
  /* Project the gradient and calculate the norm */
  ierr = VecBoundGradientProjection(pg->unprojected_gradient,tao->solution,tao->XL,tao->XU,tao->gradient);CHKERRQ(ierr);
  ierr = VecNorm(tao->gradient,NORM_2,&pg->gnorm);CHKERRQ(ierr);
  
  /* Check convergence and give info to monitors */
  ierr = TaoMonitor(tao,tao->niter,pg->f,pg->gnorm,0.0,pg->stepsize,&reason);CHKERRQ(ierr);
  
  while (reason == TAO_CONTINUE_ITERATING) {
    /* Set step direction to steepest descent */
    ierr = VecCopy(pg->unprojected_gradient,tao->stepdirection);CHKERRQ(ierr);
    ierr = VecScale(tao->stepdirection,-1.0);CHKERRQ(ierr);
    
    /* Perform line search */
    ierr = TaoLineSearchSetInitialStepLength(tao->linesearch,1.0);CHKERRQ(ierr);
    ierr = TaoLineSearchApply(tao->linesearch, tao->solution, &pg->f, pg->unprojected_gradient, tao->stepdirection, &pg->stepsize, &ls_status);CHKERRQ(ierr);
    ierr = TaoAddLineSearchCounts(tao);CHKERRQ(ierr);
    if (ls_status != TAOLINESEARCH_SUCCESS && ls_status != TAOLINESEARCH_SUCCESS_USER) reason = TAO_DIVERGED_LS_FAILURE;

    /* Project the gradient and calculate the norm */
    ierr = VecBoundGradientProjection(pg->unprojected_gradient,tao->solution,tao->XL,tao->XU,tao->gradient);CHKERRQ(ierr);
    ierr = VecNorm(tao->gradient,NORM_2,&pg->gnorm);CHKERRQ(ierr);
    
    /* Check convergence and give info to monitors */
    tao->niter++;
    ierr = TaoMonitor(tao,tao->niter,pg->f,pg->gnorm,0.0,pg->stepsize,&reason);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------- */

static PetscErrorCode TaoSetup_PGD(Tao tao)
{
  TAO_PGD        *pg = (TAO_PGD *)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Allocate some arrays */
  ierr = VecDuplicate(tao->solution, &tao->gradient);CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution, &tao->stepdirection);CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution, &pg->unprojected_gradient);CHKERRQ(ierr);
  if (!tao->XL) {
    ierr = VecDuplicate(tao->solution, &tao->XL);CHKERRQ(ierr);
    ierr = VecSet(tao->XL, PETSC_NINFINITY);CHKERRQ(ierr);
  }
  if (!tao->XU) {
    ierr = VecDuplicate(tao->solution, &tao->XU);CHKERRQ(ierr);
    ierr = VecSet(tao->XU, PETSC_INFINITY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------- */

static PetscErrorCode TaoSetFromOptions_PGD(PetscOptionItems *PetscOptionsObject,Tao tao)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TaoLineSearchSetFromOptions(tao->linesearch);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode TaoDestroy_PGD(Tao tao)
{
  TAO_PGD        *pg = (TAO_PGD *)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroy(&pg->unprojected_gradient);CHKERRQ(ierr);
  ierr = PetscFree(tao->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/*MC
  TAOPGD - The PGD algorithm is a projected gradient descent method for 
  bound constrained optimization.

  Level: beginner
M*/
PETSC_EXTERN PetscErrorCode TaoCreate_PGD(Tao tao)
{
  PetscErrorCode               ierr;
  TAO_PGD                      *pg;
  const char                   *morethuente_type = TAOLINESEARCHMT;
  
  PetscFunctionBegin;
  tao->ops->setup          = TaoSetup_PGD;
  tao->ops->solve          = TaoSolve_PGD;
  tao->ops->setfromoptions = TaoSetFromOptions_PGD;
  tao->ops->destroy        = TaoDestroy_PGD;
  
  ierr = PetscNewLog(tao,&pg);CHKERRQ(ierr);
  tao->data = (void*)pg;
  
  ierr = TaoLineSearchCreate(((PetscObject)tao)->comm, &tao->linesearch);CHKERRQ(ierr);
  ierr = PetscObjectIncrementTabLevel((PetscObject)tao->linesearch, (PetscObject)tao, 1);CHKERRQ(ierr);
  ierr = TaoLineSearchSetType(tao->linesearch,morethuente_type);CHKERRQ(ierr);
  ierr = TaoLineSearchUseTaoRoutines(tao->linesearch,tao);CHKERRQ(ierr);
  ierr = TaoLineSearchSetOptionsPrefix(tao->linesearch,tao->hdr.prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
