#include <../src/tao/complementarity/impls/ssls/ssls.h>

PetscErrorCode TaoSetUp_SSILS(Tao tao)
{
  TAO_SSLS       *ssls = (TAO_SSLS *)tao->data;

  PetscFunctionBegin;
  CHKERRQ(VecDuplicate(tao->solution,&tao->gradient));
  CHKERRQ(VecDuplicate(tao->solution,&tao->stepdirection));
  CHKERRQ(VecDuplicate(tao->solution,&ssls->ff));
  CHKERRQ(VecDuplicate(tao->solution,&ssls->dpsi));
  CHKERRQ(VecDuplicate(tao->solution,&ssls->da));
  CHKERRQ(VecDuplicate(tao->solution,&ssls->db));
  CHKERRQ(VecDuplicate(tao->solution,&ssls->t1));
  CHKERRQ(VecDuplicate(tao->solution,&ssls->t2));
  PetscFunctionReturn(0);
}

PetscErrorCode TaoDestroy_SSILS(Tao tao)
{
  TAO_SSLS       *ssls = (TAO_SSLS *)tao->data;

  PetscFunctionBegin;
  CHKERRQ(VecDestroy(&ssls->ff));
  CHKERRQ(VecDestroy(&ssls->dpsi));
  CHKERRQ(VecDestroy(&ssls->da));
  CHKERRQ(VecDestroy(&ssls->db));
  CHKERRQ(VecDestroy(&ssls->t1));
  CHKERRQ(VecDestroy(&ssls->t2));
  CHKERRQ(PetscFree(tao->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSolve_SSILS(Tao tao)
{
  TAO_SSLS                     *ssls = (TAO_SSLS *)tao->data;
  PetscReal                    psi, ndpsi, normd, innerd, t=0;
  PetscReal                    delta, rho;
  TaoLineSearchConvergedReason ls_reason;

  PetscFunctionBegin;
  /* Assume that Setup has been called!
     Set the structure for the Jacobian and create a linear solver. */
  delta = ssls->delta;
  rho = ssls->rho;

  CHKERRQ(TaoComputeVariableBounds(tao));
  CHKERRQ(VecMedian(tao->XL,tao->solution,tao->XU,tao->solution));
  CHKERRQ(TaoLineSearchSetObjectiveAndGradientRoutine(tao->linesearch,Tao_SSLS_FunctionGradient,tao));
  CHKERRQ(TaoLineSearchSetObjectiveRoutine(tao->linesearch,Tao_SSLS_Function,tao));

  /* Calculate the function value and fischer function value at the
     current iterate */
  CHKERRQ(TaoLineSearchComputeObjectiveAndGradient(tao->linesearch,tao->solution,&psi,ssls->dpsi));
  CHKERRQ(VecNorm(ssls->dpsi,NORM_2,&ndpsi));

  tao->reason = TAO_CONTINUE_ITERATING;
  while (PETSC_TRUE) {
    CHKERRQ(PetscInfo(tao, "iter: %D, merit: %g, ndpsi: %g\n",tao->niter, (double)ssls->merit, (double)ndpsi));
    /* Check the termination criteria */
    CHKERRQ(TaoLogConvergenceHistory(tao,ssls->merit,ndpsi,0.0,tao->ksp_its));
    CHKERRQ(TaoMonitor(tao,tao->niter,ssls->merit,ndpsi,0.0,t));
    CHKERRQ((*tao->ops->convergencetest)(tao,tao->cnvP));
    if (tao->reason!=TAO_CONTINUE_ITERATING) break;

    /* Call general purpose update function */
    if (tao->ops->update) {
      CHKERRQ((*tao->ops->update)(tao, tao->niter, tao->user_update));
    }
    tao->niter++;

    /* Calculate direction.  (Really negative of newton direction.  Therefore,
       rest of the code uses -d.) */
    CHKERRQ(KSPSetOperators(tao->ksp,tao->jacobian,tao->jacobian_pre));
    CHKERRQ(KSPSolve(tao->ksp,ssls->ff,tao->stepdirection));
    CHKERRQ(KSPGetIterationNumber(tao->ksp,&tao->ksp_its));
    tao->ksp_tot_its+=tao->ksp_its;
    CHKERRQ(VecNorm(tao->stepdirection,NORM_2,&normd));
    CHKERRQ(VecDot(tao->stepdirection,ssls->dpsi,&innerd));

    /* Make sure that we have a descent direction */
    if (innerd <= delta*PetscPowReal(normd, rho)) {
      CHKERRQ(PetscInfo(tao, "newton direction not descent\n"));
      CHKERRQ(VecCopy(ssls->dpsi,tao->stepdirection));
      CHKERRQ(VecDot(tao->stepdirection,ssls->dpsi,&innerd));
    }

    CHKERRQ(VecScale(tao->stepdirection, -1.0));
    innerd = -innerd;

    CHKERRQ(TaoLineSearchSetInitialStepLength(tao->linesearch,1.0));
    CHKERRQ(TaoLineSearchApply(tao->linesearch,tao->solution,&psi,ssls->dpsi,tao->stepdirection,&t,&ls_reason));
    CHKERRQ(VecNorm(ssls->dpsi,NORM_2,&ndpsi));
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------- */
/*MC
   TAOSSILS - semi-smooth infeasible linesearch algorithm for solving
       complementarity constraints

   Options Database Keys:
+ -tao_ssls_delta - descent test fraction
- -tao_ssls_rho - descent test power

   Level: beginner
M*/
PETSC_EXTERN PetscErrorCode TaoCreate_SSILS(Tao tao)
{
  TAO_SSLS       *ssls;
  const char     *armijo_type = TAOLINESEARCHARMIJO;

  PetscFunctionBegin;
  CHKERRQ(PetscNewLog(tao,&ssls));
  tao->data = (void*)ssls;
  tao->ops->solve=TaoSolve_SSILS;
  tao->ops->setup=TaoSetUp_SSILS;
  tao->ops->view=TaoView_SSLS;
  tao->ops->setfromoptions=TaoSetFromOptions_SSLS;
  tao->ops->destroy = TaoDestroy_SSILS;

  ssls->delta = 1e-10;
  ssls->rho = 2.1;

  CHKERRQ(TaoLineSearchCreate(((PetscObject)tao)->comm,&tao->linesearch));
  CHKERRQ(PetscObjectIncrementTabLevel((PetscObject)tao->linesearch, (PetscObject)tao, 1));
  CHKERRQ(TaoLineSearchSetType(tao->linesearch,armijo_type));
  CHKERRQ(TaoLineSearchSetOptionsPrefix(tao->linesearch,tao->hdr.prefix));
  CHKERRQ(TaoLineSearchSetFromOptions(tao->linesearch));
  /* Note: linesearch objective and objectivegradient routines are set in solve routine */
  CHKERRQ(KSPCreate(((PetscObject)tao)->comm,&tao->ksp));
  CHKERRQ(PetscObjectIncrementTabLevel((PetscObject)tao->ksp, (PetscObject)tao, 1));
  CHKERRQ(KSPSetOptionsPrefix(tao->ksp,tao->hdr.prefix));

  /* Override default settings (unless already changed) */
  if (!tao->max_it_changed) tao->max_it = 2000;
  if (!tao->max_funcs_changed) tao->max_funcs = 4000;
  if (!tao->gttol_changed) tao->gttol = 0;
  if (!tao->grtol_changed) tao->grtol = 0;
#if defined(PETSC_USE_REAL_SINGLE)
  if (!tao->gatol_changed) tao->gatol = 1.0e-6;
  if (!tao->fmin_changed)  tao->fmin = 1.0e-4;
#else
  if (!tao->gatol_changed) tao->gatol = 1.0e-16;
  if (!tao->fmin_changed)  tao->fmin = 1.0e-8;
#endif
  PetscFunctionReturn(0);
}
