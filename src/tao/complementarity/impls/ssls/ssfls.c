#include <../src/tao/complementarity/impls/ssls/ssls.h>

PetscErrorCode TaoSetUp_SSFLS(Tao tao)
{
  TAO_SSLS       *ssls = (TAO_SSLS *)tao->data;

  PetscFunctionBegin;
  PetscCall(VecDuplicate(tao->solution,&tao->gradient));
  PetscCall(VecDuplicate(tao->solution,&tao->stepdirection));
  PetscCall(VecDuplicate(tao->solution,&ssls->w));
  PetscCall(VecDuplicate(tao->solution,&ssls->ff));
  PetscCall(VecDuplicate(tao->solution,&ssls->dpsi));
  PetscCall(VecDuplicate(tao->solution,&ssls->da));
  PetscCall(VecDuplicate(tao->solution,&ssls->db));
  PetscCall(VecDuplicate(tao->solution,&ssls->t1));
  PetscCall(VecDuplicate(tao->solution,&ssls->t2));
  if (!tao->XL) {
    PetscCall(VecDuplicate(tao->solution,&tao->XL));
  }
  if (!tao->XU) {
    PetscCall(VecDuplicate(tao->solution,&tao->XU));
  }
  PetscCall(TaoLineSearchSetVariableBounds(tao->linesearch,tao->XL,tao->XU));
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSolve_SSFLS(Tao tao)
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

  PetscCall(TaoComputeVariableBounds(tao));
  /* Project solution inside bounds */
  PetscCall(VecMedian(tao->XL,tao->solution,tao->XU,tao->solution));
  PetscCall(TaoLineSearchSetObjectiveAndGradientRoutine(tao->linesearch,Tao_SSLS_FunctionGradient,tao));
  PetscCall(TaoLineSearchSetObjectiveRoutine(tao->linesearch,Tao_SSLS_Function,tao));

  /* Calculate the function value and fischer function value at the
     current iterate */
  PetscCall(TaoLineSearchComputeObjectiveAndGradient(tao->linesearch,tao->solution,&psi,ssls->dpsi));
  PetscCall(VecNorm(ssls->dpsi,NORM_2,&ndpsi));

  tao->reason = TAO_CONTINUE_ITERATING;
  while (PETSC_TRUE) {
    PetscCall(PetscInfo(tao, "iter: %D, merit: %g, ndpsi: %g\n",tao->niter, (double)ssls->merit, (double)ndpsi));
    /* Check the termination criteria */
    PetscCall(TaoLogConvergenceHistory(tao,ssls->merit,ndpsi,0.0,tao->ksp_its));
    PetscCall(TaoMonitor(tao,tao->niter,ssls->merit,ndpsi,0.0,t));
    PetscCall((*tao->ops->convergencetest)(tao,tao->cnvP));
    if (tao->reason!=TAO_CONTINUE_ITERATING) break;

    /* Call general purpose update function */
    if (tao->ops->update) {
      PetscCall((*tao->ops->update)(tao, tao->niter, tao->user_update));
    }
    tao->niter++;

    /* Calculate direction.  (Really negative of newton direction.  Therefore,
       rest of the code uses -d.) */
    PetscCall(KSPSetOperators(tao->ksp,tao->jacobian,tao->jacobian_pre));
    PetscCall(KSPSolve(tao->ksp,ssls->ff,tao->stepdirection));
    PetscCall(KSPGetIterationNumber(tao->ksp,&tao->ksp_its));
    tao->ksp_tot_its+=tao->ksp_its;

    PetscCall(VecCopy(tao->stepdirection,ssls->w));
    PetscCall(VecScale(ssls->w,-1.0));
    PetscCall(VecBoundGradientProjection(ssls->w,tao->solution,tao->XL,tao->XU,ssls->w));

    PetscCall(VecNorm(ssls->w,NORM_2,&normd));
    PetscCall(VecDot(ssls->w,ssls->dpsi,&innerd));

    /* Make sure that we have a descent direction */
    if (innerd >= -delta*PetscPowReal(normd, rho)) {
      PetscCall(PetscInfo(tao, "newton direction not descent\n"));
      PetscCall(VecCopy(ssls->dpsi,tao->stepdirection));
      PetscCall(VecDot(ssls->w,ssls->dpsi,&innerd));
    }

    PetscCall(VecScale(tao->stepdirection, -1.0));
    innerd = -innerd;

    PetscCall(TaoLineSearchSetInitialStepLength(tao->linesearch,1.0));
    PetscCall(TaoLineSearchApply(tao->linesearch,tao->solution,&psi,ssls->dpsi,tao->stepdirection,&t,&ls_reason));
    PetscCall(VecNorm(ssls->dpsi,NORM_2,&ndpsi));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode TaoDestroy_SSFLS(Tao tao)
{
  TAO_SSLS       *ssls = (TAO_SSLS *)tao->data;

  PetscFunctionBegin;
  PetscCall(VecDestroy(&ssls->ff));
  PetscCall(VecDestroy(&ssls->w));
  PetscCall(VecDestroy(&ssls->dpsi));
  PetscCall(VecDestroy(&ssls->da));
  PetscCall(VecDestroy(&ssls->db));
  PetscCall(VecDestroy(&ssls->t1));
  PetscCall(VecDestroy(&ssls->t2));
  PetscCall(PetscFree(tao->data));
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------- */
/*MC
   TAOSSFLS - Semi-smooth feasible linesearch algorithm for solving
       complementarity constraints

   Options Database Keys:
+ -tao_ssls_delta - descent test fraction
- -tao_ssls_rho - descent test power

   Level: beginner
M*/

PETSC_EXTERN PetscErrorCode TaoCreate_SSFLS(Tao tao)
{
  TAO_SSLS       *ssls;
  const char     *armijo_type = TAOLINESEARCHARMIJO;

  PetscFunctionBegin;
  PetscCall(PetscNewLog(tao,&ssls));
  tao->data = (void*)ssls;
  tao->ops->solve=TaoSolve_SSFLS;
  tao->ops->setup=TaoSetUp_SSFLS;
  tao->ops->view=TaoView_SSLS;
  tao->ops->setfromoptions = TaoSetFromOptions_SSLS;
  tao->ops->destroy = TaoDestroy_SSFLS;

  ssls->delta = 1e-10;
  ssls->rho = 2.1;

  PetscCall(TaoLineSearchCreate(((PetscObject)tao)->comm,&tao->linesearch));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)tao->linesearch, (PetscObject)tao, 1));
  PetscCall(TaoLineSearchSetType(tao->linesearch,armijo_type));
  PetscCall(TaoLineSearchSetOptionsPrefix(tao->linesearch,tao->hdr.prefix));
  PetscCall(TaoLineSearchSetFromOptions(tao->linesearch));
  /* Linesearch objective and objectivegradient routines are  set in solve routine */
  PetscCall(KSPCreate(((PetscObject)tao)->comm,&tao->ksp));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)tao->ksp, (PetscObject)tao, 1));
  PetscCall(KSPSetOptionsPrefix(tao->ksp,tao->hdr.prefix));

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
