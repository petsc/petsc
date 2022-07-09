#include <petsctaolinesearch.h>
#include <../src/tao/unconstrained/impls/lmvm/lmvm.h>

#define LMVM_STEP_BFGS     0
#define LMVM_STEP_GRAD     1

static PetscErrorCode TaoSolve_LMVM(Tao tao)
{
  TAO_LMVM                     *lmP = (TAO_LMVM *)tao->data;
  PetscReal                    f, fold, gdx, gnorm;
  PetscReal                    step = 1.0;
  PetscInt                     stepType = LMVM_STEP_GRAD, nupdates;
  TaoLineSearchConvergedReason ls_status = TAOLINESEARCH_CONTINUE_ITERATING;

  PetscFunctionBegin;

  if (tao->XL || tao->XU || tao->ops->computebounds) {
    PetscCall(PetscInfo(tao,"WARNING: Variable bounds have been set but will be ignored by lmvm algorithm\n"));
  }

  /*  Check convergence criteria */
  PetscCall(TaoComputeObjectiveAndGradient(tao, tao->solution, &f, tao->gradient));
  PetscCall(TaoGradientNorm(tao, tao->gradient,NORM_2,&gnorm));

  PetscCheck(!PetscIsInfOrNanReal(f) && !PetscIsInfOrNanReal(gnorm),PetscObjectComm((PetscObject)tao),PETSC_ERR_USER, "User provided compute function generated Inf or NaN");

  tao->reason = TAO_CONTINUE_ITERATING;
  PetscCall(TaoLogConvergenceHistory(tao,f,gnorm,0.0,tao->ksp_its));
  PetscCall(TaoMonitor(tao,tao->niter,f,gnorm,0.0,step));
  PetscCall((*tao->ops->convergencetest)(tao,tao->cnvP));
  if (tao->reason != TAO_CONTINUE_ITERATING) PetscFunctionReturn(0);

  /*  Set counter for gradient/reset steps */
  if (!lmP->recycle) {
    lmP->bfgs = 0;
    lmP->grad = 0;
    PetscCall(MatLMVMReset(lmP->M, PETSC_FALSE));
  }

  /*  Have not converged; continue with Newton method */
  while (tao->reason == TAO_CONTINUE_ITERATING) {
    /* Call general purpose update function */
    if (tao->ops->update) PetscCall((*tao->ops->update)(tao, tao->niter, tao->user_update));

    /*  Compute direction */
    if (lmP->H0) {
      PetscCall(MatLMVMSetJ0(lmP->M, lmP->H0));
      stepType = LMVM_STEP_BFGS;
    }
    PetscCall(MatLMVMUpdate(lmP->M,tao->solution,tao->gradient));
    PetscCall(MatSolve(lmP->M, tao->gradient, lmP->D));
    PetscCall(MatLMVMGetUpdateCount(lmP->M, &nupdates));
    if (nupdates > 0) stepType = LMVM_STEP_BFGS;

    /*  Check for success (descent direction) */
    PetscCall(VecDot(lmP->D, tao->gradient, &gdx));
    if ((gdx <= 0.0) || PetscIsInfOrNanReal(gdx)) {
      /* Step is not descent or direction produced not a number
         We can assert bfgsUpdates > 1 in this case because
         the first solve produces the scaled gradient direction,
         which is guaranteed to be descent

         Use steepest descent direction (scaled)
      */

      PetscCall(MatLMVMReset(lmP->M, PETSC_FALSE));
      PetscCall(MatLMVMClearJ0(lmP->M));
      PetscCall(MatLMVMUpdate(lmP->M, tao->solution, tao->gradient));
      PetscCall(MatSolve(lmP->M,tao->gradient, lmP->D));

      /* On a reset, the direction cannot be not a number; it is a
         scaled gradient step.  No need to check for this condition. */
      stepType = LMVM_STEP_GRAD;
    }
    PetscCall(VecScale(lmP->D, -1.0));

    /*  Perform the linesearch */
    fold = f;
    PetscCall(VecCopy(tao->solution, lmP->Xold));
    PetscCall(VecCopy(tao->gradient, lmP->Gold));

    PetscCall(TaoLineSearchApply(tao->linesearch, tao->solution, &f, tao->gradient, lmP->D, &step,&ls_status));
    PetscCall(TaoAddLineSearchCounts(tao));

    if (ls_status != TAOLINESEARCH_SUCCESS && ls_status != TAOLINESEARCH_SUCCESS_USER && (stepType != LMVM_STEP_GRAD)) {
      /*  Reset factors and use scaled gradient step */
      f = fold;
      PetscCall(VecCopy(lmP->Xold, tao->solution));
      PetscCall(VecCopy(lmP->Gold, tao->gradient));

      /*  Failed to obtain acceptable iterate with BFGS step */
      /*  Attempt to use the scaled gradient direction */

      PetscCall(MatLMVMReset(lmP->M, PETSC_FALSE));
      PetscCall(MatLMVMClearJ0(lmP->M));
      PetscCall(MatLMVMUpdate(lmP->M, tao->solution, tao->gradient));
      PetscCall(MatSolve(lmP->M, tao->solution, tao->gradient));

      /* On a reset, the direction cannot be not a number; it is a
          scaled gradient step.  No need to check for this condition. */
      stepType = LMVM_STEP_GRAD;
      PetscCall(VecScale(lmP->D, -1.0));

      /*  Perform the linesearch */
      PetscCall(TaoLineSearchApply(tao->linesearch, tao->solution, &f, tao->gradient, lmP->D, &step, &ls_status));
      PetscCall(TaoAddLineSearchCounts(tao));
    }

    if (ls_status != TAOLINESEARCH_SUCCESS && ls_status != TAOLINESEARCH_SUCCESS_USER) {
      /*  Failed to find an improving point */
      f = fold;
      PetscCall(VecCopy(lmP->Xold, tao->solution));
      PetscCall(VecCopy(lmP->Gold, tao->gradient));
      step = 0.0;
      tao->reason = TAO_DIVERGED_LS_FAILURE;
    } else {
      /* LS found valid step, so tally up step type */
      switch (stepType) {
      case LMVM_STEP_BFGS:
        ++lmP->bfgs;
        break;
      case LMVM_STEP_GRAD:
        ++lmP->grad;
        break;
      default:
        break;
      }
      /*  Compute new gradient norm */
      PetscCall(TaoGradientNorm(tao, tao->gradient,NORM_2,&gnorm));
    }

    /* Check convergence */
    tao->niter++;
    PetscCall(TaoLogConvergenceHistory(tao,f,gnorm,0.0,tao->ksp_its));
    PetscCall(TaoMonitor(tao,tao->niter,f,gnorm,0.0,step));
    PetscCall((*tao->ops->convergencetest)(tao,tao->cnvP));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSetUp_LMVM(Tao tao)
{
  TAO_LMVM       *lmP = (TAO_LMVM *)tao->data;
  PetscInt       n,N;
  PetscBool      is_set,is_spd;

  PetscFunctionBegin;
  /* Existence of tao->solution checked in TaoSetUp() */
  if (!tao->gradient) PetscCall(VecDuplicate(tao->solution,&tao->gradient));
  if (!tao->stepdirection) PetscCall(VecDuplicate(tao->solution,&tao->stepdirection));
  if (!lmP->D) PetscCall(VecDuplicate(tao->solution,&lmP->D));
  if (!lmP->Xold) PetscCall(VecDuplicate(tao->solution,&lmP->Xold));
  if (!lmP->Gold) PetscCall(VecDuplicate(tao->solution,&lmP->Gold));

  /*  Create matrix for the limited memory approximation */
  PetscCall(VecGetLocalSize(tao->solution,&n));
  PetscCall(VecGetSize(tao->solution,&N));
  PetscCall(MatSetSizes(lmP->M, n, n, N, N));
  PetscCall(MatLMVMAllocate(lmP->M,tao->solution,tao->gradient));
  PetscCall(MatIsSPDKnown(lmP->M, &is_set,&is_spd));
  PetscCheck(is_set && is_spd,PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_INCOMP, "LMVM matrix is not symmetric positive-definite.");

  /* If the user has set a matrix to solve as the initial H0, set the options prefix here, and set up the KSP */
  if (lmP->H0) PetscCall(MatLMVMSetJ0(lmP->M, lmP->H0));
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------- */
static PetscErrorCode TaoDestroy_LMVM(Tao tao)
{
  TAO_LMVM       *lmP = (TAO_LMVM *)tao->data;

  PetscFunctionBegin;
  if (tao->setupcalled) {
    PetscCall(VecDestroy(&lmP->Xold));
    PetscCall(VecDestroy(&lmP->Gold));
    PetscCall(VecDestroy(&lmP->D));
  }
  PetscCall(MatDestroy(&lmP->M));
  if (lmP->H0) PetscCall(PetscObjectDereference((PetscObject)lmP->H0));
  PetscCall(PetscFree(tao->data));
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
static PetscErrorCode TaoSetFromOptions_LMVM(PetscOptionItems *PetscOptionsObject,Tao tao)
{
  TAO_LMVM       *lm = (TAO_LMVM *)tao->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject,"Limited-memory variable-metric method for unconstrained optimization");
  PetscCall(PetscOptionsBool("-tao_lmvm_recycle","enable recycling of the BFGS matrix between subsequent TaoSolve() calls","",lm->recycle,&lm->recycle,NULL));
  PetscCall(TaoLineSearchSetFromOptions(tao->linesearch));
  PetscCall(MatSetFromOptions(lm->M));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
static PetscErrorCode TaoView_LMVM(Tao tao, PetscViewer viewer)
{
  TAO_LMVM       *lm = (TAO_LMVM *)tao->data;
  PetscBool      isascii;
  PetscInt       recycled_its;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Gradient steps: %" PetscInt_FMT "\n", lm->grad));
    if (lm->recycle) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "  Recycle: on\n"));
      recycled_its = lm->bfgs + lm->grad;
      PetscCall(PetscViewerASCIIPrintf(viewer, "  Total recycled iterations: %" PetscInt_FMT "\n", recycled_its));
    }
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------- */

/*MC
  TAOLMVM - Limited Memory Variable Metric method is a quasi-Newton
  optimization solver for unconstrained minimization. It solves
  the Newton step
          Hkdk = - gk

  using an approximation Bk in place of Hk, where Bk is composed using
  the BFGS update formula. A More-Thuente line search is then used
  to computed the steplength in the dk direction

  Options Database Keys:
+   -tao_lmvm_recycle - enable recycling LMVM updates between TaoSolve() calls
-   -tao_lmvm_no_scale - (developer) disables diagonal Broyden scaling on the LMVM approximation

  Level: beginner
M*/

PETSC_EXTERN PetscErrorCode TaoCreate_LMVM(Tao tao)
{
  TAO_LMVM       *lmP;
  const char     *morethuente_type = TAOLINESEARCHMT;

  PetscFunctionBegin;
  tao->ops->setup = TaoSetUp_LMVM;
  tao->ops->solve = TaoSolve_LMVM;
  tao->ops->view = TaoView_LMVM;
  tao->ops->setfromoptions = TaoSetFromOptions_LMVM;
  tao->ops->destroy = TaoDestroy_LMVM;

  PetscCall(PetscNewLog(tao,&lmP));
  lmP->D = NULL;
  lmP->M = NULL;
  lmP->Xold = NULL;
  lmP->Gold = NULL;
  lmP->H0   = NULL;
  lmP->recycle = PETSC_FALSE;

  tao->data = (void*)lmP;
  /* Override default settings (unless already changed) */
  if (!tao->max_it_changed) tao->max_it = 2000;
  if (!tao->max_funcs_changed) tao->max_funcs = 4000;

  PetscCall(TaoLineSearchCreate(((PetscObject)tao)->comm,&tao->linesearch));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)tao->linesearch, (PetscObject)tao, 1));
  PetscCall(TaoLineSearchSetType(tao->linesearch,morethuente_type));
  PetscCall(TaoLineSearchUseTaoRoutines(tao->linesearch,tao));
  PetscCall(TaoLineSearchSetOptionsPrefix(tao->linesearch,tao->hdr.prefix));

  PetscCall(KSPInitializePackage());
  PetscCall(MatCreate(((PetscObject)tao)->comm, &lmP->M));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)lmP->M, (PetscObject)tao, 1));
  PetscCall(MatSetType(lmP->M, MATLMVMBFGS));
  PetscCall(MatSetOptionsPrefix(lmP->M, "tao_lmvm_"));
  PetscFunctionReturn(0);
}
