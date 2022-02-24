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
    CHKERRQ(PetscInfo(tao,"WARNING: Variable bounds have been set but will be ignored by lmvm algorithm\n"));
  }

  /*  Check convergence criteria */
  CHKERRQ(TaoComputeObjectiveAndGradient(tao, tao->solution, &f, tao->gradient));
  CHKERRQ(TaoGradientNorm(tao, tao->gradient,NORM_2,&gnorm));

  PetscCheck(!PetscIsInfOrNanReal(f) && !PetscIsInfOrNanReal(gnorm),PetscObjectComm((PetscObject)tao),PETSC_ERR_USER, "User provided compute function generated Inf or NaN");

  tao->reason = TAO_CONTINUE_ITERATING;
  CHKERRQ(TaoLogConvergenceHistory(tao,f,gnorm,0.0,tao->ksp_its));
  CHKERRQ(TaoMonitor(tao,tao->niter,f,gnorm,0.0,step));
  CHKERRQ((*tao->ops->convergencetest)(tao,tao->cnvP));
  if (tao->reason != TAO_CONTINUE_ITERATING) PetscFunctionReturn(0);

  /*  Set counter for gradient/reset steps */
  if (!lmP->recycle) {
    lmP->bfgs = 0;
    lmP->grad = 0;
    CHKERRQ(MatLMVMReset(lmP->M, PETSC_FALSE));
  }

  /*  Have not converged; continue with Newton method */
  while (tao->reason == TAO_CONTINUE_ITERATING) {
    /* Call general purpose update function */
    if (tao->ops->update) {
      CHKERRQ((*tao->ops->update)(tao, tao->niter, tao->user_update));
    }

    /*  Compute direction */
    if (lmP->H0) {
      CHKERRQ(MatLMVMSetJ0(lmP->M, lmP->H0));
      stepType = LMVM_STEP_BFGS;
    }
    CHKERRQ(MatLMVMUpdate(lmP->M,tao->solution,tao->gradient));
    CHKERRQ(MatSolve(lmP->M, tao->gradient, lmP->D));
    CHKERRQ(MatLMVMGetUpdateCount(lmP->M, &nupdates));
    if (nupdates > 0) stepType = LMVM_STEP_BFGS;

    /*  Check for success (descent direction) */
    CHKERRQ(VecDot(lmP->D, tao->gradient, &gdx));
    if ((gdx <= 0.0) || PetscIsInfOrNanReal(gdx)) {
      /* Step is not descent or direction produced not a number
         We can assert bfgsUpdates > 1 in this case because
         the first solve produces the scaled gradient direction,
         which is guaranteed to be descent

         Use steepest descent direction (scaled)
      */

      CHKERRQ(MatLMVMReset(lmP->M, PETSC_FALSE));
      CHKERRQ(MatLMVMClearJ0(lmP->M));
      CHKERRQ(MatLMVMUpdate(lmP->M, tao->solution, tao->gradient));
      CHKERRQ(MatSolve(lmP->M,tao->gradient, lmP->D));

      /* On a reset, the direction cannot be not a number; it is a
         scaled gradient step.  No need to check for this condition. */
      stepType = LMVM_STEP_GRAD;
    }
    CHKERRQ(VecScale(lmP->D, -1.0));

    /*  Perform the linesearch */
    fold = f;
    CHKERRQ(VecCopy(tao->solution, lmP->Xold));
    CHKERRQ(VecCopy(tao->gradient, lmP->Gold));

    CHKERRQ(TaoLineSearchApply(tao->linesearch, tao->solution, &f, tao->gradient, lmP->D, &step,&ls_status));
    CHKERRQ(TaoAddLineSearchCounts(tao));

    if (ls_status != TAOLINESEARCH_SUCCESS && ls_status != TAOLINESEARCH_SUCCESS_USER && (stepType != LMVM_STEP_GRAD)) {
      /*  Reset factors and use scaled gradient step */
      f = fold;
      CHKERRQ(VecCopy(lmP->Xold, tao->solution));
      CHKERRQ(VecCopy(lmP->Gold, tao->gradient));

      /*  Failed to obtain acceptable iterate with BFGS step */
      /*  Attempt to use the scaled gradient direction */

      CHKERRQ(MatLMVMReset(lmP->M, PETSC_FALSE));
      CHKERRQ(MatLMVMClearJ0(lmP->M));
      CHKERRQ(MatLMVMUpdate(lmP->M, tao->solution, tao->gradient));
      CHKERRQ(MatSolve(lmP->M, tao->solution, tao->gradient));

      /* On a reset, the direction cannot be not a number; it is a
          scaled gradient step.  No need to check for this condition. */
      stepType = LMVM_STEP_GRAD;
      CHKERRQ(VecScale(lmP->D, -1.0));

      /*  Perform the linesearch */
      CHKERRQ(TaoLineSearchApply(tao->linesearch, tao->solution, &f, tao->gradient, lmP->D, &step, &ls_status));
      CHKERRQ(TaoAddLineSearchCounts(tao));
    }

    if (ls_status != TAOLINESEARCH_SUCCESS && ls_status != TAOLINESEARCH_SUCCESS_USER) {
      /*  Failed to find an improving point */
      f = fold;
      CHKERRQ(VecCopy(lmP->Xold, tao->solution));
      CHKERRQ(VecCopy(lmP->Gold, tao->gradient));
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
      CHKERRQ(TaoGradientNorm(tao, tao->gradient,NORM_2,&gnorm));
    }

    /* Check convergence */
    tao->niter++;
    CHKERRQ(TaoLogConvergenceHistory(tao,f,gnorm,0.0,tao->ksp_its));
    CHKERRQ(TaoMonitor(tao,tao->niter,f,gnorm,0.0,step));
    CHKERRQ((*tao->ops->convergencetest)(tao,tao->cnvP));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSetUp_LMVM(Tao tao)
{
  TAO_LMVM       *lmP = (TAO_LMVM *)tao->data;
  PetscInt       n,N;
  PetscBool      is_spd;

  PetscFunctionBegin;
  /* Existence of tao->solution checked in TaoSetUp() */
  if (!tao->gradient) CHKERRQ(VecDuplicate(tao->solution,&tao->gradient));
  if (!tao->stepdirection) CHKERRQ(VecDuplicate(tao->solution,&tao->stepdirection));
  if (!lmP->D) CHKERRQ(VecDuplicate(tao->solution,&lmP->D));
  if (!lmP->Xold) CHKERRQ(VecDuplicate(tao->solution,&lmP->Xold));
  if (!lmP->Gold) CHKERRQ(VecDuplicate(tao->solution,&lmP->Gold));

  /*  Create matrix for the limited memory approximation */
  CHKERRQ(VecGetLocalSize(tao->solution,&n));
  CHKERRQ(VecGetSize(tao->solution,&N));
  CHKERRQ(MatSetSizes(lmP->M, n, n, N, N));
  CHKERRQ(MatLMVMAllocate(lmP->M,tao->solution,tao->gradient));
  CHKERRQ(MatGetOption(lmP->M, MAT_SPD, &is_spd));
  PetscCheck(is_spd,PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_INCOMP, "LMVM matrix is not symmetric positive-definite.");

  /* If the user has set a matrix to solve as the initial H0, set the options prefix here, and set up the KSP */
  if (lmP->H0) {
    CHKERRQ(MatLMVMSetJ0(lmP->M, lmP->H0));
  }

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------- */
static PetscErrorCode TaoDestroy_LMVM(Tao tao)
{
  TAO_LMVM       *lmP = (TAO_LMVM *)tao->data;

  PetscFunctionBegin;
  if (tao->setupcalled) {
    CHKERRQ(VecDestroy(&lmP->Xold));
    CHKERRQ(VecDestroy(&lmP->Gold));
    CHKERRQ(VecDestroy(&lmP->D));
  }
  CHKERRQ(MatDestroy(&lmP->M));
  if (lmP->H0) {
    CHKERRQ(PetscObjectDereference((PetscObject)lmP->H0));
  }
  CHKERRQ(PetscFree(tao->data));

  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
static PetscErrorCode TaoSetFromOptions_LMVM(PetscOptionItems *PetscOptionsObject,Tao tao)
{
  TAO_LMVM       *lm = (TAO_LMVM *)tao->data;

  PetscFunctionBegin;
  CHKERRQ(PetscOptionsHead(PetscOptionsObject,"Limited-memory variable-metric method for unconstrained optimization"));
  CHKERRQ(PetscOptionsBool("-tao_lmvm_recycle","enable recycling of the BFGS matrix between subsequent TaoSolve() calls","",lm->recycle,&lm->recycle,NULL));
  CHKERRQ(TaoLineSearchSetFromOptions(tao->linesearch));
  CHKERRQ(MatSetFromOptions(lm->M));
  CHKERRQ(PetscOptionsTail());
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
static PetscErrorCode TaoView_LMVM(Tao tao, PetscViewer viewer)
{
  TAO_LMVM       *lm = (TAO_LMVM *)tao->data;
  PetscBool      isascii;
  PetscInt       recycled_its;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer, "  Gradient steps: %D\n", lm->grad));
    if (lm->recycle) {
      CHKERRQ(PetscViewerASCIIPrintf(viewer, "  Recycle: on\n"));
      recycled_its = lm->bfgs + lm->grad;
      CHKERRQ(PetscViewerASCIIPrintf(viewer, "  Total recycled iterations: %D\n", recycled_its));
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

  CHKERRQ(PetscNewLog(tao,&lmP));
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

  CHKERRQ(TaoLineSearchCreate(((PetscObject)tao)->comm,&tao->linesearch));
  CHKERRQ(PetscObjectIncrementTabLevel((PetscObject)tao->linesearch, (PetscObject)tao, 1));
  CHKERRQ(TaoLineSearchSetType(tao->linesearch,morethuente_type));
  CHKERRQ(TaoLineSearchUseTaoRoutines(tao->linesearch,tao));
  CHKERRQ(TaoLineSearchSetOptionsPrefix(tao->linesearch,tao->hdr.prefix));

  CHKERRQ(KSPInitializePackage());
  CHKERRQ(MatCreate(((PetscObject)tao)->comm, &lmP->M));
  CHKERRQ(PetscObjectIncrementTabLevel((PetscObject)lmP->M, (PetscObject)tao, 1));
  CHKERRQ(MatSetType(lmP->M, MATLMVMBFGS));
  CHKERRQ(MatSetOptionsPrefix(lmP->M, "tao_lmvm_"));
  PetscFunctionReturn(0);
}
