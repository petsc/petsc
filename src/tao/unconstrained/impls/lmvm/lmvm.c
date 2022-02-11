#include <petsctaolinesearch.h>
#include <../src/tao/unconstrained/impls/lmvm/lmvm.h>

#define LMVM_STEP_BFGS     0
#define LMVM_STEP_GRAD     1

static PetscErrorCode TaoSolve_LMVM(Tao tao)
{
  TAO_LMVM                     *lmP = (TAO_LMVM *)tao->data;
  PetscReal                    f, fold, gdx, gnorm;
  PetscReal                    step = 1.0;
  PetscErrorCode               ierr;
  PetscInt                     stepType = LMVM_STEP_GRAD, nupdates;
  TaoLineSearchConvergedReason ls_status = TAOLINESEARCH_CONTINUE_ITERATING;

  PetscFunctionBegin;

  if (tao->XL || tao->XU || tao->ops->computebounds) {
    ierr = PetscInfo(tao,"WARNING: Variable bounds have been set but will be ignored by lmvm algorithm\n");CHKERRQ(ierr);
  }

  /*  Check convergence criteria */
  ierr = TaoComputeObjectiveAndGradient(tao, tao->solution, &f, tao->gradient);CHKERRQ(ierr);
  ierr = TaoGradientNorm(tao, tao->gradient,NORM_2,&gnorm);CHKERRQ(ierr);

  PetscCheckFalse(PetscIsInfOrNanReal(f) || PetscIsInfOrNanReal(gnorm),PetscObjectComm((PetscObject)tao),PETSC_ERR_USER, "User provided compute function generated Inf or NaN");

  tao->reason = TAO_CONTINUE_ITERATING;
  ierr = TaoLogConvergenceHistory(tao,f,gnorm,0.0,tao->ksp_its);CHKERRQ(ierr);
  ierr = TaoMonitor(tao,tao->niter,f,gnorm,0.0,step);CHKERRQ(ierr);
  ierr = (*tao->ops->convergencetest)(tao,tao->cnvP);CHKERRQ(ierr);
  if (tao->reason != TAO_CONTINUE_ITERATING) PetscFunctionReturn(0);

  /*  Set counter for gradient/reset steps */
  if (!lmP->recycle) {
    lmP->bfgs = 0;
    lmP->grad = 0;
    ierr = MatLMVMReset(lmP->M, PETSC_FALSE);CHKERRQ(ierr);
  }

  /*  Have not converged; continue with Newton method */
  while (tao->reason == TAO_CONTINUE_ITERATING) {
    /* Call general purpose update function */
    if (tao->ops->update) {
      ierr = (*tao->ops->update)(tao, tao->niter, tao->user_update);CHKERRQ(ierr);
    }

    /*  Compute direction */
    if (lmP->H0) {
      ierr = MatLMVMSetJ0(lmP->M, lmP->H0);CHKERRQ(ierr);
      stepType = LMVM_STEP_BFGS;
    }
    ierr = MatLMVMUpdate(lmP->M,tao->solution,tao->gradient);CHKERRQ(ierr);
    ierr = MatSolve(lmP->M, tao->gradient, lmP->D);CHKERRQ(ierr);
    ierr = MatLMVMGetUpdateCount(lmP->M, &nupdates);CHKERRQ(ierr);
    if (nupdates > 0) stepType = LMVM_STEP_BFGS;

    /*  Check for success (descent direction) */
    ierr = VecDot(lmP->D, tao->gradient, &gdx);CHKERRQ(ierr);
    if ((gdx <= 0.0) || PetscIsInfOrNanReal(gdx)) {
      /* Step is not descent or direction produced not a number
         We can assert bfgsUpdates > 1 in this case because
         the first solve produces the scaled gradient direction,
         which is guaranteed to be descent

         Use steepest descent direction (scaled)
      */

      ierr = MatLMVMReset(lmP->M, PETSC_FALSE);CHKERRQ(ierr);
      ierr = MatLMVMClearJ0(lmP->M);CHKERRQ(ierr);
      ierr = MatLMVMUpdate(lmP->M, tao->solution, tao->gradient);CHKERRQ(ierr);
      ierr = MatSolve(lmP->M,tao->gradient, lmP->D);CHKERRQ(ierr);

      /* On a reset, the direction cannot be not a number; it is a
         scaled gradient step.  No need to check for this condition. */
      stepType = LMVM_STEP_GRAD;
    }
    ierr = VecScale(lmP->D, -1.0);CHKERRQ(ierr);

    /*  Perform the linesearch */
    fold = f;
    ierr = VecCopy(tao->solution, lmP->Xold);CHKERRQ(ierr);
    ierr = VecCopy(tao->gradient, lmP->Gold);CHKERRQ(ierr);

    ierr = TaoLineSearchApply(tao->linesearch, tao->solution, &f, tao->gradient, lmP->D, &step,&ls_status);CHKERRQ(ierr);
    ierr = TaoAddLineSearchCounts(tao);CHKERRQ(ierr);

    if (ls_status != TAOLINESEARCH_SUCCESS && ls_status != TAOLINESEARCH_SUCCESS_USER && (stepType != LMVM_STEP_GRAD)) {
      /*  Reset factors and use scaled gradient step */
      f = fold;
      ierr = VecCopy(lmP->Xold, tao->solution);CHKERRQ(ierr);
      ierr = VecCopy(lmP->Gold, tao->gradient);CHKERRQ(ierr);

      /*  Failed to obtain acceptable iterate with BFGS step */
      /*  Attempt to use the scaled gradient direction */

      ierr = MatLMVMReset(lmP->M, PETSC_FALSE);CHKERRQ(ierr);
      ierr = MatLMVMClearJ0(lmP->M);CHKERRQ(ierr);
      ierr = MatLMVMUpdate(lmP->M, tao->solution, tao->gradient);CHKERRQ(ierr);
      ierr = MatSolve(lmP->M, tao->solution, tao->gradient);CHKERRQ(ierr);

      /* On a reset, the direction cannot be not a number; it is a
          scaled gradient step.  No need to check for this condition. */
      stepType = LMVM_STEP_GRAD;
      ierr = VecScale(lmP->D, -1.0);CHKERRQ(ierr);

      /*  Perform the linesearch */
      ierr = TaoLineSearchApply(tao->linesearch, tao->solution, &f, tao->gradient, lmP->D, &step, &ls_status);CHKERRQ(ierr);
      ierr = TaoAddLineSearchCounts(tao);CHKERRQ(ierr);
    }

    if (ls_status != TAOLINESEARCH_SUCCESS && ls_status != TAOLINESEARCH_SUCCESS_USER) {
      /*  Failed to find an improving point */
      f = fold;
      ierr = VecCopy(lmP->Xold, tao->solution);CHKERRQ(ierr);
      ierr = VecCopy(lmP->Gold, tao->gradient);CHKERRQ(ierr);
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
      ierr = TaoGradientNorm(tao, tao->gradient,NORM_2,&gnorm);CHKERRQ(ierr);
    }

    /* Check convergence */
    tao->niter++;
    ierr = TaoLogConvergenceHistory(tao,f,gnorm,0.0,tao->ksp_its);CHKERRQ(ierr);
    ierr = TaoMonitor(tao,tao->niter,f,gnorm,0.0,step);CHKERRQ(ierr);
    ierr = (*tao->ops->convergencetest)(tao,tao->cnvP);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSetUp_LMVM(Tao tao)
{
  TAO_LMVM       *lmP = (TAO_LMVM *)tao->data;
  PetscInt       n,N;
  PetscErrorCode ierr;
  PetscBool      is_spd;

  PetscFunctionBegin;
  /* Existence of tao->solution checked in TaoSetUp() */
  if (!tao->gradient) {ierr = VecDuplicate(tao->solution,&tao->gradient);CHKERRQ(ierr);  }
  if (!tao->stepdirection) {ierr = VecDuplicate(tao->solution,&tao->stepdirection);CHKERRQ(ierr);  }
  if (!lmP->D) {ierr = VecDuplicate(tao->solution,&lmP->D);CHKERRQ(ierr);  }
  if (!lmP->Xold) {ierr = VecDuplicate(tao->solution,&lmP->Xold);CHKERRQ(ierr);  }
  if (!lmP->Gold) {ierr = VecDuplicate(tao->solution,&lmP->Gold);CHKERRQ(ierr);  }

  /*  Create matrix for the limited memory approximation */
  ierr = VecGetLocalSize(tao->solution,&n);CHKERRQ(ierr);
  ierr = VecGetSize(tao->solution,&N);CHKERRQ(ierr);
  ierr = MatSetSizes(lmP->M, n, n, N, N);CHKERRQ(ierr);
  ierr = MatLMVMAllocate(lmP->M,tao->solution,tao->gradient);CHKERRQ(ierr);
  ierr = MatGetOption(lmP->M, MAT_SPD, &is_spd);CHKERRQ(ierr);
  PetscCheckFalse(!is_spd,PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_INCOMP, "LMVM matrix is not symmetric positive-definite.");

  /* If the user has set a matrix to solve as the initial H0, set the options prefix here, and set up the KSP */
  if (lmP->H0) {
    ierr = MatLMVMSetJ0(lmP->M, lmP->H0);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------- */
static PetscErrorCode TaoDestroy_LMVM(Tao tao)
{
  TAO_LMVM       *lmP = (TAO_LMVM *)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (tao->setupcalled) {
    ierr = VecDestroy(&lmP->Xold);CHKERRQ(ierr);
    ierr = VecDestroy(&lmP->Gold);CHKERRQ(ierr);
    ierr = VecDestroy(&lmP->D);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&lmP->M);CHKERRQ(ierr);
  if (lmP->H0) {
    ierr = PetscObjectDereference((PetscObject)lmP->H0);CHKERRQ(ierr);
  }
  ierr = PetscFree(tao->data);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
static PetscErrorCode TaoSetFromOptions_LMVM(PetscOptionItems *PetscOptionsObject,Tao tao)
{
  TAO_LMVM       *lm = (TAO_LMVM *)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"Limited-memory variable-metric method for unconstrained optimization");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-tao_lmvm_recycle","enable recycling of the BFGS matrix between subsequent TaoSolve() calls","",lm->recycle,&lm->recycle,NULL);CHKERRQ(ierr);
  ierr = TaoLineSearchSetFromOptions(tao->linesearch);CHKERRQ(ierr);
  ierr = MatSetFromOptions(lm->M);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
static PetscErrorCode TaoView_LMVM(Tao tao, PetscViewer viewer)
{
  TAO_LMVM       *lm = (TAO_LMVM *)tao->data;
  PetscBool      isascii;
  PetscInt       recycled_its;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer, "  Gradient steps: %D\n", lm->grad);CHKERRQ(ierr);
    if (lm->recycle) {
      ierr = PetscViewerASCIIPrintf(viewer, "  Recycle: on\n");CHKERRQ(ierr);
      recycled_its = lm->bfgs + lm->grad;
      ierr = PetscViewerASCIIPrintf(viewer, "  Total recycled iterations: %D\n", recycled_its);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  tao->ops->setup = TaoSetUp_LMVM;
  tao->ops->solve = TaoSolve_LMVM;
  tao->ops->view = TaoView_LMVM;
  tao->ops->setfromoptions = TaoSetFromOptions_LMVM;
  tao->ops->destroy = TaoDestroy_LMVM;

  ierr = PetscNewLog(tao,&lmP);CHKERRQ(ierr);
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

  ierr = TaoLineSearchCreate(((PetscObject)tao)->comm,&tao->linesearch);CHKERRQ(ierr);
  ierr = PetscObjectIncrementTabLevel((PetscObject)tao->linesearch, (PetscObject)tao, 1);CHKERRQ(ierr);
  ierr = TaoLineSearchSetType(tao->linesearch,morethuente_type);CHKERRQ(ierr);
  ierr = TaoLineSearchUseTaoRoutines(tao->linesearch,tao);CHKERRQ(ierr);
  ierr = TaoLineSearchSetOptionsPrefix(tao->linesearch,tao->hdr.prefix);CHKERRQ(ierr);

  ierr = KSPInitializePackage();CHKERRQ(ierr);
  ierr = MatCreate(((PetscObject)tao)->comm, &lmP->M);CHKERRQ(ierr);
  ierr = PetscObjectIncrementTabLevel((PetscObject)lmP->M, (PetscObject)tao, 1);CHKERRQ(ierr);
  ierr = MatSetType(lmP->M, MATLMVMBFGS);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(lmP->M, "tao_lmvm_");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
