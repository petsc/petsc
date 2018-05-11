#include <petsctaolinesearch.h>
#include <../src/tao/unconstrained/impls/lmvm/lmvm.h>
#include <../src/tao/bound/impls/blmvm/blmvm.h>

#define BLMVM_STEP_BFGS     0
#define BLMVM_STEP_GRAD     1

#define BLMVM_AS_NONE       0
#define BLMVM_AS_BERTSEKAS  1
#define BLMVM_AS_SIZE       2

static const char *BLMVM_AS_TYPE[64] = {"none", "bertsekas"};

PETSC_INTERN PetscErrorCode TaoBLMVMEstimateActiveSet(Tao tao, PetscInt asType)
{
  PetscErrorCode               ierr;
  TAO_BLMVM                     *blmP = (TAO_BLMVM *)tao->data;

  PetscFunctionBegin;
  if (!tao->bounded) PetscFunctionReturn(0);
  switch (asType) {
  case BLMVM_AS_NONE:
    ierr = ISDestroy(&blmP->inactive_idx);CHKERRQ(ierr);
    ierr = VecWhichInactive(tao->XL, tao->solution, blmP->unprojected_gradient, tao->XU, PETSC_TRUE, &blmP->inactive_idx);CHKERRQ(ierr);
    ierr = ISDestroy(&blmP->active_idx);CHKERRQ(ierr);
    ierr = ISComplementVec(blmP->inactive_idx, tao->solution, &blmP->active_idx);CHKERRQ(ierr);
    break;

  case BLMVM_AS_BERTSEKAS:
    /* Use gradient descent to estimate the active set */
    ierr = VecCopy(blmP->unprojected_gradient, blmP->W);CHKERRQ(ierr);
    ierr = VecScale(blmP->W, -1.0);CHKERRQ(ierr);
    ierr = TaoEstimateActiveBounds(tao->solution, tao->XL, tao->XU, blmP->unprojected_gradient, blmP->W, blmP->work, blmP->as_step, &blmP->as_tol, 
                                   &blmP->active_lower, &blmP->active_upper, &blmP->active_fixed, &blmP->active_idx, &blmP->inactive_idx);CHKERRQ(ierr);
    break;
    
  default:
    break;
  }
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode TaoBLMVMBoundStep(Tao tao, PetscInt asType, Vec step)
{
  PetscErrorCode               ierr;
  TAO_BLMVM                     *blmP = (TAO_BLMVM *)tao->data;
  
  PetscFunctionBegin;
  if (!tao->bounded) PetscFunctionReturn(0);
  switch (asType) {
  case BLMVM_AS_NONE:
    ierr = VecISSet(step, blmP->active_idx, 0.0);CHKERRQ(ierr);
    break;

  case BLMVM_AS_BERTSEKAS:
    ierr = TaoBoundStep(tao->solution, tao->XL, tao->XU, blmP->active_lower, blmP->active_upper, blmP->active_fixed, 1.0, step);CHKERRQ(ierr);
    break;

  default:
    break;
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
PETSC_INTERN PetscErrorCode TaoSolve_BLMVM(Tao tao)
{
  PetscErrorCode               ierr;
  TAO_BLMVM                    *blmP = (TAO_BLMVM *)tao->data;
  TaoLineSearchConvergedReason ls_status = TAOLINESEARCH_CONTINUE_ITERATING;
  PetscReal                    f, fold, gdx, gnorm, resnorm;
  PetscReal                    stepsize = 1.0;
  PetscInt                     nDiff, nupdates, stepType = BLMVM_STEP_GRAD;

  PetscFunctionBegin;
  /*  Project initial point onto bounds */
  ierr = TaoComputeVariableBounds(tao);CHKERRQ(ierr);
  ierr = TaoBoundSolution(tao->solution, tao->XL,tao->XU, 0.0, &nDiff, tao->solution);CHKERRQ(ierr);
  if (tao->bounded) {
    ierr = TaoLineSearchSetVariableBounds(tao->linesearch,tao->XL,tao->XU);CHKERRQ(ierr);
  }

  /* Check convergence criteria */
  ierr = TaoComputeObjectiveAndGradient(tao, tao->solution,&f,blmP->unprojected_gradient);CHKERRQ(ierr);
  ierr = VecBoundGradientProjection(blmP->unprojected_gradient,tao->solution, tao->XL,tao->XU,tao->gradient);CHKERRQ(ierr);

  ierr = TaoGradientNorm(tao, blmP->unprojected_gradient,NORM_2,&gnorm);CHKERRQ(ierr);
  if (PetscIsInfOrNanReal(f) || PetscIsInfOrNanReal(gnorm)) SETERRQ(PETSC_COMM_SELF,1, "User provided compute function generated Inf or NaN");

  tao->reason = TAO_CONTINUE_ITERATING;
  ierr = VecFischer(tao->solution, blmP->unprojected_gradient, tao->XL, tao->XU, blmP->W);CHKERRQ(ierr);
  ierr = VecNorm(blmP->W, NORM_2, &resnorm);CHKERRQ(ierr);
  ierr = TaoLogConvergenceHistory(tao,f,resnorm,0.0,tao->ksp_its);CHKERRQ(ierr);
  ierr = TaoMonitor(tao,tao->niter,f,resnorm,0.0,stepsize);CHKERRQ(ierr);
  ierr = (*tao->ops->convergencetest)(tao,tao->cnvP);CHKERRQ(ierr);
  if (tao->reason != TAO_CONTINUE_ITERATING) PetscFunctionReturn(0);

  /* Set counter for gradient/reset steps */
  if (!blmP->recycle) {
    blmP->bfgs = 0;
    blmP->grad = 0;
    ierr = MatLMVMReset(blmP->M, PETSC_FALSE);CHKERRQ(ierr);
  }

  /* Have not converged; continue with Newton method */
  while (tao->reason == TAO_CONTINUE_ITERATING) {
    /* Estimate active set at the current iterate */
    ierr = TaoBLMVMEstimateActiveSet(tao, blmP->as_type);CHKERRQ(ierr);
    
    /* Compute direction */
    if (blmP->H0) {
      ierr = MatLMVMSetJ0(blmP->M, blmP->H0);CHKERRQ(ierr);
      stepType = BLMVM_STEP_BFGS;
    }
    ierr = MatLMVMUpdate(blmP->M, tao->solution, blmP->unprojected_gradient);CHKERRQ(ierr);
    ierr = VecCopy(blmP->unprojected_gradient, blmP->red_grad);CHKERRQ(ierr);
    ierr = VecISSet(blmP->red_grad, blmP->active_idx, 0.0);CHKERRQ(ierr);
    ierr = MatSolve(blmP->M, blmP->red_grad, tao->stepdirection);CHKERRQ(ierr);
    ierr = VecScale(tao->stepdirection,-1.0);CHKERRQ(ierr);
    ierr = TaoBLMVMBoundStep(tao, blmP->as_type, tao->stepdirection);CHKERRQ(ierr);
    ierr = MatLMVMGetUpdateCount(blmP->M, &nupdates);CHKERRQ(ierr);
    if (nupdates > 0) stepType = BLMVM_STEP_BFGS;

    /* Check for success (descent direction) */
    ierr = VecDot(tao->stepdirection, blmP->red_grad, &gdx);CHKERRQ(ierr);
    if (gdx >= 0 || PetscIsInfOrNanReal(gdx)) {
      /* Step is not descent or solve was not successful
         Use steepest descent direction (scaled) */
      ierr = MatLMVMResetJ0(blmP->M);CHKERRQ(ierr);
      ierr = MatLMVMReset(blmP->M, PETSC_FALSE);CHKERRQ(ierr);
      ierr = MatLMVMUpdate(blmP->M, tao->solution, blmP->unprojected_gradient);CHKERRQ(ierr);
      ierr = MatSolve(blmP->M, blmP->unprojected_gradient, tao->stepdirection);CHKERRQ(ierr);
      ierr = VecScale(tao->stepdirection,-1.0);CHKERRQ(ierr);
      ierr = TaoBLMVMBoundStep(tao, blmP->as_type, tao->stepdirection);CHKERRQ(ierr);
      stepType = BLMVM_STEP_GRAD;
    }

    /* Perform the linesearch */
    fold = f;
    ierr = VecCopy(tao->solution, blmP->Xold);CHKERRQ(ierr);
    ierr = VecCopy(blmP->unprojected_gradient, blmP->Gold);CHKERRQ(ierr);
    ierr = TaoLineSearchApply(tao->linesearch, tao->solution, &f, blmP->unprojected_gradient, tao->stepdirection, &stepsize, &ls_status);CHKERRQ(ierr);
    ierr = TaoAddLineSearchCounts(tao);CHKERRQ(ierr);

    if (ls_status != TAOLINESEARCH_SUCCESS && ls_status != TAOLINESEARCH_SUCCESS_USER && stepType != BLMVM_STEP_GRAD) {
      /* Linesearch failed
         Reset factors and use scaled (projected) gradient step */
      f = fold;
      ierr = VecCopy(blmP->Xold, tao->solution);CHKERRQ(ierr);
      ierr = VecCopy(blmP->Gold, blmP->unprojected_gradient);CHKERRQ(ierr);

      ierr = MatLMVMResetJ0(blmP->M);CHKERRQ(ierr);
      ierr = MatLMVMReset(blmP->M, PETSC_FALSE);CHKERRQ(ierr);
      ierr = MatLMVMUpdate(blmP->M, tao->solution, blmP->unprojected_gradient);CHKERRQ(ierr);
      ierr = MatSolve(blmP->M, blmP->unprojected_gradient, tao->stepdirection);CHKERRQ(ierr);
      ierr = VecScale(tao->stepdirection, -1.0);CHKERRQ(ierr);
      ierr = TaoBLMVMBoundStep(tao, blmP->as_type, tao->stepdirection);CHKERRQ(ierr);
      stepType = BLMVM_STEP_GRAD;
      
      ierr = TaoLineSearchApply(tao->linesearch,tao->solution,&f, blmP->unprojected_gradient, tao->stepdirection,  &stepsize, &ls_status);CHKERRQ(ierr);
      ierr = TaoAddLineSearchCounts(tao);CHKERRQ(ierr);
    }
    
    if (ls_status != TAOLINESEARCH_SUCCESS && ls_status != TAOLINESEARCH_SUCCESS_USER) {
      /* Line search failed on a gradient step, so just mark reason for divergence */
      f = fold;
      ierr = VecCopy(blmP->Xold, tao->solution);CHKERRQ(ierr);
      ierr = VecCopy(blmP->Gold, blmP->unprojected_gradient);CHKERRQ(ierr);
      tao->reason = TAO_DIVERGED_LS_FAILURE;
    } else {
      /* LS found valid step, so tally the step type and compute projected gradient */
      switch (stepType) {
      case BLMVM_STEP_BFGS:
        ++blmP->bfgs;
        break;
      case BLMVM_STEP_GRAD:
        ++blmP->grad;
        break;
      default:
        break;
      }
      ierr = VecBoundGradientProjection(blmP->unprojected_gradient, tao->solution, tao->XL, tao->XU, tao->gradient);CHKERRQ(ierr);
      ierr = TaoGradientNorm(tao, blmP->unprojected_gradient,NORM_2,&gnorm);CHKERRQ(ierr);
      if (PetscIsInfOrNanReal(f) || PetscIsInfOrNanReal(gnorm)) SETERRQ(PETSC_COMM_SELF,1, "User provided compute function generated Not-a-Number");
    }
    
    /* Check for converged */
    tao->niter++;
    ierr = VecFischer(tao->solution, blmP->unprojected_gradient, tao->XL, tao->XU, blmP->W);CHKERRQ(ierr);
    ierr = VecNorm(blmP->W, NORM_2, &resnorm);CHKERRQ(ierr);
    ierr = TaoLogConvergenceHistory(tao,f,resnorm,0.0,tao->ksp_its);CHKERRQ(ierr);
    ierr = TaoMonitor(tao,tao->niter,f,resnorm,0.0,stepsize);CHKERRQ(ierr);
    ierr = (*tao->ops->convergencetest)(tao,tao->cnvP);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode TaoSetup_BLMVM(Tao tao)
{
  TAO_BLMVM      *blmP = (TAO_BLMVM *)tao->data;
  PetscInt       n,N;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Existence of tao->solution checked in TaoSetup() */
  ierr = VecDuplicate(tao->solution,&blmP->Xold);CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution,&blmP->Gold);CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution, &blmP->unprojected_gradient);CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution, &blmP->W);CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution, &blmP->work);CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution, &blmP->red_grad);CHKERRQ(ierr);

  if (!tao->stepdirection) {
    ierr = VecDuplicate(tao->solution, &tao->stepdirection);CHKERRQ(ierr);
  }
  if (!tao->gradient) {
    ierr = VecDuplicate(tao->solution,&tao->gradient);CHKERRQ(ierr);
  }
  /* Create matrix for the limited memory approximation */
  ierr = VecGetLocalSize(tao->solution,&n);CHKERRQ(ierr);
  ierr = VecGetSize(tao->solution,&N);CHKERRQ(ierr);
  ierr = MatSetSizes(blmP->M, n, n, N, N);CHKERRQ(ierr);
  ierr = MatLMVMAllocate(blmP->M,tao->solution,blmP->unprojected_gradient);CHKERRQ(ierr);

  /* If the user has set a matrix to solve as the initial H0, set the options prefix here, and set up the KSP */
  if (blmP->H0) {
    ierr = MatLMVMSetJ0(blmP->M, blmP->H0);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------- */
PETSC_INTERN PetscErrorCode TaoDestroy_BLMVM(Tao tao)
{
  TAO_BLMVM      *blmP = (TAO_BLMVM *)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (tao->setupcalled) {
    ierr = VecDestroy(&blmP->unprojected_gradient);CHKERRQ(ierr);
    ierr = VecDestroy(&blmP->Xold);CHKERRQ(ierr);
    ierr = VecDestroy(&blmP->Gold);CHKERRQ(ierr);
    ierr = VecDestroy(&blmP->W);CHKERRQ(ierr);
    ierr = VecDestroy(&blmP->work);CHKERRQ(ierr);
    ierr = VecDestroy(&blmP->red_grad);CHKERRQ(ierr);
  }
  ierr = ISDestroy(&blmP->active_lower);CHKERRQ(ierr);
  ierr = ISDestroy(&blmP->active_upper);CHKERRQ(ierr);
  ierr = ISDestroy(&blmP->active_fixed);CHKERRQ(ierr);
  ierr = ISDestroy(&blmP->active_idx);CHKERRQ(ierr);
  ierr = ISDestroy(&blmP->inactive_idx);CHKERRQ(ierr);
  ierr = MatDestroy(&blmP->M);CHKERRQ(ierr);
  if (blmP->H0) {
    PetscObjectDereference((PetscObject)blmP->H0);
  }

  ierr = PetscFree(tao->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
PETSC_INTERN PetscErrorCode TaoSetFromOptions_BLMVM(PetscOptionItems* PetscOptionsObject,Tao tao)
{
  TAO_BLMVM      *blmP = (TAO_BLMVM *)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"Limited-memory variable-metric method for bound constrained optimization");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-tao_blmvm_recycle","enable recycling of the BFGS matrix between subsequent TaoSolve() calls","",blmP->recycle,&blmP->recycle,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEList("-tao_blmvm_as_type","active set estimation method", "", BLMVM_AS_TYPE, BLMVM_AS_SIZE, BLMVM_AS_TYPE[blmP->as_type], &blmP->as_type,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_blmvm_as_tol", "initial tolerance used when estimating actively bounded variables","",blmP->as_tol,&blmP->as_tol,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_blmvm_as_step", "step length used when estimating actively bounded variables","",blmP->as_step,&blmP->as_step,NULL);CHKERRQ(ierr);
  ierr = TaoLineSearchSetFromOptions(tao->linesearch);CHKERRQ(ierr);
  ierr = MatSetFromOptions(blmP->M);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*------------------------------------------------------------*/
PETSC_INTERN PetscErrorCode TaoView_BLMVM(Tao tao, PetscViewer viewer)
{
  TAO_BLMVM      *lmP = (TAO_BLMVM *)tao->data;
  PetscBool      isascii;
  PetscErrorCode ierr;
  PetscInt       recycled_its;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer, "  Gradient steps: %D\n", lmP->grad);CHKERRQ(ierr);
    if (lmP->recycle) {
      ierr = PetscViewerASCIIPrintf(viewer, "  Recycle: on\n");CHKERRQ(ierr);
      recycled_its = lmP->bfgs + lmP->grad;
      ierr = PetscViewerASCIIPrintf(viewer, "  Total recycled iterations: %D\n", recycled_its);CHKERRQ(ierr);
    }
    ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_INFO);CHKERRQ(ierr);
    ierr = MatView(lmP->M, viewer);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode TaoComputeDual_BLMVM(Tao tao, Vec DXL, Vec DXU)
{
  TAO_BLMVM      *blm = (TAO_BLMVM *) tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidHeaderSpecific(DXL,VEC_CLASSID,2);
  PetscValidHeaderSpecific(DXU,VEC_CLASSID,3);
  if (!tao->gradient || !blm->unprojected_gradient) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Dual variables don't exist yet or no longer exist.\n");

  ierr = VecCopy(tao->gradient,DXL);CHKERRQ(ierr);
  ierr = VecAXPY(DXL,-1.0,blm->unprojected_gradient);CHKERRQ(ierr);
  ierr = VecSet(DXU,0.0);CHKERRQ(ierr);
  ierr = VecPointwiseMax(DXL,DXL,DXU);CHKERRQ(ierr);

  ierr = VecCopy(blm->unprojected_gradient,DXU);CHKERRQ(ierr);
  ierr = VecAXPY(DXU,-1.0,tao->gradient);CHKERRQ(ierr);
  ierr = VecAXPY(DXU,1.0,DXL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------- */
/*MC
  TAOBLMVM - Bounded limited memory variable metric is a quasi-Newton method
         for nonlinear minimization with bound constraints. It is an extension
         of TAOLMVM

  Level: beginner
M*/
PETSC_EXTERN PetscErrorCode TaoCreate_BLMVM(Tao tao)
{
  TAO_BLMVM      *blmP;
  const char     *morethuente_type = TAOLINESEARCHMT;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  tao->ops->setup = TaoSetup_BLMVM;
  tao->ops->solve = TaoSolve_BLMVM;
  tao->ops->view = TaoView_BLMVM;
  tao->ops->setfromoptions = TaoSetFromOptions_BLMVM;
  tao->ops->destroy = TaoDestroy_BLMVM;
  tao->ops->computedual = TaoComputeDual_BLMVM;

  ierr = PetscNewLog(tao,&blmP);CHKERRQ(ierr);
  blmP->H0 = NULL;
  blmP->as_step = 0.001;
  blmP->as_tol = 0.001;
  blmP->as_type = BLMVM_AS_BERTSEKAS;
  blmP->recycle = PETSC_FALSE;
  
  tao->data = (void*)blmP;

  /* Override default settings (unless already changed) */
  if (!tao->max_it_changed) tao->max_it = 2000;
  if (!tao->max_funcs_changed) tao->max_funcs = 4000;

  ierr = TaoLineSearchCreate(((PetscObject)tao)->comm, &tao->linesearch);CHKERRQ(ierr);
  ierr = PetscObjectIncrementTabLevel((PetscObject)tao->linesearch, (PetscObject)tao, 1);CHKERRQ(ierr);
  ierr = TaoLineSearchSetType(tao->linesearch, morethuente_type);CHKERRQ(ierr);
  ierr = TaoLineSearchUseTaoRoutines(tao->linesearch,tao);CHKERRQ(ierr);
  ierr = TaoLineSearchSetOptionsPrefix(tao->linesearch,tao->hdr.prefix);CHKERRQ(ierr);
  
  ierr = KSPInitializePackage();CHKERRQ(ierr);
  ierr = MatCreate(((PetscObject)tao)->comm, &blmP->M);CHKERRQ(ierr);
  ierr = PetscObjectIncrementTabLevel((PetscObject)blmP->M, (PetscObject)tao, 1);CHKERRQ(ierr);
  ierr = MatSetType(blmP->M, MATLBFGS);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(blmP->M, "tao_blmvm_");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode TaoBLMVMSetH0(Tao tao, Mat H0)
{
  TAO_LMVM       *lmP;
  TAO_BLMVM      *blmP;
  TaoType        type;
  PetscBool      is_lmvm, is_blmvm;
  PetscErrorCode ierr;

  ierr = TaoGetType(tao, &type);CHKERRQ(ierr);
  ierr = PetscStrcmp(type, TAOLMVM,  &is_lmvm);CHKERRQ(ierr);
  ierr = PetscStrcmp(type, TAOBLMVM, &is_blmvm);CHKERRQ(ierr);

  if (is_lmvm) {
    lmP = (TAO_LMVM *)tao->data;
    ierr = PetscObjectReference((PetscObject)H0);CHKERRQ(ierr);
    lmP->H0 = H0;
  } else if (is_blmvm) {
    blmP = (TAO_BLMVM *)tao->data;
    ierr = PetscObjectReference((PetscObject)H0);CHKERRQ(ierr);
    blmP->H0 = H0;
  } else SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_WRONGSTATE, "This routine applies to TAO_LMVM and TAO_BLMVM.");
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode TaoBLMVMGetH0(Tao tao, Mat *H0)
{
  TAO_LMVM       *lmP;
  TAO_BLMVM      *blmP;
  TaoType        type;
  PetscBool      is_lmvm, is_blmvm;
  Mat            M;

  PetscErrorCode ierr;

  ierr = TaoGetType(tao, &type);CHKERRQ(ierr);
  ierr = PetscStrcmp(type, TAOLMVM,  &is_lmvm);CHKERRQ(ierr);
  ierr = PetscStrcmp(type, TAOBLMVM, &is_blmvm);CHKERRQ(ierr);

  if (is_lmvm) {
    lmP = (TAO_LMVM *)tao->data;
    M = lmP->M;
  } else if (is_blmvm) {
    blmP = (TAO_BLMVM *)tao->data;
    M = blmP->M;
  } else SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_WRONGSTATE, "This routine applies to TAO_LMVM and TAO_BLMVM.");
  ierr = MatLMVMGetJ0(M, H0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode TaoBLMVMGetH0KSP(Tao tao, KSP *ksp)
{
  TAO_LMVM       *lmP;
  TAO_BLMVM      *blmP;
  TaoType        type;
  PetscBool      is_lmvm, is_blmvm;
  Mat            M;
  PetscErrorCode ierr;

  ierr = TaoGetType(tao, &type);CHKERRQ(ierr);
  ierr = PetscStrcmp(type, TAOLMVM,  &is_lmvm);CHKERRQ(ierr);
  ierr = PetscStrcmp(type, TAOBLMVM, &is_blmvm);CHKERRQ(ierr);

  if (is_lmvm) {
    lmP = (TAO_LMVM *)tao->data;
    M = lmP->M;
  } else if (is_blmvm) {
    blmP = (TAO_BLMVM *)tao->data;
    M = blmP->M;
  } else SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_WRONGSTATE, "This routine applies to TAO_LMVM and TAO_BLMVM.");
  ierr = MatLMVMGetJ0KSP(M, ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
