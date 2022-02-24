#include <petsctaolinesearch.h>      /*I "petsctaolinesearch.h" I*/
#include <../src/tao/unconstrained/impls/lmvm/lmvm.h>
#include <../src/tao/bound/impls/blmvm/blmvm.h>

/*------------------------------------------------------------*/
static PetscErrorCode TaoSolve_BLMVM(Tao tao)
{
  TAO_BLMVM                    *blmP = (TAO_BLMVM *)tao->data;
  TaoLineSearchConvergedReason ls_status = TAOLINESEARCH_CONTINUE_ITERATING;
  PetscReal                    f, fold, gdx, gnorm, gnorm2;
  PetscReal                    stepsize = 1.0,delta;

  PetscFunctionBegin;
  /*  Project initial point onto bounds */
  CHKERRQ(TaoComputeVariableBounds(tao));
  CHKERRQ(VecMedian(tao->XL,tao->solution,tao->XU,tao->solution));
  CHKERRQ(TaoLineSearchSetVariableBounds(tao->linesearch,tao->XL,tao->XU));

  /* Check convergence criteria */
  CHKERRQ(TaoComputeObjectiveAndGradient(tao, tao->solution,&f,blmP->unprojected_gradient));
  CHKERRQ(VecBoundGradientProjection(blmP->unprojected_gradient,tao->solution, tao->XL,tao->XU,tao->gradient));

  CHKERRQ(TaoGradientNorm(tao, tao->gradient,NORM_2,&gnorm));
  PetscCheck(!PetscIsInfOrNanReal(f) && !PetscIsInfOrNanReal(gnorm),PetscObjectComm((PetscObject)tao),PETSC_ERR_USER, "User provided compute function generated Inf or NaN");

  tao->reason = TAO_CONTINUE_ITERATING;
  CHKERRQ(TaoLogConvergenceHistory(tao,f,gnorm,0.0,tao->ksp_its));
  CHKERRQ(TaoMonitor(tao,tao->niter,f,gnorm,0.0,stepsize));
  CHKERRQ((*tao->ops->convergencetest)(tao,tao->cnvP));
  if (tao->reason != TAO_CONTINUE_ITERATING) PetscFunctionReturn(0);

  /* Set counter for gradient/reset steps */
  if (!blmP->recycle) {
    blmP->grad = 0;
    blmP->reset = 0;
    CHKERRQ(MatLMVMReset(blmP->M, PETSC_FALSE));
  }

  /* Have not converged; continue with Newton method */
  while (tao->reason == TAO_CONTINUE_ITERATING) {
    /* Call general purpose update function */
    if (tao->ops->update) {
      CHKERRQ((*tao->ops->update)(tao, tao->niter, tao->user_update));
    }
    /* Compute direction */
    gnorm2 = gnorm*gnorm;
    if (gnorm2 == 0.0) gnorm2 = PETSC_MACHINE_EPSILON;
    if (f == 0.0) {
      delta = 2.0 / gnorm2;
    } else {
      delta = 2.0 * PetscAbsScalar(f) / gnorm2;
    }
    CHKERRQ(MatLMVMSymBroydenSetDelta(blmP->M, delta));
    CHKERRQ(MatLMVMUpdate(blmP->M, tao->solution, tao->gradient));
    CHKERRQ(MatSolve(blmP->M, blmP->unprojected_gradient, tao->stepdirection));
    CHKERRQ(VecBoundGradientProjection(tao->stepdirection,tao->solution,tao->XL,tao->XU,tao->gradient));

    /* Check for success (descent direction) */
    CHKERRQ(VecDot(blmP->unprojected_gradient, tao->gradient, &gdx));
    if (gdx <= 0) {
      /* Step is not descent or solve was not successful
         Use steepest descent direction (scaled) */
      ++blmP->grad;

      CHKERRQ(MatLMVMReset(blmP->M, PETSC_FALSE));
      CHKERRQ(MatLMVMUpdate(blmP->M, tao->solution, blmP->unprojected_gradient));
      CHKERRQ(MatSolve(blmP->M,blmP->unprojected_gradient, tao->stepdirection));
    }
    CHKERRQ(VecScale(tao->stepdirection,-1.0));

    /* Perform the linesearch */
    fold = f;
    CHKERRQ(VecCopy(tao->solution, blmP->Xold));
    CHKERRQ(VecCopy(blmP->unprojected_gradient, blmP->Gold));
    CHKERRQ(TaoLineSearchSetInitialStepLength(tao->linesearch,1.0));
    CHKERRQ(TaoLineSearchApply(tao->linesearch, tao->solution, &f, blmP->unprojected_gradient, tao->stepdirection, &stepsize, &ls_status));
    CHKERRQ(TaoAddLineSearchCounts(tao));

    if (ls_status != TAOLINESEARCH_SUCCESS && ls_status != TAOLINESEARCH_SUCCESS_USER) {
      /* Linesearch failed
         Reset factors and use scaled (projected) gradient step */
      ++blmP->reset;

      f = fold;
      CHKERRQ(VecCopy(blmP->Xold, tao->solution));
      CHKERRQ(VecCopy(blmP->Gold, blmP->unprojected_gradient));

      CHKERRQ(MatLMVMReset(blmP->M, PETSC_FALSE));
      CHKERRQ(MatLMVMUpdate(blmP->M, tao->solution, blmP->unprojected_gradient));
      CHKERRQ(MatSolve(blmP->M, blmP->unprojected_gradient, tao->stepdirection));
      CHKERRQ(VecScale(tao->stepdirection, -1.0));

      /* This may be incorrect; linesearch has values for stepmax and stepmin
         that should be reset. */
      CHKERRQ(TaoLineSearchSetInitialStepLength(tao->linesearch,1.0));
      CHKERRQ(TaoLineSearchApply(tao->linesearch,tao->solution,&f, blmP->unprojected_gradient, tao->stepdirection,  &stepsize, &ls_status));
      CHKERRQ(TaoAddLineSearchCounts(tao));

      if (ls_status != TAOLINESEARCH_SUCCESS && ls_status != TAOLINESEARCH_SUCCESS_USER) {
        tao->reason = TAO_DIVERGED_LS_FAILURE;
        break;
      }
    }

    /* Check for converged */
    CHKERRQ(VecBoundGradientProjection(blmP->unprojected_gradient, tao->solution, tao->XL, tao->XU, tao->gradient));
    CHKERRQ(TaoGradientNorm(tao, tao->gradient, NORM_2, &gnorm));
    PetscCheck(!PetscIsInfOrNanReal(f) && !PetscIsInfOrNanReal(gnorm),PetscObjectComm((PetscObject)tao),PETSC_ERR_USER, "User provided compute function generated Not-a-Number");
    tao->niter++;
    CHKERRQ(TaoLogConvergenceHistory(tao,f,gnorm,0.0,tao->ksp_its));
    CHKERRQ(TaoMonitor(tao,tao->niter,f,gnorm,0.0,stepsize));
    CHKERRQ((*tao->ops->convergencetest)(tao,tao->cnvP));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSetup_BLMVM(Tao tao)
{
  TAO_BLMVM      *blmP = (TAO_BLMVM *)tao->data;

  PetscFunctionBegin;
  /* Existence of tao->solution checked in TaoSetup() */
  CHKERRQ(VecDuplicate(tao->solution,&blmP->Xold));
  CHKERRQ(VecDuplicate(tao->solution,&blmP->Gold));
  CHKERRQ(VecDuplicate(tao->solution, &blmP->unprojected_gradient));

  if (!tao->stepdirection) {
    CHKERRQ(VecDuplicate(tao->solution, &tao->stepdirection));
  }
  if (!tao->gradient) {
    CHKERRQ(VecDuplicate(tao->solution,&tao->gradient));
  }
  if (!tao->XL) {
    CHKERRQ(VecDuplicate(tao->solution,&tao->XL));
    CHKERRQ(VecSet(tao->XL,PETSC_NINFINITY));
  }
  if (!tao->XU) {
    CHKERRQ(VecDuplicate(tao->solution,&tao->XU));
    CHKERRQ(VecSet(tao->XU,PETSC_INFINITY));
  }
  /* Allocate matrix for the limited memory approximation */
  CHKERRQ(MatLMVMAllocate(blmP->M,tao->solution,blmP->unprojected_gradient));

  /* If the user has set a matrix to solve as the initial H0, set the options prefix here, and set up the KSP */
  if (blmP->H0) {
    CHKERRQ(MatLMVMSetJ0(blmP->M, blmP->H0));
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------- */
static PetscErrorCode TaoDestroy_BLMVM(Tao tao)
{
  TAO_BLMVM      *blmP = (TAO_BLMVM *)tao->data;

  PetscFunctionBegin;
  if (tao->setupcalled) {
    CHKERRQ(VecDestroy(&blmP->unprojected_gradient));
    CHKERRQ(VecDestroy(&blmP->Xold));
    CHKERRQ(VecDestroy(&blmP->Gold));
  }
  CHKERRQ(MatDestroy(&blmP->M));
  if (blmP->H0) {
    PetscObjectDereference((PetscObject)blmP->H0);
  }
  CHKERRQ(PetscFree(tao->data));
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
static PetscErrorCode TaoSetFromOptions_BLMVM(PetscOptionItems* PetscOptionsObject,Tao tao)
{
  TAO_BLMVM      *blmP = (TAO_BLMVM *)tao->data;
  PetscBool      is_spd;

  PetscFunctionBegin;
  CHKERRQ(PetscOptionsHead(PetscOptionsObject,"Limited-memory variable-metric method for bound constrained optimization"));
  CHKERRQ(PetscOptionsBool("-tao_blmvm_recycle","enable recycling of the BFGS matrix between subsequent TaoSolve() calls","",blmP->recycle,&blmP->recycle,NULL));
  CHKERRQ(PetscOptionsTail());
  CHKERRQ(MatSetOptionsPrefix(blmP->M, ((PetscObject)tao)->prefix));
  CHKERRQ(MatAppendOptionsPrefix(blmP->M, "tao_blmvm_"));
  CHKERRQ(MatSetFromOptions(blmP->M));
  CHKERRQ(MatGetOption(blmP->M, MAT_SPD, &is_spd));
  PetscCheck(is_spd,PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_INCOMP, "LMVM matrix must be symmetric positive-definite");
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
static PetscErrorCode TaoView_BLMVM(Tao tao, PetscViewer viewer)
{
  TAO_BLMVM      *lmP = (TAO_BLMVM *)tao->data;
  PetscBool      isascii;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer, "Gradient steps: %D\n", lmP->grad));
    CHKERRQ(PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_INFO));
    CHKERRQ(MatView(lmP->M, viewer));
    CHKERRQ(PetscViewerPopFormat(viewer));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoComputeDual_BLMVM(Tao tao, Vec DXL, Vec DXU)
{
  TAO_BLMVM      *blm = (TAO_BLMVM *) tao->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidHeaderSpecific(DXL,VEC_CLASSID,2);
  PetscValidHeaderSpecific(DXU,VEC_CLASSID,3);
  PetscCheck(tao->gradient && blm->unprojected_gradient,PETSC_COMM_SELF,PETSC_ERR_ORDER,"Dual variables don't exist yet or no longer exist.");

  CHKERRQ(VecCopy(tao->gradient,DXL));
  CHKERRQ(VecAXPY(DXL,-1.0,blm->unprojected_gradient));
  CHKERRQ(VecSet(DXU,0.0));
  CHKERRQ(VecPointwiseMax(DXL,DXL,DXU));

  CHKERRQ(VecCopy(blm->unprojected_gradient,DXU));
  CHKERRQ(VecAXPY(DXU,-1.0,tao->gradient));
  CHKERRQ(VecAXPY(DXU,1.0,DXL));
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------- */
/*MC
  TAOBLMVM - Bounded limited memory variable metric is a quasi-Newton method
         for nonlinear minimization with bound constraints. It is an extension
         of TAOLMVM

  Options Database Keys:
.     -tao_lmm_recycle - enable recycling of LMVM information between subsequent TaoSolve calls

  Level: beginner
M*/
PETSC_EXTERN PetscErrorCode TaoCreate_BLMVM(Tao tao)
{
  TAO_BLMVM      *blmP;
  const char     *morethuente_type = TAOLINESEARCHMT;

  PetscFunctionBegin;
  tao->ops->setup = TaoSetup_BLMVM;
  tao->ops->solve = TaoSolve_BLMVM;
  tao->ops->view = TaoView_BLMVM;
  tao->ops->setfromoptions = TaoSetFromOptions_BLMVM;
  tao->ops->destroy = TaoDestroy_BLMVM;
  tao->ops->computedual = TaoComputeDual_BLMVM;

  CHKERRQ(PetscNewLog(tao,&blmP));
  blmP->H0 = NULL;
  blmP->recycle = PETSC_FALSE;
  tao->data = (void*)blmP;

  /* Override default settings (unless already changed) */
  if (!tao->max_it_changed) tao->max_it = 2000;
  if (!tao->max_funcs_changed) tao->max_funcs = 4000;

  CHKERRQ(TaoLineSearchCreate(((PetscObject)tao)->comm, &tao->linesearch));
  CHKERRQ(PetscObjectIncrementTabLevel((PetscObject)tao->linesearch, (PetscObject)tao, 1));
  CHKERRQ(TaoLineSearchSetType(tao->linesearch, morethuente_type));
  CHKERRQ(TaoLineSearchUseTaoRoutines(tao->linesearch,tao));

  CHKERRQ(KSPInitializePackage());
  CHKERRQ(MatCreate(((PetscObject)tao)->comm, &blmP->M));
  CHKERRQ(MatSetType(blmP->M, MATLMVMBFGS));
  CHKERRQ(PetscObjectIncrementTabLevel((PetscObject)blmP->M, (PetscObject)tao, 1));
  PetscFunctionReturn(0);
}

/*@
  TaoLMVMRecycle - Enable/disable recycling of the QN history between subsequent TaoSolve calls.

  Input Parameters:
+  tao  - the Tao solver context
-  flg - Boolean flag for recycling (PETSC_TRUE or PETSC_FALSE)

  Level: intermediate
@*/
PetscErrorCode TaoLMVMRecycle(Tao tao, PetscBool flg)
{
  TAO_LMVM       *lmP;
  TAO_BLMVM      *blmP;
  PetscBool      is_lmvm, is_blmvm;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)tao,TAOLMVM,&is_lmvm));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)tao,TAOBLMVM,&is_blmvm));
  if (is_lmvm) {
    lmP = (TAO_LMVM *)tao->data;
    lmP->recycle = flg;
  } else if (is_blmvm) {
    blmP = (TAO_BLMVM *)tao->data;
    blmP->recycle = flg;
  }
  PetscFunctionReturn(0);
}

/*@
  TaoLMVMSetH0 - Set the initial Hessian for the QN approximation

  Input Parameters:
+  tao  - the Tao solver context
-  H0 - Mat object for the initial Hessian

  Level: advanced

.seealso: TaoLMVMGetH0(), TaoLMVMGetH0KSP()
@*/
PetscErrorCode TaoLMVMSetH0(Tao tao, Mat H0)
{
  TAO_LMVM       *lmP;
  TAO_BLMVM      *blmP;
  PetscBool      is_lmvm, is_blmvm;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)tao,TAOLMVM,&is_lmvm));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)tao,TAOBLMVM,&is_blmvm));
  if (is_lmvm) {
    lmP = (TAO_LMVM *)tao->data;
    CHKERRQ(PetscObjectReference((PetscObject)H0));
    lmP->H0 = H0;
  } else if (is_blmvm) {
    blmP = (TAO_BLMVM *)tao->data;
    CHKERRQ(PetscObjectReference((PetscObject)H0));
    blmP->H0 = H0;
  }
  PetscFunctionReturn(0);
}

/*@
  TaoLMVMGetH0 - Get the matrix object for the QN initial Hessian

  Input Parameters:
.  tao  - the Tao solver context

  Output Parameters:
.  H0 - Mat object for the initial Hessian

  Level: advanced

.seealso: TaoLMVMSetH0(), TaoLMVMGetH0KSP()
@*/
PetscErrorCode TaoLMVMGetH0(Tao tao, Mat *H0)
{
  TAO_LMVM       *lmP;
  TAO_BLMVM      *blmP;
  PetscBool      is_lmvm, is_blmvm;
  Mat            M;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)tao,TAOLMVM,&is_lmvm));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)tao,TAOBLMVM,&is_blmvm));
  if (is_lmvm) {
    lmP = (TAO_LMVM *)tao->data;
    M = lmP->M;
  } else if (is_blmvm) {
    blmP = (TAO_BLMVM *)tao->data;
    M = blmP->M;
  } else SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_WRONG, "This routine applies to TAO_LMVM and TAO_BLMVM.");
  CHKERRQ(MatLMVMGetJ0(M, H0));
  PetscFunctionReturn(0);
}

/*@
  TaoLMVMGetH0KSP - Get the iterative solver for applying the inverse of the QN initial Hessian

  Input Parameters:
.  tao  - the Tao solver context

  Output Parameters:
.  ksp - KSP solver context for the initial Hessian

  Level: advanced

.seealso: TaoLMVMGetH0(), TaoLMVMGetH0KSP()
@*/
PetscErrorCode TaoLMVMGetH0KSP(Tao tao, KSP *ksp)
{
  TAO_LMVM       *lmP;
  TAO_BLMVM      *blmP;
  PetscBool      is_lmvm, is_blmvm;
  Mat            M;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)tao,TAOLMVM,&is_lmvm));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)tao,TAOBLMVM,&is_blmvm));
  if (is_lmvm) {
    lmP = (TAO_LMVM *)tao->data;
    M = lmP->M;
  } else if (is_blmvm) {
    blmP = (TAO_BLMVM *)tao->data;
    M = blmP->M;
  } else SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_WRONG, "This routine applies to TAO_LMVM and TAO_BLMVM.");
  CHKERRQ(MatLMVMGetJ0KSP(M, ksp));
  PetscFunctionReturn(0);
}
