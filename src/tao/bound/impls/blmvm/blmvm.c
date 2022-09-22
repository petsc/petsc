#include <petsctaolinesearch.h> /*I "petsctaolinesearch.h" I*/
#include <../src/tao/unconstrained/impls/lmvm/lmvm.h>
#include <../src/tao/bound/impls/blmvm/blmvm.h>

/*------------------------------------------------------------*/
static PetscErrorCode TaoSolve_BLMVM(Tao tao)
{
  TAO_BLMVM                   *blmP      = (TAO_BLMVM *)tao->data;
  TaoLineSearchConvergedReason ls_status = TAOLINESEARCH_CONTINUE_ITERATING;
  PetscReal                    f, fold, gdx, gnorm, gnorm2;
  PetscReal                    stepsize = 1.0, delta;

  PetscFunctionBegin;
  /*  Project initial point onto bounds */
  PetscCall(TaoComputeVariableBounds(tao));
  PetscCall(VecMedian(tao->XL, tao->solution, tao->XU, tao->solution));
  PetscCall(TaoLineSearchSetVariableBounds(tao->linesearch, tao->XL, tao->XU));

  /* Check convergence criteria */
  PetscCall(TaoComputeObjectiveAndGradient(tao, tao->solution, &f, blmP->unprojected_gradient));
  PetscCall(VecBoundGradientProjection(blmP->unprojected_gradient, tao->solution, tao->XL, tao->XU, tao->gradient));

  PetscCall(TaoGradientNorm(tao, tao->gradient, NORM_2, &gnorm));
  PetscCheck(!PetscIsInfOrNanReal(f) && !PetscIsInfOrNanReal(gnorm), PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "User provided compute function generated Inf or NaN");

  tao->reason = TAO_CONTINUE_ITERATING;
  PetscCall(TaoLogConvergenceHistory(tao, f, gnorm, 0.0, tao->ksp_its));
  PetscCall(TaoMonitor(tao, tao->niter, f, gnorm, 0.0, stepsize));
  PetscUseTypeMethod(tao, convergencetest, tao->cnvP);
  if (tao->reason != TAO_CONTINUE_ITERATING) PetscFunctionReturn(0);

  /* Set counter for gradient/reset steps */
  if (!blmP->recycle) {
    blmP->grad  = 0;
    blmP->reset = 0;
    PetscCall(MatLMVMReset(blmP->M, PETSC_FALSE));
  }

  /* Have not converged; continue with Newton method */
  while (tao->reason == TAO_CONTINUE_ITERATING) {
    /* Call general purpose update function */
    if (tao->ops->update) {
      PetscUseTypeMethod(tao, update, tao->niter, tao->user_update);
      PetscCall(TaoComputeObjectiveAndGradient(tao, tao->solution, &f, tao->gradient));
    }
    /* Compute direction */
    gnorm2 = gnorm * gnorm;
    if (gnorm2 == 0.0) gnorm2 = PETSC_MACHINE_EPSILON;
    if (f == 0.0) {
      delta = 2.0 / gnorm2;
    } else {
      delta = 2.0 * PetscAbsScalar(f) / gnorm2;
    }
    PetscCall(MatLMVMSymBroydenSetDelta(blmP->M, delta));
    PetscCall(MatLMVMUpdate(blmP->M, tao->solution, tao->gradient));
    PetscCall(MatSolve(blmP->M, blmP->unprojected_gradient, tao->stepdirection));
    PetscCall(VecBoundGradientProjection(tao->stepdirection, tao->solution, tao->XL, tao->XU, tao->gradient));

    /* Check for success (descent direction) */
    PetscCall(VecDot(blmP->unprojected_gradient, tao->gradient, &gdx));
    if (gdx <= 0) {
      /* Step is not descent or solve was not successful
         Use steepest descent direction (scaled) */
      ++blmP->grad;

      PetscCall(MatLMVMReset(blmP->M, PETSC_FALSE));
      PetscCall(MatLMVMUpdate(blmP->M, tao->solution, blmP->unprojected_gradient));
      PetscCall(MatSolve(blmP->M, blmP->unprojected_gradient, tao->stepdirection));
    }
    PetscCall(VecScale(tao->stepdirection, -1.0));

    /* Perform the linesearch */
    fold = f;
    PetscCall(VecCopy(tao->solution, blmP->Xold));
    PetscCall(VecCopy(blmP->unprojected_gradient, blmP->Gold));
    PetscCall(TaoLineSearchSetInitialStepLength(tao->linesearch, 1.0));
    PetscCall(TaoLineSearchApply(tao->linesearch, tao->solution, &f, blmP->unprojected_gradient, tao->stepdirection, &stepsize, &ls_status));
    PetscCall(TaoAddLineSearchCounts(tao));

    if (ls_status != TAOLINESEARCH_SUCCESS && ls_status != TAOLINESEARCH_SUCCESS_USER) {
      /* Linesearch failed
         Reset factors and use scaled (projected) gradient step */
      ++blmP->reset;

      f = fold;
      PetscCall(VecCopy(blmP->Xold, tao->solution));
      PetscCall(VecCopy(blmP->Gold, blmP->unprojected_gradient));

      PetscCall(MatLMVMReset(blmP->M, PETSC_FALSE));
      PetscCall(MatLMVMUpdate(blmP->M, tao->solution, blmP->unprojected_gradient));
      PetscCall(MatSolve(blmP->M, blmP->unprojected_gradient, tao->stepdirection));
      PetscCall(VecScale(tao->stepdirection, -1.0));

      /* This may be incorrect; linesearch has values for stepmax and stepmin
         that should be reset. */
      PetscCall(TaoLineSearchSetInitialStepLength(tao->linesearch, 1.0));
      PetscCall(TaoLineSearchApply(tao->linesearch, tao->solution, &f, blmP->unprojected_gradient, tao->stepdirection, &stepsize, &ls_status));
      PetscCall(TaoAddLineSearchCounts(tao));

      if (ls_status != TAOLINESEARCH_SUCCESS && ls_status != TAOLINESEARCH_SUCCESS_USER) {
        tao->reason = TAO_DIVERGED_LS_FAILURE;
        break;
      }
    }

    /* Check for converged */
    PetscCall(VecBoundGradientProjection(blmP->unprojected_gradient, tao->solution, tao->XL, tao->XU, tao->gradient));
    PetscCall(TaoGradientNorm(tao, tao->gradient, NORM_2, &gnorm));
    PetscCheck(!PetscIsInfOrNanReal(f) && !PetscIsInfOrNanReal(gnorm), PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "User provided compute function generated Not-a-Number");
    tao->niter++;
    PetscCall(TaoLogConvergenceHistory(tao, f, gnorm, 0.0, tao->ksp_its));
    PetscCall(TaoMonitor(tao, tao->niter, f, gnorm, 0.0, stepsize));
    PetscUseTypeMethod(tao, convergencetest, tao->cnvP);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSetup_BLMVM(Tao tao)
{
  TAO_BLMVM *blmP = (TAO_BLMVM *)tao->data;

  PetscFunctionBegin;
  /* Existence of tao->solution checked in TaoSetup() */
  PetscCall(VecDuplicate(tao->solution, &blmP->Xold));
  PetscCall(VecDuplicate(tao->solution, &blmP->Gold));
  PetscCall(VecDuplicate(tao->solution, &blmP->unprojected_gradient));
  if (!tao->stepdirection) PetscCall(VecDuplicate(tao->solution, &tao->stepdirection));
  if (!tao->gradient) PetscCall(VecDuplicate(tao->solution, &tao->gradient));
  /* Allocate matrix for the limited memory approximation */
  PetscCall(MatLMVMAllocate(blmP->M, tao->solution, blmP->unprojected_gradient));

  /* If the user has set a matrix to solve as the initial H0, set the options prefix here, and set up the KSP */
  if (blmP->H0) PetscCall(MatLMVMSetJ0(blmP->M, blmP->H0));
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------- */
static PetscErrorCode TaoDestroy_BLMVM(Tao tao)
{
  TAO_BLMVM *blmP = (TAO_BLMVM *)tao->data;

  PetscFunctionBegin;
  if (tao->setupcalled) {
    PetscCall(VecDestroy(&blmP->unprojected_gradient));
    PetscCall(VecDestroy(&blmP->Xold));
    PetscCall(VecDestroy(&blmP->Gold));
  }
  PetscCall(MatDestroy(&blmP->M));
  if (blmP->H0) PetscObjectDereference((PetscObject)blmP->H0);
  PetscCall(PetscFree(tao->data));
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
static PetscErrorCode TaoSetFromOptions_BLMVM(Tao tao, PetscOptionItems *PetscOptionsObject)
{
  TAO_BLMVM *blmP = (TAO_BLMVM *)tao->data;
  PetscBool  is_spd, is_set;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "Limited-memory variable-metric method for bound constrained optimization");
  PetscCall(PetscOptionsBool("-tao_blmvm_recycle", "enable recycling of the BFGS matrix between subsequent TaoSolve() calls", "", blmP->recycle, &blmP->recycle, NULL));
  PetscOptionsHeadEnd();
  PetscCall(MatSetOptionsPrefix(blmP->M, ((PetscObject)tao)->prefix));
  PetscCall(MatAppendOptionsPrefix(blmP->M, "tao_blmvm_"));
  PetscCall(MatSetFromOptions(blmP->M));
  PetscCall(MatIsSPDKnown(blmP->M, &is_set, &is_spd));
  PetscCheck(is_set && is_spd, PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_INCOMP, "LMVM matrix must be symmetric positive-definite");
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
static PetscErrorCode TaoView_BLMVM(Tao tao, PetscViewer viewer)
{
  TAO_BLMVM *lmP = (TAO_BLMVM *)tao->data;
  PetscBool  isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "Gradient steps: %" PetscInt_FMT "\n", lmP->grad));
    PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_INFO));
    PetscCall(MatView(lmP->M, viewer));
    PetscCall(PetscViewerPopFormat(viewer));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoComputeDual_BLMVM(Tao tao, Vec DXL, Vec DXU)
{
  TAO_BLMVM *blm = (TAO_BLMVM *)tao->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidHeaderSpecific(DXL, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(DXU, VEC_CLASSID, 3);
  PetscCheck(tao->gradient && blm->unprojected_gradient, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Dual variables don't exist yet or no longer exist.");

  PetscCall(VecCopy(tao->gradient, DXL));
  PetscCall(VecAXPY(DXL, -1.0, blm->unprojected_gradient));
  PetscCall(VecSet(DXU, 0.0));
  PetscCall(VecPointwiseMax(DXL, DXL, DXU));

  PetscCall(VecCopy(blm->unprojected_gradient, DXU));
  PetscCall(VecAXPY(DXU, -1.0, tao->gradient));
  PetscCall(VecAXPY(DXU, 1.0, DXL));
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
  TAO_BLMVM  *blmP;
  const char *morethuente_type = TAOLINESEARCHMT;

  PetscFunctionBegin;
  tao->ops->setup          = TaoSetup_BLMVM;
  tao->ops->solve          = TaoSolve_BLMVM;
  tao->ops->view           = TaoView_BLMVM;
  tao->ops->setfromoptions = TaoSetFromOptions_BLMVM;
  tao->ops->destroy        = TaoDestroy_BLMVM;
  tao->ops->computedual    = TaoComputeDual_BLMVM;

  PetscCall(PetscNew(&blmP));
  blmP->H0      = NULL;
  blmP->recycle = PETSC_FALSE;
  tao->data     = (void *)blmP;

  /* Override default settings (unless already changed) */
  if (!tao->max_it_changed) tao->max_it = 2000;
  if (!tao->max_funcs_changed) tao->max_funcs = 4000;

  PetscCall(TaoLineSearchCreate(((PetscObject)tao)->comm, &tao->linesearch));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)tao->linesearch, (PetscObject)tao, 1));
  PetscCall(TaoLineSearchSetType(tao->linesearch, morethuente_type));
  PetscCall(TaoLineSearchUseTaoRoutines(tao->linesearch, tao));

  PetscCall(KSPInitializePackage());
  PetscCall(MatCreate(((PetscObject)tao)->comm, &blmP->M));
  PetscCall(MatSetType(blmP->M, MATLMVMBFGS));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)blmP->M, (PetscObject)tao, 1));
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
  TAO_LMVM  *lmP;
  TAO_BLMVM *blmP;
  PetscBool  is_lmvm, is_blmvm;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)tao, TAOLMVM, &is_lmvm));
  PetscCall(PetscObjectTypeCompare((PetscObject)tao, TAOBLMVM, &is_blmvm));
  if (is_lmvm) {
    lmP          = (TAO_LMVM *)tao->data;
    lmP->recycle = flg;
  } else if (is_blmvm) {
    blmP          = (TAO_BLMVM *)tao->data;
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

.seealso: `TaoLMVMGetH0()`, `TaoLMVMGetH0KSP()`
@*/
PetscErrorCode TaoLMVMSetH0(Tao tao, Mat H0)
{
  TAO_LMVM  *lmP;
  TAO_BLMVM *blmP;
  PetscBool  is_lmvm, is_blmvm;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)tao, TAOLMVM, &is_lmvm));
  PetscCall(PetscObjectTypeCompare((PetscObject)tao, TAOBLMVM, &is_blmvm));
  if (is_lmvm) {
    lmP = (TAO_LMVM *)tao->data;
    PetscCall(PetscObjectReference((PetscObject)H0));
    lmP->H0 = H0;
  } else if (is_blmvm) {
    blmP = (TAO_BLMVM *)tao->data;
    PetscCall(PetscObjectReference((PetscObject)H0));
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

.seealso: `TaoLMVMSetH0()`, `TaoLMVMGetH0KSP()`
@*/
PetscErrorCode TaoLMVMGetH0(Tao tao, Mat *H0)
{
  TAO_LMVM  *lmP;
  TAO_BLMVM *blmP;
  PetscBool  is_lmvm, is_blmvm;
  Mat        M;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)tao, TAOLMVM, &is_lmvm));
  PetscCall(PetscObjectTypeCompare((PetscObject)tao, TAOBLMVM, &is_blmvm));
  if (is_lmvm) {
    lmP = (TAO_LMVM *)tao->data;
    M   = lmP->M;
  } else if (is_blmvm) {
    blmP = (TAO_BLMVM *)tao->data;
    M    = blmP->M;
  } else SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_WRONG, "This routine applies to TAO_LMVM and TAO_BLMVM.");
  PetscCall(MatLMVMGetJ0(M, H0));
  PetscFunctionReturn(0);
}

/*@
  TaoLMVMGetH0KSP - Get the iterative solver for applying the inverse of the QN initial Hessian

  Input Parameters:
.  tao  - the Tao solver context

  Output Parameters:
.  ksp - KSP solver context for the initial Hessian

  Level: advanced

.seealso: `TaoLMVMGetH0()`, `TaoLMVMGetH0KSP()`
@*/
PetscErrorCode TaoLMVMGetH0KSP(Tao tao, KSP *ksp)
{
  TAO_LMVM  *lmP;
  TAO_BLMVM *blmP;
  PetscBool  is_lmvm, is_blmvm;
  Mat        M;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)tao, TAOLMVM, &is_lmvm));
  PetscCall(PetscObjectTypeCompare((PetscObject)tao, TAOBLMVM, &is_blmvm));
  if (is_lmvm) {
    lmP = (TAO_LMVM *)tao->data;
    M   = lmP->M;
  } else if (is_blmvm) {
    blmP = (TAO_BLMVM *)tao->data;
    M    = blmP->M;
  } else SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_WRONG, "This routine applies to TAO_LMVM and TAO_BLMVM.");
  PetscCall(MatLMVMGetJ0KSP(M, ksp));
  PetscFunctionReturn(0);
}
