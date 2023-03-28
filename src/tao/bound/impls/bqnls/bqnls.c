#include <../src/tao/bound/impls/bqnk/bqnk.h>

static const char *BNK_AS[64] = {"none", "bertsekas"};

static PetscErrorCode TaoBQNLSComputeHessian(Tao tao)
{
  TAO_BNK  *bnk  = (TAO_BNK *)tao->data;
  TAO_BQNK *bqnk = (TAO_BQNK *)bnk->ctx;
  PetscReal gnorm2, delta;

  PetscFunctionBegin;
  /* Compute the initial scaling and update the approximation */
  gnorm2 = bnk->gnorm * bnk->gnorm;
  if (gnorm2 == 0.0) gnorm2 = PETSC_MACHINE_EPSILON;
  if (bnk->f == 0.0) delta = 2.0 / gnorm2;
  else delta = 2.0 * PetscAbsScalar(bnk->f) / gnorm2;
  PetscCall(MatLMVMSymBroydenSetDelta(bqnk->B, delta));
  PetscCall(MatLMVMUpdate(bqnk->B, tao->solution, bnk->unprojected_gradient));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoBQNLSComputeStep(Tao tao, PetscBool shift, KSPConvergedReason *ksp_reason, PetscInt *step_type)
{
  TAO_BNK  *bnk  = (TAO_BNK *)tao->data;
  TAO_BQNK *bqnk = (TAO_BQNK *)bnk->ctx;
  PetscInt  nupdates;

  PetscFunctionBegin;
  PetscCall(MatSolve(bqnk->B, tao->gradient, tao->stepdirection));
  PetscCall(VecScale(tao->stepdirection, -1.0));
  PetscCall(TaoBNKBoundStep(tao, bnk->as_type, tao->stepdirection));
  *ksp_reason = KSP_CONVERGED_ATOL;
  PetscCall(MatLMVMGetUpdateCount(bqnk->B, &nupdates));
  if (nupdates == 0) *step_type = BNK_SCALED_GRADIENT;
  else *step_type = BNK_BFGS;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoSetFromOptions_BQNLS(Tao tao, PetscOptionItems *PetscOptionsObject)
{
  TAO_BNK  *bnk  = (TAO_BNK *)tao->data;
  TAO_BQNK *bqnk = (TAO_BQNK *)bnk->ctx;
  PetscBool is_set, is_spd;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "Quasi-Newton-Krylov method for bound constrained optimization");
  PetscCall(PetscOptionsEList("-tao_bnk_as_type", "active set estimation method", "", BNK_AS, BNK_AS_TYPES, BNK_AS[bnk->as_type], &bnk->as_type, NULL));
  PetscCall(PetscOptionsReal("-tao_bnk_epsilon", "(developer) tolerance used when computing actual and predicted reduction", "", bnk->epsilon, &bnk->epsilon, NULL));
  PetscCall(PetscOptionsReal("-tao_bnk_as_tol", "(developer) initial tolerance used when estimating actively bounded variables", "", bnk->as_tol, &bnk->as_tol, NULL));
  PetscCall(PetscOptionsReal("-tao_bnk_as_step", "(developer) step length used when estimating actively bounded variables", "", bnk->as_step, &bnk->as_step, NULL));
  PetscCall(PetscOptionsInt("-tao_bnk_max_cg_its", "number of BNCG iterations to take for each Newton step", "", bnk->max_cg_its, &bnk->max_cg_its, NULL));
  PetscOptionsHeadEnd();

  PetscCall(TaoSetOptionsPrefix(bnk->bncg, ((PetscObject)(tao))->prefix));
  PetscCall(TaoAppendOptionsPrefix(bnk->bncg, "tao_bnk_"));
  PetscCall(TaoSetFromOptions(bnk->bncg));

  PetscCall(MatSetOptionsPrefix(bqnk->B, ((PetscObject)tao)->prefix));
  PetscCall(MatAppendOptionsPrefix(bqnk->B, "tao_bqnls_"));
  PetscCall(MatSetFromOptions(bqnk->B));
  PetscCall(MatIsSPDKnown(bqnk->B, &is_set, &is_spd));
  PetscCheck(is_set && is_spd, PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_INCOMP, "LMVM matrix must be symmetric positive-definite");
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  TAOBQNLS - Bounded Quasi-Newton Line Search method for nonlinear minimization with bound
             constraints. This method approximates the action of the inverse-Hessian with a
             limited memory quasi-Newton formula. The quasi-Newton matrix and its options are
             accessible via the prefix `-tao_bqnls_`

  Options Database Keys:
+ -tao_bnk_max_cg_its - maximum number of bounded conjugate-gradient iterations taken in each Newton loop
. -tao_bnk_as_type - active-set estimation method ("none", "bertsekas")
. -tao_bnk_epsilon - (developer) tolerance for small pred/actual ratios that trigger automatic step acceptance
. -tao_bnk_as_tol - (developer) initial tolerance used in estimating bounded active variables (-as_type bertsekas)
- -tao_bnk_as_step - (developer) trial step length used in estimating bounded active variables (-as_type bertsekas)

  Level: beginner

.seealso: `TAOBNK`
M*/
PETSC_EXTERN PetscErrorCode TaoCreate_BQNLS(Tao tao)
{
  TAO_BNK  *bnk;
  TAO_BQNK *bqnk;

  PetscFunctionBegin;
  PetscCall(TaoCreate_BQNK(tao));
  tao->ops->setfromoptions = TaoSetFromOptions_BQNLS;

  bnk                 = (TAO_BNK *)tao->data;
  bnk->update_type    = BNK_UPDATE_STEP;
  bnk->computehessian = TaoBQNLSComputeHessian;
  bnk->computestep    = TaoBQNLSComputeStep;

  bqnk        = (TAO_BQNK *)bnk->ctx;
  bqnk->solve = TaoSolve_BNLS;
  PetscCall(MatSetType(bqnk->B, MATLMVMBFGS));
  PetscFunctionReturn(PETSC_SUCCESS);
}
