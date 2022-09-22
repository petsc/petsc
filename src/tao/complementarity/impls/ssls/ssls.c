#include <../src/tao/complementarity/impls/ssls/ssls.h>

/*------------------------------------------------------------*/
PetscErrorCode TaoSetFromOptions_SSLS(Tao tao, PetscOptionItems *PetscOptionsObject)
{
  TAO_SSLS *ssls = (TAO_SSLS *)tao->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "Semismooth method with a linesearch for complementarity problems");
  PetscCall(PetscOptionsReal("-ssls_delta", "descent test fraction", "", ssls->delta, &ssls->delta, NULL));
  PetscCall(PetscOptionsReal("-ssls_rho", "descent test power", "", ssls->rho, &ssls->rho, NULL));
  PetscCall(TaoLineSearchSetFromOptions(tao->linesearch));
  PetscCall(KSPSetFromOptions(tao->ksp));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
PetscErrorCode TaoView_SSLS(Tao tao, PetscViewer pv)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
PetscErrorCode Tao_SSLS_Function(TaoLineSearch ls, Vec X, PetscReal *fcn, void *ptr)
{
  Tao       tao  = (Tao)ptr;
  TAO_SSLS *ssls = (TAO_SSLS *)tao->data;

  PetscFunctionBegin;
  PetscCall(TaoComputeConstraints(tao, X, tao->constraints));
  PetscCall(VecFischer(X, tao->constraints, tao->XL, tao->XU, ssls->ff));
  PetscCall(VecNorm(ssls->ff, NORM_2, &ssls->merit));
  *fcn = 0.5 * ssls->merit * ssls->merit;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
PetscErrorCode Tao_SSLS_FunctionGradient(TaoLineSearch ls, Vec X, PetscReal *fcn, Vec G, void *ptr)
{
  Tao       tao  = (Tao)ptr;
  TAO_SSLS *ssls = (TAO_SSLS *)tao->data;

  PetscFunctionBegin;
  PetscCall(TaoComputeConstraints(tao, X, tao->constraints));
  PetscCall(VecFischer(X, tao->constraints, tao->XL, tao->XU, ssls->ff));
  PetscCall(VecNorm(ssls->ff, NORM_2, &ssls->merit));
  *fcn = 0.5 * ssls->merit * ssls->merit;

  PetscCall(TaoComputeJacobian(tao, tao->solution, tao->jacobian, tao->jacobian_pre));

  PetscCall(MatDFischer(tao->jacobian, tao->solution, tao->constraints, tao->XL, tao->XU, ssls->t1, ssls->t2, ssls->da, ssls->db));
  PetscCall(MatDiagonalScale(tao->jacobian, ssls->db, NULL));
  PetscCall(MatDiagonalSet(tao->jacobian, ssls->da, ADD_VALUES));
  PetscCall(MatMultTranspose(tao->jacobian, ssls->ff, G));
  PetscFunctionReturn(0);
}
