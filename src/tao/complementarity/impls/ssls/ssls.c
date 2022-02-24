#include <../src/tao/complementarity/impls/ssls/ssls.h>

/*------------------------------------------------------------*/
PetscErrorCode TaoSetFromOptions_SSLS(PetscOptionItems *PetscOptionsObject,Tao tao)
{
  TAO_SSLS       *ssls = (TAO_SSLS *)tao->data;

  PetscFunctionBegin;
  CHKERRQ(PetscOptionsHead(PetscOptionsObject,"Semismooth method with a linesearch for complementarity problems"));
  CHKERRQ(PetscOptionsReal("-ssls_delta", "descent test fraction", "",ssls->delta, &ssls->delta, NULL));
  CHKERRQ(PetscOptionsReal("-ssls_rho", "descent test power", "",ssls->rho, &ssls->rho, NULL));
  CHKERRQ(TaoLineSearchSetFromOptions(tao->linesearch));
  CHKERRQ(KSPSetFromOptions(tao->ksp));
  CHKERRQ(PetscOptionsTail());
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
  Tao            tao = (Tao)ptr;
  TAO_SSLS       *ssls = (TAO_SSLS *)tao->data;

  PetscFunctionBegin;
  CHKERRQ(TaoComputeConstraints(tao, X, tao->constraints));
  CHKERRQ(VecFischer(X,tao->constraints,tao->XL,tao->XU,ssls->ff));
  CHKERRQ(VecNorm(ssls->ff,NORM_2,&ssls->merit));
  *fcn = 0.5*ssls->merit*ssls->merit;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
PetscErrorCode Tao_SSLS_FunctionGradient(TaoLineSearch ls, Vec X, PetscReal *fcn,  Vec G, void *ptr)
{
  Tao            tao = (Tao)ptr;
  TAO_SSLS       *ssls = (TAO_SSLS *)tao->data;

  PetscFunctionBegin;
  CHKERRQ(TaoComputeConstraints(tao, X, tao->constraints));
  CHKERRQ(VecFischer(X,tao->constraints,tao->XL,tao->XU,ssls->ff));
  CHKERRQ(VecNorm(ssls->ff,NORM_2,&ssls->merit));
  *fcn = 0.5*ssls->merit*ssls->merit;

  CHKERRQ(TaoComputeJacobian(tao,tao->solution,tao->jacobian,tao->jacobian_pre));

  CHKERRQ(MatDFischer(tao->jacobian, tao->solution, tao->constraints,tao->XL, tao->XU, ssls->t1, ssls->t2,ssls->da, ssls->db));
  CHKERRQ(MatDiagonalScale(tao->jacobian,ssls->db,NULL));
  CHKERRQ(MatDiagonalSet(tao->jacobian,ssls->da,ADD_VALUES));
  CHKERRQ(MatMultTranspose(tao->jacobian,ssls->ff,G));
  PetscFunctionReturn(0);
}
