#include <../src/tao/complementarity/impls/ssls/ssls.h>

/*------------------------------------------------------------*/
PetscErrorCode TaoSetFromOptions_SSLS(PetscOptionItems *PetscOptionsObject,Tao tao)
{
  TAO_SSLS       *ssls = (TAO_SSLS *)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"Semismooth method with a linesearch for complementarity problems");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-ssls_delta", "descent test fraction", "",ssls->delta, &ssls->delta, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-ssls_rho", "descent test power", "",ssls->rho, &ssls->rho, NULL);CHKERRQ(ierr);
  ierr = TaoLineSearchSetFromOptions(tao->linesearch);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(tao->ksp);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TaoComputeConstraints(tao, X, tao->constraints);CHKERRQ(ierr);
  ierr = VecFischer(X,tao->constraints,tao->XL,tao->XU,ssls->ff);CHKERRQ(ierr);
  ierr = VecNorm(ssls->ff,NORM_2,&ssls->merit);CHKERRQ(ierr);
  *fcn = 0.5*ssls->merit*ssls->merit;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
PetscErrorCode Tao_SSLS_FunctionGradient(TaoLineSearch ls, Vec X, PetscReal *fcn,  Vec G, void *ptr)
{
  Tao            tao = (Tao)ptr;
  TAO_SSLS       *ssls = (TAO_SSLS *)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TaoComputeConstraints(tao, X, tao->constraints);CHKERRQ(ierr);
  ierr = VecFischer(X,tao->constraints,tao->XL,tao->XU,ssls->ff);CHKERRQ(ierr);
  ierr = VecNorm(ssls->ff,NORM_2,&ssls->merit);CHKERRQ(ierr);
  *fcn = 0.5*ssls->merit*ssls->merit;

  ierr = TaoComputeJacobian(tao,tao->solution,tao->jacobian,tao->jacobian_pre);CHKERRQ(ierr);

  ierr = MatDFischer(tao->jacobian, tao->solution, tao->constraints,tao->XL, tao->XU, ssls->t1, ssls->t2,ssls->da, ssls->db);CHKERRQ(ierr);
  ierr = MatDiagonalScale(tao->jacobian,ssls->db,NULL);CHKERRQ(ierr);
  ierr = MatDiagonalSet(tao->jacobian,ssls->da,ADD_VALUES);CHKERRQ(ierr);
  ierr = MatMultTranspose(tao->jacobian,ssls->ff,G);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

