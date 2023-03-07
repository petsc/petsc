#include <petsc/private/taoimpl.h> /*I "petsctao.h" I*/

typedef struct {
  SNES snes;
} Tao_SNES;

static PetscErrorCode TaoSolve_SNES(Tao tao)
{
  Tao_SNES *taosnes = (Tao_SNES *)tao->data;

  PetscFunctionBegin;
  PetscCall(SNESSolve(taosnes->snes, NULL, tao->solution));
  /* TODO REASONS */
  tao->reason = TAO_CONVERGED_USER;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoDestroy_SNES(Tao tao)
{
  Tao_SNES *taosnes = (Tao_SNES *)tao->data;

  PetscFunctionBegin;
  PetscCall(SNESDestroy(&taosnes->snes));
  PetscCall(PetscFree(tao->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TAOSNESObj(SNES snes, Vec X, PetscReal *f, void *ctx)
{
  Tao tao = (Tao)ctx;

  PetscFunctionBegin;
  PetscCall(TaoComputeObjective(tao, X, f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TAOSNESFunc(SNES snes, Vec X, Vec F, void *ctx)
{
  Tao tao = (Tao)ctx;

  PetscFunctionBegin;
  PetscCall(TaoComputeGradient(tao, X, F));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TAOSNESJac(SNES snes, Vec X, Mat A, Mat P, void *ctx)
{
  Tao tao = (Tao)ctx;

  PetscFunctionBegin;
  PetscCall(TaoComputeHessian(tao, X, A, P));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoSetUp_SNES(Tao tao)
{
  Tao_SNES *taosnes = (Tao_SNES *)tao->data;
  Mat       A, P;

  PetscFunctionBegin;
  PetscCall(SNESSetSolution(taosnes->snes, tao->solution));
  PetscCall(SNESSetObjective(taosnes->snes, TAOSNESObj, tao));
  PetscCall(SNESSetFunction(taosnes->snes, NULL, TAOSNESFunc, tao));
  PetscCall(TaoGetHessian(tao, &A, &P, NULL, NULL));
  if (A) PetscCall(SNESSetJacobian(taosnes->snes, A, P, TAOSNESJac, tao));
  /* TODO TYPES */
  PetscCall(SNESSetUp(taosnes->snes));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoSetFromOptions_SNES(Tao tao, PetscOptionItems *PetscOptionsObject)
{
  Tao_SNES *taosnes = (Tao_SNES *)tao->data;

  PetscFunctionBegin;
  PetscCall(SNESSetFromOptions(taosnes->snes));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TaoView_SNES(Tao tao, PetscViewer viewer)
{
  Tao_SNES *taosnes = (Tao_SNES *)tao->data;

  PetscFunctionBegin;
  PetscCall(SNESView(taosnes->snes, viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  TAOSNES - nonlinear solver using SNES

   Level: advanced

.seealso: `TaoCreate()`, `Tao`, `TaoSetType()`, `TaoType`
M*/
PETSC_EXTERN PetscErrorCode TaoCreate_SNES(Tao tao)
{
  Tao_SNES *taosnes;

  PetscFunctionBegin;
  tao->ops->destroy        = TaoDestroy_SNES;
  tao->ops->setup          = TaoSetUp_SNES;
  tao->ops->setfromoptions = TaoSetFromOptions_SNES;
  tao->ops->view           = TaoView_SNES;
  tao->ops->solve          = TaoSolve_SNES;

  PetscCall(PetscNew(&taosnes));
  tao->data = (void *)taosnes;
  PetscCall(SNESCreate(PetscObjectComm((PetscObject)tao), &taosnes->snes));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)taosnes->snes, (PetscObject)tao, 1));
  PetscFunctionReturn(PETSC_SUCCESS);
}
