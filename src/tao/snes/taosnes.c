#include <petsc/private/taoimpl.h> /*I "petsctao.h" I*/

typedef struct {
  SNES      snes;
  PetscBool setfromoptionscalled;
} Tao_SNES;

static PetscErrorCode TaoSolve_SNES(Tao tao)
{
  Tao_SNES           *taosnes = (Tao_SNES *)tao->data;
  SNESConvergedReason reason;
  PetscInt            its;

  PetscFunctionBegin;
  /* TODO SNES fails if KSP reaches max_it, while TAO accepts whatever we got */
  PetscCall(SNESSolve(taosnes->snes, NULL, tao->solution));
  PetscCall(SNESGetConvergedReason(taosnes->snes, &reason));
  tao->reason = TaoConvergedReasonFromSNES(reason);
  PetscCall(SNESGetIterationNumber(taosnes->snes, &its));
  PetscCall(TaoSetIterationNumber(tao, its));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoDestroy_SNES(Tao tao)
{
  Tao_SNES *taosnes = (Tao_SNES *)tao->data;

  PetscFunctionBegin;
  PetscCall(SNESDestroy(&taosnes->snes));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoSNESGetSNES_C", NULL));
  PetscCall(PetscFree(tao->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TAOSNESObj(SNES snes, Vec X, PetscReal *f, PetscCtx ctx)
{
  Tao tao = (Tao)ctx;

  PetscFunctionBegin;
  PetscCall(TaoComputeObjective(tao, X, f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TAOSNESFunc(SNES snes, Vec X, Vec F, PetscCtx ctx)
{
  Tao tao = (Tao)ctx;

  PetscFunctionBegin;
  PetscCall(TaoComputeGradient(tao, X, F));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TAOSNESJac(SNES snes, Vec X, Mat A, Mat P, PetscCtx ctx)
{
  Tao tao = (Tao)ctx;

  PetscFunctionBegin;
  PetscCall(TaoComputeHessian(tao, X, A, P));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TAOSNESMonitor(SNES snes, PetscInt its, PetscReal fnorm, PetscCtx ctx)
{
  Tao       tao = (Tao)ctx;
  PetscReal obj;
  Vec       X;

  PetscFunctionBegin;
  PetscCall(SNESGetSolution(snes, &X));
  PetscCall(TaoComputeObjective(tao, X, &obj));
  PetscCall(TaoSetIterationNumber(tao, its));
  PetscCall(TaoMonitor(tao, its, obj, fnorm, 0.0, 0.0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoSetUp_SNES(Tao tao)
{
  Tao_SNES   *taosnes = (Tao_SNES *)tao->data;
  Mat         A, P;
  const char *prefix;

  PetscFunctionBegin;
  PetscCall(TaoGetOptionsPrefix(tao, &prefix));
  PetscCall(SNESSetOptionsPrefix(taosnes->snes, prefix));
  PetscCall(SNESSetSolution(taosnes->snes, tao->solution));
  PetscCall(SNESSetObjective(taosnes->snes, TAOSNESObj, tao));
  PetscCall(SNESSetFunction(taosnes->snes, NULL, TAOSNESFunc, tao));
  PetscCall(SNESMonitorSet(taosnes->snes, TAOSNESMonitor, tao, NULL));
  PetscCall(TaoGetHessian(tao, &A, &P, NULL, NULL));
  if (A) PetscCall(SNESSetJacobian(taosnes->snes, A, P, TAOSNESJac, tao));
  if (taosnes->setfromoptionscalled) PetscCall(SNESSetFromOptions(taosnes->snes));
  taosnes->setfromoptionscalled = PETSC_FALSE;
  PetscCall(SNESSetUp(taosnes->snes));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoSetFromOptions_SNES(Tao tao, PetscOptionItems PetscOptionsObject)
{
  Tao_SNES *taosnes = (Tao_SNES *)tao->data;

  PetscFunctionBegin;
  taosnes->setfromoptionscalled = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoView_SNES(Tao tao, PetscViewer viewer)
{
  Tao_SNES *taosnes = (Tao_SNES *)tao->data;

  PetscFunctionBegin;
  PetscCall(SNESView(taosnes->snes, viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoSNESGetSNES_SNES(Tao tao, SNES *snes)
{
  Tao_SNES *taosnes = (Tao_SNES *)tao->data;

  PetscFunctionBegin;
  *snes = taosnes->snes;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoSNESGetSNES - Gets the nonlinear solver used by `TAOSNES`.

  Not Collective

  Input Parameter:
. tao - the `Tao` solver

  Output Parameter:
. snes - the underlying nonlinear solver

  Level: advanced

  Notes:
  The `Tao` type must be `TAOSNES`. The returned object is owned by `tao` and must not be destroyed by the caller.

  Use `SNESGetConvergedReason()` to obtain the precise nonlinear termination reason.
  `TAOSNES` maps reasons with a `TaoConvergedReason` equivalent to that value; other reasons map to
  `TAO_CONVERGED_USER` or `TAO_DIVERGED_USER` according to their sign.
  The mapped reasons refer to the tolerances and limits of the underlying `SNES`.

.seealso: [](ch_tao), `Tao`, `TAOSNES`, `TaoGetConvergedReason()`, `TaoConvergedReasonFromSNES()`, `SNESGetType()`, `SNESGetConvergedReason()`
@*/
PetscErrorCode TaoSNESGetSNES(Tao tao, SNES *snes)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscAssertPointer(snes, 2);
  PetscUseMethod(tao, "TaoSNESGetSNES_C", (Tao, SNES *), (tao, snes));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  TAOSNES - nonlinear solver using SNES

   Level: advanced

.seealso: `TaoCreate()`, `Tao`, `TaoSetType()`, `TaoType`, `TaoSNESGetSNES()`
M*/
PETSC_EXTERN PetscErrorCode TaoCreate_SNES(Tao tao)
{
  Tao_SNES *taosnes;

  PetscFunctionBegin;
  tao->ops->destroy          = TaoDestroy_SNES;
  tao->ops->setup            = TaoSetUp_SNES;
  tao->ops->setfromoptions   = TaoSetFromOptions_SNES;
  tao->ops->view             = TaoView_SNES;
  tao->ops->solve            = TaoSolve_SNES;
  tao->uses_gradient         = PETSC_TRUE;
  tao->uses_hessian_matrices = PETSC_TRUE;

  PetscCall(PetscNew(&taosnes));
  tao->data = (void *)taosnes;
  PetscCall(SNESCreate(PetscObjectComm((PetscObject)tao), &taosnes->snes));
  {
    DM dm;

    PetscCall(TaoGetDM(tao, &dm));
    PetscCall(SNESSetDM(taosnes->snes, dm));
  }
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)taosnes->snes, (PetscObject)tao, 1));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoSNESGetSNES_C", TaoSNESGetSNES_SNES));
  PetscFunctionReturn(PETSC_SUCCESS);
}
