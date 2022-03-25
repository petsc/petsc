#include <../src/tao/bound/impls/bqnk/bqnk.h>

static PetscErrorCode TaoSetUp_BQNKTL(Tao tao)
{
  KSP               ksp;
  PetscVoidFunction valid;

  PetscFunctionBegin;
  PetscCall(TaoSetUp_BQNK(tao));
  PetscCall(TaoGetKSP(tao,&ksp));
  PetscCall(PetscObjectQueryFunction((PetscObject)ksp,"KSPCGSetRadius_C",&valid));
  PetscCheck(valid,PetscObjectComm((PetscObject)tao),PETSC_ERR_SUP,"Not for KSP type %s. Must use a trust-region CG method for KSP (e.g. KSPNASH, KSPSTCG, KSPGLTR)",((PetscObject)ksp)->type_name);
  PetscFunctionReturn(0);
}

/*MC
  TAOBQNKTL - Bounded Quasi-Newton-Krylov Trust-region with Line-search fallback, for nonlinear
              minimization with bound constraints. This method approximates the Hessian-vector
              product using a limited-memory quasi-Newton formula, and iteratively inverts the
              Hessian with a Krylov solver. The quasi-Newton matrix and its settings can be
              accessed via the prefix `-tao_bqnk_`. For options database, see TAOBNK

  Level: beginner
.seealso TAOBNK, TAOBQNKTR, TAOBQNKLS
M*/
PETSC_EXTERN PetscErrorCode TaoCreate_BQNKTL(Tao tao)
{
  TAO_BNK        *bnk;
  TAO_BQNK       *bqnk;

  PetscFunctionBegin;
  PetscCall(TaoCreate_BQNK(tao));
  tao->ops->setup = TaoSetUp_BQNKTL;
  bnk = (TAO_BNK*)tao->data;
  bqnk = (TAO_BQNK*)bnk->ctx;
  bqnk->solve = TaoSolve_BNTL;
  PetscFunctionReturn(0);
}
