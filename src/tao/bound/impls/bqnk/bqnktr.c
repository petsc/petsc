#include <../src/tao/bound/impls/bqnk/bqnk.h>
#include <petscksp.h>

static PetscErrorCode TaoSetUp_BQNKTR(Tao tao)
{
  TAO_BNK         *bnk = (TAO_BNK*)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TaoSetUp_BQNK(tao);CHKERRQ(ierr);
  if (!bnk->is_nash && !bnk->is_stcg && !bnk->is_gltr) SETERRQ(PetscObjectComm((PetscObject)tao),PETSC_ERR_SUP,"Must use a trust-region CG method for KSP (KSPNASH, KSPSTCG, KSPGLTR)");
  PetscFunctionReturn(0);
}

/*MC
  TAOBQNKTR - Bounded Quasi-Newton-Krylov Trust Region method for nonlinear minimization with
              bound constraints. This method approximates the Hessian-vector product using a
              limited-memory quasi-Newton formula, and iteratively inverts the Hessian with a
              Krylov solver. The quasi-Newton matrix and its settings can be accessed via the
              prefix `-tao_bqnk_`. For options database, see TAOBNK

  Level: beginner
.seealso TAOBNK, TAOBQNKTR, TAOBQNKLS
M*/
PETSC_EXTERN PetscErrorCode TaoCreate_BQNKTR(Tao tao)
{
  TAO_BNK        *bnk;
  TAO_BQNK       *bqnk;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TaoCreate_BQNK(tao);CHKERRQ(ierr);
  tao->ops->setup = TaoSetUp_BQNKTR;
  bnk = (TAO_BNK*)tao->data;
  bqnk = (TAO_BQNK*)bnk->ctx;
  bqnk->solve = TaoSolve_BNTR;
  PetscFunctionReturn(0);
}
