#include <../src/tao/bound/impls/bqnk/bqnk.h>

static PetscErrorCode TaoSetUp_BQNKTL(Tao tao)
{
  TAO_BNK         *bnk = (TAO_BNK*)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TaoSetUp_BQNK(tao);CHKERRQ(ierr);
  if (!bnk->is_nash && !bnk->is_stcg && !bnk->is_gltr) SETERRQ(PetscObjectComm((PetscObject)tao),PETSC_ERR_SUP,"Must use a trust-region CG method for KSP (KSPNASH, KSPSTCG, KSPGLTR)");
  PetscFunctionReturn(0);
}

/*MC
  TAOBQNKTL - Bounded Quasi-Newton-Krylov Trust-region with Line-search fallback, for nonlinear
              minimization with bound constraints. This method approximates the Hessian-vector
              product using a limited-memory quasi-Newton formula, and iteratively inverts the
              Hessian with a Krylov solver. The quasi-Newton matrix and its settings can be
              accessed via the prefix `-tao_bqnk_`

  Options Database Keys:
+ -tao_bqnk_max_cg_its - maximum number of bounded conjugate-gradient iterations taken in each Newton loop
. -tao_bqnk_init_type - trust radius initialization method ("constant", "direction", "interpolation")
. -tao_bqnk_update_type - trust radius update method ("step", "direction", "interpolation")
- -tao_bqnk_as_type - active-set estimation method ("none", "bertsekas")

  Level: beginner
M*/
PETSC_EXTERN PetscErrorCode TaoCreate_BQNKTL(Tao tao)
{
  TAO_BNK        *bnk;
  TAO_BQNK       *bqnk;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TaoCreate_BQNK(tao);CHKERRQ(ierr);
  tao->ops->setup = TaoSetUp_BQNKTL;
  bnk = (TAO_BNK*)tao->data;
  bqnk = (TAO_BQNK*)bnk->ctx;
  bqnk->solve = TaoSolve_BNTL;
  PetscFunctionReturn(0);
}
