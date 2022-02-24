#include <../src/tao/bound/impls/bqnk/bqnk.h>

/*MC
  TAOBQNKLS - Bounded Quasi-Newton-Krylov Line Search method for nonlinear minimization with
              bound constraints. This method approximates the Hessian-vector product using a
              limited-memory quasi-Newton formula, and iteratively inverts the Hessian with a
              Krylov solver. The quasi-Newton matrix and its settings can be accessed via the
              prefix `-tao_bqnk_`. For options database, see TAOBNK

  Level: beginner
.seealso TAOBNK, TAOBQNKTR, TAOBQNKTL
M*/
PETSC_EXTERN PetscErrorCode TaoCreate_BQNKLS(Tao tao)
{
  TAO_BNK        *bnk;
  TAO_BQNK       *bqnk;

  PetscFunctionBegin;
  CHKERRQ(TaoCreate_BQNK(tao));
  bnk = (TAO_BNK*)tao->data;
  bnk->update_type = BNK_UPDATE_STEP;
  bqnk = (TAO_BQNK*)bnk->ctx;
  bqnk->solve = TaoSolve_BNLS;
  PetscFunctionReturn(0);
}
