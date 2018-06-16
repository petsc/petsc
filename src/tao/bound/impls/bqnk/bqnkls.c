#include <../src/tao/bound/impls/bqnk/bqnk.h>

PETSC_EXTERN PetscErrorCode TaoCreate_BQNKLS(Tao tao)
{
  TAO_BNK        *bnk;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = TaoCreate_BQNK(tao);CHKERRQ(ierr);
  tao->ops->solve = TaoSolve_BNLS;
  bnk = (TAO_BNK*)tao->data;
  bnk->update_type = BNK_UPDATE_STEP;
  PetscFunctionReturn(0);
}
