#include <../src/tao/bound/impls/bqnk/bqnk.h>

PETSC_EXTERN PetscErrorCode TaoCreate_BQNKTL(Tao tao)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = TaoCreate_BQNK(tao);CHKERRQ(ierr);
  tao->ops->solve = TaoSolve_BNTL;
  PetscFunctionReturn(0);
}
