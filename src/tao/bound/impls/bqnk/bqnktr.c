#include <../src/tao/bound/impls/bqnk/bqnk.h>
#include <petscksp.h>

PETSC_EXTERN PetscErrorCode TaoCreate_BQNKTR(Tao tao)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = TaoCreate_BQNK(tao);CHKERRQ(ierr);
  tao->ops->solve = TaoSolve_BNTR;
  PetscFunctionReturn(0);
}
