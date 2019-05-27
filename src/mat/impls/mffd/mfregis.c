
#include <../src/mat/impls/mffd/mffdimpl.h>   /*I  "petscmat.h"   I*/

PETSC_EXTERN PetscErrorCode MatCreateMFFD_DS(MatMFFD);
PETSC_EXTERN PetscErrorCode MatCreateMFFD_WP(MatMFFD);

/*@C
  MatMFFDRegisterAll - Registers all of the compute-h in the MatMFFD package.

  Not Collective

  Level: developer

.seealso:  MatMFFDRegisterDestroy(), MatMFFDRegister(), MatCreateMFFD(),
           MatMFFDSetType()
@*/
PetscErrorCode  MatMFFDRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (MatMFFDRegisterAllCalled) PetscFunctionReturn(0);
  MatMFFDRegisterAllCalled = PETSC_TRUE;

  ierr = MatMFFDRegister(MATMFFD_DS,MatCreateMFFD_DS);CHKERRQ(ierr);
  ierr = MatMFFDRegister(MATMFFD_WP,MatCreateMFFD_WP);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

