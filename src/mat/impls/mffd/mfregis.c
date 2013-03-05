
#include <../src/mat/impls/mffd/mffdimpl.h>   /*I  "petscmat.h"   I*/

PETSC_EXTERN_C PetscErrorCode  MatCreateMFFD_DS(MatMFFD);
PETSC_EXTERN_C PetscErrorCode  MatCreateMFFD_WP(MatMFFD);

#undef __FUNCT__
#define __FUNCT__ "MatMFFDRegisterAll"
/*@C
  MatMFFDRegisterAll - Registers all of the compute-h in the MatMFFD package.

  Not Collective

  Level: developer

.keywords: MatMFFD, register, all

.seealso:  MatMFFDRegisterDestroy(), MatMFFDRegisterDynamic), MatCreateMFFD(),
           MatMFFDSetType()
@*/
PetscErrorCode  MatMFFDRegisterAll(const char *path)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  MatMFFDRegisterAllCalled = PETSC_TRUE;

  ierr = MatMFFDRegisterDynamic(MATMFFD_DS,path,"MatCreateMFFD_DS",MatCreateMFFD_DS);CHKERRQ(ierr);
  ierr = MatMFFDRegisterDynamic(MATMFFD_WP,path,"MatCreateMFFD_WP",MatCreateMFFD_WP);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

