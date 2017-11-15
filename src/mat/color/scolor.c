
#include <petscmat.h>
#include <../src/mat/color/color.h>

PETSC_EXTERN PetscErrorCode MatGetColoring_Natural(Mat,MatColoringType,ISColoring*);
PETSC_EXTERN PetscErrorCode MatGetColoring_SL_Minpack(Mat,MatColoringType,ISColoring*);
PETSC_EXTERN PetscErrorCode MatGetColoring_LF_Minpack(Mat,MatColoringType,ISColoring*);
PETSC_EXTERN PetscErrorCode MatGetColoring_ID_Minpack(Mat,MatColoringType,ISColoring*);

#undef __FUNCT__
#define __FUNCT__ "MatColoringRegisterAll"
/*@C
  MatColoringRegisterAll - Registers all of the matrix coloring routines in PETSc.

  Not Collective

  Level: developer

  Adding new methods:
  To add a new method to the registry. Copy this routine and
  modify it to incorporate a call to MatColoringRegister() for
  the new method, after the current list.

  Restricting the choices: To prevent all of the methods from being
  registered and thus save memory, copy this routine and modify it to
  register a zero, instead of the function name, for those methods you
  do not wish to register.  Make sure that the replacement routine is
  linked before libpetscmat.a.

.keywords: matrix, coloring, register, all

.seealso: MatColoringRegister(), MatColoringRegisterDestroy()
@*/
PetscErrorCode MatColoringRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  MatColoringRegisterAllCalled = PETSC_TRUE;

  ierr = MatColoringRegister(MATCOLORINGNATURAL,MatGetColoring_Natural);CHKERRQ(ierr);
  ierr = MatColoringRegister(MATCOLORINGSL,     MatGetColoring_SL_Minpack);CHKERRQ(ierr);
  ierr = MatColoringRegister(MATCOLORINGLF,     MatGetColoring_LF_Minpack);CHKERRQ(ierr);
  ierr = MatColoringRegister(MATCOLORINGID,     MatGetColoring_ID_Minpack);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



