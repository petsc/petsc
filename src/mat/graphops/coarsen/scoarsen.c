#include <petscmat.h>
#include <petsc/private/matimpl.h>

PETSC_EXTERN PetscErrorCode MatCoarsenCreate_MIS(MatCoarsen);
PETSC_EXTERN PetscErrorCode MatCoarsenCreate_HEM(MatCoarsen);
PETSC_EXTERN PetscErrorCode MatCoarsenCreate_MISK(MatCoarsen);

/*@C
  MatCoarsenRegisterAll - Registers all of the matrix Coarsen routines in PETSc.

  Not Collective

  Level: developer

.seealso: `MatCoarsen`, `MatCoarsenType`, `MatCoarsenRegister()`, `MatCoarsenRegisterDestroy()`
 @*/
PetscErrorCode MatCoarsenRegisterAll(void)
{
  PetscFunctionBegin;
  if (MatCoarsenRegisterAllCalled) PetscFunctionReturn(PETSC_SUCCESS);
  MatCoarsenRegisterAllCalled = PETSC_TRUE;

  PetscCall(MatCoarsenRegister(MATCOARSENMIS, MatCoarsenCreate_MIS));
  PetscCall(MatCoarsenRegister(MATCOARSENHEM, MatCoarsenCreate_HEM));
  PetscCall(MatCoarsenRegister(MATCOARSENMISK, MatCoarsenCreate_MISK));
  PetscFunctionReturn(PETSC_SUCCESS);
}
