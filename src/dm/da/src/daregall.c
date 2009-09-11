#define PETSCDM_DLL

#include "private/daimpl.h"     /*I  "petscda.h"  I*/
EXTERN_C_BEGIN
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT DACreate_1D(DA);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT DACreate_2D(DA);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT DACreate_3D(DA);
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "DARegisterAll"
/*@C
  DARegisterAll - Registers all of the DA components in the DA package.

  Not Collective

  Input parameter:
. path - The dynamic library path

  Level: advanced

.keywords: DA, register, all
.seealso:  DARegister(), DARegisterDestroy(), DARegisterDynamic()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DARegisterAll(const char path[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  DARegisterAllCalled = PETSC_TRUE;

  ierr = DARegisterDynamic(DA1D, path, "DACreate_1D", DACreate_1D);CHKERRQ(ierr);
  ierr = DARegisterDynamic(DA2D, path, "DACreate_2D", DACreate_2D);CHKERRQ(ierr);
  ierr = DARegisterDynamic(DA3D, path, "DACreate_3D", DACreate_3D);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

