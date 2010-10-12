#define PETSCDM_DLL

#include "private/daimpl.h"     /*I  "petscda.h"  I*/
EXTERN_C_BEGIN
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "DMRegisterAll"
/*@C
  DMRegisterAll - Registers all of the DM components in the DM package.

  Not Collective

  Input parameter:
. path - The dynamic library path

  Level: advanced

.keywords: DM, register, all
.seealso:  DMRegister(), DMRegisterDestroy(), DMRegisterDynamic()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DMRegisterAll(const char path[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  DMRegisterAllCalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

