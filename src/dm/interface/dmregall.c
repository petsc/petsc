#define PETSCDM_DLL

#include "private/daimpl.h"     /*I  "petscdm.h"  I*/
EXTERN_C_BEGIN
extern PetscErrorCode  DMCreate_DA(DM);
extern PetscErrorCode  DMCreate_Composite(DM);
extern PetscErrorCode  DMCreate_Sliced(DM);
extern PetscErrorCode  DMCreate_ADDA(DM);
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
PetscErrorCode  DMRegisterAll(const char path[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  DMRegisterAllCalled = PETSC_TRUE;
  ierr = DMRegisterDynamic(DMDA,        path, "DMCreate_DA",        DMCreate_DA);CHKERRQ(ierr);
  ierr = DMRegisterDynamic(DMCOMPOSITE, path, "DMCreate_Composite", DMCreate_Composite);CHKERRQ(ierr);
  ierr = DMRegisterDynamic(DMSLICED,    path, "DMCreate_Sliced",    DMCreate_Sliced);CHKERRQ(ierr);
  ierr = DMRegisterDynamic(DMADDA,      path, "DMCreate_ADDA",      DMCreate_ADDA);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

