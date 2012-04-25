#include <petsc-private/threadcommimpl.h>

static PetscBool PetscThreadCommPackageInitialized = PETSC_FALSE;

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommFinalizePackage"
/*@C
   PetscThreadCommFinalizePackage - Finalize PetscThreadComm package, called from PetscFinalize()

   Logically collective

   Level: developer

.seealso: PetscThreadCommInitializePackage()
@*/
PetscErrorCode PetscThreadCommFinalizePackage(void)
{
  PetscFunctionBegin;
  PetscThreadCommPackageInitialized = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommInitializePackage"
/*@C
   PetscThreadCommInitializePackage - Initializes ThreadComm package

   Logically collective

   Input Parameter:
.  path - The dynamic library path, or PETSC_NULL

   Level: developer

.seealso: PetscThreadCommFinalizePackage()
@*/
PetscErrorCode PetscThreadCommInitializePackage(const char *path)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if(PetscThreadCommPackageInitialized) PetscFunctionReturn(0);
  PetscThreadCommPackageInitialized = PETSC_TRUE;
  ierr = PetscRegisterFinalize(PetscThreadCommFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
