#include <petsc-private/sfimpl.h>

PetscClassId PETSCSF_CLASSID;

static PetscBool PetscSFPackageInitialized = PETSC_FALSE;

#undef __FUNCT__
#define __FUNCT__ "PetscSFInitializePackage"
/*@C
   PetscSFInitializePackage - Initialize SF package

   Logically Collective

   Input Arguments:
.  path - the dynamic library path or PETSC_NULL

   Level: developer

.seealso: PetscSFFinalizePackage()
@*/
PetscErrorCode PetscSFInitializePackage(const char *path)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscSFPackageInitialized) PetscFunctionReturn(0);
  PetscSFPackageInitialized = PETSC_TRUE;
  ierr = PetscClassIdRegister("Bipartite Graph",&PETSCSF_CLASSID);CHKERRQ(ierr);
  ierr = PetscRegisterFinalize(PetscSFFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFFinalizePackage"
/*@C
   PetscSFFinalizePackage - Finalize PetscSF package, it is called from PetscFinalize()

   Logically Collective

   Level: developer

.seealso: PetscSFInitializePackage()
@*/
PetscErrorCode PetscSFFinalizePackage(void)
{

  PetscFunctionBegin;
  PetscSFPackageInitialized = PETSC_FALSE;
  PetscFunctionReturn(0);
}
