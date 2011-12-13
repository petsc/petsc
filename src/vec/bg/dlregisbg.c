#include <private/bgimpl.h>

PetscClassId PETSCBG_CLASSID;

static PetscBool PetscBGPackageInitialized = PETSC_FALSE;

#undef __FUNCT__
#define __FUNCT__ "PetscBGInitializePackage"
/*@C
   PetscBGInitializePackage - Initialize BG package

   Logically Collective

   Input Arguments:
.  path - the dynamic library path or PETSC_NULL

   Level: developer

.seealso: PetscBGFinalizePackage()
@*/
PetscErrorCode PetscBGInitializePackage(const char *path)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscBGPackageInitialized) PetscFunctionReturn(0);
  PetscBGPackageInitialized = PETSC_TRUE;
  ierr = PetscClassIdRegister("Bipartite Graph",&PETSCBG_CLASSID);CHKERRQ(ierr);
  ierr = PetscRegisterFinalize(PetscBGFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscBGFinalizePackage"
/*@C
   PetscBGFinalizePackage - Finalize PetscBG package, it is called from PetscFinalize()

   Logically Collective

   Level: developer

.seealso: PetscBGInitializePackage()
@*/
PetscErrorCode PetscBGFinalizePackage(void)
{

  PetscFunctionBegin;
  PetscBGPackageInitialized = PETSC_FALSE;
  PetscFunctionReturn(0);
}
