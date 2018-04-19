#include <petsc/private/dmfieldimpl.h> /*I "petscdmfield.h" I*/

PetscClassId DMFIELD_CLASSID;

static PetscBool DMFieldPackageInitialized = PETSC_FALSE;

PetscBool DMFieldRegisterAllCalled;

/*@C
   DMFieldInitializePackage - Initialize DMField package

   Logically Collective

   Level: developer

.seealso: DMFieldFinalizePackage()
@*/
PetscErrorCode DMFieldInitializePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (DMFieldPackageInitialized) PetscFunctionReturn(0);
  DMFieldPackageInitialized = PETSC_TRUE;

  ierr = PetscClassIdRegister("Field over DM",&DMFIELD_CLASSID);CHKERRQ(ierr);
  ierr = DMFieldRegisterAll();CHKERRQ(ierr);
  ierr = PetscRegisterFinalize(DMFieldFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   DMFieldFinalizePackage - Finalize DMField package, it is called from PetscFinalize()

   Logically Collective

   Level: developer

.seealso: DMFieldInitializePackage()
@*/
PetscErrorCode DMFieldFinalizePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListDestroy(&DMFieldList);CHKERRQ(ierr);
  DMFieldPackageInitialized = PETSC_FALSE;
  DMFieldRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(0);
}
