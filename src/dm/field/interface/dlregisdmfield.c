#include <petsc/private/dmfieldimpl.h> /*I "petscdmfield.h" I*/

PetscClassId DMFIELD_CLASSID;

static PetscBool DMFieldPackageInitialized = PETSC_FALSE;

PetscBool DMFieldRegisterAllCalled;

/*@C
   DMFieldInitializePackage - Initialize `DMField` package

   Logically Collective

   Level: developer

.seealso: `DMFieldFinalizePackage()`
@*/
PetscErrorCode DMFieldInitializePackage(void)
{
  PetscFunctionBegin;
  if (DMFieldPackageInitialized) PetscFunctionReturn(PETSC_SUCCESS);
  DMFieldPackageInitialized = PETSC_TRUE;

  PetscCall(PetscClassIdRegister("Field over DM", &DMFIELD_CLASSID));
  PetscCall(DMFieldRegisterAll());
  PetscCall(PetscRegisterFinalize(DMFieldFinalizePackage));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   DMFieldFinalizePackage - Finalize `DMField` package, it is called from `PetscFinalize()`

   Logically Collective

   Level: developer

.seealso: `DMFieldInitializePackage()`
@*/
PetscErrorCode DMFieldFinalizePackage(void)
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListDestroy(&DMFieldList));
  DMFieldPackageInitialized = PETSC_FALSE;
  DMFieldRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}
