
#include <../src/vec/is/ao/aoimpl.h>

static PetscBool AOPackageInitialized = PETSC_FALSE;

/*@C
  AOFinalizePackage - This function finalizes everything in the `AO` package. It is called
  from `PetscFinalize()`.

  Level: developer

.seealso: `AOInitializePackage()`, `PetscInitialize()`
@*/
PetscErrorCode AOFinalizePackage(void)
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListDestroy(&AOList));
  AOPackageInitialized = PETSC_FALSE;
  AORegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  AOInitializePackage - This function initializes everything in the `AO` package. It is called
  from `PetscDLLibraryRegister_petscvec()` when using dynamic libraries, and on the first call to `AOCreate()`
  when using static or shared libraries.

  Level: developer

.seealso: `AOFinalizePackage()`, `PetscInitialize()`
@*/
PetscErrorCode AOInitializePackage(void)
{
  char      logList[256];
  PetscBool opt, pkg;

  PetscFunctionBegin;
  if (AOPackageInitialized) PetscFunctionReturn(PETSC_SUCCESS);
  AOPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  PetscCall(PetscClassIdRegister("Application Order", &AO_CLASSID));
  /* Register Constructors */
  PetscCall(AORegisterAll());
  /* Register Events */
  PetscCall(PetscLogEventRegister("AOPetscToApplication", AO_CLASSID, &AO_PetscToApplication));
  PetscCall(PetscLogEventRegister("AOApplicationToPetsc", AO_CLASSID, &AO_ApplicationToPetsc));
  /* Process Info */
  {
    PetscClassId classids[1];

    classids[0] = AO_CLASSID;
    PetscCall(PetscInfoProcessClass("ao", 1, classids));
  }
  /* Process summary exclusions */
  PetscCall(PetscOptionsGetString(NULL, NULL, "-log_exclude", logList, sizeof(logList), &opt));
  if (opt) {
    PetscCall(PetscStrInList("ao", logList, ',', &pkg));
    if (pkg) PetscCall(PetscLogEventExcludeClass(AO_CLASSID));
  }
  /* Register package finalizer */
  PetscCall(PetscRegisterFinalize(AOFinalizePackage));
  PetscFunctionReturn(PETSC_SUCCESS);
}
