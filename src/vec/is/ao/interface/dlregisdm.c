
#include <../src/vec/is/ao/aoimpl.h>

static PetscBool AOPackageInitialized = PETSC_FALSE;

/*@C
  AOFinalizePackage - This function finalizes everything in the AO package. It is called
  from PetscFinalize().

  Level: developer

.seealso: PetscInitialize()
@*/
PetscErrorCode  AOFinalizePackage(void)
{
  PetscFunctionBegin;
  CHKERRQ(PetscFunctionListDestroy(&AOList));
  AOPackageInitialized = PETSC_FALSE;
  AORegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
  AOInitializePackage - This function initializes everything in the AO package. It is called
  from PetscDLLibraryRegister_petscvec() when using dynamic libraries, and on the first call to AOCreate()
  when using static or shared libraries.

  Level: developer

.seealso: PetscInitialize()
@*/
PetscErrorCode  AOInitializePackage(void)
{
  char           logList[256];
  PetscBool      opt,pkg;

  PetscFunctionBegin;
  if (AOPackageInitialized) PetscFunctionReturn(0);
  AOPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  CHKERRQ(PetscClassIdRegister("Application Order",&AO_CLASSID));
  /* Register Constructors */
  CHKERRQ(AORegisterAll());
  /* Register Events */
  CHKERRQ(PetscLogEventRegister("AOPetscToApplication", AO_CLASSID,&AO_PetscToApplication));
  CHKERRQ(PetscLogEventRegister("AOApplicationToPetsc", AO_CLASSID,&AO_ApplicationToPetsc));
  /* Process Info */
  {
    PetscClassId  classids[1];

    classids[0] = AO_CLASSID;
    CHKERRQ(PetscInfoProcessClass("ao", 1, classids));
  }
  /* Process summary exclusions */
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,sizeof(logList),&opt));
  if (opt) {
    CHKERRQ(PetscStrInList("ao",logList,',',&pkg));
    if (pkg) CHKERRQ(PetscLogEventExcludeClass(AO_CLASSID));
  }
  /* Register package finalizer */
  CHKERRQ(PetscRegisterFinalize(AOFinalizePackage));
  PetscFunctionReturn(0);
}
