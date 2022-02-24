
#include <petsc/private/randomimpl.h>

static PetscBool PetscRandomPackageInitialized = PETSC_FALSE;
/*@C
  PetscRandomFinalizePackage - This function destroys everything in the Petsc interface to the Random package. It is
  called from PetscFinalize().

  Level: developer

.seealso: PetscFinalize()
@*/
PetscErrorCode  PetscRandomFinalizePackage(void)
{
  PetscFunctionBegin;
  CHKERRQ(PetscFunctionListDestroy(&PetscRandomList));
  PetscRandomPackageInitialized = PETSC_FALSE;
  PetscRandomRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
  PetscRandomInitializePackage - This function initializes everything in the PetscRandom package. It is called
  from PetscDLLibraryRegister_petsc() when using dynamic libraries, and on the first call to PetscRandomCreate()
  when using shared or static libraries.

  Level: developer

.seealso: PetscInitialize()
@*/
PetscErrorCode  PetscRandomInitializePackage(void)
{
  char           logList[256];
  PetscBool      opt,pkg;

  PetscFunctionBegin;
  if (PetscRandomPackageInitialized) PetscFunctionReturn(0);
  PetscRandomPackageInitialized = PETSC_TRUE;
  /* Register Class */
  CHKERRQ(PetscClassIdRegister("PetscRandom",&PETSC_RANDOM_CLASSID));
  /* Register Constructors */
  CHKERRQ(PetscRandomRegisterAll());
  /* Process Info */
  {
    PetscClassId  classids[1];

    classids[0] = PETSC_RANDOM_CLASSID;
    CHKERRQ(PetscInfoProcessClass("random", 1, classids));
  }
  /* Process summary exclusions */
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,sizeof(logList),&opt));
  if (opt) {
    CHKERRQ(PetscStrInList("random",logList,',',&pkg));
    if (pkg) CHKERRQ(PetscLogEventExcludeClass(PETSC_RANDOM_CLASSID));
  }
  /* Register package finalizer */
  CHKERRQ(PetscRegisterFinalize(PetscRandomFinalizePackage));
  PetscFunctionReturn(0);
}
