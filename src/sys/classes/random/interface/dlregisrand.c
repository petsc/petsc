
#include <petsc/private/randomimpl.h>

static PetscBool PetscRandomPackageInitialized = PETSC_FALSE;
/*@C
  PetscRandomFinalizePackage - This function destroys everything in the Petsc interface to the Random package. It is
  called from PetscFinalize().

  Level: developer

.seealso: `PetscFinalize()`
@*/
PetscErrorCode  PetscRandomFinalizePackage(void)
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListDestroy(&PetscRandomList));
  PetscRandomPackageInitialized = PETSC_FALSE;
  PetscRandomRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
  PetscRandomInitializePackage - This function initializes everything in the PetscRandom package. It is called
  from PetscDLLibraryRegister_petsc() when using dynamic libraries, and on the first call to PetscRandomCreate()
  when using shared or static libraries.

  Level: developer

.seealso: `PetscInitialize()`
@*/
PetscErrorCode  PetscRandomInitializePackage(void)
{
  char           logList[256];
  PetscBool      opt,pkg;

  PetscFunctionBegin;
  if (PetscRandomPackageInitialized) PetscFunctionReturn(0);
  PetscRandomPackageInitialized = PETSC_TRUE;
  /* Register Class */
  PetscCall(PetscClassIdRegister("PetscRandom",&PETSC_RANDOM_CLASSID));
  /* Register Constructors */
  PetscCall(PetscRandomRegisterAll());
  /* Process Info */
  {
    PetscClassId  classids[1];

    classids[0] = PETSC_RANDOM_CLASSID;
    PetscCall(PetscInfoProcessClass("random", 1, classids));
  }
  /* Process summary exclusions */
  PetscCall(PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,sizeof(logList),&opt));
  if (opt) {
    PetscCall(PetscStrInList("random",logList,',',&pkg));
    if (pkg) PetscCall(PetscLogEventExcludeClass(PETSC_RANDOM_CLASSID));
  }
  /* Register package finalizer */
  PetscCall(PetscRegisterFinalize(PetscRandomFinalizePackage));
  PetscFunctionReturn(0);
}
