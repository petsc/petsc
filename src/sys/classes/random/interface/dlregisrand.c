
#include <../src/sys/classes/random/randomimpl.h>

static PetscBool PetscRandomPackageInitialized = PETSC_FALSE;
/*@C
  PetscRandomFinalizePackage - This function destroys everything in the Petsc interface to the Random package. It is
  called from PetscFinalize().

  Level: developer

.seealso: PetscFinalize()
@*/
PetscErrorCode  PetscRandomFinalizePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListDestroy(&PetscRandomList);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscRandomPackageInitialized) PetscFunctionReturn(0);
  PetscRandomPackageInitialized = PETSC_TRUE;
  /* Register Class */
  ierr = PetscClassIdRegister("PetscRandom",&PETSC_RANDOM_CLASSID);CHKERRQ(ierr);
  /* Register Constructors */
  ierr = PetscRandomRegisterAll();CHKERRQ(ierr);
  /* Process Info */
  {
    PetscClassId  classids[1];

    classids[0] = PETSC_RANDOM_CLASSID;
    ierr = PetscInfoProcessClass("random", 1, classids);CHKERRQ(ierr);
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,sizeof(logList),&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrInList("random",logList,',',&pkg);CHKERRQ(ierr);
    if (pkg) {ierr = PetscLogEventExcludeClass(PETSC_RANDOM_CLASSID);CHKERRQ(ierr);}
  }
  /* Register package finalizer */
  ierr = PetscRegisterFinalize(PetscRandomFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



