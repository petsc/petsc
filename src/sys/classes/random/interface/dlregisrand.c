
#include <../src/sys/classes/random/randomimpl.h>

static PetscBool PetscRandomPackageInitialized = PETSC_FALSE;
/*@C
  PetscRandomFinalizePackage - This function destroys everything in the Petsc interface to the Random package. It is
  called from PetscFinalize().

  Level: developer

.keywords: Petsc, destroy, package, mathematica
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
  from PetscDLLibraryRegister() when using dynamic libraries, and on the first call to PetscRandomCreate()
  when using static libraries.

  Level: developer

.keywords: PetscRandom, initialize, package
.seealso: PetscInitialize()
@*/
PetscErrorCode  PetscRandomInitializePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscRandomPackageInitialized) PetscFunctionReturn(0);
  PetscRandomPackageInitialized = PETSC_TRUE;
  /* Register Class */
  ierr = PetscClassIdRegister("PetscRandom",&PETSC_RANDOM_CLASSID);CHKERRQ(ierr);
  ierr = PetscRandomRegisterAll();CHKERRQ(ierr);
  ierr = PetscRegisterFinalize(PetscRandomFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



