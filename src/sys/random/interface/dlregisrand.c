
#include <../src/sys/random/randomimpl.h>
#if defined (PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#endif

static PetscBool  PetscRandomPackageInitialized = PETSC_FALSE;
#undef __FUNCT__
#define __FUNCT__ "PetscRandomFinalizePackage"
/*@C
  PetscRandomFinalizePackage - This function destroys everything in the Petsc interface to the Random package. It is
  called from PetscFinalize().

  Level: developer

.keywords: Petsc, destroy, package, mathematica
.seealso: PetscFinalize()
@*/
PetscErrorCode  PetscRandomFinalizePackage(void)
{
  PetscFunctionBegin;
  PetscRandomPackageInitialized = PETSC_FALSE;
  PetscRandomList               = PETSC_NULL;
  PetscRandomRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscRandomInitializePackage"
/*@C
  PetscRandomInitializePackage - This function initializes everything in the PetscRandom package. It is called
  from PetscDLLibraryRegister() when using dynamic libraries, and on the first call to PetscRandomCreate()
  when using static libraries.

  Input Parameter:
  path - The dynamic library path, or PETSC_NULL

  Level: developer

.keywords: PetscRandom, initialize, package
.seealso: PetscInitialize()
@*/
PetscErrorCode  PetscRandomInitializePackage(const char path[])
{
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (PetscRandomPackageInitialized) PetscFunctionReturn(0);
  PetscRandomPackageInitialized = PETSC_TRUE;
  /* Register Class */
  ierr = PetscClassIdRegister("PetscRandom",&PETSC_RANDOM_CLASSID);CHKERRQ(ierr);
  ierr = PetscRandomRegisterAll(path);CHKERRQ(ierr);
  ierr = PetscRegisterFinalize(PetscRandomFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



