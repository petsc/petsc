#define PETSC_DLL

#include "src/sys/utils/random/randomimpl.h"
#if defined (PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#endif

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
PetscErrorCode PETSC_DLLEXPORT PetscRandomInitializePackage(const char path[]) 
{
  static PetscTruth initialized = PETSC_FALSE;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (initialized) PetscFunctionReturn(0);
  initialized = PETSC_TRUE;
  /* Register Classes */
  ierr = PetscLogClassRegister(&PETSC_RANDOM_COOKIE,"PetscRandom");CHKERRQ(ierr);
  /* ierr = PetscLogClassRegister(&PF_COOKIE,          "PointFunction");CHKERRQ(ierr); */
  ierr = PetscRandomRegisterAll(path);CHKERRQ(ierr);
  /* ierr = PFRegisterAll(path);CHKERRQ(ierr); */
  PetscFunctionReturn(0);
}



