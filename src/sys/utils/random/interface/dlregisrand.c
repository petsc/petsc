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
PetscErrorCode PETSC_DLLEXPORT PetscRandomInitializePackage(char *path) 
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

#ifdef PETSC_USE_DYNAMIC_LIBRARIES
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PetscDLLibraryRegister_petscrandom"
/*
  PetscDLLibraryRegister - This function is called when the dynamic library it is in is opened.

  This one registers all the methods that are in the basic PETSc library.

  Input Parameter:
  path - library path
 */
PetscErrorCode PETSC_DLLEXPORT PetscDLLibraryRegister_petscrandom(char *path)
{
  PetscErrorCode ierr;

  ierr = PetscInitializeNoArguments(); if (ierr) return 1;

  PetscFunctionBegin;
  /*
      If we got here then PETSc was properly loaded
  */
  ierr = PetscRandomInitializePackage(path);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#endif /* PETSC_USE_DYNAMIC_LIBRARIES */


