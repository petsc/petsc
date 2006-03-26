#define PETSC_DLL

//#include "petsc.h"
//#include "petscsys.h"        /*I "petscsys.h" I*/
#include "src/sys/utils/random/randomimpl.h"
#if defined (PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#else
/* maybe the protypes are missing */
#if defined(PETSC_HAVE_DRAND48)
EXTERN_C_BEGIN
extern double drand48();
extern void   srand48(long);
EXTERN_C_END
#else
extern double drand48();
#endif
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
PetscErrorCode PETSCVEC_DLLEXPORT PetscRandomInitializePackage(char *path) 
{
  static PetscTruth initialized = PETSC_FALSE;
  char              logList[256];
  char              *className;
  PetscTruth        opt;
  PetscErrorCode    ierr;
  PetscInt          i;

  PetscFunctionBegin;
  if (initialized) PetscFunctionReturn(0);
  initialized = PETSC_TRUE;
  /* Register Classes */
  /* ierr = PetscLogClassRegister(&IS_COOKIE,          "Index Set");CHKERRQ(ierr); */
  ierr = PetscLogClassRegister(&PETSC_RANDOM_COOKIE,         "PetscRandom");CHKERRQ(ierr);
  /* ierr = PetscLogClassRegister(&VEC_SCATTER_COOKIE, "PetscRandom Scatter");CHKERRQ(ierr); */
  /* ierr = PetscLogClassRegister(&PF_COOKIE,          "PointFunction");CHKERRQ(ierr); */
  /* Register Constructors */

  ierr = PetscRandomRegisterAll(path);CHKERRQ(ierr);
  /* ierr = PFRegisterAll(path);CHKERRQ(ierr); */

  /* Register Events */
  /* Turn off high traffic events by default */
  /* Process info exclusions */
  /* Process summary exclusions */
  /* Special processing */
  PetscFunctionReturn(0);
}
#ifdef TMP

#ifdef PETSC_USE_DYNAMIC_LIBRARIES
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PetscDLLibraryRegister_petscvec"
/*
  PetscDLLibraryRegister - This function is called when the dynamic library it is in is opened.

  This one registers all the methods that are in the basic PETSc Vec library.

  Input Parameter:
  path - library path
 */
PetscErrorCode PETSCVEC_DLLEXPORT PetscDLLibraryRegister_petscvec(char *path)
{
  PetscErrorCode ierr;

  ierr = PetscInitializeNoArguments(); if (ierr) return 1;

  PetscFunctionBegin;
  /*
      If we got here then PETSc was properly loaded
  */
  ierr = VecInitializePackage(path);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#endif /* PETSC_USE_DYNAMIC_LIBRARIES */

#endif //TMP
