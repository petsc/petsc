#define PETSC_DLL

#include "petsc.h"
#include "petscdraw.h"
#include "petscsys.h"

extern PetscLogEvent PETSC_DLLEXPORT PETSC_Barrier;

#undef __FUNCT__  
#define __FUNCT__ "PetscInitializePackage" 
/*@C
  PetscInitializePackage - This function initializes everything in the main Petsc package. It is called
  from PetscDLLibraryRegister() when using dynamic libraries, and on the call to PetscInitialize()
  when using static libraries.

  Input Parameter:
  path - The dynamic library path, or PETSC_NULL

  Level: developer

.keywords: Petsc, initialize, package
.seealso: PetscInitialize()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscInitializePackage(const char path[])
{
  static PetscTruth initialized = PETSC_FALSE;
  char              logList[256];
  char              *className;
  PetscTruth        opt;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (initialized) PetscFunctionReturn(0);
  initialized = PETSC_TRUE;
  /* Register Classes */
  ierr = PetscCookieRegister("Object",&PETSC_OBJECT_COOKIE);CHKERRQ(ierr);

  /* Register Events */
  ierr = PetscLogEventRegister("PetscBarrier", PETSC_SMALLEST_COOKIE,&PETSC_Barrier);CHKERRQ(ierr);
  /* Process info exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL, "-info_exclude", logList, 256, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList, "null", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscInfoDeactivateClass(0);CHKERRQ(ierr);
    }
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL, "-log_summary_exclude", logList, 256, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList, "null", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogEventDeactivateClass(0);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PetscDLLibraryRegister_petsc" 
/*
  PetscDLLibraryRegister - This function is called when the dynamic library it is in is opened.

  This one registers all the draw and PetscViewer objects.

  Input Parameter:
  path - library path
 */
PetscErrorCode PETSC_DLLEXPORT PetscDLLibraryRegister_petsc(const char path[])
{
  PetscErrorCode ierr;

  ierr = PetscInitializeNoArguments(); if (ierr) return 1;
  PetscFunctionBegin;
  /*
      If we got here then PETSc was properly loaded
  */
  ierr = PetscInitializePackage(path);CHKERRQ(ierr);
  ierr = PetscDrawInitializePackage(path);CHKERRQ(ierr);
  ierr = PetscViewerInitializePackage(path);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MATHEMATICA)
  ierr = PetscViewerMathematicaInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif
  ierr = PetscRandomInitializePackage(path);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END


