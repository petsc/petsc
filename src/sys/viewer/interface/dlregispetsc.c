#define PETSC_DLL

#include "petscdraw.h"

extern PetscLogEvent PETSC_DLLEXPORT PETSC_Barrier;

static PetscTruth PetscPackageInitialized = PETSC_FALSE;
#undef __FUNCT__  
#define __FUNCT__ "PetscFinalizePackage"
/*@C
  PetscFinalizePackage - This function destroys everything in the Petsc interface to Mathematica. It is
  called from PetscFinalize().

  Level: developer

.keywords: Petsc, destroy, package, mathematica
.seealso: PetscFinalize()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscFinalizePackage(void) 
{
  PetscFunctionBegin;
  PetscPackageInitialized = PETSC_FALSE;
  PetscFunctionReturn(0);
}

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
  char              logList[256];
  char              *className;
  PetscTruth        opt;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (PetscPackageInitialized) PetscFunctionReturn(0);
  PetscPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  ierr = PetscCookieRegister("Object",&PETSC_OBJECT_COOKIE);CHKERRQ(ierr);
  ierr = PetscCookieRegister("Container",&PETSC_CONTAINER_COOKIE);CHKERRQ(ierr);

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
  ierr = PetscRegisterFinalize(PetscFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN

#if defined(PETSC_USE_SINGLE_LIBRARY) && defined(PETSC_USE_DYNAMIC_LIBRARIES)
extern PetscErrorCode PetscDLLibraryRegister_petscvec(const char[]);
extern PetscErrorCode PetscDLLibraryRegister_petscmat(const char[]);
extern PetscErrorCode PetscDLLibraryRegister_petscdm(const char[]);
extern PetscErrorCode PetscDLLibraryRegister_petscksp(const char[]);
extern PetscErrorCode PetscDLLibraryRegister_petscsnes(const char[]);
extern PetscErrorCode PetscDLLibraryRegister_petscts(const char[]);
#endif

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
  ierr = PetscRandomInitializePackage(path);CHKERRQ(ierr);

#if defined(PETSC_USE_SINGLE_LIBRARY) && defined(PETSC_USE_DYNAMIC_LIBRARIES)
  ierr = PetscDLLibraryRegister_petscvec(path);CHKERRQ(ierr);
  ierr = PetscDLLibraryRegister_petscmat(path);CHKERRQ(ierr);
  ierr = PetscDLLibraryRegister_petscdm(path);CHKERRQ(ierr);
  ierr = PetscDLLibraryRegister_petscksp(path);CHKERRQ(ierr);
  ierr = PetscDLLibraryRegister_petscsnes(path);CHKERRQ(ierr);
  ierr = PetscDLLibraryRegister_petscts(path);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}
EXTERN_C_END


