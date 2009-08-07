#define PETSCSNES_DLL

#include "private/snesimpl.h"

static PetscTruth SNESPackageInitialized = PETSC_FALSE;
#undef __FUNCT__  
#define __FUNCT__ "SNESFinalizePackage"
/*@C
  SNESFinalizePackage - This function destroys everything in the Petsc interface to the charactoristics package. It is
  called from PetscFinalize().

  Level: developer

.keywords: Petsc, destroy, package, mathematica
.seealso: PetscFinalize()
@*/
PetscErrorCode PETSC_DLLEXPORT SNESFinalizePackage(void) 
{
  PetscFunctionBegin;
  SNESPackageInitialized = PETSC_FALSE;
  SNESRegisterAllCalled  = PETSC_FALSE;
  SNESList               = PETSC_NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SNESInitializePackage"
/*@C
  SNESInitializePackage - This function initializes everything in the SNES package. It is called
  from PetscDLLibraryRegister() when using dynamic libraries, and on the first call to SNESCreate()
  when using static libraries.

  Input Parameter:
  path - The dynamic library path, or PETSC_NULL

  Level: developer

.keywords: SNES, initialize, package
.seealso: PetscInitialize()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESInitializePackage(const char path[]) 
{
  char              logList[256];
  char              *className;
  PetscTruth        opt;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (SNESPackageInitialized) PetscFunctionReturn(0);
  SNESPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  ierr = PetscCookieRegister("SNES",&SNES_COOKIE);CHKERRQ(ierr);
  /* Register Constructors */
  ierr = SNESRegisterAll(path);CHKERRQ(ierr);
  /* Register Events */
  ierr = PetscLogEventRegister("SNESSolve",        SNES_COOKIE,&SNES_Solve);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("SNESLineSearch",   SNES_COOKIE,&SNES_LineSearch);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("SNESFunctionEval", SNES_COOKIE,&SNES_FunctionEval);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("SNESJacobianEval", SNES_COOKIE,&SNES_JacobianEval);CHKERRQ(ierr);
  /* Process info exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL, "-info_exclude", logList, 256, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList, "snes", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscInfoDeactivateClass(SNES_COOKIE);CHKERRQ(ierr);
    }
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL, "-log_summary_exclude", logList, 256, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList, "snes", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogEventDeactivateClass(SNES_COOKIE);CHKERRQ(ierr);
    }
  }
  ierr = PetscRegisterFinalize(SNESFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#ifdef PETSC_USE_DYNAMIC_LIBRARIES
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PetscDLLibraryRegister_petscsnes"
/*
  PetscDLLibraryRegister - This function is called when the dynamic library it is in is opened.

  This registers all of the SNES methods that are in the basic PETSc libpetscsnes library.

  Input Parameter:
  path - library path

 */
PetscErrorCode PETSCSNES_DLLEXPORT PetscDLLibraryRegister_petscsnes(const char path[])
{
  PetscErrorCode ierr;

  ierr = PetscInitializeNoArguments(); if (ierr) return 1;
  PetscFunctionBegin;
  /*
      If we got here then PETSc was properly loaded
  */
  ierr = SNESInitializePackage(path);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#endif /* PETSC_USE_DYNAMIC_LIBRARIES */
