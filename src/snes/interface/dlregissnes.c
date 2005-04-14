#define PETSCSNES_DLL

#include "petscsnes.h"

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
PetscErrorCode PETSCSNES_DLLEXPORT SNESInitializePackage(const char path[]) {
  static PetscTruth initialized = PETSC_FALSE;
  char              logList[256];
  char              *className;
  PetscTruth        opt;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (initialized) PetscFunctionReturn(0);
  initialized = PETSC_TRUE;
  /* Register Classes */
  ierr = PetscLogClassRegister(&SNES_COOKIE,         "SNES");CHKERRQ(ierr);
  ierr = PetscLogClassRegister(&MATSNESMFCTX_COOKIE, "MatSNESMFCtx");CHKERRQ(ierr);
  /* Register Constructors */
  ierr = SNESRegisterAll(path);CHKERRQ(ierr);
  /* Register Events */
  ierr = PetscLogEventRegister(&SNES_Solve,                    "SNESSolve",        SNES_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&SNES_LineSearch,               "SNESLineSearch",   SNES_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&SNES_FunctionEval,             "SNESFunctionEval", SNES_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&SNES_JacobianEval,             "SNESJacobianEval", SNES_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&MATSNESMF_Mult,                "MatMultMatrixFre", MAT_COOKIE);CHKERRQ(ierr);
  /* Process info exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL, "-log_info_exclude", logList, 256, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList, "snes", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogInfoDeactivateClass(SNES_COOKIE);CHKERRQ(ierr);
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
PetscErrorCode PETSCSNES_DLLEXPORT PetscDLLibraryRegister_petscsnes(char *path)
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
