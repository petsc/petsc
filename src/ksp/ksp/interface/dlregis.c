#define PETSCKSP_DLL

#include "petscksp.h"

#undef __FUNCT__  
#define __FUNCT__ "PCInitializePackage"
/*@C
  PCInitializePackage - This function initializes everything in the PC package. It is called
  from PetscDLLibraryRegister() when using dynamic libraries, and on the first call to PCCreate()
  when using static libraries.

  Input Parameter:
  path - The dynamic library path, or PETSC_NULL

  Level: developer

.keywords: PC, initialize, package
.seealso: PetscInitialize()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCInitializePackage(const char path[]) {
  static PetscTruth initialized = PETSC_FALSE;
  char              logList[256];
  char             *className;
  PetscTruth        opt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (initialized) PetscFunctionReturn(0);
  initialized = PETSC_TRUE;
  /* Register Classes */
  ierr = PetscLogClassRegister(&PC_COOKIE,   "Preconditioner");CHKERRQ(ierr);
  /* Register Constructors */
  ierr = PCRegisterAll(path);CHKERRQ(ierr);
  /* Register Events */
  ierr = PetscLogEventRegister(&PC_SetUp,                   "PCSetUp",          PC_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&PC_SetUpOnBlocks,           "PCSetUpOnBlocks",  PC_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&PC_Apply,                   "PCApply",          PC_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&PC_ApplyCoarse,             "PCApplyCoarse",    PC_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&PC_ApplyMultiple,           "PCApplyMultiple",  PC_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&PC_ApplySymmetricLeft,      "PCApplySymmLeft",  PC_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&PC_ApplySymmetricRight,     "PCApplySymmRight", PC_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&PC_ModifySubMatrices,       "PCModifySubMatri", PC_COOKIE);CHKERRQ(ierr);
  /* Process info exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL, "-log_info_exclude", logList, 256, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList, "pc", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogInfoDeactivateClass(PC_COOKIE);CHKERRQ(ierr);
    }
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL, "-log_summary_exclude", logList, 256, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList, "pc", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogEventDeactivateClass(PC_COOKIE);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPInitializePackage"
/*@C
  KSPInitializePackage - This function initializes everything in the KSP package. It is called
  from PetscDLLibraryRegister() when using dynamic libraries, and on the first call to KSPCreate()
  when using static libraries.

  Input Parameter:
  path - The dynamic library path, or PETSC_NULL

  Level: developer

.keywords: KSP, initialize, package
.seealso: PetscInitialize()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT KSPInitializePackage(const char path[]) {
  static PetscTruth initialized = PETSC_FALSE;
  char              logList[256];
  char             *className;
  PetscTruth        opt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (initialized) PetscFunctionReturn(0);
  initialized = PETSC_TRUE;
  /* Register Classes */
  ierr = PetscLogClassRegister(&KSP_COOKIE,  "Krylov Solver");CHKERRQ(ierr);
  /* Register Constructors */
  ierr = KSPRegisterAll(path);CHKERRQ(ierr);
  /* Register Events */
  ierr = PetscLogEventRegister(&KSP_GMRESOrthogonalization, "KSPGMRESOrthog",   KSP_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&KSP_SetUp,                  "KSPSetup",         KSP_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&KSP_Solve,                  "KSPSolve",         KSP_COOKIE);CHKERRQ(ierr);
  /* Process info exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL, "-log_info_exclude", logList, 256, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList, "ksp", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogInfoDeactivateClass(KSP_COOKIE);CHKERRQ(ierr);
    }
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL, "-log_summary_exclude", logList, 256, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList, "ksp", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogEventDeactivateClass(KSP_COOKIE);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#ifdef PETSC_USE_DYNAMIC_LIBRARIES
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PetscDLLibraryRegister"
/*
  PetscDLLibraryRegister - This function is called when the dynamic library it is in is opened.

  This one registers all the KSP and PC methods that are in the basic PETSc libpetscksp
  library.

  Input Parameter:
  path - library path
 */
PetscErrorCode PETSCKSP_DLLEXPORT PetscDLLibraryRegister(char *path)
{
  PetscErrorCode ierr;

  ierr = PetscInitializeNoArguments(); if (ierr) return 1;

  PetscFunctionBegin;
  /*
      If we got here then PETSc was properly loaded
  */
  ierr = PCInitializePackage(path);CHKERRQ(ierr);
  ierr = KSPInitializePackage(path);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

/* --------------------------------------------------------------------------*/
static const char *contents = "PETSc Krylov subspace method and preconditioner library.\n\
     GMRES, PCG, Bi-CG-stab, ...\n\
     Jacobi, ILU, Block Jacobi, LU, Additive Schwarz, ...\n";
static const char *authors  = PETSC_AUTHOR_INFO;

#include "src/sys/src/utils/dlregis.h"

#endif /* PETSC_USE_DYNAMIC_LIBRARIES */
