#define PETSCKSP_DLL

#include "private/pcimpl.h"
#include "private/kspimpl.h"


const char *PCSides[]          = {"LEFT","RIGHT","SYMMETRIC","PCSide","PC_",0};
const char *PCASMTypes[]       = {"NONE","RESTRICT","INTERPOLATE","BASIC","PCASMType","PC_ASM_",0};
const char *PCCompositeTypes[] = {"ADDITIVE","MULTIPLICATIVE","SYMMETRIC_MULTIPLICATIVE","SPECIAL","SCHUR","PCCompositeType","PC_COMPOSITE",0};

static PetscTruth PCPackageInitialized = PETSC_FALSE;
#undef __FUNCT__  
#define __FUNCT__ "PCFinalizePackage"
/*@C
  PCFinalizePackage - This function destroys everything in the Petsc interface to the charactoristics package. It is
  called from PetscFinalize().

  Level: developer

.keywords: Petsc, destroy, package, mathematica
.seealso: PetscFinalize()
@*/
PetscErrorCode PETSC_DLLEXPORT PCFinalizePackage(void) 
{
  PetscFunctionBegin;
  PCPackageInitialized = PETSC_FALSE;
  PCRegisterAllCalled  = PETSC_FALSE;
  PCList               = 0;
  PetscFunctionReturn(0);
}

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
PetscErrorCode PETSCKSP_DLLEXPORT PCInitializePackage(const char path[]) 
{
  char              logList[256];
  char             *className;
  PetscTruth        opt;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (PCPackageInitialized) PetscFunctionReturn(0);
  PCPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  ierr = PetscCookieRegister("Preconditioner",&PC_COOKIE);CHKERRQ(ierr);
  /* Register Constructors */
  ierr = PCRegisterAll(path);CHKERRQ(ierr);
  /* Register Events */
  ierr = PetscLogEventRegister("PCSetUp",          PC_COOKIE,&PC_SetUp);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("PCSetUpOnBlocks",  PC_COOKIE,&PC_SetUpOnBlocks);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("PCApplyOnBlocks",  PC_COOKIE,&PC_ApplyOnBlocks);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("PCApply",          PC_COOKIE,&PC_Apply);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("PCApplyCoarse",    PC_COOKIE,&PC_ApplyCoarse);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("PCApplyMultiple",  PC_COOKIE,&PC_ApplyMultiple);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("PCApplySymmLeft",  PC_COOKIE,&PC_ApplySymmetricLeft);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("PCApplySymmRight", PC_COOKIE,&PC_ApplySymmetricRight);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("PCModifySubMatri", PC_COOKIE,&PC_ModifySubMatrices);CHKERRQ(ierr);
  /* Process info exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL, "-info_exclude", logList, 256, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList, "pc", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscInfoDeactivateClass(PC_COOKIE);CHKERRQ(ierr);
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
  ierr = PetscRegisterFinalize(PCFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

const char *KSPCGTypes[]                  = {"SYMMETRIC","HERMITIAN","KSPCGType","KSP_CG_",0};
const char *KSPGMRESCGSRefinementTypes[]  = {"REFINE_NEVER", "REFINE_IFNEEDED", "REFINE_ALWAYS","KSPGMRESRefinementType","KSP_GMRES_CGS_",0};
const char *KSPNormTypes[]                = {"NO","PRECONDITIONED","UNPRECONDITIONED","NATURAL","KSPNormType","KSP_NORM_",0};
const char *KSPConvergedReasons_Shifted[] = {"DIVERGED_INDEFINITE_MAT","DIVERGED_NAN","DIVERGED_INDEFINITE_PC",
					     "DIVERGED_NONSYMMETRIC", "DIVERGED_BREAKDOWN_BICG","DIVERGED_BREAKDOWN",
                                             "DIVERGED_DTOL","DIVERGED_ITS","DIVERGED_NULL","","CONVERGED_ITERATING",
                                             "","CONVERGED_RTOL","CONVERGED_ATOL","CONVERGED_ITS",
                                             "CONVERGED_CG_NEG_CURVE","CONVERGED_CG_CONSTRAINED","CONVERGED_STEP_LENGTH",
                                             "CONVERGED_HAPPY_BREAKDOWN","KSPConvergedReason","KSP_",0};
const char **KSPConvergedReasons = KSPConvergedReasons_Shifted + 10;

static PetscTruth KSPPackageInitialized = PETSC_FALSE;
#undef __FUNCT__  
#define __FUNCT__ "KSPFinalizePackage"
/*@C
  KSPFinalizePackage - This function destroys everything in the Petsc interface to the charactoristics package. It is
  called from PetscFinalize().

  Level: developer

.keywords: Petsc, destroy, package, mathematica
.seealso: PetscFinalize()
@*/
PetscErrorCode PETSC_DLLEXPORT KSPFinalizePackage(void) 
{
  PetscFunctionBegin;
  KSPPackageInitialized = PETSC_FALSE;
  KSPList               = 0;
  KSPRegisterAllCalled  = PETSC_FALSE;
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
PetscErrorCode PETSCKSP_DLLEXPORT KSPInitializePackage(const char path[]) 
{
  char              logList[256];
  char              *className;
  PetscTruth        opt;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (KSPPackageInitialized) PetscFunctionReturn(0);
  KSPPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  ierr = PetscCookieRegister("Krylov Solver",&KSP_COOKIE);CHKERRQ(ierr);
  /* Register Constructors */
  ierr = KSPRegisterAll(path);CHKERRQ(ierr);
  /* Register Events */
  ierr = PetscLogEventRegister("KSPGMRESOrthog",   KSP_COOKIE,&KSP_GMRESOrthogonalization);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("KSPSetup",         KSP_COOKIE,&KSP_SetUp);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("KSPSolve",         KSP_COOKIE,&KSP_Solve);CHKERRQ(ierr);
  /* Process info exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL, "-info_exclude", logList, 256, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList, "ksp", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscInfoDeactivateClass(KSP_COOKIE);CHKERRQ(ierr);
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
  ierr = PetscRegisterFinalize(KSPFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#ifdef PETSC_USE_DYNAMIC_LIBRARIES
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PetscDLLibraryRegister_petscksp"
/*
  PetscDLLibraryRegister - This function is called when the dynamic library it is in is opened.

  This one registers all the KSP and PC methods that are in the basic PETSc libpetscksp
  library.

  Input Parameter:
  path - library path
 */
PetscErrorCode PETSCKSP_DLLEXPORT PetscDLLibraryRegister_petscksp(const char path[])
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

#endif /* PETSC_USE_DYNAMIC_LIBRARIES */
