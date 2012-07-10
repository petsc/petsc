
#include <petsc-private/pcimpl.h>
#include <petsc-private/kspimpl.h>

static const char *const PCSides_Shifted[] = {"DEFAULT","LEFT","RIGHT","SYMMETRIC","PCSide","PC_",0};
const char *const *const PCSides = PCSides_Shifted + 1;
const char *PCASMTypes[]       = {"NONE","RESTRICT","INTERPOLATE","BASIC","PCASMType","PC_ASM_",0};
const char *PCGASMTypes[]       = {"NONE","RESTRICT","INTERPOLATE","BASIC","PCGASMType","PC_GASM_",0};
const char *PCCompositeTypes[] = {"ADDITIVE","MULTIPLICATIVE","SYMMETRIC_MULTIPLICATIVE","SPECIAL","SCHUR","PCCompositeType","PC_COMPOSITE",0};
const char *PCPARMSGlobalTypes[] = {"RAS","SCHUR","BJ","PCPARMSGlobalType","PC_PARMS_",0};
const char *PCPARMSLocalTypes[]  = {"ILU0","ILUK","ILUT","ARMS","PCPARMSLocalType","PC_PARMS_",0};

static PetscBool  PCPackageInitialized = PETSC_FALSE;
#undef __FUNCT__  
#define __FUNCT__ "PCFinalizePackage"
/*@C
  PCFinalizePackage - This function destroys everything in the Petsc interface to the characteristics package. It is
  called from PetscFinalize().

  Level: developer

.keywords: Petsc, destroy, package, mathematica
.seealso: PetscFinalize()
@*/
PetscErrorCode  PCFinalizePackage(void)
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
PetscErrorCode  PCInitializePackage(const char path[]) 
{
  char              logList[256];
  char             *className;
  PetscBool         opt;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (PCPackageInitialized) PetscFunctionReturn(0);
  PCPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  ierr = PetscClassIdRegister("Preconditioner",&PC_CLASSID);CHKERRQ(ierr);
  /* Register Constructors */
  ierr = PCRegisterAll(path);CHKERRQ(ierr);
  /* Register Events */
  ierr = PetscLogEventRegister("PCSetUp",          PC_CLASSID,&PC_SetUp);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("PCSetUpOnBlocks",  PC_CLASSID,&PC_SetUpOnBlocks);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("PCApplyOnBlocks",  PC_CLASSID,&PC_ApplyOnBlocks);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("PCApplyOnMproc",   PC_CLASSID,&PC_ApplyOnMproc);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("PCApply",          PC_CLASSID,&PC_Apply);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("PCApplyCoarse",    PC_CLASSID,&PC_ApplyCoarse);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("PCApplyMultiple",  PC_CLASSID,&PC_ApplyMultiple);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("PCApplySymmLeft",  PC_CLASSID,&PC_ApplySymmetricLeft);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("PCApplySymmRight", PC_CLASSID,&PC_ApplySymmetricRight);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("PCModifySubMatri", PC_CLASSID,&PC_ModifySubMatrices);CHKERRQ(ierr);
  /* Process info exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL, "-info_exclude", logList, 256, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList, "pc", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscInfoDeactivateClass(PC_CLASSID);CHKERRQ(ierr);
    }
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL, "-log_summary_exclude", logList, 256, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList, "pc", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogEventDeactivateClass(PC_CLASSID);CHKERRQ(ierr);
    }
  }
  ierr = PetscRegisterFinalize(PCFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

const char *KSPCGTypes[]                  = {"SYMMETRIC","HERMITIAN","KSPCGType","KSP_CG_",0};
const char *KSPGMRESCGSRefinementTypes[]  = {"REFINE_NEVER", "REFINE_IFNEEDED", "REFINE_ALWAYS","KSPGMRESRefinementType","KSP_GMRES_CGS_",0};
const char *KSPNormTypes_Shifted[]        = {"DEFAULT","NONE","PRECONDITIONED","UNPRECONDITIONED","NATURAL","KSPNormType","KSP_NORM_",0};
const char *const*const KSPNormTypes = KSPNormTypes_Shifted + 1;
const char *KSPConvergedReasons_Shifted[] = {"DIVERGED_INDEFINITE_MAT","DIVERGED_NAN","DIVERGED_INDEFINITE_PC",
					     "DIVERGED_NONSYMMETRIC", "DIVERGED_BREAKDOWN_BICG","DIVERGED_BREAKDOWN",
                                             "DIVERGED_DTOL","DIVERGED_ITS","DIVERGED_NULL","","CONVERGED_ITERATING",
                                             "CONVERGED_RTOL_NORMAL","CONVERGED_RTOL","CONVERGED_ATOL","CONVERGED_ITS",
                                             "CONVERGED_CG_NEG_CURVE","CONVERGED_CG_CONSTRAINED","CONVERGED_STEP_LENGTH",
                                             "CONVERGED_HAPPY_BREAKDOWN","CONVERGED_ATOL_NORMAL","KSPConvergedReason","KSP_",0};
const char *const*KSPConvergedReasons = KSPConvergedReasons_Shifted + 10;

static PetscBool  KSPPackageInitialized = PETSC_FALSE;
#undef __FUNCT__  
#define __FUNCT__ "KSPFinalizePackage"
/*@C
  KSPFinalizePackage - This function destroys everything in the Petsc interface to the KSP package. It is
  called from PetscFinalize().

  Level: developer

.keywords: Petsc, destroy, package, mathematica
.seealso: PetscFinalize()
@*/
PetscErrorCode  KSPFinalizePackage(void)
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
PetscErrorCode  KSPInitializePackage(const char path[]) 
{
  char              logList[256];
  char              *className;
  PetscBool         opt;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (KSPPackageInitialized) PetscFunctionReturn(0);
  KSPPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  ierr = PetscClassIdRegister("Krylov Solver",&KSP_CLASSID);CHKERRQ(ierr);
  /* Register Constructors */
  ierr = KSPRegisterAll(path);CHKERRQ(ierr);
  /* Register Events */
  ierr = PetscLogEventRegister("KSPGMRESOrthog",   KSP_CLASSID,&KSP_GMRESOrthogonalization);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("KSPSetUp",         KSP_CLASSID,&KSP_SetUp);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("KSPSolve",         KSP_CLASSID,&KSP_Solve);CHKERRQ(ierr);
  /* Process info exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL, "-info_exclude", logList, 256, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList, "ksp", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscInfoDeactivateClass(KSP_CLASSID);CHKERRQ(ierr);
    }
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL, "-log_summary_exclude", logList, 256, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList, "ksp", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogEventDeactivateClass(KSP_CLASSID);CHKERRQ(ierr);
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
PetscErrorCode  PetscDLLibraryRegister_petscksp(const char path[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCInitializePackage(path);CHKERRQ(ierr);
  ierr = KSPInitializePackage(path);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#endif /* PETSC_USE_DYNAMIC_LIBRARIES */
