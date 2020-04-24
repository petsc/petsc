
#include <petsc/private/pcimpl.h>
#include <petsc/private/pcpatchimpl.h> /* For new events */
#include <petsc/private/kspimpl.h>

static const char *const PCSides_Shifted[]    = {"DEFAULT","LEFT","RIGHT","SYMMETRIC","PCSide","PC_",0};
const char *const *const PCSides              = PCSides_Shifted + 1;
const char *const        PCASMTypes[]         = {"NONE","RESTRICT","INTERPOLATE","BASIC","PCASMType","PC_ASM_",0};
const char *const        PCGASMTypes[]        = {"NONE","RESTRICT","INTERPOLATE","BASIC","PCGASMType","PC_GASM_",0};
const char *const        PCCompositeTypes[]   = {"ADDITIVE","MULTIPLICATIVE","SYMMETRIC_MULTIPLICATIVE","SPECIAL","SCHUR","GKB","PCCompositeType","PC_COMPOSITE",0};
const char *const        PCPARMSGlobalTypes[] = {"RAS","SCHUR","BJ","PCPARMSGlobalType","PC_PARMS_",0};
const char *const        PCPARMSLocalTypes[]  = {"ILU0","ILUK","ILUT","ARMS","PCPARMSLocalType","PC_PARMS_",0};
const char *const        PCPatchConstructTypes[] = {"star", "vanka", "pardecomp", "user", "python", "PCPatchSetConstructType", "PC_PATCH_", 0};

const char *const        PCFailedReasons[]    = {"FACTOR_NOERROR","FACTOR_STRUCT_ZEROPIVOT","FACTOR_NUMERIC_ZEROPIVOT","FACTOR_OUTMEMORY","FACTOR_OTHER","SUBPC_ERROR",0};

static PetscBool PCPackageInitialized = PETSC_FALSE;
/*@C
  PCFinalizePackage - This function destroys everything in the Petsc interface to the characteristics package. It is
  called from PetscFinalize().

  Level: developer

.seealso: PetscFinalize()
@*/
PetscErrorCode  PCFinalizePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListDestroy(&PCList);CHKERRQ(ierr);
  PCPackageInitialized = PETSC_FALSE;
  PCRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
  PCInitializePackage - This function initializes everything in the PC package. It is called
  from PetscDLLibraryRegister_petscksp() when using dynamic libraries, and on the first call to PCCreate()
  when using shared static libraries.

  Level: developer

.seealso: PetscInitialize()
@*/
PetscErrorCode  PCInitializePackage(void)
{
  char           logList[256];
  PetscBool      opt,pkg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PCPackageInitialized) PetscFunctionReturn(0);
  PCPackageInitialized = PETSC_TRUE;
  /* Initialize subpackages */
  ierr = PCGAMGInitializePackage();CHKERRQ(ierr);
  ierr = PCBDDCInitializePackage();CHKERRQ(ierr);
#if defined(PETSC_HAVE_HPDDM)
  ierr = PCHPDDMInitializePackage();CHKERRQ(ierr);
#endif
  /* Register Classes */
  ierr = PetscClassIdRegister("Preconditioner",&PC_CLASSID);CHKERRQ(ierr);
  /* Register Constructors */
  ierr = PCRegisterAll();CHKERRQ(ierr);
  /* Register Events */
  ierr = PetscLogEventRegister("PCSetUp",          PC_CLASSID,&PC_SetUp);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("PCSetUpOnBlocks",  PC_CLASSID,&PC_SetUpOnBlocks);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("PCApply",          PC_CLASSID,&PC_Apply);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("PCApplyOnBlocks",  PC_CLASSID,&PC_ApplyOnBlocks);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("PCApplyCoarse",    PC_CLASSID,&PC_ApplyCoarse);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("PCApplyMultiple",  PC_CLASSID,&PC_ApplyMultiple);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("PCApplySymmLeft",  PC_CLASSID,&PC_ApplySymmetricLeft);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("PCApplySymmRight", PC_CLASSID,&PC_ApplySymmetricRight);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("PCModifySubMatri", PC_CLASSID,&PC_ModifySubMatrices);CHKERRQ(ierr);

  ierr = PetscLogEventRegister("PCPATCHCreate",    PC_CLASSID, &PC_Patch_CreatePatches);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("PCPATCHComputeOp", PC_CLASSID, &PC_Patch_ComputeOp);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("PCPATCHSolve",     PC_CLASSID, &PC_Patch_Solve);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("PCPATCHApply",     PC_CLASSID, &PC_Patch_Apply);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("PCPATCHPrealloc",  PC_CLASSID, &PC_Patch_Prealloc);CHKERRQ(ierr);

  ierr = PetscLogEventRegister("KSPSolve_FS_0",    KSP_CLASSID,&KSP_Solve_FS_0);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("KSPSolve_FS_1",    KSP_CLASSID,&KSP_Solve_FS_1);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("KSPSolve_FS_2",    KSP_CLASSID,&KSP_Solve_FS_2);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("KSPSolve_FS_3",    KSP_CLASSID,&KSP_Solve_FS_3);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("KSPSolve_FS_4",    KSP_CLASSID,&KSP_Solve_FS_4);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("KSPSolve_FS_Schu", KSP_CLASSID,&KSP_Solve_FS_S);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("KSPSolve_FS_Up",   KSP_CLASSID,&KSP_Solve_FS_U);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("KSPSolve_FS_Low",  KSP_CLASSID,&KSP_Solve_FS_L);CHKERRQ(ierr);
  /* Process Info */
  {
    PetscClassId  classids[1];

    classids[0] = PC_CLASSID;
    ierr = PetscInfoProcessClass("pc", 1, classids);CHKERRQ(ierr);
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,sizeof(logList),&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrInList("pc",logList,',',&pkg);CHKERRQ(ierr);
    if (pkg) {ierr = PetscLogEventExcludeClass(PC_CLASSID);CHKERRQ(ierr);}
  }
  /* Register data */
  ierr = PetscObjectComposedDataRegister(&PetscMGLevelId);CHKERRQ(ierr);
  /* Register package finalizer */
  ierr = PetscRegisterFinalize(PCFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

const char *const KSPCGTypes[]                  = {"SYMMETRIC","HERMITIAN","KSPCGType","KSP_CG_",0};
const char *const KSPGMRESCGSRefinementTypes[]  = {"REFINE_NEVER", "REFINE_IFNEEDED", "REFINE_ALWAYS","KSPGMRESRefinementType","KSP_GMRES_CGS_",0};
const char *const KSPNormTypes_Shifted[]        = {"DEFAULT","NONE","PRECONDITIONED","UNPRECONDITIONED","NATURAL","KSPNormType","KSP_NORM_",0};
const char *const*const KSPNormTypes = KSPNormTypes_Shifted + 1;
const char *const KSPConvergedReasons_Shifted[] = {"DIVERGED_PC_FAILED","DIVERGED_INDEFINITE_MAT","DIVERGED_NANORINF","DIVERGED_INDEFINITE_PC",
                                                   "DIVERGED_NONSYMMETRIC", "DIVERGED_BREAKDOWN_BICG","DIVERGED_BREAKDOWN",
                                                   "DIVERGED_DTOL","DIVERGED_ITS","DIVERGED_NULL","","CONVERGED_ITERATING",
                                                   "CONVERGED_RTOL_NORMAL","CONVERGED_RTOL","CONVERGED_ATOL","CONVERGED_ITS",
                                                   "CONVERGED_CG_NEG_CURVE","CONVERGED_CG_CONSTRAINED","CONVERGED_STEP_LENGTH",
                                                   "CONVERGED_HAPPY_BREAKDOWN","CONVERGED_ATOL_NORMAL","KSPConvergedReason","KSP_",0};
const char *const*KSPConvergedReasons = KSPConvergedReasons_Shifted + 11;
const char *const KSPFCDTruncationTypes[] = {"STANDARD","NOTAY","KSPFCDTruncationTypes","KSP_FCD_TRUNC_TYPE_",0};

static PetscBool KSPPackageInitialized = PETSC_FALSE;
/*@C
  KSPFinalizePackage - This function destroys everything in the Petsc interface to the KSP package. It is
  called from PetscFinalize().

  Level: developer

.seealso: PetscFinalize()
@*/
PetscErrorCode  KSPFinalizePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListDestroy(&KSPList);CHKERRQ(ierr);
  ierr = PetscFunctionListDestroy(&KSPGuessList);CHKERRQ(ierr);
  KSPPackageInitialized = PETSC_FALSE;
  KSPRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
  KSPInitializePackage - This function initializes everything in the KSP package. It is called
  from PetscDLLibraryRegister_petscksp() when using dynamic libraries, and on the first call to KSPCreate()
  when using shared or static libraries.

  Level: developer

.seealso: PetscInitialize()
@*/
PetscErrorCode  KSPInitializePackage(void)
{
  char           logList[256];
  PetscBool      opt,pkg,cls;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (KSPPackageInitialized) PetscFunctionReturn(0);
  KSPPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  ierr = PetscClassIdRegister("Krylov Solver",&KSP_CLASSID);CHKERRQ(ierr);
  ierr = PetscClassIdRegister("DMKSP interface",&DMKSP_CLASSID);CHKERRQ(ierr);
  ierr = PetscClassIdRegister("KSPGuess interface",&KSPGUESS_CLASSID);CHKERRQ(ierr);
  /* Register Constructors */
  ierr = KSPRegisterAll();CHKERRQ(ierr);
  /* Register matrix implementations packaged in KSP */
  ierr = KSPMatRegisterAll();CHKERRQ(ierr);
  /* Register KSP guesses implementations */
  ierr = KSPGuessRegisterAll();CHKERRQ(ierr);
  /* Register Events */
  ierr = PetscLogEventRegister("KSPSetUp",         KSP_CLASSID,&KSP_SetUp);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("KSPSolve",         KSP_CLASSID,&KSP_Solve);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("KSPGMRESOrthog",   KSP_CLASSID,&KSP_GMRESOrthogonalization);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("KSPSolveTranspos", KSP_CLASSID,&KSP_SolveTranspose);CHKERRQ(ierr);
  /* Process Info */
  {
    PetscClassId  classids[3];

    classids[0] = KSP_CLASSID;
    classids[1] = DMKSP_CLASSID;
    classids[2] = KSPGUESS_CLASSID;
    ierr = PetscInfoProcessClass("ksp", 1, &classids[0]);CHKERRQ(ierr);
    ierr = PetscInfoProcessClass("dm", 1, &classids[1]);CHKERRQ(ierr);
    ierr = PetscInfoProcessClass("kspguess", 1, &classids[2]);CHKERRQ(ierr);
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,sizeof(logList),&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrInList("ksp",logList,',',&pkg);CHKERRQ(ierr);
    if (pkg) {ierr = PetscLogEventExcludeClass(KSP_CLASSID);CHKERRQ(ierr);}
    ierr = PetscStrInList("dm",logList,',',&cls);CHKERRQ(ierr);
    if (pkg || cls) {ierr = PetscLogEventExcludeClass(DMKSP_CLASSID);CHKERRQ(ierr);}
    ierr = PetscStrInList("kspguess",logList,',',&cls);CHKERRQ(ierr);
    if (pkg || cls) {ierr = PetscLogEventExcludeClass(KSPGUESS_CLASSID);CHKERRQ(ierr);}
  }
  /* Register package finalizer */
  ierr = PetscRegisterFinalize(KSPFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_DYNAMIC_LIBRARIES)

/*
  PetscDLLibraryRegister - This function is called when the dynamic library it is in is opened.

  This one registers all the KSP and PC methods that are in the basic PETSc libpetscksp
  library.

 */
PETSC_EXTERN PetscErrorCode PetscDLLibraryRegister_petscksp(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCInitializePackage();CHKERRQ(ierr);
  ierr = KSPInitializePackage();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#endif /* PETSC_HAVE_DYNAMIC_LIBRARIES */
