
#include <petsc/private/pcimpl.h>
#include <petsc/private/pcpatchimpl.h> /* For new events */
#include <petsc/private/kspimpl.h>

static const char *const PCSides_Shifted[]    = {"DEFAULT","LEFT","RIGHT","SYMMETRIC","PCSide","PC_",NULL};
const char *const *const PCSides              = PCSides_Shifted + 1;
const char *const        PCASMTypes[]         = {"NONE","RESTRICT","INTERPOLATE","BASIC","PCASMType","PC_ASM_",NULL};
const char *const        PCGASMTypes[]        = {"NONE","RESTRICT","INTERPOLATE","BASIC","PCGASMType","PC_GASM_",NULL};
const char *const        PCCompositeTypes[]   = {"ADDITIVE","MULTIPLICATIVE","SYMMETRIC_MULTIPLICATIVE","SPECIAL","SCHUR","GKB","PCCompositeType","PC_COMPOSITE",NULL};
const char *const        PCPARMSGlobalTypes[] = {"RAS","SCHUR","BJ","PCPARMSGlobalType","PC_PARMS_",NULL};
const char *const        PCPARMSLocalTypes[]  = {"ILU0","ILUK","ILUT","ARMS","PCPARMSLocalType","PC_PARMS_",NULL};
const char *const        PCPatchConstructTypes[] = {"star", "vanka", "pardecomp", "user", "python", "PCPatchSetConstructType", "PC_PATCH_", NULL};

const char *const        PCFailedReasons_Shifted[] = {"SETUP_ERROR","FACTOR_NOERROR","FACTOR_STRUCT_ZEROPIVOT","FACTOR_NUMERIC_ZEROPIVOT","FACTOR_OUTMEMORY","FACTOR_OTHER","SUBPC_ERROR",NULL};
const char *const *const PCFailedReasons       = PCFailedReasons_Shifted + 1;

static PetscBool PCPackageInitialized = PETSC_FALSE;
/*@C
  PCFinalizePackage - This function destroys everything in the Petsc interface to the characteristics package. It is
  called from PetscFinalize().

  Level: developer

.seealso: PetscFinalize()
@*/
PetscErrorCode  PCFinalizePackage(void)
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListDestroy(&PCList));
  PetscCall(PetscFunctionListDestroy(&PCMGCoarseList));
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

  PetscFunctionBegin;
  if (PCPackageInitialized) PetscFunctionReturn(0);
  PCPackageInitialized = PETSC_TRUE;
  /* Initialize subpackages */
  PetscCall(PCGAMGInitializePackage());
  PetscCall(PCBDDCInitializePackage());
#if defined(PETSC_HAVE_HPDDM) && defined(PETSC_HAVE_DYNAMIC_LIBRARIES) && defined(PETSC_USE_SHARED_LIBRARIES)
  PetscCall(PCHPDDMInitializePackage());
#endif
  /* Register Classes */
  PetscCall(PetscClassIdRegister("Preconditioner",&PC_CLASSID));
  /* Register Constructors */
  PetscCall(PCRegisterAll());
  /* Register Events */
  PetscCall(PetscLogEventRegister("PCSetUp",          PC_CLASSID,&PC_SetUp));
  PetscCall(PetscLogEventRegister("PCSetUpOnBlocks",  PC_CLASSID,&PC_SetUpOnBlocks));
  PetscCall(PetscLogEventRegister("PCApply",          PC_CLASSID,&PC_Apply));
  PetscCall(PetscLogEventRegister("PCMatApply",       PC_CLASSID,&PC_MatApply));
  PetscCall(PetscLogEventRegister("PCApplyOnBlocks",  PC_CLASSID,&PC_ApplyOnBlocks));
  PetscCall(PetscLogEventRegister("PCApplyCoarse",    PC_CLASSID,&PC_ApplyCoarse));
  PetscCall(PetscLogEventRegister("PCApplyMultiple",  PC_CLASSID,&PC_ApplyMultiple));
  PetscCall(PetscLogEventRegister("PCApplySymmLeft",  PC_CLASSID,&PC_ApplySymmetricLeft));
  PetscCall(PetscLogEventRegister("PCApplySymmRight", PC_CLASSID,&PC_ApplySymmetricRight));
  PetscCall(PetscLogEventRegister("PCModifySubMatri", PC_CLASSID,&PC_ModifySubMatrices));

  PetscCall(PetscLogEventRegister("PCPATCHCreate",    PC_CLASSID, &PC_Patch_CreatePatches));
  PetscCall(PetscLogEventRegister("PCPATCHComputeOp", PC_CLASSID, &PC_Patch_ComputeOp));
  PetscCall(PetscLogEventRegister("PCPATCHSolve",     PC_CLASSID, &PC_Patch_Solve));
  PetscCall(PetscLogEventRegister("PCPATCHApply",     PC_CLASSID, &PC_Patch_Apply));
  PetscCall(PetscLogEventRegister("PCPATCHPrealloc",  PC_CLASSID, &PC_Patch_Prealloc));

  PetscCall(PetscLogEventRegister("KSPSolve_FS_0",    KSP_CLASSID,&KSP_Solve_FS_0));
  PetscCall(PetscLogEventRegister("KSPSolve_FS_1",    KSP_CLASSID,&KSP_Solve_FS_1));
  PetscCall(PetscLogEventRegister("KSPSolve_FS_2",    KSP_CLASSID,&KSP_Solve_FS_2));
  PetscCall(PetscLogEventRegister("KSPSolve_FS_3",    KSP_CLASSID,&KSP_Solve_FS_3));
  PetscCall(PetscLogEventRegister("KSPSolve_FS_4",    KSP_CLASSID,&KSP_Solve_FS_4));
  PetscCall(PetscLogEventRegister("KSPSolve_FS_Schu", KSP_CLASSID,&KSP_Solve_FS_S));
  PetscCall(PetscLogEventRegister("KSPSolve_FS_Up",   KSP_CLASSID,&KSP_Solve_FS_U));
  PetscCall(PetscLogEventRegister("KSPSolve_FS_Low",  KSP_CLASSID,&KSP_Solve_FS_L));
  /* Process Info */
  {
    PetscClassId  classids[1];

    classids[0] = PC_CLASSID;
    PetscCall(PetscInfoProcessClass("pc", 1, classids));
  }
  /* Process summary exclusions */
  PetscCall(PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,sizeof(logList),&opt));
  if (opt) {
    PetscCall(PetscStrInList("pc",logList,',',&pkg));
    if (pkg) PetscCall(PetscLogEventExcludeClass(PC_CLASSID));
  }
  /* Register data */
  PetscCall(PetscObjectComposedDataRegister(&PetscMGLevelId));
  /* Register package finalizer */
  PetscCall(PetscRegisterFinalize(PCFinalizePackage));
  PetscFunctionReturn(0);
}

const char *const KSPCGTypes[]                  = {"SYMMETRIC","HERMITIAN","KSPCGType","KSP_CG_",NULL};
const char *const KSPGMRESCGSRefinementTypes[]  = {"REFINE_NEVER", "REFINE_IFNEEDED", "REFINE_ALWAYS","KSPGMRESRefinementType","KSP_GMRES_CGS_",NULL};
const char *const KSPNormTypes_Shifted[]        = {"DEFAULT","NONE","PRECONDITIONED","UNPRECONDITIONED","NATURAL","KSPNormType","KSP_NORM_",NULL};
const char *const*const KSPNormTypes = KSPNormTypes_Shifted + 1;
const char *const KSPConvergedReasons_Shifted[] = {"DIVERGED_PC_FAILED","DIVERGED_INDEFINITE_MAT","DIVERGED_NANORINF","DIVERGED_INDEFINITE_PC",
                                                   "DIVERGED_NONSYMMETRIC", "DIVERGED_BREAKDOWN_BICG","DIVERGED_BREAKDOWN",
                                                   "DIVERGED_DTOL","DIVERGED_ITS","DIVERGED_NULL","","CONVERGED_ITERATING",
                                                   "CONVERGED_RTOL_NORMAL","CONVERGED_RTOL","CONVERGED_ATOL","CONVERGED_ITS",
                                                   "CONVERGED_CG_NEG_CURVE","CONVERGED_CG_CONSTRAINED","CONVERGED_STEP_LENGTH",
                                                   "CONVERGED_HAPPY_BREAKDOWN","CONVERGED_ATOL_NORMAL","KSPConvergedReason","KSP_",NULL};
const char *const*KSPConvergedReasons = KSPConvergedReasons_Shifted + 11;
const char *const KSPFCDTruncationTypes[] = {"STANDARD","NOTAY","KSPFCDTruncationTypes","KSP_FCD_TRUNC_TYPE_",NULL};

static PetscBool KSPPackageInitialized = PETSC_FALSE;
/*@C
  KSPFinalizePackage - This function destroys everything in the Petsc interface to the KSP package. It is
  called from PetscFinalize().

  Level: developer

.seealso: PetscFinalize()
@*/
PetscErrorCode  KSPFinalizePackage(void)
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListDestroy(&KSPList));
  PetscCall(PetscFunctionListDestroy(&KSPGuessList));
  PetscCall(PetscFunctionListDestroy(&KSPMonitorList));
  PetscCall(PetscFunctionListDestroy(&KSPMonitorCreateList));
  PetscCall(PetscFunctionListDestroy(&KSPMonitorDestroyList));
  KSPPackageInitialized       = PETSC_FALSE;
  KSPRegisterAllCalled        = PETSC_FALSE;
  KSPMonitorRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
  KSPInitializePackage - This function initializes everything in the KSP package. It is called
  from PetscDLLibraryRegister_petscksp() when using dynamic libraries, and on the first call to KSPCreate()
  when using shared or static libraries.

  Level: developer

.seealso: PetscInitialize()
@*/
PetscErrorCode KSPInitializePackage(void)
{
  char           logList[256];
  PetscBool      opt,pkg,cls;

  PetscFunctionBegin;
  if (KSPPackageInitialized) PetscFunctionReturn(0);
  KSPPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  PetscCall(PetscClassIdRegister("Krylov Solver",&KSP_CLASSID));
  PetscCall(PetscClassIdRegister("DMKSP interface",&DMKSP_CLASSID));
  PetscCall(PetscClassIdRegister("KSPGuess interface",&KSPGUESS_CLASSID));
  /* Register Constructors */
  PetscCall(KSPRegisterAll());
  /* Register matrix implementations packaged in KSP */
  PetscCall(KSPMatRegisterAll());
  /* Register KSP guesses implementations */
  PetscCall(KSPGuessRegisterAll());
  /* Register Monitors */
  PetscCall(KSPMonitorRegisterAll());
  /* Register Events */
  PetscCall(PetscLogEventRegister("KSPSetUp",         KSP_CLASSID,&KSP_SetUp));
  PetscCall(PetscLogEventRegister("KSPSolve",         KSP_CLASSID,&KSP_Solve));
  PetscCall(PetscLogEventRegister("KSPGMRESOrthog",   KSP_CLASSID,&KSP_GMRESOrthogonalization));
  PetscCall(PetscLogEventRegister("KSPSolveTranspos", KSP_CLASSID,&KSP_SolveTranspose));
  PetscCall(PetscLogEventRegister("KSPMatSolve",      KSP_CLASSID,&KSP_MatSolve));
  /* Process Info */
  {
    PetscClassId  classids[3];

    classids[0] = KSP_CLASSID;
    classids[1] = DMKSP_CLASSID;
    classids[2] = KSPGUESS_CLASSID;
    PetscCall(PetscInfoProcessClass("ksp", 1, &classids[0]));
    PetscCall(PetscInfoProcessClass("dm", 1, &classids[1]));
    PetscCall(PetscInfoProcessClass("kspguess", 1, &classids[2]));
  }
  /* Process summary exclusions */
  PetscCall(PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,sizeof(logList),&opt));
  if (opt) {
    PetscCall(PetscStrInList("ksp",logList,',',&pkg));
    if (pkg) PetscCall(PetscLogEventExcludeClass(KSP_CLASSID));
    PetscCall(PetscStrInList("dm",logList,',',&cls));
    if (pkg || cls) PetscCall(PetscLogEventExcludeClass(DMKSP_CLASSID));
    PetscCall(PetscStrInList("kspguess",logList,',',&cls));
    if (pkg || cls) PetscCall(PetscLogEventExcludeClass(KSPGUESS_CLASSID));
  }
  /* Register package finalizer */
  PetscCall(PetscRegisterFinalize(KSPFinalizePackage));
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
  PetscFunctionBegin;
  PetscCall(PCInitializePackage());
  PetscCall(KSPInitializePackage());
  PetscFunctionReturn(0);
}

#endif /* PETSC_HAVE_DYNAMIC_LIBRARIES */
