#include <petsc/private/tsimpl.h>

static PetscBool TSPackageInitialized = PETSC_FALSE;
/*@C
  TSFinalizePackage - This function destroys everything in the Petsc interface to `TS`. It is
  called from `PetscFinalize()`.

  Level: developer

.seealso: [](chapter_ts), `TS`, `PetscFinalize()`, `TSInitializePackage()`
@*/
PetscErrorCode TSFinalizePackage(void)
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListDestroy(&TSList));
  PetscCall(PetscFunctionListDestroy(&TSTrajectoryList));
  TSPackageInitialized = PETSC_FALSE;
  TSRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TSInitializePackage - This function initializes everything in the `TS` package. It is called
  from `PetscDLLibraryRegister_petscts()` when using dynamic libraries, and on the first call to `TSCreate()`
  when using shared or static libraries.

  Level: developer

.seealso: [](chapter_ts), `TS`, `PetscInitialize()`, `TSFinalizePackage()`
@*/
PetscErrorCode TSInitializePackage(void)
{
  char      logList[256];
  PetscBool opt, pkg, cls;

  PetscFunctionBegin;
  if (TSPackageInitialized) PetscFunctionReturn(PETSC_SUCCESS);
  TSPackageInitialized = PETSC_TRUE;
  /* Initialize subpackages */
  PetscCall(TSAdaptInitializePackage());
  PetscCall(TSGLLEInitializePackage());
  PetscCall(TSRKInitializePackage());
  PetscCall(TSGLEEInitializePackage());
  PetscCall(TSARKIMEXInitializePackage());
  PetscCall(TSRosWInitializePackage());
  PetscCall(TSSSPInitializePackage());
  PetscCall(TSGLLEAdaptInitializePackage());
  PetscCall(TSBasicSymplecticInitializePackage());
  /* Register Classes */
  PetscCall(PetscClassIdRegister("TS", &TS_CLASSID));
  PetscCall(PetscClassIdRegister("DMTS", &DMTS_CLASSID));
  PetscCall(PetscClassIdRegister("TSTrajectory", &TSTRAJECTORY_CLASSID));

  /* Register Constructors */
  PetscCall(TSRegisterAll());
  PetscCall(TSTrajectoryRegisterAll());
  /* Register Events */
  PetscCall(PetscLogEventRegister("TSStep", TS_CLASSID, &TS_Step));
  PetscCall(PetscLogEventRegister("TSFunctionEval", TS_CLASSID, &TS_FunctionEval));
  PetscCall(PetscLogEventRegister("TSJacobianEval", TS_CLASSID, &TS_JacobianEval));
  PetscCall(PetscLogEventRegister("TSForwardStep", TS_CLASSID, &TS_ForwardStep));
  PetscCall(PetscLogEventRegister("TSAdjointStep", TS_CLASSID, &TS_AdjointStep));
  PetscCall(PetscLogEventRegister("TSTrajectorySet", TSTRAJECTORY_CLASSID, &TSTrajectory_Set));
  PetscCall(PetscLogEventRegister("TSTrajectoryGet", TSTRAJECTORY_CLASSID, &TSTrajectory_Get));
  PetscCall(PetscLogEventRegister("TSTrajGetVecs", TSTRAJECTORY_CLASSID, &TSTrajectory_GetVecs));
  PetscCall(PetscLogEventRegister("TSTrajSetUp", TSTRAJECTORY_CLASSID, &TSTrajectory_SetUp));
  PetscCall(PetscLogEventRegister("TSTrajDiskWrite", TSTRAJECTORY_CLASSID, &TSTrajectory_DiskWrite));
  PetscCall(PetscLogEventRegister("TSTrajDiskRead", TSTRAJECTORY_CLASSID, &TSTrajectory_DiskRead));
  PetscCall(PetscLogEventRegister("TSPseudoCmptTStp", TS_CLASSID, &TS_PseudoComputeTimeStep));
  /* Process Info */
  {
    PetscClassId classids[4];

    classids[0] = TS_CLASSID;
    classids[1] = DMTS_CLASSID;
    classids[2] = TSADAPT_CLASSID;
    classids[3] = TSTRAJECTORY_CLASSID;
    PetscCall(PetscInfoProcessClass("ts", 1, classids));
    PetscCall(PetscInfoProcessClass("dm", 1, &classids[1]));
    PetscCall(PetscInfoProcessClass("tsadapt", 1, &classids[2]));
    PetscCall(PetscInfoProcessClass("tstrajectory", 1, &classids[3]));
  }
  /* Process summary exclusions */
  PetscCall(PetscOptionsGetString(NULL, NULL, "-log_exclude", logList, sizeof(logList), &opt));
  if (opt) {
    PetscCall(PetscStrInList("ts", logList, ',', &pkg));
    if (pkg) PetscCall(PetscLogEventExcludeClass(TS_CLASSID));
    PetscCall(PetscStrInList("dm", logList, ',', &cls));
    if (pkg || cls) PetscCall(PetscLogEventExcludeClass(DMTS_CLASSID));
    PetscCall(PetscStrInList("tsadapt", logList, ',', &cls));
    if (pkg || cls) PetscCall(PetscLogEventExcludeClass(TSADAPT_CLASSID));
    PetscCall(PetscStrInList("tstrajectory", logList, ',', &cls));
    if (pkg || cls) PetscCall(PetscLogEventExcludeClass(TSTRAJECTORY_CLASSID));
  }
  /* Register package finalizer */
  PetscCall(PetscRegisterFinalize(TSFinalizePackage));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if defined(PETSC_HAVE_DYNAMIC_LIBRARIES)
/*
  PetscDLLibraryRegister - This function is called when the dynamic library it is in is opened.

  This one registers all the TS methods that are in the basic PETSc libpetscts library.

 */
PETSC_EXTERN PetscErrorCode PetscDLLibraryRegister_petscts(void); /*prototype*/
PETSC_EXTERN PetscErrorCode PetscDLLibraryRegister_petscts(void)
{
  PetscFunctionBegin;
  PetscCall(TSInitializePackage());
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif /* PETSC_HAVE_DYNAMIC_LIBRARIES */
