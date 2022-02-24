#include <petsc/private/tsimpl.h>

static PetscBool TSPackageInitialized = PETSC_FALSE;
/*@C
  TSFinalizePackage - This function destroys everything in the Petsc interface to Mathematica. It is
  called from PetscFinalize().

  Level: developer

.seealso: PetscFinalize()
@*/
PetscErrorCode  TSFinalizePackage(void)
{
  PetscFunctionBegin;
  CHKERRQ(PetscFunctionListDestroy(&TSList));
  CHKERRQ(PetscFunctionListDestroy(&TSTrajectoryList));
  TSPackageInitialized = PETSC_FALSE;
  TSRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
  TSInitializePackage - This function initializes everything in the TS package. It is called
  from PetscDLLibraryRegister_petscts() when using dynamic libraries, and on the first call to TSCreate()
  when using shared or static libraries.

  Level: developer

.seealso: PetscInitialize()
@*/
PetscErrorCode  TSInitializePackage(void)
{
  char           logList[256];
  PetscBool      opt,pkg,cls;

  PetscFunctionBegin;
  if (TSPackageInitialized) PetscFunctionReturn(0);
  TSPackageInitialized = PETSC_TRUE;
  /* Inialize subpackages */
  CHKERRQ(TSAdaptInitializePackage());
  CHKERRQ(TSGLLEInitializePackage());
  CHKERRQ(TSRKInitializePackage());
  CHKERRQ(TSGLEEInitializePackage());
  CHKERRQ(TSARKIMEXInitializePackage());
  CHKERRQ(TSRosWInitializePackage());
  CHKERRQ(TSSSPInitializePackage());
  CHKERRQ(TSGLLEAdaptInitializePackage());
  CHKERRQ(TSBasicSymplecticInitializePackage());
  /* Register Classes */
  CHKERRQ(PetscClassIdRegister("TS",&TS_CLASSID));
  CHKERRQ(PetscClassIdRegister("DMTS",&DMTS_CLASSID));
  CHKERRQ(PetscClassIdRegister("TSTrajectory",&TSTRAJECTORY_CLASSID));

  /* Register Constructors */
  CHKERRQ(TSRegisterAll());
  CHKERRQ(TSTrajectoryRegisterAll());
  /* Register Events */
  CHKERRQ(PetscLogEventRegister("TSStep",          TS_CLASSID,&TS_Step));
  CHKERRQ(PetscLogEventRegister("TSFunctionEval",  TS_CLASSID,&TS_FunctionEval));
  CHKERRQ(PetscLogEventRegister("TSJacobianEval",  TS_CLASSID,&TS_JacobianEval));
  CHKERRQ(PetscLogEventRegister("TSForwardStep",   TS_CLASSID,&TS_ForwardStep));
  CHKERRQ(PetscLogEventRegister("TSAdjointStep",   TS_CLASSID,&TS_AdjointStep));
  CHKERRQ(PetscLogEventRegister("TSTrajectorySet", TSTRAJECTORY_CLASSID,&TSTrajectory_Set));
  CHKERRQ(PetscLogEventRegister("TSTrajectoryGet", TSTRAJECTORY_CLASSID,&TSTrajectory_Get));
  CHKERRQ(PetscLogEventRegister("TSTrajGetVecs",   TSTRAJECTORY_CLASSID,&TSTrajectory_GetVecs));
  CHKERRQ(PetscLogEventRegister("TSTrajSetUp", TSTRAJECTORY_CLASSID,&TSTrajectory_SetUp));
  CHKERRQ(PetscLogEventRegister("TSTrajDiskWrite", TSTRAJECTORY_CLASSID,&TSTrajectory_DiskWrite));
  CHKERRQ(PetscLogEventRegister("TSTrajDiskRead",  TSTRAJECTORY_CLASSID,&TSTrajectory_DiskRead));
  CHKERRQ(PetscLogEventRegister("TSPseudoCmptTStp",TS_CLASSID,&TS_PseudoComputeTimeStep));
  /* Process Info */
  {
    PetscClassId  classids[4];

    classids[0] = TS_CLASSID;
    classids[1] = DMTS_CLASSID;
    classids[2] = TSADAPT_CLASSID;
    classids[3] = TSTRAJECTORY_CLASSID;
    CHKERRQ(PetscInfoProcessClass("ts", 1, classids));
    CHKERRQ(PetscInfoProcessClass("dm", 1, &classids[1]));
    CHKERRQ(PetscInfoProcessClass("tsadapt", 1, &classids[2]));
    CHKERRQ(PetscInfoProcessClass("tstrajectory", 1, &classids[3]));
  }
  /* Process summary exclusions */
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,sizeof(logList),&opt));
  if (opt) {
    CHKERRQ(PetscStrInList("ts",logList,',',&pkg));
    if (pkg) CHKERRQ(PetscLogEventExcludeClass(TS_CLASSID));
    CHKERRQ(PetscStrInList("dm",logList,',',&cls));
    if (pkg || cls) CHKERRQ(PetscLogEventExcludeClass(DMTS_CLASSID));
    CHKERRQ(PetscStrInList("tsadapt",logList,',',&cls));
    if (pkg || cls) CHKERRQ(PetscLogEventExcludeClass(TSADAPT_CLASSID));
    CHKERRQ(PetscStrInList("tstrajectory",logList,',',&cls));
    if (pkg || cls) CHKERRQ(PetscLogEventExcludeClass(TSTRAJECTORY_CLASSID));
  }
  /* Register package finalizer */
  CHKERRQ(PetscRegisterFinalize(TSFinalizePackage));
  PetscFunctionReturn(0);
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
  CHKERRQ(TSInitializePackage());
  PetscFunctionReturn(0);
}
#endif /* PETSC_HAVE_DYNAMIC_LIBRARIES */
