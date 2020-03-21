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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListDestroy(&TSList);CHKERRQ(ierr);
  ierr = PetscFunctionListDestroy(&TSTrajectoryList);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (TSPackageInitialized) PetscFunctionReturn(0);
  TSPackageInitialized = PETSC_TRUE;
  /* Inialize subpackages */
  ierr = TSAdaptInitializePackage();CHKERRQ(ierr);
  ierr = TSGLLEInitializePackage();CHKERRQ(ierr);
  ierr = TSRKInitializePackage();CHKERRQ(ierr);
  ierr = TSGLEEInitializePackage();CHKERRQ(ierr);
  ierr = TSARKIMEXInitializePackage();CHKERRQ(ierr);
  ierr = TSRosWInitializePackage();CHKERRQ(ierr);
  ierr = TSSSPInitializePackage();CHKERRQ(ierr);
  ierr = TSGLLEAdaptInitializePackage();CHKERRQ(ierr);
  ierr = TSBasicSymplecticInitializePackage();CHKERRQ(ierr);
  /* Register Classes */
  ierr = PetscClassIdRegister("TS",&TS_CLASSID);CHKERRQ(ierr);
  ierr = PetscClassIdRegister("DMTS",&DMTS_CLASSID);CHKERRQ(ierr);
  ierr = PetscClassIdRegister("TSTrajectory",&TSTRAJECTORY_CLASSID);CHKERRQ(ierr);

  /* Register Constructors */
  ierr = TSRegisterAll();CHKERRQ(ierr);
  ierr = TSTrajectoryRegisterAll();CHKERRQ(ierr);
  /* Register Events */
  ierr = PetscLogEventRegister("TSStep",          TS_CLASSID,&TS_Step);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("TSFunctionEval",  TS_CLASSID,&TS_FunctionEval);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("TSJacobianEval",  TS_CLASSID,&TS_JacobianEval);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("TSForwardStep",   TS_CLASSID,&TS_ForwardStep);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("TSAdjointStep",   TS_CLASSID,&TS_AdjointStep);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("TSTrajectorySet", TSTRAJECTORY_CLASSID,&TSTrajectory_Set);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("TSTrajectoryGet", TSTRAJECTORY_CLASSID,&TSTrajectory_Get);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("TSTrajGetVecs",   TSTRAJECTORY_CLASSID,&TSTrajectory_GetVecs);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("TSTrajDiskWrite", TSTRAJECTORY_CLASSID,&TSTrajectory_DiskWrite);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("TSTrajDiskRead",  TSTRAJECTORY_CLASSID,&TSTrajectory_DiskRead);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("TSPseudoCmptTStp",TS_CLASSID,&TS_PseudoComputeTimeStep);CHKERRQ(ierr);
  /* Process Info */
  {
    PetscClassId  classids[4];

    classids[0] = TS_CLASSID;
    classids[1] = DMTS_CLASSID;
    classids[2] = TSADAPT_CLASSID;
    classids[3] = TSTRAJECTORY_CLASSID;
    ierr = PetscInfoProcessClass("ts", 1, classids);CHKERRQ(ierr);
    ierr = PetscInfoProcessClass("dm", 1, &classids[1]);CHKERRQ(ierr);
    ierr = PetscInfoProcessClass("tsadapt", 1, &classids[2]);CHKERRQ(ierr);
    ierr = PetscInfoProcessClass("tstrajectory", 1, &classids[3]);CHKERRQ(ierr);
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,sizeof(logList),&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrInList("ts",logList,',',&pkg);CHKERRQ(ierr);
    if (pkg) {ierr = PetscLogEventExcludeClass(TS_CLASSID);CHKERRQ(ierr);}
    ierr = PetscStrInList("dm",logList,',',&cls);CHKERRQ(ierr);
    if (pkg || cls) {ierr = PetscLogEventExcludeClass(DMTS_CLASSID);CHKERRQ(ierr);}
    ierr = PetscStrInList("tsadapt",logList,',',&cls);CHKERRQ(ierr);
    if (pkg || cls) {ierr = PetscLogEventExcludeClass(TSADAPT_CLASSID);CHKERRQ(ierr);}
    ierr = PetscStrInList("tstrajectory",logList,',',&cls);CHKERRQ(ierr);
    if (pkg || cls) {ierr = PetscLogEventExcludeClass(TSTRAJECTORY_CLASSID);CHKERRQ(ierr);}
  }
  /* Register package finalizer */
  ierr = PetscRegisterFinalize(TSFinalizePackage);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSInitializePackage();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif /* PETSC_HAVE_DYNAMIC_LIBRARIES */
