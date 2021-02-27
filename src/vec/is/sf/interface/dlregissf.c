#include <petsc/private/sfimpl.h>

static PetscBool PetscSFPackageInitialized = PETSC_FALSE;

PetscClassId  PETSCSF_CLASSID;

PetscLogEvent PETSCSF_SetGraph;
PetscLogEvent PETSCSF_SetUp;
PetscLogEvent PETSCSF_BcastBegin;
PetscLogEvent PETSCSF_BcastEnd;
PetscLogEvent PETSCSF_ReduceBegin;
PetscLogEvent PETSCSF_ReduceEnd;
PetscLogEvent PETSCSF_FetchAndOpBegin;
PetscLogEvent PETSCSF_FetchAndOpEnd;
PetscLogEvent PETSCSF_EmbedSF;
PetscLogEvent PETSCSF_DistSect;
PetscLogEvent PETSCSF_SectSF;
PetscLogEvent PETSCSF_RemoteOff;
PetscLogEvent PETSCSF_Pack;
PetscLogEvent PETSCSF_Unpack;

/*@C
   PetscSFInitializePackage - Initialize SF package

   Logically Collective

   Level: developer

.seealso: PetscSFFinalizePackage()
@*/
PetscErrorCode PetscSFInitializePackage(void)
{
  char           logList[256];
  PetscBool      opt,pkg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscSFPackageInitialized) PetscFunctionReturn(0);
  PetscSFPackageInitialized = PETSC_TRUE;
  /* Register Class */
  ierr = PetscClassIdRegister("Star Forest Graph", &PETSCSF_CLASSID);CHKERRQ(ierr);
  /* Register Constructors */
  ierr = PetscSFRegisterAll();CHKERRQ(ierr);
  /* Register Events */
  ierr = PetscLogEventRegister("SFSetGraph"     , PETSCSF_CLASSID, &PETSCSF_SetGraph);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("SFSetUp"        , PETSCSF_CLASSID, &PETSCSF_SetUp);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("SFBcastBegin"   , PETSCSF_CLASSID, &PETSCSF_BcastBegin);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("SFBcastEnd"     , PETSCSF_CLASSID, &PETSCSF_BcastEnd);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("SFReduceBegin"  , PETSCSF_CLASSID, &PETSCSF_ReduceBegin);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("SFReduceEnd"    , PETSCSF_CLASSID, &PETSCSF_ReduceEnd);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("SFFetchOpBegin" , PETSCSF_CLASSID, &PETSCSF_FetchAndOpBegin);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("SFFetchOpEnd"   , PETSCSF_CLASSID, &PETSCSF_FetchAndOpEnd);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("SFCreateEmbed"  , PETSCSF_CLASSID, &PETSCSF_EmbedSF);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("SFDistSection"  , PETSCSF_CLASSID, &PETSCSF_DistSect);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("SFSectionSF"    , PETSCSF_CLASSID, &PETSCSF_SectSF);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("SFRemoteOff"    , PETSCSF_CLASSID, &PETSCSF_RemoteOff);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("SFPack"         , PETSCSF_CLASSID, &PETSCSF_Pack);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("SFUnpack"       , PETSCSF_CLASSID, &PETSCSF_Unpack);CHKERRQ(ierr);
  /* Process Info */
  {
    PetscClassId  classids[1];

    classids[0] = PETSCSF_CLASSID;
    ierr = PetscInfoProcessClass("sf", 1, classids);CHKERRQ(ierr);
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,sizeof(logList),&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrInList("sf",logList,',',&pkg);CHKERRQ(ierr);
    if (pkg) {ierr = PetscLogEventExcludeClass(PETSCSF_CLASSID);CHKERRQ(ierr);}
  }
  /* Register package finalizer */
  ierr = PetscRegisterFinalize(PetscSFFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscSFFinalizePackage - Finalize PetscSF package, it is called from PetscFinalize()

   Logically Collective

   Level: developer

.seealso: PetscSFInitializePackage()
@*/
PetscErrorCode PetscSFFinalizePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListDestroy(&PetscSFList);CHKERRQ(ierr);
  PetscSFPackageInitialized = PETSC_FALSE;
  PetscSFRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(0);
}
