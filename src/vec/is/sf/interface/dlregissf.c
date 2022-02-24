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

  PetscFunctionBegin;
  if (PetscSFPackageInitialized) PetscFunctionReturn(0);
  PetscSFPackageInitialized = PETSC_TRUE;
  /* Register Class */
  CHKERRQ(PetscClassIdRegister("Star Forest Graph", &PETSCSF_CLASSID));
  /* Register Constructors */
  CHKERRQ(PetscSFRegisterAll());
  /* Register Events */
  CHKERRQ(PetscLogEventRegister("SFSetGraph"     , PETSCSF_CLASSID, &PETSCSF_SetGraph));
  CHKERRQ(PetscLogEventRegister("SFSetUp"        , PETSCSF_CLASSID, &PETSCSF_SetUp));
  CHKERRQ(PetscLogEventRegister("SFBcastBegin"   , PETSCSF_CLASSID, &PETSCSF_BcastBegin));
  CHKERRQ(PetscLogEventRegister("SFBcastEnd"     , PETSCSF_CLASSID, &PETSCSF_BcastEnd));
  CHKERRQ(PetscLogEventRegister("SFReduceBegin"  , PETSCSF_CLASSID, &PETSCSF_ReduceBegin));
  CHKERRQ(PetscLogEventRegister("SFReduceEnd"    , PETSCSF_CLASSID, &PETSCSF_ReduceEnd));
  CHKERRQ(PetscLogEventRegister("SFFetchOpBegin" , PETSCSF_CLASSID, &PETSCSF_FetchAndOpBegin));
  CHKERRQ(PetscLogEventRegister("SFFetchOpEnd"   , PETSCSF_CLASSID, &PETSCSF_FetchAndOpEnd));
  CHKERRQ(PetscLogEventRegister("SFCreateEmbed"  , PETSCSF_CLASSID, &PETSCSF_EmbedSF));
  CHKERRQ(PetscLogEventRegister("SFDistSection"  , PETSCSF_CLASSID, &PETSCSF_DistSect));
  CHKERRQ(PetscLogEventRegister("SFSectionSF"    , PETSCSF_CLASSID, &PETSCSF_SectSF));
  CHKERRQ(PetscLogEventRegister("SFRemoteOff"    , PETSCSF_CLASSID, &PETSCSF_RemoteOff));
  CHKERRQ(PetscLogEventRegister("SFPack"         , PETSCSF_CLASSID, &PETSCSF_Pack));
  CHKERRQ(PetscLogEventRegister("SFUnpack"       , PETSCSF_CLASSID, &PETSCSF_Unpack));
  /* Process Info */
  {
    PetscClassId  classids[1];

    classids[0] = PETSCSF_CLASSID;
    CHKERRQ(PetscInfoProcessClass("sf", 1, classids));
  }
  /* Process summary exclusions */
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,sizeof(logList),&opt));
  if (opt) {
    CHKERRQ(PetscStrInList("sf",logList,',',&pkg));
    if (pkg) CHKERRQ(PetscLogEventExcludeClass(PETSCSF_CLASSID));
  }
  /* Register package finalizer */
  CHKERRQ(PetscRegisterFinalize(PetscSFFinalizePackage));
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
  PetscFunctionBegin;
  CHKERRQ(PetscFunctionListDestroy(&PetscSFList));
  PetscSFPackageInitialized = PETSC_FALSE;
  PetscSFRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(0);
}
