#include <petsc/private/sfimpl.h>

static PetscBool PetscSFPackageInitialized = PETSC_FALSE;

PetscClassId PETSCSF_CLASSID;

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
   PetscSFInitializePackage - Initialize `PetscSF` package

   Logically Collective

   Level: developer

.seealso: `PetscSF`, `PetscSFFinalizePackage()`
@*/
PetscErrorCode PetscSFInitializePackage(void)
{
  char      logList[256];
  PetscBool opt, pkg;

  PetscFunctionBegin;
  if (PetscSFPackageInitialized) PetscFunctionReturn(0);
  PetscSFPackageInitialized = PETSC_TRUE;
  /* Register Class */
  PetscCall(PetscClassIdRegister("Star Forest Graph", &PETSCSF_CLASSID));
  /* Register Constructors */
  PetscCall(PetscSFRegisterAll());
  /* Register Events */
  PetscCall(PetscLogEventRegister("SFSetGraph", PETSCSF_CLASSID, &PETSCSF_SetGraph));
  PetscCall(PetscLogEventRegister("SFSetUp", PETSCSF_CLASSID, &PETSCSF_SetUp));
  PetscCall(PetscLogEventRegister("SFBcastBegin", PETSCSF_CLASSID, &PETSCSF_BcastBegin));
  PetscCall(PetscLogEventRegister("SFBcastEnd", PETSCSF_CLASSID, &PETSCSF_BcastEnd));
  PetscCall(PetscLogEventRegister("SFReduceBegin", PETSCSF_CLASSID, &PETSCSF_ReduceBegin));
  PetscCall(PetscLogEventRegister("SFReduceEnd", PETSCSF_CLASSID, &PETSCSF_ReduceEnd));
  PetscCall(PetscLogEventRegister("SFFetchOpBegin", PETSCSF_CLASSID, &PETSCSF_FetchAndOpBegin));
  PetscCall(PetscLogEventRegister("SFFetchOpEnd", PETSCSF_CLASSID, &PETSCSF_FetchAndOpEnd));
  PetscCall(PetscLogEventRegister("SFCreateEmbed", PETSCSF_CLASSID, &PETSCSF_EmbedSF));
  PetscCall(PetscLogEventRegister("SFDistSection", PETSCSF_CLASSID, &PETSCSF_DistSect));
  PetscCall(PetscLogEventRegister("SFSectionSF", PETSCSF_CLASSID, &PETSCSF_SectSF));
  PetscCall(PetscLogEventRegister("SFRemoteOff", PETSCSF_CLASSID, &PETSCSF_RemoteOff));
  PetscCall(PetscLogEventRegister("SFPack", PETSCSF_CLASSID, &PETSCSF_Pack));
  PetscCall(PetscLogEventRegister("SFUnpack", PETSCSF_CLASSID, &PETSCSF_Unpack));
  /* Process Info */
  {
    PetscClassId classids[1];

    classids[0] = PETSCSF_CLASSID;
    PetscCall(PetscInfoProcessClass("sf", 1, classids));
  }
  /* Process summary exclusions */
  PetscCall(PetscOptionsGetString(NULL, NULL, "-log_exclude", logList, sizeof(logList), &opt));
  if (opt) {
    PetscCall(PetscStrInList("sf", logList, ',', &pkg));
    if (pkg) PetscCall(PetscLogEventExcludeClass(PETSCSF_CLASSID));
  }
  /* Register package finalizer */
  PetscCall(PetscRegisterFinalize(PetscSFFinalizePackage));
  PetscFunctionReturn(0);
}

/*@C
   PetscSFFinalizePackage - Finalize `PetscSF` package, it is called from `PetscFinalize()`

   Logically Collective

   Level: developer

.seealso: `PetscSF`, `PetscSFInitializePackage()`
@*/
PetscErrorCode PetscSFFinalizePackage(void)
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListDestroy(&PetscSFList));
  PetscSFPackageInitialized = PETSC_FALSE;
  PetscSFRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(0);
}
