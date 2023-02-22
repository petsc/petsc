#include <petsc/private/characteristicimpl.h>

static PetscBool CharacteristicPackageInitialized = PETSC_FALSE;
/*@C
  CharacteristicFinalizePackage - This function destroys everything in the `Characteristics` package. It is
  called from `PetscFinalize()`.

  Level: developer

.seealso: [](chapter_ts), `PetscFinalize()`, `CharacteristicInitializePackage()`
@*/
PetscErrorCode CharacteristicFinalizePackage(void)
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListDestroy(&CharacteristicList));
  CharacteristicPackageInitialized = PETSC_FALSE;
  CharacteristicRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  CharacteristicInitializePackage - This function initializes everything in the Characteristic package. It is called
  from PetscDLLibraryRegister() when using dynamic libraries, and on the first call to CharacteristicCreate()
  when using static libraries.

  Level: developer

.seealso: [](chapter_ts), `PetscInitialize()`, `CharacteristicFinalizePackage()`
@*/
PetscErrorCode CharacteristicInitializePackage(void)
{
  char      logList[256];
  PetscBool opt, pkg;

  PetscFunctionBegin;
  if (CharacteristicPackageInitialized) PetscFunctionReturn(PETSC_SUCCESS);
  CharacteristicPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  PetscCall(PetscClassIdRegister("Method of Characteristics", &CHARACTERISTIC_CLASSID));
  /* Register Constructors */
  PetscCall(CharacteristicRegisterAll());
  /* Register Events */
  PetscCall(PetscLogEventRegister("MOCSetUp", CHARACTERISTIC_CLASSID, &CHARACTERISTIC_SetUp));
  PetscCall(PetscLogEventRegister("MOCSolve", CHARACTERISTIC_CLASSID, &CHARACTERISTIC_Solve));
  PetscCall(PetscLogEventRegister("MOCQueueSetup", CHARACTERISTIC_CLASSID, &CHARACTERISTIC_QueueSetup));
  PetscCall(PetscLogEventRegister("MOCDAUpdate", CHARACTERISTIC_CLASSID, &CHARACTERISTIC_DAUpdate));
  PetscCall(PetscLogEventRegister("MOCHalfTimeLocal", CHARACTERISTIC_CLASSID, &CHARACTERISTIC_HalfTimeLocal));
  PetscCall(PetscLogEventRegister("MOCHalfTimeRemot", CHARACTERISTIC_CLASSID, &CHARACTERISTIC_HalfTimeRemote));
  PetscCall(PetscLogEventRegister("MOCHalfTimeExchg", CHARACTERISTIC_CLASSID, &CHARACTERISTIC_HalfTimeExchange));
  PetscCall(PetscLogEventRegister("MOCFullTimeLocal", CHARACTERISTIC_CLASSID, &CHARACTERISTIC_FullTimeLocal));
  PetscCall(PetscLogEventRegister("MOCFullTimeRemot", CHARACTERISTIC_CLASSID, &CHARACTERISTIC_FullTimeRemote));
  PetscCall(PetscLogEventRegister("MOCFullTimeExchg", CHARACTERISTIC_CLASSID, &CHARACTERISTIC_FullTimeExchange));
  /* Process Info */
  {
    PetscClassId classids[1];

    classids[0] = CHARACTERISTIC_CLASSID;
    PetscCall(PetscInfoProcessClass("characteristic", 1, classids));
  }
  /* Process summary exclusions */
  PetscCall(PetscOptionsGetString(NULL, NULL, "-log_exclude", logList, sizeof(logList), &opt));
  if (opt) {
    PetscCall(PetscStrInList("characteristic", logList, ',', &pkg));
    if (pkg) PetscCall(PetscLogEventExcludeClass(CHARACTERISTIC_CLASSID));
  }
  /* Process package finalizer */
  PetscCall(PetscRegisterFinalize(CharacteristicFinalizePackage));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if defined(PETSC_HAVE_DYNAMIC_LIBRARIES)
/*
  PetscDLLibraryRegister - This function is called when the dynamic library it is in is opened.

  This one registers the method of characteristics code
 */
PETSC_EXTERN PetscErrorCode PetscDLLibraryRegister_petsccharacteristic(void)
{
  PetscFunctionBegin;
  PetscCall(CharacteristicInitializePackage());
  PetscFunctionReturn(PETSC_SUCCESS);
}

#endif /* PETSC_HAVE_DYNAMIC_LIBRARIES */
