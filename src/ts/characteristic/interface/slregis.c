#include <petsc/private/characteristicimpl.h>

static PetscBool CharacteristicPackageInitialized = PETSC_FALSE;
/*@C
  CharacteristicFinalizePackage - This function destroys everything in the Petsc interface to the characteristics package. It is
  called from PetscFinalize().

  Level: developer

.seealso: PetscFinalize()
@*/
PetscErrorCode CharacteristicFinalizePackage(void)
{
  PetscFunctionBegin;
  CHKERRQ(PetscFunctionListDestroy(&CharacteristicList));
  CharacteristicPackageInitialized = PETSC_FALSE;
  CharacteristicRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
  CharacteristicInitializePackage - This function initializes everything in the Characteristic package. It is called
  from PetscDLLibraryRegister() when using dynamic libraries, and on the first call to CharacteristicCreate()
  when using static libraries.

  Level: developer

.seealso: PetscInitialize()
@*/
PetscErrorCode CharacteristicInitializePackage(void)
{
  char           logList[256];
  PetscBool      opt,pkg;

  PetscFunctionBegin;
  if (CharacteristicPackageInitialized) PetscFunctionReturn(0);
  CharacteristicPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  CHKERRQ(PetscClassIdRegister("Method of Characteristics",&CHARACTERISTIC_CLASSID));
  /* Register Constructors */
  CHKERRQ(CharacteristicRegisterAll());
  /* Register Events */
  CHKERRQ(PetscLogEventRegister("MOCSetUp",         CHARACTERISTIC_CLASSID,&CHARACTERISTIC_SetUp));
  CHKERRQ(PetscLogEventRegister("MOCSolve",         CHARACTERISTIC_CLASSID,&CHARACTERISTIC_Solve));
  CHKERRQ(PetscLogEventRegister("MOCQueueSetup",    CHARACTERISTIC_CLASSID,&CHARACTERISTIC_QueueSetup));
  CHKERRQ(PetscLogEventRegister("MOCDAUpdate",      CHARACTERISTIC_CLASSID,&CHARACTERISTIC_DAUpdate));
  CHKERRQ(PetscLogEventRegister("MOCHalfTimeLocal", CHARACTERISTIC_CLASSID,&CHARACTERISTIC_HalfTimeLocal));
  CHKERRQ(PetscLogEventRegister("MOCHalfTimeRemot", CHARACTERISTIC_CLASSID,&CHARACTERISTIC_HalfTimeRemote));
  CHKERRQ(PetscLogEventRegister("MOCHalfTimeExchg", CHARACTERISTIC_CLASSID,&CHARACTERISTIC_HalfTimeExchange));
  CHKERRQ(PetscLogEventRegister("MOCFullTimeLocal", CHARACTERISTIC_CLASSID,&CHARACTERISTIC_FullTimeLocal));
  CHKERRQ(PetscLogEventRegister("MOCFullTimeRemot", CHARACTERISTIC_CLASSID,&CHARACTERISTIC_FullTimeRemote));
  CHKERRQ(PetscLogEventRegister("MOCFullTimeExchg", CHARACTERISTIC_CLASSID,&CHARACTERISTIC_FullTimeExchange));
  /* Process Info */
  {
    PetscClassId  classids[1];

    classids[0] = CHARACTERISTIC_CLASSID;
    CHKERRQ(PetscInfoProcessClass("characteristic", 1, classids));
  }
  /* Process summary exclusions */
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,sizeof(logList),&opt));
  if (opt) {
    CHKERRQ(PetscStrInList("characteristic",logList,',',&pkg));
    if (pkg) CHKERRQ(PetscLogEventExcludeClass(CHARACTERISTIC_CLASSID));
  }
  /* Process package finalizer */
  CHKERRQ(PetscRegisterFinalize(CharacteristicFinalizePackage));
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_DYNAMIC_LIBRARIES)
/*
  PetscDLLibraryRegister - This function is called when the dynamic library it is in is opened.

  This one registers the method of characteristics code

 */
PETSC_EXTERN PetscErrorCode PetscDLLibraryRegister_petsccharacteristic(void)
{
  PetscFunctionBegin;
  CHKERRQ(CharacteristicInitializePackage());
  PetscFunctionReturn(0);
}

#endif /* PETSC_HAVE_DYNAMIC_LIBRARIES */
