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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListDestroy(&CharacteristicList);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (CharacteristicPackageInitialized) PetscFunctionReturn(0);
  CharacteristicPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  ierr = PetscClassIdRegister("Method of Characteristics",&CHARACTERISTIC_CLASSID);CHKERRQ(ierr);
  /* Register Constructors */
  ierr = CharacteristicRegisterAll();CHKERRQ(ierr);
  /* Register Events */
  ierr = PetscLogEventRegister("MOCSetUp",         CHARACTERISTIC_CLASSID,&CHARACTERISTIC_SetUp);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MOCSolve",         CHARACTERISTIC_CLASSID,&CHARACTERISTIC_Solve);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MOCQueueSetup",    CHARACTERISTIC_CLASSID,&CHARACTERISTIC_QueueSetup);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MOCDAUpdate",      CHARACTERISTIC_CLASSID,&CHARACTERISTIC_DAUpdate);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MOCHalfTimeLocal", CHARACTERISTIC_CLASSID,&CHARACTERISTIC_HalfTimeLocal);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MOCHalfTimeRemot", CHARACTERISTIC_CLASSID,&CHARACTERISTIC_HalfTimeRemote);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MOCHalfTimeExchg", CHARACTERISTIC_CLASSID,&CHARACTERISTIC_HalfTimeExchange);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MOCFullTimeLocal", CHARACTERISTIC_CLASSID,&CHARACTERISTIC_FullTimeLocal);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MOCFullTimeRemot", CHARACTERISTIC_CLASSID,&CHARACTERISTIC_FullTimeRemote);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MOCFullTimeExchg", CHARACTERISTIC_CLASSID,&CHARACTERISTIC_FullTimeExchange);CHKERRQ(ierr);
  /* Process Info */
  {
    PetscClassId  classids[1];

    classids[0] = CHARACTERISTIC_CLASSID;
    ierr = PetscInfoProcessClass("characteristic", 1, classids);CHKERRQ(ierr);
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,sizeof(logList),&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrInList("characteristic",logList,',',&pkg);CHKERRQ(ierr);
    if (pkg) {ierr = PetscLogEventExcludeClass(CHARACTERISTIC_CLASSID);CHKERRQ(ierr);}
  }
  /* Process package finalizer */
  ierr = PetscRegisterFinalize(CharacteristicFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_DYNAMIC_LIBRARIES)
/*
  PetscDLLibraryRegister - This function is called when the dynamic library it is in is opened.

  This one registers the method of characteristics code

 */
PETSC_EXTERN PetscErrorCode PetscDLLibraryRegister_petsccharacteristic(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = CharacteristicInitializePackage();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#endif /* PETSC_HAVE_DYNAMIC_LIBRARIES */
