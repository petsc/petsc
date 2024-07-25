#include <../src/vec/is/ao/aoimpl.h> /*I "petscao.h"  I*/

static PetscBool AOPackageInitialized = PETSC_FALSE;
static PetscBool AORegisterAllCalled  = PETSC_FALSE;

/*@C
  AOFinalizePackage - This function finalizes everything in the `AO` package. It is called
  from `PetscFinalize()`.

  Level: developer

.seealso: `AOInitializePackage()`, `PetscInitialize()`
@*/
PetscErrorCode AOFinalizePackage(void)
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListDestroy(&AOList));
  AOPackageInitialized = PETSC_FALSE;
  AORegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  AOInitializePackage - This function initializes everything in the `AO` package. It is called
  from `PetscDLLibraryRegister_petscvec()` when using dynamic libraries, and on the first call to `AOCreate()`
  when using static or shared libraries.

  Level: developer

.seealso: `AOFinalizePackage()`, `PetscInitialize()`
@*/
PetscErrorCode AOInitializePackage(void)
{
  char      logList[256];
  PetscBool opt, pkg;

  PetscFunctionBegin;
  if (AOPackageInitialized) PetscFunctionReturn(PETSC_SUCCESS);
  AOPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  PetscCall(PetscClassIdRegister("Application Order", &AO_CLASSID));
  /* Register Constructors */
  PetscCall(AORegisterAll());
  /* Register Events */
  PetscCall(PetscLogEventRegister("AOPetscToApplication", AO_CLASSID, &AO_PetscToApplication));
  PetscCall(PetscLogEventRegister("AOApplicationToPetsc", AO_CLASSID, &AO_ApplicationToPetsc));
  /* Process Info */
  {
    PetscClassId classids[1];

    classids[0] = AO_CLASSID;
    PetscCall(PetscInfoProcessClass("ao", 1, classids));
  }
  /* Process summary exclusions */
  PetscCall(PetscOptionsGetString(NULL, NULL, "-log_exclude", logList, sizeof(logList), &opt));
  if (opt) {
    PetscCall(PetscStrInList("ao", logList, ',', &pkg));
    if (pkg) PetscCall(PetscLogEventExcludeClass(AO_CLASSID));
  }
  /* Register package finalizer */
  PetscCall(PetscRegisterFinalize(AOFinalizePackage));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  AOSetType - Builds an application ordering for a particular `AOType`

  Collective

  Input Parameters:
+ ao     - The `AO` object
- method - The name of the AO type

  Options Database Key:
. -ao_type <type> - Sets the `AO` type; use -help for a list of available types

  Level: intermediate

  Notes:
  See "petsc/include/petscao.h" for available AO types (for instance, `AOBASIC` and `AOMEMORYSCALABLE`).

  `AO` are usually created via the convenience routines such as `AOCreateBasic()` or `AOCreateMemoryScalable()`

.seealso: `AO`, `AOType`, `AOCreateBasic()`, `AOCreateMemoryScalable()`, `AOGetType()`, `AOCreate()`
@*/
PetscErrorCode AOSetType(AO ao, AOType method)
{
  PetscErrorCode (*r)(AO);
  PetscBool match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ao, AO_CLASSID, 1);
  PetscCall(PetscObjectTypeCompare((PetscObject)ao, method, &match));
  if (match) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(AORegisterAll());
  PetscCall(PetscFunctionListFind(AOList, method, &r));
  PetscCheck(r, PetscObjectComm((PetscObject)ao), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown AO type: %s", method);
  PetscTryTypeMethod(ao, destroy);
  ao->ops->destroy = NULL;

  PetscCall((*r)(ao));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  AOGetType - Gets the `AO` type name (as a string) from the AO.

  Not Collective

  Input Parameter:
. ao - The vector

  Output Parameter:
. type - The `AO` type name

  Level: intermediate

.seealso: `AO`, `AOType`, `AOSetType()`, `AOCreate()`
@*/
PetscErrorCode AOGetType(AO ao, AOType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ao, AO_CLASSID, 1);
  PetscAssertPointer(type, 2);
  PetscCall(AORegisterAll());
  *type = ((PetscObject)ao)->type_name;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscFunctionList AOList = NULL;

/*@C
  AORegister - Register  an application ordering method

  Not Collective, No Fortran Support

  Input Parameters:
+ sname    - the name (`AOType`) of the `AO` scheme
- function - the create routine for the application ordering method

  Level: advanced

.seealso: `AO`, `AOType`, `AOCreate()`, `AORegisterAll()`, `AOBASIC`, `AOADVANCED`, `AOMAPPING`, `AOMEMORYSCALABLE`
@*/
PetscErrorCode AORegister(const char sname[], PetscErrorCode (*function)(AO))
{
  PetscFunctionBegin;
  PetscCall(AOInitializePackage());
  PetscCall(PetscFunctionListAdd(&AOList, sname, function));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode AOCreate_Basic(AO ao);
PETSC_INTERN PetscErrorCode AOCreate_MemoryScalable(AO ao);

/*@C
  AORegisterAll - Registers all of the application ordering components in the `AO` package.

  Not Collective

  Level: advanced

.seealso: `AO`, `AOType`, `AORegister()`, `AORegisterDestroy()`
@*/
PetscErrorCode AORegisterAll(void)
{
  PetscFunctionBegin;
  if (AORegisterAllCalled) PetscFunctionReturn(PETSC_SUCCESS);
  AORegisterAllCalled = PETSC_TRUE;

  PetscCall(AORegister(AOBASIC, AOCreate_Basic));
  PetscCall(AORegister(AOMEMORYSCALABLE, AOCreate_MemoryScalable));
  PetscFunctionReturn(PETSC_SUCCESS);
}
