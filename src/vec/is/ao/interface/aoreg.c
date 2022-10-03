
#include <../src/vec/is/ao/aoimpl.h> /*I "petscao.h"  I*/

PetscFunctionList AOList              = NULL;
PetscBool         AORegisterAllCalled = PETSC_FALSE;

/*@C
  AOSetType - Builds an application ordering for a particular `AOType`

  Collective on ao

  Input Parameters:
+ ao    - The `AO` object
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
  if (match) PetscFunctionReturn(0);

  PetscCall(AORegisterAll());
  PetscCall(PetscFunctionListFind(AOList, method, &r));
  PetscCheck(r, PETSC_COMM_SELF, PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown AO type: %s", method);
  PetscTryTypeMethod(ao, destroy);
  ao->ops->destroy = NULL;

  PetscCall((*r)(ao));
  PetscFunctionReturn(0);
}

/*@C
  AOGetType - Gets the `AO` type name (as a string) from the AO.

  Not Collective

  Input Parameter:
. ao  - The vector

  Output Parameter:
. type - The `AO` type name

  Level: intermediate

.seealso: `AO`, `AOType`, `AOSetType()`, `AOCreate()`
@*/
PetscErrorCode AOGetType(AO ao, AOType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ao, AO_CLASSID, 1);
  PetscValidPointer(type, 2);
  PetscCall(AORegisterAll());
  *type = ((PetscObject)ao)->type_name;
  PetscFunctionReturn(0);
}

/*--------------------------------------------------------------------------------------------------------------------*/

/*@C
  AORegister - Register  an application ordering method

    Not Collective

   Input Parameters:
+   sname - the name (`AOType`) of the `AO` scheme
-   function - the create routine for the application ordering method

  Level: advanced

.seealso: `AO`, `AOType`, `AOCreate()`, `AORegisterAll()`, `AOBASIC`, `AOADVANCED`, `AOMAPPING`, `AOMEMORYSCALABLE`
@*/
PetscErrorCode AORegister(const char sname[], PetscErrorCode (*function)(AO))
{
  PetscFunctionBegin;
  PetscCall(AOInitializePackage());
  PetscCall(PetscFunctionListAdd(&AOList, sname, function));
  PetscFunctionReturn(0);
}
