
#include <../src/vec/is/ao/aoimpl.h>    /*I "petscao.h"  I*/

PetscFunctionList AOList              = NULL;
PetscBool         AORegisterAllCalled = PETSC_FALSE;

/*@C
  AOSetType - Builds an application ordering for a particular implementation.

  Collective on AO

  Input Parameters:
+ ao    - The AO object
- method - The name of the AO type

  Options Database Key:
. -ao_type <type> - Sets the AO type; use -help for a list of available types

  Notes:
  See "petsc/include/petscao.h" for available AO types (for instance, AOBASIC and AOMEMORYSCALABLE).

  Level: intermediate

.seealso: AOGetType(), AOCreate()
@*/
PetscErrorCode  AOSetType(AO ao, AOType method)
{
  PetscErrorCode (*r)(AO);
  PetscBool      match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ao, AO_CLASSID,1);
  CHKERRQ(PetscObjectTypeCompare((PetscObject)ao, method, &match));
  if (match) PetscFunctionReturn(0);

  CHKERRQ(AORegisterAll());
  CHKERRQ(PetscFunctionListFind(AOList,method,&r));
  PetscCheck(r,PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown AO type: %s", method);
  if (ao->ops->destroy) {
    CHKERRQ((*ao->ops->destroy)(ao));
    ao->ops->destroy = NULL;
  }

  CHKERRQ((*r)(ao));
  PetscFunctionReturn(0);
}

/*@C
  AOGetType - Gets the AO type name (as a string) from the AO.

  Not Collective

  Input Parameter:
. ao  - The vector

  Output Parameter:
. type - The AO type name

  Level: intermediate

.seealso: AOSetType(), AOCreate()
@*/
PetscErrorCode  AOGetType(AO ao, AOType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ao, AO_CLASSID,1);
  PetscValidPointer(type,2);
  CHKERRQ(AORegisterAll());
  *type = ((PetscObject)ao)->type_name;
  PetscFunctionReturn(0);
}

/*--------------------------------------------------------------------------------------------------------------------*/

/*@C
  AORegister - Register  an application ordering method

    Not Collective

   Input Parameters:
+   sname - the name of the AO scheme
-   function - the create routine for the application ordering method

  Level: advanced

.seealso:   AOCreate(), AORegisterAll(), AOBASIC, AOADVANCED, AOMAPPING, AOMEMORYSCALABLE

@*/
PetscErrorCode  AORegister(const char sname[], PetscErrorCode (*function)(AO))
{
  PetscFunctionBegin;
  CHKERRQ(AOInitializePackage());
  CHKERRQ(PetscFunctionListAdd(&AOList,sname,function));
  PetscFunctionReturn(0);
}
