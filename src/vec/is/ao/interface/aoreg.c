
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ao, AO_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)ao, method, &match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  ierr = AORegisterAll();CHKERRQ(ierr);
  ierr = PetscFunctionListFind(AOList,method,&r);CHKERRQ(ierr);
  PetscAssertFalse(!r,PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown AO type: %s", method);
  if (ao->ops->destroy) {
    ierr             = (*ao->ops->destroy)(ao);CHKERRQ(ierr);
    ao->ops->destroy = NULL;
  }

  ierr = (*r)(ao);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ao, AO_CLASSID,1);
  PetscValidPointer(type,2);
  ierr = AORegisterAll();CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = AOInitializePackage();CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&AOList,sname,function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

