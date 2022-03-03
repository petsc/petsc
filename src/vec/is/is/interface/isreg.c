
#include <petsc/private/isimpl.h>    /*I "petscis.h"  I*/

PetscFunctionList ISList              = NULL;
PetscBool         ISRegisterAllCalled = PETSC_FALSE;

/*@
   ISCreate - Creates an index set object.

   Collective

   Input Parameters:
.  comm - the MPI communicator

   Output Parameter:
.  is - the new index set

   Notes:
   When the communicator is not MPI_COMM_SELF, the operations on IS are NOT
   conceptually the same as MPI_Group operations. The IS are then
   distributed sets of indices and thus certain operations on them are
   collective.

   Level: beginner

.seealso: ISCreateGeneral(), ISCreateStride(), ISCreateBlock(), ISAllGather()
@*/
PetscErrorCode  ISCreate(MPI_Comm comm,IS *is)
{
  PetscFunctionBegin;
  PetscValidPointer(is,2);
  CHKERRQ(ISInitializePackage());

  CHKERRQ(PetscHeaderCreate(*is,IS_CLASSID,"IS","Index Set","IS",comm,ISDestroy,ISView));
  CHKERRQ(PetscLayoutCreate(comm, &(*is)->map));
  PetscFunctionReturn(0);
}

/*@C
  ISSetType - Builds a index set, for a particular implementation.

  Collective on IS

  Input Parameters:
+ is    - The index set object
- method - The name of the index set type

  Options Database Key:
. -is_type <type> - Sets the index set type; use -help for a list of available types

  Notes:
  See "petsc/include/petscis.h" for available istor types (for instance, ISGENERAL, ISSTRIDE, or ISBLOCK).

  Use ISDuplicate() to make a duplicate

  Level: intermediate

.seealso: ISGetType(), ISCreate()
@*/
PetscErrorCode  ISSetType(IS is, ISType method)
{
  PetscErrorCode (*r)(IS);
  PetscBool      match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is, IS_CLASSID,1);
  CHKERRQ(PetscObjectTypeCompare((PetscObject) is, method, &match));
  if (match) PetscFunctionReturn(0);

  CHKERRQ(ISRegisterAll());
  CHKERRQ(PetscFunctionListFind(ISList,method,&r));
  PetscCheck(r,PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown IS type: %s", method);
  if (is->ops->destroy) {
    CHKERRQ((*is->ops->destroy)(is));
    is->ops->destroy = NULL;
  }
  CHKERRQ((*r)(is));
  CHKERRQ(PetscObjectChangeTypeName((PetscObject)is,method));
  PetscFunctionReturn(0);
}

/*@C
  ISGetType - Gets the index set type name (as a string) from the IS.

  Not Collective

  Input Parameter:
. is  - The index set

  Output Parameter:
. type - The index set type name

  Level: intermediate

.seealso: ISSetType(), ISCreate()
@*/
PetscErrorCode  ISGetType(IS is, ISType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(is, IS_CLASSID,1);
  PetscValidPointer(type,2);
  if (!ISRegisterAllCalled) {
    CHKERRQ(ISRegisterAll());
  }
  *type = ((PetscObject)is)->type_name;
  PetscFunctionReturn(0);
}

/*--------------------------------------------------------------------------------------------------------------------*/

/*@C
  ISRegister - Adds a new index set implementation

  Not Collective

  Input Parameters:
+ name        - The name of a new user-defined creation routine
- create_func - The creation routine itself

  Notes:
  ISRegister() may be called multiple times to add several user-defined vectors

  Sample usage:
.vb
    ISRegister("my_is_name",  MyISCreate);
.ve

  Then, your vector type can be chosen with the procedural interface via
.vb
    ISCreate(MPI_Comm, IS *);
    ISSetType(IS,"my_is_name");
.ve
   or at runtime via the option
.vb
    -is_type my_is_name
.ve

  This is no ISSetFromOptions() and the current implementations do not have a way to dynamically determine type, so
  dynamic registration of custom IS types will be of limited use to users.

  Level: developer

.seealso: ISRegisterAll(), ISRegisterDestroy(), ISRegister()

  Level: advanced
@*/
PetscErrorCode  ISRegister(const char sname[], PetscErrorCode (*function)(IS))
{
  PetscFunctionBegin;
  CHKERRQ(ISInitializePackage());
  CHKERRQ(PetscFunctionListAdd(&ISList,sname,function));
  PetscFunctionReturn(0);
}
