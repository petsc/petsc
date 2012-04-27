
#include <petsc-private/isimpl.h>    /*I "petscis.h"  I*/

PetscFList ISList                       = PETSC_NULL;
PetscBool  ISRegisterAllCalled          = PETSC_FALSE;

#undef __FUNCT__  
#define __FUNCT__ "ISCreate" 
/*@
   ISCreate - Creates an index set object.

   Collective on MPI_Comm

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

  Concepts: index sets^creating
  Concepts: IS^creating

.seealso: ISCreateGeneral(), ISCreateStride(), ISCreateBlock(), ISAllGather()
@*/
PetscErrorCode  ISCreate(MPI_Comm comm,IS *is)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(is,2);
#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = ISInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif

  ierr = PetscHeaderCreate(*is,_p_IS,struct _ISOps,IS_CLASSID,-1,"IS","Index Set","IS",comm,ISDestroy,ISView);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISSetType"
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
PetscErrorCode  ISSetType(IS is, const ISType method)
{
  PetscErrorCode (*r)(IS);
  PetscBool      match;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is, IS_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject) is, method, &match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  if (!ISRegisterAllCalled) {ierr = ISRegisterAll(PETSC_NULL);CHKERRQ(ierr);}
  ierr = PetscFListFind(ISList, ((PetscObject)is)->comm, method,PETSC_TRUE,(void (**)(void)) &r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown IS type: %s", method);
  if (is->ops->destroy) {
    ierr = (*is->ops->destroy)(is);CHKERRQ(ierr);
    is->ops->destroy = PETSC_NULL;
  }
  ierr = (*r)(is);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)is,method);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISGetType"
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
PetscErrorCode  ISGetType(IS is, const ISType *type)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is, IS_CLASSID,1);
  PetscValidCharPointer(type,2);
  if (!ISRegisterAllCalled) {
    ierr = ISRegisterAll(PETSC_NULL);CHKERRQ(ierr);
  }
  *type = ((PetscObject)is)->type_name;
  PetscFunctionReturn(0);
}


/*--------------------------------------------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "ISRegister"
/*@C
  ISRegister - See ISRegisterDynamic()

  Level: advanced
@*/
PetscErrorCode  ISRegister(const char sname[], const char path[], const char name[], PetscErrorCode (*function)(IS))
{
  char           fullname[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscStrcpy(fullname, path);CHKERRQ(ierr);
  ierr = PetscStrcat(fullname, ":");CHKERRQ(ierr);
  ierr = PetscStrcat(fullname, name);CHKERRQ(ierr);
  ierr = PetscFListAdd(&ISList, sname, fullname, (void (*)(void)) function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*--------------------------------------------------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "ISRegisterDestroy"
/*@C
   ISRegisterDestroy - Frees the list of IS methods that were registered by ISRegister()/ISRegisterDynamic().

   Not Collective

   Level: advanced

.keywords: IS, register, destroy
.seealso: ISRegister(), ISRegisterAll(), ISRegisterDynamic()
@*/
PetscErrorCode  ISRegisterDestroy(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFListDestroy(&ISList);CHKERRQ(ierr);
  ISRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

