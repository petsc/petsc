#define PETSCDM_DLL

#include "private/daimpl.h"    /*I "petscda.h"  I*/

PetscFList DAList                       = PETSC_NULL;
PetscTruth DARegisterAllCalled          = PETSC_FALSE;

#undef __FUNCT__  
#define __FUNCT__ "DASetType"
/*@C
  DASetType - Builds a DA, for a particular DA implementation.

  Collective on DA

  Input Parameters:
+ da     - The DA object
- method - The name of the DA type

  Options Database Key:
. -da_type <type> - Sets the DA type; use -help for a list of available types

  Notes:
  See "petsc/include/petscda.h" for available DA types (for instance, DA1D, DA2D, or DA3D).

  Level: intermediate

.keywords: DA, set, type
.seealso: DAGetType(), DACreate()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DASetType(DA da, const DAType method)
{
  PetscErrorCode (*r)(DA);
  PetscTruth     match;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, DM_COOKIE,1);
  ierr = PetscTypeCompare((PetscObject) da, method, &match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  if (!DARegisterAllCalled) {ierr = DARegisterAll(PETSC_NULL);CHKERRQ(ierr);}
  ierr = PetscFListFind(DAList, ((PetscObject)da)->comm, method,(void (**)(void)) &r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown DA type: %s", method);

  if (da->ops->destroy) {
    ierr = (*da->ops->destroy)(da);CHKERRQ(ierr);
  }
  ierr = (*r)(da);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)da,method);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DAGetType"
/*@C
  DAGetType - Gets the DA type name (as a string) from the DA.

  Not Collective

  Input Parameter:
. da  - The DA

  Output Parameter:
. type - The DA type name

  Level: intermediate

.keywords: DA, get, type, name
.seealso: DASetType(), DACreate()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DAGetType(DA da, const DAType *type)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, DM_COOKIE,1);
  PetscValidCharPointer(type,2);
  if (!DARegisterAllCalled) {
    ierr = DARegisterAll(PETSC_NULL);CHKERRQ(ierr);
  }
  *type = ((PetscObject)da)->type_name;
  PetscFunctionReturn(0);
}


/*--------------------------------------------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "DARegister"
/*@C
  DARegister - See DARegisterDynamic()

  Level: advanced
@*/
PetscErrorCode PETSCDM_DLLEXPORT DARegister(const char sname[], const char path[], const char name[], PetscErrorCode (*function)(DA))
{
  char fullname[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscStrcpy(fullname, path);CHKERRQ(ierr);
  ierr = PetscStrcat(fullname, ":");CHKERRQ(ierr);
  ierr = PetscStrcat(fullname, name);CHKERRQ(ierr);
  ierr = PetscFListAdd(&DAList, sname, fullname, (void (*)(void)) function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*--------------------------------------------------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "DARegisterDestroy"
/*@C
   DARegisterDestroy - Frees the list of DA methods that were registered by DARegister()/DARegisterDynamic().

   Not Collective

   Level: advanced

.keywords: DA, register, destroy
.seealso: DARegister(), DARegisterAll(), DARegisterDynamic()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DARegisterDestroy(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFListDestroy(&DAList);CHKERRQ(ierr);
  DARegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}
