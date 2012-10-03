
#include <petsc-private/vecimpl.h>    /*I "petscvec.h"  I*/

PetscFList VecList                       = PETSC_NULL;
PetscBool  VecRegisterAllCalled          = PETSC_FALSE;

#undef __FUNCT__
#define __FUNCT__ "VecSetType"
/*@C
  VecSetType - Builds a vector, for a particular vector implementation.

  Collective on Vec

  Input Parameters:
+ vec    - The vector object
- method - The name of the vector type

  Options Database Key:
. -vec_type <type> - Sets the vector type; use -help for a list
                     of available types

  Notes:
  See "petsc/include/petscvec.h" for available vector types (for instance, VECSEQ, VECMPI, or VECSHARED).

  Use VecDuplicate() or VecDuplicateVecs() to form additional vectors of the same type as an existing vector.

  Level: intermediate

.keywords: vector, set, type
.seealso: VecGetType(), VecCreate()
@*/
PetscErrorCode  VecSetType(Vec vec, VecType method)
{
  PetscErrorCode (*r)(Vec);
  PetscBool      match;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec, VEC_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject) vec, method, &match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  ierr = PetscFListFind(VecList, ((PetscObject)vec)->comm, method,PETSC_TRUE,(void (**)(void)) &r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown vector type: %s", method);
  if (vec->ops->destroy) {
    ierr = (*vec->ops->destroy)(vec);CHKERRQ(ierr);
    vec->ops->destroy = PETSC_NULL;
  }
  if (vec->map->n < 0 && vec->map->N < 0) {
    vec->ops->create = r;
    vec->ops->load   = VecLoad_Default;
  } else {
    ierr = (*r)(vec);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecGetType"
/*@C
  VecGetType - Gets the vector type name (as a string) from the Vec.

  Not Collective

  Input Parameter:
. vec  - The vector

  Output Parameter:
. type - The vector type name

  Level: intermediate

.keywords: vector, get, type, name
.seealso: VecSetType(), VecCreate()
@*/
PetscErrorCode  VecGetType(Vec vec, VecType *type)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec, VEC_CLASSID,1);
  PetscValidCharPointer(type,2);
  if (!VecRegisterAllCalled) {
    ierr = VecRegisterAll(PETSC_NULL);CHKERRQ(ierr);
  }
  *type = ((PetscObject)vec)->type_name;
  PetscFunctionReturn(0);
}


/*--------------------------------------------------------------------------------------------------------------------*/

#undef __FUNCT__
#define __FUNCT__ "VecRegister"
/*@C
  VecRegister - See VecRegisterDynamic()

  Level: advanced
@*/
PetscErrorCode  VecRegister(const char sname[], const char path[], const char name[], PetscErrorCode (*function)(Vec))
{
  char fullname[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscStrcpy(fullname, path);CHKERRQ(ierr);
  ierr = PetscStrcat(fullname, ":");CHKERRQ(ierr);
  ierr = PetscStrcat(fullname, name);CHKERRQ(ierr);
  ierr = PetscFListAdd(&VecList, sname, fullname, (void (*)(void)) function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*--------------------------------------------------------------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "VecRegisterDestroy"
/*@C
   VecRegisterDestroy - Frees the list of Vec methods that were registered by VecRegister()/VecRegisterDynamic().

   Not Collective

   Level: advanced

.keywords: Vec, register, destroy
.seealso: VecRegister(), VecRegisterAll(), VecRegisterDynamic()
@*/
PetscErrorCode  VecRegisterDestroy(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFListDestroy(&VecList);CHKERRQ(ierr);
  VecRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

