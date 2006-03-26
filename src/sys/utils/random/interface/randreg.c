#define PETSC_DLL

#include "src/sys/utils/random/randomimpl.h"
#if defined (PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#else
/* maybe the protypes are missing */
#if defined(PETSC_HAVE_DRAND48)
EXTERN_C_BEGIN
extern double drand48();
extern void   srand48(long);
EXTERN_C_END
#else
extern double drand48();
#endif
#endif

PetscFList PetscRandomList              = PETSC_NULL;
PetscTruth PetscRandomRegisterAllCalled = PETSC_FALSE;

#undef __FUNCT__  
#define __FUNCT__ "PetscRandomSetType"
/*@C
  PetscRandomSetType - Builds a context for generating particular type of random numbers.

  Collective on PetscRandom

  Input Parameters:
+ rand   - The random object
- type - The name of the random type

  Options Database Key:
. -random_type <type> - Sets the random type; use -help for a list 
                     of available types

  Notes:
  See "petsc/include/petscsys.h" for available random types (for instance, RANDOM_RAND48, RANDOM_RAND).

  Level: intermediate

.keywords: random, set, type
.seealso: PetscRandomGetType(), PetscRandomCreate()
@*/

#define PetscRandomType const char*  // change original!

PetscErrorCode PETSCVEC_DLLEXPORT PetscRandomSetType(PetscRandom rand, PetscRandomType type)
{
  PetscErrorCode (*r)(PetscRandom);
  PetscTruth     match;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rand, PETSC_RANDOM_COOKIE,1);
  ierr = PetscTypeCompare((PetscObject)rand, type, &match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  /* Get the function pointers for the random requested */
  if (!PetscRandomRegisterAllCalled) {
    ierr = PetscRandomRegisterAll(PETSC_NULL);CHKERRQ(ierr);
  }
  /* ierr = PetscFListFind(rand->comm, PetscRandomList, type,(void (**)(void)) &r);CHKERRQ(ierr); */
  ierr = PetscFListFind(PETSC_COMM_SELF, PetscRandomList, type,(void (**)(void)) &r);CHKERRQ(ierr)
  if (!r) SETERRQ1(PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown random type: %s", type);
#ifdef TMP
  if (vec->ops->destroy) {
    ierr = (*vec->ops->destroy)(vec);CHKERRQ(ierr);
  }

  ierr = (*r)(rand);CHKERRQ(ierr); /* PetscRandomCreate_xxx() ? */
#endif
  ierr = PetscObjectChangeTypeName((PetscObject)rand, type);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#ifdef TMP
#undef __FUNCT__  
#define __FUNCT__ "PetscRandomGetType"
/*@C
  PetscRandomGetType - Gets the vector type name (as a string) from the PetscRandom.

  Not Collective

  Input Parameter:
. vec  - The vector

  Output Parameter:
. type - The vector type name

  Level: intermediate

.keywords: vector, get, type, name
.seealso: PetscRandomSetType(), PetscRandomCreate()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT PetscRandomGetType(PetscRandom vec, PetscRandomType *type)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec, VEC_COOKIE,1);
  PetscValidCharPointer(type,2);
  if (!PetscRandomRegisterAllCalled) {
    ierr = PetscRandomRegisterAll(PETSC_NULL);CHKERRQ(ierr);
  }
  *type = vec->type_name;
  PetscFunctionReturn(0);
}


/*--------------------------------------------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "PetscRandomRegister"
/*@C
  PetscRandomRegister - See PetscRandomRegisterDynamic()

  Level: advanced
@*/
PetscErrorCode PETSCVEC_DLLEXPORT PetscRandomRegister(const char sname[], const char path[], const char name[], PetscErrorCode (*function)(Random))
{
  char fullname[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscStrcpy(fullname, path);CHKERRQ(ierr);
  ierr = PetscStrcat(fullname, ":");CHKERRQ(ierr);
  ierr = PetscStrcat(fullname, name);CHKERRQ(ierr);
  ierr = PetscFListAdd(&RandomList, sname, fullname, (void (*)(void)) function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*--------------------------------------------------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "PetscRandomRegisterDestroy"
/*@C
   PetscRandomRegisterDestroy - Frees the list of Random types that were registered by PetscRandomRegister()/PetscRandomRegisterDynamic().

   Not Collective

   Level: advanced

.keywords: PetscRandom, register, destroy
.seealso: PetscRandomRegister(), PetscRandomRegisterAll(), PetscRandomRegisterDynamic()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT PetscRandomRegisterDestroy(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (RandomList) {
    ierr = PetscFListDestroy(&PetscRandomList);CHKERRQ(ierr);
    PetscRandomList = PETSC_NULL;
  }
  PetscRandomRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#endif  /* TMP */

EXTERN_C_BEGIN
//EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecCreate_Seq(Vec);
//EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecCreate_MPI(Vec);
//EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecCreate_Shared(Vec);
//EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecCreate_FETI(Vec);
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "PetscRandomRegisterAll"
/*@C
  PetscRandomRegisterAll - Registers all of the components in the PetscRandom package.

  Not Collective

  Input parameter:
. path - The dynamic library path

  Level: advanced

.keywords: PetscRandom, register, all
.seealso:  PetscRandomRegister(), PetscRandomRegisterDestroy(), PetscRandomRegisterDynamic()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT PetscRandomRegisterAll(const char path[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscRandomRegisterAllCalled = PETSC_TRUE;
  /*
  ierr = PetscRandomRegisterDynamic(PETSC_RAND,  path,"PetscRandomCreate_Rand",  PetscRandomCreate_Rand);CHKERRQ(ierr);
  ierr = PetscRandomRegisterDynamic(PETSC_RAND48,path,"PetscRandomCreate_Rand48",PetscRandomCreate_Rand48);CHKERRQ(ierr);
  */
  PetscFunctionReturn(0);
}

