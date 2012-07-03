
#include <../src/sys/random/randomimpl.h>         /*I "petscsys.h" I*/

PetscFList PetscRandomList              = PETSC_NULL;
PetscBool  PetscRandomRegisterAllCalled = PETSC_FALSE;

#undef __FUNCT__  
#define __FUNCT__ "PetscRandomSetType"
/*@C
  PetscRandomSetType - Builds a context for generating particular type of random numbers.

  Collective on PetscRandom

  Input Parameters:
+ rnd   - The random number generator context
- type - The name of the random type

  Options Database Key:
. -random_type <type> - Sets the random type; use -help for a list 
                     of available types

  Notes:
  See "petsc/include/petscsys.h" for available random types (for instance, PETSCRAND48, PETSCRAND).

  Level: intermediate

.keywords: random, set, type
.seealso: PetscRandomGetType(), PetscRandomCreate()
@*/

PetscErrorCode  PetscRandomSetType(PetscRandom rnd, const PetscRandomType type)
{
  PetscErrorCode (*r)(PetscRandom);
  PetscBool      match;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rnd, PETSC_RANDOM_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)rnd, type, &match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  ierr = PetscFListFind(PetscRandomList,((PetscObject)rnd)->comm,  type,PETSC_TRUE,(void (**)(void)) &r);CHKERRQ(ierr); 
  if (!r) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown random type: %s", type);

  if (rnd->ops->destroy) {
    ierr = (*rnd->ops->destroy)(rnd);CHKERRQ(ierr);
    rnd->ops->destroy = PETSC_NULL;
  }
  ierr = (*r)(rnd);CHKERRQ(ierr); 
  ierr = PetscRandomSeed(rnd);CHKERRQ(ierr);

  ierr = PetscObjectChangeTypeName((PetscObject)rnd, type);CHKERRQ(ierr);
#if defined(PETSC_HAVE_AMS)
  if (PetscAMSPublishAll) {
    ierr = PetscObjectAMSPublish((PetscObject)rnd);CHKERRQ(ierr);
  }
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscRandomGetType"
/*@C
  PetscRandomGetType - Gets the type name (as a string) from the PetscRandom.

  Not Collective

  Input Parameter:
. rnd  - The random number generator context

  Output Parameter:
. type - The type name

  Level: intermediate

.keywords: random, get, type, name
.seealso: PetscRandomSetType(), PetscRandomCreate()
@*/
PetscErrorCode  PetscRandomGetType(PetscRandom rnd, const PetscRandomType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(rnd, PETSC_RANDOM_CLASSID,1);
  PetscValidPointer(type,2);
  *type = ((PetscObject)rnd)->type_name;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscRandomRegister"
/*@C
  PetscRandomRegister - See PetscRandomRegisterDynamic()

  Level: advanced
@*/
PetscErrorCode  PetscRandomRegister(const char sname[], const char path[], const char name[], PetscErrorCode (*function)(PetscRandom))
{
  char           fullname[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFListConcat(path,name,fullname);CHKERRQ(ierr);
  ierr = PetscFListAdd(&PetscRandomList,sname,fullname,(void (*)(void))function);CHKERRQ(ierr);
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
PetscErrorCode  PetscRandomRegisterDestroy(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFListDestroy(&PetscRandomList);CHKERRQ(ierr);
  PetscRandomRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#if defined(PETSC_HAVE_RAND)
extern PetscErrorCode  PetscRandomCreate_Rand(PetscRandom);
#endif
#if defined(PETSC_HAVE_DRAND48)
extern PetscErrorCode  PetscRandomCreate_Rand48(PetscRandom);
#endif
#if defined(PETSC_HAVE_SPRNG)
extern PetscErrorCode  PetscRandomCreate_Sprng(PetscRandom);
#endif
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
PetscErrorCode  PetscRandomRegisterAll(const char path[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscRandomRegisterAllCalled = PETSC_TRUE;
#if defined(PETSC_HAVE_RAND)
  ierr = PetscRandomRegisterDynamic(PETSCRAND,  path,"PetscRandomCreate_Rand",  PetscRandomCreate_Rand);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_DRAND48)
  ierr = PetscRandomRegisterDynamic(PETSCRAND48,path,"PetscRandomCreate_Rand48",PetscRandomCreate_Rand48);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_SPRNG)
  ierr = PetscRandomRegisterDynamic(PETSCSPRNG,path,"PetscRandomCreate_Sprng",PetscRandomCreate_Sprng);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

