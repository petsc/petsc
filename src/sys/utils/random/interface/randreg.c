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
+ rand   - The random number generator context
- type - The name of the random type

  Options Database Key:
. -random_type <type> - Sets the random type; use -help for a list 
                     of available types

  Notes:
  See "petsc/include/petscsys.h" for available random types (for instance, PETSC_RAND48, PETSC_RAND).

  Level: intermediate

.keywords: random, set, type
.seealso: PetscRandomGetType(), PetscRandomCreate()
@*/

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
  ierr = PetscFListFind(rand->comm, PetscRandomList, type,(void (**)(void)) &r);CHKERRQ(ierr); 
  if (!r) SETERRQ1(PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown random type: %s", type);

  if (rand->ops->destroy) {
    ierr = (*rand->ops->destroy)(rand);CHKERRQ(ierr);
  }
  ierr = (*r)(rand);CHKERRQ(ierr); 

  ierr = PetscObjectChangeTypeName((PetscObject)rand, type);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscRandomGetType"
/*@C
  PetscRandomGetType - Gets the type name (as a string) from the PetscRandom.

  Not Collective

  Input Parameter:
. rand  - The random number generator context

  Output Parameter:
. type - The type name

  Level: intermediate

.keywords: random, get, type, name
.seealso: PetscRandomSetType(), PetscRandomCreate()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT PetscRandomGetType(PetscRandom rand, PetscRandomType *type)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rand, PETSC_RANDOM_COOKIE,1);
  PetscValidCharPointer(type,2);
  if (!PetscRandomRegisterAllCalled) {
    ierr = PetscRandomRegisterAll(PETSC_NULL);CHKERRQ(ierr);
  }
  *type = rand->type_name;
  PetscFunctionReturn(0);
}
//-------------------------------------------------------------------
#undef __FUNCT__  
#define __FUNCT__ "PetscRandomSetTypeFromOptions_Private"
/*
  PetscRandomSetTypeFromOptions_Private - Sets the type of random generator from user options. Defaults to type PETSC_RAND48 or PETSC_RAND.

  Collective on PetscRandom

  Input Parameter:
. rand - The random number generator context

  Level: intermediate

.keywords: PetscRandom, set, options, database, type
.seealso: PetscRandomSetFromOptions(), PetscRandomSetType()
*/
static PetscErrorCode PetscRandomSetTypeFromOptions_Private(PetscRandom rand)
{
  PetscTruth     opt;
  const char     *defaultType;
  char           typeName[256];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (rand->type_name) {
    defaultType = rand->type_name;
  } else {
#if defined(PETSC_HAVE_DRAND48)    
    defaultType = PETSC_RAND48;
#elif defined(PETSC_HAVE_RAND)
    defaultType = PETSC_RAND;
#endif
  }

  if (!PetscRandomRegisterAllCalled) {
    ierr = PetscRandomRegisterAll(PETSC_NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsList("-random_type","PetscRandom type","PetscRandomSetType",PetscRandomList,defaultType,typeName,256,&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscRandomSetType(rand, typeName);CHKERRQ(ierr);
  } else {
    ierr = PetscRandomSetType(rand, defaultType);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscRandomSetFromOptions"
/*@
  PetscRandomSetFromOptions - Configures the random number generator from the options database.

  Collective on PetscRandom

  Input Parameter:
. rand - The random number generator context

  Notes:  To see all options, run your program with the -help option, or consult the users manual.
          Must be called after PetscRandomCreate() but before the rand is used.

  Level: beginner

.keywords: PetscRandom, set, options, database
.seealso: PetscRandomCreate(), PetscRandomSetType()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscRandomSetFromOptions(PetscRandom rand)
{
  PetscTruth     opt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rand,PETSC_RANDOM_COOKIE,1);

  ierr = PetscOptionsBegin(rand->comm, rand->prefix, "PetscRandom options", "PetscRandom");CHKERRQ(ierr);

  /* Handle generic options */
  ierr = PetscOptionsHasName(PETSC_NULL, "-help", &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscRandomPrintHelp(rand);CHKERRQ(ierr);
  }

  /* Handle PetscRandom type options */
  ierr = PetscRandomSetTypeFromOptions_Private(rand);CHKERRQ(ierr);

  /* Handle specific random generator's options */
  if (rand->ops->setfromoptions) {
    ierr = (*rand->ops->setfromoptions)(rand);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscRandomPrintHelp"
/*@
  PetscRandomPrintHelp - Prints some options for the PetscRandom.

  Input Parameter:
. rand - The random number generator context

  Options Database Keys:
$  -help, -h

  Level: intermediate

.keywords: PetscRandom, help
.seealso: PetscRandomSetFromOptions()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscRandomPrintHelp(PetscRandom rand)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(rand, PETSC_RANDOM_COOKIE,1);
  PetscFunctionReturn(0);
}
/*--------------------------------------------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "PetscRandomRegister"
/*@C
  PetscRandomRegister - See PetscRandomRegisterDynamic()

  Level: advanced
@*/
PetscErrorCode PETSCVEC_DLLEXPORT PetscRandomRegister(const char sname[], const char path[], const char name[], PetscErrorCode (*function)(PetscRandom))
{
  char fullname[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscStrcpy(fullname, path);CHKERRQ(ierr);
  ierr = PetscStrcat(fullname, ":");CHKERRQ(ierr);
  ierr = PetscStrcat(fullname, name);CHKERRQ(ierr);
  ierr = PetscFListAdd(&PetscRandomList, sname, fullname, (void (*)(void)) function);CHKERRQ(ierr);
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
  if (PetscRandomList) {
    ierr = PetscFListDestroy(&PetscRandomList);CHKERRQ(ierr);
    PetscRandomList = PETSC_NULL;
  }
  PetscRandomRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#if defined(PETSC_HAVE_DRAND)
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT PetscRandomCreate_Rand(PetscRandom);
#endif
#if defined(PETSC_HAVE_DRAND48)
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT PetscRandomCreate_Rand48(PetscRandom);
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
PetscErrorCode PETSCVEC_DLLEXPORT PetscRandomRegisterAll(const char path[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscRandomRegisterAllCalled = PETSC_TRUE;
#if defined(PETSC_HAVE_DRAND)
  ierr = PetscRandomRegisterDynamic(PETSC_RAND,  path,"PetscRandomCreate_Rand",  PetscRandomCreate_Rand);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_DRAND48)
  ierr = PetscRandomRegisterDynamic(PETSC_RAND48,path,"PetscRandomCreate_Rand48",PetscRandomCreate_Rand48);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

