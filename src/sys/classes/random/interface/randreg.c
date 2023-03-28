
#include <petsc/private/randomimpl.h> /*I "petscsys.h" I*/

PetscFunctionList PetscRandomList              = NULL;
PetscBool         PetscRandomRegisterAllCalled = PETSC_FALSE;

/*@C
  PetscRandomSetType - Builds a context for generating a particular type of random numbers.

  Collective

  Input Parameters:
+ rnd   - The random number generator context
- type - The name of the random type

  Options Database Key:
. -random_type <type> - Sets the random type; use -help for a list
                     of available types

  Level: intermediate

  Note:
  See `PetscRandomType` for available random types (for instance, `PETSCRAND48`, `PETSCRAND`).

.seealso: `PetscRandom`, `PetscRandomType`, `PetscRandomGetType()`, `PetscRandomCreate()`
@*/

PetscErrorCode PetscRandomSetType(PetscRandom rnd, PetscRandomType type)
{
  PetscErrorCode (*r)(PetscRandom);
  PetscBool match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rnd, PETSC_RANDOM_CLASSID, 1);
  PetscCall(PetscObjectTypeCompare((PetscObject)rnd, type, &match));
  if (match) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscFunctionListFind(PetscRandomList, type, &r));
  PetscCheck(r, PETSC_COMM_SELF, PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown random type: %s", type);

  PetscTryTypeMethod(rnd, destroy);
  rnd->ops->destroy = NULL;

  PetscCall((*r)(rnd));
  PetscCall(PetscRandomSeed(rnd));

  PetscCall(PetscObjectChangeTypeName((PetscObject)rnd, type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscRandomGetType - Gets the type name (as a string) from the `PetscRandom`.

  Not Collective

  Input Parameter:
. rnd  - The random number generator context

  Output Parameter:
. type - The type name

  Level: intermediate

.seealso: `PetscRandom`, `PetscRandomType`, `PetscRandomSetType()`, `PetscRandomCreate()`
@*/
PetscErrorCode PetscRandomGetType(PetscRandom rnd, PetscRandomType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(rnd, PETSC_RANDOM_CLASSID, 1);
  PetscValidPointer(type, 2);
  *type = ((PetscObject)rnd)->type_name;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscRandomRegister -  Adds a new `PetscRandom` implementation

  Not Collective

  Input Parameters:
+ sname    - The name of a new user-defined creation routine
- function - The creation routine

  Level: advanced

  Notes:
  `PetscRandomRegister()` may be called multiple times to add several user-defined randome number generators

  For an example of the code needed to interface your own random number generator see src/sys/random/impls/rand/rand.c

  Sample usage:
.vb
    PetscRandomRegister("my_rand",  MyPetscRandomtorCreate);
.ve

  Then, your random type can be chosen with the procedural interface via
.vb
    PetscRandomCreate(MPI_Comm, PetscRandom *);
    PetscRandomSetType(PetscRandom,"my_random_name");
.ve
   or at runtime via the option
.vb
    -random_type my_random_name
.ve

.seealso: `PetscRandom`, `PetscRandomRegisterAll()`, `PetscRandomRegisterDestroy()`, `PetscRandomRegister()`
@*/
PetscErrorCode PetscRandomRegister(const char sname[], PetscErrorCode (*function)(PetscRandom))
{
  PetscFunctionBegin;
  PetscCall(PetscRandomInitializePackage());
  PetscCall(PetscFunctionListAdd(&PetscRandomList, sname, function));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if defined(PETSC_HAVE_RAND)
PETSC_EXTERN PetscErrorCode PetscRandomCreate_Rand(PetscRandom);
#endif
#if defined(PETSC_HAVE_DRAND48)
PETSC_EXTERN PetscErrorCode PetscRandomCreate_Rand48(PetscRandom);
#endif
#if defined(PETSC_HAVE_SPRNG)
PETSC_EXTERN PetscErrorCode PetscRandomCreate_Sprng(PetscRandom);
#endif
PETSC_EXTERN PetscErrorCode PetscRandomCreate_Rander48(PetscRandom);
#if defined(PETSC_HAVE_RANDOM123)
PETSC_EXTERN PetscErrorCode PetscRandomCreate_Random123(PetscRandom);
#endif
#if defined(PETSC_HAVE_CUDA)
PETSC_EXTERN PetscErrorCode PetscRandomCreate_CURAND(PetscRandom);
#endif

/*@C
  PetscRandomRegisterAll - Registers all of the implementations in the `PetscRandom` package.

   Not Collective

  Level: advanced

.seealso: `PetscRandom`, `PetscRandomType`, `PetscRandomRegister()`, `PetscRandomRegisterDestroy()`
@*/
PetscErrorCode PetscRandomRegisterAll(void)
{
  PetscFunctionBegin;
  if (PetscRandomRegisterAllCalled) PetscFunctionReturn(PETSC_SUCCESS);
  PetscRandomRegisterAllCalled = PETSC_TRUE;
#if defined(PETSC_HAVE_RAND)
  PetscCall(PetscRandomRegister(PETSCRAND, PetscRandomCreate_Rand));
#endif
#if defined(PETSC_HAVE_DRAND48)
  PetscCall(PetscRandomRegister(PETSCRAND48, PetscRandomCreate_Rand48));
#endif
#if defined(PETSC_HAVE_SPRNG)
  PetscCall(PetscRandomRegister(PETSCSPRNG, PetscRandomCreate_Sprng));
#endif
  PetscCall(PetscRandomRegister(PETSCRANDER48, PetscRandomCreate_Rander48));
#if defined(PETSC_HAVE_RANDOM123)
  PetscCall(PetscRandomRegister(PETSCRANDOM123, PetscRandomCreate_Random123));
#endif
#if defined(PETSC_HAVE_CUDA)
  PetscCall(PetscRandomRegister(PETSCCURAND, PetscRandomCreate_CURAND));
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}
