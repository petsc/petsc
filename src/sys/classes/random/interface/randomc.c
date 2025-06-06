/*
    This file contains routines for interfacing to random number generators.
    This provides more than just an interface to some system random number
    generator:

    Numbers can be shuffled for use as random tuples

    Multiple random number generators may be used

    We are still not sure what interface we want here.  There should be
    one to reinitialize and set the seed.
 */

#include <petsc/private/randomimpl.h> /*I "petscsys.h" I*/
#include <petscviewer.h>

/* Logging support */
PetscClassId PETSC_RANDOM_CLASSID;

/*@
  PetscRandomDestroy - Destroys a `PetscRandom` object that was created by `PetscRandomCreate()`.

  Collective

  Input Parameter:
. r - the random number generator object

  Level: intermediate

.seealso: `PetscRandom`, `PetscRandomGetValue()`, `PetscRandomCreate()`, `VecSetRandom()`
@*/
PetscErrorCode PetscRandomDestroy(PetscRandom *r)
{
  PetscFunctionBegin;
  if (!*r) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific(*r, PETSC_RANDOM_CLASSID, 1);
  if (--((PetscObject)*r)->refct > 0) {
    *r = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscTryTypeMethod(*r, destroy);
  PetscCall(PetscHeaderDestroy(r));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscRandomGetSeed - Gets the random seed.

  Not collective

  Input Parameter:
. r - The random number generator context

  Output Parameter:
. seed - The random seed

  Level: intermediate

.seealso: `PetscRandom`, `PetscRandomCreate()`, `PetscRandomSetSeed()`, `PetscRandomSeed()`
@*/
PetscErrorCode PetscRandomGetSeed(PetscRandom r, PetscInt64 *seed)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(r, PETSC_RANDOM_CLASSID, 1);
  if (seed) {
    PetscAssertPointer(seed, 2);
    *seed = (PetscInt64)r->seed;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscRandomSetSeed - Sets the random seed. You MUST call `PetscRandomSeed()` after this call to have the new seed used.

  Not collective

  Input Parameters:
+ r    - The random number generator context
- seed - The random seed

  Level: intermediate

  Example Usage:
.vb
      PetscRandomSetSeed(r,a positive integer);
      PetscRandomSeed(r);
      PetscRandomGetValue() will now start with the new seed.

      PetscRandomSeed(r) without a call to PetscRandomSetSeed() re-initializes
      the seed. The random numbers generated will be the same as before.
.ve

.seealso: `PetscRandom`, `PetscRandomCreate()`, `PetscRandomGetSeed()`, `PetscRandomSeed()`
@*/
PetscErrorCode PetscRandomSetSeed(PetscRandom r, PetscInt64 seed)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(r, PETSC_RANDOM_CLASSID, 1);
  r->seed = (unsigned long)seed;
  PetscCall(PetscInfo(NULL, "Setting seed to %d\n", (int)seed));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ------------------------------------------------------------------- */
/*
  PetscRandomSetTypeFromOptions_Private - Sets the type of random generator from user options. Defaults to type PETSCRAND48 or PETSCRAND.

  Collective

  Input Parameter:
. rnd - The random number generator context

  Level: intermediate

.seealso: `PetscRandomSetFromOptions()`, `PetscRandomSetType()`
*/
static PetscErrorCode PetscRandomSetTypeFromOptions_Private(PetscRandom rnd, PetscOptionItems PetscOptionsObject)
{
  PetscBool   opt;
  const char *defaultType;
  char        typeName[256];

  PetscFunctionBegin;
  if (((PetscObject)rnd)->type_name) {
    defaultType = ((PetscObject)rnd)->type_name;
  } else {
    defaultType = PETSCRANDER48;
  }

  PetscCall(PetscRandomRegisterAll());
  PetscCall(PetscOptionsFList("-random_type", "PetscRandom type", "PetscRandomSetType", PetscRandomList, defaultType, typeName, 256, &opt));
  if (opt) {
    PetscCall(PetscRandomSetType(rnd, typeName));
  } else {
    PetscCall(PetscRandomSetType(rnd, defaultType));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscRandomSetFromOptions - Configures the random number generator from the options database.

  Collective

  Input Parameter:
. rnd - The random number generator context

  Options Database Keys:
+ -random_seed <integer>    - provide a seed to the random number generator
- -random_no_imaginary_part - makes the imaginary part of the random number zero, this is useful when you want the
                              same code to produce the same result when run with real numbers or complex numbers for regression testing purposes

  Level: beginner

  Note:
  Must be called after `PetscRandomCreate()` but before the rnd is used.

.seealso: `PetscRandom`, `PetscRandomCreate()`, `PetscRandomSetType()`
@*/
PetscErrorCode PetscRandomSetFromOptions(PetscRandom rnd)
{
  PetscBool set, noimaginary = PETSC_FALSE;
  PetscInt  seed;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rnd, PETSC_RANDOM_CLASSID, 1);

  PetscObjectOptionsBegin((PetscObject)rnd);

  /* Handle PetscRandom type options */
  PetscCall(PetscRandomSetTypeFromOptions_Private(rnd, PetscOptionsObject));

  /* Handle specific random generator's options */
  PetscTryTypeMethod(rnd, setfromoptions, PetscOptionsObject);
  PetscCall(PetscOptionsInt("-random_seed", "Seed to use to generate random numbers", "PetscRandomSetSeed", 0, &seed, &set));
  if (set) {
    PetscCall(PetscRandomSetSeed(rnd, (unsigned long int)seed));
    PetscCall(PetscRandomSeed(rnd));
  }
  PetscCall(PetscOptionsBool("-random_no_imaginary_part", "The imaginary part of the random number will be zero", "PetscRandomSetInterval", noimaginary, &noimaginary, &set));
#if defined(PETSC_HAVE_COMPLEX)
  if (set) {
    if (noimaginary) {
      PetscScalar low, high;
      PetscCall(PetscRandomGetInterval(rnd, &low, &high));
      low  = low - PetscImaginaryPart(low);
      high = high - PetscImaginaryPart(high);
      PetscCall(PetscRandomSetInterval(rnd, low, high));
    }
  }
#endif
  PetscOptionsEnd();
  PetscCall(PetscRandomViewFromOptions(rnd, NULL, "-random_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscRandomSetOptionsPrefix - Sets the prefix used for searching for all
  `PetscRandom` options in the database.

  Logically Collective

  Input Parameters:
+ r      - the random number generator context
- prefix - the prefix to prepend to all option names

  Level: advanced

  Note:
  A hyphen (-) must NOT be given at the beginning of the prefix name.
  The first character of all runtime options is AUTOMATICALLY the hyphen.

.seealso: `PetscRandom`, `PetscRandomSetFromOptions()`
@*/
PetscErrorCode PetscRandomSetOptionsPrefix(PetscRandom r, const char prefix[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(r, PETSC_RANDOM_CLASSID, 1);
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)r, prefix));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if defined(PETSC_HAVE_SAWS)
  #include <petscviewersaws.h>
#endif

/*@
  PetscRandomViewFromOptions - View a `PetscRandom` object based on the options database

  Collective

  Input Parameters:
+ A    - the random number generator context
. obj  - Optional object
- name - command line option

  Level: intermediate

.seealso: `PetscRandom`, `PetscRandomView`, `PetscObjectViewFromOptions()`, `PetscRandomCreate()`
@*/
PetscErrorCode PetscRandomViewFromOptions(PetscRandom A, PetscObject obj, const char name[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, PETSC_RANDOM_CLASSID, 1);
  PetscCall(PetscObjectViewFromOptions((PetscObject)A, obj, name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscRandomView - Views a random number generator object.

  Collective

  Input Parameters:
+ rnd    - The random number generator context
- viewer - an optional visualization context

  Level: beginner

  Note:
  The available visualization contexts include
+     `PETSC_VIEWER_STDOUT_SELF` - standard output (default)
-     `PETSC_VIEWER_STDOUT_WORLD` - synchronized standard
  output where only the first processor opens
  the file.  All other processors send their
  data to the first processor to print.

.seealso: `PetscRandom`, `PetscRealView()`, `PetscScalarView()`, `PetscIntView()`
@*/
PetscErrorCode PetscRandomView(PetscRandom rnd, PetscViewer viewer)
{
  PetscBool iascii;
#if defined(PETSC_HAVE_SAWS)
  PetscBool issaws;
#endif

  PetscFunctionBegin;
  PetscValidHeaderSpecific(rnd, PETSC_RANDOM_CLASSID, 1);
  PetscValidType(rnd, 1);
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)rnd), &viewer));
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCheckSameComm(rnd, 1, viewer, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
#if defined(PETSC_HAVE_SAWS)
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERSAWS, &issaws));
#endif
  if (iascii) {
    PetscMPIInt rank;
    PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)rnd, viewer));
    PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)rnd), &rank));
    PetscCall(PetscViewerASCIIPushSynchronized(viewer));
    PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "[%d] Random type %s, seed %lu\n", rank, ((PetscObject)rnd)->type_name, rnd->seed));
    PetscCall(PetscViewerFlush(viewer));
    PetscCall(PetscViewerASCIIPopSynchronized(viewer));
#if defined(PETSC_HAVE_SAWS)
  } else if (issaws) {
    PetscMPIInt rank;
    const char *name;

    PetscCall(PetscObjectGetName((PetscObject)rnd, &name));
    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
    if (!((PetscObject)rnd)->amsmem && rank == 0) {
      char dir[1024];

      PetscCall(PetscObjectViewSAWs((PetscObject)rnd, viewer));
      PetscCall(PetscSNPrintf(dir, 1024, "/PETSc/Objects/%s/Low", name));
      PetscCallSAWs(SAWs_Register, (dir, &rnd->low, 1, SAWs_READ, SAWs_DOUBLE));
    }
#endif
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscRandomCreate - Creates an object for generating random numbers,
  and initializes the random-number generator.

  Collective

  Input Parameter:
. comm - MPI communicator

  Output Parameter:
. r - the random number generator object

  Level: intermediate

  Notes:
  The random type has to be set by `PetscRandomSetType()`.

  This is only a primitive "parallel" random number generator, it should NOT
  be used for sophisticated parallel Monte Carlo methods since it will very likely
  not have the correct statistics across processors. You can provide your own
  parallel generator using `PetscRandomRegister()`;

  If you create a `PetscRandom()` using `PETSC_COMM_SELF` on several processors then
  the SAME random numbers will be generated on all those processors. Use `PETSC_COMM_WORLD`
  or the appropriate parallel communicator to eliminate this issue.

  Use `VecSetRandom()` to set the elements of a vector to random numbers.

  Example of Usage:
.vb
      PetscRandomCreate(PETSC_COMM_SELF,&r);
      PetscRandomSetType(r,PETSCRAND48);
      PetscRandomGetValue(r,&value1);
      PetscRandomGetValueReal(r,&value2);
      PetscRandomDestroy(&r);
.ve

.seealso: `PetscRandomSetType()`, `PetscRandomGetValue()`, `PetscRandomGetValueReal()`, `PetscRandomSetInterval()`,
          `PetscRandomDestroy()`, `VecSetRandom()`, `PetscRandomType`, `PetscRandom`
@*/
PetscErrorCode PetscRandomCreate(MPI_Comm comm, PetscRandom *r)
{
  PetscRandom rr;
  PetscMPIInt rank;

  PetscFunctionBegin;
  PetscAssertPointer(r, 2);
  PetscCall(PetscRandomInitializePackage());

  PetscCall(PetscHeaderCreate(rr, PETSC_RANDOM_CLASSID, "PetscRandom", "Random number generator", "Sys", comm, PetscRandomDestroy, PetscRandomView));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  rr->data  = NULL;
  rr->low   = 0.0;
  rr->width = 1.0;
  rr->iset  = PETSC_FALSE;
  rr->seed  = 0x12345678 + 76543 * rank;
  PetscCall(PetscRandomSetType(rr, PETSCRANDER48));
  *r = rr;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscRandomSeed - Seed the random number generator.

  Not collective

  Input Parameter:
. r - The random number generator context

  Level: intermediate

  Example Usage:
.vb
      PetscRandomSetSeed(r,a positive integer);
      PetscRandomSeed(r);
      PetscRandomGetValue() will now start with the new seed.

      PetscRandomSeed(r) without a call to PetscRandomSetSeed() re-initializes
      the seed. The random numbers generated will be the same as before.
.ve

.seealso: `PetscRandomCreate()`, `PetscRandomGetSeed()`, `PetscRandomSetSeed()`
@*/
PetscErrorCode PetscRandomSeed(PetscRandom r)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(r, PETSC_RANDOM_CLASSID, 1);
  PetscValidType(r, 1);

  PetscUseTypeMethod(r, seed);
  PetscCall(PetscObjectStateIncrease((PetscObject)r));
  PetscFunctionReturn(PETSC_SUCCESS);
}
