#include <petsc/private/randomimpl.h>

static PetscErrorCode PetscRandomSeed_Rand(PetscRandom r)
{
  PetscFunctionBegin;
  srand((unsigned int)r->seed);
  PetscFunctionReturn(PETSC_SUCCESS);
}

#define RAND_WRAP ((PetscReal)(rand() / (double)((unsigned int)RAND_MAX + 1)))
static PetscErrorCode PetscRandomGetValue_Rand(PetscRandom r, PetscScalar *val)
{
  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  if (r->iset) *val = PetscRealPart(r->width) * RAND_WRAP + PetscRealPart(r->low) + (PetscImaginaryPart(r->width) * RAND_WRAP + PetscImaginaryPart(r->low)) * PETSC_i;
  else *val = RAND_WRAP + RAND_WRAP * PETSC_i;
#else
  if (r->iset) *val = r->width * RAND_WRAP + r->low;
  else *val = RAND_WRAP;
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscRandomGetValueReal_Rand(PetscRandom r, PetscReal *val)
{
  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  if (r->iset) *val = PetscRealPart(r->width) * RAND_WRAP + PetscRealPart(r->low);
  else *val = RAND_WRAP;
#else
  if (r->iset) *val = r->width * RAND_WRAP + r->low;
  else *val = RAND_WRAP;
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

static struct _PetscRandomOps PetscRandomOps_Values = {
  PetscDesignatedInitializer(seed, PetscRandomSeed_Rand),
  PetscDesignatedInitializer(getvalue, PetscRandomGetValue_Rand),
  PetscDesignatedInitializer(getvaluereal, PetscRandomGetValueReal_Rand),
  PetscDesignatedInitializer(getvalues, NULL),
  PetscDesignatedInitializer(getvaluesreal, NULL),
  PetscDesignatedInitializer(destroy, NULL),
  PetscDesignatedInitializer(setfromoptions, NULL),
};

/*MC
   PETSCRAND - access to the basic Unix random number generator

   Options Database Key:
. -random_type <rand,rand48,sprng> - set the random number generator from the options database

  Level: beginner

  Note:
  Not recommended since it can produce different numbers on different systems

.seealso: `PetscRandomCreate()`, `PetscRandomSetType()`, `PETSCRAND48`, `PETSCSPRNG`, `PetscRandomSetFromOptions()`, `PetscRandomType`
M*/

PETSC_EXTERN PetscErrorCode PetscRandomCreate_Rand(PetscRandom r)
{
  PetscFunctionBegin;
  r->ops[0] = PetscRandomOps_Values;
  PetscCall(PetscObjectChangeTypeName((PetscObject)r, PETSCRAND));
  PetscFunctionReturn(PETSC_SUCCESS);
}
