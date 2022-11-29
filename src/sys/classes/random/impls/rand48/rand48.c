#define PETSC_DESIRE_FEATURE_TEST_MACROS /* for drand48() */
#include <petsc/private/randomimpl.h>

PetscErrorCode PetscRandomSeed_Rand48(PetscRandom r)
{
  PetscFunctionBegin;
  srand48(r->seed);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscRandomGetValue_Rand48(PetscRandom r, PetscScalar *val)
{
  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  if (r->iset) {
    *val = PetscRealPart(r->width) * (PetscReal)drand48() + PetscRealPart(r->low) + (PetscImaginaryPart(r->width) * (PetscReal)drand48() + PetscImaginaryPart(r->low)) * PETSC_i;
  } else {
    *val = (PetscReal)drand48() + (PetscReal)drand48() * PETSC_i;
  }
#else
  if (r->iset) *val = r->width * drand48() + r->low;
  else *val = drand48();
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode PetscRandomGetValueReal_Rand48(PetscRandom r, PetscReal *val)
{
  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  if (r->iset) *val = PetscRealPart(r->width) * drand48() + PetscRealPart(r->low);
  else *val = drand48();
#else
  if (r->iset) *val = r->width * drand48() + r->low;
  else *val = drand48();
#endif
  PetscFunctionReturn(0);
}

static struct _PetscRandomOps PetscRandomOps_Values = {
  PetscDesignatedInitializer(seed, PetscRandomSeed_Rand48),
  PetscDesignatedInitializer(getvalue, PetscRandomGetValue_Rand48),
  PetscDesignatedInitializer(getvaluereal, PetscRandomGetValueReal_Rand48),
};

/*MC
   PETSCRAND48 - access to the basic Unix drand48() random number generator

   Options Database Keys:
. -random_type <rand,rand48,sprng> - select the random number generator at runtime

  Level: beginner

  Note:
  Not recommended because it may produce different results on different systems.

.seealso: `PetscRandomCreate()`, `PetscRandomSetType()`, `PETSCRAND`, `PETSCSPRNG`, `PetscRandomSetFromOptions()`
M*/

PETSC_EXTERN PetscErrorCode PetscRandomCreate_Rand48(PetscRandom r)
{
  PetscFunctionBegin;
  PetscCall(PetscMemcpy(r->ops, &PetscRandomOps_Values, sizeof(PetscRandomOps_Values)));
  PetscCall(PetscObjectChangeTypeName((PetscObject)r, PETSCRAND48));
  PetscFunctionReturn(0);
}
