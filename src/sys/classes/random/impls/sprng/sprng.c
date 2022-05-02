
#include <petsc/private/randomimpl.h>

#define USE_MPI
#define SIMPLE_SPRNG
EXTERN_C_BEGIN
#include <sprng.h>
EXTERN_C_END

PetscErrorCode  PetscRandomSeed_Sprng(PetscRandom r)
{
  PetscFunctionBegin;
  init_sprng(r->seed,SPRNG_DEFAULT);
  PetscFunctionReturn(0);
}

PetscErrorCode  PetscRandomGetValue_Sprng(PetscRandom r,PetscScalar *val)
{
  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  if (r->iset) {
    *val = PetscRealPart(r->width)*sprng() + PetscRealPart(r->low) + (PetscImaginaryPart(r->width)*sprng() + PetscImaginaryPart(r->low)) * PETSC_i;
  } else {
    *val = sprng() + sprng()*PETSC_i;
  }
#else
  if (r->iset) *val = r->width * sprng() + r->low;
  else         *val = sprng();
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode  PetscRandomGetValueReal_Sprng(PetscRandom r,PetscReal *val)
{
  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  if (r->iset) *val = PetscRealPart(r->width)*sprng() + PetscRealPart(r->low);
  else         *val = sprng();
#else
  if (r->iset) *val = r->width * sprng() + r->low;
  else         *val = sprng();
#endif
  PetscFunctionReturn(0);
}

static struct _PetscRandomOps PetscRandomOps_Values = {
  PetscDesignatedInitializer(seed,PetscRandomSeed_Sprng),
  PetscDesignatedInitializer(getvalue,PetscRandomGetValue_Sprng),
  PetscDesignatedInitializer(getvaluereal,PetscRandomGetValueReal_Sprng),
};

/*MC
   PETSCSPRNG- access to the publically available random number generator sprng

   Options Database Keys:
. -random_type <rand,rand48,sprng>

  Level: beginner

   PETSc must have been ./configure with the option --download-sprng to use
   this random number generator.

   This is NOT currently using a parallel random number generator. Sprng does have
   an MPI version we should investigate.

.seealso: `RandomCreate()`, `RandomSetType()`, `PETSCRAND`, `PETSCRAND48`
M*/

PETSC_EXTERN PetscErrorCode PetscRandomCreate_Sprng(PetscRandom r)
{
  PetscFunctionBegin;
  PetscCall(PetscMemcpy(r->ops,&PetscRandomOps_Values,sizeof(PetscRandomOps_Values)));
  PetscCall(PetscObjectChangeTypeName((PetscObject)r,PETSCSPRNG));
  PetscFunctionReturn(0);
}
