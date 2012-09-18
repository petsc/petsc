
#include <../src/sys/random/randomimpl.h>
#if defined (PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#endif

#undef __FUNCT__
#define __FUNCT__ "PetscRandomSeed_Rand"
PetscErrorCode  PetscRandomSeed_Rand(PetscRandom r)
{
  PetscFunctionBegin;
  srand(r->seed);
  PetscFunctionReturn(0);
}

#define RAND_WRAP ((PetscReal)((rand()/(double)((unsigned int)RAND_MAX+1))))
#undef __FUNCT__
#define __FUNCT__ "PetscRandomGetValue_Rand"
PetscErrorCode  PetscRandomGetValue_Rand(PetscRandom r,PetscScalar *val)
{
  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  if (r->iset)
    *val = PetscRealPart(r->width)*RAND_WRAP + PetscRealPart(r->low) +
      (PetscImaginaryPart(r->width)*RAND_WRAP + PetscImaginaryPart(r->low)) * PETSC_i;
  else *val = RAND_WRAP + RAND_WRAP*PETSC_i;
#else
  if (r->iset) *val = r->width * RAND_WRAP + r->low;
  else         *val = RAND_WRAP;
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscRandomGetValueReal_Rand"
PetscErrorCode  PetscRandomGetValueReal_Rand(PetscRandom r,PetscReal *val)
{
  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
    if (r->iset) *val = PetscRealPart(r->width)*RAND_WRAP + PetscRealPart(r->low);
    else         *val = RAND_WRAP;
#else
  if (r->iset) *val = r->width * RAND_WRAP + r->low;
  else         *val = RAND_WRAP;
#endif
  PetscFunctionReturn(0);
}

static struct _PetscRandomOps PetscRandomOps_Values = {
  /* 0 */
  PetscRandomSeed_Rand,
  PetscRandomGetValue_Rand,
  PetscRandomGetValueReal_Rand,
   0,
  /* 5 */
  0
};

/*MC
   PETSCRAND - access to the basic Unix random number generator

   Options Database Keys:
. -random_type <rand,rand48,sprng>

  Level: beginner

.seealso: RandomCreate(), RandomSetType(), PETSCRAND48, PETSCSPRNG
M*/

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PetscRandomCreate_Rand"
PetscErrorCode  PetscRandomCreate_Rand(PetscRandom r)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMemcpy(r->ops,&PetscRandomOps_Values,sizeof(PetscRandomOps_Values));CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)r,PETSCRAND);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
