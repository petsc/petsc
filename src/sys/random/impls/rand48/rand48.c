
#include <../src/sys/random/randomimpl.h>
#if defined (PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#endif

#undef __FUNCT__  
#define __FUNCT__ "PetscRandomSeed_Rand48"
PetscErrorCode  PetscRandomSeed_Rand48(PetscRandom r)
{
  PetscFunctionBegin;
  srand48(r->seed);   
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscRandomGetValue_Rand48"
PetscErrorCode  PetscRandomGetValue_Rand48(PetscRandom r,PetscScalar *val)
{
  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)  
  if (r->iset) {
    *val = PetscRealPart(r->width)*(PetscReal)drand48() + PetscRealPart(r->low) +
      (PetscImaginaryPart(r->width)*(PetscReal)drand48() + PetscImaginaryPart(r->low)) * PETSC_i;
  } else {
    *val = (PetscReal)drand48() + (PetscReal)drand48()*PETSC_i;
  } 
#else
  if (r->iset) *val = r->width * drand48() + r->low;
  else         *val = drand48();
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscRandomGetValueReal_Rand48"
PetscErrorCode  PetscRandomGetValueReal_Rand48(PetscRandom r,PetscReal *val)
{
  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  if (r->iset) *val = PetscRealPart(r->width)*drand48() + PetscRealPart(r->low);
  else         *val = drand48();
#else
  if (r->iset) *val = r->width * drand48() + r->low;
  else         *val = drand48();
#endif
  PetscFunctionReturn(0);
}

static struct _PetscRandomOps PetscRandomOps_Values = {
  /* 0 */
  PetscRandomSeed_Rand48,
  PetscRandomGetValue_Rand48,
  PetscRandomGetValueReal_Rand48,
  0,
  /* 5 */
  0
};

/*MC
   PETSCRAND48 - access to the basic Unix drand48() random number generator

   Options Database Keys:
. -random_type <rand,rand48,sprng> 

  Level: beginner

.seealso: RandomCreate(), RandomSetType(), PETSCRAND, PETSCSPRNG
M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PetscRandomCreate_Rand48" 
PetscErrorCode  PetscRandomCreate_Rand48(PetscRandom r)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMemcpy(r->ops,&PetscRandomOps_Values,sizeof(PetscRandomOps_Values));CHKERRQ(ierr);
  /* r->bops->publish   = PetscRandomPublish; */
  /*  r->petscnative     = PETSC_TRUE;  */

  ierr = PetscObjectChangeTypeName((PetscObject)r,PETSCRAND48);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
