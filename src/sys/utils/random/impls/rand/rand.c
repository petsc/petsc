#define PETSC_DLL

#include "src/sys/utils/random/randomimpl.h"
#if defined (PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#endif


#undef __FUNCT__  
#define __FUNCT__ "PetscRandomSeed_Rand"
PetscErrorCode PETSC_DLLEXPORT PetscRandomSeed_Rand(PetscRandom r)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(r,PETSC_RANDOM_COOKIE,1);
  srand(r->seed);  
  PetscFunctionReturn(0);
}

#define RAND_WRAP() (rand()/(double)((unsigned int)RAND_MAX+1))
#undef __FUNCT__  
#define __FUNCT__ "PetscRandomGetValue_Rand"
PetscErrorCode PETSC_DLLEXPORT PetscRandomGetValue_Rand(PetscRandom r,PetscScalar *val)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(r,PETSC_RANDOM_COOKIE,1);
  PetscValidScalarPointer(val,2);
#if defined(PETSC_USE_COMPLEX)
  if (r->type == RANDOM_DEFAULT) {
    if (r->iset)
         *val = PetscRealPart(r->width)*RAND_WRAP() + PetscRealPart(r->low) +
                (PetscImaginaryPart(r->width)*RAND_WRAP() + PetscImaginaryPart(r->low)) * PETSC_i;
    else *val = RAND_WRAP() + RAND_WRAP()*PETSC_i;
  } else if (r->type == RANDOM_DEFAULT_REAL) {
    if (r->iset) *val = PetscRealPart(r->width)*RAND_WRAP() + PetscRealPart(r->low);
    else         *val = RAND_WRAP();
  } else if (r->type == RANDOM_DEFAULT_IMAGINARY) {
    if (r->iset) *val = (PetscImaginaryPart(r->width)*RAND_WRAP()+PetscImaginaryPart(r->low))*PETSC_i;
    else         *val = RAND_WRAP()*PETSC_i;
  } else SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Invalid random number type");
#else
  if (r->iset) *val = r->width * RAND_WRAP() + r->low;
  else         *val = RAND_WRAP();
#endif
  PetscFunctionReturn(0);
}

static struct _PetscRandomOps PetscRandomOps_Values = {
  /* 0 */
  PetscRandomSeed_Rand,
  0,
  0,
  PetscRandomGetValue_Rand,
  0,
  /* 5 */
  0,
  0
};

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PetscRandomCreate_Rand" 
PetscErrorCode PETSC_DLLEXPORT PetscRandomCreate_Rand(PetscRandom r)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMemcpy(r->ops,&PetscRandomOps_Values,sizeof(PetscRandomOps_Values));CHKERRQ(ierr);
  srand(r->seed);   
  ierr = PetscObjectChangeTypeName((PetscObject)r,PETSC_RAND48);CHKERRQ(ierr);
  ierr = PetscPublishAll(r);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
