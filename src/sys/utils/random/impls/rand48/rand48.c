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

#undef __FUNCT__  
#define __FUNCT__ "PetscRandomSeed_Rand48"
PetscErrorCode PETSC_DLLEXPORT PetscRandomSeed_Rand48(PetscRandom r)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(r,PETSC_RANDOM_COOKIE,1);
  srand48(r->seed);   
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscRandomGetValue_Rand48"
PetscErrorCode PETSC_DLLEXPORT PetscRandomGetValue_Rand48(PetscRandom r,PetscScalar *val)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(r,PETSC_RANDOM_COOKIE,1);
  PetscValidIntPointer(val,2);
#if defined(PETSC_USE_COMPLEX)
  if (r->type == RANDOM_DEFAULT) {
    if (r->iset) {
         *val = PetscRealPart(r->width)*drand48() + PetscRealPart(r->low) +
                (PetscImaginaryPart(r->width)*drand48() + PetscImaginaryPart(r->low)) * PETSC_i;
    }
    else *val = drand48() + drand48()*PETSC_i;
  } else if (r->type == RANDOM_DEFAULT_REAL) {
    if (r->iset) *val = PetscRealPart(r->width)*drand48() + PetscRealPart(r->low);
    else                       *val = drand48();
  } else if (r->type == RANDOM_DEFAULT_IMAGINARY) {
    if (r->iset) *val = (PetscImaginaryPart(r->width)*drand48()+PetscImaginaryPart(r->low))*PETSC_i;
    else         *val = drand48()*PETSC_i;
  } else {
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Invalid random number type");
  }
#else
  if (r->iset) *val = r->width * drand48() + r->low;
  else         *val = drand48();
#endif
  PetscFunctionReturn(0);
}


static struct _PetscRandomOps PetscRandomOps_Values = {
  /* 0 */
  PetscRandomSeed_Rand48,
  0,
  0,
  PetscRandomGetValue_Rand48,
  0,
  /* 5 */
  0,
  0
};

/*
   For now we have set up using the DRAND48() generater. We need to deal 
   with other variants of random number generators. We should also add
   a routine to enable restarts [seed48()] 
*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PetscRandomCreate_Rand48" 
PetscErrorCode PETSC_DLLEXPORT PetscRandomCreate_Rand48(PetscRandom r)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMemcpy(r->ops,&PetscRandomOps_Values,sizeof(PetscRandomOps_Values));CHKERRQ(ierr);
  /* r->bops->publish   = PetscRandomPublish; */
  /*  r->petscnative     = PETSC_TRUE;  */

  srand48(r->seed);  
  ierr = PetscObjectChangeTypeName((PetscObject)r,PETSC_RAND48);CHKERRQ(ierr);
  ierr = PetscPublishAll(r);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
