
#include "petsc.h"
#if defined(PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#endif

EXTERN_C_BEGIN

#include "ad_deriv.h"

void PetscADResetIndep(void){
  ad_AD_ResetIndep();
}

void PetscADSetValArray(DERIV_TYPE *var,int size,double *values){
  ad_AD_SetValArray(var,size,values);
}

void PetscADSetIndepVector(DERIV_TYPE *var, int size,double *values){
  ad_AD_SetIndepVector(var, size, values);
}

void PetscADSetIndepArrayColored(DERIV_TYPE *var,int size,int *coloring){
  ad_AD_SetIndepArrayColored(var,size,coloring);
}

void PetscADIncrementTotalGradSize(int num){
  ad_AD_IncrementTotalGradSize(num);
}

void PetscADSetIndepDone(void){
  ad_AD_SetIndepDone();
}

/* Note that we pass a pointer to DERIV_TYPE, then dereference to match ad_AD_ExtractGrad format */
void PetscADExtractGrad(double *grad, DERIV_TYPE *deriv){
  ad_AD_ExtractGrad(grad,*deriv);
}

int PetscADGetDerivTypeSize(void){
  return sizeof(DERIV_TYPE);
}

double *PetscADGetGradArray(DERIV_TYPE *deriv){
  return deriv->grad;
}

EXTERN_C_END

