#include <stdio.h> /* why do I need this? */
#include <string.h> /* why do I need this? */

#if defined(__cplusplus)
extern "C" {
#endif

#include "ad_deriv.h"

void PetscADResetIndep(){
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

void PetscADSetIndepDone(){
  ad_AD_SetIndepDone();
}

/* Note that we pass a pointer to DERIV_TYPE, then dereference to match ad_AD_ExtractGrad format */
void PetscADExtractGrad(double *grad, DERIV_TYPE *deriv){
  ad_AD_ExtractGrad(grad,*deriv);
}

int PetscADGetDerivTypeSize(){
  return sizeof(DERIV_TYPE);
}

double *PetscADGetGradArray(DERIV_TYPE *deriv){
  return deriv->grad;
}

#if defined(__cplusplus)
}
#endif
