
#include "petsc.h"
#include "petscis.h"

#if defined(PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#endif

EXTERN_C_BEGIN

#include "ad_deriv.h"

void PetscADSetValueAndColor(DERIV_TYPE *vars,int n,ISColoringValue *colors,double *values)
{
  int         i,j;
  PetscScalar *d;

  for (i=0; i<n; i++) {
    DERIV_val(vars[i]) = values[i];
    d = DERIV_grad(vars[i]);
    for (j=0; j<ad_GRAD_MAX; j++) {
      d[j] = 0.0; 
    }
    d[colors[i]] = 1.0; 
  } 
}

void PetscADResetIndep(void)
{
  ad_AD_ResetIndep();
}

void PetscADSetValArray(DERIV_TYPE *var,int size,double *values)
{
  ad_AD_SetValArray(var,size,values);
}

void PetscADSetIndepVector(DERIV_TYPE *var, int size,double *values)
{
  ad_AD_SetIndepVector(var, size, values);
}

void PetscADSetIndepArrayColored(DERIV_TYPE *var,int size,int *coloring)
{
  ad_AD_SetIndepArrayColored(var,size,coloring);
}

void PetscADIncrementTotalGradSize(int num)
{
  ad_AD_IncrementTotalGradSize(num);
}

void PetscADSetIndepDone(void)
{
  ad_AD_SetIndepDone();
}

/* Note that we pass a pointer to DERIV_TYPE, then dereference to match ad_AD_ExtractGrad format */
void PetscADExtractGrad(double *grad, DERIV_TYPE *deriv)
{
  ad_AD_ExtractGrad(grad,*deriv);
}

int PetscADGetDerivTypeSize(void)
{
  return sizeof(DERIV_TYPE);
}

double *PetscADGetGradArray(DERIV_TYPE *deriv)
{
  return deriv->grad;
}

EXTERN_C_END

