/*
  THIS PROGRAM DISCLOSES MATERIAL PROTECTABLE UNDER COPYRIGHT
  LAWS OF THE UNITED STATES.  FOR LICENSING INFORMATION CONTACT:

  Christian Bischof or Lucas Roh, Mathematics and Computer Science Division,
  Argonne National Laboratory, 9700 S. Cass Avenue, Argonne IL 60439, 
  {bischof,roh}@mcs.anl.gov.
*/

#include "petsc.h"
#include <stdarg.h>

#include "ad_deriv.h"
#include "ad_grad.h"

int ad_grad_size = 0;
int ad_total_grad_size = 0;
int ad_grad_size_shadow = 0;
int iWiLlNeVeRCoNfLiCt = 0;

EXTERN_C_BEGIN

int ad_AD_IncrShadowVar(void)
{ return ad_grad_size_shadow++; }

void ad_AD_CommitShadowVar(void) 
{ ad_grad_size = ad_grad_size_shadow; }

void ad_AD_ResetShadowVar(void) 
{ ad_grad_size_shadow = 0; }

void ad_grad_axpy_n(int arity, void* ddz, ...)
{
  int                i, j, count = 0,found;
  static double      alphas[100], *z,alpha;
  static DERIV_TYPE* grads[100],*grad;
  va_list            parg;

  va_start(parg, ddz);
  for (i = 0; i < arity; i++) {
    alpha = va_arg(parg, double);
    grad  = (DERIV_TYPE*)va_arg(parg, DERIV_TYPE*);

    found = 0;
    for (j=0; j<count; j++) {
      if (grad == grads[j]) {
        alphas[j] += alpha;
        found     = 1;
        break;
      }
    }
    if (!found) {
      alphas[count]  = alpha;
      grads[count++] = grad;
    }
  }
  va_end(parg);

  z = DERIV_grad(*((DERIV_TYPE*)ddz));
  { 
    double  *gradv = DERIV_grad(*grads[0]);
    alpha = alphas[0];
    for (i = 0; i < ad_grad_size_dynamic; i++) {
      z[i] = alpha*gradv[i];
    }
  }
  for (j = 1; j < count; j++) {
    double  *gradv = DERIV_grad(*grads[j]);
    alpha = alphas[j];
    for (i = 0; i < ad_grad_size_dynamic; i++) {
      z[i] += alpha*gradv[i];
    }
  }   
}

EXTERN_C_END

