/*
  THIS PROGRAM DISCLOSES MATERIAL PROTECTABLE UNDER COPYRIGHT
  LAWS OF THE UNITED STATES.  FOR LICENSING INFORMATION CONTACT:

  Christian Bischof or Lucas Roh, Mathematics and Computer Science Division,
  Argonne National Laboratory, 9700 S. Cass Avenue, Argonne IL 60439, 
  {bischof,roh}@mcs.anl.gov.
*/

#include <string.h>
#include <stdarg.h>

#include "ad_deriv.h"
#include "ad_grad.h"
#if defined(__cplusplus)
extern "C" {
#endif

int ad_grad_size = 0;
int ad_total_grad_size = 0;
int ad_grad_size_shadow = 0;
int iWiLlNeVeRCoNfLiCt = 0;

int ad_AD_IncrShadowVar(void)
{ return ad_grad_size_shadow++; }

void ad_AD_CommitShadowVar(void) 
{ ad_grad_size = ad_grad_size_shadow; }

void ad_AD_ResetShadowVar(void) 
{ ad_grad_size_shadow = 0; }

void ad_grad_axpy_n(int arity, void* ddz, ...)
{
  int     i, j, count = 0;
  static double   alphas[100], *z;
  static DERIV_TYPE* grads[100];
  va_list parg;
  va_start(parg, ddz);
  
  for (i = 0; i < arity; i++) {
    alphas[count] = va_arg(parg, double);
    grads[count] = (DERIV_TYPE*)va_arg(parg, DERIV_TYPE*);
    if ((grads[count]) != NULL) {
      count++;
    }
  }
  va_end(parg);

  z = DERIV_grad(*((DERIV_TYPE*)ddz));
  { 
    double  *grad = DERIV_grad(*grads[0]);
    for (i = 0; i < ad_grad_size; i++) {
      z[i] = alphas[0]*grad[i];
    }
  }
  for (j = 1; j < count; j++) {
    double  *grad = DERIV_grad(*grads[j]);
    for (i = 0; i < ad_grad_size; i++) {
      z[i] += alphas[j]*grad[i];
    }
  }   
}


#if defined(__cplusplus)
}
#endif

