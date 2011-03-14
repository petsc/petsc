
/*
  THIS PROGRAM DISCLOSES MATERIAL PROTECTABLE UNDER COPYRIGHT
  LAWS OF THE UNITED STATES.  FOR LICENSING INFORMATION CONTACT:

  Christian Bischof or Lucas Roh, Mathematics and Computer Science Division,
  Argonne National Laboratory, 9700 S. Cass Avenue, Argonne IL 60439, 
  {bischof,roh}@mcs.anl.gov.
*/

#include <petscsys.h>
#include <stdarg.h>

#include <ad_deriv.h>
#include <ad_grad.h>

int ad_grad_size = 0;
int ad_total_grad_size = 0;
int ad_grad_size_shadow = 0;


EXTERN_C_BEGIN

int ad_AD_IncrShadowVar(void)
{ return ad_grad_size_shadow++; }

void ad_AD_CommitShadowVar(void) 
{ ad_grad_size = ad_grad_size_shadow; }

void ad_AD_ResetShadowVar(void) 
{ ad_grad_size_shadow = 0; }

void ad_grad_axpy_n(int arity, void* ddz, ...)
{
  int                i, j;
  double             *z,alpha,*gradv;
  static double      alphas[100];
  static DERIV_TYPE* grads[100];
  va_list            parg;

  va_start(parg, ddz);
  for (i = 0; i < arity; i++) {
    alphas[i] = va_arg(parg, double);
    grads[i]  = (DERIV_TYPE*)va_arg(parg, DERIV_TYPE*);
  }
  va_end(parg);

  z = DERIV_grad(*((DERIV_TYPE*)ddz));
  { 
    gradv = DERIV_grad(*grads[0]);
    alpha = alphas[0];
    for (i = 0; i < ad_grad_size; i++) {
      z[i] = alpha*gradv[i];
    }
  }
  for (j = 1; j < arity; j++) {
    gradv = DERIV_grad(*grads[j]);
    alpha = alphas[j];
    for (i = 0; i < ad_grad_size; i++) {
      z[i] += alpha*gradv[i];
    }
  }   
  PetscLogFlops(2.0*ad_grad_size*(arity-.5));
}

void mfad_grad_axpy_n(int arity, void* ddz, ...)
{
  int                j;
  double             *z,*gradv;
  static double      alphas[100];
  static DERIV_TYPE* grads[100];
  va_list            parg;

  va_start(parg, ddz);
  for (j = 0; j < arity; j++) {
    alphas[j] = va_arg(parg, double);
    grads[j]  = (DERIV_TYPE*)va_arg(parg, DERIV_TYPE*);
  }
  va_end(parg);

  z = DERIV_grad(*((DERIV_TYPE*)ddz));
  { 
    gradv = DERIV_grad(*grads[0]);
    z[0] = alphas[0]*gradv[0];
  }

  for (j = 1; j < arity; j++) {
    gradv = DERIV_grad(*grads[j]);
    z[0] += alphas[j]*gradv[0];
  }
  PetscLogFlops(2.0*(arity-.5));
}

EXTERN_C_END

