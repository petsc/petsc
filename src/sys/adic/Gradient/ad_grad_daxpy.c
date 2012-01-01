/*
  THIS PROGRAM DISCLOSES MATERIAL PROTECTABLE UNDER COPYRIGHT
  LAWS OF THE UNITED STATES.  FOR LICENSING INFORMATION CONTACT:

  Paul Hovland and Boyana Norris, Mathematics and Computer Science Division,
  Argonne National Laboratory, 9700 S. Cass Avenue, Argonne IL 60439,
  {hovland,norris}@mcs.anl.gov.
*/

#include <string.h>
#include <stdarg.h>
#include <ad_grad.h>
#include <ad_grad_daxpy.h>
void ad_grad_daxpy_init(void) { 
    ad_adic_deriv_init( ad_grad_size*sizeof(double), 0 );
}
void ad_grad_daxpy_final(void) { 
    ad_adic_deriv_final();
}


void ad_grad_daxpy_0(double** ppz)
{
    INVALIDATE(ppz);
}


void ad_grad_daxpy_copy(double** ppz, double* pa)
{
    if (IS_ZERO(pa)) {
        INVALIDATE(ppz);
    }
    else {
        VALIDATE(ppz);
        memcpy(*ppz, pa, sizeof(double)*ad_grad_size);
    }
}


void ad_grad_daxpy_1(double** ppz, double a, double* pa)
{
    if (IS_ZERO(pa)) {
        INVALIDATE(ppz);
    }
    else {
        DAXPY1(ppz,a,pa);
    }
}


void ad_grad_daxpy_2(double** ppz, double a, double* pa, 
                     double b, double* pb)
{
    if (IS_ZERO(pa)) {
        if (IS_ZERO(pb)) {
            INVALIDATE(ppz);
        }
        else {
            DAXPY1(ppz,b,pb);
        }
    }
    else if (IS_ZERO(pb)) {
        DAXPY1(ppz,a,pa);
    }
    else {
        DAXPY2(ppz,a,pa,b,pb);
    }
}

void ad_grad_daxpy_3(double** ppz, double a, double* pa, 
                     double b, double* pb, double c, double* pc)
{
    if (IS_ZERO(pa)) {
        if (IS_ZERO(pb)) {
            if (IS_ZERO(pc)) { 
                INVALIDATE(ppz);
            }
            else {      
                DAXPY1(ppz,c,pc);
            }
        }
        else if (IS_ZERO(pc)) {
            DAXPY1(ppz,b,pb);
        }
        else { 
            DAXPY2(ppz,b,pb,c,pc);
        }
    }
    else if (IS_ZERO(pb)) {
        if (IS_ZERO(pc)) {
            DAXPY1(ppz,a,pa);
        }
        else { 
            DAXPY2(ppz,a,pa,c,pc);
        }
    }
    else if (IS_ZERO(pc)) {
        DAXPY2(ppz,a,pa,b,pb);
    }
    else {
        DAXPY3(ppz,a,pa,b,pb,c,pc);
    }
}
void ad_grad_daxpy_4(double** ppz, double ca, double* pa, double cb, double* pb, double cc, double* pc, double cd, double* pd){ double *pz; int i, flag = 0;
SET_ZERO_FLAG(flag, pa, 0);
SET_ZERO_FLAG(flag, pb, 1);
SET_ZERO_FLAG(flag, pc, 2);
SET_ZERO_FLAG(flag, pd, 3);
switch (flag) {
case 0:
VALIDATE(ppz); pz = *ppz;
                        for (i = 0; i < ad_grad_size; i++) { pz[i] =  +(ca)*pa[i] +(cb)*pb[i] +(cc)*pc[i] +(cd)*pd[i];}
break;
case 1:
VALIDATE(ppz); pz = *ppz;
                        for (i = 0; i < ad_grad_size; i++) { pz[i] =  +(cb)*pb[i] +(cc)*pc[i] +(cd)*pd[i];}
break;
case 2:
VALIDATE(ppz); pz = *ppz;
                        for (i = 0; i < ad_grad_size; i++) { pz[i] =  +(ca)*pa[i] +(cc)*pc[i] +(cd)*pd[i];}
break;
case 3:
VALIDATE(ppz); pz = *ppz;
                        for (i = 0; i < ad_grad_size; i++) { pz[i] =  +(cc)*pc[i] +(cd)*pd[i];}
break;
case 4:
VALIDATE(ppz); pz = *ppz;
                        for (i = 0; i < ad_grad_size; i++) { pz[i] =  +(ca)*pa[i] +(cb)*pb[i] +(cd)*pd[i];}
break;
case 5:
VALIDATE(ppz); pz = *ppz;
                        for (i = 0; i < ad_grad_size; i++) { pz[i] =  +(cb)*pb[i] +(cd)*pd[i];}
break;
case 6:
VALIDATE(ppz); pz = *ppz;
                        for (i = 0; i < ad_grad_size; i++) { pz[i] =  +(ca)*pa[i] +(cd)*pd[i];}
break;
case 7:
VALIDATE(ppz); pz = *ppz;
                        for (i = 0; i < ad_grad_size; i++) { pz[i] =  +(cd)*pd[i];}
break;
case 8:
VALIDATE(ppz); pz = *ppz;
                        for (i = 0; i < ad_grad_size; i++) { pz[i] =  +(ca)*pa[i] +(cb)*pb[i] +(cc)*pc[i];}
break;
case 9:
VALIDATE(ppz); pz = *ppz;
                        for (i = 0; i < ad_grad_size; i++) { pz[i] =  +(cb)*pb[i] +(cc)*pc[i];}
break;
case 10:
VALIDATE(ppz); pz = *ppz;
                        for (i = 0; i < ad_grad_size; i++) { pz[i] =  +(ca)*pa[i] +(cc)*pc[i];}
break;
case 11:
VALIDATE(ppz); pz = *ppz;
                        for (i = 0; i < ad_grad_size; i++) { pz[i] =  +(cc)*pc[i];}
break;
case 12:
VALIDATE(ppz); pz = *ppz;
                        for (i = 0; i < ad_grad_size; i++) { pz[i] =  +(ca)*pa[i] +(cb)*pb[i];}
break;
case 13:
VALIDATE(ppz); pz = *ppz;
                        for (i = 0; i < ad_grad_size; i++) { pz[i] =  +(cb)*pb[i];}
break;
case 14:
VALIDATE(ppz); pz = *ppz;
                        for (i = 0; i < ad_grad_size; i++) { pz[i] =  +(ca)*pa[i];}
break;
case 15:
INVALIDATE(ppz);
}}
void ad_grad_daxpy_5(double** ppz, double ca, double* pa, double cb, double* pb, double cc, double* pc, double cd, double* pd, double ce, double* pe){ double *pz; int i, flag = 0;
SET_ZERO_FLAG(flag, pa, 0);
SET_ZERO_FLAG(flag, pb, 1);
SET_ZERO_FLAG(flag, pc, 2);
SET_ZERO_FLAG(flag, pd, 3);
SET_ZERO_FLAG(flag, pe, 4);
switch (flag) {
case 0:
VALIDATE(ppz); pz = *ppz;
                        for (i = 0; i < ad_grad_size; i++) { pz[i] =  +(ca)*pa[i] +(cb)*pb[i] +(cc)*pc[i] +(cd)*pd[i] +(ce)*pe[i];}
break;
case 1:
VALIDATE(ppz); pz = *ppz;
                        for (i = 0; i < ad_grad_size; i++) { pz[i] =  +(cb)*pb[i] +(cc)*pc[i] +(cd)*pd[i] +(ce)*pe[i];}
break;
case 2:
VALIDATE(ppz); pz = *ppz;
                        for (i = 0; i < ad_grad_size; i++) { pz[i] =  +(ca)*pa[i] +(cc)*pc[i] +(cd)*pd[i] +(ce)*pe[i];}
break;
case 3:
VALIDATE(ppz); pz = *ppz;
                        for (i = 0; i < ad_grad_size; i++) { pz[i] =  +(cc)*pc[i] +(cd)*pd[i] +(ce)*pe[i];}
break;
case 4:
VALIDATE(ppz); pz = *ppz;
                        for (i = 0; i < ad_grad_size; i++) { pz[i] =  +(ca)*pa[i] +(cb)*pb[i] +(cd)*pd[i] +(ce)*pe[i];}
break;
case 5:
VALIDATE(ppz); pz = *ppz;
                        for (i = 0; i < ad_grad_size; i++) { pz[i] =  +(cb)*pb[i] +(cd)*pd[i] +(ce)*pe[i];}
break;
case 6:
VALIDATE(ppz); pz = *ppz;
                        for (i = 0; i < ad_grad_size; i++) { pz[i] =  +(ca)*pa[i] +(cd)*pd[i] +(ce)*pe[i];}
break;
case 7:
VALIDATE(ppz); pz = *ppz;
                        for (i = 0; i < ad_grad_size; i++) { pz[i] =  +(cd)*pd[i] +(ce)*pe[i];}
break;
case 8:
VALIDATE(ppz); pz = *ppz;
                        for (i = 0; i < ad_grad_size; i++) { pz[i] =  +(ca)*pa[i] +(cb)*pb[i] +(cc)*pc[i] +(ce)*pe[i];}
break;
case 9:
VALIDATE(ppz); pz = *ppz;
                        for (i = 0; i < ad_grad_size; i++) { pz[i] =  +(cb)*pb[i] +(cc)*pc[i] +(ce)*pe[i];}
break;
case 10:
VALIDATE(ppz); pz = *ppz;
                        for (i = 0; i < ad_grad_size; i++) { pz[i] =  +(ca)*pa[i] +(cc)*pc[i] +(ce)*pe[i];}
break;
case 11:
VALIDATE(ppz); pz = *ppz;
                        for (i = 0; i < ad_grad_size; i++) { pz[i] =  +(cc)*pc[i] +(ce)*pe[i];}
break;
case 12:
VALIDATE(ppz); pz = *ppz;
                        for (i = 0; i < ad_grad_size; i++) { pz[i] =  +(ca)*pa[i] +(cb)*pb[i] +(ce)*pe[i];}
break;
case 13:
VALIDATE(ppz); pz = *ppz;
                        for (i = 0; i < ad_grad_size; i++) { pz[i] =  +(cb)*pb[i] +(ce)*pe[i];}
break;
case 14:
VALIDATE(ppz); pz = *ppz;
                        for (i = 0; i < ad_grad_size; i++) { pz[i] =  +(ca)*pa[i] +(ce)*pe[i];}
break;
case 15:
VALIDATE(ppz); pz = *ppz;
                        for (i = 0; i < ad_grad_size; i++) { pz[i] =  +(ce)*pe[i];}
break;
case 16:
VALIDATE(ppz); pz = *ppz;
                        for (i = 0; i < ad_grad_size; i++) { pz[i] =  +(ca)*pa[i] +(cb)*pb[i] +(cc)*pc[i] +(cd)*pd[i];}
break;
case 17:
VALIDATE(ppz); pz = *ppz;
                        for (i = 0; i < ad_grad_size; i++) { pz[i] =  +(cb)*pb[i] +(cc)*pc[i] +(cd)*pd[i];}
break;
case 18:
VALIDATE(ppz); pz = *ppz;
                        for (i = 0; i < ad_grad_size; i++) { pz[i] =  +(ca)*pa[i] +(cc)*pc[i] +(cd)*pd[i];}
break;
case 19:
VALIDATE(ppz); pz = *ppz;
                        for (i = 0; i < ad_grad_size; i++) { pz[i] =  +(cc)*pc[i] +(cd)*pd[i];}
break;
case 20:
VALIDATE(ppz); pz = *ppz;
                        for (i = 0; i < ad_grad_size; i++) { pz[i] =  +(ca)*pa[i] +(cb)*pb[i] +(cd)*pd[i];}
break;
case 21:
VALIDATE(ppz); pz = *ppz;
                        for (i = 0; i < ad_grad_size; i++) { pz[i] =  +(cb)*pb[i] +(cd)*pd[i];}
break;
case 22:
VALIDATE(ppz); pz = *ppz;
                        for (i = 0; i < ad_grad_size; i++) { pz[i] =  +(ca)*pa[i] +(cd)*pd[i];}
break;
case 23:
VALIDATE(ppz); pz = *ppz;
                        for (i = 0; i < ad_grad_size; i++) { pz[i] =  +(cd)*pd[i];}
break;
case 24:
VALIDATE(ppz); pz = *ppz;
                        for (i = 0; i < ad_grad_size; i++) { pz[i] =  +(ca)*pa[i] +(cb)*pb[i] +(cc)*pc[i];}
break;
case 25:
VALIDATE(ppz); pz = *ppz;
                        for (i = 0; i < ad_grad_size; i++) { pz[i] =  +(cb)*pb[i] +(cc)*pc[i];}
break;
case 26:
VALIDATE(ppz); pz = *ppz;
                        for (i = 0; i < ad_grad_size; i++) { pz[i] =  +(ca)*pa[i] +(cc)*pc[i];}
break;
case 27:
VALIDATE(ppz); pz = *ppz;
                        for (i = 0; i < ad_grad_size; i++) { pz[i] =  +(cc)*pc[i];}
break;
case 28:
VALIDATE(ppz); pz = *ppz;
                        for (i = 0; i < ad_grad_size; i++) { pz[i] =  +(ca)*pa[i] +(cb)*pb[i];}
break;
case 29:
VALIDATE(ppz); pz = *ppz;
                        for (i = 0; i < ad_grad_size; i++) { pz[i] =  +(cb)*pb[i];}
break;
case 30:
VALIDATE(ppz); pz = *ppz;
                        for (i = 0; i < ad_grad_size; i++) { pz[i] =  +(ca)*pa[i];}
break;
case 31:
INVALIDATE(ppz);
}}

void ad_grad_daxpy_n(int n, double** ppz, ...)
{
    static double   alphas[100];
    static double*  grads[100];
    int             i, j, count = 0;
    double*         z;
    va_list         parg;

    va_start(parg, ppz);
    for (i = 0; i < n; i++) {
         alphas[count] = va_arg(parg, double);
         grads[count] = va_arg(parg, double*);
         if (!IS_ZERO(grads[count])) {
             count++;
         }
    }
    va_end(parg);

    switch (count) {
      case 0:
          INVALIDATE(ppz);
          break;

      case 1:
          DAXPY1(ppz,alphas[0],grads[0]);
          break;
          
      case 2:
          DAXPY2(ppz,alphas[0],grads[0],alphas[1],grads[1]);
          break;
          
      case 3:
          VALIDATE(ppz);
          DAXPY3(ppz,alphas[0],grads[0],alphas[1],grads[1],alphas[2],grads[2]);
          break;
          
      case 4:
          VALIDATE(ppz);
          z = *ppz;
          for (i = 0; i < ad_grad_size; i++) {
              z[i] = alphas[0]*grads[0][i] + alphas[1]*grads[1][i] +
                     alphas[2]*grads[2][i] + alphas[3]*grads[3][i];
          }
          break;
          
      case 5:
          VALIDATE(ppz);
          z = *ppz;
          for (i = 0; i < ad_grad_size; i++) {
              z[i] = alphas[0]*grads[0][i] + alphas[1]*grads[1][i] +
                     alphas[2]*grads[2][i] + alphas[3]*grads[3][i] +
                     alphas[4]*grads[4][i];
          }
          break;
          
      default:
          z = *ppz;
          for (i = 0; i < ad_grad_size; i++) {
              z[i] = alphas[0]*grads[0][i] + alphas[1]*grads[1][i] +
                     alphas[2]*grads[2][i] + alphas[3]*grads[3][i] +
                     alphas[4]*grads[4][i];
          }
          for (j = 5; j < count; j++) {
              double    a = alphas[j], *grad = grads[j];
              for (i = 0; i < ad_grad_size; i++) {
                  z[i] += a*grad[i];
              }
          }
          break;
    }
    
}
