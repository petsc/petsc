/*
,
  THIS PROGRAM DISCLOSES MATERIAL PROTECTABLE UNDER COPYRIGHT

  LAWS OF THE UNITED STATES.  FOR LICENSING INFORMATION CONTACT:



  Christian Bischof or Lucas Roh, Mathematics and Computer Science Division,

  Argonne National Laboratory, 9700 S. Cass Avenue, Argonne IL 60439,

  {bischof,roh}@mcs.anl.gov.

*/

#if !defined(AD_GRAD_DYN_H)
#define AD_GRAD_DYN_H

#include "../adic/run-alloc.h"


#if defined(__cplusplus)
extern "C" {
#endif

#define VALIDATE(px) \
    if (!*px) { \
        *px = (double*)ad_adic_deriv_alloc(); \
    } \

#define INVALIDATE(ppx) \
    if (*ppx) { \
        ad_adic_deriv_free(*ppx); \
        *ppx = (double*)0; \
    } 

#define IS_ZERO(px) \
    !px

#define SET_ZERO_FLAG(flag, px, pos)\
    if (IS_ZERO(px)) {\
        flag |= (1<<pos);\
    }

    
#define DAXPY1(ppz, a, pa) \
{\
    int _i; double*pz;\
    VALIDATE(ppz);\
    pz = *ppz; \
    for (_i = 0; _i < ad_grad_size; _i++) {\
        pz[_i] = a*pa[_i];\
    }\
}

#define DAXPY2(ppz, a, pa, b, pb) \
{\
    int _i; double*pz;\
    VALIDATE(ppz);\
    pz = *ppz;\
    for (_i = 0; _i < ad_grad_size; _i++) {\
        pz[_i] = a*pa[_i] + b*pb[_i];\
    }\
}

#define DAXPY3(ppz, a, pa, b, pb, c, pc) \
{\
    int _i; double*pz;\
    VALIDATE(ppz);\
    pz = *ppz;\
    for (_i = 0; _i < ad_grad_size; _i++) {\
        pz[_i] = a*pa[_i] + b*pb[_i] + c*pc[_i];\
    }\
}
void ad_grad_daxpy_init(void);
void ad_grad_daxpy_final(void);
#define ad_grad_daxpy_free(pz) ad_adic_deriv_free(pz)
void ad_grad_daxpy_0(double** ppz);

void ad_grad_daxpy_copy(double** ppz, double* pa);

void ad_grad_daxpy_1(double** pz, double a, double* pa);

void ad_grad_daxpy_2(double** ppz, double a, double* pa, 
                     double b, double* pb);

void ad_grad_daxpy_3(double** ppz, double a, double* pa, 
                     double b, double* pb, double c, double* pc);

void ad_grad_daxpy_n(int n, double** ppz, ...);
void ad_grad_daxpy_4(double** ppz, double ca, double* pa, double cb, double* pb, double cc, double* pc, double cd, double* pd);
void ad_grad_daxpy_5(double** ppz, double ca, double* pa, double cb, double* pb, double cc, double* pc, double cd, double* pd, double ce, double* pe);

#if defined(__cplusplus)
}
#endif
#endif /*AD_GRAD_DYN_H*/
