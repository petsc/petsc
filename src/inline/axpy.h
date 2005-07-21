
/* 
   These are macros for daxpy like operations.  The format is
   APXY(U,Alpha,P,n)
   for
   U += Alpha * P

   In addition,versions that process 2 and 4 vectors are provided; 
   these can give significantly better use of memory resources than
   successive calls to the regular daxpy.
 */

#ifndef APXY

#include "petscblaslapack.h"

/* BGL kernels */
#if defined(PETSC_USE_BGL_KERNELS)
#define fortrancopy   fortrancopy_bgl
#define fortranzero   fortranzero_bgl
#define fortranmaxpy4 fortranmaxpy4_bgl
#define fortranmaxpy3 fortranmaxpy3_bgl
#define fortranmaxpy2 fortranmaxpy2_bgl
#define fortranaypx   fortranaypx_bgl
#define fortranwaxpy  fortranwaxpy_bgl

#endif

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define fortrancopy_ FORTRANCOPY
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define fortrancopy_ fortrancopy
#endif
EXTERN_C_BEGIN
extern void fortrancopy_(PetscInt*,PetscScalar*,PetscScalar*); 
EXTERN_C_END

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define fortranzero_ FORTRANZERO
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define fortranzero_ fortranzero
#endif
EXTERN_C_BEGIN
extern void fortranzero_(PetscInt*,PetscScalar*);
EXTERN_C_END


#if defined(PETSC_USE_FORTRAN_KERNEL_AYPX)
#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define fortranaypx_ FORTRANAYPX
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define fortranaypx_ fortranaypx
#endif
EXTERN_C_BEGIN
extern void fortranaypx_(PetscInt*,const PetscScalar*,PetscScalar*,PetscScalar*); 
EXTERN_C_END
#endif

#if defined(PETSC_USE_FORTRAN_KERNEL_WAXPY)
#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define fortranwaxpy_ FORTRANWAXPY
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define fortranwaxpy_ fortranwaxpy
#endif
EXTERN_C_BEGIN
extern void fortranwaxpy_(PetscInt*,const PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*); 
EXTERN_C_END
#endif

#if defined(PETSC_USE_FORTRAN_KERNEL_MAXPY)

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define fortranmaxpy4_ FORTRANMAXPY4
#define fortranmaxpy3_ FORTRANMAXPY3
#define fortranmaxpy2_ FORTRANMAXPY2
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define fortranmaxpy4_ fortranmaxpy4
#define fortranmaxpy3_ fortranmaxpy3
#define fortranmaxpy2_ fortranmaxpy2
#endif

EXTERN_C_BEGIN
EXTERN void fortranmaxpy4_(void*,void*,void*,void*,void*,void*,void*,void*,void*,PetscInt*);
EXTERN void fortranmaxpy3_(void*,void*,void*,void*,void*,void*,void*,PetscInt*);
EXTERN void fortranmaxpy2_(void*,void*,void*,void*,void*,PetscInt*);
EXTERN_C_END

#define APXY(U,a1,p1,n)  {PetscBLASInt one=1;\
  BLASaxpy_(&n,&a1,p1,&one,U,&one);}
#define APXY2(U,a1,a2,p1,p2,n) { \
  fortranmaxpy2_(U,&a1,&a2,p1,p2,&n);}
#define APXY3(U,a1,a2,a3,p1,p2,p3,n) { \
  fortranmaxpy3_(U,&a1,&a2,&a3,p1,p2,p3,&n);}
#define APXY4(U,a1,a2,a3,a4,p1,p2,p3,p4,n){ \
  fortranmaxpy4_(U,&a1,&a2,&a3,&a4,p1,p2,p3,p4,&n);}

#elif defined(PETSC_USE_UNROLL_KERNELS)

#define APXY(U,Alpha,P,n) {\
  switch (n & 0x3) {\
  case 3: *U++    += Alpha * *P++;\
  case 2: *U++    += Alpha * *P++;\
  case 1: *U++    += Alpha * *P++;\
  n -= 4;case 0: break;}while (n>0) {U[0] += Alpha * P[0];U[1] += Alpha * P[1];\
                                     U[2] += Alpha * P[2]; U[3] += Alpha * P[3]; \
                                     U += 4; P += 4; n -= 4;}}
#define APXY2(U,a1,a2,p1,p2,n) {\
  switch (n & 0x3) {\
  case 3: *U++    += a1 * *p1++ + a2 * *p2++;\
  case 2: *U++    += a1 * *p1++ + a2 * *p2++;\
  case 1: *U++    += a1 * *p1++ + a2 * *p2++;\
  n -= 4;case 0: break;}\
  while (n>0) {U[0]+=a1*p1[0]+a2*p2[0];U[1]+=a1*p1[1]+a2*p2[1];\
               U[2]+=a1*p1[2]+a2*p2[2];U[3]+=a1*p1[3]+a2*p2[3];U+=4;p1+=4;p2+=4;n -= 4;}}
#define APXY3(U,a1,a2,a3,p1,p2,p3,n) {\
  switch (n & 0x3) {\
  case 3: *U++    += a1 * *p1++ + a2 * *p2++ + a3 * *p3++;\
  case 2: *U++    += a1 * *p1++ + a2 * *p2++ + a3 * *p3++;\
  case 1: *U++    += a1 * *p1++ + a2 * *p2++ + a3 * *p3++;\
  n -= 4;case 0:break;}while (n>0) {U[0]+=a1*p1[0]+a2*p2[0]+a3*p3[0];\
  U[1]+=a1*p1[1]+a2*p2[1]+a3*p3[1];\
  U[2]+=a1*p1[2]+a2*p2[2]+a3*p3[2];\
  U[3]+=a1*p1[3]+a2*p2[3]+a3*p3[3];U+=4;p1+=4;p2+=4;p3+=4;n-=4;}}
#define APXY4(U,a1,a2,a3,a4,p1,p2,p3,p4,n) {\
  switch (n & 0x3) {\
  case 3: *U++    += a1 * *p1++ + a2 * *p2++ + a3 * *p3++ + a4 * *p4++;\
  case 2: *U++    += a1 * *p1++ + a2 * *p2++ + a3 * *p3++ + a4 * *p4++;\
  case 1: *U++    += a1 * *p1++ + a2 * *p2++ + a3 * *p3++ + a4 * *p4++;\
  n -= 4;case 0:break;}while (n>0) {U[0]+=a1*p1[0]+a2*p2[0]+a3*p3[0]+a4*p4[0];\
  U[1]+=a1*p1[1]+a2*p2[1]+a3*p3[1]+a4*p4[1];\
  U[2]+=a1*p1[2]+a2*p2[2]+a3*p3[2]+a4*p4[2];\
  U[3]+=a1*p1[3]+a2*p2[3]+a3*p3[3]+a4*p4[3];U+=4;p1+=4;p2+=4;p3+=4;p4+=4;n-=4;}}

#elif defined(PETSC_USE_WHILE_KERNELS)

#define APXY(U,a1,p1,n)  {\
  while (n--) *U++ += a1 * *p1++;}
#define APXY2(U,a1,a2,p1,p2,n)  {\
  while (n--) *U++ += a1 * *p1++ + a2 * *p2++;}
#define APXY3(U,a1,a2,a3,p1,p2,p3,n) {\
  while (n--) *U++ += a1 * *p1++ + a2 * *p2++ + a3 * *p3++;}
#define APXY4(U,a1,a2,a3,a4,p1,p2,p3,p4,n) {\
  while (n--) *U++ += a1 * *p1++ + a2 * *p2++ + a3 * *p3++ + a4 * *p4++;}

#elif defined(PETSC_USE_BLAS_KERNELS)

#define APXY(U,a1,p1,n)  {PetscBLASInt one=1;\
  BLASaxpy_(&n,&a1,p1,&one,U,&one);}
#define APXY2(U,a1,a2,p1,p2,n){APXY(U,a1,p1,n);\
  APXY(U,a2,p2,n);}
#define APXY3(U,a1,a2,a3,p1,p2,p3,n){APXY2(U,a1,a2,p1,p2,n);\
  APXY(U,a3,p3,n);}
#define APXY4(U,a1,a2,a3,a4,p1,p2,p3,p4,n){APXY2(U,a1,a2,p1,p2,n);\
  APXY2(U,a3,a4,p3,p4,n);}

#elif defined(PETSC_USE_FOR_KERNELS)

#define APXY(U,a1,p1,n)  {PetscInt __i;PetscScalar __s1,__s2; \
  for(__i=0;__i<n-1;__i+=2){__s1=a1*p1[__i];__s2=a1*p1[__i+1];\
  __s1+=U[__i];__s2+=U[__i+1];U[__i]=__s1;U[__i+1]=__s2;}\
  if (n & 0x1) U[__i] += a1 * p1[__i];}
#define APXY2(U,a1,a2,p1,p2,n) {PetscInt __i;\
  for(__i=0;__i<n;__i++)U[__i] += a1 * p1[__i] + a2 * p2[__i];}
#define APXY3(U,a1,a2,a3,p1,p2,p3,n){PetscInt __i;\
  for(__i=0;__i<n;__i++)U[__i]+=a1*p1[__i]+a2*p2[__i]+a3*p3[__i];}
#define APXY4(U,a1,a2,a3,a4,p1,p2,p3,p4,n){PetscInt __i;\
  for(__i=0;__i<n;__i++)U[__i]+=a1*p1[__i]+a2*p2[__i]+a3*p3[__i]+a4*p4[__i];}

#else

#define APXY(U,a1,p1,n)  {PetscInt __i;PetscScalar _a1=a1;\
  for(__i=0;__i<n;__i++)U[__i]+=_a1 * p1[__i];}
#define APXY2(U,a1,a2,p1,p2,n) {PetscInt __i;\
  for(__i=0;__i<n;__i++)U[__i] += a1 * p1[__i] + a2 * p2[__i];}
#define APXY3(U,a1,a2,a3,p1,p2,p3,n){PetscInt __i;\
  for(__i=0;__i<n;__i++)U[__i]+=a1*p1[__i]+a2*p2[__i]+a3*p3[__i];}
#define APXY4(U,a1,a2,a3,a4,p1,p2,p3,p4,n){PetscInt __i;\
  for(__i=0;__i<n;__i++)U[__i]+=a1*p1[__i]+a2*p2[__i]+a3*p3[__i]+a4*p4[__i];}

#endif


/* ----------------------------------------------------------------------------
      axpy() but for increments of inc in both U and P 
   ---------------------------------------------------------------------------*/
#ifdef PETSC_USE_UNROLL_KERNELS
#define APXYINC(U,Alpha,P,n,inc) {\
if (n & 0x1) {\
  *U    += Alpha * *P; U += inc; P += inc; n--;}\
while (n>0) {U[0] += Alpha * P[0];U[inc] += Alpha * P[inc];\
  U += 2*inc; P += 2*inc; n -= 2;}}
#define APXY2INC(U,a1,a2,p1,p2,n,inc) {\
if (n & 0x1) {\
  *U    += a1 * *p1 + a2 * *p2; U += inc; p1 += inc; p2 += inc;n--;}\
while (n>0) {U[0] += a1*p1[0]+a2*p2[0];U[inc]+=a1*p1[inc]+a2*p2[inc];\
  U += 2*inc;p1 += 2*inc;p2+=2*inc; n -= 2;}}
#define APXY3INC(U,a1,a2,a3,p1,p2,p3,n,inc) {\
if (n & 0x1) {\
   *U    += a1 * *p1 + a2 * *p2 + a3 * *p3; \
    U += inc; p1 += inc; p2 += inc; p3 += inc;n--;}\
while (n>0) {U[0] += a1*p1[0]+a2*p2[0]+a3*p3[0];\
  U[inc]+=a1*p1[inc]+a2*p2[inc]+a3*p3[inc];\
  U += 2*inc;p1 += 2*inc;p2+=2*inc;p3+=2*inc;n -= 2;}}
#define APXY4INC(U,a1,a2,a3,a4,p1,p2,p3,p4,n,inc) {\
if (n & 0x1) {\
   *U    += a1 * *p1 + a2 * *p2 + a3 * *p3 + a4 * *p4; \
    U += inc; p1 += inc; p2 += inc; p3 += inc; p4 += inc;n--;}\
while (n>0) {U[0] += a1*p1[0]+a2*p2[0]+a3*p3[0]+a4*p4[0];\
  U[inc]+=a1*p1[inc]+a2*p2[inc]+a3*p3[inc]+a4*p4[inc];\
  U += 2*inc;p1 += 2*inc;p2+=2*inc;p3+=2*inc;p4+=2*inc; n -= 2;}}

#elif defined(PETSC_USE_WHILE_KERNELS)
#define APXYINC(U,a1,p1,n,inc) {\
while (n--){*U += a1 * *p1; U += inc; p1 += inc;}}
#define APXY2INC(U,a1,a2,p1,p2,n,inc)  {\
while (n--) {*U += a1 * *p1 + a2 * *p2;\
U+=inc;p1+=inc;p2+=inc;}}
#define APXY3INC(U,a1,a2,a3,p1,p2,p3,n,inc){\
while (n--) {*U+=a1**p1+a2**p2+a3 * *p3;U+=inc;p1+=inc;p2+=inc;p3+=inc;}}
#define APXY4INC(U,a1,a2,a3,a4,p1,p2,p3,p4,n,inc) {\
while (n--) {*U += a1 * *p1 + a2 * *p2 + a3 * *p3 + a4 * *p4;U+=inc;p1+=inc;\
p2+=inc;p3+=inc;p4+=inc;}}

#else
/* These need to be converted to for loops */
#define APXYINC(U,a1,p1,n,inc) {\
while (n--){*U += a1 * *p1; U += inc; p1 += inc;}}
#define APXY2INC(U,a1,a2,p1,p2,n,inc) {\
while (n--) {*U += a1 * *p1 + a2 * *p2;\
U+=inc;p1+=inc;p2+=inc;}}
#define APXY3INC(U,a1,a2,a3,p1,p2,p3,n,inc) {\
while (n--) {*U+=a1**p1+a2**p2+a3 * *p3;U+=inc;p1+=inc;p2+=inc;p3+=inc;}}
#define APXY4INC(U,a1,a2,a3,a4,p1,p2,p3,p4,n,inc){\
while (n--) {*U += a1 * *p1 + a2 * *p2 + a3 * *p3 + a4 * *p4;U+=inc;p1+=inc;\
p2+=inc;p3+=inc;p4+=inc;}}
#endif

/* --------------------------------------------------------------------
   This is aypx:
    for (i=0; i<n; i++) 
       y[i] = x[i] + alpha * y[i];
  ---------------------------------------------------------------------*/
#if defined(PETSC_USE_UNROLL_KERNELS)
#define AYPX(U,Alpha,P,n) {\
switch (n & 0x3) {\
case 3: *U    = *P++ + Alpha * *U;U++;\
case 2: *U    = *P++ + Alpha * *U;U++;\
case 1: *U    = *P++ + Alpha * *U;U++;\
n -= 4;case 0: break;}while (n>0) {U[0] = P[0]+Alpha * U[0];\
U[1] = P[1] + Alpha * U[1];\
U[2] = P[2] + Alpha * U[2]; U[3] = P[3] + Alpha * U[3]; \
U += 4; P += 4; n -= 4;}}

#elif defined(PETSC_USE_WHILE_KERNELS)
#define AYPX(U,a1,p1,n)  {\
while (n--) {*U = *p1++ + a1 * *U;U++;}

#elif defined(PETSC_USE_FOR_KERNELS)
#define AYPX(U,a1,p1,n)  {PetscInt __i;PetscScalar __s1,__s2; \
for(__i=0;__i<n-1;__i+=2){__s1=p1[__i];__s2=p1[__i+1];\
__s1+=a1*U[__i];__s2+=a1*U[__i+1];\
U[__i]=__s1;U[__i+1]=__s2;}\
if (n & 0x1) U[__i] = p1[__i] + a1 * U[__i];}

#else
#define AYPX(U,a1,p1,n)  {PetscInt __i;\
for(__i=0;__i<n;__i++)U[__i]=p1[__i]+a1 * U[__i];}
#endif

/* ----------------------------------------------------------------------------------
       Useful for APXY where alpha == -1 
  ----------------------------------------------------------------------------------
  */
#define YMX(U,p1,n)  {PetscInt __i;\
for(__i=0;__i<n;__i++)U[__i]-=p1[__i];}
/* Useful for APXY where alpha == 1 */
#define YPX(U,p1,n)  {PetscInt __i;\
for(__i=0;__i<n;__i++)U[__i]+=p1[__i];}

#endif
