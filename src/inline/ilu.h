/* $Id: ilu.h,v 1.2 1997/03/26 01:34:10 bsmith Exp bsmith $ */
/*
    Kernels used in sparse ILU (and LU) and in the resulting triangular
 solves. These are for block algorithms where the block sizes are on 
 the order of 6+.

*/
#if !defined(__ILU_H)
#define __ILU_H

#include "pinclude/plapack.h"
extern int  Linpack_DGEFA(Scalar *,int, int *);
extern int  Linpack_DGEDI(Scalar *,int, int *,Scalar*);
extern int  Kernel_A_gets_inverse_A_3(Scalar *);
extern int  Kernel_A_gets_inverse_A_4(Scalar *);
extern int  Kernel_A_gets_inverse_A_5(Scalar *);

/*
      A = A * B   A_gets_A_times_B

   A, B - square bs by bs arrays stored in column major order
   W    - square bs by bs work array

*/
#define Kernel_A_gets_A_times_B(bs,A,B,W) \
{ \
  Scalar _one = 1.0, _zero = 0.0; \
  PetscMemcpy((W),(A),(bs)*(bs)*sizeof(Scalar)); \
  BLgemm_("N","N",&(bs),&(bs),&(bs),&_one,(W),&(bs),(B),&(bs),&_zero,(A),&(bs));\
}

/*

    A = A - B * C  A_gets_A_minus_B_times_C 

   A, B, C - square bs by bs arrays stored in column major order
*/ 
#define Kernel_A_gets_A_minus_B_times_C(bs,A,B,C) \
{ \
  Scalar _mone = -1.0,_one = 1.0; \
  BLgemm_("N","N",&(bs),&(bs),&(bs),&_mone,(B),&(bs),(C),&(bs),&_one,(A),&(bs));\
}

/*
    A = inv(A)    A_gets_inverse_A

   A      - square bs by bs array stored in column major order
   pivots - integer work array of length bs
   W      - square bs by bs work array
*/
#define Kernel_A_gets_inverse_A(bs,A,pivots,W)\
{ \
  ierr = Linpack_DGEFA((A),(bs),(pivots)); CHKERRQ(ierr); \
  ierr = Linpack_DGEDI((A),(bs),(pivots),(W)); CHKERRQ(ierr); \
}

/*
    v = v - A w  v_gets_v_minus_A_times_w

   v - array of length bs
   A - square bs by bs array
   w - array of length bs
*/
#define  Kernel_v_gets_v_minus_A_times_w(bs,v,A,w) \
{  \
  Scalar _mone = -1.0, _one = 1.0; \
  int    _ione = 1; \
  LAgemv_("N",&(bs),&(bs),&_mone,A,&(bs),w,&_ione,&_one,v,&_ione); \
}

/*
    v = v + A w  v_gets_v_plus_A_times_w

   v - array of length bs
   A - square bs by bs array
   w - array of length bs
*/
#define  Kernel_v_gets_v_plus_A_times_w(bs,v,A,w) \
{  \
  Scalar _one = 1.0; \
  int    _ione = 1; \
  LAgemv_("N",&(bs),&(bs),&_one,A,&(bs),w,&_ione,&_one,v,&_ione); \
}

/*
    w = A v   w_gets_A_times_v

   v - array of length bs
   A - square bs by bs array
   w - array of length bs
*/
#define Kernel_w_gets_A_times_v(bs,v,A,w) \
{  \
  Scalar _zero = 0.0, _one = 1.0; \
  int    _ione = 1; \
  LAgemv_("N",&(bs),&(bs),&_one,A,&(bs),v,&_ione,&_zero,w,&_ione); \
}

   
#endif


