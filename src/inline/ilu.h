/* $Id: ilu.h,v 1.11 1999/01/06 16:00:43 balay Exp bsmith $ */
/*
    Kernels used in sparse ILU (and LU) and in the resulting triangular
 solves. These are for block algorithms where the block sizes are on 
 the order of 3-6+.

*/
#if !defined(__ILU_H)
#define __ILU_H

#include "pinclude/blaslapack.h"
/*
      These are C kernels, they are contained in 
   src/mat/impls/baij/seq
*/

extern int  LINPACKdgefa(MatScalar *,int, int *);
extern int  LINPACKdgedi(MatScalar *,int, int *,MatScalar*);
extern int  Kernel_A_gets_inverse_A_2(MatScalar *);
extern int  Kernel_A_gets_inverse_A_3(MatScalar *);
extern int  Kernel_A_gets_inverse_A_4(MatScalar *);
extern int  Kernel_A_gets_inverse_A_5(MatScalar *);
extern int  Kernel_A_gets_inverse_A_6(MatScalar *);
extern int  Kernel_A_gets_inverse_A_7(MatScalar *);

/*
     These are Fortran kernels: They replace certain BLAS routines but
   have some arguments that may be single precision, rather than double
   These routines are provided in src/fortran/kernels/sgemv.F 
   They are pretty pitiful but get the job done. The intention is 
   that for important block sizes (currently 3,4,5) custom inlined 
   code is used.
*/
#ifdef HAVE_FORTRAN_CAPS
#define msgemv_  MSGEMV
#define msgemvp_ MSGEMVP
#define msgemvm_ MSGEMVM
#define msgemvt_ MSGEMVT
#define msgemmi_ MSGEMMI
#define msgemm_  MSGEMM
#elif !defined(HAVE_FORTRAN_UNDERSCORE)
#define msgemv_  msgemv
#define msgemvp_ msgemvp
#define msgemvm_ msgemvm
#define msgemvt_ msgemvt
#define msgemmi_ msgemmi
#define msgemm_  msgemm
#endif
EXTERN_C_BEGIN
extern void msgemv_(int *,int *,MatScalar*,Scalar*,Scalar*);
extern void msgemvp_(int *,int *,MatScalar*,Scalar*,Scalar*);
extern void msgemvm_(int *,int *,MatScalar*,Scalar*,Scalar*);
extern void msgemvt_(int *,int *,MatScalar*,Scalar*,Scalar*);
extern void msgemmi_(int *,MatScalar*,MatScalar*,MatScalar*);
extern void msgemm_(int *,MatScalar*,MatScalar*,MatScalar*);
EXTERN_C_END

/*
    A = inv(A)    A_gets_inverse_A

   A      - square bs by bs array stored in column major order
   pivots - integer work array of length bs
   W      -  bs by 1 work array
*/
#define Kernel_A_gets_inverse_A(bs,A,pivots,W)\
{ \
  ierr = LINPACKdgefa((A),(bs),(pivots)); CHKERRQ(ierr); \
  ierr = LINPACKdgedi((A),(bs),(pivots),(W)); CHKERRQ(ierr); \
}

/* -----------------------------------------------------------------------*/

#if !defined(USE_MAT_SINGLE)
/*
        Version that calls the BLAS directly
*/
/*
      A = A * B   A_gets_A_times_B

   A, B - square bs by bs arrays stored in column major order
   W    - square bs by bs work array

*/
#define Kernel_A_gets_A_times_B(bs,A,B,W) \
{ \
  Scalar _one = 1.0, _zero = 0.0; \
  PetscMemcpy((W),(A),(bs)*(bs)*sizeof(MatScalar)); \
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
    v = v + A w  v_gets_v_plus_Ar_times_w

   v - array of length bs
   A - square bs by bs array
   w - array of length bs
*/
#define  Kernel_w_gets_w_plus_Ar_times_v(bs,ncols,v,A,w) \
{  \
  Scalar _one = 1.0; \
  int    _ione = 1; \
  LAgemv_("N",&(bs),&(ncols),&_one,A,&(bs),v,&_ione,&_one,w,&_ione); \
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

/*
        z = A*x
*/
#define Kernel_w_gets_Ar_times_v(bs,ncols,x,A,z) \
{ \
  Scalar _one = 1.0, _zero = 0.0; \
  int    _ione = 1; \
  LAgemv_("N",&bs,&ncols,&_one,A,&bs,x,&_ione,&_zero,z,&_ione); \
}

/*
        z = trans(A)*x
*/
#define Kernel_w_gets_w_plus_trans_Ar_times_v(bs,ncols,x,A,z) \
{ \
  Scalar _one = 1.0; \
  int    _ione = 1; \
  LAgemv_("T",&bs,&ncols,&_one,A,&bs,x,&_ione,&_one,z,&_ione); \
}

#else 
/*
       Version that calls Fortran routines; can handle different precision
   of matrix (array) and vectors
*/
/*
      A = A * B   A_gets_A_times_B

   A, B - square bs by bs arrays stored in column major order
   W    - square bs by bs work array

*/
#define Kernel_A_gets_A_times_B(bs,A,B,W) \
{ \
  PetscMemcpy((W),(A),(bs)*(bs)*sizeof(MatScalar)); \
  msgemmi_(&bs,A,B,W); \
}

/*

    A = A - B * C  A_gets_A_minus_B_times_C 

   A, B, C - square bs by bs arrays stored in column major order
*/ 
#define Kernel_A_gets_A_minus_B_times_C(bs,A,B,C) \
{ \
  msgemm_(&bs,A,B,C); \
}

/*
    v = v - A w  v_gets_v_minus_A_times_w

   v - array of length bs
   A - square bs by bs array
   w - array of length bs
*/
#define  Kernel_v_gets_v_minus_A_times_w(bs,v,A,w) \
{  \
  msgemvm_(&bs,&bs,A,w,v); \
}

/*
    v = v + A w  v_gets_v_plus_A_times_w

   v - array of length bs
   A - square bs by bs array
   w - array of length bs
*/
#define  Kernel_v_gets_v_plus_A_times_w(bs,v,A,w) \
{  \
  msgemvp_(&bs,&bs,A,w,v);\
}

/*
    v = v + A w  v_gets_v_plus_Ar_times_w

   v - array of length bs
   A - square bs by bs array
   w - array of length bs
*/
#define  Kernel_w_gets_w_plus_Ar_times_v(bs,ncol,v,A,w) \
{  \
  msgemvp_(&bs,&ncol,A,v,w);\
}

/*
    w = A v   w_gets_A_times_v

   v - array of length bs
   A - square bs by bs array
   w - array of length bs
*/
#define Kernel_w_gets_A_times_v(bs,v,A,w) \
{  \
  msgemv_(&bs,&bs,A,v,w);\
}
   
/*
        z = A*x
*/
#define Kernel_w_gets_Ar_times_v(bs,ncols,x,A,z) \
{ \
  msgemv_(&bs,&ncols,A,x,z);\
}

/*
        z = trans(A)*x
*/
#define Kernel_w_gets_w_plus_trans_Ar_times_v(bs,ncols,x,A,z) \
{ \
  msgemvt_(&bs,&ncols,A,x,z);\
}

#endif

#endif


