/* $Id: ilu.h,v 1.29 2001/08/07 03:01:46 balay Exp $ */
/*
    Kernels used in sparse ILU (and LU) and in the resulting triangular
 solves. These are for block algorithms where the block sizes are on 
 the order of 2-6+.

    There are TWO versions of the macros below. 
    1) standard for MatScalar == PetscScalar use the standard BLAS for 
       block size larger than 7 and
    2) handcoded Fortran single precision for the matrices, since BLAS
       does not have some arguments in single and some in double.

*/
#if !defined(__ILU_H)
#define __ILU_H

#include "petscblaslapack.h"

/*
      These are C kernels,they are contained in 
   src/mat/impls/baij/seq
*/

EXTERN int  LINPACKdgefa(MatScalar *,int,int *);
EXTERN int  LINPACKdgedi(MatScalar *,int,int *,MatScalar*);
EXTERN int  Kernel_A_gets_inverse_A_2(MatScalar *);
EXTERN int  Kernel_A_gets_inverse_A_3(MatScalar *);

#define Kernel_A_gets_inverse_A_4_nopivot(mat) 0;\
{\
  MatScalar d, di;\
\
  di = mat[0];\
  mat[0] = d = 1.0 / di;\
  mat[4] *= -d;\
  mat[8] *= -d;\
  mat[12] *= -d;\
  mat[1] *= d;\
  mat[2] *= d;\
  mat[3] *= d;\
  mat[5] += mat[4] * mat[1] * di;\
  mat[6] += mat[4] * mat[2] * di;\
  mat[7] += mat[4] * mat[3] * di;\
  mat[9] += mat[8] * mat[1] * di;\
  mat[10] += mat[8] * mat[2] * di;\
  mat[11] += mat[8] * mat[3] * di;\
  mat[13] += mat[12] * mat[1] * di;\
  mat[14] += mat[12] * mat[2] * di;\
  mat[15] += mat[12] * mat[3] * di;\
  di = mat[5];\
  mat[5] = d = 1.0 / di;\
  mat[1] *= -d;\
  mat[9] *= -d;\
  mat[13] *= -d;\
  mat[4] *= d;\
  mat[6] *= d;\
  mat[7] *= d;\
  mat[0] += mat[1] * mat[4] * di;\
  mat[2] += mat[1] * mat[6] * di;\
  mat[3] += mat[1] * mat[7] * di;\
  mat[8] += mat[9] * mat[4] * di;\
  mat[10] += mat[9] * mat[6] * di;\
  mat[11] += mat[9] * mat[7] * di;\
  mat[12] += mat[13] * mat[4] * di;\
  mat[14] += mat[13] * mat[6] * di;\
  mat[15] += mat[13] * mat[7] * di;\
  di = mat[10];\
  mat[10] = d = 1.0 / di;\
  mat[2] *= -d;\
  mat[6] *= -d;\
  mat[14] *= -d;\
  mat[8] *= d;\
  mat[9] *= d;\
  mat[11] *= d;\
  mat[0] += mat[2] * mat[8] * di;\
  mat[1] += mat[2] * mat[9] * di;\
  mat[3] += mat[2] * mat[11] * di;\
  mat[4] += mat[6] * mat[8] * di;\
  mat[5] += mat[6] * mat[9] * di;\
  mat[7] += mat[6] * mat[11] * di;\
  mat[12] += mat[14] * mat[8] * di;\
  mat[13] += mat[14] * mat[9] * di;\
  mat[15] += mat[14] * mat[11] * di;\
  di = mat[15];\
  mat[15] = d = 1.0 / di;\
  mat[3] *= -d;\
  mat[7] *= -d;\
  mat[11] *= -d;\
  mat[12] *= d;\
  mat[13] *= d;\
  mat[14] *= d;\
  mat[0] += mat[3] * mat[12] * di;\
  mat[1] += mat[3] * mat[13] * di;\
  mat[2] += mat[3] * mat[14] * di;\
  mat[4] += mat[7] * mat[12] * di;\
  mat[5] += mat[7] * mat[13] * di;\
  mat[6] += mat[7] * mat[14] * di;\
  mat[8] += mat[11] * mat[12] * di;\
  mat[9] += mat[11] * mat[13] * di;\
  mat[10] += mat[11] * mat[14] * di;\
}

EXTERN int  Kernel_A_gets_inverse_A_4(MatScalar *);
# if defined(PETSC_HAVE_SSE)
EXTERN int  Kernel_A_gets_inverse_A_4_SSE(MatScalar *);
# endif

EXTERN int  Kernel_A_gets_inverse_A_5(MatScalar *);
EXTERN int  Kernel_A_gets_inverse_A_6(MatScalar *);
EXTERN int  Kernel_A_gets_inverse_A_7(MatScalar *);


/*
    A = inv(A)    A_gets_inverse_A

   A      - square bs by bs array stored in column major order
   pivots - integer work array of length bs
   W      -  bs by 1 work array
*/
#define Kernel_A_gets_inverse_A(bs,A,pivots,W)\
{ \
  ierr = LINPACKdgefa((A),(bs),(pivots));CHKERRQ(ierr); \
  ierr = LINPACKdgedi((A),(bs),(pivots),(W));CHKERRQ(ierr); \
}

/* -----------------------------------------------------------------------*/

#if !defined(PETSC_USE_MAT_SINGLE)
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
  PetscScalar _one = 1.0,_zero = 0.0; \
  int    _ierr; \
  _ierr = PetscMemcpy((W),(A),(bs)*(bs)*sizeof(MatScalar));CHKERRQ(_ierr); \
  BLgemm_("N","N",&(bs),&(bs),&(bs),&_one,(W),&(bs),(B),&(bs),&_zero,(A),&(bs));\
}

/*

    A = A - B * C  A_gets_A_minus_B_times_C 

   A, B, C - square bs by bs arrays stored in column major order
*/ 
#define Kernel_A_gets_A_minus_B_times_C(bs,A,B,C) \
{ \
  PetscScalar _mone = -1.0,_one = 1.0; \
  BLgemm_("N","N",&(bs),&(bs),&(bs),&_mone,(B),&(bs),(C),&(bs),&_one,(A),&(bs));\
}

/*

    A = A + B^T * C  A_gets_A_plus_Btranspose_times_C 

   A, B, C - square bs by bs arrays stored in column major order
*/ 
#define Kernel_A_gets_A_plus_Btranspose_times_C(bs,A,B,C) \
{ \
  PetscScalar _one = 1.0; \
  BLgemm_("T","N",&(bs),&(bs),&(bs),&_one,(B),&(bs),(C),&(bs),&_one,(A),&(bs));\
}

/*
    v = v + A^T w  v_gets_v_plus_Atranspose_times_w

   v - array of length bs
   A - square bs by bs array
   w - array of length bs
*/
#define  Kernel_v_gets_v_plus_Atranspose_times_w(bs,v,A,w) \
{  \
  PetscScalar _one = 1.0; \
  int    _ione = 1; \
  LAgemv_("T",&(bs),&(bs),&_one,A,&(bs),w,&_ione,&_one,v,&_ione); \
} 

/*
    v = v - A w  v_gets_v_minus_A_times_w

   v - array of length bs
   A - square bs by bs array
   w - array of length bs
*/
#define  Kernel_v_gets_v_minus_A_times_w(bs,v,A,w) \
{  \
  PetscScalar _mone = -1.0,_one = 1.0; \
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
  PetscScalar _one = 1.0; \
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
  PetscScalar _one = 1.0; \
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
  PetscScalar _zero = 0.0,_one = 1.0; \
  int    _ione = 1; \
  LAgemv_("N",&(bs),&(bs),&_one,A,&(bs),v,&_ione,&_zero,w,&_ione); \
}

/*
        z = A*x
*/
#define Kernel_w_gets_Ar_times_v(bs,ncols,x,A,z) \
{ \
  PetscScalar _one = 1.0,_zero = 0.0; \
  int    _ione = 1; \
  LAgemv_("N",&bs,&ncols,&_one,A,&bs,x,&_ione,&_zero,z,&_ione); \
}

/*
        z = trans(A)*x
*/
#define Kernel_w_gets_w_plus_trans_Ar_times_v(bs,ncols,x,A,z) \
{ \
  PetscScalar _one = 1.0; \
  int    _ione = 1; \
  LAgemv_("T",&bs,&ncols,&_one,A,&bs,x,&_ione,&_one,z,&_ione); \
}

#else 
/*
       Version that calls Fortran routines; can handle different precision
   of matrix (array) and vectors
*/
/*
     These are Fortran kernels: They replace certain BLAS routines but
   have some arguments that may be single precision,rather than double
   These routines are provided in src/fortran/kernels/sgemv.F 
   They are pretty pitiful but get the job done. The intention is 
   that for important block sizes (currently 1,2,3,4,5,6,7) custom inlined 
   code is used.
*/
#ifdef PETSC_HAVE_FORTRAN_CAPS
#define msgemv_  MSGEMV
#define msgemvp_ MSGEMVP
#define msgemvm_ MSGEMVM
#define msgemvt_ MSGEMVT
#define msgemmi_ MSGEMMI
#define msgemm_  MSGEMM
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define msgemv_  msgemv
#define msgemvp_ msgemvp
#define msgemvm_ msgemvm
#define msgemvt_ msgemvt
#define msgemmi_ msgemmi
#define msgemm_  msgemm
#endif
EXTERN_C_BEGIN
EXTERN void msgemv_(int *,int *,MatScalar*,PetscScalar*,PetscScalar*);
EXTERN void msgemvp_(int *,int *,MatScalar*,PetscScalar*,PetscScalar*);
EXTERN void msgemvm_(int *,int *,MatScalar*,PetscScalar*,PetscScalar*);
EXTERN void msgemvt_(int *,int *,MatScalar*,PetscScalar*,PetscScalar*);
EXTERN void msgemmi_(int *,MatScalar*,MatScalar*,MatScalar*);
EXTERN void msgemm_(int *,MatScalar*,MatScalar*,MatScalar*);
EXTERN_C_END

/*
      A = A * B   A_gets_A_times_B

   A, B - square bs by bs arrays stored in column major order
   W    - square bs by bs work array

*/
#define Kernel_A_gets_A_times_B(bs,A,B,W) \
{ \
  int _ierr; \
  _ierr = PetscMemcpy((W),(A),(bs)*(bs)*sizeof(MatScalar));CHKERRQ(_ierr); \
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

/* These do not work yet */
#define Kernel_A_gets_A_plus_Btranspose_times_C(bs,A,B,C) 
#define  Kernel_v_gets_v_plus_Atranspose_times_w(bs,v,A,w) 


#endif

#endif




