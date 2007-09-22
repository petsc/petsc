/*
   This file provides some name space protection from LAPACK and BLAS and
allows the appropriate single or double precision version to be used.

This file also deals with unmangled Fortran 77 naming convention.
*/
#if !defined(_BLASLAPACK_C_H)
#define _BLASLAPACK_C_H
#include "petsc.h"
PETSC_EXTERN_CXX_BEGIN
EXTERN_C_BEGIN

#if !defined(PETSC_USE_COMPLEX)
# if defined(PETSC_USE_SINGLE)
/* Real single precision with no character string arguments */
#  define LAPACKgeqrf_ sgeqrf
#  define LAPACKungqr_ sorgqr
#  define LAPACKgetrf_ sgetrf
#  define BLASdot_     sdot
#  define BLASnrm2_    snrm2
#  define BLASscal_    sscal
#  define BLAScopy_    scopy
#  define BLASswap_    sswap
#  define BLASaxpy_    saxpy
#  define BLASasum_    sasum
#  define LAPACKpttrf_ spttrf /* factorization of a spd tridiagonal matrix */
#  define LAPACKpttrs_ spttrs /* solve a spd tridiagonal matrix system */
#  define LAPACKstein_ sstein /* eigenvectors of real symm tridiagonal matrix */
/* Real single precision with character string arguments. */
#  define LAPACKormqr_ sormqr
#  define LAPACKtrtrs_ strtrs
#  define LAPACKpotrf_ spotrf
#  define LAPACKpotrs_ spotrs
#  define BLASgemv_    sgemv
#  define LAPACKgetrs_ sgetrs
#  define BLAStrmv_    strmv
#  define BLASgemm_    sgemm
#  define LAPACKgesvd_ sgesvd
#  define LAPACKgeev_  sgeev
#  define LAPACKsyev_  ssyev  /* eigenvalues and eigenvectors of a symm matrix */
#  define LAPACKsyevx_ ssyevx /* selected eigenvalues and eigenvectors of a symm matrix */
#  define LAPACKsygv_  ssygv
#  define LAPACKsygvx_ ssygvx
#  define LAPACKstebz_ sstebz /* eigenvalues of symm tridiagonal matrix */
# else
/* Real double precision with no character string arguments */
#  define LAPACKgeqrf_ dgeqrf
#  define LAPACKungqr_ dorgqr
#  define LAPACKgetrf_ dgetrf
#  define BLASdot_     ddot
#  define BLASnrm2_    dnrm2
#  define BLASscal_    dscal
#  define BLAScopy_    dcopy
#  define BLASswap_    dswap
#  define BLASaxpy_    daxpy
#  define BLASasum_    dasum
#  define LAPACKpttrf_ dpttrf 
#  define LAPACKpttrs_ dpttrs 
#  define LAPACKstein_ dstein
/* Real double precision with character string arguments. */
#  define LAPACKormqr_ dormqr
#  define LAPACKtrtrs_ dtrtrs
#  define LAPACKpotrf_ dpotrf
#  define LAPACKpotrs_ dpotrs
#  define BLASgemv_    dgemv
#  define LAPACKgetrs_ dgetrs
#  define BLAStrmv_    dtrmv
#  define BLASgemm_    dgemm
#  define LAPACKgesvd_ dgesvd
#  define LAPACKgeev_  dgeev
#  define LAPACKsyev_  dsyev
#  define LAPACKsyevx_ dsyevx
#  define LAPACKsygv_  dsygv
#  define LAPACKsygvx_ dsygvx
#  define LAPACKstebz_ dstebz
# endif
#else
/* Complex double precision with no character string arguments */
# define LAPACKgeqrf_ zgeqrf
# define LAPACKungqr_ zungqr
# define LAPACKgetrf_ zgetrf

# define BLASdot_     zdotc
# define BLASnrm2_    dznrm2
# define BLASscal_    zscal
# define BLAScopy_    zcopy
# define BLASswap_    zswap
# define BLASaxpy_    zaxpy
# define BLASasum_    dzasum
#  define LAPACKpttrf_ zpttrf 
#  define LAPACKstein_ zstein
/* Complex double precision with character string arguments */
/* LAPACKormqr_ does not exist for complex. */
# define LAPACKtrtrs_ ztrtrs
# define LAPACKpotrf_ zpotrf
# define LAPACKpotrs_ zpotrs
# define BLASgemv_    zgemv
# define LAPACKgetrs_ zgetrs
# define BLAStrmv_    ztrmv
# define BLASgemm_    zgemm
# define LAPACKgesvd_ zgesvd
# define LAPACKgeev_  zgeev
# define LAPACKsyev_  zheev /* zsyev */
# define LAPACKsyevx_ zheevx /* zsyevx */
# define LAPACKsygv_  zsygv
# define LAPACKsygvx_ zsygvx
# define LAPACKpttrs_ zpttrs 
/* LAPACKstebz_ does not exist for complex. */
#endif

EXTERN void LAPACKgetrf_(PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
EXTERN void LAPACKungqr_(PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN void LAPACKgeqrf_(PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);

EXTERN PetscReal BLASdot_(PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*);
EXTERN PetscReal BLASnrm2_(PetscBLASInt*,PetscScalar*,PetscBLASInt*);
EXTERN void      BLASscal_(PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*);
EXTERN void      BLAScopy_(PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*);
EXTERN void      BLASswap_(PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*);
EXTERN void      BLASaxpy_(PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*);
EXTERN PetscReal BLASasum_(PetscBLASInt*,PetscScalar*,PetscBLASInt*);
EXTERN void LAPACKpttrf_(PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*);
EXTERN void LAPACKstein_(PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);

EXTERN void LAPACKormqr_(const char*,const char*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN void LAPACKtrtrs_(const char*,const char*,const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN void LAPACKpotrf_(const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN void LAPACKpotrs_(const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN void BLASgemv_(const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar *,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*);
EXTERN void LAPACKgetrs_(const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN void BLAStrmv_(const char*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*);
EXTERN void BLASgemm_(const char*,const char*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*);

/* ESSL uses a different calling sequence for dgeev(), zgeev() than LAPACK; */
#if defined(PETSC_HAVE_ESSL) && defined(PETSC_USE_COMPLEX)
EXTERN void LAPACKgeev_(PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*);
EXTERN void LAPACKgesvd_(const char*,const char*,PetscBLASInt *,PetscBLASInt*,PetscScalar *,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*);
#elif defined(PETSC_HAVE_ESSL)
EXTERN void LAPACKgeev_(PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*);
EXTERN void LAPACKgesvd_(const char*,const char*,PetscBLASInt *,PetscBLASInt*,PetscScalar *,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#elif !defined(PETSC_USE_COMPLEX)
EXTERN void LAPACKgeev_(const char*,const char*,PetscBLASInt *,PetscScalar *,PetscBLASInt*,PetscReal*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN void LAPACKgesvd_(const char*,const char*,PetscBLASInt *,PetscBLASInt*,PetscScalar *,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);

EXTERN void LAPACKsyev_(const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN void LAPACKsyevx_(const char*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);

EXTERN void LAPACKsygv_(PetscBLASInt*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN void LAPACKsygvx_(PetscBLASInt*,const char*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
EXTERN void LAPACKpttrs_(PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN void LAPACKstebz_(const char*,const char*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*);
#else
EXTERN void LAPACKgeev_(const char*,const char*,PetscBLASInt *,PetscScalar *,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*);

EXTERN void LAPACKsyev_(const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*);
EXTERN void LAPACKsyevx_(const char*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*);

EXTERN void LAPACKsygv_(PetscBLASInt*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN void LAPACKsygvx_(PetscBLASInt*,const char*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
EXTERN void LAPACKgesvd_(const char*,const char*,PetscBLASInt *,PetscBLASInt*,PetscScalar *,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*);
EXTERN void LAPACKpttrs_(const char*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#endif

EXTERN_C_END
PETSC_EXTERN_CXX_END
#endif
