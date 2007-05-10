/*
   This file provides some name space protection from LAPACK and BLAS and
allows the appropriate single or double precision version to be used.

This file also deals with underscore Fortran 77 naming conventions.
This file is also used for the f2cblaslapack distribution.
*/
#if !defined(_BLASLAPACK_USCORE_H)
#define _BLASLAPACK_USCORE_H
#include "petsc.h"
PETSC_EXTERN_CXX_BEGIN
EXTERN_C_BEGIN

#if !defined(PETSC_USE_COMPLEX)
# if defined(PETSC_USE_SINGLE)
/* Real single precision with no character string arguments */
#  define LAPACKgeqrf_ sgeqrf_
#  define LAPACKgetrf_ sgetrf_
#  define BLASdot_     sdot_
#  define BLASnrm2_    snrm2_
#  define BLASscal_    sscal_
#  define BLAScopy_    scopy_
#  define BLASswap_    sswap_
#  define BLASaxpy_    saxpy_
#  define BLASasum_    sasum_
#  define LAPACKpttrf_ spttrf_
#  define LAPACKpttrs_ spttrs_ 
#  define LAPACKstein_ sstein_ 
/* Real single precision with character string arguments. */
#  define LAPACKormqr_ sormqr_
#  define LAPACKtrtrs_ strtrs_
#  define LAPACKpotrf_ spotrf_
#  define LAPACKpotrs_ spotrs_
#  define BLASgemv_    sgemv_
#  define LAPACKgetrs_ sgetrs_
#  define BLAStrmv_    strmv_
#  define BLASgemm_    sgemm_
#  define LAPACKgesvd_ sgesvd_
#  define LAPACKgeev_  sgeev_
#  define LAPACKsyev_  ssyev_
#  define LAPACKsyevx_ ssyevx_
#  define LAPACKsygv_  ssygv_
#  define LAPACKsygvx_ ssygvx_
#  define LAPACKstebz_ sstebz_
# else
/* Real double precision with no character string arguments */
#  define LAPACKgeqrf_ dgeqrf_
#  define LAPACKgetrf_ dgetrf_
#  define BLASdot_     ddot_
#  define BLASnrm2_    dnrm2_
#  define BLASscal_    dscal_
#  define BLAScopy_    dcopy_
#  define BLASswap_    dswap_
#  define BLASaxpy_    daxpy_
#  define BLASasum_    dasum_
#  define LAPACKpttrf_ dpttrf_
#  define LAPACKpttrs_ dpttrs_
#  define LAPACKstein_ dstein_
/* Real double precision with character string arguments. */
#  define LAPACKormqr_ dormqr_
#  define LAPACKtrtrs_ dtrtrs_
#  define LAPACKpotrf_ dpotrf_
#  define LAPACKpotrs_ dpotrs_
#  define BLASgemv_    dgemv_
#  define LAPACKgetrs_ dgetrs_
#  define BLAStrmv_    dtrmv_
#  define BLASgemm_    dgemm_
#  define LAPACKgesvd_ dgesvd_
#  define LAPACKgeev_  dgeev_
#  define LAPACKsyev_  dsyev_
#  define LAPACKsyevx_ dsyevx_
#  define LAPACKsygv_  dsygv_
#  define LAPACKsygvx_ dsygvx_
#  define LAPACKstebz_ dstebz_
# endif
#else
/* Complex double precision with no character string arguments */
# define LAPACKgeqrf_ zgeqrf_
# define LAPACKgetrf_ zgetrf_
# define BLASdot_     zdotc_
# define BLASnrm2_    dznrm2_
# define BLASscal_    zscal_
# define BLAScopy_    zcopy_
# define BLASswap_    zswap_
# define BLASaxpy_    zaxpy_
# define BLASasum_    dzasum_
#  define LAPACKpttrf_ zpttrf_
#  define LAPACKstein_ zstein_
/* Complex double precision with character string arguments */
/* LAPACKormqr_ does not exist for complex. */
# define LAPACKtrtrs_ ztrtrs_
# define LAPACKpotrf_ zpotrf_
# define LAPACKpotrs_ zpotrs_
# define BLASgemv_    zgemv_
# define LAPACKgetrs_ zgetrs_
# define BLAStrmv_    ztrmv_
# define BLASgemm_    zgemm_
# define LAPACKgesvd_ zgesvd_
# define LAPACKgeev_  zgeev_
# define LAPACKsyev_  zsyev_
# define LAPACKsyevx_ zsyevx_
# define LAPACKsygv_  zsygv_
# define LAPACKsygvx_ zsygvx_
#  define LAPACKpttrs_ zpttrs_ 
/* LAPACKstebz_ does not exist for complex. */
#endif

EXTERN void LAPACKgetrf_(PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
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
// --- new
EXTERN void LAPACKpttrs_(const char*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#endif

EXTERN_C_END
PETSC_EXTERN_CXX_END
#endif
