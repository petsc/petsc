/*
   This file provides some name space protection from LAPACK and BLAS and
allows the appropriate single or double precision version to be used.

This file also deals with CAPS Fortran 77 naming convention.
*/
#if !defined(_BLASLAPACK_CAPS_H)
#define _BLASLAPACK_CAPS_H
#include "petsc.h"
PETSC_EXTERN_CXX_BEGIN
EXTERN_C_BEGIN

#if !defined(PETSC_USE_COMPLEX)
# if defined(PETSC_USE_SINGLE) || defined(PETSC_USES_FORTRAN_SINGLE)
/* Real single precision with no character string arguments */
#  define LAPACKgeqrf_ SGEQRF
#  define LAPACKungqr_ SORGQR
#  define LAPACKgetrf_ SGETRF
#  define BLASdot_     SDOT
#  define BLASnrm2_    SNRM2
#  define BLASscal_    SSCAL
#  define BLAScopy_    SCOPY
#  define BLASswap_    SSWAP
#  define BLASaxpy_    SAXPY
#  define BLASasum_    SASUM
#  define LAPACKpttrf_ SPTTRF
#  define LAPACKpttrs_ SPTTRS
#  define LAPACKstein_ STEIN
/* Real single precision with character string arguments. */
#  define LAPACKormqr_ SORMQR
#  define LAPACKungqr_ DORGQR
#  define LAPACKtrtrs_ STRTRS
#  define LAPACKpotrf_ SPOTRF
#  define LAPACKpotrs_ SPOTRS
#  define BLASgemv_    SGEMV
#  define LAPACKgetrs_ SGETRS
#  define BLAStrmv_    STRMV
#  define LAPACKgesvd_ SGESVD
#  define LAPACKgeev_  SGEEV
#  define LAPACKsyev_  SSYEV
#  define LAPACKsyevx_ SSYEVX
#  define LAPACKsygv_  SSYGV
#  define LAPACKsygvx_ SSYGVX
#  define BLASgemm_    SGEMM
#  define LAPACKstebz_ SSTEBZ
# else
/* Real double precision with no character string arguments */
#  define LAPACKgeqrf_ DGEQRF
#  define LAPACKgetrf_ DGETRF
#  define BLASdot_     DDOT
#  define BLASnrm2_    DNRM2
#  define BLASscal_    DSCAL
#  define BLAScopy_    DCOPY
#  define BLASswap_    DSWAP
#  define BLASaxpy_    DAXPY
#  define BLASasum_    DASUM
#  define LAPACKpttrf_ DPTTRF
#  define LAPACKpttrs_ DPTTRS 
#  define LAPACKstein_ DSTEIN
/* Real double precision with character string arguments. */
#  define LAPACKormqr_ DORMQR
#  define LAPACKtrtrs_ DTRTRS
#  define LAPACKpotrf_ DPOTRF
#  define LAPACKpotrs_ DPOTRS
#  define BLASgemv_    DGEMV
#  define LAPACKgetrs_ DGETRS
#  define BLAStrmv_    DTRMV
#  define LAPACKgesvd_ DGESVD
#  define LAPACKgeev_  DGEEV
#  define LAPACKsyev_  DSYEV
#  define LAPACKsyevx_ DSYEVX
#  define LAPACKsygv_  DSYGV
#  define LAPACKsygvx_ DSYGVX
#  define BLASgemm_    DGEMM
#  define LAPACKstebz_ DSTEBZ
# endif

#else
# if defined(PETSC_USES_FORTRAN_SINGLE)
/* Complex single precision with no character string arguments */
#  define LAPACKgeqrf_ CGEQRF
#  define LAPACKungqr_ CUNGQR
#  define LAPACKgetrf_ CGETRF
#  define BLASdot_     CDOTC
#  define BLASnrm2_    SCNRM2
#  define BLASscal_    CSCAL
#  define BLAScopy_    CCOPY
#  define BLASswap_    CSWAP
#  define BLASaxpy_    CAXPY
#  define BLASasum_    SCASUM
#  define LAPACKpttrf_ CPTTRF
#  define LAPACKstein_ CSTEIN
/* Complex single precision with character string arguments */
/* LAPACKormqr_ does not exist for complex. */
#  define LAPACKtrtrs_ CTRTRS
#  define LAPACKpotrf_ CPOTRF
#  define LAPACKpotrs_ CPOTRS
#  define BLASgemv_    CGEMV
#  define LAPACKgetrs_ CGETRS
#  define BLAStrmv_    CTRMV
#  define BLASgemm_    SGEMM
#  define LAPACKgesvd_ CGESVD
#  define LAPACKgeev_  CGEEV
#  define LAPACKsyev_  CSYEV
#  define LAPACKsyevx_ CSYEVX
#  define LAPACKsygv_  CSYGV
#  define LAPACKsygvx_ CSYGVX
#  define LAPACKpttrs_ CPTTRS 
/* LAPACKstebz_ does not exist for complex. */
# else
/* Complex double precision with no character string arguments */
#  define LAPACKgeqrf_ ZGEQRF
#  define LAPACKgetrf_ ZGETRF
#  define BLASdot_     ZDOTC
#  define BLASnrm2_    DZNRM2
#  define BLASscal_    ZSCAL
#  define BLAScopy_    ZCOPY
#  define BLASswap_    ZSWAP
#  define BLASaxpy_    ZAXPY
#  define BLASasum_    DZASUM
#  define LAPACKpttrf_ ZPTTRF
#  define LAPACKstein_ ZSTEIN
/* Complex double precision with character string arguments */
/* LAPACKormqr_ does not exist for complex. */
#  define LAPACKtrtrs_ ZTRTRS
#  define LAPACKpotrf_ ZPOTRF
#  define LAPACKpotrs_ ZPOTRS
#  define BLASgemv_    ZGEMV
#  define LAPACKgetrs_ ZGETRS
#  define BLAStrmv_    ZTRMV
#  define BLASgemm_    ZGEMM
#  define LAPACKgesvd_ ZGESVD
#  define LAPACKgeev_  ZGEEV
#  define LAPACKsyev_  ZSYEV
#  define LAPACKsyevx_ ZSYEVX
#  define LAPACKsygv_  ZSYGV
#  define LAPACKsygvx_ ZSYGVX
#  define LAPACKpttrs_ ZPTTRS 
/* LAPACKstebz_ does not exist for complex. */
# endif
#endif

EXTERN void LAPACKgetrf_(PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
EXTERN void LAPACKgeqrf_(PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN void LAPACKungqr_(PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);

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
EXTERN void LAPACKgetrs_(const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN void BLASgemv_(const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar *,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*);
EXTERN void BLAStrmv_(const char*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*);
EXTERN void BLASgemm_(const char*,const char*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*);


#if !defined(PETSC_USE_COMPLEX)
EXTERN void LAPACKgeev_(const char*,const char*,PetscBLASInt *,PetscScalar *,PetscBLASInt*,PetscReal*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN void LAPACKgesvd_(const char*,const char*,PetscBLASInt *,PetscBLASInt*,PetscScalar *,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);

EXTERN void LAPACKsyev_(const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN void LAPACKsyevx_(const char*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);

EXTERN void LAPACKsygv_(PetscBLASInt*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN void LAPACKsygvx_(PetscBLASInt*,const char*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
EXTERN void  LAPACKpttrs_(PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN void LAPACKstebz_(const char*,const char*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*);
#else
EXTERN void LAPACKgeev_(const char*,const char*,PetscBLASInt *,PetscScalar *,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*);

EXTERN void LAPACKsyev_(const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*);
EXTERN void LAPACKsyevx_(const char*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*);

EXTERN void LAPACKsygv_(PetscBLASInt*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN void LAPACKsygvx_(PetscBLASInt*,const char*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
EXTERN void LAPACKgesvd_(const char*,const char*,PetscBLASInt *,PetscBLASInt*,PetscScalar *,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*);
EXTERN void  LAPACKpttrs_(const char*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#endif

EXTERN_C_END
PETSC_EXTERN_CXX_END
#endif
