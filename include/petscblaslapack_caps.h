/*
     This file deals with CAPS Fortran 77 naming convention.
*/
#if !defined(_BLASLAPACK_CAPS_H)
#define _BLASLAPACK_CAPS_H

#if !defined(PETSC_USE_COMPLEX)
# if defined(PETSC_USE_SCALAR_SINGLE) || defined(PETSC_USES_FORTRAN_SINGLE)
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
#  define LAPACKstein_ SSTEIN
#  define LAPACKgesv_  SGESV
#  define LAPACKgelss_ SGELSS
/* Real single precision with character string arguments. */
#  define LAPACKormqr_ SORMQR
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
#  define LAPACKungqr_ DORGQR
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
#  define LAPACKgesv_  DGESV
#  define LAPACKgelss_ DGELSS
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
#  define LAPACKgesv_  CGESV
#  define LAPACKgelss_ CGELSS
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
#  define LAPACKsygv_  CHEGV
#  define LAPACKsygvx_ CHEGVX
#  define LAPACKpttrs_ CPTTRS 
/* LAPACKstebz_ does not exist for complex. */
# else
/* Complex double precision with no character string arguments */
#  define LAPACKgeqrf_ ZGEQRF
#  define LAPACKungqr_ ZUNGQR
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
#  define LAPACKgesv_  ZGESV
#  define LAPACKgelss_ ZGELSS
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
#  define LAPACKsyev_  ZHEEV  
#  define LAPACKsyevx_ ZHEEVX 
#  define LAPACKsygv_  ZHEGV
#  define LAPACKsygvx_ ZHEGVX
#  define LAPACKpttrs_ ZPTTRS 
/* LAPACKstebz_ does not exist for complex. */
# endif
#endif

#endif
