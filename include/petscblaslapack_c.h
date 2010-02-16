/*
      This file deals with unmangled Fortran 77 naming convention.
*/
#if !defined(_BLASLAPACK_C_H)
#define _BLASLAPACK_C_H

#if !defined(PETSC_USE_COMPLEX)
# if defined(PETSC_USE_SCALAR_SINGLE)
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
#  define LAPACKgesv_  sgesv
#  define LAPACKgelss_ sgelss
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
#  define LAPACKgesv_  dgesv
#  define LAPACKgelss_ dgelss
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
# if defined(PETSC_USE_SCALAR_SINGLE)
/* Complex single precision with no character string arguments */
#  define LAPACKgeqrf_ cgeqrf
#  define LAPACKungqr_ cungqr
#  define LAPACKgetrf_ cgetrf

#  define BLASdot_     cdotc
#  define BLASnrm2_    scnrm2
#  define BLASscal_    cscal
#  define BLAScopy_    ccopy
#  define BLASswap_    cswap
#  define BLASaxpy_    caxpy
#  define BLASasum_    scasum
#  define LAPACKpttrf_ cpttrf 
#  define LAPACKstein_ cstein
#  define LAPACKgelss_ cgelss
/* Complex single precision with character string arguments */
/* LAPACKormqr_ does not exist for complex. */
#  define LAPACKtrtrs_ ctrtrs
#  define LAPACKpotrf_ cpotrf
#  define LAPACKpotrs_ cpotrs
#  define BLASgemv_    cgemv
#  define LAPACKgetrs_ cgetrs
#  define BLAStrmv_    ctrmv
#  define BLASgemm_    cgemm
#  define LAPACKgesvd_ cgesvd
#  define LAPACKgesv_  cgesv
#  define LAPACKgeev_  cgeev
#  define LAPACKsyev_  cheev 
#  define LAPACKsyevx_ cheevx 
#  define LAPACKsygv_  chegv 
#  define LAPACKsygvx_ chegvx 
#  define LAPACKpttrs_ cpttrs 
/* LAPACKstebz_ does not exist for complex. */
# else
/* Complex double precision with no character string arguments */
#  define LAPACKgeqrf_ zgeqrf
#  define LAPACKungqr_ zungqr
#  define LAPACKgetrf_ zgetrf

#  define BLASdot_     zdotc
#  define BLASnrm2_    dznrm2
#  define BLASscal_    zscal
#  define BLAScopy_    zcopy
#  define BLASswap_    zswap
#  define BLASaxpy_    zaxpy
#  define BLASasum_    dzasum
#  define LAPACKpttrf_ zpttrf 
#  define LAPACKstein_ zstein
# define LAPACKgesv_   zgesv
# define LAPACKgelss_  zgelss
/* Complex double precision with character string arguments */
/* LAPACKormqr_ does not exist for complex. */
#  define LAPACKtrtrs_ ztrtrs
#  define LAPACKpotrf_ zpotrf
#  define LAPACKpotrs_ zpotrs
#  define BLASgemv_    zgemv
#  define LAPACKgetrs_ zgetrs
#  define BLAStrmv_    ztrmv
#  define BLASgemm_    zgemm
#  define LAPACKgesvd_ zgesvd
#  define LAPACKgeev_  zgeev
#  define LAPACKsyev_  zheev 
#  define LAPACKsyevx_ zheevx 
#  define LAPACKsygv_  zhegv 
#  define LAPACKsygvx_ zhegvx 
#  define LAPACKpttrs_ zpttrs 
/* LAPACKstebz_ does not exist for complex. */
# endif
#endif

#endif
