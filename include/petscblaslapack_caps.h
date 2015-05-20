/*
     This file deals with CAPS BLAS/LAPACK naming convention.
*/
#if !defined(_BLASLAPACK_CAPS_H)
#define _BLASLAPACK_CAPS_H

#if !defined(PETSC_USE_COMPLEX)
# if defined(PETSC_USE_REAL_SINGLE) || defined(PETSC_BLASLAPACK_SINGLEISDOUBLE)
/* Real single precision with no character string arguments */
#  define LAPACKgeqrf_ SGEQRF
#  define LAPACKungqr_ SORGQR
#  define LAPACKgetrf_ SGETRF
#  define LAPACKgetri_ SGETRI
#  define BLASdot_     SDOT
#  define BLASdotu_    SDOT
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
#  define LAPACKgerfs_ SGERFS
#  define LAPACKtgsen_ STGSEN
/* Real single precision with character string arguments. */
#  define LAPACKpotrf_ SPOTRF
#  define LAPACKpotrs_ SPOTRS
#  define LAPACKpotri_ SPOTRI
#  define LAPACKsytrf_ SSYTRF
#  define LAPACKsytrs_ SSYTRS
#  define LAPACKsytri_ SSYTRI
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
#  define BLASsymm_    SSYMM
#  define BLAStrsm_    STRSM
#  define LAPACKstebz_ SSTEBZ
#  define LAPACKsteqr_ SSTEQR  /* eigenvalues of symm tridiagonal matrix */
#  define LAPACKhseqr_ SHSEQR
#  define LAPACKgges_  SGGES
#  define LAPACKtrsen_ STRSEN
#  define LAPACKormqr_ SORMQR
#  define LAPACKhgeqz_ SHGEQZ
#  define LAPACKtrtrs_ STRTRS
# else
/* Real double precision with no character string arguments */
#  define LAPACKgeqrf_ DGEQRF
#  define LAPACKungqr_ DORGQR
#  define LAPACKgetrf_ DGETRF
#  define LAPACKgetri_ DGETRI
#  define BLASdot_     DDOT
#  define BLASdotu_    DDOT
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
#  define LAPACKgerfs_ DGERFS
#  define LAPACKtgsen_ DTGSEN
/* Real double precision with character string arguments. */
#  define LAPACKpotrf_ DPOTRF
#  define LAPACKpotrs_ DPOTRS
#  define LAPACKpotri_ DPOTRI
#  define LAPACKsytrf_ DSYTRF
#  define LAPACKsytrs_ DSYTRS
#  define LAPACKsytri_ DSYTRI
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
#  define BLASsymm_    DSYMM
#  define BLAStrsm_    DTRSM
#  define LAPACKstebz_ DSTEBZ
#  define LAPACKsteqr_ DSTEQR
#  define LAPACKhseqr_ DHSEQR
#  define LAPACKgges_  DGGES
#  define LAPACKtrsen_ DTRSEN
#  define LAPACKormqr_ DORMQR
#  define LAPACKhgeqz_ DHGEQZ
#  define LAPACKtrtrs_ DTRTRS
# endif

#else
# if defined(PETSC_BLASLAPACK_SINGLEISDOUBLE)
/* Complex single precision with no character string arguments */
#  define LAPACKgeqrf_ CGEQRF
#  define LAPACKungqr_ CUNGQR
#  define LAPACKgetrf_ CGETRF
#  define LAPACKgetri_ CGETRI
/* #  define BLASdot_     CDOTC */
/* #  define BLASdotu_    CDOTU */
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
#  define LAPACKgerfs_ CGERFS
#  define LAPACKtgsen_ CTGSEN
/* Complex single precision with character string arguments */
#  define LAPACKpotrf_ CPOTRF
#  define LAPACKpotrs_ CPOTRS
#  define LAPACKpotri_ CPOTRI
#  define LAPACKsytrf_ CSYTRF
#  define LAPACKsytrs_ CSYTRS
#  define LAPACKsytri_ CSYTRI
#  define BLASgemv_    CGEMV
#  define LAPACKgetrs_ CGETRS
#  define BLAStrmv_    CTRMV
#  define BLASgemm_    CGEMM
#  define BLASsymm_    CSYMM
#  define BLAStrsm_    CTRSM
#  define LAPACKgesvd_ CGESVD
#  define LAPACKgeev_  CGEEV
#  define LAPACKsyev_  CSYEV
#  define LAPACKsyevx_ CSYEVX
#  define LAPACKsygv_  CHEGV
#  define LAPACKsygvx_ CHEGVX
#  define LAPACKpttrs_ CPTTRS
/* LAPACKstebz_ does not exist for complex. */
#  define LAPACKsteqr_ CSTEQR
#  define LAPACKhseqr_ CHSEQR
#  define LAPACKgges_  CGGES
#  define LAPACKtrsen_ CTRSEN
#  define LAPACKormqr_ CORMQR
#  define LAPACKhgeqz_ CHGEQZ
#  define LAPACKtrtrs_ CTRTRS
# else
/* Complex double precision with no character string arguments */
#  define LAPACKgeqrf_ ZGEQRF
#  define LAPACKungqr_ ZUNGQR
#  define LAPACKgetrf_ ZGETRF
#  define LAPACKgetri_ ZGETRI
/* #  define BLASdot_     ZDOTC */
/* #  define BLASdotu_    ZDOTU */
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
#  define LAPACKgerfs_ ZGERFS
#  define LAPACKtgsen_ ZTGSEN
/* Complex double precision with character string arguments */
#  define LAPACKpotrf_ ZPOTRF
#  define LAPACKpotrs_ ZPOTRS
#  define LAPACKpotri_ ZPOTRI
#  define LAPACKsytrf_ ZSYTRF
#  define LAPACKsytrs_ ZSYTRS
#  define LAPACKsytri_ ZSYTRI
#  define BLASgemv_    ZGEMV
#  define LAPACKgetrs_ ZGETRS
#  define BLAStrmv_    ZTRMV
#  define BLASgemm_    ZGEMM
#  define BLASsymm_    ZSYMM
#  define BLAStrsm_    ZTRSM
#  define LAPACKgesvd_ ZGESVD
#  define LAPACKgeev_  ZGEEV
#  define LAPACKsyev_  ZHEEV
#  define LAPACKsyevx_ ZHEEVX
#  define LAPACKsygv_  ZHEGV
#  define LAPACKsygvx_ ZHEGVX
#  define LAPACKpttrs_ ZPTTRS
/* LAPACKstebz_ does not exist for complex. */
#  define LAPACKsteqr_ ZSTEQR
#  define LAPACKhseqr_ ZHSEQR
#  define LAPACKgges_  ZGGES
#  define LAPACKtrsen_ ZTRSEN
#  define LAPACKormqr_ ZORMQR
#  define LAPACKhgeqz_ ZHGEQZ
#  define LAPACKtrtrs_ ZTRTRS
# endif
#endif

#endif
