/*
      This file deals with unmangled BLAS/LAPACK naming convention on systems that do not modify BLAS/LAPACK function names
      adding an underscore at the end or changing to CAPS. IBM is one such compiler.
*/
#if !defined(_BLASLAPACK_C_H)
#define _BLASLAPACK_C_H

#if !defined(PETSC_USE_COMPLEX)
# if defined(PETSC_USE_REAL_SINGLE)
/* Real single precision with no character string arguments */
#  define LAPACKgeqrf_ sgeqrf
#  define LAPACKungqr_ sorgqr
#  define LAPACKgetrf_ sgetrf
#  define LAPACKgetri_ sgetri
#  define BLASdot_     sdot
#  define BLASdotu_    sdot
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
#  define LAPACKgerfs_ sgerfs
#  define LAPACKtgsen_ stgsen
/* Real single precision with character string arguments. */
#  define LAPACKpotrf_ spotrf
#  define LAPACKpotri_ spotri
#  define LAPACKpotrs_ spotrs
#  define LAPACKpotri_ spotri
#  define LAPACKsytrf_ ssytrf
#  define LAPACKsytrs_ ssytrs
#  define LAPACKsytri_ ssytri
#  define BLASgemv_    sgemv
#  define LAPACKgetrs_ sgetrs
#  define BLAStrmv_    strmv
#  define BLASgemm_    sgemm
#  define BLASsymm_    ssymm
#  define BLAStrsm_    strsm
#  define LAPACKgesvd_ sgesvd
#  define LAPACKgeev_  sgeev
#  define LAPACKsyev_  ssyev  /* eigenvalues and eigenvectors of a symm matrix */
#  define LAPACKsyevx_ ssyevx /* selected eigenvalues and eigenvectors of a symm matrix */
#  define LAPACKsygv_  ssygv
#  define LAPACKsygvx_ ssygvx
#  define LAPACKstebz_ sstebz /* eigenvalues of symm tridiagonal matrix */
#  define LAPACKsteqr_ ssteqr  /* eigenvalues and eigenvectors of symm tridiagonal */
#  define LAPACKhseqr_ shseqr
#  define LAPACKgges_  sgges
#  define LAPACKtrsen_ strsen
#  define LAPACKormqr_ sormqr
#  define LAPACKhgeqz_ shgeqz
#  define LAPACKtrtrs_ strtrs
# else
/* Real double precision with no character string arguments */
#  define LAPACKgeqrf_ dgeqrf
#  define LAPACKungqr_ dorgqr
#  define LAPACKgetrf_ dgetrf
#  define LAPACKgetri_ dgetri
#  define BLASdot_     ddot
#  define BLASdotu_    ddot
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
#  define LAPACKgerfs_ dgerfs
#  define LAPACKtgsen_ dtgsen
/* Real double precision with character string arguments. */
#  define LAPACKpotrf_ dpotrf
#  define LAPACKpotri_ dpotri
#  define LAPACKpotrs_ dpotrs
#  define LAPACKpotri_ dpotri
#  define LAPACKsytrf_ dsytrf
#  define LAPACKsytrs_ dsytrs
#  define LAPACKsytri_ dsytri
#  define BLASgemv_    dgemv
#  define LAPACKgetrs_ dgetrs
#  define BLAStrmv_    dtrmv
#  define BLASgemm_    dgemm
#  define BLASsymm_    dsymm
#  define BLAStrsm_    dtrsm
#  define LAPACKgesvd_ dgesvd
#  define LAPACKgeev_  dgeev
#  define LAPACKsyev_  dsyev
#  define LAPACKsyevx_ dsyevx
#  define LAPACKsygv_  dsygv
#  define LAPACKsygvx_ dsygvx
#  define LAPACKstebz_ dstebz
#  define LAPACKsteqr_ dsteqr
#  define LAPACKhseqr_ dhseqr
#  define LAPACKgges_  dgges
#  define LAPACKtrsen_ dtrsen
#  define LAPACKormqr_ dormqr
#  define LAPACKhgeqz_ dhgeqz
#  define LAPACKtrtrs_ dtrtrs
# endif
#else
# if defined(PETSC_USE_REAL_SINGLE)
/* Complex single precision with no character string arguments */
#  define LAPACKgeqrf_ cgeqrf
#  define LAPACKungqr_ cungqr
#  define LAPACKgetrf_ cgetrf
#  define LAPACKgetri_ cgetri
/* #  define BLASdot_     cdotc */
/* #  define BLASdotu_    cdotu */
#  define BLASnrm2_    scnrm2
#  define BLASscal_    cscal
#  define BLAScopy_    ccopy
#  define BLASswap_    cswap
#  define BLASaxpy_    caxpy
#  define BLASasum_    scasum
#  define LAPACKpttrf_ cpttrf
#  define LAPACKstein_ cstein
#  define LAPACKgelss_ cgelss
#  define LAPACKgerfs_ cgerfs
#  define LAPACKtgsen_ ctgsen
/* Complex single precision with character string arguments */
#  define LAPACKpotrf_ cpotrf
#  define LAPACKpotri_ cpotri
#  define LAPACKpotrs_ cpotrs
#  define LAPACKpotri_ cpotri
#  define LAPACKsytrf_ csytrf
#  define LAPACKsytrs_ csytrs
#  define LAPACKsytri_ csytri
#  define BLASgemv_    cgemv
#  define LAPACKgetrs_ cgetrs
#  define BLAStrmv_    ctrmv
#  define BLASgemm_    cgemm
#  define BLASsymm_    csymm
#  define BLAStrsm_    ctrsm
#  define LAPACKgesvd_ cgesvd
#  define LAPACKgesv_  cgesv
#  define LAPACKgeev_  cgeev
#  define LAPACKsyev_  cheev
#  define LAPACKsyevx_ cheevx
#  define LAPACKsygv_  chegv
#  define LAPACKsygvx_ chegvx
#  define LAPACKpttrs_ cpttrs
#  define LAPACKhseqr_ chseqr
#  define LAPACKgges_  cgges
#  define LAPACKtrsen_ ctrsen
#  define LAPACKormqr_ cormqr
#  define LAPACKhgeqz_ chgeqz
#  define LAPACKtrtrs_ ctrtrs
/* LAPACKstebz_ does not exist for complex. */
#  define LAPACKsteqr_ csteqr
# else
/* Complex double precision with no character string arguments */
#  define LAPACKgeqrf_ zgeqrf
#  define LAPACKungqr_ zungqr
#  define LAPACKgetrf_ zgetrf
#  define LAPACKgetri_ zgetri
/* #  define BLASdot_     zdotc */
/* #  define BLASdotu_    zdotu */
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
#  define LAPACKgerfs_ zgerfs
#  define LAPACKtgsen_ ztgsen
/* Complex double precision with character string arguments */
#  define LAPACKpotrf_ zpotrf
#  define LAPACKpotri_ zpotri
#  define LAPACKpotrs_ zpotrs
#  define LAPACKpotri_ zpotri
#  define LAPACKsytrf_ zsytrf
#  define LAPACKsytrs_ zsytrs
#  define LAPACKsytri_ zsytri
#  define BLASgemv_    zgemv
#  define LAPACKgetrs_ zgetrs
#  define BLAStrmv_    ztrmv
#  define BLASgemm_    zgemm
#  define BLASsymm_    zsymm
#  define BLAStrsm_    ztrsm
#  define LAPACKgesvd_ zgesvd
#  define LAPACKgeev_  zgeev
#  define LAPACKsyev_  zheev
#  define LAPACKsyevx_ zheevx
#  define LAPACKsygv_  zhegv
#  define LAPACKsygvx_ zhegvx
#  define LAPACKpttrs_ zpttrs
#  define LAPACKhseqr_ zhseqr
#  define LAPACKtrsen_ ztrsen
#  define LAPACKgges_  zgges
#  define LAPACKormqr_ zormqr
#  define LAPACKhgeqz_ zhgeqz
#  define LAPACKtrtrs_ ztrtrs
/* LAPACKstebz_ does not exist for complex. */
#  define LAPACKsteqr_ zsteqr
# endif
#endif

#endif
