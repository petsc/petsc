/*
   This file deals with underscore Fortran 77 naming conventions and those from the f2cblaslapack distribution.
*/
#if !defined(_BLASLAPACK_USCORE_H)
#define _BLASLAPACK_USCORE_H

#if !defined(PETSC_USE_COMPLEX)
# if defined(PETSC_USE_REAL_SINGLE)
/* Real single precision with no character string arguments */
#  define LAPACKgeqrf_ sgeqrf_
#  define LAPACKungqr_ sorgqr_
#  define LAPACKgetrf_ sgetrf_
#  define LAPACKgetri_ sgetri_
#  define BLASdot_     sdot_
#  define BLASdotu_    sdot_
#  define BLASnrm2_    snrm2_
#  define BLASscal_    sscal_
#  define BLAScopy_    scopy_
#  define BLASswap_    sswap_
#  define BLASaxpy_    saxpy_
#  define BLASasum_    sasum_
#  define LAPACKpttrf_ spttrf_
#  define LAPACKpttrs_ spttrs_
#  define LAPACKstein_ sstein_
#  define LAPACKgesv_  sgesv_
#  define LAPACKgelss_ sgelss_
#  define LAPACKgerfs_ sgerfs_
#  define LAPACKtgsen_ stgsen_
/* Real single precision with character string arguments. */
#  define LAPACKpotrf_ spotrf_
#  define LAPACKpotrs_ spotrs_
#  define LAPACKsytrf_ ssytrf_
#  define LAPACKsytrs_ ssytrs_
#  define BLASgemv_    sgemv_
#  define LAPACKgetrs_ sgetrs_
#  define BLAStrmv_    strmv_
#  define BLASgemm_    sgemm_
#  define BLAStrsm_    strsm_
#  define LAPACKgesvd_ sgesvd_
#  define LAPACKgeev_  sgeev_
#  define LAPACKsyev_  ssyev_
#  define LAPACKsyevx_ ssyevx_
#  define LAPACKsygv_  ssygv_
#  define LAPACKsygvx_ ssygvx_
#  define LAPACKstebz_ sstebz_
#  define LAPACKsteqr_ ssteqr_
#  define LAPACKhseqr_ shseqr_
#  define LAPACKgges_  sgges_
#  define LAPACKtrsen_ strsen_
#  define LAPACKormqr_ sormqr_
#  define LAPACKhgeqz_ shgeqz_
#  define LAPACKtrtrs_ strtrs_
# elif defined(PETSC_USE_REAL_DOUBLE)
/* Real double precision with no character string arguments */
#  define LAPACKgeqrf_ dgeqrf_
#  define LAPACKungqr_ dorgqr_
#  define LAPACKgetrf_ dgetrf_
#  define LAPACKgetri_ dgetri_
#  define BLASdot_     ddot_
#  define BLASdotu_    ddot_
#  define BLASnrm2_    dnrm2_
#  define BLASscal_    dscal_
#  define BLAScopy_    dcopy_
#  define BLASswap_    dswap_
#  define BLASaxpy_    daxpy_
#  define BLASasum_    dasum_
#  define LAPACKpttrf_ dpttrf_
#  define LAPACKpttrs_ dpttrs_
#  define LAPACKstein_ dstein_
#  define LAPACKgesv_  dgesv_
#  define LAPACKgelss_ dgelss_
#  define LAPACKgerfs_ dgerfs_
#  define LAPACKtgsen_ dtgsen_
/* Real double precision with character string arguments. */
#  define LAPACKpotrf_ dpotrf_
#  define LAPACKpotrs_ dpotrs_
#  define LAPACKsytrf_ dsytrf_
#  define LAPACKsytrs_ dsytrs_
#  define BLASgemv_    dgemv_
#  define LAPACKgetrs_ dgetrs_
#  define BLAStrmv_    dtrmv_
#  define BLASgemm_    dgemm_
#  define BLAStrsm_    dtrsm_
#  define LAPACKgesvd_ dgesvd_
#  define LAPACKgeev_  dgeev_
#  define LAPACKsyev_  dsyev_
#  define LAPACKsyevx_ dsyevx_
#  define LAPACKsygv_  dsygv_
#  define LAPACKsygvx_ dsygvx_
#  define LAPACKstebz_ dstebz_
#  define LAPACKsteqr_ dsteqr_
#  define LAPACKhseqr_ dhseqr_
#  define LAPACKgges_  dgges_
#  define LAPACKtrsen_ dtrsen_
#  define LAPACKormqr_ dormqr_
#  define LAPACKhgeqz_ dhgeqz_
#  define LAPACKtrtrs_ dtrtrs_
# else
/* Real quad precision with no character string arguments */
#  define LAPACKgeqrf_ qgeqrf_
#  define LAPACKungqr_ qorgqr_
#  define LAPACKgetrf_ qgetrf_
#  define LAPACKgetri_ qgetri_
#  define BLASdot_     qdot_
#  define BLASdotu_    qdot_
#  define BLASnrm2_    qnrm2_
#  define BLASscal_    qscal_
#  define BLAScopy_    qcopy_
#  define BLASswap_    qswap_
#  define BLASaxpy_    qaxpy_
#  define BLASasum_    qasum_
#  define LAPACKpttrf_ qpttrf_
#  define LAPACKpttrs_ qpttrs_
#  define LAPACKstein_ qstein_
#  define LAPACKgesv_  qgesv_
#  define LAPACKgelss_ qgelss_
#  define LAPACKgerfs_ qgerfs_
#  define LAPACKtgsen_ qtgsen_
/* Real quad precision with character string arguments. */
#  define LAPACKpotrf_ qpotrf_
#  define LAPACKpotrs_ qpotrs_
#  define LAPACKsytrf_ qsytrf_
#  define LAPACKsytrs_ qsytrs_
#  define BLASgemv_    qgemv_
#  define LAPACKgetrs_ qgetrs_
#  define BLAStrmv_    qtrmv_
#  define BLASgemm_    qgemm_
#  define BLAStrsm_    qtrsm_
#  define LAPACKgesvd_ qgesvd_
#  define LAPACKgeev_  qgeev_
#  define LAPACKsyev_  qsyev_
#  define LAPACKsyevx_ qsyevx_
#  define LAPACKsygv_  qsygv_
#  define LAPACKsygvx_ qsygvx_
#  define LAPACKstebz_ qstebz_
#  define LAPACKsteqr_ qsteqr_
#  define LAPACKhseqr_ qhseqr_
#  define LAPACKgges_  qgges_
#  define LAPACKtrsen_ qtrsen_
#  define LAPACKormqr_ qormqr_
#  define LAPACKhgeqz_ qhgeqz_
#  define LAPACKtrtrs_ qtrtrs_
# endif
#else
# if defined(PETSC_USE_REAL_SINGLE)
/* Complex single precision with no character string arguments */
#  define LAPACKgeqrf_ cgeqrf_
#  define LAPACKungqr_ cungqr_
#  define LAPACKgetrf_ cgetrf_
#  define LAPACKgetri_ cgetri_
/* #  define BLASdot_     cdotc_ */
/* #  define BLASdotu_    cdotu_ */
#  define BLASnrm2_    scnrm2_
#  define BLASscal_    cscal_
#  define BLAScopy_    ccopy_
#  define BLASswap_    cswap_
#  define BLASaxpy_    caxpy_
#  define BLASasum_    scasum_
#  define LAPACKpttrf_ cpttrf_
#  define LAPACKstein_ cstein_
#  define LAPACKgelss_ cgelss_
#  define LAPACKgerfs_ cgerfs_
#  define LAPACKtgsen_ ctgsen_
/* Complex single precision with character string arguments */
#  define LAPACKpotrf_ cpotrf_
#  define LAPACKpotrs_ cpotrs_
#  define LAPACKsytrf_ csytrf_
#  define LAPACKsytrs_ csytrs_
#  define BLASgemv_    cgemv_
#  define LAPACKgetrs_ cgetrs_
#  define BLAStrmv_    ctrmv_
#  define BLASgemm_    cgemm_
#  define BLAStrsm_    ctrsm_
#  define LAPACKgesvd_ cgesvd_
#  define LAPACKgesv_  cgesv_
#  define LAPACKgeev_  cgeev_
#  define LAPACKsyev_  cheev_
#  define LAPACKsyevx_ cheevx_
#  define LAPACKsygv_  chegv_
#  define LAPACKsygvx_ chegvx_
#  define LAPACKpttrs_ cpttrs_
/* LAPACKstebz_ does not exist for complex. */
#  define LAPACKsteqr_ csteqr_
#  define LAPACKhseqr_ chseqr_
#  define LAPACKgges_  cgges_
#  define LAPACKtrsen_ ctrsen_
#  define LAPACKormqr_ cormqr_
#  define LAPACKhgeqz_ chgeqz_
#  define LAPACKtrtrs_ ctrtrs_
# elif defined(PETSC_USE_REAL_DOUBLE)
/* Complex double precision with no character string arguments */
#  define LAPACKgeqrf_ zgeqrf_
#  define LAPACKungqr_ zungqr_
#  define LAPACKgetrf_ zgetrf_
#  define LAPACKgetri_ zgetri_
/* #  define BLASdot_     zdotc_ */
/* #  define BLASdotu_    zdotu_ */
#  define BLASnrm2_    dznrm2_
#  define BLASscal_    zscal_
#  define BLAScopy_    zcopy_
#  define BLASswap_    zswap_
#  define BLASaxpy_    zaxpy_
#  define BLASasum_    dzasum_
#  define LAPACKpttrf_ zpttrf_
#  define LAPACKstein_ zstein_
#  define LAPACKgesv_  zgesv_
#  define LAPACKgelss_ zgelss_
#  define LAPACKgerfs_ zgerfs_
#  define LAPACKtgsen_ ztgsen_
/* Complex double precision with character string arguments */
#  define LAPACKpotrf_ zpotrf_
#  define LAPACKpotrs_ zpotrs_
#  define LAPACKsytrf_ zsytrf_
#  define LAPACKsytrs_ zsytrs_
#  define BLASgemv_    zgemv_
#  define LAPACKgetrs_ zgetrs_
#  define BLAStrmv_    ztrmv_
#  define BLASgemm_    zgemm_
#  define BLAStrsm_    ztrsm_
#  define LAPACKgesvd_ zgesvd_
#  define LAPACKgeev_  zgeev_
#  define LAPACKsyev_  zheev_
#  define LAPACKsyevx_ zheevx_
#  define LAPACKsygv_  zhegv_
#  define LAPACKsygvx_ zhegvx_
#  define LAPACKpttrs_ zpttrs_
/* LAPACKstebz_ does not exist for complex. */
#  define LAPACKsteqr_ zsteqr_
#  define LAPACKhseqr_ zhseqr_
#  define LAPACKgges_  zgges_
#  define LAPACKtrsen_ ztrsen_
#  define LAPACKormqr_ zormqr_
#  define LAPACKhgeqz_ zhgeqz_
#  define LAPACKtrtrs_ ztrtrs_
# else
/* Complex quad precision with no character string arguments */
#  define LAPACKgeqrf_ wgeqrf_
#  define LAPACKungqr_ wungqr_
#  define LAPACKgetrf_ wgetrf_
#  define LAPACKgetri_ wgetri_
/* #  define BLASdot_     wdotc_ */
/* #  define BLASdotu_    wdotu_ */
#  define BLASnrm2_    qwnrm2_
#  define BLASscal_    wscal_
#  define BLAScopy_    wcopy_
#  define BLASswap_    wswap_
#  define BLASaxpy_    waxpy_
#  define BLASasum_    qwasum_
#  define LAPACKpttrf_ wpttrf_
#  define LAPACKstein_ wstein_
#  define LAPACKgesv_  wgesv_
#  define LAPACKgelss_ wgelss_
#  define LAPACKgerfs_ wgerfs_
#  define LAPACKtgsen_ wtgsen_
/* Complex quad precision with character string arguments */
#  define LAPACKpotrf_ wpotrf_
#  define LAPACKpotrs_ wpotrs_
#  define LAPACKsytrf_ wsytrf_
#  define LAPACKsytrs_ wsytrs_
#  define BLASgemv_    wgemv_
#  define LAPACKgetrs_ wgetrs_
#  define BLAStrmv_    wtrmv_
#  define BLASgemm_    wgemm_
#  define BLAStrsm_    wtrsm_
#  define LAPACKgesvd_ wgesvd_
#  define LAPACKgeev_  wgeev_
#  define LAPACKsyev_  wheev_
#  define LAPACKsyevx_ wheevx_
#  define LAPACKsygv_  whegv_
#  define LAPACKsygvx_ whegvx_
#  define LAPACKpttrs_ wpttrs_
/* LAPACKstebz_ does not exist for complex. */
#  define LAPACKsteqr_ wsteqr_
#  define LAPACKhseqr_ whseqr_
#  define LAPACKgges_  wgges_
#  define LAPACKtrsen_ wtrsen_
# endif
#endif

#endif
