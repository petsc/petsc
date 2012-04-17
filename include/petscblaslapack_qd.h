/*
  This is for the mpack multi-precision version of BLAS and LAPACK using the qd multiprecision package.
*/
#if !defined(_BLASLAPACK_QD_H)
#define _BLASLAPACK_QD_H

#if !defined(PETSC_USE_COMPLEX)
/* Real with no character string arguments */
#  define LAPACKgeqrf_ sgeqrf_
#  define LAPACKungqr_ sorgqr_
#  define LAPACKgetrf_ sgetrf_
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
/* Real with character string arguments. */
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
#else
/* Complex with no character string arguments */
#  define LAPACKgeqrf_ cgeqrf_
#  define LAPACKungqr_ cungqr_
#  define LAPACKgetrf_ cgetrf_
#  define BLASdot_     cdotc_
#  define BLASdotu_    cdotu_
#  define BLASnrm2_    scnrm2_
#  define BLASscal_    cscal_
#  define BLAScopy_    ccopy_
#  define BLASswap_    cswap_
#  define BLASaxpy_    caxpy_
#  define BLASasum_    scasum_
#  define LAPACKpttrf_ cpttrf_
#  define LAPACKstein_ cstein_
/* Complex with character string arguments */
#  define LAPACKpotrf_ cpotrf_
#  define LAPACKpotrs_ cpotrs_
#  define BLASgemv_    cgemv_
#  define LAPACKgetrs_ cgetrs_
#  define BLAStrmv_    ctrmv_
#  define BLASgemm_    cgemm_
#  define LAPACKgesvd_ cgesvd_
#  define LAPACKgeev_  cgeev_
#  define LAPACKsyev_  cheev_
#  define LAPACKsyevx_ cheevx_
#  define LAPACKsygv_  chegv_
#  define LAPACKsygvx_ chegvx_
#  define LAPACKpttrs_ cpttrs_ 
#endif

#endif
