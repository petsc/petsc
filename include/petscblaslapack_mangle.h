/*
      This file deals with the BLAS/LAPACK naming convention on
      non-Microsoft Windows systems, which may append an underscore, use
      upper or lower case, and/or use a configurable symbol suffix.
*/
#pragma once

/* MANSEC = Sys */

/* macros to mangle BLAS/LAPACK names as needed for linking */

/* token pasting, with an extra level of indirection so that we
   can paste the contents of other preprocessor #definitions */
#define PETSC_PASTE2_(a, b)    a##b
#define PETSC_PASTE2(a, b)     PETSC_PASTE2_(a, b)
#define PETSC_PASTE3_(a, b, c) a##b##c
#define PETSC_PASTE3(a, b, c)  PETSC_PASTE3_(a, b, c)

#if !defined(PETSC_BLASLAPACK_SUFFIX)
  #if defined(PETSC_BLASLAPACK_UNDERSCORE)
    #define PETSC_BLASLAPACK_SUFFIX_ _
  #else
    #define PETSC_BLASLAPACK_SUFFIX_
  #endif
#else
  #if defined(PETSC_BLASLAPACK_UNDERSCORE)
    #define PETSC_BLASLAPACK_SUFFIX_ PETSC_PASTE2(PETSC_BLASLAPACK_SUFFIX, _)
  #else
    #define PETSC_BLASLAPACK_SUFFIX_ PETSC_BLASLAPACK_SUFFIX
  #endif
#endif

/* complex/real and single/double/quad/half precision prefixes: */
#if !defined(PETSC_USE_COMPLEX)
  #if defined(PETSC_BLASLAPACK_CAPS)
    #if defined(PETSC_USE_REAL_SINGLE)
      #define PETSC_BLASLAPACK_PREFIX_        S
      #define PETSC_BLASLAPACK_PREFIX_COMPLEX C
    #elif defined(PETSC_USE_REAL_DOUBLE)
      #define PETSC_BLASLAPACK_PREFIX_        D
      #define PETSC_BLASLAPACK_PREFIX_COMPLEX Z
    #elif defined(PETSC_USE_REAL___FLOAT128)
      #define PETSC_BLASLAPACK_PREFIX_        Q
      #define PETSC_BLASLAPACK_PREFIX_COMPLEX W
    #else
      #define PETSC_BLASLAPACK_PREFIX_        H
      #define PETSC_BLASLAPACK_PREFIX_COMPLEX K
    #endif
  #else
    #if defined(PETSC_USE_REAL_SINGLE)
      #define PETSC_BLASLAPACK_PREFIX_        s
      #define PETSC_BLASLAPACK_PREFIX_COMPLEX c
    #elif defined(PETSC_USE_REAL_DOUBLE)
      #define PETSC_BLASLAPACK_PREFIX_        d
      #define PETSC_BLASLAPACK_PREFIX_COMPLEX z
    #elif defined(PETSC_USE_REAL___FLOAT128)
      #define PETSC_BLASLAPACK_PREFIX_        q
      #define PETSC_BLASLAPACK_PREFIX_COMPLEX w
    #else
      #define PETSC_BLASLAPACK_PREFIX_        h
      #define PETSC_BLASLAPACK_PREFIX_COMPLEX k
    #endif
  #endif
  #define PETSC_BLASLAPACK_RPREFIX_    PETSC_BLASLAPACK_PREFIX_
  #define PETSC_BLASLAPACK_PREFIX_REAL PETSC_BLASLAPACK_PREFIX_
#else
  #if defined(PETSC_BLASLAPACK_CAPS)
    #if defined(PETSC_USE_REAL_SINGLE)
      #define PETSC_BLASLAPACK_PREFIX_     C
      #define PETSC_BLASLAPACK_PREFIX_REAL S
      #define PETSC_BLASLAPACK_RPREFIX_    SC
    #elif defined(PETSC_USE_REAL_DOUBLE)
      #define PETSC_BLASLAPACK_PREFIX_     Z
      #define PETSC_BLASLAPACK_PREFIX_REAL D
      #define PETSC_BLASLAPACK_RPREFIX_    DZ
    #elif defined(PETSC_USE_REAL___FLOAT128)
      #define PETSC_BLASLAPACK_PREFIX_     W
      #define PETSC_BLASLAPACK_PREFIX_REAL Q
      #define PETSC_BLASLAPACK_RPREFIX_    QW
    #else
      #define PETSC_BLASLAPACK_PREFIX_     K
      #define PETSC_BLASLAPACK_PREFIX_REAL H
      #define PETSC_BLASLAPACK_RPREFIX_    HK
    #endif
  #else
    #if defined(PETSC_USE_REAL_SINGLE)
      #define PETSC_BLASLAPACK_PREFIX_     c
      #define PETSC_BLASLAPACK_PREFIX_REAL s
      #define PETSC_BLASLAPACK_RPREFIX_    sc
    #elif defined(PETSC_USE_REAL_DOUBLE)
      #define PETSC_BLASLAPACK_PREFIX_     z
      #define PETSC_BLASLAPACK_PREFIX_REAL d
      #define PETSC_BLASLAPACK_RPREFIX_    dz
    #elif defined(PETSC_USE_REAL___FLOAT128)
      #define PETSC_BLASLAPACK_PREFIX_     w
      #define PETSC_BLASLAPACK_PREFIX_REAL q
      #define PETSC_BLASLAPACK_RPREFIX_    qw
    #else
      #define PETSC_BLASLAPACK_PREFIX_     k
      #define PETSC_BLASLAPACK_PREFIX_REAL h
      #define PETSC_BLASLAPACK_RPREFIX_    hk
    #endif
  #endif
  #define PETSC_BLASLAPACK_PREFIX_COMPLEX PETSC_BLASLAPACK_PREFIX_
#endif

/* define macros PETSCBLAS to mangle BLAS/LAPACK subroutine names, and
   PETSCBLASR for functions returning real values */
#if defined(PETSC_BLASLAPACK_CAPS)
  #define PETSCBLAS(x, X)        PETSC_PASTE3(PETSC_BLASLAPACK_PREFIX_, X, PETSC_BLASLAPACK_SUFFIX_)
  #define PETSCBLASREAL(x, X)    PETSC_PASTE3(PETSC_BLASLAPACK_PREFIX_REAL, X, PETSC_BLASLAPACK_SUFFIX_)
  #define PETSCBLASCOMPLEX(x, X) PETSC_PASTE3(PETSC_BLASLAPACK_PREFIX_COMPLEX, X, PETSC_BLASLAPACK_SUFFIX_)
  #define PETSCBLASR(x, X)       PETSC_PASTE3(PETSC_BLASLAPACK_RPREFIX_, X, PETSC_BLASLAPACK_SUFFIX_)
#else
  #define PETSCBLAS(x, X)        PETSC_PASTE3(PETSC_BLASLAPACK_PREFIX_, x, PETSC_BLASLAPACK_SUFFIX_)
  #define PETSCBLASREAL(x, X)    PETSC_PASTE3(PETSC_BLASLAPACK_PREFIX_REAL, x, PETSC_BLASLAPACK_SUFFIX_)
  #define PETSCBLASCOMPLEX(x, X) PETSC_PASTE3(PETSC_BLASLAPACK_PREFIX_COMPLEX, x, PETSC_BLASLAPACK_SUFFIX_)
  #define PETSCBLASR(x, X)       PETSC_PASTE3(PETSC_BLASLAPACK_RPREFIX_, x, PETSC_BLASLAPACK_SUFFIX_)
#endif

/* definitions of BLAS and LAPACK symbols */

/* Subroutine names that are the same for real/complex data: */
/* no character-string arguments: */
#define LAPACKgeqrf_     PETSCBLAS(geqrf, GEQRF)
#define LAPACKgetrf_     PETSCBLAS(getrf, GETRF)
#define LAPACKgetri_     PETSCBLAS(getri, GETRI)
#define LAPACKREALgetrf_ PETSCBLASREAL(getrf, GETRF)
#define LAPACKREALgetri_ PETSCBLASREAL(getri, GETRI)
#define BLASnrm2_        PETSCBLASR(nrm2, NRM2)
#define BLASscal_        PETSCBLAS(scal, SCAL)
#define BLAScopy_        PETSCBLAS(copy, COPY)
#define BLASswap_        PETSCBLAS(swap, SWAP)
#define BLASaxpy_        PETSCBLAS(axpy, AXPY)
#define BLASasum_        PETSCBLASR(asum, ASUM)
#define LAPACKpttrf_     PETSCBLAS(pttrf, PTTRF) /* factorization of a spd tridiagonal matrix */
#define LAPACKpttrs_     PETSCBLAS(pttrs, PTTRS) /* solve a spd tridiagonal matrix system */
#if !defined(PETSC_MISSING_LAPACK_STEIN)
  #define LAPACKstein_ PETSCBLAS(stein, STEIN) /* eigenvectors of real symm tridiagonal matrix */
#endif
#define LAPACKgesv_ PETSCBLAS(gesv, GESV)
#if !defined(PETSC_MISSING_LAPACK_GELSS)
  #define LAPACKgelss_ PETSCBLAS(gelss, GELSS)
#endif
#if !defined(PETSC_MISSING_LAPACK_GERFS)
  #define LAPACKgerfs_ PETSCBLAS(gerfs, GERFS)
#endif
#if !defined(PETSC_MISSING_LAPACK_TGSEN)
  #define LAPACKtgsen_ PETSCBLAS(tgsen, TGSEN)
#endif
/* character-string arguments: */
#define LAPACKtrtri_ PETSCBLAS(trtri, TRTRI)
#define LAPACKpotrf_ PETSCBLAS(potrf, POTRF)
#define LAPACKpotri_ PETSCBLAS(potri, POTRI)
#define LAPACKpotrs_ PETSCBLAS(potrs, POTRS)
#define LAPACKsytrf_ PETSCBLAS(sytrf, SYTRF)
#define LAPACKsytrs_ PETSCBLAS(sytrs, SYTRS)
#if !defined(PETSC_MISSING_LAPACK_SYTRI)
  #define LAPACKsytri_ PETSCBLAS(sytri, SYTRI)
#endif
#define BLASgemv_     PETSCBLAS(gemv, GEMV)
#define LAPACKgetrs_  PETSCBLAS(getrs, GETRS)
#define BLAStrmv_     PETSCBLAS(trmv, TRMV)
#define BLAStrsv_     PETSCBLAS(trsv, TRSV)
#define BLASgemm_     PETSCBLAS(gemm, GEMM)
#define BLASsymm_     PETSCBLAS(symm, SYMM)
#define BLASsyrk_     PETSCBLAS(syrk, SYRK)
#define BLASsyr2k_    PETSCBLAS(syr2k, SYR2K)
#define BLAStrsm_     PETSCBLAS(trsm, TRSM)
#define BLASREALgemm_ PETSCBLASREAL(gemm, GEMM)
#define LAPACKgesvd_  PETSCBLAS(gesvd, GESVD)
#define LAPACKgeev_   PETSCBLAS(geev, GEEV)
#define LAPACKgels_   PETSCBLAS(gels, GELS)
#if !defined(PETSC_MISSING_LAPACK_STEGR)
  #define LAPACKstegr_ PETSCBLAS(stegr, STEGR) /* eigenvalues and eigenvectors of symm tridiagonal */
#endif
#if !defined(PETSC_MISSING_LAPACK_STEQR)
  #define LAPACKsteqr_     PETSCBLAS(steqr, STEQR) /* eigenvalues and eigenvectors of symm tridiagonal */
  #define LAPACKREALsteqr_ PETSCBLASREAL(steqr, STEQR)
#endif
#if !defined(PETSC_MISSING_LAPACK_STEV)
  #define LAPACKstev_     PETSCBLAS(stev, STEV) /* eigenvalues and eigenvectors of symm tridiagonal */
  #define LAPACKREALstev_ PETSCBLASREAL(stev, STEV)
#endif
#if !defined(PETSC_MISSING_LAPACK_HSEQR)
  #define LAPACKhseqr_ PETSCBLAS(hseqr, HSEQR)
#endif
#if !defined(PETSC_MISSING_LAPACK_GGES)
  #define LAPACKgges_ PETSCBLAS(gges, GGES)
#endif
#if !defined(PETSC_MISSING_LAPACK_TRSEN)
  #define LAPACKtrsen_ PETSCBLAS(trsen, TRSEN)
#endif
#if !defined(PETSC_MISSING_LAPACK_HGEQZ)
  #define LAPACKhgeqz_ PETSCBLAS(hgeqz, HGEQZ)
#endif
#if !defined(PETSC_MISSING_LAPACK_TRTRS)
  #define LAPACKtrtrs_ PETSCBLAS(trtrs, TRTRS)
#endif

/* Subroutine names that differ for real/complex data: */
#if !defined(PETSC_USE_COMPLEX)
  #if !defined(PETSC_MISSING_LAPACK_ORGQR)
    #define LAPACKorgqr_ PETSCBLAS(orgqr, ORGQR)
  #endif
  #if !defined(PETSC_MISSING_LAPACK_ORMQR)
    #define LAPACKormqr_ PETSCBLAS(ormqr, ORMQR)
  #endif
  #define BLASdot_  PETSCBLAS(dot, DOT)
  #define BLASdotu_ PETSCBLAS(dot, DOT)

  #define LAPACKsyev_  PETSCBLAS(syev, SYEV)   /* eigenvalues and eigenvectors of a symm matrix */
  #define LAPACKsyevx_ PETSCBLAS(syevx, SYEVX) /* selected eigenvalues and eigenvectors of a symm matrix */
  #define LAPACKsygv_  PETSCBLAS(sygv, SYGV)
  #define LAPACKsygvx_ PETSCBLAS(sygvx, SYGVX)

  /* stebz does not exist for complex data */
  #if !defined(PETSC_MISSING_LAPACK_STEBZ)
    #define LAPACKstebz_ PETSCBLAS(stebz, STEBZ) /* eigenvalues of symm tridiagonal matrix */
  #endif
  #define LAPACKgerc_ PETSCBLAS(ger, GER)
  #define BLAShemv_   PETSCBLAS(symv, SYMV)
#else
  #define LAPACKhetrf_ PETSCBLAS(hetrf, HETRF)
  #define LAPACKhetrs_ PETSCBLAS(hetrs, HETRS)
  #define LAPACKhetri_ PETSCBLAS(hetri, HETRI)
  #define LAPACKheev_  PETSCBLAS(heev, HEEV)
  #if !defined(PETSC_MISSING_LAPACK_ORGQR)
    #define LAPACKorgqr_ PETSCBLAS(ungqr, UNGQR)
  #endif
  #if !defined(PETSC_MISSING_LAPACK_ORMQR)
    #define LAPACKormqr_ PETSCBLAS(unmqr, UNMQR)
  #endif
/* note: dot and dotu are handled separately for complex data */

  #define LAPACKsyev_  PETSCBLAS(heev, HEEV)   /* eigenvalues and eigenvectors of a symm matrix */
  #define LAPACKsyevx_ PETSCBLAS(heevx, HEEVX) /* selected eigenvalues and eigenvectors of a symm matrix */
  #define LAPACKsygv_  PETSCBLAS(hegv, HEGV)
  #define LAPACKsygvx_ PETSCBLAS(hegvx, HEGVX)

  #define LAPACKgerc_ PETSCBLAS(gerc, GERC)
  #define BLAShemv_   PETSCBLAS(hemv, HEMV)
#endif
