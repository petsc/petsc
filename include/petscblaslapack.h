/*
   This file provides some name space protection from LAPACK and BLAS and
allows the appropriate single or double precision version to be used.
This file also deals with different Fortran 77 naming conventions on machines.

   Another problem is character strings are represented differently on 
on some machines in C and Fortran 77. This problem comes up on the 
Cray T3D/T3E.

*/
#if !defined(_BLASLAPACK_H)
#define _BLASLAPACK_H
#include "petsc.h"
PETSC_EXTERN_CXX_BEGIN

#if defined(PETSC_BLASLAPACK_MKL64_ONLY)
#define PETSC_MISSING_LAPACK_GESVD
#define PETSC_MISSING_LAPACK_GEEV
#define PETSC_MISSING_LAPACK_SYGV
#define PETSC_MISSING_LAPACK_SYGVX
#define PETSC_MISSING_LAPACK_GETRF
#define PETSC_MISSING_LAPACK_POTRF
#define PETSC_MISSING_LAPACK_GETRS
#define PETSC_MISSING_LAPACK_POTRS
#elif defined(PETSC_BLASLAPACK_MKL_ONLY)
#define PETSC_MISSING_LAPACK_GESVD
#define PETSC_MISSING_LAPACK_GEEV
#define PETSC_MISSING_LAPACK_SYGV
#define PETSC_MISSING_LAPACK_SYGVX
#elif defined(PETSC_BLASLAPACK_CRAY_ONLY)
#define PETSC_MISSING_LAPACK_GESVD
#elif defined(PETSC_BLASLAPACK_ESSL_ONLY)
#define PETSC_MISSING_LAPACK_GESVD
#define PETSC_MISSING_LAPACK_GETRF
#define PETSC_MISSING_LAPACK_GETRS
#define PETSC_MISSING_LAPACK_POTRF
#define PETSC_MISSING_LAPACK_POTRS
#endif

/*
   This include file on the Cray T3D/T3E defines the interface between 
  Fortran and C representations of character strings.
*/
#if defined(PETSC_USES_CPTOFCD)
#include <fortran.h>
#endif

#if !defined(PETSC_USE_COMPLEX)

/*
    These are real case with no character string arguments
*/

#if defined(PETSC_USES_FORTRAN_SINGLE) 
/*
   For these machines we must call the single precision Fortran version
*/
#define DGEQRF   SGEQRF
#define DGETRF   SGETRF
#define DDOT     SDOT
#define DNRM2    SNRM2
#define DSCAL    SSCAL
#define DCOPY    SCOPY
#define DSWAP    SSWAP
#define DAXPY    SAXPY
#define DASUM    SASUM
#define DSORMQR  SORMQR
#define DTRTRS   STRTRS
#define DPOTRF   SPOTRF
#define DPOTRS   SPOTRS
#define DGEMV    SGEMV
#define DGETRS   SGETRS
#define DGETRS   SGETRS
#define DGEMM    SGEMM
#define DGESVD   SGESVD
#define DGEEV    SGEEV
#define DSYGV    SSYGV 
#define DSYGVX   SSYGVX
#define DTRMV    STRMV
#define DTRSL    STRSL
#endif

#if defined(PETSC_USE_SINGLE)

#if defined(PETSC_HAVE_FORTRAN_UNDERSCORE) || defined(PETSC_BLASLAPACK_F2C)
#define LAPACKgeqrf_ sgeqrf_
#define LAPACKgetrf_ sgetrf_
#define LAPACKgetf2_ sgetf2_
#define BLASdot_     sdot_
#define BLASnrm2_    snrm2_
#define BLASscal_    sscal_
#define BLAScopy_    scopy_
#define BLASswap_    sswap_
#define BLASaxpy_    saxpy_
#define BLASasum_    sasum_
#elif defined(PETSC_HAVE_FORTRAN_CAPS)
#define LAPACKgeqrf_ SGEQRF
#define LAPACKgetrf_ SGETRF
#define LAPACKgetf2_ SGETF2
#define BLASdot_     SDOT
#define BLASnrm2_    SNRM2
#define BLASscal_    SSCAL
#define BLAScopy_    SCOPY
#define BLASswap_    SSWAP
#define BLASaxpy_    SAXPY
#define BLASasum_    SASUM
#else
#define LAPACKgeqrf_ sgeqrf
#define LAPACKgetrf_ sgetrf
#define LAPACKgetf2_ sgetf2
#define BLASdot_     sdot
#define BLASnrm2_    snrm2
#define BLASscal_    sscal
#define BLAScopy_    scopy
#define BLASswap_    sswap
#define BLASaxpy_    saxpy
#define BLASasum_    sasum
#endif

#else

#if defined(PETSC_HAVE_FORTRAN_UNDERSCORE) || defined(PETSC_BLASLAPACK_F2C)
#define LAPACKgeqrf_ dgeqrf_
#define LAPACKgetrf_ dgetrf_
#define LAPACKgetf2_ dgetf2_
#define BLASdot_     ddot_
#define BLASnrm2_    dnrm2_
#define BLASscal_    dscal_
#define BLAScopy_    dcopy_
#define BLASswap_    dswap_
#define BLASaxpy_    daxpy_
#define BLASasum_    dasum_
#elif defined(PETSC_HAVE_FORTRAN_CAPS)
#define LAPACKgeqrf_ DGEQRF
#define LAPACKgetrf_ DGETRF
#define LAPACKgetf2_ DGETF2
#define BLASdot_     DDOT
#define BLASnrm2_    DNRM2
#define BLASscal_    DSCAL
#define BLAScopy_    DCOPY
#define BLASswap_    DSWAP
#define BLASaxpy_    DAXPY
#define BLASasum_    DASUM
#else
#define LAPACKgeqrf_ dgeqrf
#define LAPACKgetrf_ dgetrf
#define LAPACKgetf2_ dgetf2
#define BLASdot_     ddot
#define BLASnrm2_    dnrm2
#define BLASscal_    dscal
#define BLAScopy_    dcopy
#define BLASswap_    dswap
#define BLASaxpy_    daxpy
#define BLASasum_    dasum
#endif

#endif

/*
   Real with character string arguments.
*/
#if defined(PETSC_USES_CPTOFCD)
/*
   Note that this assumes that machines which use cptofcd() use 
  the PETSC_HAVE_FORTRAN_CAPS option. This is true on the Cray T3D/T3E.
*/
#define LAPACKormqr_(a,b,c,d,e,f,g,h,i,j,k,l,m)   DORMQR(_cptofcd((a),1),_cptofcd((b),1),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m))
#define LAPACKtrtrs_(a,b,c,d,e,f,g,h,i,j)         DTRTRS(_cptofcd((a),1),_cptofcd((b),1),_cptofcd((c),1),(d),(e),(f),(g),(h),(i),(j))
#define LAPACKpotrf_(a,b,c,d,e)                   DPOTRF(_cptofcd((a),1),(b),(c),(d),(e))
#define LAPACKpotrs_(a,b,c,d,e,f,g,h)             DPOTRS(_cptofcd((a),1),(b),(c),(d),(e),(f),(g),(h))
#define BLASgemv_(a,b,c,d,e,f,g,h,i,j,k)        DGEMV(_cptofcd((a),1),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k))
#define LAPACKgetrs_(a,b,c,d,e,f,g,h,i)           DGETRS(_cptofcd((a),1),(b),(c),(d),(e),(f),(g),(h),(i))
#define BLASgemm_(a,b,c,d,e,f,g,h,i,j,k,l,m)    DGEMM(_cptofcd((a),1), _cptofcd((a),1),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m))
#define LAPACKgesvd_(a,b,c,d,e,f,g,h,i,j,k,l,m,n) DGESVD(_cptofcd((a),1),_cptofcd((a),1),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n))
#define LAPACKgeev_(a,b,c,d,e,f,g,h,i,j,k,l,m,n)  DGEEV(_cptofcd((a),1),_cptofcd((a),1),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n))
#define LAPACKsygv_(a,b,c,d,e,f,g,h,i,j,k,l)  DSYGV((a),_cptofcd((a),1),_cptofcd((a),1),(d),(e),(f),(g),(h),(i),(j),(k),(l)) 
#define LAPACKsygvx_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w) DSYGVX((a),_cptofcd((a),1),_cptofcd((a),1),_cptofcd((a),1),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t),(u),(v),(w))
#define BLAStrmv_  DTRMV
#define LAPACKtrsl_  DTRSL
#define LAPACKgetrf_ DGETRF

#elif defined(PETSC_USE_SINGLE)

#if defined(PETSC_HAVE_FORTRAN_UNDERSCORE) || defined(PETSC_BLASLAPACK_F2C)
#define LAPACKormqr_ sormqr_
#define LAPACKtrtrs_ strtrs_
#define LAPACKpotrf_ spotrf_
#define LAPACKpotrs_ spotrs_
#define BLASgemv_  sgemv_
#define LAPACKgetrs_ sgetrs_
#define BLAStrmv_  strmv_
#define LAPACKtrsl_  strsl_
#define BLASgemm_  sgemm_
#define LAPACKgesvd_ sgesvd_
#define LAPACKgeev_  sgeev_
#define LAPACKsygv_  ssygv_
#define LAPACKsygvx_  ssygvx_

#elif defined(PETSC_HAVE_FORTRAN_CAPS)
#define LAPACKormqr_ SORMQR
#define LAPACKtrtrs_ STRTRS
#define LAPACKpotrf_ SPOTRF
#define LAPACKpotrs_ SPOTRS
#define BLASgemv_  SGEMV
#define LAPACKgetrs_ SGETRS
#define BLAStrmv_  STRMV
#define LAPACKtrsl_  STRSL
#define LAPACKgesvd_ SGESVD
#define LAPACKgeev_  SGEEV
#define LAPACKsygv_  SSYGV
#define LAPACKsygvx_ SSYGVX
#define BLASgemm_  SGEMM
#else
#define LAPACKormqr_ sormqr
#define LAPACKtrtrs_ strtrs
#define LAPACKpotrf_ spotrf
#define LAPACKpotrs_ spotrs
#define BLASgemv_  sgemv
#define LAPACKgetrs_ sgetrs
#define BLAStrmv_  strmv
#define LAPACKtrsl_  strsl
#define BLASgemm_  sgemm
#define LAPACKgesvd_ sgesvd
#define LAPACKgeev_  sgeev
#define LAPACKsygv_  ssygv_
#define LAPACKsygvx_  ssygvx_
#endif

#else /* PETSC_USE_SINGLE */

#if defined(PETSC_HAVE_FORTRAN_UNDERSCORE) || defined(PETSC_BLASLAPACK_F2C)
#define LAPACKormqr_ dormqr_
#define LAPACKtrtrs_ dtrtrs_
#define LAPACKpotrf_ dpotrf_
#define LAPACKpotrs_ dpotrs_
#define BLASgemv_  dgemv_
#define LAPACKgetrs_ dgetrs_
#define BLAStrmv_  dtrmv_
#define LAPACKtrsl_  dtrsl_
#define BLASgemm_  dgemm_
#define LAPACKgesvd_ dgesvd_
#define LAPACKgeev_  dgeev_
#define LAPACKsygv_  dsygv_
#define LAPACKsygvx_  dsygvx_

#elif defined(PETSC_HAVE_FORTRAN_CAPS)
#define LAPACKormqr_ DORMQR
#define LAPACKtrtrs_ DTRTRS
#define LAPACKpotrf_ DPOTRF
#define LAPACKpotrs_ DPOTRS
#define BLASgemv_  DGEMV
#define LAPACKgetrs_ DGETRS
#define BLAStrmv_  DTRMV
#define LAPACKtrsl_  DTRSL
#define LAPACKgesvd_ DGESVD
#define LAPACKgeev_  DGEEV
#define LAPACKsygv_  DSYGV
#define LAPACKsygvx_  DSYGVX
#define BLASgemm_  DGEMM
#else
#define LAPACKormqr_ dormqr
#define LAPACKtrtrs_ dtrtrs
#define LAPACKpotrf_ dpotrf
#define LAPACKpotrs_ dpotrs
#define BLASgemv_  dgemv
#define LAPACKgetrs_ dgetrs
#define BLAStrmv_  dtrmv
#define LAPACKtrsl_  dtrsl
#define BLASgemm_  dgemm
#define LAPACKgesvd_ dgesvd
#define LAPACKgeev_  dgeev
#define LAPACKsygv_  dsygv
#define LAPACKsygvx_  dsygvx
#endif

#endif /* PETSC_USES_CPTOFCD */

#else
/*
   Complex with no character string arguments
*/
#if defined(PETSC_USES_FORTRAN_SINGLE)
#define ZGEQRF  CGEQRF
#define ZDOTC   CDOTC
#define DZNRM2  SCNRM2
#define ZSCAL   CSCAL
#define ZCOPY   CCOPY
#define ZSWAP   CSWAP
#define ZAXPY   CAXPY
#define DZASUM  SCASUM
#define ZGETRF  CGETRF
#define ZTRTRS  CTRTRS
#define ZPOTRF  CPOTRF
#define ZPOTRS  CPOTRS
#define ZGEMV   CGEMV
#define ZGETRS  CGETRS
#define ZGEMM   SGEMM
#define ZTRMV   CTRMV
#define ZTRSL   CTRSL
#define ZGEEV   CGEEV
#define ZSYGV   CSYGV
#define ZSYGVX  CSYGVX
#endif

#if defined(PETSC_HAVE_FORTRAN_UNDERSCORE) || defined(PETSC_BLASLAPACK_F2C)
#define LAPACKgeqrf_ zgeqrf_
#define LAPACKgetrf_ zgetrf_
#define LAPACKgetf2_ zgetf2_
#define BLASdot_     zdotc_
#define BLASnrm2_    dznrm2_
#define BLASscal_    zscal_
#define BLAScopy_    zcopy_
#define BLASswap_    zswap_
#define BLASaxpy_    zaxpy_
#define BLASasum_    dzasum_
#elif defined(PETSC_HAVE_FORTRAN_CAPS)
#define LAPACKgeqrf_ ZGEQRF
#define LAPACKgetrf_ ZGETRF
#define BLASdot_     ZDOTC
#define BLASnrm2_    DZNRM2
#define BLASscal_    ZSCAL
#define BLAScopy_    ZCOPY
#define BLASswap_    ZSWAP
#define BLASaxpy_    ZAXPY
#define BLASasum_    DZASUM
#else
#define LAPACKgeqrf_ zgeqrf
#define LAPACKgetrf_ zgetrf
#define LAPACKgetf2_ zgetf2
#define BLASdot_     zdotc
#define BLASnrm2_    dznrm2
#define BLASscal_    zscal
#define BLAScopy_    zcopy
#define BLASswap_    zswap
#define BLASaxpy_    zaxpy
#define BLASasum_    dzasum
#endif

#if defined(PETSC_USES_CPTOFCD)
#define LAPACKtrtrs_(a,b,c,d,e,f,g,h,i,j)           ZTRTRS(_cptofcd((a),1),_cptofcd((b),1),_cptofcd((c),1),(d),(e),(f),(g),(h),(i),(j))
#define LAPACKpotrf_(a,b,c,d,e)                     ZPOTRF(_cptofcd((a),1),(b),(c),(d),(e))
#define LAPACKpotrs_(a,b,c,d,e,f,g,h)               ZPOTRS(_cptofcd((a),1),(b),(c),(d),(e),(f),(g),(h))
#define BLASgemv_(a,b,c,d,e,f,g,h,i,j,k)          ZGEMV(_cptofcd((a),1),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k))
#define LAPACKgetrs_(a,b,c,d,e,f,g,h,i)             ZGETRS(_cptofcd((a),1),(b),(c),(d),(e),(f),(g),(h),(i))
#define BLASgemm_(a,b,c,d,e,f,g,h,i,j,k,l,m)      ZGEMM(_cptofcd((a),1),_cptofcd((a),1),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m))
#define LAPACKgesvd_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,p) ZGESVD(_cptofcd((a),1),_cptofcd((a),1),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(p))
#define LAPACKgeev_(a,b,c,d,e,f,g,h,i,j,k,l,m,n)    ZGEEV(_cptofcd((a),1),_cptofcd((a),1),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n))
#define LAPACKsygv_(a,b,c,d,e,f,g,h,i,j,k,l)    ZSYGV((a),_cptofcd((a),1),_cptofcd((a),1),(d),(e),(f),(g),(h),(i),(j),(k),(l))
#define LAPACKsygvx_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w)    ZSYGVX((a),_cptofcd((a),1),_cptofcd((a),1),_cptofcd((a),1),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t),(u),(v),(w))

#define BLAStrmv_    ZTRMV
#define LAPACKtrsl_  ZTRSL
#elif defined(PETSC_HAVE_FORTRAN_UNDERSCORE) || defined(PETSC_BLASLAPACK_F2C)
#define LAPACKtrtrs_ ztrtrs_
#define LAPACKpotrf_ zpotrf_
#define LAPACKpotrs_ zpotrs_
#define BLASgemv_    zgemv_
#define LAPACKgetrs_ zgetrs_
#define BLAStrmv_    ztrmv_
#define LAPACKtrsl_  ztrsl_
#define BLASgemm_    zgemm_
#define LAPACKgesvd_ zgesvd_
#define LAPACKgeev_  zgeev_
#define LAPACKsygv_  zsygv_
#define LAPACKsygvx_  zsygvx_

#elif defined(PETSC_HAVE_FORTRAN_CAPS)
#define LAPACKtrtrs_ ZTRTRS
#define LAPACKpotrf_ ZPOTRF
#define LAPACKpotrs_ ZPOTRS
#define BLASgemv_    ZGEMV
#define LAPACKgetrf_ ZGETRF
#define LAPACKgetf2_ ZGETF2
#define LAPACKgetrs_ ZGETRS
#define BLAStrmv_    ZTRMV
#define LAPACKtrsl_  ZTRSL
#define BLASgemm_    ZGEMM
#define LAPACKgesvd_ ZGESVD
#define LAPACKgeev_  ZGEEV
#define LAPACKsygv_  ZSYGV
#define LAPACKsygvx_  ZSYGVX

#else
#define LAPACKtrtrs_ ztrtrs
#define LAPACKpotrf_ zpotrf
#define LAPACKpotrs_ zpotrs
#define BLASgemv_    zgemv
#define LAPACKgetrs_ zgetrs
#define BLAStrmv_    ztrmv
#define LAPACKtrsl_  ztrsl
#define BLASgemm_    zgemm
#define LAPACKgesvd_ zgesvd
#define LAPACKgeev_  zgeev
#define LAPACKsygv_  zsygv
#define LAPACKsygvx_  zsygvx
#endif

#endif

EXTERN_C_BEGIN

/* 
   BLASdot cannot be used with COMPLEX because it cannot 
   handle returing a double complex to C++.
*/
EXTERN PetscReal BLASdot_(PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*);
EXTERN PetscReal BLASnrm2_(PetscBLASInt*,PetscScalar*,PetscBLASInt*);
EXTERN PetscReal BLASasum_(PetscBLASInt*,PetscScalar*,PetscBLASInt*);
EXTERN void      BLASscal_(PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*);
EXTERN void      BLAScopy_(PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*);
EXTERN void      BLASswap_(PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*);
EXTERN void      BLASaxpy_(PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*);
EXTERN void      LAPACKgetrf_(PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
EXTERN void      LAPACKgetf2_(PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
EXTERN void      LAPACKgeqrf_(PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);

#if defined(PETSC_USES_CPTOFCD)

#if defined(PETSC_USE_COMPLEX)
EXTERN void   ZORMQR(_fcd,_fcd,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN void   ZTRTRS(_fcd,_fcd,_fcd,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN void   ZPOTRF(_fcd,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN void   ZGEMV(_fcd,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar *,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*);
EXTERN void   ZPOTRS(_fcd,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN void   ZGETRS(_fcd,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN void   ZGEMM(_fcd,_fcd,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*);
EXTERN void   ZGESVD(_fcd,_fcd,PetscBLASInt *,PetscBLASInt*,PetscScalar *,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*);
EXTERN void   ZGEEV(_fcd,_fcd,PetscBLASInt *,PetscScalar *,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*);
EXTERN void   ZSYGV(PetscBLASInt*,_fcd,_fcd,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN void   ZSYGVX(PetscBLASInt*,_fcd,_fcd,_fcd,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);

#else
EXTERN void   DORMQR(_fcd,_fcd,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN void   DTRTRS(_fcd,_fcd,_fcd,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN void   DPOTRF(_fcd,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN void   DGEMV(_fcd,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar *,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*);
EXTERN void   DPOTRS(_fcd,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN void   DGETRS(_fcd,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN void   DGEMM(_fcd,_fcd,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*);
EXTERN void   DGESVD(_fcd,_fcd,PetscBLASInt *,PetscBLASInt*,PetscScalar *,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN void   DGEEV(_fcd,_fcd,PetscBLASInt *,PetscScalar *,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN void   DSYGV(PetscBLASInt*,_fcd,_fcd,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN void   DSYGVX(PetscBLASInt*,_fcd,_fcd,_fcd,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
#endif

#else
EXTERN void   LAPACKormqr_(const char*,const char*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN void   LAPACKtrtrs_(const char*,const char*,const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN void   LAPACKpotrf_(const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN void   BLASgemv_(const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar *,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*);
EXTERN void   LAPACKpotrs_(const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN void   LAPACKgetrs_(const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN void   BLASgemm_(const char*,const char*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*);

/* ESSL uses a different calling sequence for dgeev(), zgeev() than LAPACK; */
#if defined(PETSC_HAVE_ESSL) && defined(PETSC_USE_COMPLEX)
EXTERN void   LAPACKgeev_(PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*);
EXTERN void   LAPACKgesvd_(const char*,const char*,PetscBLASInt *,PetscBLASInt*,PetscScalar *,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*);
#elif defined(PETSC_HAVE_ESSL)
EXTERN void   LAPACKgeev_(PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*);
EXTERN void   LAPACKgesvd_(const char*,const char*,PetscBLASInt *,PetscBLASInt*,PetscScalar *,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#elif !defined(PETSC_USE_COMPLEX)
EXTERN void   LAPACKgeev_(const char*,const char*,PetscBLASInt *,PetscScalar *,PetscBLASInt*,PetscReal*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN void   LAPACKgesvd_(const char*,const char*,PetscBLASInt *,PetscBLASInt*,PetscScalar *,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN void   LAPACKsygv_(PetscBLASInt*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN void   LAPACKsygvx_(PetscBLASInt*,const char*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
#else
EXTERN void   LAPACKgeev_(const char*,const char*,PetscBLASInt *,PetscScalar *,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*);
EXTERN void   LAPACKsygv_(PetscBLASInt*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN void   LAPACKsygvx_(PetscBLASInt*,const char*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
EXTERN void   LAPACKgesvd_(const char*,const char*,PetscBLASInt *,PetscBLASInt*,PetscScalar *,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*);
#endif
#endif

EXTERN_C_END
PETSC_EXTERN_CXX_END
#endif
