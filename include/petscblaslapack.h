/* $Id: petscblaslapack.h,v 1.51 2001/08/22 18:03:11 balay Exp $ */
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


#if defined(PETSC_BLASLAPACK_MKL64_ONLY)
#define PETSC_MISSING_LAPACK_GESVD
#define PETSC_MISSING_LAPACK_GEEV
#define PETSC_MISSING_LAPACK_GETRF
#define PETSC_MISSING_LAPACK_POTRF
#define PETSC_MISSING_LAPACK_GETRS
#define PETSC_MISSING_LAPACK_POTRS
#elif defined(PETSC_BLASLAPACK_MKL_ONLY)
#define PETSC_MISSING_LAPACK_GESVD
#define PETSC_MISSING_LAPACK_GEEV
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
#define DTRMV    STRMV
#define DTRSL    STRSL
#endif

#if defined(PETSC_USE_SINGLE)

#if defined(PETSC_HAVE_FORTRAN_UNDERSCORE) || defined(PETSC_BLASLAPACK_F2C)
#define LAgeqrf_ sgeqrf_
#define LAgetrf_ sgetrf_
#define LAgetf2_ sgetf2_
#define BLdot_   sdot_
#define BLnrm2_  snrm2_
#define BLscal_  sscal_
#define BLcopy_  scopy_
#define BLswap_  sswap_
#define BLaxpy_  saxpy_
#define BLasum_  sasum_
#elif defined(PETSC_HAVE_FORTRAN_CAPS)
#define LAgeqrf_ SGEQRF
#define LAgetrf_ SGETRF
#define LAgetf2_ SGETF2
#define BLdot_   SDOT
#define BLnrm2_  SNRM2
#define BLscal_  SSCAL
#define BLcopy_  SCOPY
#define BLswap_  SSWAP
#define BLaxpy_  SAXPY
#define BLasum_  SASUM
#else
#define LAgeqrf_ sgeqrf
#define LAgetrf_ sgetrf
#define LAgetf2_ sgetf2
#define BLdot_   sdot
#define BLnrm2_  snrm2
#define BLscal_  sscal
#define BLcopy_  scopy
#define BLswap_  sswap
#define BLaxpy_  saxpy
#define BLasum_  sasum
#endif

#else

#if defined(PETSC_HAVE_FORTRAN_UNDERSCORE) || defined(PETSC_BLASLAPACK_F2C)
#define LAgeqrf_ dgeqrf_
#define LAgetrf_ dgetrf_
#define LAgetf2_ dgetf2_
#define BLdot_   ddot_
#define BLnrm2_  dnrm2_
#define BLscal_  dscal_
#define BLcopy_  dcopy_
#define BLswap_  dswap_
#define BLaxpy_  daxpy_
#define BLasum_  dasum_
#elif defined(PETSC_HAVE_FORTRAN_CAPS)
#define LAgeqrf_ DGEQRF
#define LAgetrf_ DGETRF
#define LAgetf2_ DGETF2
#define BLdot_   DDOT
#define BLnrm2_  DNRM2
#define BLscal_  DSCAL
#define BLcopy_  DCOPY
#define BLswap_  DSWAP
#define BLaxpy_  DAXPY
#define BLasum_  DASUM
#else
#define LAgeqrf_ dgeqrf
#define LAgetrf_ dgetrf
#define LAgetf2_ dgetf2
#define BLdot_   ddot
#define BLnrm2_  dnrm2
#define BLscal_  dscal
#define BLcopy_  dcopy
#define BLswap_  dswap
#define BLaxpy_  daxpy
#define BLasum_  dasum
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
#define LAormqr_(a,b,c,d,e,f,g,h,i,j,k,l,m)   DORMQR(_cptofcd((a),1),_cptofcd((b),1),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m))
#define LAtrtrs_(a,b,c,d,e,f,g,h,i,j)         DTRTRS(_cptofcd((a),1),_cptofcd((b),1),_cptofcd((c),1),(d),(e),(f),(g),(h),(i),(j))
#define LApotrf_(a,b,c,d,e)                   DPOTRF(_cptofcd((a),1),(b),(c),(d),(e))
#define LApotrs_(a,b,c,d,e,f,g,h)             DPOTRS(_cptofcd((a),1),(b),(c),(d),(e),(f),(g),(h))
#define LAgemv_(a,b,c,d,e,f,g,h,i,j,k)        DGEMV(_cptofcd((a),1),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k))
#define LAgetrs_(a,b,c,d,e,f,g,h,i)           DGETRS(_cptofcd((a),1),(b),(c),(d),(e),(f),(g),(h),(i))
#define LAgetrs_(a,b,c,d,e,f,g,h,i)           DGETRS(_cptofcd((a),1),(b),(c),(d),(e),(f),(g),(h),(i))
#define BLgemm_(a,b,c,d,e,f,g,h,i,j,k,l,m)    DGEMM(_cptofcd((a),1), _cptofcd((a),1),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m))
#define LAgesvd_(a,b,c,d,e,f,g,h,i,j,k,l,m,n) DGESVD(_cptofcd((a),1),_cptofcd((a),1),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n))
#define LAgeev_(a,b,c,d,e,f,g,h,i,j,k,l,m,n)  DGEEV(_cptofcd((a),1),_cptofcd((a),1),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n))
#define LAtrmv_  DTRMV
#define LAtrsl_  DTRSL
#define LAgetrf_ DGETRF

#elif defined(PETSC_USE_SINGLE)

#if defined(PETSC_HAVE_FORTRAN_UNDERSCORE) || defined(PETSC_BLASLAPACK_F2C)
#define LAormqr_ sormqr_
#define LAtrtrs_ strtrs_
#define LApotrf_ spotrf_
#define LApotrs_ spotrs_
#define LAgemv_  sgemv_
#define LAgetrs_ sgetrs_
#define LAtrmv_  strmv_
#define LAtrsl_  strsl_
#define BLgemm_  sgemm_
#define LAgesvd_ sgesvd_
#define LAgeev_  sgeev_
#elif defined(PETSC_HAVE_FORTRAN_CAPS)
#define LAormqr_ SORMQR
#define LAtrtrs_ STRTRS
#define LApotrf_ SPOTRF
#define LApotrs_ SPOTRS
#define LAgemv_  SGEMV
#define LAgetrs_ SGETRS
#define LAtrmv_  STRMV
#define LAtrsl_  STRSL
#define LAgesvd_ SGESVD
#define LAgeev_  SGEEV
#define BLgemm_  SGEMM
#else
#define LAormqr_ sormqr
#define LAtrtrs_ strtrs
#define LApotrf_ spotrf
#define LApotrs_ spotrs
#define LAgemv_  sgemv
#define LAgetrs_ sgetrs
#define LAtrmv_  strmv
#define LAtrsl_  strsl
#define BLgemm_  sgemm
#define LAgesvd_ sgesvd
#define LAgeev_  sgeev
#endif

#else /* PETSC_USE_SINGLE */

#if defined(PETSC_HAVE_FORTRAN_UNDERSCORE) || defined(PETSC_BLASLAPACK_F2C)
#define LAormqr_ dormqr_
#define LAtrtrs_ dtrtrs_
#define LApotrf_ dpotrf_
#define LApotrs_ dpotrs_
#define LAgemv_  dgemv_
#define LAgetrs_ dgetrs_
#define LAtrmv_  dtrmv_
#define LAtrsl_  dtrsl_
#define BLgemm_  dgemm_
#define LAgesvd_ dgesvd_
#define LAgeev_  dgeev_
#elif defined(PETSC_HAVE_FORTRAN_CAPS)
#define LAormqr_ DORMQR
#define LAtrtrs_ DTRTRS
#define LApotrf_ DPOTRF
#define LApotrs_ DPOTRS
#define LAgemv_  DGEMV
#define LAgetrs_ DGETRS
#define LAtrmv_  DTRMV
#define LAtrsl_  DTRSL
#define LAgesvd_ DGESVD
#define LAgeev_  DGEEV
#define BLgemm_  DGEMM
#else
#define LAormqr_ dormqr
#define LAtrtrs_ dtrtrs
#define LApotrf_ dpotrf
#define LApotrs_ dpotrs
#define LAgemv_  dgemv
#define LAgetrs_ dgetrs
#define LAtrmv_  dtrmv
#define LAtrsl_  dtrsl
#define BLgemm_  dgemm
#define LAgesvd_ dgesvd
#define LAgeev_  dgeev
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
#endif

#if defined(PETSC_HAVE_FORTRAN_UNDERSCORE) || defined(PETSC_BLASLAPACK_F2C)
#define LAgeqrf_ zgeqrf_
#define LAgetrf_ zgetrf_
#define LAgetf2_ zgetf2_
#define BLdot_   zdotc_
#define BLnrm2_  dznrm2_
#define BLscal_  zscal_
#define BLcopy_  zcopy_
#define BLswap_  zswap_
#define BLaxpy_  zaxpy_
#define BLasum_  dzasum_
#elif defined(PETSC_HAVE_FORTRAN_CAPS)
#define LAgeqrf_ ZGEQRF
#define BLdot_   ZDOTC
#define BLnrm2_  DZNRM2
#define BLscal_  ZSCAL
#define BLcopy_  ZCOPY
#define BLswap_  ZSWAP
#define BLaxpy_  ZAXPY
#define BLasum_  DZASUM
#define LAgetrf_ ZGETRF
#else
#define LAgeqrf_ zgeqrf
#define LAgetrf_ zgetrf
#define LAgetf2_ zgetf2
#define BLdot_   zdotc
#define BLnrm2_  dznrm2
#define BLscal_  zscal
#define BLcopy_  zcopy
#define BLswap_  zswap
#define BLaxpy_  zaxpy
#define BLasum_  dzasum
#endif

#if defined(PETSC_USES_CPTOFCD)
#define LAtrtrs_(a,b,c,d,e,f,g,h,i,j)           ZTRTRS(_cptofcd((a),1),_cptofcd((b),1),_cptofcd((c),1),(d),(e),(f),(g),(h),(i),(j))
#define LApotrf_(a,b,c,d,e)                     ZPOTRF(_cptofcd((a),1),(b),(c),(d),(e))
#define LApotrs_(a,b,c,d,e,f,g,h)               ZPOTRS(_cptofcd((a),1),(b),(c),(d),(e),(f),(g),(h))
#define LAgemv_(a,b,c,d,e,f,g,h,i,j,k)          ZGEMV(_cptofcd((a),1),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k))
#define LAgetrs_(a,b,c,d,e,f,g,h,i)             ZGETRS(_cptofcd((a),1),(b),(c),(d),(e),(f),(g),(h),(i))
#define BLgemm_(a,b,c,d,e,f,g,h,i,j,k,l,m)      ZGEMM(_cptofcd((a),1),_cptofcd((a),1),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m))
#define LAgesvd_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,p) ZGESVD(_cptofcd((a),1),_cptofcd((a),1),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(p))
#define LAgeev_(a,b,c,d,e,f,g,h,i,j,k,l,m,n)    ZGEEV(_cptofcd((a),1),_cptofcd((a),1),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n))
#define LAtrmv_  ZTRMV
#define LAtrsl_  ZTRSL
#elif defined(PETSC_HAVE_FORTRAN_UNDERSCORE) || defined(PETSC_BLASLAPACK_F2C)
#define LAtrtrs_ ztrtrs_
#define LApotrf_ zpotrf_
#define LApotrs_ zpotrs_
#define LAgemv_  zgemv_
#define LAgetrs_ zgetrs_
#define LAtrmv_  ztrmv_
#define LAtrsl_  ztrsl_
#define BLgemm_  zgemm_
#define LAgesvd_ zgesvd_
#define LAgeev_  zgeev_
#elif defined(PETSC_HAVE_FORTRAN_CAPS)
#define LAtrtrs_ ZTRTRS
#define LApotrf_ ZPOTRF
#define LApotrs_ ZPOTRS
#define LAgemv_  ZGEMV
#define LAgetrf_ ZGETRF
#define LAgetf2_ ZGETF2
#define LAgetrs_ ZGETRS
#define LAtrmv_  ZTRMV
#define LAtrsl_  ZTRSL
#define BLgemm_  ZGEMM
#define LAgesvd_ ZGESVD
#define LAgeev_  ZGEEV
#else
#define LAtrtrs_ ztrtrs
#define LApotrf_ zpotrf
#define LApotrs_ zpotrs
#define LAgemv_  zgemv
#define LAgetrs_ zgetrs
#define LAtrmv_  ztrmv
#define LAtrsl_  ztrsl
#define BLgemm_  zgemm
#define LAgesvd_ zgesvd
#define LAgeev_  zgeev
#endif

#endif

EXTERN_C_BEGIN

/* 
   BLdot cannot be used with COMPLEX because it cannot 
   handle returing a double complex to C++.
*/
EXTERN PetscReal BLdot_(int*,PetscScalar*,int*,PetscScalar*,int*);
EXTERN PetscReal BLnrm2_(int*,PetscScalar*,int*);
EXTERN PetscReal BLasum_(int*,PetscScalar*,int*);
EXTERN void      BLscal_(int*,PetscScalar*,PetscScalar*,int*);
EXTERN void      BLcopy_(int*,PetscScalar*,int*,PetscScalar*,int*);
EXTERN void      BLswap_(int*,PetscScalar*,int*,PetscScalar*,int*);
EXTERN void      BLaxpy_(int*,PetscScalar*,PetscScalar*,int*,PetscScalar*,int*);
EXTERN void      LAgetrf_(int*,int*,PetscScalar*,int*,int*,int*);
EXTERN void      LAgetf2_(int*,int*,PetscScalar*,int*,int*,int*);
EXTERN void      LAgeqrf_(int*,int*,PetscScalar*,int*,PetscScalar*,PetscScalar*,int*,int*);

#if defined(PETSC_USES_CPTOFCD)

#if defined(PETSC_USE_COMPLEX)
EXTERN void   ZORMQR(_fcd,_fcd,int*,int*,int*,PetscScalar*,int*,PetscScalar*,PetscScalar*,int*,PetscScalar*,int*,int*);
EXTERN void   ZTRTRS(_fcd,_fcd,_fcd,int*,int*,PetscScalar*,int*,PetscScalar*,int*,int*);
EXTERN void   ZPOTRF(_fcd,int*,PetscScalar*,int*,int*);
EXTERN void   ZGEMV(_fcd,int*,int*,PetscScalar*,PetscScalar*,int*,PetscScalar *,int*,PetscScalar*,PetscScalar*,int*);
EXTERN void   ZPOTRS(_fcd,int*,int*,PetscScalar*,int*,PetscScalar*,int*,int*);
EXTERN void   ZGETRS(_fcd,int*,int*,PetscScalar*,int*,int*,PetscScalar*,int*,int*);
EXTERN void   ZGEMM(_fcd,_fcd,int*,int*,int*,PetscScalar*,PetscScalar*,int*,PetscScalar*,int*,PetscScalar*,PetscScalar*,int*);
EXTERN void   ZGESVD(_fcd,_fcd,int *,int*,PetscScalar *,int*,PetscReal*,PetscScalar*,int*,PetscScalar*,int*,PetscScalar*,int*,PetscReal*,int*);
EXTERN void   ZGEEV(_fcd,_fcd,int *,PetscScalar *,int*,PetscScalar*,PetscScalar*,int*,PetscScalar*,int*,PetscScalar*,int*,PetscReal*,int*);
#else
EXTERN void   DORMQR(_fcd,_fcd,int*,int*,int*,PetscScalar*,int*,PetscScalar*,PetscScalar*,int*,PetscScalar*,int*,int*);
EXTERN void   DTRTRS(_fcd,_fcd,_fcd,int*,int*,PetscScalar*,int*,PetscScalar*,int*,int*);
EXTERN void   DPOTRF(_fcd,int*,PetscScalar*,int*,int*);
EXTERN void   DGEMV(_fcd,int*,int*,PetscScalar*,PetscScalar*,int*,PetscScalar *,int*,PetscScalar*,PetscScalar*,int*);
EXTERN void   DPOTRS(_fcd,int*,int*,PetscScalar*,int*,PetscScalar*,int*,int*);
EXTERN void   DGETRS(_fcd,int*,int*,PetscScalar*,int*,int*,PetscScalar*,int*,int*);
EXTERN void   DGEMM(_fcd,_fcd,int*,int*,int*,PetscScalar*,PetscScalar*,int*,PetscScalar*,int*,PetscScalar*,PetscScalar*,int*);
EXTERN void   DGESVD(_fcd,_fcd,int *,int*,PetscScalar *,int*,PetscScalar*,PetscScalar*,int*,PetscScalar*,int*,PetscScalar*,int*,int*);
EXTERN void   DGEEV(_fcd,_fcd,int *,PetscScalar *,int*,PetscScalar*,PetscScalar*,PetscScalar*,int*,PetscScalar*,int*,PetscScalar*,int*,int*);
#endif

#else
EXTERN void   LAormqr_(char*,char*,int*,int*,int*,PetscScalar*,int*,PetscScalar*,PetscScalar*,int*,PetscScalar*,int*,int*);
EXTERN void   LAtrtrs_(char*,char*,char*,int*,int*,PetscScalar*,int*,PetscScalar*,int*,int*);
EXTERN void   LApotrf_(char*,int*,PetscScalar*,int*,int*);
EXTERN void   LAgemv_(char*,int*,int*,PetscScalar*,PetscScalar*,int*,PetscScalar *,int*,PetscScalar*,PetscScalar*,int*);
EXTERN void   LApotrs_(char*,int*,int*,PetscScalar*,int*,PetscScalar*,int*,int*);
EXTERN void   LAgetrs_(char*,int*,int*,PetscScalar*,int*,int*,PetscScalar*,int*,int*);
EXTERN void   BLgemm_(char *,char*,int*,int*,int*,PetscScalar*,PetscScalar*,int*,PetscScalar*,int*,PetscScalar*,PetscScalar*,int*);

/* ESSL uses a different calling sequence for dgeev(), zgeev() than LAPACK; */
#if defined(PETSC_HAVE_ESSL) && defined(PETSC_USE_COMPLEX)
EXTERN void   LAgeev_(int*,PetscScalar*,int*,PetscScalar*,PetscScalar*,int*,int*,int*,PetscReal*,int*);
EXTERN void   LAgesvd_(char *,char *,int *,int*,PetscScalar *,int*,PetscReal*,PetscScalar*,int*,PetscScalar*,int*,PetscScalar*,int*,PetscReal*,int*);
#elif defined(PETSC_HAVE_ESSL)
EXTERN void   LAgeev_(int*,PetscScalar*,int*,PetscScalar*,PetscScalar*,int*,int*,int*,PetscReal*,int*);
EXTERN void   LAgesvd_(char *,char *,int *,int*,PetscScalar *,int*,PetscReal*,PetscScalar*,int*,PetscScalar*,int*,PetscScalar*,int*,int*);
#elif !defined(PETSC_USE_COMPLEX)
EXTERN void   LAgeev_(char *,char *,int *,PetscScalar *,int*,PetscReal*,PetscReal*,PetscScalar*,int*,PetscScalar*,int*,PetscScalar*,int*,int*);
EXTERN void   LAgesvd_(char *,char *,int *,int*,PetscScalar *,int*,PetscReal*,PetscScalar*,int*,PetscScalar*,int*,PetscScalar*,int*,int*);
#else
EXTERN void   LAgeev_(char *,char *,int *,PetscScalar *,int*,PetscScalar*,PetscScalar*,int*,PetscScalar*,int*,PetscScalar*,int*,PetscReal*,int*);
EXTERN void   LAgesvd_(char *,char *,int *,int*,PetscScalar *,int*,PetscReal*,PetscScalar*,int*,PetscScalar*,int*,PetscScalar*,int*,PetscReal*,int*);
#endif
#endif

EXTERN_C_END

#endif

