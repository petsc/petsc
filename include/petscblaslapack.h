/*
   This is to provide some name space protection from Lapack and Blas 
allow the appropriate single or double precision version to be used.
Also deals with different Fortran 77 naming conventions on machines.
*/
#if !defined(_PLAPACK_H)
#define _PLAPACK_H

#include "ptscimpl.h"

#if !defined(PETSC_COMPLEX)
#if defined(cray)
#define BLdot_   SDOT
#define BLnrm2_  SNRM2
#define BLscal_  SSCAL
#define BLcopy_  SCOPY
#define BLswap_  SSWAP
#define BLaxpy_  SAXPY
#define BLasum_  SASUM
#elif defined(FORTRANCAPS)
#define BLdot_   DDOT
#define BLnrm2_  DNRM2
#define BLscal_  DSCAL
#define BLcopy_  DCOPY
#define BLswap_  DSWAP
#define BLaxpy_  DAXPY
#define BLasum_  DASUM
#elif !defined(FORTRANUNDERSCORE)
#define BLdot_   ddot
#define BLnrm2_  dnrm2
#define BLscal_  dscal
#define BLcopy_  dcopy
#define BLswap_  dswap
#define BLaxpy_  daxpy
#define BLasum_  dasum
#else
#define BLdot_   ddot_
#define BLnrm2_  dnrm2_
#define BLscal_  dscal_
#define BLcopy_  dcopy_
#define BLswap_  dswap_
#define BLaxpy_  daxpy_
#define BLasum_  dasum_
#endif

#if defined(cray)
#define LApotrf_ SPOTRF
#define LApotrs_ SPOTRS
#define LAgemv_  SGEMV
#define LAgetrf_ SGETRF
#define LAgetrs_ SGETRS
#define LAgemv_  SGEMV
#define LAtrmv_  STRMV
#define LAtrsl_  STRSL
#define LAger_   SGER
#elif defined(FORTRANCAPS)
#define LApotrf_ DPOTRF
#define LApotrs_ DPOTRS
#define LAgemv_  DGEMV
#define LAgetrf_ DGETRF
#define LAgetrs_ DGETRS
#define LAger_   DGER
#define LAtrmv_  DTRMV
#define LAtrsl_  DTRSL
#elif !defined(FORTRANUNDERSCORE)
#define LApotrf_ dpotrf
#define LApotrs_ dpotrs
#define LAgemv_  dgemv
#define LAgetrf_ dgetrf
#define LAgetrs_ dgetrs
#define LAger_   dger
#define LAtrmv_  dtrmv
#define LAtrsl_  dtrsl
#else
#define LApotrf_ dpotrf_
#define LApotrs_ dpotrs_
#define LAgemv_  dgemv_
#define LAgetrf_ dgetrf_
#define LAgetrs_ dgetrs_
#define LAger_   dger_
#define LAtrmv_  dtrmv_
#define LAtrsl_  dtrsl_
#endif

#else

#if defined(cray)
#define BLdot_   CDOTC
#define BLnrm2_  SCNRM2
#define BLscal_  CSCAL
#define BLcopy_  CCOPY
#define BLswap_  CSWAP
#define BLaxpy_  CAXPY
#define BLasum_  SCASUM
#elif defined(FORTRANCAPS)
#define BLdot_   ZDOTC
#define BLnrm2_  DZNRM2
#define BLscal_  ZSCAL
#define BLcopy_  ZCOPY
#define BLswap_  ZSWAP
#define BLaxpy_  ZAXPY
#define BLasum_  DZASUM
#elif !defined(FORTRANUNDERSCORE)
#define BLdot_   zdotc
#define BLnrm2_  dznrm2
#define BLscal_  zscal
#define BLcopy_  zcopy
#define BLswap_  zswap
#define BLaxpy_  zaxpy
#define BLasum_  dzasum
#else
#define BLdot_   zdotc_
#define BLnrm2_  dznrm2_
#define BLscal_  zscal_
#define BLcopy_  zcopy_
#define BLswap_  zswap_
#define BLaxpy_  zaxpy_
#define BLasum_  dzasum_
#endif

#if defined(cray)
#define LApotrf_ CPOTRF
#define LApotrs_ CPOTRS
#define LAgemv_  CGEMV
#define LAgetrf_ CGETRF
#define LAgetrs_ CGETRS
#define LAgemv_  CGEMV
#define LAtrmv_  CTRMV
#define LAtrsl_  CTRSL
#define LAger_   CGER
#elif defined(FORTRANCAPS)
#define LApotrf_ ZPOTRF
#define LApotrs_ ZPOTRS
#define LAgemv_  ZGEMV
#define LAgetrf_ ZGETRF
#define LAgetrs_ ZGETRS
#define LAger_   ZGER
#define LAtrmv_  ZTRMV
#define LAtrsl_  ZTRSL
#elif !defined(FORTRANUNDERSCORE)
#define LApotrf_ zpotrf
#define LApotrs_ zpotrs
#define LAgemv_  zgemv
#define LAgetrf_ zgetrf
#define LAgetrs_ zgetrs
#define LAger_   zger
#define LAtrmv_  ztrmv
#define LAtrsl_  ztrsl
#else
#define LApotrf_ zpotrf_
#define LApotrs_ zpotrs_
#define LAgemv_  zgemv_
#define LAgetrf_ zgetrf_
#define LAgetrs_ zgetrs_
#define LAger_   zger_
#define LAtrmv_  ztrmv_
#define LAtrsl_  ztrsl_
#endif

#endif

#if defined(__cplusplus)
extern "C" {
#endif

Scalar BLdot_(int*,Scalar*,int*,Scalar*,int*);
double BLnrm2_(int*,Scalar*,int*),BLasum_(int*,Scalar*,int*);
void   BLscal_(int*,Scalar*,Scalar*,int*);
void   BLcopy_(int*,Scalar*,int*,Scalar*,int*);
void   BLswap_(int*,Scalar*,int*,Scalar*,int*);
void   BLaxpy_(int*,Scalar*,Scalar*,int*,Scalar*,int*);

void   LAgetrf_(int*,int*,Scalar*,int*,int*,int*);
void   LApotrf_(char*,int*,Scalar*,int*,int*);
void   LAgemv_(char*,int*,int*,Scalar*,Scalar*,int*,Scalar *,int*,
               Scalar*,Scalar*,int*);
void   LApotrs_(char*,int*,int*,Scalar*,int*,Scalar*,int*,int*);
void   LAgetrs_(char*,int*,int*,Scalar*,int*,int*,Scalar*,int*,int*);

#if defined(__cplusplus)
};
#endif

#endif
