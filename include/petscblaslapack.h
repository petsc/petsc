/* $Id: plapack.h,v 1.18 1996/04/04 22:05:54 bsmith Exp bsmith $ */
/*
   This file provides some name space protection from LAPACK and BLAS and
allows the appropriate single or double precision version to be used.
This file also deals with different Fortran 77 naming conventions on machines.

   Another problem is charactor strings are represented differently on 
on some machines in C and Fortran 77. This problem comes up on the 
Cray T3D.  Yet another reason to hate ...

*/
#if !defined(_PLAPACK_H)
#define _PLAPACK_H

#include "petsc.h"

#if defined(PARCH_t3d)
#include "fortran.h"
#endif

#if !defined(PETSC_COMPLEX)

/*
    These are real case with no character string arguments
*/
#if defined(PARCH_t3d)
#define LAgeqrf_ SGEQRF
#define LAgetrf_ SGETRF
#define LAgetf2_ SGETRF
#define BLdot_   SDOT
#define BLnrm2_  SNRM2
#define BLscal_  SSCAL
#define BLcopy_  SCOPY
#define BLswap_  SSWAP
#define BLaxpy_  SAXPY
#define BLasum_  SASUM
#elif defined(HAVE_FORTRAN_CAPS)
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
#elif !defined(HAVE_FORTRAN_UNDERSCORE)
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
#else
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
#endif

/*
   Real with character string arguments.
*/
#if defined(PARCH_t3d)
#define LAormqr_(a,b,c,d,e,f,g,h,i,j,k,l,m)  SORMQR(_cptofcd((a),1),\
             _cptofcd((b),1),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m))
#define LAtrtrs_(a,b,c,d,e,f,g,h,i,j) STRTRS(_cptofcd((a),1),_cptofcd((b),1),\
                             _cptofcd((c),1),(d),(e),(f),(g),(h),(i),(j))
#define LApotrf_(a,b,c,d,e) SPOTRF(_cptofcd((a),1),(b),(c),(d),(e))
#define LApotrs_(a,b,c,d,e,f,g,h) SPOTRS(_cptofcd((a),1),(b),(c),(d),(e),\
                                         (f),(g),(h))
#define LAgemv_(a,b,c,d,e,f,g,h,i,j,k) SGEMV(_cptofcd((a),1),(b),(c),(d),(e),\
                                        (f),(g),(h),(i),(j),(k))
#define LAgetrs_(a,b,c,d,e,f,g,h,i) SGETRS(_cptofcd((a),1),(b),(c),(d),(e),\
                                        (f),(g),(h),(i))
#define LAgetrs_(a,b,c,d,e,f,g,h,i) SGETRS(_cptofcd((a),1),(b),(c),(d),(e),\
                                        (f),(g),(h),(i))
#define BLgemm_(a,b,c,d,e,f,g,h,i,j,k,l,m) SGEMM(_cptofcd((a),1), \
                                            _cptofcd((a),1),(c),(d),(e),\
                                        (f),(g),(h),(i),(j),(k),(l),(m))
#define LAgesvd_(a,b,c,d,e,f,g,h,i,j,k,l,m) SGESVD(_cptofcd((a),1), \
                                            _cptofcd((a),1),(c),(d),(e),\
                                        (f),(g),(h),(i),(j),(k),(l),(m))
#define LAtrmv_  STRMV
#define LAtrsl_  STRSL
#elif defined(HAVE_FORTRAN_CAPS)
#define LAormqr_ DORMQR
#define LAtrtrs_ DTRTRS
#define LApotrf_ DPOTRF
#define LApotrs_ DPOTRS
#define LAgemv_  DGEMV
#define LAgetrs_ DGETRS
#define LAtrmv_  DTRMV
#define LAtrsl_  DTRSL
#define LAgesvd_ DGESVD
#define BLgemm_  DGEMM
#elif !defined(HAVE_FORTRAN_UNDERSCORE)
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
#else
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
#endif

#else

/*
   Complex with no character string arguments
*/
#if defined(PARCH_t3d)
#define LAgeqrf_ CGEQRF
#define BLdot_   CDOTC
#define BLnrm2_  SCNRM2
#define BLscal_  CSCAL
#define BLcopy_  CCOPY
#define BLswap_  CSWAP
#define BLaxpy_  CAXPY
#define BLasum_  SCASUM
#define LAgetrf_ CGETRF
#define LAgetf2_ CGETRF
#elif defined(HAVE_FORTRAN_CAPS)
#define LAgeqrf_ ZGEQRF
#define BLdot_   ZDOTC
#define BLnrm2_  DZNRM2
#define BLscal_  ZSCAL
#define BLcopy_  ZCOPY
#define BLswap_  ZSWAP
#define BLaxpy_  ZAXPY
#define BLasum_  DZASUM
#elif !defined(HAVE_FORTRAN_UNDERSCORE)
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
#else
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
#endif

/*
    Complex with character string arguments.
  Who the F&%&^ was stupid enough to put character strings
  into low-level computational kernels? It was a mistake!
*/
#if defined(PARCH_t3d)
#define LAtrtrs_(a,b,c,d,e,f,g,h,i,j) CTRTRS(_cptofcd((a),1),_cptofcd((b),1),\
                              _cptofcd((c),1),(d),(e),(f),(g),(h),(i),(j))
#define LApotrf_(a,b,c,d,e)       CPOTRF(_cptofcd((a),1),(b),(c),(d),(e))
#define LApotrs_(a,b,c,d,e,f,g,h) CPOTRS(_cptofcd((a),1),(b),(c),(d),(e),\
                                         (f),(g),(h))
#define LAgemv_(a,b,c,d,e,f,g,h,i,j,k) CGEMV(_cptofcd((a),1),(b),(c),(d),(e),\
                                        (f),(g),(h),(i),(j),(k))
#define LAgetrs_(a,b,c,d,e,f,g,h,i) CGETRS(_cptofcd((a),1),(b),(c),(d),(e),\
                                        (f),(g),(h),(i))
#define BLgemm_(a,b,c,d,e,f,g,h,i,j,k,l,m) SGEMM(_cptofcd((a),1), \
                                            _cptofcd((a),1),(c),(d),(e),\
                                        (f),(g),(h),(i),(j),(k),(l),(m))
#define LAtrmv_  CTRMV
#define LAtrsl_  CTRSL
#elif defined(HAVE_FORTRAN_CAPS)
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
#elif !defined(HAVE_FORTRAN_UNDERSCORE)
#define LAtrtrs_ ztrtrs
#define LApotrf_ zpotrf
#define LApotrs_ zpotrs
#define LAgemv_  zgemv
#define LAgetrs_ zgetrs
#define LAtrmv_  ztrmv
#define LAtrsl_  ztrsl
#define BLgemm_  zgemm
#else
#define LAtrtrs_ ztrtrs_
#define LApotrf_ zpotrf_
#define LApotrs_ zpotrs_
#define LAgemv_  zgemv_
#define LAgetrs_ zgetrs_
#define LAtrmv_  ztrmv_
#define LAtrsl_  ztrsl_
#define BLgemm_  zgemm_
#endif

#endif

#if defined(__cplusplus)
extern "C" {
#endif

/* note that BLdot cannot be used with COMPLEX because it cannot 
   handle returing a double complex!!
*/
extern double BLdot_(int*,Scalar*,int*,Scalar*,int*);
extern double BLnrm2_(int*,Scalar*,int*),BLasum_(int*,Scalar*,int*);
extern void   BLscal_(int*,Scalar*,Scalar*,int*);
extern void   BLcopy_(int*,Scalar*,int*,Scalar*,int*);
extern void   BLswap_(int*,Scalar*,int*,Scalar*,int*);
extern void   BLaxpy_(int*,Scalar*,Scalar*,int*,Scalar*,int*);
extern void   LAgetrf_(int*,int*,Scalar*,int*,int*,int*);
extern void   LAgetf2_(int*,int*,Scalar*,int*,int*,int*);
extern void   LAgeqrf_(int*,int*,Scalar*,int*,Scalar*,Scalar*,int*,int*);

#if defined(PARCH_t3d)

#if defined(PETSC_COMPLEX)
extern void   CPOTRF(_fcd,int*,Scalar*,int*,int*);
extern void   CGEMV(_fcd,int*,int*,Scalar*,Scalar*,int*,Scalar *,int*,
                        Scalar*,Scalar*,int*);
extern void   CPOTRS(_fcd,int*,int*,Scalar*,int*,Scalar*,int*,int*);
extern void   CGETRS(_fcd,int*,int*,Scalar*,int*,int*,Scalar*,int*,int*);
extern void   CGEMM(_fcd,_fcd,int*,int*,int*,Scalar*,Scalar*,int*,
                      Scalar*,int*,Scalar*,Scalar*,int*);
#else
extern void   SPOTRF(_fcd,int*,Scalar*,int*,int*);
extern void   SGEMV(_fcd,int*,int*,Scalar*,Scalar*,int*,Scalar *,int*,
                        Scalar*,Scalar*,int*);
extern void   SPOTRS(_fcd,int*,int*,Scalar*,int*,Scalar*,int*,int*);
extern void   SGETRS(_fcd,int*,int*,Scalar*,int*,int*,Scalar*,int*,int*);
extern void   SGEMM(_fcd,_fcd,int*,int*,int*,Scalar*,Scalar*,int*,
                      Scalar*,int*,Scalar*,Scalar*,int*);
extern void   SGESVD(_fcd,_fcd,int *,int*, Scalar *,int*,Scalar*,Scalar*,
                      int*,Scalar*,int*,Scalar*,int*,int*);
#endif

#else
extern void   LAormqr_(char*,char*,int*,int*,int*,Scalar*,int*,Scalar*,Scalar*,
                       int*,Scalar*,int*,int*);
extern void   LAtrtrs_(char*,char*,char*,int*,int*,Scalar*,int*,Scalar*,int*,
                       int*);
extern void   LApotrf_(char*,int*,Scalar*,int*,int*);
extern void   LAgemv_(char*,int*,int*,Scalar*,Scalar*,int*,Scalar *,int*,
                       Scalar*,Scalar*,int*);
extern void   LApotrs_(char*,int*,int*,Scalar*,int*,Scalar*,int*,int*);
extern void   LAgetrs_(char*,int*,int*,Scalar*,int*,int*,Scalar*,int*,int*);
extern void   BLgemm_(char *,char*,int*,int*,int*,Scalar*,Scalar*,int*,
                      Scalar*,int*,Scalar*,Scalar*,int*);
extern void   LAgesvd_(char *,char *,int *,int*, Scalar *,int*,Scalar*,Scalar*,
                      int*,Scalar*,int*,Scalar*,int*,int*);
#endif

#if defined(__cplusplus)
}
#endif

#endif



