/*
  This file provides some name space protection from LAPACK and BLAS and
  allows the appropriate single or double precision version to be used.

  Another problem is character strings are represented differently on 
  on some machines in C and Fortran 77. This problem comes up on the 
  Cray T3D/T3E where cptofcd is used to resolve the problem.

  Note that this assumes that machines which use cptofcd() use 
  the PETSC_HAVE_FORTRAN_CAPS option. This is true on the Cray T3D/T3E.

  Note also that there is no single precision support here.
*/
#if !defined(_BLASLAPACK_CPTOFCD_H)
#define _BLASLAPACK_CPTOFCD_H
#include "petsc.h"

/*
   This include file on the Cray T3D/T3E defines the interface between 
  Fortran and C representations of character strings.
*/
#if defined(PETSC_USES_CPTOFCD)
# include <fortran.h>
#endif

PETSC_EXTERN_CXX_BEGIN
EXTERN_C_BEGIN
#if !defined(PETSC_USE_COMPLEX)
/* Real double precision without character string arguments. */
#define LAPACKgeqrf_ DGEQRF
#define LAPACKgetrf_ DGETRF
#define BLASdot_     DDOT
#define BLASnrm2_    DNRM2
#define BLASscal_    DSCAL
#define BLAScopy_    DCOPY
#define BLASswap_    DSWAP
#define BLASaxpy_    DAXPY
#define BLASasum_    DASUM
/* Real double precision with character string arguments. */
#define LAPACKormqr_(a,b,c,d,e,f,g,h,i,j,k,l,m)   DORMQR(_cptofcd((a),1),_cptofcd((b),1),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m))
EXTERN void  DORMQR(_fcd,_fcd,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#define LAPACKtrtrs_(a,b,c,d,e,f,g,h,i,j)         DTRTRS(_cptofcd((a),1),_cptofcd((b),1),_cptofcd((c),1),(d),(e),(f),(g),(h),(i),(j))
EXTERN void  DTRTRS(_fcd,_fcd,_fcd,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#define LAPACKpotrf_(a,b,c,d,e)                   DPOTRF(_cptofcd((a),1),(b),(c),(d),(e))
EXTERN void  DPOTRF(_fcd,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#define LAPACKpotrs_(a,b,c,d,e,f,g,h)             DPOTRS(_cptofcd((a),1),(b),(c),(d),(e),(f),(g),(h))
EXTERN void  DPOTRS(_fcd,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#define  BLASgemv_(a,b,c,d,e,f,g,h,i,j,k)          DGEMV(_cptofcd((a),1),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k))
EXTERN void DGEMV(_fcd,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar *,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*);
#define LAPACKgetrs_(a,b,c,d,e,f,g,h,i)           DGETRS(_cptofcd((a),1),(b),(c),(d),(e),(f),(g),(h),(i))
EXTERN void  DGETRS(_fcd,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#define  BLASgemm_(a,b,c,d,e,f,g,h,i,j,k,l,m)      DGEMM(_cptofcd((a),1), _cptofcd((b),1),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m))
EXTERN void DGEMM(_fcd,_fcd,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*);
#define LAPACKgesvd_(a,b,c,d,e,f,g,h,i,j,k,l,m,n) DGESVD(_cptofcd((a),1),_cptofcd((b),1),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n))
EXTERN void  DGESVD(_fcd,_fcd,PetscBLASInt *,PetscBLASInt*,PetscScalar *,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#define LAPACKgeev_(a,b,c,d,e,f,g,h,i,j,k,l,m,n)  DGEEV(_cptofcd((a),1),_cptofcd((b),1),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n))
EXTERN void  DGEEV(_fcd,_fcd,PetscBLASInt *,PetscScalar *,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#define LAPACKsygv_(a,b,c,d,e,f,g,h,i,j,k,l)      DSYGV((a),_cptofcd((b),1),_cptofcd((c),1),(d),(e),(f),(g),(h),(i),(j),(k),(l)) 
EXTERN void  DSYGV(PetscBLASInt*,_fcd,_fcd,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#define LAPACKsygvx_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w) DSYGVX((a),_cptofcd((b),1),_cptofcd((c),1),_cptofcd((d),1),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t),(u),(v),(w))
EXTERN void  DSYGVX(PetscBLASInt*,_fcd,_fcd,_fcd,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
#define BLAStrmv_    DTRMV
#else
/* Complex double precision without character string arguments. */
#define LAPACKgeqrf_ ZGEQRF
#define LAPACKgetrf_ ZGETRF
#define BLASdot_     ZDOTC
#define BLASnrm2_    DZNRM2
#define BLASscal_    ZSCAL
#define BLAScopy_    ZCOPY
#define BLASswap_    ZSWAP
#define BLASaxpy_    ZAXPY
#define BLASasum_    DZASUM
/* Complex double precision with character string arguments. */
#define LAPACKtrtrs_(a,b,c,d,e,f,g,h,i,j)           ZTRTRS(_cptofcd((a),1),_cptofcd((b),1),_cptofcd((c),1),(d),(e),(f),(g),(h),(i),(j))
EXTERN void  ZTRTRS(_fcd,_fcd,_fcd,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#define LAPACKpotrf_(a,b,c,d,e)                     ZPOTRF(_cptofcd((a),1),(b),(c),(d),(e))
EXTERN void  ZPOTRF(_fcd,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#define LAPACKpotrs_(a,b,c,d,e,f,g,h)               ZPOTRS(_cptofcd((a),1),(b),(c),(d),(e),(f),(g),(h))
EXTERN void  ZPOTRS(_fcd,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#define   BLASgemv_(a,b,c,d,e,f,g,h,i,j,k)            ZGEMV(_cptofcd((a),1),(b),(c),(d),(e),(f),(g),(h),(i),(j),(k))
EXTERN void  ZGEMV(_fcd,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar *,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*);
#define LAPACKgetrs_(a,b,c,d,e,f,g,h,i)             ZGETRS(_cptofcd((a),1),(b),(c),(d),(e),(f),(g),(h),(i))
EXTERN void  ZGETRS(_fcd,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#define   BLASgemm_(a,b,c,d,e,f,g,h,i,j,k,l,m)        ZGEMM(_cptofcd((a),1),_cptofcd((b),1),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m))
EXTERN void  ZGEMM(_fcd,_fcd,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*);
#define LAPACKgesvd_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,p) ZGESVD(_cptofcd((a),1),_cptofcd((b),1),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(p))
EXTERN void  ZGESVD(_fcd,_fcd,PetscBLASInt *,PetscBLASInt*,PetscScalar *,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*);
#define LAPACKgeev_(a,b,c,d,e,f,g,h,i,j,k,l,m,n)    ZGEEV(_cptofcd((a),1),_cptofcd((b),1),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n))
EXTERN void  ZGEEV(_fcd,_fcd,PetscBLASInt *,PetscScalar *,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*);
#define LAPACKsygv_(a,b,c,d,e,f,g,h,i,j,k,l)        ZSYGV((a),_cptofcd((b),1),_cptofcd((c),1),(d),(e),(f),(g),(h),(i),(j),(k),(l))
EXTERN void  ZSYGV(PetscBLASInt*,_fcd,_fcd,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#define LAPACKsygvx_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w)    ZSYGVX((a),_cptofcd((b),1),_cptofcd((c),1),_cptofcd((d),1),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t),(u),(v),(w))
EXTERN void  ZSYGVX(PetscBLASInt*,_fcd,_fcd,_fcd,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
#define BLAStrmv_    ZTRMV
#endif

EXTERN void      LAPACKgetrf_(PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
EXTERN void      LAPACKgeqrf_(PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);

EXTERN PetscReal BLASdot_(PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*);
EXTERN PetscReal BLASnrm2_(PetscBLASInt*,PetscScalar*,PetscBLASInt*);
EXTERN void      BLASscal_(PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*);
EXTERN void      BLAScopy_(PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*);
EXTERN void      BLASswap_(PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*);
EXTERN void      BLASaxpy_(PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*);
EXTERN PetscReal BLASasum_(PetscBLASInt*,PetscScalar*,PetscBLASInt*);

EXTERN_C_END
PETSC_EXTERN_CXX_END
#endif
