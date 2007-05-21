/*
  This file provides some name space protection from LAPACK and BLAS and
  allows the appropriate single or double precision version to be used.

  This file also deals with STDCALL Fortran 77 naming conventions.  This also
  assumes PETSC_HAVE_FORTRAN_CAPS is also defined, which is the case on certain Windows
  FORTRAN compilers which use STDCALL.

  Another problem is character strings are represented differently on 
  on some machines in C and Fortran 77. This problem comes up on some Windows compilers.
*/
#if !defined(_BLASLAPACK_STDCALL_H)
#define _BLASLAPACK_STDCALL_H
#include "petsc.h"
PETSC_EXTERN_CXX_BEGIN
EXTERN_C_BEGIN

#if !defined(PETSC_USE_COMPLEX)
# if defined(PETSC_USES_FORTRAN_SINGLE) || defined(PETSC_USE_SINGLE)
/* Real single precision without character string arguments. */
#  define LAPACKgeqrf_ SGEQRF
#  define LAPACKgetrf_ SGETRF
#  define BLASdot_     SDOT
#  define BLASnrm2_    SNRM2
#  define BLASscal_    SSCAL
#  define BLAScopy_    SCOPY
#  define BLASswap_    SSWAP
#  define BLASaxpy_    SAXPY
#  define BLASasum_    SASUM
#  define BLAStrmv_    STRMV
#  define LAPACKpttrf_ SPTTRF
#  define LAPACKpttrs_ SPTTRS
#  define LAPACKstein_ SSTEIN
/* Real single precision with character string arguments. */
#  define LAPACKormqr_(a,b,c,d,e,f,g,h,i,j,k,l,m)   SORMQR((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m))
EXTERN void PETSC_STDCALL                           SORMQR(char*,int,char*,int,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#  define LAPACKtrtrs_(a,b,c,d,e,f,g,h,i,j)         STRTRS((a),1,(b),1,(c),1,(d),(e),(f),(g),(h),(i),(j))
EXTERN void PETSC_STDCALL                           STRTRS(char*,int,char*,int,char*,int,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#  define LAPACKpotrf_(a,b,c,d,e)                   SPOTRF((a),1,(b),(c),(d),(e))
EXTERN void PETSC_STDCALL                           SPOTRF(char*,int,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#  define LAPACKpotrs_(a,b,c,d,e,f,g,h)             SPOTRS((a),1,(b),(c),(d),(e),(f),(g),(h))
EXTERN void PETSC_STDCALL                           SPOTRS(char*,int,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#  define BLASgemm_(a,b,c,d,e,f,g,h,i,j,k,l,m)      SGEMM((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m))
EXTERN void PETSC_STDCALL                           SGEMM(char *,int,char*,int,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*);
#  define LAPACKgetrs_(a,b,c,d,e,f,g,h,i)           SGETRS((a),1,(b),(c),(d),(e),(f),(g),(h),(i))
EXTERN void PETSC_STDCALL                           SGETRS(char*,int,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#  define BLASgemv_(a,b,c,d,e,f,g,h,i,j,k)          SGEMV((a),1,(b),(c),(d),(e),(f),(g),(h),(i),(j),(k))
EXTERN void PETSC_STDCALL                           SGEMV(char*,int,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar *,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*);
#  define LAPACKgeev_(a,b,c,d,e,f,g,h,i,j,k,l,m,n)  SGEEV((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n))
EXTERN void PETSC_STDCALL                           SGEEV(char *,int,char *,int,PetscBLASInt*,PetscScalar *,PetscBLASInt*,PetscReal*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#  define LAPACKgesvd_(a,b,c,d,e,f,g,h,i,j,k,l,m,n) SGESVD((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n))
EXTERN void PETSC_STDCALL                           SGESVD(char *,int,char *,int,PetscBLASInt*,PetscBLASInt*,PetscScalar *,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);

#  define LAPACKsyev_(a,b,c,d,e,f,g,h,i)            SSYEV((a),(b),1,(c),1,(d),(e),(f),(g),(h),(i)) 
EXTERN void PETSC_STDCALL                           SSYEV(const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#  define LAPACKsyevx_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t) SSYEVX((a),(b),1,(c),1,(d),1,(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t))
EXTERN void PETSC_STDCALL                           SSYEVX(const char*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);

#  define LAPACKsygv_(a,b,c,d,e,f,g,h,i,j,k,l)      SSYGV((a),(b),1,(c),1,(d),(e),(f),(g),(h),(i),(j),(k),(l)) 
EXTERN void PETSC_STDCALL                           SSYGV(PetscBLASInt*,const char*,int,const char*,int,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#  define LAPACKsygvx_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w) SSYGVX((a),(b),1,(c),1,(d),1,(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t),(u),(v),(w))
EXTERN void PETSC_STDCALL                           SSYGVX(PetscBLASInt*,const char*,int,const char*,int,const char*,int,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);

#  define LAPACKstebz_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r) SSTEBZ((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r))
EXTERN void PETSC_STDCALL                          SSTEBZ(const char*,const char*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*);
# else
/* Real double precision without character string arguments. */
#  define LAPACKgeqrf_ DGEQRF
#  define LAPACKgetrf_ DGETRF
#  define BLASdot_     DDOT
#  define BLASnrm2_    DNRM2
#  define BLASscal_    DSCAL
#  define BLAScopy_    DCOPY
#  define BLASswap_    DSWAP
#  define BLASaxpy_    DAXPY
#  define BLASasum_    DASUM
#  define BLAStrmv_    DTRMV
#  define LAPACKpttrf_ DPTTRF
#  define LAPACKpttrs_ DPTTRS
#  define LAPACKstein_ DSTEIN
/* Real double precision with character string arguments. */
#  define LAPACKormqr_(a,b,c,d,e,f,g,h,i,j,k,l,m)   DORMQR((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m))
EXTERN void PETSC_STDCALL                           DORMQR(char*,int,char*,int,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#  define LAPACKtrtrs_(a,b,c,d,e,f,g,h,i,j)         DTRTRS((a),1,(b),1,(c),1,(d),(e),(f),(g),(h),(i),(j))
EXTERN void PETSC_STDCALL                           DTRTRS(char*,int,char*,int,char*,int,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#  define LAPACKpotrf_(a,b,c,d,e)                   DPOTRF((a),1,(b),(c),(d),(e))
EXTERN void PETSC_STDCALL                           DPOTRF(char*,int,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#  define LAPACKpotrs_(a,b,c,d,e,f,g,h)             DPOTRS((a),1,(b),(c),(d),(e),(f),(g),(h))
EXTERN void PETSC_STDCALL                           DPOTRS(char*,int,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#  define BLASgemv_(a,b,c,d,e,f,g,h,i,j,k)          DGEMV((a),1,(b),(c),(d),(e),(f),(g),(h),(i),(j),(k))
EXTERN void PETSC_STDCALL                           DGEMV(char*,int,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar *,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*);
#  define LAPACKgetrs_(a,b,c,d,e,f,g,h,i)           DGETRS((a),1,(b),(c),(d),(e),(f),(g),(h),(i))
EXTERN void PETSC_STDCALL                           DGETRS(char*,int,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#  define BLASgemm_(a,b,c,d,e,f,g,h,i,j,k,l,m)      DGEMM((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m))
EXTERN void PETSC_STDCALL                           DGEMM(char *,int,char*,int,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*);
#  define LAPACKgesvd_(a,b,c,d,e,f,g,h,i,j,k,l,m,n) DGESVD((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n))
EXTERN void PETSC_STDCALL                           DGESVD(char *,int,char *,int,PetscBLASInt*,PetscBLASInt*,PetscScalar *,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#  define LAPACKgeev_(a,b,c,d,e,f,g,h,i,j,k,l,m,n)  DGEEV((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n))
EXTERN void PETSC_STDCALL                           DGEEV(char *,int,char *,int,PetscBLASInt*,PetscScalar *,PetscBLASInt*,PetscReal*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);

#  define LAPACKsyev_(a,b,c,d,e,f,g,h,i)            DSYEV((a),(b),1,(c),1,(d),(e),(f),(g),(h),(i)) 
EXTERN void PETSC_STDCALL                           DSYEV(const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#  define LAPACKsyevx_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t) DSYEVX((a),(b),1,(c),1,(d),1,(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t))
EXTERN void PETSC_STDCALL                           DSYEVX(const char*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);

#  define LAPACKsygv_(a,b,c,d,e,f,g,h,i,j,k,l)      DSYGV((a),(b),1,(c),1,(d),(e),(f),(g),(h),(i),(j),(k),(l)) 
EXTERN void PETSC_STDCALL                           DSYGV(PetscBLASInt*,const char*,int,const char*,int,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#  define LAPACKsygvx_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w) DSYGVX((a),(b),1,(c),1,(d),1,(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t),(u),(v),(w))
EXTERN void PETSC_STDCALL                           DSYGVX(PetscBLASInt*,const char*,int,const char*,int,const char*,int,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
#  define LAPACKstebz_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r) DSTEBZ((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r))
EXTERN void PETSC_STDCALL                           DSTEBZ(const char*,int,const char*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*);
# endif
#else
# if defined(PETSC_USES_FORTRAN_SINGLE) || defined(PETSC_USE_SINGLE)
/* Complex single precision without character string arguments. */
#  define ZGEQRF  CGEQRF
#  define ZGETRF  CGETRF
#  define ZDOTC   CDOTC
#  define DZNRM2  SCNRM2
#  define ZSCAL   CSCAL
#  define ZCOPY   CCOPY
#  define ZSWAP   CSWAP
#  define ZAXPY   CAXPY
#  define DZASUM  SCASUM
#  define ZTRMV   CTRMV
#  define ZPTTRF  CPTTRF
#  define ZSTEIN  CSTEIN
/* Complex single precision with character string arguments. */
#  define LAPACKtrtrs_(a,b,c,d,e,f,g,h,i,j)         CTRTRS((a),1,(b),1,(c),1,(d),(e),(f),(g),(h),(i),(j))
EXTERN void PETSC_STDCALL                           CTRTRS(char*,int,char*,int,char*,int,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#  define LAPACKpotrf_(a,b,c,d,e)                   CPOTRF((a),1,(b),(c),(d),(e))
EXTERN void PETSC_STDCALL                           CPOTRF(char*,int,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#  define LAPACKpotrs_(a,b,c,d,e,f,g,h)             CPOTRS((a),1,(b),(c),(d),(e),(f),(g),(h))
EXTERN void PETSC_STDCALL                           CPOTRS(char*,int,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#  define BLASgemv_(a,b,c,d,e,f,g,h,i,j,k)          CGEMV((a),1,(b),(c),(d),(e),(f),(g),(h),(i),(j),(k))
EXTERN void PETSC_STDCALL                           CGEMV(char*,int,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar *,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*);
#  define LAPACKgetrs_(a,b,c,d,e,f,g,h,i)           CGETRS((a),1,(b),(c),(d),(e),(f),(g),(h),(i))
EXTERN void PETSC_STDCALL                           CGETRS(char*,int,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#  define BLASgemm_(a,b,c,d,e,f,g,h,i,j,k,l,m)      SGEMM((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m))
EXTERN void PETSC_STDCALL                           SGEMM(char *,int,char*,int,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*);
#  define LAPACKgeev_(a,b,c,d,e,f,g,h,i,j,k,l,m,n)  CGEEV((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n))
EXTERN void PETSC_STDCALL                           CGEEV(char *,int,char *,int,PetscBLASInt*,PetscScalar *,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*);
#  define LAPACKgesvd_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o) CGESVD((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o))
EXTERN void PETSC_STDCALL                             CGESVD(char *,int,char *,int,PetscBLASInt*,PetscBLASInt*,PetscScalar *,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*);

#  define LAPACKsyev_(a,b,c,d,e,f,g,h,i,j)          CSYEV((a),(b),(c),(d),(e),(f),(g),(h),(i),(j))
EXTERN void PETSC_STDCALL                           CSYEV(const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*);
#  define LAPACKsyevx_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u) CSYEVX((a),(b),1,(c),1,(d),1,(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t),(u))
EXTERN void PETSC_STDCALL                           CSYEVX(const char*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*);

#  define LAPACKsygv_(a,b,c,d,e,f,g,h,i,j,k,l)      CSYGV((a),(b),1,(c),1,(d),(e),(f),(g),(h),(i),(j),(k),(l))
EXTERN void PETSC_STDCALL                           CSYGV(PetscBLASInt*,const char*,int,const char*,int,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#  define LAPACKsygvx_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w) CSYGVX((a),(b),1,(c),1,(d),1,(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t),(u),(v),(w))
EXTERN void PETSC_STDCALL                           CSYGVX(PetscBLASInt*,const char*,int,const char*,int,const char*,int,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
// --- new
#  define LAPACKpttrs_(a,b,c,d,e,f,g,h) CPTTRS((a),1),(b),(c),(d),(e),(f),(g),(h))
EXTERN void PETSC_STDCALL                           CPTTRS(const char*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
# else
/* Complex double precision without character string arguments */
#  define LAPACKgeqrf_ ZGEQRF
#  define LAPACKgetrf_ ZGETRF
#  define BLASdot_     ZDOTC
#  define BLASnrm2_    DZNRM2
#  define BLASscal_    ZSCAL
#  define BLAScopy_    ZCOPY
#  define BLASswap_    ZSWAP
#  define BLASaxpy_    ZAXPY
#  define BLASasum_    DZASUM
#  define BLAStrmv_    ZTRMV
#  define LAPACKpttrf_ ZPTTRF
#  define LAPACKstein_ ZSTEIN
/* Complex double precision with character string arguments */
#  define LAPACKtrtrs_(a,b,c,d,e,f,g,h,i,j)         ZTRTRS((a),1,(b),1,(c),1,(d),(e),(f),(g),(h),(i),(j))
EXTERN void PETSC_STDCALL                           ZTRTRS(char*,int,char*,int,char*,int,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#  define LAPACKpotrf_(a,b,c,d,e)                   ZPOTRF((a),1,(b),(c),(d),(e))
EXTERN void PETSC_STDCALL                           ZPOTRF(char*,int,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#  define LAPACKpotrs_(a,b,c,d,e,f,g,h)             ZPOTRS((a),1,(b),(c),(d),(e),(f),(g),(h))
EXTERN void PETSC_STDCALL                           ZPOTRS(char*,int,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#  define BLASgemv_(a,b,c,d,e,f,g,h,i,j,k)          ZGEMV((a),1,(b),(c),(d),(e),(f),(g),(h),(i),(j),(k))
EXTERN void PETSC_STDCALL                           ZGEMV(char*,int,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar *,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*);
#  define LAPACKgetrs_(a,b,c,d,e,f,g,h,i)           ZGETRS((a),1,(b),(c),(d),(e),(f),(g),(h),(i))
EXTERN void PETSC_STDCALL                           ZGETRS(char*,int,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#  define BLASgemm_(a,b,c,d,e,f,g,h,i,j,k,l,m)      ZGEMM((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m))
EXTERN void PETSC_STDCALL                           ZGEMM(char *,int,char*,int,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*);
#  define LAPACKgeev_(a,b,c,d,e,f,g,h,i,j,k,l,m,n)  ZGEEV((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n))
EXTERN void PETSC_STDCALL                           ZGEEV(char *,int,char *,int,PetscBLASInt*,PetscScalar *,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*);
#  define LAPACKgesvd_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o) ZGESVD((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o))
EXTERN void PETSC_STDCALL                             ZGESVD(char *,int,char *,int,PetscBLASInt*,PetscBLASInt*,PetscScalar *,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*);

#  define LAPACKsyev_(a,b,c,d,e,f,g,h,i,j)          ZSYEV((a),(b),(c),(d),(e),(f),(g),(h),(i),(j))
EXTERN void PETSC_STDCALL                           ZSYEV(const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*);
#  define LAPACKsyevx_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u)    ZSYEVX((a),(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t),(u))
EXTERN void PETSC_STDCALL                           ZSYEVX(const char*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*);

#  define LAPACKsygv_(a,b,c,d,e,f,g,h,i,j,k,l)      ZSYGV((a),(b),1,(c),1,(d),(e),(f),(g),(h),(i),(j),(k),(l))
EXTERN void PETSC_STDCALL                           ZSYGV(PetscBLASInt*,const char*,int,const char*,int,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#  define LAPACKsygvx_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w)    ZSYGVX((a),(b),1,(c),1,(d),1,(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t),(u),(v),(w))
EXTERN void PETSC_STDCALL                           ZSYGVX(PetscBLASInt*,const char*,int,const char*,int,const char*,int,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);

#  define LAPACKpttrs_(a,b,c,d,e,f,g,h)             ZPTTRS((a),(b),(c),(d),(e),(f),(g),(h))
EXTERN void PETSC_STDCALL                           ZPTTRS(const char*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
# endif
#endif

EXTERN void      PETSC_STDCALL LAPACKgetrf_(PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
EXTERN void      PETSC_STDCALL LAPACKgeqrf_(PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN PetscReal PETSC_STDCALL BLASdot_(PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*);
EXTERN PetscReal PETSC_STDCALL BLASnrm2_(PetscBLASInt*,PetscScalar*,PetscBLASInt*);
EXTERN void      PETSC_STDCALL BLASscal_(PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*);
EXTERN void      PETSC_STDCALL BLAScopy_(PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*);
EXTERN void      PETSC_STDCALL BLASswap_(PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*);
EXTERN void      PETSC_STDCALL BLASaxpy_(PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*);
EXTERN PetscReal PETSC_STDCALL BLASasum_(PetscBLASInt*,PetscScalar*,PetscBLASInt*);

EXTERN_C_END
PETSC_EXTERN_CXX_END
#endif
