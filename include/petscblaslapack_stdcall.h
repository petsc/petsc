/*
  This file deals with STDCALL Fortran 77 naming conventions.  This also
  assumes PETSC_HAVE_FORTRAN_CAPS is also defined, which is the case on certain Windows
  FORTRAN compilers which use STDCALL.

  Another problem is character strings are represented differently on 
  on some machines in C and Fortran 77. This problem comes up on some Windows compilers.
*/
#if !defined(_BLASLAPACK_STDCALL_H)
#define _BLASLAPACK_STDCALL_H
PETSC_EXTERN_CXX_BEGIN
EXTERN_C_BEGIN

#if !defined(PETSC_USE_COMPLEX)
# if defined(PETSC_USES_FORTRAN_SINGLE) || defined(PETSC_USE_SCALAR_SINGLE)
/* Real single precision without character string arguments. */
#  define LAPACKgeqrf_ SGEQRF
#  define LAPACKungqr_ SORGQR
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
#  define LAPACKgesv_  SGESV
#  define LAPACKgelss_ SGELSS
/* Real single precision with character string arguments. */
#  define LAPACKormqr_(a,b,c,d,e,f,g,h,i,j,k,l,m)   SORMQR((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m))
EXTERN void PETSC_STDCALL                           SORMQR(const char*,int,const char*,int,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#  define LAPACKtrtrs_(a,b,c,d,e,f,g,h,i,j)         STRTRS((a),1,(b),1,(c),1,(d),(e),(f),(g),(h),(i),(j))
EXTERN void PETSC_STDCALL                           STRTRS(const char*,int,const char*,int,const char*,int,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#  define LAPACKpotrf_(a,b,c,d,e)                   SPOTRF((a),1,(b),(c),(d),(e))
EXTERN void PETSC_STDCALL                           SPOTRF(const char*,int,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#  define LAPACKpotrs_(a,b,c,d,e,f,g,h)             SPOTRS((a),1,(b),(c),(d),(e),(f),(g),(h))
EXTERN void PETSC_STDCALL                           SPOTRS(const char*,int,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#  define BLASgemm_(a,b,c,d,e,f,g,h,i,j,k,l,m)      SGEMM((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m))
EXTERN void PETSC_STDCALL                           SGEMM(const char*,int,const char*,int,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*);
#  define LAPACKgetrs_(a,b,c,d,e,f,g,h,i)           SGETRS((a),1,(b),(c),(d),(e),(f),(g),(h),(i))
EXTERN void PETSC_STDCALL                           SGETRS(const char*,int,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#  define BLASgemv_(a,b,c,d,e,f,g,h,i,j,k)          SGEMV((a),1,(b),(c),(d),(e),(f),(g),(h),(i),(j),(k))
EXTERN void PETSC_STDCALL                           SGEMV(const char*,int,PetscBLASInt*,PetscBLASInt*,PetscScalar*,const PetscScalar*,PetscBLASInt*,PetscScalar *,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*);
#  define LAPACKgeev_(a,b,c,d,e,f,g,h,i,j,k,l,m,n)  SGEEV((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n))
EXTERN void PETSC_STDCALL                           SGEEV(const char*,int,const char*,int,PetscBLASInt*,PetscScalar *,PetscBLASInt*,PetscReal*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#  define LAPACKgesvd_(a,b,c,d,e,f,g,h,i,j,k,l,m,n) SGESVD((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n))
EXTERN void PETSC_STDCALL                           SGESVD(const char*,int,const char*,int,PetscBLASInt*,PetscBLASInt*,PetscScalar *,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);

#  define LAPACKsyev_(a,b,c,d,e,f,g,h,i)            SSYEV((a),(b),1,(c),1,(d),(e),(f),(g),(h),(i)) 
EXTERN void PETSC_STDCALL                           SSYEV(const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#  define LAPACKsyevx_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t) SSYEVX((a),(b),1,(c),1,(d),1,(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t))
EXTERN void PETSC_STDCALL                           SSYEVX(const char*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);

#  define LAPACKsygv_(a,b,c,d,e,f,g,h,i,j,k,l)      SSYGV((a),(b),1,(c),1,(d),(e),(f),(g),(h),(i),(j),(k),(l)) 
EXTERN void PETSC_STDCALL                           SSYGV(PetscBLASInt*,const char*,int,const char*,int,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#  define LAPACKsygvx_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w) SSYGVX((a),(b),1,(c),1,(d),1,(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t),(u),(v),(w))
EXTERN void PETSC_STDCALL                           SSYGVX(PetscBLASInt*,const char*,int,const char*,int,const char*,int,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);

#  define LAPACKstebz_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r) SSTEBZ((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r))
EXTERN void PETSC_STDCALL                          SSTEBZ(const char*,int,const char*,int,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*);
EXTERN void PETSC_STDCALL                          SPTTRS(PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
# else
/* Real double precision without character string arguments. */
#  define LAPACKgeqrf_ DGEQRF
#  define LAPACKungqr_ DORGQR
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
#  define LAPACKgesv_  DGESV
#  define LAPACKgelss_ DGELSS
/* Real double precision with character string arguments. */
#  define LAPACKormqr_(a,b,c,d,e,f,g,h,i,j,k,l,m)   DORMQR((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m))
EXTERN void PETSC_STDCALL                           DORMQR(const char*,int,const char*,int,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#  define LAPACKtrtrs_(a,b,c,d,e,f,g,h,i,j)         DTRTRS((a),1,(b),1,(c),1,(d),(e),(f),(g),(h),(i),(j))
EXTERN void PETSC_STDCALL                           DTRTRS(const char*,int,const char*,int,const char*,int,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#  define LAPACKpotrf_(a,b,c,d,e)                   DPOTRF((a),1,(b),(c),(d),(e))
EXTERN void PETSC_STDCALL                           DPOTRF(const char*,int,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#  define LAPACKpotrs_(a,b,c,d,e,f,g,h)             DPOTRS((a),1,(b),(c),(d),(e),(f),(g),(h))
EXTERN void PETSC_STDCALL                           DPOTRS(const char*,int,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#  define BLASgemv_(a,b,c,d,e,f,g,h,i,j,k)          DGEMV((a),1,(b),(c),(d),(e),(f),(g),(h),(i),(j),(k))
EXTERN void PETSC_STDCALL                           DGEMV(const char*,int,PetscBLASInt*,PetscBLASInt*,PetscScalar*,const PetscScalar*,PetscBLASInt*,PetscScalar *,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*);
#  define LAPACKgetrs_(a,b,c,d,e,f,g,h,i)           DGETRS((a),1,(b),(c),(d),(e),(f),(g),(h),(i))
EXTERN void PETSC_STDCALL                           DGETRS(const char*,int,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#  define BLASgemm_(a,b,c,d,e,f,g,h,i,j,k,l,m)      DGEMM((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m))
EXTERN void PETSC_STDCALL                           DGEMM(const char*,int,const char*,int,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*);
#  define LAPACKgesvd_(a,b,c,d,e,f,g,h,i,j,k,l,m,n) DGESVD((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n))
EXTERN void PETSC_STDCALL                           DGESVD(const char*,int,const char*,int,PetscBLASInt*,PetscBLASInt*,PetscScalar *,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#  define LAPACKgeev_(a,b,c,d,e,f,g,h,i,j,k,l,m,n)  DGEEV((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n))
EXTERN void PETSC_STDCALL                           DGEEV(const char*,int,const char*,int,PetscBLASInt*,PetscScalar *,PetscBLASInt*,PetscReal*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);

#  define LAPACKsyev_(a,b,c,d,e,f,g,h,i)            DSYEV((a),(b),1,(c),1,(d),(e),(f),(g),(h),(i)) 
EXTERN void PETSC_STDCALL                           DSYEV(const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#  define LAPACKsyevx_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t) DSYEVX((a),(b),1,(c),1,(d),1,(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t))
EXTERN void PETSC_STDCALL                           DSYEVX(const char*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);

#  define LAPACKsygv_(a,b,c,d,e,f,g,h,i,j,k,l)      DSYGV((a),(b),1,(c),1,(d),(e),(f),(g),(h),(i),(j),(k),(l)) 
EXTERN void PETSC_STDCALL                           DSYGV(PetscBLASInt*,const char*,int,const char*,int,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#  define LAPACKsygvx_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w) DSYGVX((a),(b),1,(c),1,(d),1,(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t),(u),(v),(w))
EXTERN void PETSC_STDCALL                           DSYGVX(PetscBLASInt*,const char*,int,const char*,int,const char*,int,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
#  define LAPACKstebz_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r) DSTEBZ((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r))
EXTERN void PETSC_STDCALL                           DSTEBZ(const char*,int,const char*,int,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*);
EXTERN void PETSC_STDCALL                           DPTTRS(PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
# endif
#else
# if defined(PETSC_USES_FORTRAN_SINGLE) || defined(PETSC_USE_SCALAR_SINGLE)
/* Complex single precision without character string arguments. */
#  define LAPACKgeqrf_ CGEQRF
#  define LAPACKungqr_ CUNGQR
#  define LAPACKgetrf_ CGETRF
#  define BLASdot_     CDOTC
#  define BLASnrm2_    SCNRM2
#  define BLASscal_    CSCAL
#  define BLAScopy_    CCOPY
#  define BLASswap_    CSWAP
#  define BLASaxpy_    CAXPY
#  define BLASasum_    SCASUM
#  define BLAStrmv_    CTRMV
#  define LAPACKpttrf_ CPTTRF
#  define LAPACKstein_ CSTEIN
#  define ZGESV   CGESV
#  define ZGELSS  CGELSS
/* Complex single precision with character string arguments. */
#  define LAPACKtrtrs_(a,b,c,d,e,f,g,h,i,j)         CTRTRS((a),1,(b),1,(c),1,(d),(e),(f),(g),(h),(i),(j))
EXTERN void PETSC_STDCALL                           CTRTRS(const char*,int,const char*,int,const char*,int,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#  define LAPACKpotrf_(a,b,c,d,e)                   CPOTRF((a),1,(b),(c),(d),(e))
EXTERN void PETSC_STDCALL                           CPOTRF(const char*,int,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#  define LAPACKpotrs_(a,b,c,d,e,f,g,h)             CPOTRS((a),1,(b),(c),(d),(e),(f),(g),(h))
EXTERN void PETSC_STDCALL                           CPOTRS(const char*,int,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#  define BLASgemv_(a,b,c,d,e,f,g,h,i,j,k)          CGEMV((a),1,(b),(c),(d),(e),(f),(g),(h),(i),(j),(k))
EXTERN void PETSC_STDCALL                           CGEMV(const char*,int,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar *,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*);
#  define LAPACKgetrs_(a,b,c,d,e,f,g,h,i)           CGETRS((a),1,(b),(c),(d),(e),(f),(g),(h),(i))
EXTERN void PETSC_STDCALL                           CGETRS(const char*,int,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#  define BLASgemm_(a,b,c,d,e,f,g,h,i,j,k,l,m)      CGEMM((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m))
EXTERN void PETSC_STDCALL                           CGEMM(const char*,int,const char*,int,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*);
#  define LAPACKgeev_(a,b,c,d,e,f,g,h,i,j,k,l,m,n)  CGEEV((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n))
EXTERN void PETSC_STDCALL                           CGEEV(const char*,int,const char*,int,PetscBLASInt*,PetscScalar *,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*);
#  define LAPACKgesvd_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o) CGESVD((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o))
EXTERN void PETSC_STDCALL                             CGESVD(const char*,int,const char*,int,PetscBLASInt*,PetscBLASInt*,PetscScalar *,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*);

#  define LAPACKsyev_(a,b,c,d,e,f,g,h,i,j)          CSYEV((a),(b),(c),(d),(e),(f),(g),(h),(i),(j))
EXTERN void PETSC_STDCALL                           CSYEV(const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*);
#  define LAPACKsyevx_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u) CSYEVX((a),(b),1,(c),1,(d),1,(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t),(u))
EXTERN void PETSC_STDCALL                           CSYEVX(const char*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*);

#  define LAPACKsygv_(a,b,c,d,e,f,g,h,i,j,k,l,m)      CHEGV((a),(b),1,(c),1,(d),(e),(f),(g),(h),(i),(j),(k),(l),(m))
EXTERN void PETSC_STDCALL                           CHEGV(PetscBLASInt*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*);
#  define LAPACKsygvx_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x) CHEGVX((a),(b),1,(c),1,(d),1,(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t),(u),(v),(w),(x))
EXTERN void PETSC_STDCALL                           CHEGVX(PetscBLASInt*,const char*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
#  define LAPACKpttrs_(a,b,c,d,e,f,g,h) CPTTRS((a),1,(b),(c),(d),(e),(f),(g),(h))
EXTERN void PETSC_STDCALL                           CPTTRS(const char*,int,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
# else
/* Complex double precision without character string arguments */
#  define LAPACKgeqrf_ ZGEQRF
#  define LAPACKungqr_ ZUNGQR
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
#  define LAPACKgesv_  ZGESV
#  define LAPACKgelss_ ZGELSS
/* Complex double precision with character string arguments */
#  define LAPACKtrtrs_(a,b,c,d,e,f,g,h,i,j)         ZTRTRS((a),1,(b),1,(c),1,(d),(e),(f),(g),(h),(i),(j))
EXTERN void PETSC_STDCALL                           ZTRTRS(const char*,int,const char*,int,const char*,int,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#  define LAPACKpotrf_(a,b,c,d,e)                   ZPOTRF((a),1,(b),(c),(d),(e))
EXTERN void PETSC_STDCALL                           ZPOTRF(const char*,int,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#  define LAPACKpotrs_(a,b,c,d,e,f,g,h)             ZPOTRS((a),1,(b),(c),(d),(e),(f),(g),(h))
EXTERN void PETSC_STDCALL                           ZPOTRS(const char*,int,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#  define BLASgemv_(a,b,c,d,e,f,g,h,i,j,k)          ZGEMV((a),1,(b),(c),(d),(e),(f),(g),(h),(i),(j),(k))
EXTERN void PETSC_STDCALL                           ZGEMV(const char*,int,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar *,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*);
#  define LAPACKgetrs_(a,b,c,d,e,f,g,h,i)           ZGETRS((a),1,(b),(c),(d),(e),(f),(g),(h),(i))
EXTERN void PETSC_STDCALL                           ZGETRS(const char*,int,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#  define BLASgemm_(a,b,c,d,e,f,g,h,i,j,k,l,m)      ZGEMM((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m))
EXTERN void PETSC_STDCALL                           ZGEMM(const char*,int,const char*,int,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*);
#  define LAPACKgeev_(a,b,c,d,e,f,g,h,i,j,k,l,m,n)  ZGEEV((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n))
EXTERN void PETSC_STDCALL                           ZGEEV(const char*,int,const char*,int,PetscBLASInt*,PetscScalar *,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*);
#  define LAPACKgesvd_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o) ZGESVD((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o))
EXTERN void PETSC_STDCALL                             ZGESVD(const char*,int,const char*,int,PetscBLASInt*,PetscBLASInt*,PetscScalar *,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*);

#  define LAPACKsyev_(a,b,c,d,e,f,g,h,i,j)           ZHEEV((a),(b),(c),(d),(e),(f),(g),(h),(i),(j))
EXTERN void PETSC_STDCALL                            ZHEEV(const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*);
#  define LAPACKsyevx_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u)    ZHEEVX((a),(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t),(u))
EXTERN void PETSC_STDCALL                           ZHEEVX(const char*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);

#  define LAPACKsygv_(a,b,c,d,e,f,g,h,i,j,k,l,m)      ZHEGV((a),(b),1,(c),1,(d),(e),(f),(g),(h),(i),(j),(k),(l),(m))
EXTERN void PETSC_STDCALL                           ZHEGV(PetscBLASInt*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*);
#  define LAPACKsygvx_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x)    ZHEGVX((a),(b),1,(c),1,(d),1,(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t),(u),(v),(w),(x))
EXTERN void PETSC_STDCALL                           ZHEGVX(PetscBLASInt*,const char*,int,const char*,int,const char*,int,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);

#  define LAPACKpttrs_(a,b,c,d,e,f,g,h)             ZPTTRS((a),1,(b),(c),(d),(e),(f),(g),(h))
EXTERN void PETSC_STDCALL                           ZPTTRS(const char*,int,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
# endif
#endif

EXTERN void      PETSC_STDCALL LAPACKgetrf_(PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
EXTERN void      PETSC_STDCALL LAPACKgeqrf_(PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN void      PETSC_STDCALL LAPACKungqr_(PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN void      PETSC_STDCALL LAPACKpttrf_(PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*);
EXTERN void      PETSC_STDCALL LAPACKstein_(PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
EXTERN void      PETSC_STDCALL LAPACKgesv_(const PetscBLASInt*,const PetscBLASInt*,PetscScalar*,const PetscBLASInt*,PetscBLASInt*,PetscScalar*,const PetscBLASInt*,PetscBLASInt*);

#if defined(PETSC_USE_COMPLEX)
EXTERN void      PETSC_STDCALL LAPACKgelss_(const PetscBLASInt*,const PetscBLASInt*,const PetscBLASInt*,PetscScalar*,const PetscBLASInt*,PetscScalar*,const PetscBLASInt*,PetscReal*,const PetscReal*,PetscBLASInt*,PetscScalar*,const PetscBLASInt*,PetscReal*,PetscBLASInt*);
#else
EXTERN void      PETSC_STDCALL LAPACKgelss_(const PetscBLASInt*,const PetscBLASInt*,const PetscBLASInt*,PetscScalar*,const PetscBLASInt*,PetscScalar*,const PetscBLASInt*,PetscReal*,const PetscReal*,PetscBLASInt*,PetscScalar*,const PetscBLASInt*,PetscBLASInt*);
#endif


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
