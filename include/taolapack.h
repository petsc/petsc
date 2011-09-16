#if !defined __TAO_LAPACK_H
#define __TAO_LAPACK_H
#include "petsc.h"
#include "private/fortranimpl.h"

#if defined(PETSC_BLASLAPACK_STDCALL) 
# if defined(PETSC_USE_FORTRAN_SINGLE) || defined(PETSC_USE_REAL_SINGLE)
PETSC_EXTERN_CXX_BEGIN
EXTERN_C_BEGIN
#  define LAPACKormqr_(a,b,c,d,e,f,g,h,i,j,k,l,m)   SORMQR((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m))
#  define LAPACKtrtrs_(a,b,c,d,e,f,g,h,i,j)         STRTRS((a),1,(b),1,(c),1,(d),(e),(f),(g),(h),(i),(j))
EXTERN void PETSC_STDCALL                           SORMQR(const char*,int,const char*,int,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN void PETSC_STDCALL                           STRTRS(const char*,int,const char*,int,const char*,int,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
# else
#  define LAPACKormqr_(a,b,c,d,e,f,g,h,i,j,k,l,m)   DORMQR((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m))
#  define LAPACKtrtrs_(a,b,c,d,e,f,g,h,i,j)         DTRTRS((a),1,(b),1,(c),1,(d),(e),(f),(g),(h),(i),(j))
EXTERN void PETSC_STDCALL                           DORMQR(const char*,int,const char*,int,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN void PETSC_STDCALL                           DTRTRS(const char*,int,const char*,int,const char*,int,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
# endif
EXTERN_C_END
PETSC_EXTERN_CXX_END
#elif defined(PETSC_BLASLAPACK_UNDERSCORE)
# if defined(PETSC_USE_REAL_SINGLE)
#  define LAPACKormqr_ sormqr_
#  define LAPACKtrtrs_ strtrs_
# elif defined(PETSC_USE_REAL_DOUBLE)
#  define LAPACKormqr_ dormqr_
#  define LAPACKtrtrs_ dtrtrs_
# else
#  define LAPACKormqr_ qormqr_
#  define LAPACKtrtrs_ qtrtrs_
# endif

#elif defined(PETSC_BLASLAPACK_CAPS)
# if defined(PETSC_USE_REAL_SINGLE) || defined(PETSC_USE_FORTRAN_SINGLE)
#  define LAPACKormqr_ SORMQR
#  define LAPACKtrtrs_ STRTRS
# else
#  define LAPACKormqr_ DORMQR
#  define LAPACKtrtrs_ DTRTRS
# endif

#else
# if defined(PETSC_USE_REAL_SINGLE)
#  define LAPACKormqr_ sormqr
#  define LAPACKtrtrs_ strtrs
# else
#  define LAPACKormqr_ dormqr
#  define LAPACKtrtrs_ dtrtrs
# endif

#endif


EXTERN_C_BEGIN
extern void LAPACKormqr_(const char*,const char*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
extern void LAPACKtrtrs_(const char*,const char*,const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
EXTERN_C_END


#endif /* defined __TAO_LAPACK_H */

