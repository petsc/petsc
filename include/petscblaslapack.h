/*
  This file dispatches between various header files for blas/lapack distributions to handle the name mangling.
  It also provides C prototypes for all the BLAS/LAPACK functions that PETSc uses

  This is not included automatically by petscsys.h because some external packages include their own prototypes for
  certain BLAS/LAPACK functions that conflict with the ones given here. Hence this should only be included when needed.

*/
#if !defined(_BLASLAPACK_H)
#define _BLASLAPACK_H

/* Do not use PetscStackPush/PetscStackPop because  */
#define PetscStackCallBLAS(name,routine) do {                   \
    PetscStackPushNoCheck(name,PETSC_FALSE,PETSC_TRUE);         \
    routine;                                                    \
    PetscStackPop;                                              \
  } while(0)

#if defined(PETSC_BLASLAPACK_STDCALL)
#include <petscblaslapack_stdcall.h>
#else

#if defined(PETSC_BLASLAPACK_UNDERSCORE)
#include <petscblaslapack_uscore.h>
#elif defined(PETSC_BLASLAPACK_CAPS)
#include <petscblaslapack_caps.h>
#else
#include <petscblaslapack_c.h>
#endif

PETSC_EXTERN void LAPACKgetrf_(PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void LAPACKgetri_(PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void LAPACKungqr_(PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void LAPACKgeqrf_(PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN PetscReal BLASnrm2_(const PetscBLASInt*,const PetscScalar*,const PetscBLASInt*);
PETSC_EXTERN void BLASscal_(const PetscBLASInt*,const PetscScalar*,PetscScalar*,const PetscBLASInt*);
PETSC_EXTERN void BLAScopy_(const PetscBLASInt*,const PetscScalar*,const PetscBLASInt*,PetscScalar*,const PetscBLASInt*);
PETSC_EXTERN void BLASswap_(const PetscBLASInt*,PetscScalar*,const PetscBLASInt*,PetscScalar*,const PetscBLASInt*);
PETSC_EXTERN void BLASaxpy_(const PetscBLASInt*,const PetscScalar*,const PetscScalar*,const PetscBLASInt*,PetscScalar*,const PetscBLASInt*);
PETSC_EXTERN PetscReal BLASasum_(const PetscBLASInt*,const PetscScalar*,const PetscBLASInt*);
PETSC_EXTERN void LAPACKpttrf_(const PetscBLASInt*,PetscReal*,PetscScalar*,const PetscBLASInt*);
PETSC_EXTERN void LAPACKstein_(const PetscBLASInt*,PetscReal*,PetscReal*,const PetscBLASInt*,PetscReal*,const PetscBLASInt*,const PetscBLASInt*,PetscScalar*,const PetscBLASInt*,PetscReal*,const PetscBLASInt*,const PetscBLASInt*,const PetscBLASInt*);
PETSC_EXTERN void LAPACKgesv_(const PetscBLASInt*,const PetscBLASInt*,PetscScalar*,const PetscBLASInt*,PetscBLASInt*,PetscScalar*,const PetscBLASInt*,PetscBLASInt*);

PETSC_EXTERN void LAPACKpotrf_(const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void LAPACKpotrs_(const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void BLASgemv_(const char*,const PetscBLASInt*,const PetscBLASInt*,const PetscScalar*,const PetscScalar*,const PetscBLASInt*,const PetscScalar *,const PetscBLASInt*,const PetscScalar*,PetscScalar*,const PetscBLASInt*);
PETSC_EXTERN void LAPACKgetrs_(const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void BLAStrmv_(const char*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*);
PETSC_EXTERN void BLASgemm_(const char*,const char*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*);
PETSC_EXTERN void BLAStrsm_(const char*,const char*,const char*,const char*,const PetscBLASInt*,const PetscBLASInt*,const PetscScalar*,const PetscScalar*,const PetscBLASInt*,PetscScalar*,const PetscBLASInt*);
PETSC_EXTERN void LAPACKormqr_(const char *,const char *t,PetscBLASInt * ,PetscBLASInt *,PetscBLASInt *,PetscScalar *,PetscBLASInt *,PetscScalar *,PetscScalar *,PetscBLASInt *,PetscScalar *,PetscBLASInt *,PetscBLASInt *);
PETSC_EXTERN void LAPACKsteqr_(const char*,PetscBLASInt*,PetscReal*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*);
PETSC_EXTERN void LAPACKhgeqz_(const char *,const char *,const char *,PetscBLASInt *,PetscBLASInt *,PetscBLASInt *,PetscScalar *,PetscBLASInt *,PetscScalar *,PetscBLASInt *,PetscScalar *,PetscScalar *,PetscScalar *,PetscScalar *,PetscBLASInt *,PetscScalar *,PetscBLASInt *,PetscScalar *,PetscBLASInt *,PetscBLASInt *);
PETSC_EXTERN void LAPACKtrtrs_(const char *,const char *, const char *,PetscBLASInt *,PetscBLASInt *,PetscScalar *,PetscBLASInt *,PetscScalar *,PetscBLASInt *,PetscBLASInt *);

/* handle complex dot() with special code */
#if defined(PETSC_USE_COMPLEX)
PETSC_STATIC_INLINE PetscScalar BLASdot_(const PetscBLASInt *n,const PetscScalar *x,const PetscBLASInt *sx,const PetscScalar *y,const PetscBLASInt *sy)
{
  PetscScalar sum=0.0;
  PetscInt    i,j,k;
  if (*sx==1 && *sy==1) {
    for (i=0; i < *n; i++) sum += PetscConj(x[i])*y[i];
  } else {
    for (i=0,j=0,k=0; i < *n; i++,j+=*sx,k+=*sy) sum += PetscConj(x[j])*y[k];
  }
  return sum;
}
PETSC_STATIC_INLINE PetscScalar BLASdotu_(const PetscBLASInt *n,const PetscScalar *x,const PetscBLASInt *sx,const PetscScalar *y,const PetscBLASInt *sy)
{
  PetscScalar sum=0.0;
  PetscInt    i,j,k;
  if (*sx==1 && *sy==1) {
    for (i=0; i < *n; i++) sum += x[i]*y[i];
  } else {
    for (i=0,j=0,k=0; i < *n; i++,j+=*sx,k+=*sy) sum += x[j]*y[k];
  }
  return sum;
}
#else
PETSC_EXTERN PetscScalar BLASdot_(const PetscBLASInt*,const PetscScalar*,const PetscBLASInt*,const PetscScalar*,const PetscBLASInt*);
PETSC_EXTERN PetscScalar BLASdotu_(const PetscBLASInt*,const PetscScalar*,const PetscBLASInt*,const PetscScalar*,const PetscBLASInt*);
#endif

/* Some functions prototypes differ between real and complex */
#if defined(PETSC_USE_COMPLEX)
PETSC_EXTERN void LAPACKgelss_(const PetscBLASInt*,const PetscBLASInt*,const PetscBLASInt*,PetscScalar*,const PetscBLASInt*,PetscScalar*,const PetscBLASInt*,PetscReal*,const PetscReal*,PetscBLASInt*,PetscScalar*,const PetscBLASInt*,PetscReal*,PetscBLASInt*);
PETSC_EXTERN void LAPACKsyev_(const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*);
PETSC_EXTERN void LAPACKsyevx_(const char*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void LAPACKsygv_(PetscBLASInt*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*);
PETSC_EXTERN void LAPACKsygvx_(PetscBLASInt*,const char*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void LAPACKpttrs_(const char*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void LAPACKgerfs_(const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscReal*,PetscBLASInt*);
PETSC_EXTERN void LAPACKtrsen_(const char*,const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void LAPACKtgsen_(PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void LAPACKgges_(const char*,const char*,const char*,PetscBLASInt(*)(),PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void LAPACKhseqr_(const char*,const char*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#else
PETSC_EXTERN void LAPACKgelss_(const PetscBLASInt*,const PetscBLASInt*,const PetscBLASInt*,PetscScalar*,const PetscBLASInt*,PetscScalar*,const PetscBLASInt*,PetscReal*,const PetscReal*,PetscBLASInt*,PetscScalar*,const PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void LAPACKsyev_(const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void LAPACKsyevx_(const char*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void LAPACKsygv_(PetscBLASInt*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void LAPACKsygvx_(PetscBLASInt*,const char*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void LAPACKpttrs_(PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void LAPACKstebz_(const char*,const char*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void LAPACKgerfs_(const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void LAPACKtrsen_(const char*,const char*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void LAPACKtgsen_(PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void LAPACKgges_(const char*,const char*,const char*,PetscBLASInt(*)(),PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void LAPACKhseqr_(const char*,const char*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#endif

/* ESSL uses a different calling sequence for dgeev(), zgeev() than LAPACK; */
#if defined(PETSC_HAVE_ESSL) && defined(PETSC_USE_COMPLEX)
PETSC_EXTERN void LAPACKgeev_(PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*);
PETSC_EXTERN void LAPACKgesvd_(const char*,const char*,PetscBLASInt *,PetscBLASInt*,PetscScalar *,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*);
#elif defined(PETSC_HAVE_ESSL)
PETSC_EXTERN void LAPACKgeev_(PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*);
PETSC_EXTERN void LAPACKgesvd_(const char*,const char*,PetscBLASInt *,PetscBLASInt*,PetscScalar *,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#elif defined(PETSC_USE_COMPLEX)
PETSC_EXTERN void LAPACKgeev_(const char*,const char*,PetscBLASInt *,PetscScalar *,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*);
PETSC_EXTERN void LAPACKgesvd_(const char*,const char*,PetscBLASInt *,PetscBLASInt*,PetscScalar *,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*);
#else
PETSC_EXTERN void LAPACKgeev_(const char*,const char*,PetscBLASInt *,PetscScalar *,PetscBLASInt*,PetscReal*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void LAPACKgesvd_(const char*,const char*,PetscBLASInt *,PetscBLASInt*,PetscScalar *,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#endif

#endif
#endif
