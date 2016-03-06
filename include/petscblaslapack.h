/*
  This file dispatches between various header files for blas/lapack distributions to handle the name mangling.
  It also provides C prototypes for all the BLAS/LAPACK functions that PETSc uses

  This is not included automatically by petscsys.h because some external packages include their own prototypes for
  certain BLAS/LAPACK functions that conflict with the ones given here. Hence this should only be included when needed.

  The BLAS/LAPACK name mangling is almost (but not always) the same as the Fortran mangling; and exists even if there is
  not Fortran compiler.

  PETSC_BLASLAPACK_STDCALL  is for Microsoft Windows compiles, it also implicitly has capitalized function names
  PETSC_BLASLAPACK_UNDERSCORE BLAS/LAPACK function have an underscore at the end of each function name
  PETSC_BLASLAPACK_CAPS BLAS/LAPACK function names are all in capital letters
  PETSC_BLASLAPACK_C BLAS/LAPACK function names have no mangling

  PETSC_BLASLAPACK_SINGLEISDOUBLE - for Cray systems where the BLAS/LAPACK single precision (i.e. Fortran single precision is actually 64 bits)
                                    old Cray vector machines used to be this way, it is is not clear if any exist now.

  PetscBLASInt is almost always 32 bit integers but can be 64 bit integers for certain usages of MKL BLAS/LAPACK libraries

*/
#if !defined(_BLASLAPACK_H)
#define _BLASLAPACK_H

#define PetscStackCallBLAS(name,routine) do {                   \
    PetscStackPushNoCheck(name,PETSC_FALSE,PETSC_TRUE);         \
    routine;                                                    \
    PetscStackPop;                                              \
  } while(0)

#if defined(PETSC_BLASLAPACK_STDCALL)
#include <petscblaslapack_stdcall.h>
#else

#include <petscblaslapack_mangle.h>

PETSC_EXTERN void LAPACKgetrf_(PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void LAPACKgetri_(PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void LAPACKungqr_(PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void LAPACKgeqrf_(PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#if defined(PETSC_USE_REAL_SINGLE) && defined(PETSC_BLASLAPACK_SNRM2_RETURNS_DOUBLE)
PETSC_EXTERN double BLASnrm2_(const PetscBLASInt*,const PetscScalar*,const PetscBLASInt*);
#else
PETSC_EXTERN PetscReal BLASnrm2_(const PetscBLASInt*,const PetscScalar*,const PetscBLASInt*);
#endif
PETSC_EXTERN void BLASscal_(const PetscBLASInt*,const PetscScalar*,PetscScalar*,const PetscBLASInt*);
PETSC_EXTERN void BLAScopy_(const PetscBLASInt*,const PetscScalar*,const PetscBLASInt*,PetscScalar*,const PetscBLASInt*);
PETSC_EXTERN void BLASswap_(const PetscBLASInt*,PetscScalar*,const PetscBLASInt*,PetscScalar*,const PetscBLASInt*);
PETSC_EXTERN void BLASaxpy_(const PetscBLASInt*,const PetscScalar*,const PetscScalar*,const PetscBLASInt*,PetscScalar*,const PetscBLASInt*);
#if defined(PETSC_USE_REAL_SINGLE) && defined(PETSC_BLASLAPACK_SNRM2_RETURNS_DOUBLE)
PETSC_EXTERN double BLASasum_(const PetscBLASInt*,const PetscScalar*,const PetscBLASInt*);
#else
PETSC_EXTERN PetscReal BLASasum_(const PetscBLASInt*,const PetscScalar*,const PetscBLASInt*);
#endif
PETSC_EXTERN void LAPACKpttrf_(const PetscBLASInt*,PetscReal*,PetscScalar*,const PetscBLASInt*);
PETSC_EXTERN void LAPACKstein_(const PetscBLASInt*,PetscReal*,PetscReal*,const PetscBLASInt*,PetscReal*,const PetscBLASInt*,const PetscBLASInt*,PetscScalar*,const PetscBLASInt*,PetscReal*,const PetscBLASInt*,const PetscBLASInt*,const PetscBLASInt*);
PETSC_EXTERN void LAPACKgesv_(const PetscBLASInt*,const PetscBLASInt*,PetscScalar*,const PetscBLASInt*,PetscBLASInt*,PetscScalar*,const PetscBLASInt*,PetscBLASInt*);

PETSC_EXTERN void LAPACKpotrf_(const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void LAPACKpotri_(const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void LAPACKpotrs_(const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void LAPACKpotri_(const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void LAPACKsytrf_(const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void LAPACKsytrs_(const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void LAPACKsytri_(const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*);
PETSC_EXTERN void BLASgemv_(const char*,const PetscBLASInt*,const PetscBLASInt*,const PetscScalar*,const PetscScalar*,const PetscBLASInt*,const PetscScalar *,const PetscBLASInt*,const PetscScalar*,PetscScalar*,const PetscBLASInt*);
PETSC_EXTERN void LAPACKgetrs_(const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void BLAStrmv_(const char*,const char*,const char*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*);
PETSC_EXTERN void BLASgemm_(const char*,const char*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*);
PETSC_EXTERN void BLASsymm_(const char*,const char*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*);
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
#if defined(PETSC_USE_REAL_SINGLE) && defined(PETSC_BLASLAPACK_SDOT_RETURNS_DOUBLE)
PETSC_EXTERN double BLASdot_(const PetscBLASInt*,const PetscScalar*,const PetscBLASInt*,const PetscScalar*,const PetscBLASInt*);
PETSC_EXTERN double BLASdotu_(const PetscBLASInt*,const PetscScalar*,const PetscBLASInt*,const PetscScalar*,const PetscBLASInt*);
#else
PETSC_EXTERN PetscScalar BLASdot_(const PetscBLASInt*,const PetscScalar*,const PetscBLASInt*,const PetscScalar*,const PetscBLASInt*);
PETSC_EXTERN PetscScalar BLASdotu_(const PetscBLASInt*,const PetscScalar*,const PetscBLASInt*,const PetscScalar*,const PetscBLASInt*);
#endif
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
