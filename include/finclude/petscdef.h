!
!
!  Part of the base include file for Fortran use of PETSc.
!  Note: This file should contain only define statements and
!  not the declaration of variables.

! No spaces for #defines as some compilers (PGI) also adds
! those additional spaces during preprocessing - bad for fixed format
!
#if !defined (__PETSCDEF_H)
#define __PETSCDEF_H
!
#include "petscconf.h"
!
#define MPI_Comm integer
!
#define PetscEnum integer
!
#define PetscTruth PetscEnum
#define PetscDataType PetscEnum
#define PetscFPTrap PetscEnum

#define PetscErrorCode integer
#define PetscCookie integer
#define PetscEvent integer
#define PetscMPIInt integer
!
!
! The real*8,complex*16 notatiton is used so that the 
! PETSc double/complex variables are not affected by 
! compiler options like -r4,-r8, sometimes invoked 
! by the user. NAG compiler does not like integer*4,real*8

#if defined(PETSC_USE_FORTRANKIND)
#define integer8 integer(kind=selected_int_kind(10))
#define integer4 integer(kind=selected_int_kind(5))
#else
#define integer8 integer*8
#define integer4 integer*4
#endif

#if (PETSC_SIZEOF_VOID_P == 8)
#define PetscFortranAddr integer8
#define PetscOffset integer8
#else
#define PetscOffset integer4
#define PetscFortranAddr integer4
#endif

#if defined(PETSC_USE_64BIT_INDICES)
#define PetscInt integer8
#else
#define PetscInt integer4
#endif

#if defined (PETSC_USE_FORTRANKIND)
#define PetscFortranFloat real(kind=selected_real_kind(5))
#define PetscFortranDouble real(kind=selected_real_kind(10))
#define PetscFortranComplex complex(kind=selected_real_kind(10))
#define PetscChar(a) character(len = a) ::
#else
#define PetscFortranFloat real*4
#define PetscFortranDouble real*8
#define PetscFortranComplex complex*16
#define PetscChar(a) character*(a)
#endif

#if defined(PETSC_USE_COMPLEX)
#define PETSC_SCALAR PETSC_COMPLEX
#else
#if defined(PETSC_USE_SINGLE)
#define PETSC_SCALAR PETSC_FLOAT
#else
#define PETSC_SCALAR PETSC_DOUBLE
#endif     
#endif
!
!     Macro for templating between real and complex
!
#if defined(PETSC_USE_COMPLEX)
#define PetscScalar PetscFortranComplex
!
! F90 uses real(), conjg() when KIND parameter is used.
!
#if defined (PETSC_MISSING_DREAL)
#define PetscRealPart(a) real(a)
#define PetscConj(a) conjg(a)
#define PetscImaginaryPart(a) aimg(a)
#else
#define PetscRealPart(a) dreal(a)
#define PetscConj(a) dconjg(a)
#define PetscImaginaryPart(a) daimg(a)
#endif
#define MPIU_SCALAR MPI_DOUBLE_COMPLEX
#else
#if defined (PETSC_USE_SINGLE)
#define PetscScalar PetscFortranFloat
#else
#define PetscScalar PetscFortranDouble
#endif
#define PetscRealPart(a) a
#define PetscConj(a) a
#define PetscImaginaryPart(a) a
#define MPIU_SCALAR MPI_DOUBLE_PRECISION
#endif

#if defined (PETSC_USE_SINGLE)
#define PetscReal PetscFortranFloat
#else
#define PetscReal PetscFortranDouble
#endif

!
!    Allows the matrix Fortran Kernels to work with single precision
!    matrix data structures
!
#if defined(PETSC_USE_COMPLEX)
#define MatScalar PetscScalar 
#elif defined(PETSC_USE_MAT_SINGLE)
#define MatScalar real*4
#else
#define MatScalar PetscScalar
#endif
!
!     PetscLogDouble variables are used to contain double precision numbers
!     that are not used in the numerical computations, but rather in logging,
!     timing etc.
!
#define PetscObject PetscFortranAddr
#define PetscLogDouble PetscFortranDouble
!
!     Macros for error checking
!
#if defined(PETSC_USE_DEBUG)
#define SETERRQ(n,s,ierr) call MPI_Abort(PETSC_COMM_WORLD,n,ierr)
#define CHKERRQ(n) if (n .ne. 0) call MPI_Abort(PETSC_COMM_WORLD,n,n)
#define CHKMEMQ call chkmemfortran(__LINE__,__FILE__,ierr)
#else
#define SETERRQ(n,s,ierr)
#define CHKERRQ(n)
#define CHKMEMQ
#endif

#define PetscMatlabEngine PetscFortranAddr

#if defined(PETSC_STDCALL)
#define PETSC_DEC_ATTRIBUTES(A,B) DEC$ ATTRIBUTESC, ALIAS:B ::A
#else
#define PETSC_DEC_ATTRIBUTES(A,B)
#endif

#endif
