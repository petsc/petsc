!
!  $Id: petscdef.h,v 1.31 2001/09/04 15:42:43 balay Exp $;
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
#define PetscTruth integer
#define PetscDataType integer
#define PetscFPTrap integer
!
!
! The real*8,complex*16 notatiton is used so that the 
! PETSc double/complex variables are not affected by 
! compiler options like -r4,-r8, sometimes invoked 
! by the user. NAG compiler does not like integer*4,real*8
!
! ???? All integers should also be changed to PetscFortranInt ?????
!

#if (PETSC_SIZEOF_VOID_P == 8)
#define PetscOffset integer*8
#define PetscFortranAddr integer*8
#elif defined (PETSC_MISSING_FORTRANSTAR)
#define PetscOffset integer
#define PetscFortranAddr integer
#else
#define PetscOffset integer*4
#define PetscFortranAddr integer*4
#endif

#if (PETSC_SIZEOF_INT == 8)
#define PetscFortranInt integer*8
#elif defined (PETSC_MISSING_FORTRANSTAR)
#define PetscFortranInt integer
#else
#define PetscFortranInt integer*4
#endif

#if defined (PETSC_MISSING_FORTRANSTAR)
#define PetscFortranFloat real
#define PetscFortranDouble double precision
#define PetscFortranComplex complex (KIND=SELECTED_REAL_KIND(14))
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
#else
#define PetscRealPart(a) dreal(a)
#define PetscConj(a) dconjg(a)
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
!     Declare PETSC_NULL_OBJECT
!
#define PETSC_NULL_OBJECT PETSC_NULL_INTEGER
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
#define CHKMEMQ call chkmemfortran(__LINE__,__FILE__)
#define CHKMEMA CHKMEMQ
#else
#define SETERRQ(n,s)
#define CHKERRQ(n)
#define CHKMEMQ
#define CHKMEMA
#endif

#define PetscMatlabEngine PetscFortranAddr

#endif
