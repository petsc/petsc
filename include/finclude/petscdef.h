!
!  $Id: petscdef.h,v 1.9 1999/03/24 16:21:19 bsmith Exp balay $;
!
!  Part of the base include file for Fortran use of PETSc.
!  Note: This file should contain only define statements and
!  not the declaration of variables. This is the only fortran
! include file which is protected from multiple inclusions.
!
#if !defined (__PETSCDEF_H)
#define __PETSCDEF_H
!
#include "petscconf.h"
!
#define MPI_Comm integer
!
#define PetscTruth    integer
#define PetscDataType integer

!
! The real*8,complex*16 notatiton is used so that the 
! PETSc double/complex variables are not affected by 
! compiler options like -r4,-r8, sometimes invoked 
! by the user. NAG compiler does not like integer*4,real*8
!
! ???? All integers should also be changed to PetscFortranInt ?????
!

#if (SIZEOF_VOIDP == 8)
#define PetscOffset        integer*8
#define PetscFortranAddr   integer*8
#elif defined (HAVE_NAGF90) || defined (HAVE_SOLARISF90)
#define PetscOffset        integer
#define PetscFortranAddr   integer
#else
#define PetscOffset        integer*4
#define PetscFortranAddr   integer*4
#endif

#if (SIZEOF_INT == 8)
#define PetscFortranInt integer*8
#elif defined (HAVE_NAGF90) || defined (HAVE_SOLARISF90)
#define PetscFortranInt integer
#else
#define PetscFortranInt integer*4
#endif

#if defined (HAVE_NAGF90) || defined (HAVE_SOLARISF90)
#define PetscFortranDouble  double precision
#define PetscFortranComplex complex (KIND=SELECTED_REAL_KIND(14))
#else
#define PetscFortranDouble  real*8
#define PetscFortranComplex complex*16
#endif

#if defined(USE_PETSC_COMPLEX)
#define PETSC_SCALAR PETSC_COMPLEX
#else
#define PETSC_SCALAR PETSC_DOUBLE
#endif     
!
!     Macro for templating between real and complex
!
#if defined(USE_PETSC_COMPLEX)
#define Scalar       PetscFortranComplex
!
! F90 uses real(), conjg() when KIND parameter is used.
!
#if defined (HAVE_NAGF90) || defined (HAVE_IRIXF90) || defined (PARCH_freebsd)
#define PetscReal(a) real(a)
#define PetscConj(a) conjg(a)
#else
#define PetscReal(a) dreal(a)
#define PetscConj(a) dconjg(a)
#endif
#define MPIU_SCALAR   MPI_DOUBLE_COMPLEX
#else
#define Scalar       PetscFortranDouble
#define PetscReal(a) a
#define PetscConj(a) a
#define MPIU_SCALAR  MPI_DOUBLE_PRECISION
#endif
!
!    Allows the matrix Fortran Kernels to work with single precision
!    matrix data structures
!
#if defined(USE_PETSC_COMPLEX)
#define MatScalar Scalar 
#elif defined(USE_MAT_SINGLE)
#define MatScalar real*4
#else
#define MatScalar Scalar
#endif
!
!     Declare PETSC_NULL_OBJECT
!
#define PETSC_NULL_OBJECT PETSC_NULL_INTEGER
!
!     PLogDouble variables are used to contain double precision numbers
!     that are not used in the numerical computations, but rather in logging,
!     timing etc.
!
#define PetscObject PetscFortranAddr
#define PLogDouble  PetscFortranDouble
!
!     Macros for error checking
!
#if defined(USE_PETSC_DEBUG)
#define SETERRA(n,p,s) call MPI_Abort(PETSC_COMM_WORLD,n)
#define CHKERRA(n) if (n .ne. 0) call MPI_Abort(PETSC_COMM_WORLD,n)
#else
#define SETERRA(n,p,s)
#define CHKERRA(n)
#endif

#endif
