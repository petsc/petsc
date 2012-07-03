!
!
!  Part of the base include file for Fortran use of PETSc.
!  Note: This file should contain only define statements and
!  not the declaration of variables.

! No spaces for #defines as some compilers (PGI) also adds
! those additional spaces during preprocessing - bad for fixed format
!
#if !defined (__PETSCSYSDEF_H)
#define __PETSCSYSDEF_H
#include "petscconf.h"
#include "finclude/petscviewerdef.h"
#include "finclude/petscerrordef.h"
#include "finclude/petsclogdef.h"
#include "finclude/petscdrawdef.h"

!
! The real*8,complex*16 notatiton is used so that the 
! PETSc double/complex variables are not affected by 
! compiler options like -r4,-r8, sometimes invoked 
! by the user. NAG compiler does not like integer*4,real*8

#if defined(PETSC_USE_FORTRANKIND)
#define integer8 integer(kind=selected_int_kind(10))
#define integer4 integer(kind=selected_int_kind(5))
#define PetscBool  logical(kind=4)
#else
#define integer8 integer*8
#define integer4 integer*4
#define PetscBool  logical*4
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

#if (PETSC_SIZEOF_INT == 4)
#define PetscFortranInt integer4
#elif (PETSC_SIZEOF_INT == 8)
#define PetscFortranInt integer8
#endif
!
#if (PETSC_SIZEOF_SIZE_T == 8)
#define PetscSizeT integer8
#else
#define PetscSizeT integer4
#endif
!
#if defined(PETSC_HAVE_MPIUNI)
#define MPI_Comm PetscFortranInt
#define MPI_Group PetscFortranInt
#define PetscMPIInt PetscFortranInt
#else
#define MPI_Comm integer
#define MPI_Group integer
#define PetscMPIInt integer
#endif
!
#define PetscEnum PetscFortranInt
#define PetscErrorCode PetscFortranInt
#define PetscClassId PetscFortranInt
#define PetscLogEvent PetscFortranInt
#define PetscLogStage PetscFortranInt
#define PetscVoid PetscFortranAddr
!
#if defined(PETSC_FORTRAN_PETSCTRUTH_INT)
#undef PetscBool 
#define PetscBool  PetscEnum
#endif
!
#define PetscCopyMode PetscEnum
!
#define PetscDataType PetscEnum
#define PetscFPTrap PetscEnum
!
#if defined (PETSC_USE_FORTRANKIND)
#define PetscFortranFloat real(kind=selected_real_kind(5))
#define PetscFortranDouble real(kind=selected_real_kind(10))
#define PetscFortranLongDouble real(kind=selected_real_kind(19))
#if defined(PETSC_USE_REAL_SINGLE)
#define PetscFortranComplex complex(kind=selected_real_kind(5))
#else
#define PetscFortranComplex complex(kind=selected_real_kind(10))
#endif
#define PetscChar(a) character(len = a) ::
#else
#define PetscFortranFloat real*4
#define PetscFortranDouble real*8
#define PetscFortranLongDouble real*16
#if defined(PETSC_USE_REAL_SINGLE)
#define PetscFortranComplex complex*8
#else
#define PetscFortranComplex complex*16
#endif
#define PetscChar(a) character*(a)
#endif

#if defined(PETSC_USE_COMPLEX)
#define PETSC_SCALAR PETSC_COMPLEX
#else
#if defined(PETSC_USE_REAL_SINGLE)
#define PETSC_SCALAR PETSC_FLOAT
#elif defined(PETSC_USE_REAL___FLOAT128)
#define PETSC_SCALAR PETSC___FLOAT128
#else
#define PETSC_SCALAR PETSC_DOUBLE
#endif
#endif
#if defined(PETSC_USE_REAL_SINGLE)
#define  PETSC_REAL  PETSC_FLOAT
#elif defined(PETSC_USE_REAL___FLOAT128)
#define PETSC_REAL PETSC___FLOAT128
#else
#define  PETSC_REAL  PETSC_DOUBLE
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
#define PetscImaginaryPart(a) aimag(a)
#else
#define PetscRealPart(a) dreal(a)
#define PetscConj(a) dconjg(a)
#define PetscImaginaryPart(a) daimag(a)
#endif
#else
#if defined (PETSC_USE_REAL_SINGLE)
#define PetscScalar PetscFortranFloat
#elif defined(PETSC_USE_REAL___FLOAT128)
#define PetscScalar PetscFortranLongDouble
#elif defined(PETSC_USE_REAL_DOUBLE)
#define PetscScalar PetscFortranDouble
#endif
#define PetscRealPart(a) a
#define PetscConj(a) a
#define PetscImaginaryPart(a) a
#endif

#if defined (PETSC_USE_REAL_SINGLE)
#define PetscReal PetscFortranFloat
#elif defined(PETSC_USE_REAL___FLOAT128)
#define PetscReal PetscFortranLongDouble
#elif defined(PETSC_USE_REAL_DOUBLE)
#define PetscReal PetscFortranDouble
#endif

!
!    Allows the matrix Fortran Kernels to work with single precision
!    matrix data structures
!
#define MatScalar PetscScalar 
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
#if defined(PETSC_USE_ERRORCHECKING)
#define SETERRQ(c,n,s,ierr) call MPI_Abort(PETSC_COMM_WORLD,n,ierr)
#define CHKERRQ(n) if (n .ne. 0) call MPI_Abort(PETSC_COMM_WORLD,n,n)
#define CHKMEMQ call chkmemfortran(__LINE__,__FILE__,ierr)
#else
#define SETERRQ(c,n,s,ierr)
#define CHKERRQ(n)
#define CHKMEMQ
#endif

#define PetscMatlabEngine PetscFortranAddr

#if !defined(PetscFlush)
#if defined(PETSC_HAVE_FLUSH)
#define PetscFlush(a)    call flush(a)
#elif defined(PETSC_HAVE_FLUSH_)
#define PetscFlush(a)    call flush_(a)
#else
#define PetscFlush(a)
#endif
#endif

#define PetscRandom PetscFortranAddr
#define PetscRandomType character*(80)
#define PetscBinarySeekType PetscEnum

#define PetscSF PetscFortranAddr

#endif
