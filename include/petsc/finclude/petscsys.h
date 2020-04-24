!
!
!  Part of the base include file for Fortran use of PETSc.
!  Note: This file should contain only define statements and
!  not the declaration of variables.

! No spaces for #defines as some compilers (PGI) also adds
! those additional spaces during preprocessing - bad for fixed format
!
#if !defined (PETSCSYSDEF_H)
#define PETSCSYSDEF_H
#include "petscconf.h"
#if defined (PETSC_HAVE_MPIUNI)
#include "petsc/mpiuni/mpiunifdef.h"
#endif
#include "petscversion.h"
#include "petsc/finclude/petscviewer.h"
#include "petsc/finclude/petsclog.h"
#include "petsc/finclude/petscbag.h"

!
! The real*8,complex*16 notatiton is used so that the
! PETSc double/complex variables are not affected by
! compiler options like -r4,-r8, sometimes invoked
! by the user. NAG compiler does not like integer*4,real*8

#define integer8 integer(kind=selected_int_kind(10))
#define integer4 integer(kind=selected_int_kind(5))
#define integer2 integer(kind=selected_int_kind(3))
#define integer1 integer(kind=selected_int_kind(1))
#define PetscBool  logical(kind=4)

#if (PETSC_SIZEOF_VOID_P == 8)
#define PetscOffset integer8
#define PetscFortranAddr integer8
#else
#define PetscOffset integer4
#define PetscFortranAddr integer4
#endif

#if defined(PETSC_USE_64BIT_INDICES)
#define PetscInt integer8
#else
#define PetscInt integer4
#endif
#define PetscInt64 integer8
#define PetscObjectState PetscInt64
#define PetscObjectId PetscInt64

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
#define MPI_Comm MPIUNI_FInt
#define MPI_Group MPIUNI_FInt
#define PetscMPIInt MPIUNI_FInt
#else
#define MPI_Comm integer4
#define MPI_Group integer4
#define PetscMPIInt integer4
#endif
!
#define PetscEnum PetscFortranInt
#define PetscErrorCode PetscFortranInt
#define PetscClassId PetscFortranInt
#define PetscLogEvent PetscFortranInt
#define PetscLogStage PetscFortranInt
#define PetscVoid PetscFortranAddr
!
#define PetscCopyMode PetscEnum
!
#define PetscDataType PetscEnum
#define PetscFPTrap PetscEnum
!
#define PetscFortranFloat real(kind=selected_real_kind(5))
#define PetscFortranDouble real(kind=selected_real_kind(10))
#define PetscFortranLongDouble real(kind=selected_real_kind(19))
#if defined(PETSC_USE_REAL_SINGLE)
#define PetscFortranComplex complex(kind=selected_real_kind(5))
#elif defined(PETSC_USE_REAL_DOUBLE)
#define PetscFortranComplex complex(kind=selected_real_kind(10))
#elif defined(PETSC_USE_REAL___FLOAT128)
#define PetscFortranComplex complex(kind=selected_real_kind(20))
#endif
#define PetscChar(a) character(len = a) ::

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
#define PetscIntToReal(a) real(a)
#elif defined(PETSC_USE_REAL___FLOAT128)
#define PETSC_REAL PETSC___FLOAT128
#define PetscIntToReal(a) dble(a)
#else
#define  PETSC_REAL  PETSC_DOUBLE
#define PetscIntToReal(a) dble(a)
#endif
!
!     Macro for templating between real and complex
!
#define PetscComplex PetscFortranComplex
#if defined(PETSC_USE_COMPLEX)
#define PetscScalar PetscFortranComplex
!
! F90 uses real(), conjg() when KIND parameter is used.
!
#define PetscRealPart(a) real(a)
#define PetscConj(a) conjg(a)
#define PetscImaginaryPart(a) aimag(a)
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
#define PetscImaginaryPart(a) 0.0
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
#define SETERRQ(c,ierr,s)  call PetscError(c,ierr,0,s); return
#define SETERRA(c,ierr,s)  call PetscError(c,ierr,0,s); call MPIU_Abort(c,ierr)
#define CHKERRQ(ierr) if (ierr .ne. 0) then;call PetscErrorF(ierr);return;endif
#define CHKERRA(ierr) if (ierr .ne. 0) then;call PetscErrorF(ierr);call MPIU_Abort(MPI_COMM_SELF,ierr);endif
#define CHKMEMQ call chkmemfortran(__LINE__,__FILE__,ierr)

#define PetscMatlabEngine PetscFortranAddr

#if !defined(PetscFlush)
#if defined(PETSC_HAVE_FORTRAN_FLUSH)
#define PetscFlush(a)    flush(a)
#elif defined(PETSC_HAVE_FORTRAN_FLUSH_)
#define PetscFlush(a)    flush_(a)
#else
#define PetscFlush(a)
#endif
#endif

#define PetscRandom type(tPetscRandom)
#define PetscRandomType character*(80)
#define PetscBinarySeekType PetscEnum

#define PetscBuildTwoSidedType PetscEnum

#define PetscSubcomm type(tPetscSubcomm)
#define PetscSubcommType PetscEnum

#define PetscOptions type(tPetscOptions)

#define PetscFunctionList PetscFortranAddr

#define PetscInfoCommFlag PetscEnum
#endif
