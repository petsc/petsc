!
!  Manually maintained part of the base include file for Fortran use of PETSc.
!  Note: This file should contain only define statements
!
#if !defined (PETSCSYSBASEDEF_H)
#define PETSCSYSBASEDEF_H
#include "petscconf.h"
#if defined (PETSC_HAVE_MPIUNI)
#include "petsc/mpiuni/mpiunifdef.h"
#endif
#include "petscversion.h"

!
#define integer8 integer(kind=C_INT64_T)
#define integer4 integer(kind=C_INT32_T)
#define integer2 integer(kind=C_INT16_T)
#define integer1 integer(kind=C_INT8_T)
#define PetscBool logical(kind=C_BOOL)

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

#if defined(PETSC_USE_64BIT_BLAS_INDICES)
#define PetscBLASInt integer8
#else
#define PetscBLASInt integer4
#endif
#define PetscCuBLASInt integer4
#define PetscHipBLASInt integer4

!
#define PetscSizeT integer(kind=C_SIZE_T)
!
#define MPI_Comm integer4
#define MPI_Group integer4
!
#define PetscEnum integer4
#define PetscVoid PetscFortranAddr
!
#define PetscFortranFloat real(kind=C_FLOAT)
#define PetscFortranDouble real(kind=C_DOUBLE)
#define PetscFortranLongDouble real(kind=C_FLOAT128)
#if defined(PETSC_USE_REAL_SINGLE)
#define PetscComplex complex(kind=C_FLOAT_COMPLEX)
#elif defined(PETSC_USE_REAL_DOUBLE)
#define PetscComplex complex(kind=C_DOUBLE_COMPLEX)
#elif defined(PETSC_USE_REAL___FLOAT128)
#define PetscComplex complex(kind=C_FLOAT128_COMPLEX)
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
#if defined(PETSC_USE_COMPLEX)
#define PetscScalar PetscComplex
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

#define PetscReal2d type(tPetscReal2d)

#define PETSC_FORTRAN_TYPE_INITIALIZE -2
#define PetscObjectIsNull(obj) (obj%v == 0 .or. obj%v ==  PETSC_FORTRAN_TYPE_INITIALIZE .or. obj%v == -3)
#define PetscObjectNullify(obj) obj%v = PETSC_FORTRAN_TYPE_INITIALIZE
!
!     Macros for error checking
!
#define SETERRQ(c, ierr, s)  call PetscError(c, ierr, PETSC_ERROR_INITIAL, s); return
#define SETERRA(c, ierr, s)  call PetscError(c, ierr, PETSC_ERROR_INITIAL, s); call MPIU_Abort(c, ierr)
#if defined(PETSC_HAVE_FORTRAN_FREE_LINE_LENGTH_NONE)
#define CHKERRQ(ierr) if (ierr .ne. 0) then;call PetscErrorF(ierr,__LINE__,__FILE__);return;endif
#define CHKERRA(ierr) if (ierr .ne. 0) then;call PetscErrorF(ierr,__LINE__,__FILE__);call MPIU_Abort(PETSC_COMM_SELF,ierr);endif
#define CHKERRMPI(ierr) if (ierr .ne. 0) then;call PetscErrorMPI(ierr,__LINE__,__FILE__);return;endif
#define CHKERRMPIA(ierr) if (ierr .ne. 0) then;call PetscErrorMPI(ierr,__LINE__,__FILE__);call MPIU_Abort(PETSC_COMM_SELF,ierr);endif
#else
#define CHKERRQ(ierr) if (ierr .ne. 0) then;call PetscErrorF(ierr);return;endif
#define CHKERRA(ierr) if (ierr .ne. 0) then;call PetscErrorF(ierr);call MPIU_Abort(PETSC_COMM_SELF,ierr);endif
#define CHKERRMPI(ierr) if (ierr .ne. 0) then;call PetscErrorMPI(ierr);return;endif
#define CHKERRMPIA(ierr) if (ierr .ne. 0) then;call PetscErrorMPI(ierr);call MPIU_Abort(PETSC_COMM_SELF,ierr);endif
#endif
#define CHKMEMQ call chkmemfortran(__LINE__,__FILE__,ierr)
#define PetscCall(func) call func; CHKERRQ(ierr)
#define PetscCallMPI(func) call func; CHKERRMPI(ierr)
#define PetscCallA(func) call func; CHKERRA(ierr)
#define PetscCallMPIA(func) call func; CHKERRMPIA(ierr)
#define PetscCheckA(err, c, ierr, s) if (.not.(err)) then; SETERRA(c, ierr, s); endif
#define PetscCheck(err, c, ierr, s) if (.not.(err)) then; SETERRQ(c, ierr, s); endif

#if !defined(PetscFlush)
#if defined(PETSC_HAVE_FORTRAN_FLUSH)
#define PetscFlush(a)    flush(a)
#elif defined(PETSC_HAVE_FORTRAN_FLUSH_)
#define PetscFlush(a)    flush_(a)
#else
#define PetscFlush(a)
#endif
#endif

#define PetscEnumCase(e) case(e%v)

#define PetscObjectSpecificCast(sp,ob) sp%v = ob%v

#endif
