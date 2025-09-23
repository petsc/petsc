module petscmpi
  use, intrinsic :: ISO_C_binding
#include <petscconf.h>
#include "petsc/finclude/petscsys.h"
#if defined(PETSC_HAVE_MPIUNI)
  use mpiuni
#else
#if defined(PETSC_HAVE_MPI_F90MODULE)
  use mpi
#else
#include "mpif.h"
#endif
#endif

  public:: MPIU_REAL, MPIU_SUM, MPIU_SCALAR, MPIU_INTEGER
  public:: PETSC_COMM_WORLD, PETSC_COMM_SELF

  integer4 :: MPIU_REAL
  integer4 :: MPIU_SUM
  integer4 :: MPIU_SCALAR
  integer4 :: MPIU_INTEGER

  MPI_Comm::PETSC_COMM_WORLD = 0
  MPI_Comm::PETSC_COMM_SELF = 0

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::MPIU_REAL
!DEC$ ATTRIBUTES DLLEXPORT::MPIU_SUM
!DEC$ ATTRIBUTES DLLEXPORT::MPIU_SCALAR
!DEC$ ATTRIBUTES DLLEXPORT::MPIU_INTEGER
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_COMM_SELF
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_COMM_WORLD
#endif
end module petscmpi

! ------------------------------------------------------------------------
module petscsysdef
  use, intrinsic :: ISO_C_binding
#if defined(PETSC_HAVE_MPI_F90MODULE_VISIBILITY)
  use petscmpi
#else
  use petscmpi, only: MPIU_REAL, MPIU_SUM, MPIU_SCALAR, MPIU_INTEGER, PETSC_COMM_WORLD, PETSC_COMM_SELF
#endif
  PetscReal, parameter :: PetscReal_Private = 1.0
  Integer, parameter   :: PETSC_REAL_KIND = kind(PetscReal_Private)

  PetscBool, parameter :: PETSC_TRUE = .true._C_BOOL
  PetscBool, parameter :: PETSC_FALSE = .false._C_BOOL

  PetscInt, parameter :: PETSC_DECIDE = -1
  PetscInt, parameter :: PETSC_DECIDE_INTEGER = -1
  PetscReal, parameter :: PETSC_DECIDE_REAL = -1.0_PETSC_REAL_KIND
#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_DECIDE
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_DECIDE_INTEGER
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_DECIDE_REAL
#endif

  PetscInt, parameter :: PETSC_DETERMINE = -1
  PetscInt, parameter :: PETSC_DETERMINE_INTEGER = -1
  PetscReal, parameter :: PETSC_DETERMINE_REAL = -1.0_PETSC_REAL_KIND
#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_DETERMINE
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_DETERMINE_INTEGER
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_DETERMINE_REAL
#endif

  PetscInt, parameter :: PETSC_CURRENT = -2
  PetscInt, parameter :: PETSC_CURRENT_INTEGER = -2
  PetscReal, parameter :: PETSC_CURRENT_REAL = -2.0_PETSC_REAL_KIND
#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_CURRENT
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_CURRENT_INTEGER
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_CURRENT_REAL
#endif

  PetscInt, parameter :: PETSC_DEFAULT = -2
  PetscInt, parameter :: PETSC_DEFAULT_INTEGER = -2
  PetscReal, parameter :: PETSC_DEFAULT_REAL = -2.0_PETSC_REAL_KIND
#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_DEFAULT
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_DEFAULT_INTEGER
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_DEFAULT_REAL
#endif

  PetscFortranAddr, parameter :: PETSC_STDOUT = 0
!
!  PETSc DataTypes
!
#if defined(PETSC_USE_REAL_SINGLE)
#define PETSC_REAL PETSC_FLOAT
#elif defined(PETSC_USE_REAL___FLOAT128)
#define PETSC_REAL PETSC___FLOAT128
#else
#define PETSC_REAL PETSC_DOUBLE
#endif
#define PETSC_FORTRANADDR PETSC_LONG

! PETSc mathematics include file. Defines certain basic mathematical
! constants and functions for working with single and double precision
! floating point numbers as well as complex and integers.
!
! Representation of complex i
  PetscComplex, parameter :: PETSC_i = (0.0_PETSC_REAL_KIND, 1.0_PETSC_REAL_KIND)

! A PETSC_NULL_FUNCTION pointer
!
  external PETSC_NULL_FUNCTION
!
! Possible arguments to PetscPushErrorHandler()
!
  external PETSCTRACEBACKERRORHANDLER
  external PETSCABORTERRORHANDLER
  external PETSCEMACSCLIENTERRORHANDLER
  external PETSCATTACHDEBUGGERERRORHANDLER
  external PETSCIGNOREERRORHANDLER
!
  external PetscIsInfOrNanScalar
  external PetscIsInfOrNanReal
  PetscBool PetscIsInfOrNanScalar
  PetscBool PetscIsInfOrNanReal

#include <../ftn/sys/petscall.h>

  PetscViewer, parameter :: PETSC_VIEWER_STDOUT_SELF = tPetscViewer(9)
  PetscViewer, parameter :: PETSC_VIEWER_DRAW_WORLD = tPetscViewer(4)
  PetscViewer, parameter :: PETSC_VIEWER_DRAW_SELF = tPetscViewer(5)
  PetscViewer, parameter :: PETSC_VIEWER_SOCKET_WORLD = tPetscViewer(6)
  PetscViewer, parameter :: PETSC_VIEWER_SOCKET_SELF = tPetscViewer(7)
  PetscViewer, parameter :: PETSC_VIEWER_STDOUT_WORLD = tPetscViewer(8)
  PetscViewer, parameter :: PETSC_VIEWER_STDERR_WORLD = tPetscViewer(10)
  PetscViewer, parameter :: PETSC_VIEWER_STDERR_SELF = tPetscViewer(11)
  PetscViewer, parameter :: PETSC_VIEWER_BINARY_WORLD = tPetscViewer(12)
  PetscViewer, parameter :: PETSC_VIEWER_BINARY_SELF = tPetscViewer(13)
  PetscViewer, parameter :: PETSC_VIEWER_MATLAB_WORLD = tPetscViewer(14)
  PetscViewer, parameter :: PETSC_VIEWER_MATLAB_SELF = tPetscViewer(15)

  PetscViewer PETSC_VIEWER_STDOUT_
  PetscViewer PETSC_VIEWER_DRAW_
  external PETSC_VIEWER_STDOUT_
  external PETSC_VIEWER_DRAW_
  external PetscViewerAndFormatDestroy

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_VIEWER_STDOUT_SELF
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_VIEWER_DRAW_WORLD
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_VIEWER_DRAW_SELF
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_VIEWER_SOCKET_WORLD
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_VIEWER_SOCKET_SELF
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_VIEWER_STDOUT_WORLD
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_VIEWER_STDERR_WORLD
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_VIEWER_STDERR_SELF
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_VIEWER_BINARY_WORLD
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_VIEWER_BINARY_SELF
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_VIEWER_MATLAB_WORLD
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_VIEWER_MATLAB_SELF
#endif

  PetscErrorCode, parameter :: PETSC_ERR_MEM = 55
  PetscErrorCode, parameter :: PETSC_ERR_SUP = 56
  PetscErrorCode, parameter :: PETSC_ERR_SUP_SYS = 57
  PetscErrorCode, parameter :: PETSC_ERR_ORDER = 58
  PetscErrorCode, parameter :: PETSC_ERR_SIG = 59
  PetscErrorCode, parameter :: PETSC_ERR_FP = 72
  PetscErrorCode, parameter :: PETSC_ERR_COR = 74
  PetscErrorCode, parameter :: PETSC_ERR_LIB = 76
  PetscErrorCode, parameter :: PETSC_ERR_PLIB = 77
  PetscErrorCode, parameter :: PETSC_ERR_MEMC = 78
  PetscErrorCode, parameter :: PETSC_ERR_CONV_FAILED = 82
  PetscErrorCode, parameter :: PETSC_ERR_USER = 83
  PetscErrorCode, parameter :: PETSC_ERR_SYS = 88
  PetscErrorCode, parameter :: PETSC_ERR_POINTER = 70
  PetscErrorCode, parameter :: PETSC_ERR_MPI_LIB_INCOMP = 87

  PetscErrorCode, parameter :: PETSC_ERR_ARG_SIZ = 60
  PetscErrorCode, parameter :: PETSC_ERR_ARG_IDN = 61
  PetscErrorCode, parameter :: PETSC_ERR_ARG_WRONG = 62
  PetscErrorCode, parameter :: PETSC_ERR_ARG_CORRUPT = 64
  PetscErrorCode, parameter :: PETSC_ERR_ARG_OUTOFRANGE = 63
  PetscErrorCode, parameter :: PETSC_ERR_ARG_BADPTR = 68
  PetscErrorCode, parameter :: PETSC_ERR_ARG_NOTSAMETYPE = 69
  PetscErrorCode, parameter :: PETSC_ERR_ARG_NOTSAMECOMM = 80
  PetscErrorCode, parameter :: PETSC_ERR_ARG_WRONGSTATE = 73
  PetscErrorCode, parameter :: PETSC_ERR_ARG_TYPENOTSET = 89
  PetscErrorCode, parameter :: PETSC_ERR_ARG_INCOMP = 75
  PetscErrorCode, parameter :: PETSC_ERR_ARG_NULL = 85
  PetscErrorCode, parameter :: PETSC_ERR_ARG_UNKNOWN_TYPE = 86

  PetscErrorCode, parameter :: PETSC_ERR_FILE_OPEN = 65
  PetscErrorCode, parameter :: PETSC_ERR_FILE_READ = 66
  PetscErrorCode, parameter :: PETSC_ERR_FILE_WRITE = 67
  PetscErrorCode, parameter :: PETSC_ERR_FILE_UNEXPECTED = 79

  PetscErrorCode, parameter :: PETSC_ERR_MAT_LU_ZRPVT = 71
  PetscErrorCode, parameter :: PETSC_ERR_MAT_CH_ZRPVT = 81

  PetscErrorCode, parameter :: PETSC_ERR_INT_OVERFLOW = 84

  PetscErrorCode, parameter :: PETSC_ERR_FLOP_COUNT = 90
  PetscErrorCode, parameter :: PETSC_ERR_NOT_CONVERGED = 91
  PetscErrorCode, parameter :: PETSC_ERR_MISSING_FACTOR = 92
  PetscErrorCode, parameter :: PETSC_ERR_OPT_OVERWRITE = 93
  PetscErrorCode, parameter :: PETSC_ERR_WRONG_MPI_SIZE = 94
  PetscErrorCode, parameter :: PETSC_ERR_USER_INPUT = 95
  PetscErrorCode, parameter :: PETSC_ERR_GPU_RESOURCE = 96
  PetscErrorCode, parameter :: PETSC_ERR_GPU = 97
  PetscErrorCode, parameter :: PETSC_ERR_MPI = 98
  PetscErrorCode, parameter :: PETSC_ERR_RETURN = 99

  character(len=80) :: PETSC_NULL_CHARACTER = ''
  PetscInt PETSC_NULL_INTEGER, PETSC_NULL_INTEGER_ARRAY(1)
  PetscInt, pointer :: PETSC_NULL_INTEGER_POINTER(:)
  PetscScalar, pointer :: PETSC_NULL_SCALAR_POINTER(:)
  PetscFortranDouble PETSC_NULL_DOUBLE
  PetscScalar PETSC_NULL_SCALAR, PETSC_NULL_SCALAR_ARRAY(1)
  PetscReal PETSC_NULL_REAL, PETSC_NULL_REAL_ARRAY(1)
  PetscReal, pointer :: PETSC_NULL_REAL_POINTER(:)
  PetscBool PETSC_NULL_BOOL
  PetscEnum PETSC_NULL_ENUM
  MPI_Comm PETSC_NULL_MPI_COMM
!
!     Basic math constants
!
  PetscReal PETSC_PI
  PetscReal PETSC_MAX_REAL
  PetscReal PETSC_MIN_REAL
  PetscReal PETSC_MACHINE_EPSILON
  PetscReal PETSC_SQRT_MACHINE_EPSILON
  PetscReal PETSC_SMALL
  PetscReal PETSC_INFINITY
  PetscReal PETSC_NINFINITY

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_CHARACTER
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_INTEGER
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_INTEGER_ARRAY
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_INTEGER_POINTER
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_SCALAR_POINTER
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_REAL_POINTER
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_DOUBLE
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_SCALAR
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_SCALAR_ARRAY
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_REAL
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_REAL_ARRAY
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_BOOL
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_ENUM
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_MPI_COMM
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_PI
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_MAX_REAL
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_MIN_REAL
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_MACHINE_EPSILON
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_SQRT_MACHINE_EPSILON
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_SMALL
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_INFINITY
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NINFINITY
#endif

  type tPetscReal2d
    sequence
    PetscReal, dimension(:), pointer :: ptr
  end type tPetscReal2D

end module petscsysdef

!   ------------------------------------------------------------------------

module petscsys
  use, intrinsic :: ISO_C_binding
  use petscsysdef

#include <../src/sys/ftn-mod/petscsys.h90>
#include <../src/sys/ftn-mod/petscviewer.h90>
#include <../ftn/sys/petscall.h90>

  interface PetscInitialize
    module procedure PetscInitializeWithHelp, PetscInitializeNoHelp, PetscInitializeNoArguments
  end interface PetscInitialize

  interface
    subroutine PetscSetFortranBasePointers( &
      PETSC_NULL_CHARACTER, &
      PETSC_NULL_INTEGER, PETSC_NULL_SCALAR, &
      PETSC_NULL_DOUBLE, PETSC_NULL_REAL, &
      PETSC_NULL_BOOL, PETSC_NULL_ENUM, PETSC_NULL_FUNCTION, &
      PETSC_NULL_MPI_COMM, &
      PETSC_NULL_INTEGER_ARRAY, PETSC_NULL_SCALAR_ARRAY, &
      PETSC_NULL_REAL_ARRAY, APETSC_NULL_INTEGER_POINTER, &
      PETSC_NULL_SCALAR_POINTER, PETSC_NULL_REAL_POINTER)
      use, intrinsic :: ISO_C_binding
      character(*) PETSC_NULL_CHARACTER
      PetscInt PETSC_NULL_INTEGER
      PetscScalar PETSC_NULL_SCALAR
      PetscFortranDouble PETSC_NULL_DOUBLE
      PetscReal PETSC_NULL_REAL
      PetscBool PETSC_NULL_BOOL
      PetscEnum PETSC_NULL_ENUM
      external PETSC_NULL_FUNCTION
      MPI_Comm PETSC_NULL_MPI_COMM
      PetscInt PETSC_NULL_INTEGER_ARRAY(*)
      PetscScalar PETSC_NULL_SCALAR_ARRAY(*)
      PetscReal PETSC_NULL_REAL_ARRAY(*)
      PetscInt, pointer :: APETSC_NULL_INTEGER_POINTER(:)
      PetscScalar, pointer :: PETSC_NULL_SCALAR_POINTER(:)
      PetscReal, pointer :: PETSC_NULL_REAL_POINTER(:)
    end subroutine PetscSetFortranBasePointers

    subroutine PetscOptionsString(string, text, man, default, value, flg, ierr)
      use, intrinsic :: ISO_C_binding
      character(*) string, text, man, default, value
      PetscBool flg
      PetscErrorCode ierr
    end subroutine PetscOptionsString
  end interface

  interface petscbinaryread
    subroutine petscbinaryreadcomplex(fd, data, num, count, type, z)
      use, intrinsic :: ISO_C_binding
      import ePetscDataType
      integer4 fd
      PetscComplex data(*)
      PetscInt num
      PetscInt count
      PetscDataType type
      PetscErrorCode z
    end subroutine petscbinaryreadcomplex
    subroutine petscbinaryreadreal(fd, data, num, count, type, z)
      use, intrinsic :: ISO_C_binding
      import ePetscDataType
      integer4 fd
      PetscReal data(*)
      PetscInt num
      PetscInt count
      PetscDataType type
      PetscErrorCode z
    end subroutine petscbinaryreadreal
    subroutine petscbinaryreadint(fd, data, num, count, type, z)
      use, intrinsic :: ISO_C_binding
      import ePetscDataType
      integer4 fd
      PetscInt data(*)
      PetscInt num
      PetscInt count
      PetscDataType type
      PetscErrorCode z
    end subroutine petscbinaryreadint
    subroutine petscbinaryreadcomplex1(fd, data, num, count, type, z)
      use, intrinsic :: ISO_C_binding
      import ePetscDataType
      integer4 fd
      PetscComplex data
      PetscInt num
      PetscInt count
      PetscDataType type
      PetscErrorCode z
    end subroutine petscbinaryreadcomplex1
    subroutine petscbinaryreadreal1(fd, data, num, count, type, z)
      use, intrinsic :: ISO_C_binding
      import ePetscDataType
      integer4 fd
      PetscReal data
      PetscInt num
      PetscInt count
      PetscDataType type
      PetscErrorCode z
    end subroutine petscbinaryreadreal1
    subroutine petscbinaryreadint1(fd, data, num, count, type, z)
      use, intrinsic :: ISO_C_binding
      import ePetscDataType
      integer4 fd
      PetscInt data
      PetscInt num
      PetscInt count
      PetscDataType type
      PetscErrorCode z
    end subroutine petscbinaryreadint1
    subroutine petscbinaryreadcomplexcnt(fd, data, num, count, type, z)
      use, intrinsic :: ISO_C_binding
      import ePetscDataType
      integer4 fd
      PetscComplex data(*)
      PetscInt num
      PetscInt count(1)
      PetscDataType type
      PetscErrorCode z
    end subroutine petscbinaryreadcomplexcnt
    subroutine petscbinaryreadrealcnt(fd, data, num, count, type, z)
      use, intrinsic :: ISO_C_binding
      import ePetscDataType
      integer4 fd
      PetscReal data(*)
      PetscInt num
      PetscInt count(1)
      PetscDataType type
      PetscErrorCode z
    end subroutine petscbinaryreadrealcnt
    subroutine petscbinaryreadintcnt(fd, data, num, count, type, z)
      use, intrinsic :: ISO_C_binding
      import ePetscDataType
      integer4 fd
      PetscInt data(*)
      PetscInt num
      PetscInt count(1)
      PetscDataType type
      PetscErrorCode z
    end subroutine petscbinaryreadintcnt
    subroutine petscbinaryreadcomplex1cnt(fd, data, num, count, type, z)
      use, intrinsic :: ISO_C_binding
      import ePetscDataType
      integer4 fd
      PetscComplex data
      PetscInt num
      PetscInt count(1)
      PetscDataType type
      PetscErrorCode z
    end subroutine petscbinaryreadcomplex1cnt
    subroutine petscbinaryreadreal1cnt(fd, data, num, count, type, z)
      use, intrinsic :: ISO_C_binding
      import ePetscDataType
      integer4 fd
      PetscReal data
      PetscInt num
      PetscInt count(1)
      PetscDataType type
      PetscErrorCode z
    end subroutine petscbinaryreadreal1cnt
    subroutine petscbinaryreadint1cnt(fd, data, num, count, type, z)
      use, intrinsic :: ISO_C_binding
      import ePetscDataType
      integer4 fd
      PetscInt data
      PetscInt num
      PetscInt count(1)
      PetscDataType type
      PetscErrorCode z
    end subroutine petscbinaryreadint1cnt
  end interface petscbinaryread

  interface petscbinarywrite
    subroutine petscbinarywritecomplex(fd, data, num, type, z)
      use, intrinsic :: ISO_C_binding
      import ePetscDataType
      integer4 fd
      PetscComplex data(*)
      PetscInt num
      PetscDataType type
      PetscErrorCode z
    end subroutine petscbinarywritecomplex
    subroutine petscbinarywritereal(fd, data, num, type, z)
      use, intrinsic :: ISO_C_binding
      import ePetscDataType
      integer4 fd
      PetscReal data(*)
      PetscInt num
      PetscDataType type
      PetscErrorCode z
    end subroutine petscbinarywritereal
    subroutine petscbinarywriteint(fd, data, num, type, z)
      use, intrinsic :: ISO_C_binding
      import ePetscDataType
      integer4 fd
      PetscInt data(*)
      PetscInt num
      PetscDataType type
      PetscErrorCode z
    end subroutine petscbinarywriteint
    subroutine petscbinarywritecomplex1(fd, data, num, type, z)
      use, intrinsic :: ISO_C_binding
      import ePetscDataType
      integer4 fd
      PetscComplex data
      PetscInt num
      PetscDataType type
      PetscErrorCode z
    end subroutine petscbinarywritecomplex1
    subroutine petscbinarywritereal1(fd, data, num, type, z)
      use, intrinsic :: ISO_C_binding
      import ePetscDataType
      integer4 fd
      PetscReal data
      PetscInt num
      PetscDataType type
      PetscErrorCode z
    end subroutine petscbinarywritereal1
    subroutine petscbinarywriteint1(fd, data, num, type, z)
      use, intrinsic :: ISO_C_binding
      import ePetscDataType
      integer4 fd
      PetscInt data
      PetscInt num
      PetscDataType type
      PetscErrorCode z
    end subroutine petscbinarywriteint1
  end interface petscbinarywrite

contains
#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::PetscInitializeWithHelp
#endif
  subroutine PetscInitializeWithHelp(filename, help, ierr)
    character(len=*) :: filename
    character(len=*) :: help
    PetscErrorCode   :: ierr

    if (filename /= PETSC_NULL_CHARACTER) then
      call PetscInitializeF(trim(filename), help, ierr)
      CHKERRQ(ierr)
    else
      call PetscInitializeF(filename, help, ierr)
      CHKERRQ(ierr)
    end if
  end subroutine PetscInitializeWithHelp

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::PetscInitializeNoHelp
#endif
  subroutine PetscInitializeNoHelp(filename, ierr)
    character(len=*) :: filename
    PetscErrorCode   :: ierr

    if (filename /= PETSC_NULL_CHARACTER) then
      call PetscInitializeF(trim(filename), PETSC_NULL_CHARACTER, ierr)
      CHKERRQ(ierr)
    else
      call PetscInitializeF(filename, PETSC_NULL_CHARACTER, ierr)
      CHKERRQ(ierr)
    end if
  end subroutine PetscInitializeNoHelp

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::PetscInitializeNoArguments
#endif
  subroutine PetscInitializeNoArguments(ierr)
    PetscErrorCode :: ierr

    call PetscInitializeF(PETSC_NULL_CHARACTER, PETSC_NULL_CHARACTER, ierr)
    CHKERRQ(ierr)
  end subroutine PetscInitializeNoArguments

#include <../ftn/sys/petscall.hf90>
end module petscsys

subroutine F90ArraySetRealPointer(array, sz, j, T)
  use petscsysdef
  PetscInt :: j, sz
  PetscReal, target    :: array(1:sz)
  PetscReal2d, pointer :: T(:)

  T(j + 1)%ptr => array
end subroutine F90ArraySetRealPointer
#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT:: F90ArraySetRealPointer
#endif

!------------------------------------------------------------------------
!TODO: generate the modules below by looping over
!      ftn/sys/XXX.h90
!      and skipping those in petscall.h

module petscbag
  use petscsys
#include <../include/petsc/finclude/petscbag.h>
#include <../ftn/sys/petscbag.h>
#include <../ftn/sys/petscbag.h90>
contains
#include <../ftn/sys/petscbag.hf90>
end module petscbag

!------------------------------------------------------------------------
module petscbm
  use petscsys
#include <../include/petsc/finclude/petscbm.h>
#include <../ftn/sys/petscbm.h>
#include <../ftn/sys/petscbm.h90>
contains

#include <../ftn/sys/petscbm.hf90>
end module petscbm

!------------------------------------------------------------------------
module petscmatlab
  use petscsys
#include <../include/petsc/finclude/petscmatlab.h>
#include <../ftn/sys/petscmatlab.h>
#include <../ftn/sys/petscmatlab.h90>

contains

#include <../ftn/sys/petscmatlab.hf90>
end module petscmatlab

!------------------------------------------------------------------------
module petscdraw
  use petscsys
#include <../include/petsc/finclude/petscdraw.h>
#include <../ftn/sys/petscdraw.h>
#include <../ftn/sys/petscdraw.h90>

  PetscEnum, parameter :: PETSC_DRAW_BASIC_COLORS = 33
  PetscEnum, parameter :: PETSC_DRAW_ROTATE = -1
  PetscEnum, parameter :: PETSC_DRAW_WHITE = 0
  PetscEnum, parameter :: PETSC_DRAW_BLACK = 1
  PetscEnum, parameter :: PETSC_DRAW_RED = 2
  PetscEnum, parameter :: PETSC_DRAW_GREEN = 3
  PetscEnum, parameter :: PETSC_DRAW_CYAN = 4
  PetscEnum, parameter :: PETSC_DRAW_BLUE = 5
  PetscEnum, parameter :: PETSC_DRAW_MAGENTA = 6
  PetscEnum, parameter :: PETSC_DRAW_AQUAMARINE = 7
  PetscEnum, parameter :: PETSC_DRAW_FORESTGREEN = 8
  PetscEnum, parameter :: PETSC_DRAW_ORANGE = 9
  PetscEnum, parameter :: PETSC_DRAW_VIOLET = 10
  PetscEnum, parameter :: PETSC_DRAW_BROWN = 11
  PetscEnum, parameter :: PETSC_DRAW_PINK = 12
  PetscEnum, parameter :: PETSC_DRAW_CORAL = 13
  PetscEnum, parameter :: PETSC_DRAW_GRAY = 14
  PetscEnum, parameter :: PETSC_DRAW_YELLOW = 15
  PetscEnum, parameter :: PETSC_DRAW_GOLD = 16
  PetscEnum, parameter :: PETSC_DRAW_LIGHTPINK = 17
  PetscEnum, parameter :: PETSC_DRAW_MEDIUMTURQUOISE = 18
  PetscEnum, parameter :: PETSC_DRAW_KHAKI = 19
  PetscEnum, parameter :: PETSC_DRAW_DIMGRAY = 20
  PetscEnum, parameter :: PETSC_DRAW_YELLOWGREEN = 21
  PetscEnum, parameter :: PETSC_DRAW_SKYBLUE = 22
  PetscEnum, parameter :: PETSC_DRAW_DARKGREEN = 23
  PetscEnum, parameter :: PETSC_DRAW_NAVYBLUE = 24
  PetscEnum, parameter :: PETSC_DRAW_SANDYBROWN = 25
  PetscEnum, parameter :: PETSC_DRAW_CADETBLUE = 26
  PetscEnum, parameter :: PETSC_DRAW_POWDERBLUE = 27
  PetscEnum, parameter :: PETSC_DRAW_DEEPPINK = 28
  PetscEnum, parameter :: PETSC_DRAW_THISTLE = 29
  PetscEnum, parameter :: PETSC_DRAW_LIMEGREEN = 30
  PetscEnum, parameter :: PETSC_DRAW_LAVENDERBLUSH = 31
  PetscEnum, parameter :: PETSC_DRAW_PLUM = 32

contains

#include <../ftn/sys/petscdraw.hf90>
end module petscdraw

!------------------------------------------------------------------------
subroutine PetscSetCOMM(c1, c2)
  use, intrinsic :: ISO_C_binding
  use petscmpi, only: PETSC_COMM_WORLD, PETSC_COMM_SELF

  implicit none
  MPI_Comm c1, c2

  PETSC_COMM_WORLD = c1
  PETSC_COMM_SELF = c2
end

subroutine PetscGetCOMM(c1)
  use, intrinsic :: ISO_C_binding
  use petscmpi, only: PETSC_COMM_WORLD
  implicit none
  MPI_Comm c1

  c1 = PETSC_COMM_WORLD
end subroutine PetscGetCOMM

subroutine PetscSetModuleBlock()
  use, intrinsic :: ISO_C_binding
  use petscsys!, only: PETSC_NULL_CHARACTER,PETSC_NULL_INTEGER,&
  !  PETSC_NULL_SCALAR,PETSC_NULL_DOUBLE,PETSC_NULL_REAL,&
  !  PETSC_NULL_BOOL,PETSC_NULL_FUNCTION,PETSC_NULL_MPI_COMM
  implicit none

  call PetscSetFortranBasePointers(PETSC_NULL_CHARACTER, &
                                   PETSC_NULL_INTEGER, PETSC_NULL_SCALAR, &
                                   PETSC_NULL_DOUBLE, PETSC_NULL_REAL, &
                                   PETSC_NULL_BOOL, PETSC_NULL_ENUM, PETSC_NULL_FUNCTION, &
                                   PETSC_NULL_MPI_COMM, &
                                   PETSC_NULL_INTEGER_ARRAY, PETSC_NULL_SCALAR_ARRAY, &
                                   PETSC_NULL_REAL_ARRAY, PETSC_NULL_INTEGER_POINTER, &
                                   PETSC_NULL_SCALAR_POINTER, PETSC_NULL_REAL_POINTER)
end subroutine PetscSetModuleBlock

subroutine PetscSetModuleBlockMPI(freal, fscalar, fsum, finteger)
  use, intrinsic :: ISO_C_binding
  use petscmpi, only: MPIU_REAL, MPIU_SUM, MPIU_SCALAR, MPIU_INTEGER
  implicit none

  integer4 freal, fscalar, fsum, finteger

  MPIU_REAL = freal
  MPIU_SCALAR = fscalar
  MPIU_SUM = fsum
  MPIU_INTEGER = finteger
end subroutine PetscSetModuleBlockMPI

subroutine PetscSetModuleBlockNumeric(pi, maxreal, minreal, eps, seps, small, pinf, pninf)
  use petscsys, only: PETSC_PI, PETSC_MAX_REAL, PETSC_MIN_REAL, &
                      PETSC_MACHINE_EPSILON, PETSC_SQRT_MACHINE_EPSILON, &
                      PETSC_SMALL, PETSC_INFINITY, PETSC_NINFINITY
  use, intrinsic :: ISO_C_binding
  implicit none

  PetscReal pi, maxreal, minreal, eps, seps
  PetscReal small, pinf, pninf

  PETSC_PI = pi
  PETSC_MAX_REAL = maxreal
  PETSC_MIN_REAL = minreal
  PETSC_MACHINE_EPSILON = eps
  PETSC_SQRT_MACHINE_EPSILON = seps
  PETSC_SMALL = small
  PETSC_INFINITY = pinf
  PETSC_NINFINITY = pninf
end subroutine PetscSetModuleBlockNumeric
