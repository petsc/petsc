!
!
!  Base include file for Fortran use of the PETSc package.
!
#include "petscconf.h"
#include "petscversion.h"
#include "petsc/finclude/petscsys.h"

!
!    The following block allows one to write constants that match the
!    precision of PetscReal as, for example,  x = .7_PETSC_REAL_KIND
!
       PetscReal,Parameter ::                                                 &
     &                        PetscReal_Private = 1.0
       Integer,Parameter   :: PETSC_REAL_KIND                                 &
     &  = Selected_Real_Kind(Precision(PetscReal_Private))


#if !defined(PETSC_AVOID_MPIF_H)
#if defined(PETSC_HAVE_MPIUNI)
#include "petsc/mpiuni/mpif.h"
#else
!
!  This code is extremely fragile; it assumes the format of the mpif.h file has
!  a particular structure that does not change with MPI implementation versions. But since
!  mpif.h is a bit of a deadwater and PETSC_PROMOTE_FORTRAN_INTEGER is
!  rarely used it is maybe ok to include fragile code
!
#if defined(PETSC_HAVE_MPICH_NUMVERSION) && defined(PETSC_PROMOTE_FORTRAN_INTEGER)
#define INTEGER integer4
#define MPI_STATUS_IGNORE(A) mpi_status_ignore(5)
#define MPI_STATUSES_IGNORE(B,C) mpi_statuses_ignore(5,1)
#elif defined(PETSC_HAVE_OMPI_MAJOR_VERSION)  && defined(PETSC_PROMOTE_FORTRAN_INTEGER)
#define integer integer4
#define INTEGER integer4
#endif
#include "mpif.h"
#if defined(PETSC_HAVE_MPICH_NUMVERSION) && defined(PETSC_PROMOTE_FORTRAN_INTEGER)
#undef INTEGER
#undef MPI_STATUS_IGNORE
#undef MPI_STATUSES_IGNORE
#elif defined(PETSC_HAVE_OMPI_MAJOR_VERSION) && defined(PETSC_PROMOTE_FORTRAN_INTEGER)
#undef integer
#undef INTEGER
#endif
#endif
#endif

      type tPetscOptions
        PetscFortranAddr:: v PETSC_FORTRAN_TYPE_INITIALIZE
      end type tPetscOptions

      PetscOptions, parameter :: PETSC_NULL_OPTIONS =                        &
     &                           tPetscOptions(0)

! ------------------------------------------------------------------------
!     Non Common block Stuff declared first
!
!     Flags
!
      PetscBool  PETSC_TRUE
      PetscBool  PETSC_FALSE
      parameter (PETSC_TRUE = .true.,PETSC_FALSE = .false.)
      PetscInt   PETSC_DECIDE,PETSC_DETERMINE
      parameter (PETSC_DECIDE=-1,PETSC_DETERMINE=-1)

      PetscInt  PETSC_DEFAULT_INTEGER
      parameter (PETSC_DEFAULT_INTEGER = -2)

      PetscReal PETSC_DEFAULT_REAL
      parameter (PETSC_DEFAULT_REAL=-2.0d0)

      PetscEnum PETSC_FP_TRAP_OFF
      PetscEnum PETSC_FP_TRAP_ON
      parameter (PETSC_FP_TRAP_OFF = 0,PETSC_FP_TRAP_ON = 1)

      PetscFortranAddr PETSC_STDOUT

      parameter (PETSC_STDOUT  = 0)
!
!     PETSc DataTypes
!
      PetscEnum PETSC_INT
      PetscEnum PETSC_DOUBLE
      PetscEnum PETSC_COMPLEX
      PetscEnum PETSC_LONG
      PetscEnum PETSC_SHORT
      PetscEnum PETSC_FLOAT
      PetscEnum PETSC_CHAR
      PetscEnum PETSC_BIT_LOGICAL
      PetscEnum PETSC_ENUM
      PetscEnum PETSC_BOOL
      PetscEnum PETSC___FLOAT128
      PetscEnum PETSC_OBJECT
      PetscEnum PETSC_FUNCTION
      PetscEnum PETSC_STRING
      PetscEnum PETSC___FP16
      PetscEnum PETSC_STRUCT
      PetscEnum PETSC_DATATYPE_UNKNOWN

#if defined(PETSC_USE_REAL_SINGLE)
#define PETSC_REAL PETSC_FLOAT
#elif defined(PETSC_USE_REAL___FLOAT128)
#define PETSC_REAL PETSC___FLOAT128
#else
#define PETSC_REAL PETSC_DOUBLE
#endif
#define PETSC_FORTRANADDR PETSC_LONG

      parameter (PETSC_DATATYPE_UNKNOWN=0)
      parameter (PETSC_DOUBLE=1,PETSC_COMPLEX=2)
      parameter (PETSC_LONG=3,PETSC_SHORT=4,PETSC_FLOAT=5)
      parameter (PETSC_CHAR=6,PETSC_BIT_LOGICAL=7,PETSC_ENUM=8)
      parameter (PETSC_BOOL=9,PETSC___FLOAT128=10)
      parameter (PETSC_OBJECT=11,PETSC_FUNCTION=12)
      parameter (PETSC_STRING=13,PETSC___FP16=14,PETSC_STRUCT=15)
      parameter (PETSC_INT=16)
!
!
!
      PetscEnum PETSC_COPY_VALUES
      PetscEnum PETSC_OWN_POINTER
      PetscEnum PETSC_USE_POINTER

      parameter (PETSC_COPY_VALUES = 0)
      parameter (PETSC_OWN_POINTER = 1)
      parameter (PETSC_USE_POINTER = 2)
!
! ------------------------------------------------------------------------
!     PETSc mathematics include file. Defines certain basic mathematical
!    constants and functions for working with single and double precision
!    floating point numbers as well as complex and integers.
!
!     Representation of complex i
!
      PetscFortranComplex PETSC_i
#if defined(PETSC_USE_REAL_SINGLE)
      parameter (PETSC_i = (0.0e0,1.0e0))
#else
      parameter (PETSC_i = (0.0d0,1.0d0))
#endif

!
! ----------------------------------------------------------------------------
!    BEGIN PETSc aliases for MPI_ constants
!
!   These values for __float128 are handled in the common block (below)
!     and transmitted from the C code
!
#if !defined(PETSC_USE_REAL___FLOAT128)
      integer4 MPIU_REAL
#if defined (PETSC_USE_REAL_SINGLE)
      parameter (MPIU_REAL = MPI_REAL)
#else
      parameter(MPIU_REAL = MPI_DOUBLE_PRECISION)
#endif

      integer4 MPIU_SUM
      parameter (MPIU_SUM = MPI_SUM)

      integer4 MPIU_SCALAR
#if defined(PETSC_USE_COMPLEX)
#if defined (PETSC_USE_REAL_SINGLE)
      parameter(MPIU_SCALAR = MPI_COMPLEX)
#else
      parameter(MPIU_SCALAR = MPI_DOUBLE_COMPLEX)
#endif
#else
#if defined (PETSC_USE_REAL_SINGLE)
      parameter (MPIU_SCALAR = MPI_REAL)
#else
      parameter(MPIU_SCALAR = MPI_DOUBLE_PRECISION)
#endif
#endif
#endif

      integer4 MPIU_INTEGER
#if defined(PETSC_USE_64BIT_INDICES)
      parameter(MPIU_INTEGER = MPI_INTEGER8)
#else
      parameter(MPIU_INTEGER = MPI_INTEGER)
#endif

!      A PETSC_NULL_FUNCTION pointer
!
      external PETSC_NULL_FUNCTION
!
!     Possible arguments to PetscPushErrorHandler()
!
      external PETSCTRACEBACKERRORHANDLER
      external PETSCABORTERRORHANDLER
      external PETSCEMACSCLIENTERRORHANDLER
      external PETSCATTACHDEBUGGERERRORHANDLER
      external PETSCIGNOREERRORHANDLER
!
      external  PetscIsInfOrNanScalar
      external  PetscIsInfOrNanReal
      PetscBool PetscIsInfOrNanScalar
      PetscBool PetscIsInfOrNanReal


! ----------------------------------------------------------------------------
!
!     Random numbers
!
      type tPetscRandom
        sequence
        PetscFortranAddr:: v PETSC_FORTRAN_TYPE_INITIALIZE
      end type tPetscRandom

      PetscRandom, parameter :: PETSC_NULL_RANDOM                                  &
     &             = tPetscRandom(0)
!
#define PETSCRAND 'rand'
#define PETSCRAND48 'rand48'
#define PETSCSPRNG 'sprng'
#define PETSCRANDER48 'rander48'
!
!
!
      PetscEnum PETSC_BINARY_INT_SIZE
      PetscEnum PETSC_BINARY_FLOAT_SIZE
      PetscEnum PETSC_BINARY_CHAR_SIZE
      PetscEnum PETSC_BINARY_SHORT_SIZE
      PetscEnum PETSC_BINARY_DOUBLE_SIZE
      PetscEnum PETSC_BINARY_SCALAR_SIZE

      parameter (PETSC_BINARY_INT_SIZE = 4)
      parameter (PETSC_BINARY_FLOAT_SIZE = 4)
      parameter (PETSC_BINARY_CHAR_SIZE = 1)
      parameter (PETSC_BINARY_SHORT_SIZE = 2)
      parameter (PETSC_BINARY_DOUBLE_SIZE = 8)
#if defined(PETSC_USE_COMPLEX)
      parameter (PETSC_BINARY_SCALAR_SIZE = 16)
#else
      parameter (PETSC_BINARY_SCALAR_SIZE = 8)
#endif

      PetscEnum PETSC_BINARY_SEEK_SET
      PetscEnum PETSC_BINARY_SEEK_CUR
      PetscEnum PETSC_BINARY_SEEK_END

      parameter (PETSC_BINARY_SEEK_SET = 0,PETSC_BINARY_SEEK_CUR = 1)
      parameter (PETSC_BINARY_SEEK_END = 2)

      PetscEnum PETSC_BUILDTWOSIDED_ALLREDUCE
      PetscEnum PETSC_BUILDTWOSIDED_IBARRIER
      PetscEnum PETSC_BUILDTWOSIDED_REDSCATTER
      parameter (PETSC_BUILDTWOSIDED_ALLREDUCE = 0)
      parameter (PETSC_BUILDTWOSIDED_IBARRIER = 1)
      parameter (PETSC_BUILDTWOSIDED_REDSCATTER = 2)
!
!     PetscSubcommType
!
      PetscEnum PETSC_SUBCOMM_GENERAL
      PetscEnum PETSC_SUBCOMM_CONTIGUOUS
      PetscEnum PETSC_SUBCOMM_INTERLACED
      parameter(PETSC_SUBCOMM_GENERAL=0)
      parameter(PETSC_SUBCOMM_CONTIGUOUS=1)
      parameter(PETSC_SUBCOMM_INTERLACED=2)

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::PetscReal_Private
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_REAL_KIND
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_OPTIONS
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_TRUE
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_FALSE
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_DECIDE
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_DETERMINE
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_DEFAULT_INTEGER
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_DEFAULT_REAL
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_FP_TRAP_OFF
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_FP_TRAP_ON
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_STDOUT
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_INT
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_DOUBLE
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_COMPLEX
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_LONG
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_SHORT
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_FLOAT
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_CHAR
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_BIT_LOGICAL
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_ENUM
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_BOOL
!DEC$ ATTRIBUTES DLLEXPORT::PETSC___FLOAT128
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_OBJECT
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_FUNCTION
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_STRING
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_STRUC
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_DATATYPE_UNKNOWN
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_COPY_VALUES
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_OWN_POINTER
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_USE_POINTER
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_i
!DEC$ ATTRIBUTES DLLEXPORT::MPIU_REAL
!DEC$ ATTRIBUTES DLLEXPORT::MPIU_SUM
!DEC$ ATTRIBUTES DLLEXPORT::MPIU_SCALAR
!DEC$ ATTRIBUTES DLLEXPORT::MPIU_INTEGER
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_RANDOM
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_BINARY_INT_SIZE
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_BINARY_FLOAT_SIZE
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_BINARY_CHAR_SIZE
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_BINARY_SHORT_SIZE
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_BINARY_DOUBLE_SIZE
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_BINARY_SCALAR_SIZE
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_BINARY_SEEK_SET
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_BINARY_SEEK_CUR
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_BINARY_SEEK_END
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_BUILDTWOSIDED_ALLREDUCE
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_BUILDTWOSIDED_IBARRIER
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_BUILDTWOSIDED_REDSCATTER
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_SUBCOMM_GENERAL
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_SUBCOMM_CONTIGUOUS
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_SUBCOMM_INTERLACED
#endif
!
! include other sys components
!
#include "../src/sys/f90-mod/petscerror.h"
#include "../src/sys/f90-mod/petsclog.h"
#include "../src/sys/f90-mod/petscbag.h"

