!
!
!  Base include file for Fortran use of the PETSc package.
!
#include "petscconf.h"
#include "petscversion.h"
#include "finclude/petscsysdef.h"

#if !defined(PETSC_AVOID_MPIF_H)
#include "mpif.h"
#endif

! ------------------------------------------------------------------------
!     Non Common block Stuff declared first
!    
!     Flags
!
      PetscTruth PETSC_TRUE
      PetscTruth PETSC_FALSE
      PetscTruth PETSC_YES
      PetscTruth PETSC_NO
#if defined(PETSC_FORTRAN_PETSCTRUTH_INT)
      parameter (PETSC_TRUE = 1,PETSC_FALSE = 0)
      parameter (PETSC_YES= 1, PETSC_NO = 0)
#else
      parameter (PETSC_TRUE = .true.,PETSC_FALSE = .false.)
      parameter (PETSC_YES=.true., PETSC_NO=.false.)
#endif
      PetscInt   PETSC_DECIDE,PETSC_DETERMINE
      parameter (PETSC_DECIDE=-1,PETSC_DETERMINE=-1)

      PetscInt  PETSC_DEFAULT_INTEGER
      parameter (PETSC_DEFAULT_INTEGER = -2)

      PetscFortranDouble PETSC_DEFAULT_DOUBLE_PRECISION
      parameter (PETSC_DEFAULT_DOUBLE_PRECISION=-2.0d0)

      PetscEnum PETSC_FP_TRAP_OFF
      PetscEnum PETSC_FP_TRAP_ON
      parameter (PETSC_FP_TRAP_OFF = 0,PETSC_FP_TRAP_ON = 1) 



!
!     Default PetscViewers.
!
      PetscFortranAddr PETSC_VIEWER_DRAW_WORLD
      PetscFortranAddr PETSC_VIEWER_DRAW_SELF
      PetscFortranAddr PETSC_VIEWER_SOCKET_WORLD
      PetscFortranAddr PETSC_VIEWER_SOCKET_SELF
      PetscFortranAddr PETSC_VIEWER_STDOUT_WORLD
      PetscFortranAddr PETSC_VIEWER_STDOUT_SELF
      PetscFortranAddr PETSC_VIEWER_STDERR_WORLD
      PetscFortranAddr PETSC_VIEWER_STDERR_SELF
      PetscFortranAddr PETSC_VIEWER_BINARY_WORLD
      PetscFortranAddr PETSC_VIEWER_BINARY_SELF
      PetscFortranAddr PETSC_VIEWER_MATLAB_WORLD
      PetscFortranAddr PETSC_VIEWER_MATLAB_SELF

!
!     The numbers used below should match those in 
!     private/fortranimpl.h
!
      parameter (PETSC_VIEWER_DRAW_WORLD   = -4) 
      parameter (PETSC_VIEWER_DRAW_SELF    = -5)
      parameter (PETSC_VIEWER_SOCKET_WORLD = -6)
      parameter (PETSC_VIEWER_SOCKET_SELF  = -7)
      parameter (PETSC_VIEWER_STDOUT_WORLD = -8)
      parameter (PETSC_VIEWER_STDOUT_SELF  = -9)
      parameter (PETSC_VIEWER_STDERR_WORLD = -10)
      parameter (PETSC_VIEWER_STDERR_SELF  = -11)
      parameter (PETSC_VIEWER_BINARY_WORLD = -12)
      parameter (PETSC_VIEWER_BINARY_SELF  = -13)
      parameter (PETSC_VIEWER_MATLAB_WORLD = -14)
      parameter (PETSC_VIEWER_MATLAB_SELF  = -15)
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
      PetscEnum PETSC_LOGICAL
      PetscEnum PETSC_ENUM
      PetscEnum PETSC_TRUTH
      PetscEnum PETSC_LONG_DOUBLE

#if defined(PETSC_USE_SCALAR_SINGLE)
#define PETSC_REAL PETSC_FLOAT
#elif defined(PETSC_USE_SCALAR_LONG_DOUBLE)
#define PETSC_REAL PETSC_LONG_DOUBLE
#elif defined(PETSC_USE_SCALAR_INT)
#define PETSC_REAL PETSC_INT
#else
#define PETSC_REAL PETSC_DOUBLE
#endif
#define PETSC_FORTRANADDR PETSC_LONG

      parameter (PETSC_INT=0,PETSC_DOUBLE=1,PETSC_COMPLEX=2)
      parameter (PETSC_LONG=3,PETSC_SHORT=4,PETSC_FLOAT=5)
      parameter (PETSC_CHAR=6,PETSC_LOGICAL=7,PETSC_ENUM=8)
      parameter (PETSC_TRUTH=9,PETSC_LONG_DOUBLE=10)
!
! ------------------------------------------------------------------------
!     PETSc mathematics include file. Defines certain basic mathematical 
!    constants and functions for working with single and double precision
!    floating point numbers as well as complex and integers.
!
!     Representation of complex i
!
      PetscFortranComplex PETSC_i
      parameter (PETSC_i = (0.0d0,1.0d0))
!
!     Basic constants
! 
      PetscFortranDouble PETSC_PI
      PetscFortranDouble PETSC_DEGREES_TO_RADIANS
      PetscFortranDouble PETSC_MAX
      PetscFortranDouble PETSC_MIN

      parameter (PETSC_PI = 3.14159265358979323846264d0)
      parameter (PETSC_DEGREES_TO_RADIANS = 0.01745329251994d0)
      parameter (PETSC_MAX = 1.d300,PETSC_MIN = -1.d300)

      PetscFortranDouble PETSC_MACHINE_EPSILON
      PetscFortranDouble PETSC_SQRT_MACHINE_EPSILON
      PetscFortranDouble PETSC_SMALL

#if defined(PETSC_USE_SCALAR_SINGLE)
      parameter (PETSC_MACHINE_EPSILON = 1.e-7)
      parameter (PETSC_SQRT_MACHINE_EPSILON = 3.e-4)
      parameter (PETSC_SMALL = 1.e-5)
#else
      parameter (PETSC_MACHINE_EPSILON = 1.d-14)
      parameter (PETSC_SQRT_MACHINE_EPSILON = 1.d-7)
      parameter (PETSC_SMALL = 1.d-10)
#endif
!
! ----------------------------------------------------------------------------
!    BEGIN PETSc aliases for MPI_ constants
!
      integer MPIU_SCALAR 
#if defined(PETSC_USE_COMPLEX)
#if defined (PETSC_USE_SCALAR_SINGLE)
      parameter(MPIU_SCALAR = MPI_COMPLEX)
#else
      parameter(MPIU_SCALAR = MPI_DOUBLE_COMPLEX)
#endif
#else
#if defined (PETSC_USE_SCALAR_SINGLE)
      parameter (MPIU_SCALAR = MPI_REAL)
#elif defined(PETSC_USE_SCALAR_LONG_DOUBLE)
      parameter(MPIU_SCALAR = MPI_2DOUBLE_PRECISION)
#else
      parameter(MPIU_SCALAR = MPI_DOUBLE_PRECISION)
#endif
#endif

      integer MPIU_INTEGER
#if defined(PETSC_USE_64BIT_INDICES)
      parameter(MPIU_INTEGER = MPI_INTEGER8)
#else
      parameter(MPIU_INTEGER = MPI_INTEGER)
#endif

!
! ----------------------------------------------------------------------------
!    BEGIN COMMON-BLOCK VARIABLES
!
!
!     PETSc world communicator
!
      MPI_Comm PETSC_COMM_WORLD
      MPI_Comm PETSC_COMM_SELF
!
!     Fortran Null
!
      PetscChar(80)       PETSC_NULL_CHARACTER
      PetscInt            PETSC_NULL_INTEGER
      PetscFortranDouble  PETSC_NULL_DOUBLE
      PetscInt            PETSC_NULL
      PetscObject         PETSC_NULL_OBJECT
!
!      A PETSC_NULL_FUNCTION pointer
!
      external PETSC_NULL_FUNCTION
      PetscScalar   PETSC_NULL_SCALAR
      PetscReal     PETSC_NULL_REAL
      PetscTruth    PETSC_NULL_TRUTH
!
!     Common Block to store some of the PETSc constants.
!     which can be set - only at runtime.
!
!
!     A string should be in a different common block
!  
      common /petscfortran1/ PETSC_NULL_CHARACTER
      common /petscfortran2/ PETSC_NULL_INTEGER
      common /petscfortran3/ PETSC_NULL
      common /petscfortran4/ PETSC_NULL_SCALAR
      common /petscfortran5/ PETSC_NULL_DOUBLE
      common /petscfortran6/ PETSC_NULL_REAL
      common /petscfortran7/ PETSC_NULL_TRUTH
      common /petscfortran8/ PETSC_NULL_OBJECT
      common /petscfortran9/ PETSC_COMM_WORLD
      common /petscfortran10/ PETSC_COMM_SELF
!
!     Possible arguments to PetscPushErrorHandler()
!
      external PETSCTRACEBACKERRORHANDLER
      external PETSCABORTERRORHANDLER
      external PETSCEMACSCLIENTERRORHANDLER
      external PETSCATTACHDEBUGGERERRORHANDLER
      external PETSCIGNOREERRORHANDLER
!
      external PetscIsInfOrNanScalar
      external PetscIsInfOrNanReal
      PetscTruth PetscIsInfOrNanScalar
      PetscTruth PetscIsInfOrNanReal

  
!    END COMMON-BLOCK VARIABLES
! ----------------------------------------------------------------------------
!
!
!     Random numbers
!
#define PETSCRAND 'rand'
#define PETSCRAND48 'rand48'
#define PETSCSPRNG 'sprng'          
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

!
