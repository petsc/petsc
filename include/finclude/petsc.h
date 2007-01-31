!
!
!  Base include file for Fortran use of the PETSc package.
!
#include "petscconf.h"

#if !defined(PETSC_AVOID_MPIF_H) && !defined(PETSC_AVOID_DECLARATIONS)
#include "mpif.h"
#endif

#include "include/finclude/petscdef.h"

#if !defined (PETSC_AVOID_DECLARATIONS)
! ------------------------------------------------------------------------
!     Non Common block Stuff declared first
!    
!     Flags
!
      PetscEnum   PETSC_TRUE,PETSC_FALSE
      PetscEnum   PETSC_YES,PETSC_NO
      parameter (PETSC_TRUE = 1,PETSC_FALSE = 0)
      parameter (PETSC_YES=1, PETSC_NO=0)

      PetscInt   PETSC_DECIDE,PETSC_DETERMINE
      parameter (PETSC_DECIDE=-1,PETSC_DETERMINE=-1)

      PetscInt  PETSC_DEFAULT_INTEGER
      parameter (PETSC_DEFAULT_INTEGER = -2)

      PetscFortranDouble PETSC_DEFAULT_DOUBLE_PRECISION
      parameter (PETSC_DEFAULT_DOUBLE_PRECISION=-2.0d0)

      PetscEnum   PETSC_FP_TRAP_OFF,PETSC_FP_TRAP_ON
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
!     src/fortran/custom/zpetsc.h
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
      PetscEnum PETSC_INT,PETSC_DOUBLE,PETSC_COMPLEX
      PetscEnum PETSC_LONG,PETSC_SHORT,PETSC_FLOAT
      PetscEnum PETSC_CHAR,PETSC_LOGICAL,PETSC_ENUM
      PetscEnum PETSC_TRUTH,PETSC_LONG_DOUBLE


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

#if defined(PETSC_USE_SINGLE)
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
!    BEGIN COMMON-BLOCK VARIABLES
!
!
!     PETSc world communicator
!
      MPI_Comm PETSC_COMM_WORLD,PETSC_COMM_SELF
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
      common /petscfortran7/ PETSC_COMM_WORLD,PETSC_COMM_SELF
      common /petscfortran8/ PETSC_NULL_OBJECT
!
!     Possible arguments to PetscPushErrorHandler()
!
      external PETSCTRACEBACKERRORHANDLER
      external PETSCABORTERRORHANDLER
      external PETSCEMACSCLIENTERRORHANDLER
      external PETSCATTACHDEBUGGERERRORHANDLER
      external PETSCIGNOREERRORHANDLER

  
!    END COMMON-BLOCK VARIABLES
! ----------------------------------------------------------------------------

#endif
