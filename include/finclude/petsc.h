!
!  $Id: petsc.h,v 1.92 2000/11/15 19:43:04 bsmith Exp balay $;
!
!  Base include file for Fortran use of the PETSc package.
!
#include "petscconf.h"

#if !defined(PETSC_AVOID_MPIF_H) && !defined(PETSC_AVOID_MPIF_H)
#include "mpif.h"
#endif

#include "finclude/petscdef.h"

#if !defined (PETSC_AVOID_DECLARATIONS)
! ------------------------------------------------------------------------
!     Non Common block Stuff declared first
!    
!     Flags
!
      integer   PETSC_TRUE,PETSC_FALSE,PETSC_DECIDE
      integer   PETSC_DEFAULT_INTEGER,PETSC_DETERMINE
      integer   PETSC_FP_TRAP_OFF,PETSC_FP_TRAP_ON

      parameter (PETSC_TRUE = 1,PETSC_FALSE = 0,PETSC_DECIDE = -1)
      parameter (PETSC_DEFAULT_INTEGER = -2,PETSC_DETERMINE = -1)
      parameter (PETSC_FP_TRAP_OFF = 0,PETSC_FP_TRAP_ON = 1) 


      PetscFortranDouble PETSC_DEFAULT_DOUBLE_PRECISION
      parameter (PETSC_DEFAULT_DOUBLE_PRECISION=-2.0d0)

!
!     Default Viewers.
!
      PetscFortranAddr VIEWER_DRAW_WORLD
      PetscFortranAddr VIEWER_DRAW_SELF
      PetscFortranAddr VIEWER_SOCKET_WORLD
      PetscFortranAddr VIEWER_SOCKET_SELF
      PetscFortranAddr VIEWER_STDOUT_WORLD
      PetscFortranAddr VIEWER_STDOUT_SELF
      PetscFortranAddr VIEWER_STDERR_WORLD
      PetscFortranAddr VIEWER_STDERR_SELF

!
!     The numbers used below should match those in 
!     src/fortran/custom/zpetsc.h
!
      parameter (VIEWER_DRAW_WORLD   = -4) 
      parameter (VIEWER_DRAW_SELF    = -5)
      parameter (VIEWER_SOCKET_WORLD = -6)
      parameter (VIEWER_SOCKET_SELF  = -7)
      parameter (VIEWER_STDOUT_WORLD = -8)
      parameter (VIEWER_STDOUT_SELF  = -9)
      parameter (VIEWER_STDERR_WORLD = -10)
      parameter (VIEWER_STDERR_SELF  = -11)
!
!     PETSc DataTypes
!
      integer PETSC_INT,PETSC_DOUBLE,PETSC_COMPLEX
      integer PETSC_LONG,PETSC_SHORT,PETSC_FLOAT
      integer PETSC_CHAR,PETSC_LOGICAL

      parameter (PETSC_INT=0,PETSC_DOUBLE=1,PETSC_COMPLEX=2)
      parameter (PETSC_LONG=3,PETSC_SHORT=4,PETSC_FLOAT=5)
      parameter (PETSC_CHAR=6,PETSC_LOGICAL=7)
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
      PetscFortranDouble PETSC_PI,PETSC_DEGREES_TO_RADIANS
      PetscFortranDouble PETSC_MAX,PETSC_MIN

      parameter (PETSC_PI = 3.14159265358979323846264d0)
      parameter (PETSC_DEGREES_TO_RADIANS = 0.01745329251994d0)
      parameter (PETSC_MAX = 1.d300,PETSC_MIN = -1.d300)
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
      character*(80)      PETSC_NULL_CHARACTER
      PetscFortranInt     PETSC_NULL_INTEGER
      PetscFortranDouble  PETSC_NULL_DOUBLE
!
!      A PETSC_NULL_FUNCTION pointer
!
      external PETSC_NULL_FUNCTION
      Scalar   PETSC_NULL_SCALAR
!
!     Common Block to store some of the PETSc constants.
!     which can be set - only at runtime.
!
!
!     A string should be in a different common block
!  
      common /petscfortran1/ PETSC_NULL_CHARACTER
      common /petscfortran2/ PETSC_NULL_INTEGER
      common /petscfortran3/ PETSC_NULL_SCALAR
      common /petscfortran4/ PETSC_NULL_DOUBLE
      common /petscfortran5/ PETSC_COMM_WORLD,PETSC_COMM_SELF

!    END COMMON-BLOCK VARIABLES
! ----------------------------------------------------------------------------

#endif
