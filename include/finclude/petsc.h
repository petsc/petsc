!
!  $Id: petsc.h,v 1.79 1998/07/08 14:47:23 balay Exp bsmith $;
!
!  Base include file for Fortran use of the PETSc package
!
#include "finclude/petscdef.h"

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
      common /petscfortran5/ VIEWER_STDOUT_SELF,VIEWER_STDERR_SELF
      common /petscfortran6/ VIEWER_STDOUT_WORLD
      common /petscfortran7/ PETSC_COMM_WORLD,PETSC_COMM_SELF

