!
!  $Id: petsc.h,v 1.83 1998/08/27 15:02:24 balay Exp balay $;
!
!  Base include file for Fortran use of the PETSc package
!  Note that the external functions and common-block variables 
!  are declared in this file. The rest are defined in petsvar.h
!
#include "mpif.h"
#include "finclude/petscdef.h"
#include "finclude/petscvar.h"
!
!     Default Viewers.
!     Other viewers which need not be in the common block
!     are declared in petscvar.h file
!
      PetscFortranAddr VIEWER_STDOUT_SELF, VIEWER_STDERR_SELF
      PetscFortranAddr VIEWER_STDOUT_WORLD

!
!     PETSc world communicator
!
      MPI_Comm PETSC_COMM_WORLD, PETSC_COMM_SELF
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
      common /petscfortran5/ VIEWER_STDOUT_SELF,VIEWER_STDERR_SELF
      common /petscfortran6/ VIEWER_STDOUT_WORLD
      common /petscfortran7/ PETSC_COMM_WORLD,PETSC_COMM_SELF

