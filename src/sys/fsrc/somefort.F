!
!
!     Utility routine used to set constants into the
!     PETSc Fortran common block
!

      subroutine PetscSetCommonBlock(c1,c2)
      implicit none

#include      "../../../include/finclude/petscsys.h"

      integer c1,c2

      PETSC_COMM_WORLD    = c1
      PETSC_COMM_SELF     = c2

      call PetscSetFortranBasePointers(PETSC_NULL_CHARACTER,            &
     &     PETSC_NULL_INTEGER,PETSC_NULL,PETSC_NULL_SCALAR,             &
     &     PETSC_NULL_DOUBLE,PETSC_NULL_REAL,PETSC_NULL_OBJECT,         &
     &     PETSC_NULL_TRUTH,PETSC_NULL_FUNCTION)

      return
      end

      subroutine PetscGetCommonCOMM(c1)
      implicit none
#include      "../../../include/finclude/petscsys.h"
      integer c1

      c1 = PETSC_COMM_WORLD
      return
      end

      block data PetscCommInit
      implicit none
!
!     this code is duplicated - because including finclude/petscsys.h here
!     gives compile errors.
!
      MPI_Comm PETSC_COMM_WORLD
      MPI_Comm PETSC_COMM_SELF
      common /petscfortran9/ PETSC_COMM_WORLD
      common /petscfortran10/ PETSC_COMM_SELF
      data   PETSC_COMM_WORLD /0/
      data   PETSC_COMM_SELF /0/
      end
