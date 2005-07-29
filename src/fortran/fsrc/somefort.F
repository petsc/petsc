!
!
!     Utility routine used to set constants into the
!     PETSc Fortran common block
!

      subroutine PetscSetCommonBlock(c1,c2)
      implicit none

#include      "../../../include/finclude/petsc.h"

      integer c1,c2

      PETSC_COMM_WORLD    = c1
      PETSC_COMM_SELF     = c2

      call PetscSetFortranBasePointers(PETSC_NULL_CHARACTER,            &
     &     PETSC_NULL_INTEGER,PETSC_NULL,PETSC_NULL_SCALAR,             &
     &     PETSC_NULL_DOUBLE,PETSC_NULL_REAL,PETSC_NULL_OBJECT,         &
     &     PETSC_NULL_FUNCTION)

      return
      end

