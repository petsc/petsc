!
!  "$Id: somefort.F,v 1.19 1998/04/23 01:55:16 balay Exp balay $";
!
!  Utility routine used to set some values into a Fortran common block

      subroutine PetscSetCommonBlock(s1,s2,s3,c1,c2)
      implicit none

#include      "../../../include/finclude/petsc.h"

      PetscFortranAddr s1,s2,s3
      integer c1,c2


      VIEWER_STDOUT_SELF  = s1
      VIEWER_STDERR_SELF  = s2
      VIEWER_STDOUT_WORLD = s3
      PETSC_COMM_WORLD    = c1
      PETSC_COMM_SELF     = c2

      call PetscSetFortranBasePointers(PETSC_NULL_CHARACTER,            &
     &     PETSC_NULL_INTEGER,PETSC_NULL_SCALAR,PETSC_NULL_DOUBLE,      &
     &     PETSC_NULL_FUNCTION)

      return
      end

