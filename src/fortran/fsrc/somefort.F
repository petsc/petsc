

C
C  "$Id: somefort.F,v 1.9 1996/03/04 21:30:31 bsmith Exp bsmith $";
C
C  Utility routine used to set some values into a Fortran common block

      subroutine PetscSetCommonBlock(s1,s2,s3)
      integer s1,s2,s3

#include      "../../../include/FINCLUDE/petsc.h"

      VIEWER_STDOUT_SELF  = s1
      VIEWER_STDERR_SELF  = s2
      VIEWER_STDOUT_WORLD = s3

      call PetscSetFortranBasePointers(PETSC_NULL,PETSC_NULL_CHARACTER)

      return
      end




