
C
C  "$Id: somefort.F,v 1.7 1995/12/29 22:31:07 curfman Exp bsmith $";
C
C  Utility routine used to set some values into a Fortran common block

      subroutine PetscSetCommonBlock(s1,s2,s3)
      integer s1,s2,s3

#include      "../../../include/finclude/petsc.h"

      STDOUT_VIEWER_SELF  = s1
      STDERR_VIEWER_SELF  = s2
      STDOUT_VIEWER_WORLD = s3

      call PetscSetFortranBasePointers(PETSC_NULL)

      return
      end


