C
C  "$Id: somefort.F,v 1.5 1995/11/23 04:15:38 bsmith Exp curfman $";
C
C  Utility routine used to set some values into a Fortran common block

      subroutine PetscSetCommonBlock(s1,s2,s3)
      integer s1,s2,s3

#include      "../../../include/finclude/petsc.h"

      STDOUT_VIEWER_SELF  = s1
      STDERR_VIEWER_SELF  = s2
      STDOUT_VIEWER_WORLD = s3

      call PetscSetFortranBasePointers(PetscNull)

      return
      end

