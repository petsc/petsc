C
C  "$Id: somefort.F,v 1.4 1995/09/04 17:18:58 bsmith Exp bsmith $";
C
C    Utility routine used to set some values into a Fortran
C  common block

      subroutine PetscSetCommonBlock(s1,s2,s3)
      integer s1,s2,s3

#include      "../../../include/finclude/petsc.h"

      STDOUT_VIEWER_SELF  = s1
      STDERR_VIEWER_SELF  = s2
      STDOUT_VIEWER_WORLD = s3

      call PetscSetFortranBasePointers(PetscInt,PetscDouble,PetscNull)

      return
      end

