C
C    Utility routine used to set some values into a Fortran
C  common block
#include      "include/finclude/petsc.h"

      subroutine PetscSetCommonBlock(s1,s2,s3)

      STDOUT_VIEWER_SELF = s1
      STDERR_VIEWER_SELF = s2
      STDOUT_VIEWER_WORLD = s3

      return
      end
