C
C  $Id: zoptions.c,v 1.1 1995/08/21 19:56:20 bsmith Exp bsmith $";
C
C    Utility routine used to set some values into a Fortran
C  common block


      subroutine PetscSetCommonBlock(s1,s2,s3)
      integer s1,s2,s3

#include      "include/finclude/petsc.h"

      STDOUT_VIEWER_SELF  = s1
      STDERR_VIEWER_SELF  = s2
      STDOUT_VIEWER_WORLD = s3

      return
      end
