

C
C  "$Id: somefort.F,v 1.11 1996/09/14 12:56:10 bsmith Exp bsmith $";
C
C  Utility routine used to set some values into a Fortran common block

      subroutine PetscSetCommonBlock(s1,s2,s3,s4)
      integer s1,s2,s3,s4

#include      "../../../include/FINCLUDE/petsc.h"

      VIEWER_STDOUT_SELF  = s1
      VIEWER_STDERR_SELF  = s2
      VIEWER_STDOUT_WORLD = s3
      PETSC_COMM_WORLD    = s4

      call PetscSetFortranBasePointers(PETSC_NULL,PETSC_NULL_CHARACTER)

      return
      end

#if defined(HAVE_NAGF90)
C
C  These functions are needed if you are using 
C  MPICH, have not compiled MPICH for the NAG 
C  Fortran 90 compiler, but are compiling PETSc
C  with the NAG Fortran 90 compiler. 
C 
      integer function mpir_iargc()
C
      use f90_unix

      mpir_iargc = iargc()
      return
      end
c     
      subroutine mpir_getarg( i, s )
C
      use f90_unix

      integer       i
      character*(*) s
      call getarg(i,s)
      return
      end
#endif



