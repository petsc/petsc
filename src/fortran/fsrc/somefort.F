

C
C  "$Id: somefort.F,v 1.12 1997/01/15 23:17:41 bsmith Exp bsmith $";
C
C  Utility routine used to set some values into a Fortran common block

      subroutine PetscSetCommonBlock(s1,s2,s3,s4,s5)
      implicit none

      integer s1,s2,s3,s4,s5

#include      "../../../include/FINCLUDE/petsc.h"

      VIEWER_STDOUT_SELF  = s1
      VIEWER_STDERR_SELF  = s2
      VIEWER_STDOUT_WORLD = s3
      PETSC_COMM_WORLD    = s4
      PETSC_COMM_SELF     = s5

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
      use f90_unix
      implicit none
C


      mpir_iargc = iargc()
      return
      end
c     
      subroutine mpir_getarg( i, s )
      use f90_unix
      implicit none
C


      integer       i
      character*(*) s
      call getarg(i,s)
      return
      end
#endif



