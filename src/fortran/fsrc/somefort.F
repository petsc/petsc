!
!  "$Id: somefort.F,v 1.14 1998/03/25 17:13:21 balay Exp balay $";
!
!  Utility routine used to set some values into a Fortran common block

      subroutine PetscSetCommonBlock(s1,s2,s3,c1,c2)
      implicit none

#include      "../../../include/FINCLUDE/petsc.h"

      PetscFortranAddr s1,s2,s3
      integer c1,c2


      VIEWER_STDOUT_SELF  = s1
      VIEWER_STDERR_SELF  = s2
      VIEWER_STDOUT_WORLD = s3
      PETSC_COMM_WORLD    = c1
      PETSC_COMM_SELF     = c2

      call PetscSetFortranBasePointers(PETSC_NULL,PETSC_NULL_CHARACTER)

      return
      end

#if defined(HAVE_NAGF90)
!
!  These functions are needed if you are using 
!  MPICH, have not compiled MPICH for the NAG 
!  Fortran 90 compiler, but are compiling PETSc
!  with the NAG Fortran 90 compiler. 
! 
      integer function mpir_iargc()
      use f90_unix
      implicit none
!


      mpir_iargc = iargc()
      return
      end
!     
      subroutine mpir_getarg( i, s )
      use f90_unix
      implicit none
!


      integer       i
      character*(*) s
      call getarg(i,s)
      return
      end
#endif



