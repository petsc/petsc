!
!
       subroutine ex7f(vec,comm)
#include <petsc/finclude/petscvec.h>
       use petscvec
       implicit none
!
!  This routine demonstates how a computational module may be written
!  in Fortran and called from a C routine, passing down PETSc objects.
!

       PetscScalar, parameter ::  two = 2.0
       Vec              vec
       MPI_Comm         comm
       PetscErrorCode ierr
       PetscMPIInt rank

!
!     The Objects vec,comm created in a C routine are now
!     used in fortran routines.
!
       call VecSet(vec,two,ierr)
       call MPI_Comm_rank(comm,rank,ierr)
       call MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr)

!
!  Now call C routine from Fortran, passing in the vector, communicator
!
       call ex7c(vec,comm,ierr)
!
!     IO from the fortran routines may cause all kinds of
!
! 100   format ('[',i1,']',' Calling VecView from Fortran')
!       write(6,100) rank
!
!  Now Call a Petsc Routine from Fortran
!
       call VecView(vec,PETSC_VIEWER_STDOUT_WORLD,ierr)
       return
       end


