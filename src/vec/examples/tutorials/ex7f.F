!
!    "$Id: ex7f.F,v 1.17 2000/05/05 22:15:21 balay Exp balay $";
!
       subroutine ex7f(vec,comm)

#include "include/finclude/petsc.h"
#include "include/finclude/petscvec.h"
!
!  This routine demonstates how a computational module may be written
!  in Fortran and called from a C routine, passing down PETSc objects.
!

       Scalar           two
       Vec              vec
       MPI_Comm         comm
       integer          ierr,rank

       two  = 2.0

!
!     The Objects vec,comm created in a C routine are now
!     used in fortran routines.
!       
       call VecSet(two,vec,ierr)
       call MPI_Comm_rank(comm,rank,ierr)

!
!  Now call C routine from Fortran, passing in the vector, communicator
!
       call ex7c(vec,comm)
!
!     IO from the fortran routines may cause all kinds of
!     grief while linking. *_SYS_LIB in base_variables is supporsed
!     to take care of it. 
!
 100   format ('[',i1,']',' Calling VecView from Fortran')
       write(6,100) rank
!
!  Now Call a Petsc Routine from Fortran
!
       call VecView(vec,VIEWER_STDOUT_WORLD,ierr)
       return 
       end

 
