
C    "$Id: ex7f.F,v 1.10 1996/11/20 00:08:44 balay Exp balay $";

       subroutine ex7f(vec,comm)

#include "include/FINCLUDE/petsc.h"
#include "include/FINCLUDE/vec.h"
C
C  This routine demonstates how a computational module may be written
C  in Fortran and called from a C routine, passing down PETSc objects.
C

       Scalar           two
       Vec              vec
       MPI_Comm         comm
       integer          ierr,rank

       two  = 2.0

C
C     The Objects vec,comm created in a C routine are now
C     used in fortran routines.
C       
       call VecSet(two,vec,ierr)
       call MPI_Comm_rank(comm,rank,ierr)

C
C  Now call C routine from Fortran, passing in the vector, communicator
C
       call ex7c(vec,comm)
C
C     IO from the fortran routines may cause all kinds of
C     grief while linking. FC_LIBS in base.site is supporsed
C     to take care of it. 
C
 100   format ('[',i1,']',' Calling VecView from Fortran')
       write(6,100) rank
C
C  Now Call a Petsc Routine form Fortran
C
       call VecView(vec,VIEWER_STDOUT_WORLD,ierr)
       return 
       end

 
