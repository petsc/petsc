
C    "$Id: ex7f.F,v 1.9 1996/11/19 19:47:23 balay Exp balay $";

       subroutine ex7f(vec)

#include "include/FINCLUDE/petsc.h"
#include "include/FINCLUDE/vec.h"
C
C  This routine demonstates how a computational module may be written
C  in Fortran and called from a C routine, passing down PETSc objects.
C

       Scalar           two
       Vec              vec
       integer          ierr

       two  = 2.0

       call VecSet(two,vec,ierr)

C
C  Now call C routine from Fortran, passing in the vector
C
       call ex7c(vec)
C
C     IO from the fortran routines may cause all kinds of
C     grief while linking. FC_LIBS in base.site is supporsed
C     to take care of it. 
C
 100   format ('Calling VecView from Fortran')
       write(6,100)
C
C  Now Call a Petsc Routine form Fortran
C
       call VecView(vec,VIEWER_STDOUT_WORLD,ierr)
       return 
       end

 
