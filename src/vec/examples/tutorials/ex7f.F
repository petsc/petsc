
C    "$Id: ex7f.F,v 1.7 1996/11/12 16:55:45 curfman Exp curfman $";

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
       call VecView(vec,VIEWER_STDOUT_WORLD,ierr)
       return 
       end

 
