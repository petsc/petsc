
C    "$Id: ex24f.F,v 1.3 1996/04/01 01:09:59 curfman Exp balay $";

       subroutine ex24f(vec)

#include "include/FINCLUDE/petsc.h"
#include "include/FINCLUDE/vec.h"
C
C  This routine demonstates how a compute module may be written in 
C  Fortran and called from a C routine, passing down PETSc objects.
C

       Scalar           two
       Vec              vec
       integer          ierr

       two  = 2.0

       call VecSet(two,vec,ierr)

C
C  Now call C routine from Fortran, passing in the vector
C
       call ex24c(vec)
       call VecView(vec,STDOUT_VIEWER_WORLD,ierr)
       return 
       end

 
