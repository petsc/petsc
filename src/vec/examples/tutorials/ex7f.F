
C    "$Id: ex7f.F,v 1.6 1996/10/23 22:14:12 bsmith Exp curfman $";

       subroutine ex7f(vec)

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
       call ex7c(vec)
       call VecView(vec,VIEWER_STDOUT_WORLD,ierr)
       return 
       end

 
