
C    "$Id: ex7f.F,v 1.8 1996/11/16 23:00:00 curfman Exp balay $";

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
C  If IO is done in the fortran routine, then on some machines
C  (for eg: SUN4) use FLINKER (fortran linker) instead of 
C  CLINKER (C linker ). This does not work on DEC Alpha
C
C       write(6,*) 'Calling VecView from Fortran'
C
C  Now Call A Petsc Routine form Fortran
C
       call VecView(vec,VIEWER_STDOUT_WORLD,ierr)
       return 
       end

 
