
C    "$Id: ex19.F,v 1.16 1996/01/12 22:05:06 bsmith Exp bsmith $";

#include "include/finclude/petsc.h"
#include "include/finclude/vec.h"
C
C  This example demonstrates basic use of the PETSc Fortran interface
C  to vectors.
C
       integer          n, ierr,flg
       Scalar           one, two, three, dot, norm
       Vec              x,y,w

       n     = 20
       one   = 1.0
       two   = 2.0
       three = 3.0

       call PetscInitialize(ierr)                  
       call OptionsGetInt(PETSC_NULL,'-n',n,flg,ierr)

C Create a vector, then duplicate it
       call VecCreate(MPI_COMM_WORLD,n,x,ierr)
       call VecDuplicate(x,y,ierr)
       call VecDuplicate(x,w,ierr)

       call VecSet(one,x,ierr)
       call VecSet(two,y,ierr)

       call VecDot(x,y,dot,ierr)
       write(6,100) dot
  100  format('Result of inner product ',f10.4)

       call VecScale(two,x,ierr)
       call VecNorm(x,NORM_2,norm,ierr)
       write(6,110) norm
  110  format('Result of scaling ',f10.4)

       call VecCopy(x,w,ierr)
       call VecNorm(w,NORM_2,norm,ierr)
       write(6,120) norm
  120  format('Result of copy ',f10.4)

       call VecAXPY(three,x,y,ierr)
       call VecNorm(y,NORM_2,norm,ierr)
       write(6,130) norm
  130  format('Result of axpy ',f10.4)

       call VecDestroy(x,ierr)
       call VecDestroy(y,ierr)
       call VecDestroy(w,ierr)
       call PetscFinalize(ierr)
       stop 
       end

 
