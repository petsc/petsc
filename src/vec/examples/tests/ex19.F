
#include "include/finclude/petsc.h"
#include "include/finclude/vec.h"

       integer          n, ierr
       double precision one, two, three, dot, norm
       Vec              x,y,w

       n     = 20
       one   = 1.0
       two   = 2.0
       three = 3.0

       call  PetscInitialize(ierr)
       call  OptionsGetInt(' ','-n',n,ierr)

C      /* create a vector */
       call VecCreate(MPI_COMM_WORLD,n,x,ierr)
       call VecDuplicate(x,y,ierr)
       call VecDuplicate(x,w,ierr)

       call VecSet(one,x,ierr)
       call VecSet(two,y,ierr)

       call VecDot(x,y,dot,ierr)
       print*,'Result of inner product ',dot

       call VecScale(two,x,ierr)
       call VecNorm(x,norm,ierr)
       print*,'Result of scaling ',norm

       call VecCopy(x,w,ierr)
       call VecNorm(w,norm,ierr)
       print*,'Result of copy ',norm

       call VecAXPY(three,x,y,ierr)
       call VecNorm(y,norm,ierr)
       print*,'Result of axpy ',norm

       call PetscFinalize(ierr)
       stop 
       end

 
