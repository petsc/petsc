
#include "mpif.h"

       integer          n, ierr
       double precision one, two, three, dot, norm
       integer          x,y,w

       n     = 20
       one   = 1.0
       two   = 2.0
       three = 3.0
       call  PetscInitialize(ierr)
       call  OptionsGetInt(0,'-n',n)

C      /* create a vector */
       call VecCreate(MPI_COMM_WORLD,n,x)
       call VecDuplicate(x,y)
       call VecDuplicate(x,w)

       call VecSet(one,x)
       call VecSet(two,y)

       call VecDot(x,y,dot)
       print*,'Result of inner product ',dot

       call VecScale(two,x)
       call VecNorm(x,norm)
       print*,'Result of scaling ',norm

       call VecCopy(x,w)
       call VecNorm(w,norm)
       print*,'Result of copy ',norm

       call VecAXPY(three,x,y)
       call VecNorm(y,norm)
       print*,'Result of axpy ',norm

       call PetscFinalize()
       stop 
       end

 
