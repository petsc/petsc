
C    "$Id: ex21.F,v 1.10 1995/10/11 15:18:37 bsmith Exp $";

#include "include/finclude/petsc.h"
#include "include/finclude/vec.h"

C  This is the Fortran analog of ex6.c, which demonstrates a scatter with a 
C  stride and general index set.

       integer          n, ierr, idx1(3), loc(6)
       Vec              x, y
       VecScatterCtx    ctx
       double precision two, vals(6)
       IS               is1, is2

       n       = 6
       two     = 2.0
       ctx     = 0
       idx1(1) = 0
       idx1(2) = 1
       idx1(3) = 2
       loc(1)  = 0
       loc(2)  = 1
       loc(3)  = 2
       loc(4)  = 3
       loc(5)  = 4
       loc(6)  = 5
       vals(1) = 10
       vals(2) = 11
       vals(3) = 12
       vals(4) = 13
       vals(5) = 14
       vals(6) = 15

       call PetscInitialize(ierr)

C Create two vectors
       call VecCreateSeq(MPI_COMM_SELF,n,&x,ierr)
       call VecDuplicate(x,&y,ierr)

C Create two index sets
       call ISCreateSeq(MPI_COMM_SELF,3,idx1,&is1,ierr)
       call ISCreateStrideSeq(MPI_COMM_SELF,3,0,2,&is2,ierr)

       call VecSetValues(x,6,loc,vals,INSERT_VALUES,ierr)
       call VecView(x,STDOUT_VIEWER_SELF,ierr)
       write(6,*) '----'
       call VecSet(&two,y,ierr)
       call VecScatterCtxCreate(x,is1,y,is2,&ctx,ierr)
       call VecScatterBegin(x,y,INSERT_VALUES,SCATTER_ALL,ctx,ierr)
       call VecScatterEnd(x,y,INSERT_VALUES,SCATTER_ALL,ctx,ierr)
       call VecScatterCtxDestroy(ctx,ierr)
  
       call VecView(y,STDOUT_VIEWER_SELF,ierr)

       call ISDestroy(is1,ierr)
       call ISDestroy(is2,ierr)
       call VecDestroy(x,ierr)
       call VecDestroy(y,ierr)

       call PetscFinalize(ierr)
       stop
       end
 
