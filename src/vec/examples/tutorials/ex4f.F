
C    "$Id: ex21.F,v 1.2 1995/10/18 16:46:56 curfman Exp curfman $";

#include "include/finclude/petsc.h"
#include "include/finclude/vec.h"

C
C  This Fortran example illustrates the use of VecSetValues() and VecGetArray().
C

       integer          i, n, ierr, loc(6)
       double precision val(6), xwork(6)
       Vec              x

       n = 6
       do 10 i=1,n
          loc(i) = i-1
          val(i) = 10.0*i
  10   continue

       call PetscInitialize(ierr)

C Create initial vector and set values
       call VecCreateSeq(MPI_COMM_SELF,n,x,ierr)
       call VecSetValues(x,6,loc,val,INSERT_VALUES,ierr)
       write(6,*) 'initial vector:'
       call VecView(x,STDOUT_VIEWER_SELF,ierr)

C Extract vector elements and modify them
       call VecGetArray(x,xwork,ierr)
       do 20 i=1,n
          xwork(i) = 100.0*i
  20   continue
       call VecRestoreArray(x,xwork,ierr)
       write(6,*) 'new vector:'
       call VecView(x,STDOUT_VIEWER_SELF,ierr)

       call VecDestroy(x,ierr)
       call PetscFinalize(ierr)

       stop
       end
 
