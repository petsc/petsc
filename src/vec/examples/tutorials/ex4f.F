
C    "$Id: ex21.F,v 1.4 1995/10/19 15:03:00 curfman Exp bsmith $";

#include "include/finclude/petsc.h"
#include "include/finclude/vec.h"

C  This Fortran example illustrates the use of VecSetValues() to set
C  multiple values at once.  This also demonstrates VecGetArray().

       double precision xwork(6)
       integer          i, n, ierr, loc(6), iwork
       Vec              x

C  Note that the vector indices are 0-based in PETSc (for both Fortran
C  and C vectors)
       n = 6
       do 10 i=1,n
          loc(i) = i-1
          xwork(i) = 10.0*i
  10   continue

       call PetscInitialize(ierr)

C Create initial vector and set values
       call VecCreateSeq(MPI_COMM_SELF,n,x,ierr)
       call VecSetValues(x,6,loc,xwork,INSERT_VALUES,ierr)
       write(6,20)
  20   format('initial vector:')
       call VecView(x,STDOUT_VIEWER_SELF,ierr)

C Extract vector elements and modify them
       call VecGetArray(x,iwork,ierr)
       do 30 i=1,n
          PetscDouble(iwork+i) = 100.0*i
  30   continue
       call VecRestoreArray(x,iwork,ierr)
       write(6,40)
  40   format('new vector:')
       call VecView(x,STDOUT_VIEWER_SELF,ierr)

       call VecDestroy(x,ierr)
       call PetscFinalize(ierr)

       stop
       end
 
