
C      "$Id: ex21.F,v 1.7 1995/11/29 22:08:42 curfman Exp bsmith $";

#include "include/finclude/petsc.h"
#include "include/finclude/vec.h"

C  This Fortran example illustrates the use of VecSetValues() to set
C  multiple values at once.  This also demonstrates VecGetArray().

       Scalar           xwork(6)
       Scalar           vxx(1), vyy(1)
       integer          i, n, ierr, loc(6), ixx, iyy
       Vec              x, y

#define xx(ib)  vxx(ixx + (ib))
#define yy(ib)  vyy(iyy + (ib))

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
       call VecDuplicate(x,y,ierr)
       call VecSetValues(x,6,loc,xwork,INSERT_VALUES,ierr)
       call VecAssemblyBegin(x,ierr)
       call VecAssemblyEnd(x,ierr)
       write(6,20)
  20   format('initial vector:')
       call VecView(x,STDOUT_VIEWER_SELF,ierr)
       call VecCopy(x,y,ierr)

C Extract vector elements and modify them
       call VecGetArray(x,vxx,ixx,ierr)
       call VecGetArray(y,vyy,iyy,ierr)
       do 30 i=1,n
          xx(i) = 100.0*i
          yy(i) = 1000.0*i
  30   continue
       call VecRestoreArray(x,vxx,ixx,ierr)
       call VecRestoreArray(y,vyy,iyy,ierr)

       write(6,40)
  40   format('new vector 1:')
       call VecView(x,STDOUT_VIEWER_SELF,ierr)

       write(6,50)
  50   format('new vector 2:')
       call VecView(y,STDOUT_VIEWER_SELF,ierr)

       call VecDestroy(x,ierr)
       call VecDestroy(y,ierr)
       call PetscFinalize(ierr)

       stop
       end
 
