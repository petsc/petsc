
C      "$Id: ex21.F,v 1.13 1996/08/27 20:24:00 curfman Exp curfman $";

#include "include/FINCLUDE/petsc.h"
#include "include/FINCLUDE/vec.h"

C  This Fortran example illustrates the use of VecSetValues() to set
C  multiple values at once.  This also demonstrates VecGetArray().

       Scalar  xwork(6)
       Scalar  xx_v(1), yy_v(1)
       integer i, n, ierr, loc(6), xx_i, yy_i
       Vec     x, y

       call PetscInitialize(PETSC_NULL_CHARACTER,ierr)

#define xx_a(ib)  xx_v(xx_i + (ib))
#define yy_a(ib)  yy_v(yy_i + (ib))

C  Note that the vector indices are 0-based in PETSc (for both Fortran
C  and C vectors)
       n = 6
       do 10 i=1,n
          loc(i) = i-1
          xwork(i) = 10.0*i
  10   continue

C Create initial vector and set values
       call VecCreateSeq(MPI_COMM_SELF,n,x,ierr)
       call VecDuplicate(x,y,ierr)
       call VecSetValues(x,6,loc,xwork,INSERT_VALUES,ierr)
       call VecAssemblyBegin(x,ierr)
       call VecAssemblyEnd(x,ierr)
       write(6,20)
  20   format('initial vector:')
       call VecView(x,VIEWER_STDOUT_SELF,ierr)
       call VecCopy(x,y,ierr)

C Extract vector elements and modify them
       call VecGetArray(x,xx_v,xx_i,ierr)
       call VecGetArray(y,yy_v,yy_i,ierr)
       do 30 i=1,n
          xx_a(i) = 100.0*i
          yy_a(i) = 1000.0*i
  30   continue
       call VecRestoreArray(x,xx_v,xx_i,ierr)
       call VecRestoreArray(y,yy_v,yy_i,ierr)

       write(6,40)
  40   format('new vector 1:')
       call VecView(x,VIEWER_STDOUT_SELF,ierr)

       write(6,50)
  50   format('new vector 2:')
       call VecView(y,VIEWER_STDOUT_SELF,ierr)

       call VecDestroy(x,ierr)
       call VecDestroy(y,ierr)
       call PetscFinalize(ierr)

       stop
       end
 
