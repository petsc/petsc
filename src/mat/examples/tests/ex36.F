
C      "$Id: ex36.F,v 1.1 1995/11/23 00:33:54 bsmith Exp bsmith $";

#include "include/finclude/petsc.h"
#include "include/finclude/vec.h"
#include "include/finclude/mat.h"
#include "include/finclude/pc.h"
#include "include/finclude/sles.h"
#include "include/finclude/viewer.h"
C
C  Demonstrates how a Fortran user can access the array storing a PETSc dense matrix
C The "global" varable PetscDouble() is indexed by the integer obtained from
C MatGetArray(). This index indicates the position in PetscDouble() that 
C is were the Matrix array storage begins. Note also the use of PetscNull to 
C indicate no storage is being provided (do not pass a zero in that location).
C

      integer          ierr,idx,n,m
      Mat              A

      call PetscInitialize(ierr)
      m = 5
      n = 5

      call MatCreateSeqDense(MPI_COMM_WORLD,m,n,PetscNull,A,ierr)
      call MatGetArray(A,idx,ierr)

      call FillUpMatrix(m,n,PetscDouble(idx))

      call MatView(A,STDOUT_VIEWER_SELF,ierr)
     
      call PetscFinalize(ierr)
      stop
      end

      subroutine FillUpMatrix(m,n,X)
      integer          m,n,i,j
      double precision X(m,n)

      do 10, j=1,n
        do 20, i=1,m
          X(i,j) = 1.0/(i+j-1)
 20     continue
 10   continue
      return
      end


