
C      "$Id: ex36.F,v 1.3 1995/11/24 20:26:33 curfman Exp bsmith $";

#include "include/finclude/petsc.h"
#include "include/finclude/mat.h"
#include "include/finclude/viewer.h"

C
C  This example demonstrates the use of MatGetArray() for direct access to
C  the array that stores a PETSc dense matrix.  The user declares an array
C  (aa(1)) and index variable (ia), which are then used together to
C  manipulate the array contents.
C
C  Note the use of PetscNull in MatCreateSeqDense() to indicate that no
C  storage is being provided by the user. (Do NOT pass a zero in that
C  location.)
C
      double precision aa(1)
      integer          ierr, ia, n, m
      Mat              A

      call PetscInitialize(ierr)
      m = 5
      n = 5

C  Create matrix
      call MatCreateSeqDense(MPI_COMM_SELF,m,n,PetscNull,A,ierr)

C  Access array storage
      call MatGetArray(A,aa,ia,ierr)

C  Set matrix values directly
      call FillUpMatrix(m,n,aa(ia+1))
      call MatView(A,STDOUT_VIEWER_SELF,ierr)

C  Clean up
      call MatDestroy(A,ierr)     
      call PetscFinalize(ierr)
      stop
      end

C -----------------------------------------------------------------

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


