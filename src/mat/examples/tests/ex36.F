
C      "$Id: ex36.F,v 1.4 1995/11/29 22:10:22 curfman Exp balay $";

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
      Mat              A
      integer          ia, n, m, err
      double precision aa(1)

      call PetscInitialize(err)
      n = 5
      m = 5

C  Create matrix
      call MatCreateSeqDense(MPI_COMM_SELF,m,n,PetscNull,A,err)

C  Access array storage
      call MatGetArray(A,aa,ia,err)

C  Set matrix values directly
      call FillUpMatrix(m,n,aa(ia+1))
      call MatView(A,STDOUT_VIEWER_SELF,err)

C  Clean up
      call MatDestroy(A,err)     
      call PetscFinalize(err)
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


