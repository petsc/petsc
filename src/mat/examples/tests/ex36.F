
C      "$Id: ex36.F,v 1.7 1996/01/30 12:28:17 curfman Exp bsmith $";

#include "include/finclude/petsc.h"
#include "include/finclude/mat.h"
#include "include/finclude/viewer.h"

C
C  This example demonstrates the use of MatGetArray() for direct access to
C  the array that stores a PETSc dense matrix.  The user declares an array
C  (aa(1)) and index variable (ia), which are then used together to
C  manipulate the array contents.
C
C  Note the use of PETSC_NULL in MatCreateSeqDense() to indicate that no
C  storage is being provided by the user. (Do NOT pass a zero in that
C  location.)
C
      Mat              A
      integer          ia, n, m, err
      Scalar           aa(1)

      call PetscInitialize(err)
      n = 5
      m = 5

C  Create matrix
      call MatCreateSeqDense(MPI_COMM_SELF,m,n,PETSC_NULL,A,err)

C  Access array storage
      call MatGetArray(A,aa,ia,err)

C  Set matrix values directly
      call FillUpMatrix(m,n,aa(ia+1))

C  Finalize matrix assembly
      call MatAssemblyBegin(A,FINAL_ASSEMBLY,err)
      call MatAssemblyEnd(A,FINAL_ASSEMBLY,err)

C  View matrix
      call MatView(A,STDOUT_VIEWER_SELF,err)

C  Clean up
      call MatDestroy(A,err)     
      call PetscFinalize(err)
      stop
      end

C -----------------------------------------------------------------

      subroutine FillUpMatrix(m,n,X)
      integer          m,n,i,j
      Scalar           X(m,n)

      do 10, j=1,n
        do 20, i=1,m
          X(i,j) = 1.0/(i+j-1)
 20     continue
 10   continue
      return
      end




