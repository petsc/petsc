
C      "$Id: ex36.F,v 1.13 1996/03/08 20:48:32 balay Exp bsmith $";
C
C   This program demonstrates use of PETSc dense matrices.
C

#include "include/FINCLUDE/petsc.h"

      integer err

      call PetscInitialize(PETSC_NULL_CHAR,err)

C  Demo of PETSc-allocated dense matrix storage
      call Demo1()

C  Demo of user-allocated dense matrix storage
      call Demo2()

      call PetscFinalize(err)
      stop
      end

C -----------------------------------------------------------------
C
C  Demo1 -  This subroutine demonstrates the use of PETSc-allocated dense
C  matrix storage.  Here MatGetArray() is used for direct access to the
C  array that stores the dense matrix.  The user declares an array (aa(1))
C  and index variable (ia), which are then used together to manipulate
C  the array contents.
C
C  Note the use of PETSC_NULL in MatCreateSeqDense() to indicate that no
C  storage is being provided by the user. (Do NOT pass a zero in that
C  location.)
C
      subroutine Demo1()

#include "include/FINCLUDE/petsc.h"
#include "include/FINCLUDE/mat.h"
#include "include/FINCLUDE/viewer.h"

      Mat      A
      integer  ia, n, m, err
      Scalar   aa(1)

      n = 4
      m = 5

C  Create matrix
      call MatCreateSeqDense(MPI_COMM_SELF,m,n,PETSC_NULL,A,err)
C      call MatCreateMPIDense(MPI_COMM_WORLD,m,n,m,n,PETSC_NULL,A,err)

C  Access array storage
      call MatGetArray(A,aa,ia,err)

C  Set matrix values directly
      call FillUpMatrix(m,n,aa(ia+1))

      call MatRestoreArray(A,aa,ia,err)

C  Finalize matrix assembly
      call MatAssemblyBegin(A,FINAL_ASSEMBLY,err)
      call MatAssemblyEnd(A,FINAL_ASSEMBLY,err)

C  View matrix
      call MatView(A,STDOUT_VIEWER_SELF,err)

C  Clean up
      call MatDestroy(A,err)     
      return 
      end

C -----------------------------------------------------------------
C
C  Demo2 -  This subroutine demonstrates the use of user-allocated dense
C  matrix storage.
C
      subroutine Demo2()

#include "include/FINCLUDE/petsc.h"
#include "include/FINCLUDE/mat.h"
#include "include/FINCLUDE/viewer.h"

      integer   n, m, err
      parameter (m=5, n=4)
      Mat       A
      Scalar    aa(m,n)

C  Create matrix
      call MatCreateSeqDense(MPI_COMM_SELF,m,n,aa,A,err)

C  Set matrix values directly
      call FillUpMatrix(m,n,aa)

C  Finalize matrix assembly
      call MatAssemblyBegin(A,FINAL_ASSEMBLY,err)
      call MatAssemblyEnd(A,FINAL_ASSEMBLY,err)

C  View matrix
      call MatView(A,STDOUT_VIEWER_SELF,err)

C  Clean up
      call MatDestroy(A,err)     
      return 
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




