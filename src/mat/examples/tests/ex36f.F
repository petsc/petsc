!
!
!   This program demonstrates use of PETSc dense matrices.
!
      program main

#include "include/finclude/petsc.h"

      integer err

      call PetscInitialize(PETSC_NULL_CHARACTER,err)

!  Demo of PETSc-allocated dense matrix storage
      call Demo1()

!  Demo of user-allocated dense matrix storage
      call Demo2()

      call PetscFinalize(err)
      end

! -----------------------------------------------------------------
!
!  Demo1 -  This subroutine demonstrates the use of PETSc-allocated dense
!  matrix storage.  Here MatGetArray() is used for direct access to the
!  array that stores the dense matrix.  The user declares an array (aa(1))
!  and index variable (ia), which are then used together to manipulate
!  the array contents.
!
!  Note the use of PETSC_NULL_SCALAR in MatCreateSeqDense() to indicate that no
!  storage is being provided by the user. (Do NOT pass a zero in that
!  location.)
!
      subroutine Demo1()

#include "include/finclude/petsc.h"
#include "include/finclude/petscmat.h"
#include "include/finclude/petscviewer.h"

      Mat         A
      integer     n,m,err
      PetscScalar aa(1)
      PetscOffset ia

      n = 4
      m = 5

!  Create matrix
      call MatCreateSeqDense(PETSC_COMM_SELF,m,n,PETSC_NULL_SCALAR,     &
     &     A,err)
!      call MatCreateMPIDense(PETSC_COMM_WORLD,m,n,m,n,PETSC_NULL_SCALAR,A,err)

!  Access array storage
      call MatGetArray(A,aa,ia,err)

!  Set matrix values directly
      call FillUpMatrix(m,n,aa(ia+1))

      call MatRestoreArray(A,aa,ia,err)

!  Finalize matrix assembly
      call MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY,err)
      call MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY,err)

!  View matrix
      call MatView(A,PETSC_VIEWER_STDOUT_SELF,err)

!  Clean up
      call MatDestroy(A,err)     
      return 
      end

! -----------------------------------------------------------------
!
!  Demo2 -  This subroutine demonstrates the use of user-allocated dense
!  matrix storage.
!
      subroutine Demo2()

#include "include/finclude/petsc.h"
#include "include/finclude/petscmat.h"
#include "include/finclude/petscviewer.h"

      integer   n,m,err
      parameter (m=5,n=4)
      Mat       A
      PetscScalar    aa(m,n)

!  Create matrix
      call MatCreateSeqDense(PETSC_COMM_SELF,m,n,aa,A,err)

!  Set matrix values directly
      call FillUpMatrix(m,n,aa)

!  Finalize matrix assembly
      call MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY,err)
      call MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY,err)

!  View matrix
      call MatView(A,PETSC_VIEWER_STDOUT_SELF,err)

!  Clean up
      call MatDestroy(A,err)     
      return 
      end

! -----------------------------------------------------------------

      subroutine FillUpMatrix(m,n,X)
      integer          m,n,i,j
      PetscScalar      X(m,n)

      do 10, j=1,n
        do 20, i=1,m
          X(i,j) = 1.0/(i+j-1)
 20     continue
 10   continue
      return
      end




