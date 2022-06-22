!
!
!   This program demonstrates use of PETSc dense matrices.
!
      program main
#include <petsc/finclude/petscsys.h>
      use petscsys
      implicit none

      PetscErrorCode ierr

      PetscCallA(PetscInitialize(ierr))

!  Demo of PETSc-allocated dense matrix storage
      call Demo1()

!  Demo of user-allocated dense matrix storage
      call Demo2()

      PetscCallA(PetscFinalize(ierr))
      end

! -----------------------------------------------------------------
!
!  Demo1 -  This subroutine demonstrates the use of PETSc-allocated dense
!  matrix storage.  Here MatDenseGetArray() is used for direct access to the
!  array that stores the dense matrix.  The user declares an array (aa(1))
!  and index variable (ia), which are then used together to manipulate
!  the array contents.
!
!  Note the use of PETSC_NULL_SCALAR in MatCreateSeqDense() to indicate that no
!  storage is being provided by the user. (Do NOT pass a zero in that
!  location.)
!
      subroutine Demo1()
#include <petsc/finclude/petscmat.h>
      use petscmat
      implicit none

      Mat         A
      PetscInt   n,m
      PetscErrorCode ierr
      PetscScalar aa(1)
      PetscOffset ia

      n = 4
      m = 5

!  Create matrix

      PetscCall(MatCreate(PETSC_COMM_SELF,A,ierr))
      PetscCall(MatSetSizes(A,m,n,m,n,ierr))
      PetscCall(MatSetType(A,MATSEQDENSE,ierr))
      PetscCall(MatSetUp(A,ierr))

!  Access array storage
      PetscCall(MatDenseGetArray(A,aa,ia,ierr))

!  Set matrix values directly
      PetscCall(FillUpMatrix(m,n,aa(ia+1))

      PetscCall(MatDenseRestoreArray(A,aa,ia,ierr))

!  Finalize matrix assembly
      PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY,ierr))
      PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY,ierr))

!  View matrix
      PetscCall(MatView(A,PETSC_VIEWER_STDOUT_SELF,ierr))

!  Clean up
      PetscCall(MatDestroy(A,ierr))
      return
      end

! -----------------------------------------------------------------
!
!  Demo2 -  This subroutine demonstrates the use of user-allocated dense
!  matrix storage.
!
      subroutine Demo2()
#include <petsc/finclude/petscmat.h>
      use petscmat
      implicit none

      PetscInt   n,m
      PetscErrorCode ierr
      parameter (m=5,n=4)
      Mat       A
      PetscScalar    aa(m,n)

!  Create matrix
      PetscCall(MatCreateSeqDense(PETSC_COMM_SELF,m,n,aa,A,ierr))

!  Set matrix values directly
      PetscCall(FillUpMatrix(m,n,aa)

!  Finalize matrix assembly
      PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY,ierr))
      PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY,ierr))

!  View matrix
      PetscCall(MatView(A,PETSC_VIEWER_STDOUT_SELF,ierr))

!  Clean up
      PetscCall(MatDestroy(A,ierr))
      return
      end

! -----------------------------------------------------------------

      subroutine FillUpMatrix(m,n,X)
      PetscInt          m,n,i,j
      PetscScalar      X(m,n)

      do 10, j=1,n
        do 20, i=1,m
          X(i,j) = 1.0/real(i+j-1)
 20     continue
 10   continue
      return
      end
