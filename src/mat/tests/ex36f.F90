!
!
!   This program demonstrates use of PETSc dense matrices.
!
#include <petsc/finclude/petscmat.h>
module ex36fmodule
  use petscmat
  implicit none
! -----------------------------------------------------------------
!
!  Demo1 -  This subroutine demonstrates the use of PETSc-allocated dense
!  matrix storage.  Here MatDenseGetArray() is used for direct access to the
!  array that stores the dense matrix.
!
!  Note the use of PETSC_NULL_SCALAR_ARRAY in MatCreateSeqDense() to indicate that no
!  storage is being provided by the user. (Do NOT pass a zero in that
!  location.)
!
contains
  subroutine Demo1()

    Mat A
    PetscInt n, m
    PetscErrorCode ierr
    PetscScalar, pointer :: aa(:, :)

    n = 4
    m = 5

!  Create matrix

    PetscCall(MatCreate(PETSC_COMM_SELF, A, ierr))
    PetscCall(MatSetSizes(A, m, n, m, n, ierr))
    PetscCall(MatSetType(A, MATSEQDENSE, ierr))
    PetscCall(MatSetUp(A, ierr))

!  Access array storage
    PetscCall(MatDenseGetArray(A, aa, ierr))

!  Set matrix values directly
    PetscCall(FillUpMatrix(m, n, aa))

    PetscCall(MatDenseRestoreArray(A, aa, ierr))

!  Finalize matrix assembly
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY, ierr))
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY, ierr))

!  View matrix
    PetscCall(MatView(A, PETSC_VIEWER_STDOUT_SELF, ierr))

!  Clean up
    PetscCall(MatDestroy(A, ierr))
  end

! -----------------------------------------------------------------
!
!  Demo2 -  This subroutine demonstrates the use of user-allocated dense
!  matrix storage.
!
  subroutine Demo2()

    PetscInt n, m
    PetscErrorCode ierr
    parameter(m=5, n=4)
    Mat A
    PetscScalar aa(m, n)

!  Create matrix
    PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, m, n, aa, A, ierr))

!  Set matrix values directly
    PetscCall(FillUpMatrix(m, n, aa))

!  Finalize matrix assembly
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY, ierr))
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY, ierr))

!  View matrix
    PetscCall(MatView(A, PETSC_VIEWER_STDOUT_SELF, ierr))

!  Clean up
    PetscCall(MatDestroy(A, ierr))
  end

! -----------------------------------------------------------------

  subroutine FillUpMatrix(m, n, X)
    PetscInt m, n, i, j
    PetscScalar X(m, n)

    do j = 1, n
      do i = 1, m
        X(i, j) = 1.0/real(i + j - 1)
      end do
    end do
    end module ex36fmodule

    end program main
    use ex36fmodule
    implicit none

    PetscErrorCode ierr

    PetscCallA(PetscInitialize(ierr))

!  Demo of PETSc-allocated dense matrix storage
    call Demo1()

!  Demo of user-allocated dense matrix storage
    call Demo2()

    PetscCallA(PetscFinalize(ierr))
  end
