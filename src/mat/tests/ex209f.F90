#include <petsc/finclude/petscmat.h>
program main
  use petscmat
  implicit none

  Mat A
  PetscErrorCode ierr
  PetscScalar, pointer :: km(:, :)
  PetscInt i, j
  PetscScalar v(1)

  PetscCallA(PetscInitialize(ierr))

  PetscCallA(MatCreate(PETSC_COMM_WORLD, A, ierr))
  PetscCallA(MatSetSizes(A, 3_PETSC_INT_KIND, 3_PETSC_INT_KIND, 3_PETSC_INT_KIND, 3_PETSC_INT_KIND, ierr))
  PetscCallA(MatSetBlockSize(A, 3_PETSC_INT_KIND, ierr))
  PetscCallA(MatSetType(A, MATSEQBAIJ, ierr))
  PetscCallA(MatSetUp(A, ierr))

  allocate (km(3_PETSC_INT_KIND, 3_PETSC_INT_KIND))
  do i = 1, 3
    do j = 1, 3
      km(i, j) = i + j
    end do
  end do

  PetscCallA(MatSetValuesBlocked(A, 1_PETSC_INT_KIND, [0_PETSC_INT_KIND], 1_PETSC_INT_KIND, [0_PETSC_INT_KIND], reshape(km, [3_PETSC_INT_KIND**2]), ADD_VALUES, ierr))
  PetscCallA(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY, ierr))
  PetscCallA(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY, ierr))
  PetscCallA(MatView(A, PETSC_VIEWER_STDOUT_WORLD, ierr))

  PetscCallA(MatGetValues(A, 1_PETSC_INT_KIND, [0_PETSC_INT_KIND], 1_PETSC_INT_KIND, [0_PETSC_INT_KIND], v, ierr))

  PetscCallA(MatDestroy(A, ierr))

  deallocate (km)
  PetscCallA(PetscFinalize(ierr))
end

!/*TEST
!
!     test:
!       requires: double !complex
!
!TEST*/
