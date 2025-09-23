program main
#include <petsc/finclude/petscmat.h>
  use petscmat
  implicit none

  Mat A, B
  PetscErrorCode ierr
  PetscScalar, pointer :: km(:, :)
  PetscInt three, one
  PetscInt idxm(1), idxmj(1), i, j
  PetscMPIInt rank, size

  PetscCallA(PetscInitialize(ierr))

  PetscCallMPIA(MPI_Comm_rank(PETSC_COMM_WORLD, rank, ierr))
  PetscCallMPIA(MPI_Comm_size(PETSC_COMM_WORLD, size, ierr))

  PetscCallA(MatCreate(PETSC_COMM_WORLD, A, ierr))
  three = 3
  PetscCallA(MatSetSizes(A, three, three, PETSC_DECIDE, PETSC_DECIDE, ierr))
  PetscCallA(MatSetBlockSize(A, three, ierr))
  PetscCallA(MatSetUp(A, ierr))
  PetscCallA(MatDuplicate(A, MAT_DO_NOT_COPY_VALUES, B, ierr))
  one = 1
  idxm(1) = 0
  allocate (km(three, three))
  do i = 1, 3
    do j = 1, 3
      km(1, 1) = i + j
      idxm(1) = i - 1 + 3*rank
      idxmj(1) = j - 1 + 3*rank
      PetscCallA(MatSetValues(B, one, idxm, one, idxmj, reshape(km, [three*three]), ADD_VALUES, ierr))
    end do
  end do

  PetscCallA(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY, ierr))
  PetscCallA(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY, ierr))
  PetscCallA(MatView(B, PETSC_VIEWER_STDOUT_WORLD, ierr))

  PetscCallA(MatDestroy(A, ierr))
  PetscCallA(MatDestroy(B, ierr))

  deallocate (km)
  PetscCallA(PetscFinalize(ierr))
end

!/*TEST
!
!   test:
!     nsize: 2
!
!TEST*/
