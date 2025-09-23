! Test MatCreateNest() with NULL index sets

program main
#include <petsc/finclude/petscmat.h>
  use petscmat
  implicit none

  Mat                   :: A, D, Id, Acopy
  Mat, dimension(4)      :: mats
  Vec                   :: v, w
  PetscInt              :: i, rstart, rend
  PetscInt, parameter    :: n = 6, nb = 2
  PetscScalar, parameter :: one = 1.0, two = 2.0
  PetscErrorCode        :: ierr

  PetscCallA(PetscInitialize(ierr))

  PetscCallA(MatCreateConstantDiagonal(PETSC_COMM_WORLD, PETSC_DETERMINE, PETSC_DETERMINE, n, n, one, Id, ierr))
  PetscCallA(MatCreateVecs(Id, v, w, ierr))
  PetscCallA(VecGetOwnershipRange(v, rstart, rend, ierr))
  do i = rstart, rend - 1
    PetscCallA(VecSetValue(v, i, two/(i + 1), INSERT_VALUES, ierr))
  end do
  PetscCallA(VecAssemblyBegin(v, ierr))
  PetscCallA(VecAssemblyEnd(v, ierr))
  PetscCallA(MatCreateDiagonal(v, D, ierr))

  mats = [PETSC_NULL_MAT, D, Id, PETSC_NULL_MAT]
  PetscCallA(MatCreateNest(PETSC_COMM_WORLD, nb, PETSC_NULL_IS_ARRAY, nb, PETSC_NULL_IS_ARRAY, mats, A, ierr))
  PetscCallA(MatView(A, PETSC_VIEWER_STDOUT_WORLD, ierr))
  PetscCallA(MatNestSetSubMats(A, nb, PETSC_NULL_IS_ARRAY, nb, PETSC_NULL_IS_ARRAY, mats, ierr))
  PetscCallA(MatView(A, PETSC_VIEWER_STDOUT_WORLD, ierr))

  ! test MatCopy()
  PetscCallA(MatDuplicate(A, MAT_DO_NOT_COPY_VALUES, Acopy, ierr))
  PetscCallA(MatCopy(A, Acopy, DIFFERENT_NONZERO_PATTERN, ierr))

  PetscCallA(MatDestroy(Acopy, ierr))
  PetscCallA(MatDestroy(Id, ierr))
  PetscCallA(VecDestroy(v, ierr))
  PetscCallA(VecDestroy(w, ierr))
  PetscCallA(MatDestroy(D, ierr))
  PetscCallA(MatDestroy(A, ierr))
  PetscCallA(PetscFinalize(ierr))

end program

!/*TEST
!
!   test:
!      nsize: 2
!
!TEST*/
