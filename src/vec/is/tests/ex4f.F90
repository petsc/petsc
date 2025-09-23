!
!     Test for bug with ISGetIndices() when length of indices is 0
!
!     Contributed by: Jakub Fabian
!
program main
#include <petsc/finclude/petscis.h>
  use petscis
  implicit none

  PetscErrorCode ierr
  PetscInt n, bs
  PetscInt, pointer :: indices(:) => NULL()
  PetscInt, pointer :: idx(:) => NULL()
  IS is

  n = 0
  allocate (indices(n), source=n)

  PetscCallA(PetscInitialize(ierr))

  PetscCallA(ISCreateGeneral(PETSC_COMM_SELF, n, indices, PETSC_USE_POINTER, is, ierr))
  PetscCallA(ISGetIndices(is, idx, ierr))
  PetscCallA(ISRestoreIndices(is, idx, ierr))
  PetscCallA(ISDestroy(is, ierr))

  bs = 2
  PetscCallA(ISCreateBlock(PETSC_COMM_SELF, bs, n, indices, PETSC_USE_POINTER, is, ierr))
  PetscCallA(ISGetIndices(is, idx, ierr))
  PetscCallA(ISRestoreIndices(is, idx, ierr))
  PetscCallA(ISDestroy(is, ierr))
  PetscCallA(PetscFinalize(ierr))
end

!/*TEST
!
!   test:
!      output_file: output/empty.out
!
!TEST*/
