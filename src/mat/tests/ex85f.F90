!
!   This program tests MatGetDiagonal()
!
#include <petsc/finclude/petscmat.h>
program main
  use petscmat
  implicit none

  PetscErrorCode ierr
  PetscInt, parameter :: one = 1, twelve = 12
  Vec v
  Mat m
  PetscScalar, parameter :: value = 3.0

  PetscCallA(PetscInitialize(ierr))

  PetscCallA(MatCreate(PETSC_COMM_SELF, m, ierr))
  PetscCallA(MatSetSizes(m, twelve, twelve, twelve, twelve, ierr))
  PetscCallA(MatSetFromOptions(m, ierr))
  PetscCallA(MatSetUp(m, ierr))

  PetscCallA(MatSetValues(m, one, [4_PETSC_INT_KIND], one, [4_PETSC_INT_KIND], [value], INSERT_VALUES, ierr))
  PetscCallA(MatAssemblyBegin(m, MAT_FINAL_ASSEMBLY, ierr))
  PetscCallA(MatAssemblyEnd(m, MAT_FINAL_ASSEMBLY, ierr))

  PetscCallA(VecCreateSeq(PETSC_COMM_SELF, twelve, v, ierr))
  PetscCallA(MatGetDiagonal(m, v, ierr))
  PetscCallA(VecView(v, PETSC_VIEWER_STDOUT_WORLD, ierr))

  PetscCallA(MatDestroy(m, ierr))
  PetscCallA(VecDestroy(v, ierr))
  PetscCallA(PetscFinalize(ierr))
end

!/*TEST
!
!   test:
!
!TEST*/
