!
!   This program tests MatNullSpaceCreate()
!
program main
#include <petsc/finclude/petscmat.h>
  use petscmat
  implicit none

  PetscErrorCode ierr
  MatNullSpace nsp
  Vec v(1)
  PetscInt nloc, on
  PetscScalar one
  PetscReal norm
  Vec, pointer :: vnsp(:)

  PetscCallA(PetscInitialize(ierr))

  nloc = 12
  on = 1
  PetscCallA(VecCreateFromOptions(PETSC_COMM_WORLD, PETSC_NULL_CHARACTER, on, nloc, PETSC_DETERMINE, v(1), ierr))
  one = 1.0
  PetscCallA(VecSet(v(1), one, ierr))
  PetscCallA(VecNormalize(v(1), norm, ierr))
  PetscCallA(MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_FALSE, on, [v], nsp, ierr))
  PetscCallA(MatNullSpaceGetVecs(nsp, PETSC_NULL_BOOL, PETSC_NULL_INTEGER, vnsp, ierr))
  PetscCallA(MatNullSpaceRestoreVecs(nsp, PETSC_NULL_BOOL, PETSC_NULL_INTEGER, vnsp, ierr))
  PetscCallA(MatNullSpaceDestroy(nsp, ierr))
  PetscCallA(VecDestroy(v(1), ierr))
  PetscCallA(PetscFinalize(ierr))
end

!/*TEST
!
!   test:
!      output_file: output/empty.out
!
!TEST*/
