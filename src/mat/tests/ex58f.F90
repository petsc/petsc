!
!
!   This program demonstrates use of MatGetRow() and MatGetRowMaxAbs() from Fortran
!
program main
#include <petsc/finclude/petscmat.h>
  use petscmat
  implicit none

  Mat A
  PetscErrorCode ierr
  PetscInt M, N, row, ncol
  PetscViewer v
  Vec rowmax
  PetscBool flg
  IS isrow, iscol
  character*(256) f
  PetscInt, pointer :: cols(:)
  PetscScalar, pointer :: vals(:)

  PetscCallA(PetscInitialize(ierr))

  PetscCallA(PetscOptionsGetString(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, '-f', f, flg, ierr))
  PetscCallA(PetscViewerBinaryOpen(PETSC_COMM_WORLD, f, FILE_MODE_READ, v, ierr))

  PetscCallA(MatCreate(PETSC_COMM_WORLD, A, ierr))
  PetscCallA(MatSetType(A, MATSEQAIJ, ierr))
  PetscCallA(MatLoad(A, v, ierr))

  PetscCallA(MatView(A, PETSC_VIEWER_STDOUT_WORLD, ierr))

  row = 1
  PetscCallA(MatGetRow(A, row, ncol, cols, vals, ierr))
  !print*,cols,vals
  PetscCallA(MatRestoreRow(A, row, ncol, cols, vals, ierr))
  PetscCallA(MatGetRow(A, row, PETSC_NULL_INTEGER, cols, PETSC_NULL_SCALAR_POINTER, ierr))
  !print*,cols
  PetscCallA(MatRestoreRow(A, row, PETSC_NULL_INTEGER, cols, PETSC_NULL_SCALAR_POINTER, ierr))
  PetscCallA(MatGetRow(A, row, PETSC_NULL_INTEGER, PETSC_NULL_INTEGER_POINTER, vals, ierr))
  !print*,vals
  PetscCallA(MatRestoreRow(A, row, PETSC_NULL_INTEGER, PETSC_NULL_INTEGER_POINTER, vals, ierr))
  PetscCallA(MatGetRow(A, row, ncol, PETSC_NULL_INTEGER_POINTER, PETSC_NULL_SCALAR_POINTER, ierr))
  PetscCallA(MatRestoreRow(A, row, ncol, PETSC_NULL_INTEGER_POINTER, PETSC_NULL_SCALAR_POINTER, ierr))

!
!     Test MatGetRowMaxAbs()
  PetscCallA(MatGetSize(A, M, N, ierr))
  PetscCallA(VecCreate(PETSC_COMM_WORLD, rowmax, ierr))
  PetscCallA(VecSetSizes(rowmax, M, M, ierr))
  PetscCallA(VecSetFromOptions(rowmax, ierr))

  PetscCallA(MatGetRowMaxAbs(A, rowmax, PETSC_NULL_INTEGER_ARRAY, ierr))
  PetscCallA(VecView(rowmax, PETSC_VIEWER_STDOUT_WORLD, ierr))

  PetscCallA(MatGetRowMax(A, rowmax, PETSC_NULL_INTEGER_ARRAY, ierr))
  PetscCallA(VecView(rowmax, PETSC_VIEWER_STDOUT_WORLD, ierr))

  PetscCallA(MatGetRowMinAbs(A, rowmax, PETSC_NULL_INTEGER_ARRAY, ierr))
  PetscCallA(VecView(rowmax, PETSC_VIEWER_STDOUT_WORLD, ierr))

  PetscCallA(MatGetRowMin(A, rowmax, PETSC_NULL_INTEGER_ARRAY, ierr))
  PetscCallA(VecView(rowmax, PETSC_VIEWER_STDOUT_WORLD, ierr))

  PetscCallA(MatGetOwnershipIS(A, isrow, iscol, ierr))
  PetscCallA(ISDestroy(isrow, ierr))
  PetscCallA(ISDestroy(iscol, ierr))
  PetscCallA(MatGetOwnershipIS(A, PETSC_NULL_IS, PETSC_NULL_IS, ierr))

  PetscCallA(MatDestroy(A, ierr))
  PetscCallA(PetscViewerDestroy(v, ierr))
  PetscCallA(VecDestroy(rowmax, ierr))

  PetscCallA(PetscFinalize(ierr))
end

!/*TEST
!
!     test:
!       args: -f ${DATAFILESPATH}/matrices/tiny
!       requires: datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES)
!
!TEST*/
