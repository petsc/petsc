!
!   This program demonstrates use of MatCreateSubMatrices() from Fortran
!
program main
#include <petsc/finclude/petscmat.h>
  use petscmat
  implicit none

  Mat A
  Mat, pointer :: B(:)
  PetscErrorCode ierr
  PetscInt nis, zero(1)
  PetscViewer v
  IS isrow
  PetscMPIInt rank

  PetscCallA(PetscInitialize(ierr))
  PetscCallMPIA(MPI_Comm_rank(PETSC_COMM_WORLD, rank, ierr))

#if defined(PETSC_USE_64BIT_INDICES)
  PetscCallA(PetscViewerBinaryOpen(PETSC_COMM_WORLD, '${PETSC_DIR}/share/petsc/datafiles/matrices/'//'ns-real-int64-float64', FILE_MODE_READ, v, ierr))
#else
  PetscCallA(PetscViewerBinaryOpen(PETSC_COMM_WORLD, '${PETSC_DIR}/share/petsc/datafiles/matrices/'//'ns-real-int32-float64', FILE_MODE_READ, v, ierr))
#endif

  PetscCallA(MatCreate(PETSC_COMM_WORLD, A, ierr))
  PetscCallA(MatSetType(A, MATAIJ, ierr))
  PetscCallA(MatLoad(A, v, ierr))

  nis = 1
  zero(1) = 0
  if (rank == 1) then
    nis = 0 ! test nis = 0
  end if
  PetscCallA(ISCreateGeneral(PETSC_COMM_SELF, nis, zero, PETSC_COPY_VALUES, isrow, ierr))

  PetscCallA(MatCreateSubmatrices(A, nis, [isrow], [isrow], MAT_INITIAL_MATRIX, B, ierr))

  if (rank == 0) then
    PetscCallA(MatView(B(1), PETSC_VIEWER_STDOUT_SELF, ierr))
  end if

  PetscCallA(MatCreateSubmatrices(A, nis, [isrow], [isrow], MAT_REUSE_MATRIX, B, ierr))

  if (rank == 0) then
    PetscCallA(MatView(B(1), PETSC_VIEWER_STDOUT_SELF, ierr))
  end if

  PetscCallA(ISDestroy(isrow, ierr))
  PetscCallA(MatDestroy(A, ierr))
  PetscCallA(MatDestroySubMatrices(nis, B, ierr))
  PetscCallA(PetscViewerDestroy(v, ierr))

  PetscCallA(PetscFinalize(ierr))
end

!/*TEST
!
!     test:
!        requires: double !complex
!
!TEST*/
