!     Test code contributed by Thibaut Appel <t.appel17@imperial.ac.uk>
#include <petsc/finclude/petscmat.h>
program test_assembly
  use petscmat

  implicit none
  PetscInt, parameter :: n = 10
  Mat      :: L
  PetscInt :: istart, iend, row
  PetscErrorCode :: ierr
  PetscInt cols(1), rows(1)
  PetscScalar vals(1)

  PetscCallA(PetscInitialize(ierr))

  PetscCallA(MatCreate(PETSC_COMM_WORLD, L, ierr))
  PetscCallA(MatSetType(L, MATAIJ, ierr))
  PetscCallA(MatSetSizes(L, PETSC_DECIDE, PETSC_DECIDE, n, n, ierr))

  PetscCallA(MatSeqAIJSetPreallocation(L, 1_PETSC_INT_KIND, PETSC_NULL_INTEGER_ARRAY, ierr))
  PetscCallA(MatMPIAIJSetPreallocation(L, 1_PETSC_INT_KIND, PETSC_NULL_INTEGER_ARRAY, 0_PETSC_INT_KIND, PETSC_NULL_INTEGER_ARRAY, ierr)) ! No allocated non-zero in off-diagonal part
  PetscCallA(MatSetOption(L, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE, ierr))
  PetscCallA(MatSetOption(L, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE, ierr))
  PetscCallA(MatSetOption(L, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE, ierr))

  PetscCallA(MatGetOwnershipRange(L, istart, iend, ierr))

  ! assembling a diagonal matrix
  do row = istart, iend - 1
    cols = [row]; vals = [1.0]; rows = [row]
    PetscCallA(MatSetValues(L, 1_PETSC_INT_KIND, rows, 1_PETSC_INT_KIND, cols, vals, ADD_VALUES, ierr))
  end do

  PetscCallA(MatAssemblyBegin(L, MAT_FINAL_ASSEMBLY, ierr))
  PetscCallA(MatAssemblyEnd(L, MAT_FINAL_ASSEMBLY, ierr))

  PetscCallA(MatSetOption(L, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE, ierr))

  !PetscCallA(MatZeroEntries(L,ierr))

  ! assembling a diagonal matrix, adding a zero value to non-diagonal part
  do row = istart, iend - 1

    if (row == 0) then
      cols = [n - 1]
      vals = [0.0]
      rows = [row]
      PetscCallA(MatSetValues(L, 1_PETSC_INT_KIND, rows, 1_PETSC_INT_KIND, cols, vals, ADD_VALUES, ierr))
    end if
    cols = [row]; vals = [1.0]; rows = [row]
    PetscCallA(MatSetValues(L, 1_PETSC_INT_KIND, rows, 1_PETSC_INT_KIND, cols, vals, ADD_VALUES, ierr))

  end do

  PetscCallA(MatAssemblyBegin(L, MAT_FINAL_ASSEMBLY, ierr))
  PetscCallA(MatAssemblyEnd(L, MAT_FINAL_ASSEMBLY, ierr))
  PetscCallA(MatDestroy(L, ierr))

  PetscCallA(PetscFinalize(ierr))

end program test_assembly

!/*TEST
!
!   build:
!      requires: complex
!
!   test:
!      nsize: 2
!      output_file: output/empty.out
!
!TEST*/
