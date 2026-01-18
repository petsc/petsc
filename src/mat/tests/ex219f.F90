#include <petsc/finclude/petscmat.h>
program newnonzero
  use petscmat
  implicit none

  Mat :: A
  PetscInt, parameter :: n = 3, m = n
  PetscInt :: idxm(1), idxn(1), nl1, nl2, i
  PetscScalar :: v(1), value(1), values(2)
  PetscErrorCode :: ierr
  IS :: is
  ISLocalToGlobalMapping :: ismap

  PetscCallA(PetscInitialize(ierr))
  PetscCallA(MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, n, m, 1_PETSC_INT_KIND, PETSC_NULL_INTEGER_ARRAY, 0_PETSC_INT_KIND, PETSC_NULL_INTEGER_ARRAY, A, ierr))

  PetscCallA(MatGetOwnershipRange(A, nl1, nl2, ierr))
  do i = nl1, nl2 - 1
    idxn(1) = i
    idxm(1) = i
    v(1) = 1.0
    PetscCallA(MatSetValues(A, 1_PETSC_INT_KIND, idxn, 1_PETSC_INT_KIND, idxm, v, INSERT_VALUES, ierr))
  end do
  PetscCallA(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY, ierr))
  PetscCallA(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY, ierr))

! Ignore any values set into new nonzero locations
  PetscCallA(MatSetOption(A, MAT_NEW_NONZERO_LOCATIONS, PETSC_FALSE, ierr))

  idxn(1) = 0
  idxm(1) = n - 1
  if ((idxn(1) >= nl1) .and. (idxn(1) <= nl2 - 1)) then
    v(1) = 2.0
    PetscCallA(MatSetValues(A, 1_PETSC_INT_KIND, idxn, 1_PETSC_INT_KIND, idxm, v, INSERT_VALUES, ierr))
  end if
  PetscCallA(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY, ierr))
  PetscCallA(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY, ierr))

  if ((idxn(1) >= nl1) .and. (idxn(1) <= nl2 - 1)) then
    PetscCallA(MatGetValues(A, 1_PETSC_INT_KIND, idxn, 1_PETSC_INT_KIND, idxm, v, ierr))
    write (6, *) PetscRealPart(v)
  end if

  PetscCallA(ISCreateStride(PETSC_COMM_WORLD, nl2 - nl1, nl1, 1_PETSC_INT_KIND, is, ierr))
  PetscCallA(ISLocalToGlobalMappingCreateIS(is, ismap, ierr))
  PetscCallA(MatSetLocalToGlobalMapping(A, ismap, ismap, ierr))
  PetscCallA(ISLocalToGlobalMappingDestroy(ismap, ierr))
  PetscCallA(ISDestroy(is, ierr))
  PetscCallA(MatGetValuesLocal(A, 1_PETSC_INT_KIND, [0_PETSC_INT_KIND], 1_PETSC_INT_KIND, [0_PETSC_INT_KIND], value, ierr))
  PetscCallA(MatGetValuesLocal(A, 1_PETSC_INT_KIND, [0_PETSC_INT_KIND], 1_PETSC_INT_KIND, [0_PETSC_INT_KIND], values, ierr))
  idxn(1) = 0
  PetscCallA(MatGetValuesLocal(A, 1_PETSC_INT_KIND, idxn, 1_PETSC_INT_KIND, [0_PETSC_INT_KIND], values, ierr))
  PetscCallA(MatGetValuesLocal(A, 1_PETSC_INT_KIND, idxn, 1_PETSC_INT_KIND, idxn, values, ierr))

  PetscCallA(MatDestroy(A, ierr))
  PetscCallA(PetscFinalize(ierr))

end program newnonzero

!/*TEST
!
!     test:
!       nsize: 2
!       filter: Error:
!
!     test:
!       requires: defined(PETSC_USE_INFO)
!       suffix: 2
!       nsize: 2
!       args: -info
!       filter: grep "Skipping"
!
!TEST*/
