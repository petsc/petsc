! test verifies DMShellSetCreateFieldDecomposition interface in Fortran
#include "petsc/finclude/petsc.h"

module ex54fmodule
  use petsc
  implicit none

contains
  ! a simple Fortran callback for field decomposition.
  subroutine myFieldDecomp(dm, nfields, fieldNames, isFields, subDms, ierr)
    type(tDM), intent(in) :: dm
    PetscInt, intent(out) :: nfields
    character(len=30), allocatable, intent(out) :: fieldNames(:)
    type(tIS), allocatable, intent(out) :: isFields(:)
    type(tDM), allocatable, intent(out) :: subDms(:)
    PetscErrorCode, intent(out) :: ierr
    ! defining a simple decomposition with two fields
    nfields = 2
    allocate (fieldNames(nfields))
    allocate (isFields(nfields))
    allocate (subDms(nfields))
    fieldNames(1) = 'field1'
    fieldNames(2) = 'field2'
    ! set the pointer arrays to NULL (using pointer assignment)
    isFields(1:nfields) = PETSC_NULL_IS
    subDms(1:nfields) = PETSC_NULL_DM
    ierr = 0
    print *, 'myFieldDecomp callback invoked.'
  end subroutine myFieldDecomp
end module ex54fmodule

program ex54f
  use petsc
  use ex54fmodule
  implicit none
  type(tDM)          :: dm
  PetscErrorCode     :: ierr
  ! initializing PETSc
  PetscCallA(PetscInitialize(PETSC_NULL_CHARACTER, ierr))
  ! creating a DMShell object
  PetscCallA(DMShellCreate(PETSC_COMM_WORLD, dm, ierr))
  ! registering the Fortran field decomposition callback
  PetscCallA(DMShellSetCreateFieldDecomposition(dm, myFieldDecomp, ierr))
  ! for this minimal test, we simply print a success message to the console
  print *, 'DMShellSetCreateFieldDecomposition set successfully.'
  ! cleanup
  PetscCallA(DMDestroy(dm, ierr))
  PetscCallA(PetscFinalize(ierr))
end program ex54f

!/*TEST
!
!   test:
!TEST*/
