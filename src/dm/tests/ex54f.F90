! test verifies DMShellSetCreateFieldDecomposition interface in Fortran
program main
#include "petsc/finclude/petsc.h"
#include "petsc/finclude/petscdmshell.h"

  use petsc
  use petscdmshell
  implicit none
  type(tDM)          :: dm
  PetscErrorCode     :: ierr
  interface
    subroutine myFieldDecomp(dm, nfields, fieldNames, isFields, subDms, ierr)
      use petsc
      implicit none
      type(tDM), intent(in) :: dm
      PetscInt, intent(out) :: nfields
      character(len=30), allocatable, intent(out) :: fieldNames(:)
      type(tIS), allocatable, intent(out) :: isFields(:)
      type(tDM), allocatable, intent(out) :: subDms(:)
      PetscErrorCode, intent(out) :: ierr
    end subroutine myFieldDecomp
  end interface
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
end program main

! a simple Fortran callback for field decomposition.
subroutine myFieldDecomp(dm, nfields, fieldNames, isFields, subDms, ierr)
  use petsc
  implicit none
  type(tDM), intent(in) :: dm
  PetscInt, intent(out) :: nfields
  character(len=30), allocatable, intent(out) :: fieldNames(:)
  type(tIS), allocatable, intent(out) :: isFields(:)
  type(tDM), allocatable, intent(out) :: subDms(:)
  PetscErrorCode, intent(out) :: ierr
  PetscInt :: i
  ! defining a simple decomposition with two fields
  nfields = 2
  allocate (fieldNames(nfields))
  allocate (isFields(nfields))
  allocate (subDms(nfields))
  fieldNames(1) = 'field1'
  fieldNames(2) = 'field2'
  ! set the pointer arrays to NULL (using pointer assignment)
  do i = 1, nfields
    isFields(i) = PETSC_NULL_IS
    subDms(i) = PETSC_NULL_DM
  end do
  ierr = 0
  print *, 'myFieldDecomp callback invoked.'
end subroutine myFieldDecomp
!/*TEST
!
!   test:
!TEST*/
