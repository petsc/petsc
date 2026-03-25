!
!
!  This example demonstrates use of PetscDrawZoom()
!
!          This function is called repeatedly by PetscDrawZoom() to
!      redraw the figure
!
#include <petsc/finclude/petscsys.h>
#include <petsc/finclude/petscdraw.h>
module ex4fmodule
  use petscsys
  use petscdraw
  implicit none
contains
  subroutine zoomfunction(draw, dummy, ierr)

    PetscReal val
    PetscInt, parameter :: max = 256
    PetscDraw draw
    integer dummy
    PetscErrorCode, intent(out) :: ierr
    PetscInt32 i

    do i = 0, max - 1
      val = real(i, PETSC_REAL_KIND)/real(max, PETSC_REAL_KIND)
      PetscCall(PetscDrawLine(draw, 0.0_PETSC_REAL_KIND, val, 1.0_PETSC_REAL_KIND, val, i, ierr))
    end do
    ierr = 0
  end
end module
program main
  use petscsys
  use petscdraw
  use ex4fmodule
  implicit none

  PetscDraw draw
  PetscErrorCode ierr
  integer4, parameter :: x = 0, y = 0, width = 256, height = 256

  PetscCallA(PetscInitialize(ierr))
  PetscCallA(PetscDrawCreate(PETSC_COMM_WORLD, PETSC_NULL_CHARACTER, 'Title', x, y, width, height, draw, ierr))
  PetscCallA(PetscDrawSetFromOptions(draw, ierr))
  PetscCallA(PetscDrawZoom(draw, zoomfunction, PETSC_NULL_INTEGER, ierr))
  PetscCallA(PetscDrawDestroy(draw, ierr))
  PetscCallA(PetscFinalize(ierr))
end

!/*TEST
!
!   build:
!     requires: x
!
!   test:
!     output_file: output/empty.out
!
!TEST*/
