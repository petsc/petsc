!
!
#include <petsc/finclude/petscsys.h>
#include <petsc/finclude/petscdraw.h>
program main
  use petscsys
  use petscdraw
  implicit none
!
!  This example demonstrates basic use of the Fortran interface for
!  PetscDraw routines.
!
  PetscDraw draw
  PetscDrawLG lg
  PetscDrawAxis axis
  PetscErrorCode ierr
  PetscBool flg
  integer4, parameter :: x = 0, y = 0
  integer4 width, height
  PetscReal xd, yd
  PetscInt i, n, w, h

  PetscCallA(PetscInitialize(ierr))

!  GetInt requires a PetscInt so have to do this ugly setting
  w = 400
  PetscCallA(PetscOptionsGetInt(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, '-width', w, flg, ierr))
  width = int(w, kind=kind(width))
  h = 300
  PetscCallA(PetscOptionsGetInt(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, '-height', h, flg, ierr))
  height = int(h, kind=kind(height))
  n = 15
  PetscCallA(PetscOptionsGetInt(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, '-n', n, flg, ierr))

  PetscCallA(PetscDrawCreate(PETSC_COMM_WORLD, PETSC_NULL_CHARACTER, PETSC_NULL_CHARACTER, x, y, width, height, draw, ierr))
  PetscCallA(PetscDrawSetFromOptions(draw, ierr))

  PetscCallA(PetscDrawLGCreate(draw, 1_PETSC_INT_KIND, lg, ierr))
  PetscCallA(PetscDrawLGGetAxis(lg, axis, ierr))
  PetscCallA(PetscDrawAxisSetColors(axis, PETSC_DRAW_BLACK, PETSC_DRAW_RED, PETSC_DRAW_BLUE, ierr))
  PetscCallA(PetscDrawAxisSetLabels(axis, 'toplabel', 'xlabel', 'ylabel', ierr))

  do i = 0, n - 1
    xd = real(i) - 5.0
    yd = xd**2
    PetscCallA(PetscDrawLGAddPoint(lg, xd, yd, ierr))
  end do

  PetscCallA(PetscDrawLGSetUseMarkers(lg, PETSC_TRUE, ierr))
  PetscCallA(PetscDrawLGDraw(lg, ierr))

  PetscCallA(PetscSleep(10._PETSC_REAL_KIND, ierr))

  PetscCallA(PetscDrawLGDestroy(lg, ierr))
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
