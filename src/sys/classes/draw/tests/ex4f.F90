!
!
!  This example demonstrates use of PetscDrawZoom()
!
!          This function is called repeatedly by PetscDrawZoom() to
!      redraw the figure
!
      subroutine zoomfunction(draw,dummy,ierr)
#include <petsc/finclude/petscsys.h>
#include <petsc/finclude/petscdraw.h>
      use petscsys
      implicit none
      PetscReal zero, one,value, max
      PetscDraw    draw
      integer dummy
      PetscErrorCode ierr

      PetscInt i

      zero = 0
      one  = 1
      max = 256.0
      do 10, i=0,255
        value = i/max
        PetscCall(PetscDrawLine(draw,zero,value,one,value,i,ierr))
 10   continue
      return
      end

      program main
      use petscsys
      implicit none

      PetscDraw draw
      PetscErrorCode ierr
      integer  x,y,width,height
      External zoomfunction
      x      = 0
      y      = 0
      width  = 256
      height = 256

      PetscCallA(PetscInitialize(ierr))
      PetscCallA(PetscDrawCreate(PETSC_COMM_WORLD,PETSC_NULL_CHARACTER,'Title',x,y,width,height,draw,ierr))
      PetscCallA(PetscDrawSetFromOptions(draw,ierr))
      PetscCallA(PetscDrawZoom(draw,zoomfunction,PETSC_NULL_INTEGER,ierr))
      PetscCallA(PetscDrawDestroy(draw,ierr))
      PetscCallA(PetscFinalize(ierr))
      end

!/*TEST
!
!   build:
!     requires: x
!
!   test:
!     output_file: output/ex1_1.out
!
!TEST*/
