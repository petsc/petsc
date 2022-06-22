!
!
      program main
#include <petsc/finclude/petscsys.h>
#include <petsc/finclude/petscdraw.h>
      use petscsys
      implicit none
!
!  This example demonstrates basic use of the Fortran interface for
!  PetscDraw routines.
!
      PetscDraw         draw
      PetscDrawLG       lg
      PetscDrawAxis     axis
      PetscErrorCode    ierr
      PetscBool         flg
      integer           x,y,width,height
      PetscScalar       xd,yd
      PetscReal         ten
      PetscInt          i,n,w,h
      PetscInt          one

      n      = 15
      x      = 0
      y      = 0
      w      = 400
      h      = 300
      ten    = 10.0
      one    = 1

      PetscCallA(PetscInitialize(ierr))

!  GetInt requires a PetscInt so have to do this ugly setting
      PetscCallA(PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-width',w, flg,ierr))
      width = int(w)
      PetscCallA(PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-height',h,flg,ierr))
      height = int(h)
      PetscCallA(PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-n',n,flg,ierr))

      PetscCallA(PetscDrawCreate(PETSC_COMM_WORLD,PETSC_NULL_CHARACTER,PETSC_NULL_CHARACTER,x,y,width,height,draw,ierr))
      PetscCallA(PetscDrawSetFromOptions(draw,ierr))

      PetscCallA(PetscDrawLGCreate(draw,one,lg,ierr))
      PetscCallA(PetscDrawLGGetAxis(lg,axis,ierr))
      PetscCallA(PetscDrawAxisSetColors(axis,PETSC_DRAW_BLACK,PETSC_DRAW_RED,PETSC_DRAW_BLUE,ierr))
      PetscCallA(PetscDrawAxisSetLabels(axis,'toplabel','xlabel','ylabel',ierr))

      do 10, i=0,n-1
        xd = real(i) - 5.0
        yd = xd*xd
        PetscCallA(PetscDrawLGAddPoint(lg,xd,yd,ierr))
 10   continue

      PetscCallA(PetscDrawLGSetUseMarkers(lg,PETSC_TRUE,ierr))
      PetscCallA(PetscDrawLGDraw(lg,ierr))

      PetscCallA(PetscSleep(ten,ierr))

      PetscCallA(PetscDrawLGDestroy(lg,ierr))
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
