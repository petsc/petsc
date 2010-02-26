!
!
      program main
#include "finclude/petscsys.h"
#include "finclude/petscdraw.h"
!
!  This example demonstrates basic use of the Fortran interface for
!  PetscDraw routines.
!
      PetscDraw         draw
      PetscDrawLG       lg
      PetscDrawAxis     axis
      PetscErrorCode    ierr
      PetscTruth        flg
      integer           x,y,width,height
      PetscScalar       xd,yd
      PetscInt          i,n,w,h

      n      = 20
      x      = 0
      y      = 0
      w      = 300
      h      = 300

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)

!  GetInt requires a PetscInt so have to do this ugly setting
      call PetscOptionsGetInt(PETSC_NULL_CHARACTER,'-width',w,          &
     &     flg,ierr) 
      width = w
      call PetscOptionsGetInt(PETSC_NULL_CHARACTER,'-height',h,         &
     &     flg,ierr)
      height = h
      call PetscOptionsGetInt(PETSC_NULL_CHARACTER,'-n',n,flg,ierr)

      call PetscDrawCreate(PETSC_COMM_SELF,PETSC_NULL_CHARACTER,        &
     &               PETSC_NULL_CHARACTER,x,y,width,height,draw,ierr)
      call PetscDrawSetType(draw,PETSC_DRAW_X,ierr)

      call PetscDrawLGCreate(draw,1,lg,ierr)
      call PetscDrawLGGetAxis(lg,axis,ierr)
      call PetscDrawAxisSetColors(axis,PETSC_DRAW_BLACK,PETSC_DRAW_RED, &
     &     PETSC_DRAW_BLUE,ierr)
      call PetscDrawAxisSetLabels(axis,'toplabel','xlabel','ylabel',    &
     &     ierr)

      do 10, i=0,n-1
        xd = i - 5.0
        yd = xd*xd
        call PetscDrawLGAddPoint(lg,xd,yd,ierr)
 10   continue

      call PetscDrawLGIndicateDataPoints(lg,ierr)
      call PetscDrawLGDraw(lg,ierr)
      call PetscDrawFlush(draw,ierr)

      call PetscSleep(10,ierr)

      call PetscDrawLGDestroy(lg,ierr)
      call PetscDrawDestroy(draw,ierr)
      call PetscFinalize(ierr)
      end
 
