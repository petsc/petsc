!
!
      program main
#include "include/finclude/petsc.h"
#include "include/finclude/petscdraw.h"
!
!  This example demonstrates basic use of the Fortran interface for
!  PetscDraw routines.
!
      PetscDraw         draw
      PetscDrawLG       lg
      PetscDrawAxis     axis
      integer           n,i,ierr,x,y,width,height,flg
      PetscScalar       xd,yd

      n      = 20
      x      = 0
      y      = 0
      width  = 300
      height = 300

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)

      call PetscOptionsGetInt(PETSC_NULL_CHARACTER,'-width',width,      &
     &     flg,ierr) 
      call PetscOptionsGetInt(PETSC_NULL_CHARACTER,'-height',height,    &
     &     flg,ierr)
      call PetscOptionsGetInt(PETSC_NULL_CHARACTER,'-n',n,flg,ierr)

!     call PetscDrawOpenX(PETSC_COMM_SELF,PETSC_NULL_CHARACTER,         &
!                   PETSC_NULL_CHARACTER,x,y,width,height,draw,ierr)
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
 
