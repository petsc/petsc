!
!    "$Id: ex5.F,v 1.22 1999/11/24 21:52:55 bsmith Exp bsmith $";
!
#include "include/finclude/petsc.h"
#include "include/finclude/draw.h"
!
!  This example demonstrates basic use of the Fortran interface for
!  Draw routines.
!
      Draw              draw
      DrawLG            lg
      DrawAxis          axis
      integer           n,i,ierr,x,y,width,height,flg
      Scalar            xd,yd

      n      = 20
      x      = 0
      y      = 0
      width  = 300
      height = 300

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)

      call OptionsGetInt(PETSC_NULL_CHARACTER,'-width',width,flg,ierr) 
      call OptionsGetInt(PETSC_NULL_CHARACTER,'-height',height,flg,ierr)
      call OptionsGetInt(PETSC_NULL_CHARACTER,'-n',n,flg,ierr)

!     call DrawOpenX(PETSC_COMM_SELF,PETSC_NULL_CHARACTER,              
!                   PETSC_NULL_CHARACTER,x,y,width,height,draw,ierr)
      call DrawCreate(PETSC_COMM_SELF,PETSC_NULL_CHARACTER,             &
     &               PETSC_NULL_CHARACTER,x,y,width,height,draw,ierr)
      call DrawSetType(draw,DRAW_X,ierr)

      call DrawLGCreate(draw,1,lg,ierr)
      call DrawLGGetAxis(lg,axis,ierr)
      call DrawAxisSetColors(axis,DRAW_BLACK,DRAW_RED,DRAW_BLUE,ierr)
      call DrawAxisSetLabels(axis,'toplabel','xlabel','ylabel',ierr)

      do 10, i=0,n-1
        xd = i - 5.0
        yd = xd*xd
        call DrawLGAddPoint(lg,xd,yd,ierr)
 10   continue

      call DrawLGIndicateDataPoints(lg,ierr)
      call DrawLGDraw(lg,ierr)
      call DrawFlush(draw,ierr)

      call PetscSleep(10,ierr)

      call DrawLGDestroy(lg,ierr)
      call DrawDestroy(draw,ierr)
      call PetscFinalize(ierr)
      end
 
