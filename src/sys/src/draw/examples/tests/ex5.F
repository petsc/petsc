
C    "$Id: ex5.F,v 1.6 1995/11/09 22:31:45 bsmith Exp bsmith $";

#include "include/finclude/petsc.h"
#include "include/finclude/draw.h"
C
C  This example demonstrates basic use of the Fortran interface for
C  Draw routines.
C
      Draw              draw
      DrawLG            lg
      DrawAxis          axis
      integer           n,i, ierr, x, y, width, height
      double precision  xd,yd

      n      = 20
      x      = 0
      y      = 0
      width  = 300
      height = 300

      call PetscInitialize(ierr)

      call OptionsGetInt(PetscNull,'-width',width,ierr) 
      call OptionsGetInt(PetscNull,'-height',height,ierr)
      call OptionsGetInt(PetscNull,'-n',n,ierr)

      call DrawOpenX(MPI_COMM_SELF,PetscNull,PetscNull,x,y,width,height,draw,ierr)

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

      call PetscSleep(10)

      call DrawLGDestroy(lg,ierr)
      call PetscFinalize(ierr)
      stop 
      end
 
