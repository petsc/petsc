
#include "include/finclude/petsc.h"
#include "include/finclude/draw.h"

      DrawCtx           draw
      DrawLGCtx         lg
      DrawAxisCtx       axis
      integer           n,i, ierr, x, y, width, height
      double precision  xd,yd

      n      = 20
      x      = 0
      y      = 0
      width  = 300
      height = 300

      call PetscInitialize(ierr)

      call OptionsGetInt(' ','-width',width,ierr) 
      call OptionsGetInt(' ','-height',height,ierr)
      call OptionsGetInt(' ','-n',n,ierr)

      call DrawOpenX(MPI_COMM_SELF,' ',' ',x,y,width,height,draw,ierr)

      call DrawLGCreate(draw,1,lg,ierr)
      call DrawLGGetAxisCtx(lg,axis,ierr)
      call DrawAxisSetColors(axis,DRAW_BLACK,DRAW_RED,DRAW_BLUE,ierr)
      call DrawAxisSetLabels(axis,'toplabel','xlabel','ylabel',ierr)

      do 10, i=0,n-1
        xd = i - 5.0
        yd = xd*xd
        call DrawLGAddPoint(lg,xd,yd,ierr)
 10   continue

      call DrawLGIndicateDataPoints(lg,ierr)
      call DrawLG(lg,ierr)
      call DrawFlush(draw,ierr)

      call PetscSleep(10)

      call DrawLGDestroy(lg,ierr)
      call PetscFinalize(ierr)
      stop 
      end
 
