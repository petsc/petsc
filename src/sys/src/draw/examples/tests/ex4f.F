!
!    "$Id: ex4f.F,v 1.2 1999/09/08 18:56:43 bsmith Exp bsmith $";
!
!
!  This example demonstrates use of DrawZoom()
!
!          This function is called repeatedly by DrawZoom() to
!      redraw the figure
!
      subroutine zoomfunction(draw,dummy,ierr)
#include "include/finclude/petsc.h"
#include "include/finclude/draw.h"
      Draw    draw
      integer dummy,ierr
    
      integer i

      do 10, i=0,255
        call DrawLine(draw,0.d0,i/256.d0,1.d0,i/256.d0,i,ierr)
 10   continue
      return 
      end

#include "include/finclude/petsc.h"
#include "include/finclude/draw.h"
      Draw draw
      integer  ierr, x, y, width, height
      External zoomfunction
      x      = 0
      y      = 0
      width  = 256
      height = 256

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)

      call DrawOpenX(PETSC_COMM_SELF,0,'Title',x,y,width,height,draw,   &
     &               ierr)
      call DrawZoom(draw,zoomfunction,PETSC_NULL_INTEGER,ierr)
      call DrawDestroy(draw,ierr)
      call PetscFinalize(ierr)
      end

 
