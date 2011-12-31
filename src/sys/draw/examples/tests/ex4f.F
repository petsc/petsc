!
!
!  This example demonstrates use of PetscDrawZoom()
!
!          This function is called repeatedly by PetscDrawZoom() to
!      redraw the figure
!
      subroutine zoomfunction(draw,dummy,ierr)
#include <finclude/petscsys.h>
#include <finclude/petscdraw.h>
      PetscDraw    draw
      integer dummy
      PetscErrorCode ierr
    
      integer i

      do 10, i=0,255
        call PetscDrawLine(draw,0.d0,i/256.d0,1.d0,i/256.d0,i,ierr)
 10   continue
      return 
      end

      program main
#include <finclude/petscsys.h>
#include <finclude/petscdraw.h>
      PetscDraw draw
      PetscErrorCode ierr
      integer  x,y,width,height
      External zoomfunction
      x      = 0
      y      = 0
      width  = 256
      height = 256

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      call PetscDrawCreate(PETSC_COMM_SELF,PETSC_NULL_CHARACTER,        &
     & 'Title',x,y,width,height,draw,ierr)
      call PetscDrawSetType(draw,PETSC_DRAW_X,ierr)
      call PetscDrawZoom(draw,zoomfunction,PETSC_NULL_INTEGER,ierr)
      call PetscDrawDestroy(draw,ierr)
      call PetscFinalize(ierr)
      end

 
