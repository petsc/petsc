#define PETSC_DLL

#include "petscdraw.h"     /*I "petscdraw.h"  I*/

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawZoom" 
/*@C
    PetscDrawZoom - Allows one to create a graphic that users may zoom into.

    Collective on PetscDraw

    Input Parameters:
+   draw - the window where the graph will be made.
.   func - users function that draws the graphic
-   ctx - pointer to any user required data

  Level: advanced

  Concepts: graphics^zooming
  Concepts: drawing^zooming
  Concepts: zooming^in graphics

.seealso:  
@*/
PetscErrorCode PETSC_DLLEXPORT PetscDrawZoom(PetscDraw draw,PetscErrorCode (*func)(PetscDraw,void *),void *ctx)
{
  PetscErrorCode  ierr;
  PetscDrawButton button;
  PetscReal       dpause,xc,yc,scale = 1.0,w,h,xr,xl,yr,yl,xmin,xmax,ymin,ymax;
  PetscTruth      isnull;

  PetscFunctionBegin;
  ierr = PetscDrawIsNull(draw,&isnull);CHKERRQ(ierr);
  if (isnull) PetscFunctionReturn(0);

  ierr = PetscDrawSynchronizedClear(draw);CHKERRQ(ierr);
  ierr = (*func)(draw,ctx);CHKERRQ(ierr);
  ierr = PetscDrawSynchronizedFlush(draw);CHKERRQ(ierr);

  ierr = PetscDrawGetPause(draw,&dpause);CHKERRQ(ierr);
  if (dpause >= 0) { 
    ierr = PetscSleep(dpause);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  ierr = PetscDrawCheckResizedWindow(draw);CHKERRQ(ierr);
  ierr = PetscDrawSynchronizedGetMouseButton(draw,&button,&xc,&yc,0,0);CHKERRQ(ierr); 
  ierr = PetscDrawGetCoordinates(draw,&xl,&yl,&xr,&yr);CHKERRQ(ierr);
  w    = xr - xl; xmin = xl; ymin = yl; xmax = xr; ymax = yr;
  h    = yr - yl;

  if (button != BUTTON_NONE) {
    while (button != BUTTON_RIGHT) {

      ierr = PetscDrawSynchronizedClear(draw);CHKERRQ(ierr);
      if (button == BUTTON_LEFT)        scale = .5;
      else if (button == BUTTON_CENTER) scale = 2.;
      xl = scale*(xl + w - xc) + xc - w*scale;
      xr = scale*(xr - w - xc) + xc + w*scale;
      yl = scale*(yl + h - yc) + yc - h*scale;
      yr = scale*(yr - h - yc) + yc + h*scale;
      w *= scale; h *= scale;
      ierr = PetscDrawSetCoordinates(draw,xl,yl,xr,yr);CHKERRQ(ierr);

      ierr = (*func)(draw,ctx);CHKERRQ(ierr);
      ierr = PetscDrawCheckResizedWindow(draw);CHKERRQ(ierr);
      ierr = PetscDrawSynchronizedGetMouseButton(draw,&button,&xc,&yc,0,0);CHKERRQ(ierr);
    }
  }
  ierr = PetscDrawSetCoordinates(draw,xmin,ymin,xmax,ymax);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

