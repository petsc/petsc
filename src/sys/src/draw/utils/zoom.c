/*$Id: zoom.c,v 1.16 2000/09/22 20:42:13 bsmith Exp bsmith $*/

#include "petscdraw.h"     /*I "petscdraw.h"  I*/

#undef __FUNC__  
#define __FUNC__ /*<a name="DrawZoom"></a>*/"DrawZoom" 
/*@C
    DrawZoom - Allows one to create a graphic that users may zoom into.

    Collective on Draw

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
int DrawZoom(Draw draw,int (*func)(Draw,void *),void *ctx)
{
  int        ierr,pause;
  DrawButton button;
  PetscReal  xc,yc,scale = 1.0,w,h,xr,xl,yr,yl,xmin,xmax,ymin,ymax;
  PetscTruth isnull;

  PetscFunctionBegin;
  ierr = DrawIsNull(draw,&isnull);CHKERRQ(ierr);
  if (isnull) PetscFunctionReturn(0);

  ierr = DrawSynchronizedClear(draw);CHKERRQ(ierr);
  ierr = (*func)(draw,ctx);CHKERRQ(ierr);
  ierr = DrawSynchronizedFlush(draw);CHKERRQ(ierr);

  ierr = DrawGetPause(draw,&pause);CHKERRQ(ierr);
  if (pause >= 0) { 
    ierr = PetscSleep(pause);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  ierr = DrawCheckResizedWindow(draw);CHKERRQ(ierr);
  ierr = DrawSynchronizedGetMouseButton(draw,&button,&xc,&yc,0,0);CHKERRQ(ierr); 
  ierr = DrawGetCoordinates(draw,&xl,&yl,&xr,&yr);CHKERRQ(ierr);
  w    = xr - xl; xmin = xl; ymin = yl; xmax = xr; ymax = yr;
  h    = yr - yl;

  if (button != BUTTON_NONE) {
    while (button != BUTTON_RIGHT) {

      ierr = DrawSynchronizedClear(draw);CHKERRQ(ierr);
      if (button == BUTTON_LEFT)        scale = .5;
      else if (button == BUTTON_CENTER) scale = 2.;
      xl = scale*(xl + w - xc) + xc - w*scale;
      xr = scale*(xr - w - xc) + xc + w*scale;
      yl = scale*(yl + h - yc) + yc - h*scale;
      yr = scale*(yr - h - yc) + yc + h*scale;
      w *= scale; h *= scale;
      ierr = DrawSetCoordinates(draw,xl,yl,xr,yr);CHKERRQ(ierr);

      ierr = (*func)(draw,ctx);CHKERRQ(ierr);
      ierr = DrawCheckResizedWindow(draw);CHKERRQ(ierr);
      ierr = DrawSynchronizedGetMouseButton(draw,&button,&xc,&yc,0,0);CHKERRQ(ierr);
    }
  }

  ierr = DrawSetCoordinates(draw,xmin,ymin,xmax,ymax);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

