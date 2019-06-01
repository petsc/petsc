
#include <petscdraw.h>     /*I "petscdraw.h"  I*/

/*@C
    PetscDrawZoom - Allows one to create a graphic that users may zoom into.

    Collective on PetscDraw

    Input Parameters:
+   draw - the window where the graph will be made.
.   func - users function that draws the graphic
-   ctx - pointer to any user required data

  Level: advanced

.seealso:
@*/
PetscErrorCode  PetscDrawZoom(PetscDraw draw,PetscErrorCode (*func)(PetscDraw,void*),void *ctx)
{
  PetscErrorCode  ierr;
  PetscDrawButton button;
  PetscReal       dpause,xc,yc,scale = 1.0,w,h,xr,xl,yr,yl,xmin,xmax,ymin,ymax;
  PetscBool       isnull;

  PetscFunctionBegin;
  ierr = PetscDrawIsNull(draw,&isnull);CHKERRQ(ierr);
  if (isnull) PetscFunctionReturn(0);

  ierr = PetscDrawCheckResizedWindow(draw);CHKERRQ(ierr);
  ierr = PetscDrawClear(draw);CHKERRQ(ierr);
  ierr = PetscDrawCollectiveBegin(draw);CHKERRQ(ierr);
  ierr = (*func)(draw,ctx);CHKERRQ(ierr);
  ierr = PetscDrawCollectiveEnd(draw);CHKERRQ(ierr);
  ierr = PetscDrawFlush(draw);CHKERRQ(ierr);

  ierr = PetscDrawGetPause(draw,&dpause);CHKERRQ(ierr);
  if (dpause >= 0) {
    ierr = PetscSleep(dpause);CHKERRQ(ierr);
    goto theend;
  }
  if (dpause != -1) goto theend;

  ierr = PetscDrawGetMouseButton(draw,&button,&xc,&yc,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscDrawGetCoordinates(draw,&xl,&yl,&xr,&yr);CHKERRQ(ierr);
  xmin = xl; xmax = xr; w = xr - xl;
  ymin = yl; ymax = yr; h = yr - yl;

  while (button != PETSC_BUTTON_NONE && button != PETSC_BUTTON_RIGHT) {
    switch (button) {
    case PETSC_BUTTON_LEFT:       scale = 0.5;   break;
    case PETSC_BUTTON_CENTER:     scale = 2.0;   break;
    case PETSC_BUTTON_WHEEL_UP:   scale = 8/10.; break;
    case PETSC_BUTTON_WHEEL_DOWN: scale = 10/8.; break;
    default:                      scale = 1.0;
    }
    xl = scale*(xl + w - xc) + xc - w*scale;
    xr = scale*(xr - w - xc) + xc + w*scale;
    yl = scale*(yl + h - yc) + yc - h*scale;
    yr = scale*(yr - h - yc) + yc + h*scale;
    w *= scale; h *= scale;
    ierr = PetscDrawClear(draw);CHKERRQ(ierr);
    ierr = PetscDrawSetCoordinates(draw,xl,yl,xr,yr);CHKERRQ(ierr);
    ierr = PetscDrawCollectiveBegin(draw);CHKERRQ(ierr);
    ierr = (*func)(draw,ctx);CHKERRQ(ierr);
    ierr = PetscDrawCollectiveEnd(draw);CHKERRQ(ierr);
    ierr = PetscDrawFlush(draw);CHKERRQ(ierr);
    ierr = PetscDrawGetMouseButton(draw,&button,&xc,&yc,NULL,NULL);CHKERRQ(ierr);
  }
  ierr = PetscDrawSetCoordinates(draw,xmin,ymin,xmax,ymax);CHKERRQ(ierr);
theend:
  PetscFunctionReturn(0);
}

