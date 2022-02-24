
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
  PetscDrawButton button;
  PetscReal       dpause,xc,yc,scale = 1.0,w,h,xr,xl,yr,yl,xmin,xmax,ymin,ymax;
  PetscBool       isnull;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  CHKERRQ(PetscDrawIsNull(draw,&isnull));
  if (isnull) PetscFunctionReturn(0);

  CHKERRQ(PetscDrawCheckResizedWindow(draw));
  CHKERRQ(PetscDrawClear(draw));
  ierr = PetscDrawCollectiveBegin(draw);CHKERRQ(ierr);
  CHKERRQ((*func)(draw,ctx));
  ierr = PetscDrawCollectiveEnd(draw);CHKERRQ(ierr);
  CHKERRQ(PetscDrawFlush(draw));

  CHKERRQ(PetscDrawGetPause(draw,&dpause));
  if (dpause >= 0) {
    CHKERRQ(PetscSleep(dpause));
    goto theend;
  }
  if (dpause != -1) goto theend;

  CHKERRQ(PetscDrawGetMouseButton(draw,&button,&xc,&yc,NULL,NULL));
  CHKERRQ(PetscDrawGetCoordinates(draw,&xl,&yl,&xr,&yr));
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
    CHKERRQ(PetscDrawClear(draw));
    CHKERRQ(PetscDrawSetCoordinates(draw,xl,yl,xr,yr));
    ierr = PetscDrawCollectiveBegin(draw);CHKERRQ(ierr);
    CHKERRQ((*func)(draw,ctx));
    ierr = PetscDrawCollectiveEnd(draw);CHKERRQ(ierr);
    CHKERRQ(PetscDrawFlush(draw));
    CHKERRQ(PetscDrawGetMouseButton(draw,&button,&xc,&yc,NULL,NULL));
  }
  CHKERRQ(PetscDrawSetCoordinates(draw,xmin,ymin,xmax,ymax));
theend:
  PetscFunctionReturn(0);
}
