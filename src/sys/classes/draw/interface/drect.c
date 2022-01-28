
/*
       Provides the calling sequences for all the basic PetscDraw routines.
*/
#include <petsc/private/drawimpl.h>  /*I "petscdraw.h" I*/

/*@C
   PetscDrawIndicatorFunction - Draws an indicator function (where a relationship is true) on a PetscDraw

   Not collective

   Input Parameters:
+  draw - a PetscDraw
.  xmin,xmax,ymin,ymax - region to draw indicator function
-  f - the indicator function

   Level: developer

@*/
PetscErrorCode PetscDrawIndicatorFunction(PetscDraw draw,PetscReal xmin,PetscReal xmax,PetscReal ymin,PetscReal ymax,int c,PetscErrorCode (*indicator)(void*,PetscReal,PetscReal,PetscBool*),void *ctx)
{
  int            i,j,xstart,ystart,xend,yend;
  PetscReal      x,y;
  PetscBool      isnull,flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  ierr = PetscDrawIsNull(draw,&isnull);CHKERRQ(ierr);
  if (isnull) PetscFunctionReturn(0);

  ierr = PetscDrawCoordinateToPixel(draw,xmin,ymin,&xstart,&ystart);CHKERRQ(ierr);
  ierr = PetscDrawCoordinateToPixel(draw,xmax,ymax,&xend,&yend);CHKERRQ(ierr);
  if (yend < ystart) { PetscInt tmp = ystart; ystart = yend; yend = tmp; }

  for (i=xstart; i<=xend; i++) {
    for (j=ystart; j<=yend; j++) {
      ierr = PetscDrawPixelToCoordinate(draw,i,j,&x,&y);CHKERRQ(ierr);
      ierr = indicator(ctx,x,y,&flg);CHKERRQ(ierr);
      if (flg) {
        ierr = PetscDrawPointPixel(draw,i,j,c);CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}

/*@C
   PetscDrawCoordinateToPixel - given a coordinate in a PetscDraw returns the pixel location

   Not collective

   Input Parameters:
+  draw - the draw where the coordinates are defined
.  x - the horizontal coordinate
-  y - the vertical coordinate

   Output Parameters:
+  i - the horizontal pixel location
-  j - the vertical pixel location

   Level: developer

@*/
PetscErrorCode PetscDrawCoordinateToPixel(PetscDraw draw,PetscReal x,PetscReal y,int *i,int *j)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  PetscAssertFalse(!draw->ops->coordinatetopixel,PETSC_COMM_SELF,PETSC_ERR_SUP,"This draw type %s does not support locating pixels",((PetscObject)draw)->type_name);
  ierr = (*draw->ops->coordinatetopixel)(draw,x,y,i,j);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscDrawPixelToCoordinate - given a pixel in a PetscDraw returns the coordinate

   Not collective

   Input Parameters:
+  draw - the draw where the coordinates are defined
.  i - the horizontal pixel location
-  j - the vertical pixel location

   Output Parameters:
+  x - the horizontal coordinate
-  y - the vertical coordinate

   Level: developer

@*/
PetscErrorCode PetscDrawPixelToCoordinate(PetscDraw draw,int i,int j,PetscReal *x,PetscReal *y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  PetscAssertFalse(!draw->ops->pixeltocoordinate,PETSC_COMM_SELF,PETSC_ERR_SUP,"This draw type %s does not support locating coordinates",((PetscObject)draw)->type_name);
  ierr = (*draw->ops->pixeltocoordinate)(draw,i,j,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   PetscDrawRectangle - PetscDraws a rectangle  onto a drawable.

   Not Collective

   Input Parameters:
+  draw - the drawing context
.  xl,yl,xr,yr - the coordinates of the lower left, upper right corners
-  c1,c2,c3,c4 - the colors of the four corners in counter clockwise order

   Level: beginner

.seealso: PetscDrawLine(), PetscDrawRectangle(), PetscDrawTriangle(), PetscDrawEllipse(),
          PetscDrawMarker(), PetscDrawPoint(), PetscDrawString(), PetscDrawPoint(), PetscDrawArrow()

@*/
PetscErrorCode  PetscDrawRectangle(PetscDraw draw,PetscReal xl,PetscReal yl,PetscReal xr,PetscReal yr,int c1,int c2,int c3,int c4)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  PetscAssertFalse(!draw->ops->rectangle,PETSC_COMM_SELF,PETSC_ERR_SUP,"This draw type %s does not support drawing rectangles",((PetscObject)draw)->type_name);
  ierr = (*draw->ops->rectangle)(draw,xl,yl,xr,yr,c1,c2,c3,c4);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
