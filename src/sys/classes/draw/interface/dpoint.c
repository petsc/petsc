
/*
       Provides the calling sequences for all the basic PetscDraw routines.
*/
#include <petsc/private/drawimpl.h> /*I "petscdraw.h" I*/

/*@
   PetscDrawPoint - draws a point onto a drawable.

   Not collective

   Input Parameters:
+  draw - the drawing context
.  xl,yl - the coordinates of the point
-  cl - the color of the point

   Level: beginner

.seealso: `PetscDraw`, `PetscDrawPointPixel()`, `PetscDrawPointSetSize()`, `PetscDrawLine()`, `PetscDrawRectangle()`, `PetscDrawTriangle()`, `PetscDrawEllipse()`,
          `PetscDrawMarker()`, `PetscDrawString()`, `PetscDrawArrow()`
@*/
PetscErrorCode PetscDrawPoint(PetscDraw draw, PetscReal xl, PetscReal yl, int cl)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw, PETSC_DRAW_CLASSID, 1);
  PetscUseTypeMethod(draw, point, xl, yl, cl);
  PetscFunctionReturn(0);
}

/*@
   PetscDrawPointPixel - draws a point onto a drawable, in pixel coordinates

   Not collective

   Input Parameters:
+  draw - the drawing context
.  x,y - the pixel coordinates of the point
-  c - the color of the point

   Level: beginner

.seealso: `PetscDraw`, `PetscDrawPoint()`, `PetscDrawPointSetSize()`
@*/
PetscErrorCode PetscDrawPointPixel(PetscDraw draw, int x, int y, int c)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw, PETSC_DRAW_CLASSID, 1);
  PetscUseTypeMethod(draw, pointpixel, x, y, c);
  PetscFunctionReturn(0);
}

/*@
   PetscDrawPointSetSize - Sets the point size for future draws.  The size is
   relative to the user coordinates of the window; 0.0 denotes the natural
   width, 1.0 denotes the entire viewport.

   Not collective

   Input Parameters:
+  draw - the drawing context
-  width - the width in user coordinates

   Level: advanced

   Note:
   Even a size of zero insures that a single pixel is colored.

.seealso: `PetscDraw`, `PetscDrawPoint()`, `PetscDrawMarker()`
@*/
PetscErrorCode PetscDrawPointSetSize(PetscDraw draw, PetscReal width)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw, PETSC_DRAW_CLASSID, 1);
  PetscCheck(width >= 0.0 && width <= 1.0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Bad size %g, should be between 0 and 1", (double)width);
  PetscTryTypeMethod(draw, pointsetsize, width);
  PetscFunctionReturn(0);
}
