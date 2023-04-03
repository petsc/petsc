#include <petsc/private/drawimpl.h> /*I "petscdraw.h" I*/

/*@C
   PetscDrawIndicatorFunction - Draws an indicator function (where a relationship is true) on a `PetscDraw`

   Not Collective

   Input Parameters:
+  draw - a `PetscDraw`
.  xmin - region to draw indicator function
.  xmax - region to draw indicator function
.  ymin - region to draw indicator function
.  ymax - region to draw indicator function
-  f - the indicator function

   Level: developer

.seealso: `PetscDraw`
@*/
PetscErrorCode PetscDrawIndicatorFunction(PetscDraw draw, PetscReal xmin, PetscReal xmax, PetscReal ymin, PetscReal ymax, int c, PetscErrorCode (*indicator)(void *, PetscReal, PetscReal, PetscBool *), void *ctx)
{
  int       i, j, xstart, ystart, xend, yend;
  PetscReal x, y;
  PetscBool isnull, flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw, PETSC_DRAW_CLASSID, 1);
  PetscCall(PetscDrawIsNull(draw, &isnull));
  if (isnull) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscDrawCoordinateToPixel(draw, xmin, ymin, &xstart, &ystart));
  PetscCall(PetscDrawCoordinateToPixel(draw, xmax, ymax, &xend, &yend));
  if (yend < ystart) {
    PetscInt tmp = ystart;
    ystart       = yend;
    yend         = tmp;
  }

  for (i = xstart; i <= xend; i++) {
    for (j = ystart; j <= yend; j++) {
      PetscCall(PetscDrawPixelToCoordinate(draw, i, j, &x, &y));
      PetscCall(indicator(ctx, x, y, &flg));
      if (flg) PetscCall(PetscDrawPointPixel(draw, i, j, c));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PetscDrawCoordinateToPixel - given a coordinate in a `PetscDraw` returns the pixel location

   Not Collective

   Input Parameters:
+  draw - the draw where the coordinates are defined
.  x - the horizontal coordinate
-  y - the vertical coordinate

   Output Parameters:
+  i - the horizontal pixel location
-  j - the vertical pixel location

   Level: developer

.seealso: `PetscDraw`
@*/
PetscErrorCode PetscDrawCoordinateToPixel(PetscDraw draw, PetscReal x, PetscReal y, int *i, int *j)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw, PETSC_DRAW_CLASSID, 1);
  PetscUseTypeMethod(draw, coordinatetopixel, x, y, i, j);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PetscDrawPixelToCoordinate - given a pixel in a `PetscDraw` returns the coordinate

   Not Collective

   Input Parameters:
+  draw - the draw where the coordinates are defined
.  i - the horizontal pixel location
-  j - the vertical pixel location

   Output Parameters:
+  x - the horizontal coordinate
-  y - the vertical coordinate

   Level: developer

.seealso: `PetscDraw`
@*/
PetscErrorCode PetscDrawPixelToCoordinate(PetscDraw draw, int i, int j, PetscReal *x, PetscReal *y)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw, PETSC_DRAW_CLASSID, 1);
  PetscUseTypeMethod(draw, pixeltocoordinate, i, j, x, y);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   PetscDrawRectangle - draws a rectangle onto a `PetscDraw` object

   Not Collective

   Input Parameters:
+  draw - the drawing context
.  xl - coordinates of the lower left corner
.  yl - coordinates of the lower left corner
.  xr - coordinate of the upper right corner
.  yr - coordinate of the upper right corner
.  c1 - the color of the first corner
.  c2 - the color of the second corner
.  c3 - the color of the third corner
-  c4 - the color of the fourth corner

   Level: beginner

.seealso: `PetscDraw`, `PetscDrawLine()`, `PetscDrawRectangle()`, `PetscDrawTriangle()`, `PetscDrawEllipse()`,
          `PetscDrawMarker()`, `PetscDrawPoint()`, `PetscDrawString()`, `PetscDrawPoint()`, `PetscDrawArrow()`
@*/
PetscErrorCode PetscDrawRectangle(PetscDraw draw, PetscReal xl, PetscReal yl, PetscReal xr, PetscReal yr, int c1, int c2, int c3, int c4)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw, PETSC_DRAW_CLASSID, 1);
  PetscUseTypeMethod(draw, rectangle, xl, yl, xr, yr, c1, c2, c3, c4);
  PetscFunctionReturn(PETSC_SUCCESS);
}
