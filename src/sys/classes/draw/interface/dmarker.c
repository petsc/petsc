
/*
       Provides the calling sequences for all the basic PetscDraw routines.
*/
#include <petsc/private/drawimpl.h> /*I "petscdraw.h" I*/
const char *const PetscDrawMarkerTypes[] = {"CROSS", "POINT", "PLUS", "CIRCLE", "PetscDrawMarkerType", "PETSC_DRAW_MARKER_", NULL};

/*@
   PetscDrawMarker - draws a marker onto a drawable.

   Not Collective

   Input Parameters:
+  draw - the drawing context
.  xl - horizontal coordinate of the marker
.  yl - vertical coordinate of the marker
-  cl - the color of the marker

   Level: beginner

.seealso: `PetscDraw`, `PetscDrawPoint()`, `PetscDrawString()`, `PetscDrawSetMarkerType()`, `PetscDrawGetMarkerType()`
@*/
PetscErrorCode PetscDrawMarker(PetscDraw draw, PetscReal xl, PetscReal yl, int cl)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw, PETSC_DRAW_CLASSID, 1);
  if (draw->markertype == PETSC_DRAW_MARKER_CROSS) {
    if (draw->ops->coordinatetopixel && draw->ops->pointpixel) {
      int i, j, k;
      PetscUseTypeMethod(draw, coordinatetopixel, xl, yl, &i, &j);
      for (k = -2; k <= 2; k++) {
        PetscCall((*draw->ops->pointpixel)(draw, i + k, j + k, cl));
        PetscCall((*draw->ops->pointpixel)(draw, i + k, j - k, cl));
      }
    } else PetscUseTypeMethod(draw, string, xl, yl, cl, "x");
  } else if (draw->markertype == PETSC_DRAW_MARKER_PLUS) {
    if (draw->ops->coordinatetopixel && draw->ops->pointpixel) {
      int i, j, k;
      PetscUseTypeMethod(draw, coordinatetopixel, xl, yl, &i, &j);
      for (k = -2; k <= 2; k++) {
        PetscCall((*draw->ops->pointpixel)(draw, i, j + k, cl));
        PetscCall((*draw->ops->pointpixel)(draw, i + k, j, cl));
      }
    } else PetscUseTypeMethod(draw, string, xl, yl, cl, "+");
  } else if (draw->markertype == PETSC_DRAW_MARKER_CIRCLE) {
    if (draw->ops->coordinatetopixel && draw->ops->pointpixel) {
      int i, j, k;
      PetscUseTypeMethod(draw, coordinatetopixel, xl, yl, &i, &j);
      for (k = -1; k <= 1; k++) {
        PetscCall((*draw->ops->pointpixel)(draw, i + 2, j + k, cl));
        PetscCall((*draw->ops->pointpixel)(draw, i - 2, j + k, cl));
        PetscCall((*draw->ops->pointpixel)(draw, i + k, j + 2, cl));
        PetscCall((*draw->ops->pointpixel)(draw, i + k, j - 2, cl));
      }
    } else PetscUseTypeMethod(draw, string, xl, yl, cl, "+");
  } else PetscUseTypeMethod(draw, point, xl, yl, cl);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   PetscDrawSetMarkerType - sets the type of marker to display with `PetscDrawMarker()`

   Not Collective

   Input Parameters:
+  draw - the drawing context
-  mtype - either `PETSC_DRAW_MARKER_CROSS` (default) or `PETSC_DRAW_MARKER_POINT`

   Options Database Key:
.  -draw_marker_type - x or point

   Level: beginner

.seealso: `PetscDraw`, `PetscDrawPoint()`, `PetscDrawMarker()`, `PetscDrawGetMarkerType()`, `PetscDrawMarkerType`
@*/
PetscErrorCode PetscDrawSetMarkerType(PetscDraw draw, PetscDrawMarkerType mtype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw, PETSC_DRAW_CLASSID, 1);
  draw->markertype = mtype;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   PetscDrawGetMarkerType - gets the type of marker to display with `PetscDrawMarker()`

   Not Collective

   Input Parameters:
+  draw - the drawing context
-  mtype - either `PETSC_DRAW_MARKER_CROSS` (default) or `PETSC_DRAW_MARKER_POINT`

   Level: beginner

.seealso: `PetscDraw`, `PetscDrawPoint()`, `PetscDrawMarker()`, `PetscDrawSetMarkerType()`, `PetscDrawMarkerType`
@*/
PetscErrorCode PetscDrawGetMarkerType(PetscDraw draw, PetscDrawMarkerType *mtype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw, PETSC_DRAW_CLASSID, 1);
  *mtype = draw->markertype;
  PetscFunctionReturn(PETSC_SUCCESS);
}
