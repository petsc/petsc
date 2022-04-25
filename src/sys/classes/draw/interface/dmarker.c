
/*
       Provides the calling sequences for all the basic PetscDraw routines.
*/
#include <petsc/private/drawimpl.h>  /*I "petscdraw.h" I*/
const char *const PetscDrawMarkerTypes[]     = {"CROSS","POINT","PLUS","CIRCLE","PetscDrawMarkerType","PETSC_DRAW_MARKER_",NULL};

/*@
   PetscDrawMarker - PetscDraws a marker onto a drawable.

   Not collective

   Input Parameters:
+  draw - the drawing context
.  xl,yl - the coordinates of the marker
-  cl - the color of the marker

   Level: beginner

.seealso: `PetscDrawPoint()`, `PetscDrawString()`, `PetscDrawSetMarkerType()`, `PetscDrawGetMarkerType()`

@*/
PetscErrorCode  PetscDrawMarker(PetscDraw draw,PetscReal xl,PetscReal yl,int cl)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  if (draw->markertype == PETSC_DRAW_MARKER_CROSS) {
    if (draw->ops->coordinatetopixel && draw->ops->pointpixel) {
      int i,j,k;
      PetscCall((*draw->ops->coordinatetopixel)(draw,xl,yl,&i,&j));
      for (k=-2; k<=2; k++) {
        PetscCall((*draw->ops->pointpixel)(draw,i+k,j+k,cl));
        PetscCall((*draw->ops->pointpixel)(draw,i+k,j-k,cl));
      }
    } else if (draw->ops->string) {
       PetscCall((*draw->ops->string)(draw,xl,yl,cl,"x"));
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for drawing marker type CROSS");
  } else if (draw->markertype == PETSC_DRAW_MARKER_PLUS) {
    if (draw->ops->coordinatetopixel && draw->ops->pointpixel) {
      int i,j,k;
      PetscCall((*draw->ops->coordinatetopixel)(draw,xl,yl,&i,&j));
      for (k=-2; k<=2; k++) {
        PetscCall((*draw->ops->pointpixel)(draw,i,j+k,cl));
        PetscCall((*draw->ops->pointpixel)(draw,i+k,j,cl));
      }
    } else if (draw->ops->string) {
       PetscCall((*draw->ops->string)(draw,xl,yl,cl,"+"));
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for drawing marker type PLUS");
  } else if (draw->markertype == PETSC_DRAW_MARKER_CIRCLE) {
    if (draw->ops->coordinatetopixel && draw->ops->pointpixel) {
      int i,j,k;
      PetscCall((*draw->ops->coordinatetopixel)(draw,xl,yl,&i,&j));
      for (k=-1; k<=1; k++) {
        PetscCall((*draw->ops->pointpixel)(draw,i+2,j+k,cl));
        PetscCall((*draw->ops->pointpixel)(draw,i-2,j+k,cl));
        PetscCall((*draw->ops->pointpixel)(draw,i+k,j+2,cl));
        PetscCall((*draw->ops->pointpixel)(draw,i+k,j-2,cl));
      }
    } else if (draw->ops->string) {
       PetscCall((*draw->ops->string)(draw,xl,yl,cl,"+"));
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for drawing marker type CIRCLE");
  } else {
    PetscCall((*draw->ops->point)(draw,xl,yl,cl));
  }
  PetscFunctionReturn(0);
}

/*@
   PetscDrawSetMarkerType - sets the type of marker to display with PetscDrawMarker()

   Not collective

   Input Parameters:
+  draw - the drawing context
-  mtype - either PETSC_DRAW_MARKER_CROSS (default) or PETSC_DRAW_MARKER_POINT

   Options Database:
.  -draw_marker_type - x or point

   Level: beginner

.seealso: `PetscDrawPoint()`, `PetscDrawMarker()`, `PetscDrawGetMarkerType()`

@*/
PetscErrorCode  PetscDrawSetMarkerType(PetscDraw draw,PetscDrawMarkerType mtype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  draw->markertype = mtype;
  PetscFunctionReturn(0);
}

/*@
   PetscDrawGetMarkerType - gets the type of marker to display with PetscDrawMarker()

   Not collective

   Input Parameters:
+  draw - the drawing context
-  mtype - either PETSC_DRAW_MARKER_CROSS (default) or PETSC_DRAW_MARKER_POINT

   Level: beginner

.seealso: `PetscDrawPoint()`, `PetscDrawMarker()`, `PetscDrawSetMarkerType()`

@*/
PetscErrorCode  PetscDrawGetMarkerType(PetscDraw draw,PetscDrawMarkerType *mtype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  *mtype = draw->markertype;
  PetscFunctionReturn(0);
}
