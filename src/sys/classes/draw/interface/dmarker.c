
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


.seealso: PetscDrawPoint(), PetscDrawString(), PetscDrawSetMarkerType(), PetscDrawGetMarkerType()

@*/
PetscErrorCode  PetscDrawMarker(PetscDraw draw,PetscReal xl,PetscReal yl,int cl)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  if (draw->markertype == PETSC_DRAW_MARKER_CROSS){
    if (draw->ops->coordinatetopixel && draw->ops->pointpixel) {
      int i,j,k;
      ierr = (*draw->ops->coordinatetopixel)(draw,xl,yl,&i,&j);CHKERRQ(ierr);
      for (k=-2; k<=2; k++) {
        ierr = (*draw->ops->pointpixel)(draw,i+k,j+k,cl);CHKERRQ(ierr);
        ierr = (*draw->ops->pointpixel)(draw,i+k,j-k,cl);CHKERRQ(ierr);
      }
    } else if (draw->ops->string) {
       ierr = (*draw->ops->string)(draw,xl,yl,cl,"x");CHKERRQ(ierr);
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for drawing marker type CROSS");
  } else if (draw->markertype == PETSC_DRAW_MARKER_PLUS){
    if (draw->ops->coordinatetopixel && draw->ops->pointpixel) {
      int i,j,k;
      ierr = (*draw->ops->coordinatetopixel)(draw,xl,yl,&i,&j);CHKERRQ(ierr);
      for (k=-2; k<=2; k++) {
        ierr = (*draw->ops->pointpixel)(draw,i,j+k,cl);CHKERRQ(ierr);
        ierr = (*draw->ops->pointpixel)(draw,i+k,j,cl);CHKERRQ(ierr);
      }
    } else if (draw->ops->string) {
       ierr = (*draw->ops->string)(draw,xl,yl,cl,"+");CHKERRQ(ierr);
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for drawing marker type PLUS");
  } else if (draw->markertype == PETSC_DRAW_MARKER_CIRCLE){
    if (draw->ops->coordinatetopixel && draw->ops->pointpixel) {
      int i,j,k;
      ierr = (*draw->ops->coordinatetopixel)(draw,xl,yl,&i,&j);CHKERRQ(ierr);
      for (k=-1; k<=1; k++) {
        ierr = (*draw->ops->pointpixel)(draw,i+2,j+k,cl);CHKERRQ(ierr);
        ierr = (*draw->ops->pointpixel)(draw,i-2,j+k,cl);CHKERRQ(ierr);
        ierr = (*draw->ops->pointpixel)(draw,i+k,j+2,cl);CHKERRQ(ierr);
        ierr = (*draw->ops->pointpixel)(draw,i+k,j-2,cl);CHKERRQ(ierr);
      }
    } else if (draw->ops->string) {
       ierr = (*draw->ops->string)(draw,xl,yl,cl,"+");CHKERRQ(ierr);
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for drawing marker type CIRCLE");
  } else {
    ierr = (*draw->ops->point)(draw,xl,yl,cl);CHKERRQ(ierr);
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


.seealso: PetscDrawPoint(), PetscDrawMarker(), PetscDrawGetMarkerType()

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


.seealso: PetscDrawPoint(), PetscDrawMarker(), PetscDrawSetMarkerType()

@*/
PetscErrorCode  PetscDrawGetMarkerType(PetscDraw draw,PetscDrawMarkerType *mtype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  *mtype = draw->markertype;
  PetscFunctionReturn(0);
}
