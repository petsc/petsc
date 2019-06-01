
/*
       Provides the calling sequences for all the basic PetscDraw routines.
*/
#include <petsc/private/drawimpl.h>  /*I "petscdraw.h" I*/

/*@
   PetscDrawPoint - PetscDraws a point onto a drawable.

   Not collective

   Input Parameters:
+  draw - the drawing context
.  xl,yl - the coordinates of the point
-  cl - the color of the point

   Level: beginner


.seealso: PetscDrawPointPixel(), PetscDrawPointSetSize(), PetscDrawLine(), PetscDrawRectangle(), PetscDrawTriangle(), PetscDrawEllipse(),
          PetscDrawMarker(), PetscDrawString(), PetscDrawArrow()

@*/
PetscErrorCode  PetscDrawPoint(PetscDraw draw,PetscReal xl,PetscReal yl,int cl)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  if (!draw->ops->point) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"This draw type %s does not support drawing points",((PetscObject)draw)->type_name);
  ierr = (*draw->ops->point)(draw,xl,yl,cl);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   PetscDrawPointPixel - PetscDraws a point onto a drawable, in pixel coordinates

   Not collective

   Input Parameters:
+  draw - the drawing context
.  x,y - the pixel coordinates of the point
-  c - the color of the point

   Level: beginner


.seealso: PetscDrawPoint(), PetscDrawPointSetSize()

@*/
PetscErrorCode  PetscDrawPointPixel(PetscDraw draw,int x,int y,int c)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  if (!draw->ops->pointpixel) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"This draw type %s does not support drawing point pixels",((PetscObject)draw)->type_name);
  ierr = (*draw->ops->pointpixel)(draw,x,y,c);CHKERRQ(ierr);
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

.seealso: PetscDrawPoint(), PetscDrawMarker()
@*/
PetscErrorCode  PetscDrawPointSetSize(PetscDraw draw,PetscReal width)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  if (width < 0.0 || width > 1.0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Bad size %g, should be between 0 and 1",(double)width);
  if (draw->ops->pointsetsize) {
    ierr = (*draw->ops->pointsetsize)(draw,width);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
