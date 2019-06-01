
/*
       Provides the calling sequences for all the basic PetscDraw routines.
*/
#include <petsc/private/drawimpl.h>  /*I "petscdraw.h" I*/

/*@
   PetscDrawGetBoundingBox - Gets the bounding box of all PetscDrawStringBoxed() commands

   Not collective

   Input Parameter:
.  draw - the drawing context

   Output Parameters:
.   xl,yl,xr,yr - coordinates of lower left and upper right corners of bounding box

   Level: intermediate

.seealso:  PetscDrawPushCurrentPoint(), PetscDrawPopCurrentPoint(), PetscDrawSetCurrentPoint()
@*/
PetscErrorCode  PetscDrawGetBoundingBox(PetscDraw draw,PetscReal *xl,PetscReal *yl,PetscReal *xr,PetscReal *yr)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  if (xl) PetscValidRealPointer(xl,2);
  if (yl) PetscValidRealPointer(yl,3);
  if (xr) PetscValidRealPointer(xr,4);
  if (yr) PetscValidRealPointer(yr,5);
  if (xl) *xl = draw->boundbox_xl;
  if (yl) *yl = draw->boundbox_yl;
  if (xr) *xr = draw->boundbox_xr;
  if (yr) *yr = draw->boundbox_yr;
  PetscFunctionReturn(0);
}

/*@
   PetscDrawGetCurrentPoint - Gets the current draw point, some codes use this point to determine where to draw next

   Not collective

   Input Parameter:
.  draw - the drawing context

   Output Parameters:
.   x,y - the current point

   Level: intermediate

.seealso:  PetscDrawPushCurrentPoint(), PetscDrawPopCurrentPoint(), PetscDrawSetCurrentPoint()
@*/
PetscErrorCode  PetscDrawGetCurrentPoint(PetscDraw draw,PetscReal *x,PetscReal *y)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  PetscValidRealPointer(x,2);
  PetscValidRealPointer(y,3);
  *x = draw->currentpoint_x[draw->currentpoint];
  *y = draw->currentpoint_y[draw->currentpoint];
  PetscFunctionReturn(0);
}

/*@
   PetscDrawSetCurrentPoint - Sets the current draw point, some codes use this point to determine where to draw next

   Not collective

   Input Parameters:
+  draw - the drawing context
-  x,y - the location of the current point

   Level: intermediate

.seealso:  PetscDrawPushCurrentPoint(), PetscDrawPopCurrentPoint(), PetscDrawGetCurrentPoint()
@*/
PetscErrorCode  PetscDrawSetCurrentPoint(PetscDraw draw,PetscReal x,PetscReal y)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  draw->currentpoint_x[draw->currentpoint] = x;
  draw->currentpoint_y[draw->currentpoint] = y;
  PetscFunctionReturn(0);
}

/*@
   PetscDrawPushCurrentPoint - Pushes a new current draw point, retaining the old one, some codes use this point to determine where to draw next

   Not collective

   Input Parameters:
+  draw - the drawing context
-  x,y - the location of the current point

   Level: intermediate

.seealso:  PetscDrawPushCurrentPoint(), PetscDrawPopCurrentPoint(), PetscDrawGetCurrentPoint()
@*/
PetscErrorCode  PetscDrawPushCurrentPoint(PetscDraw draw,PetscReal x,PetscReal y)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  if (draw->currentpoint > 19) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"You have pushed too many current points");
  draw->currentpoint_x[++draw->currentpoint] = x;
  draw->currentpoint_y[draw->currentpoint]   = y;
  PetscFunctionReturn(0);
}

/*@
   PetscDrawPopCurrentPoint - Pops a current draw point (discarding it)

   Not collective

   Input Parameter:
.  draw - the drawing context

   Level: intermediate

.seealso:  PetscDrawPushCurrentPoint(), PetscDrawSetCurrentPoint(), PetscDrawGetCurrentPoint()
@*/
PetscErrorCode  PetscDrawPopCurrentPoint(PetscDraw draw)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  if (draw->currentpoint-- == 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"You have popped too many current points");
  PetscFunctionReturn(0);
}

/*@
   PetscDrawLine - PetscDraws a line onto a drawable.

   Not collective

   Input Parameters:
+  draw - the drawing context
.  xl,yl,xr,yr - the coordinates of the line endpoints
-  cl - the colors of the endpoints

   Level: beginner


.seealso: PetscDrawArrow(), PetscDrawLineSetWidth(), PetscDrawLineGetWidth(), PetscDrawRectangle(), PetscDrawTriangle(), PetscDrawEllipse(),
          PetscDrawMarker(), PetscDrawPoint()

@*/
PetscErrorCode  PetscDrawLine(PetscDraw draw,PetscReal xl,PetscReal yl,PetscReal xr,PetscReal yr,int cl)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  if (!draw->ops->line) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"This draw type %s does not support drawing lines",((PetscObject)draw)->type_name);
  ierr = (*draw->ops->line)(draw,xl,yl,xr,yr,cl);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   PetscDrawArrow - PetscDraws a line with arrow head at end if the line is long enough

   Not collective

   Input Parameters:
+  draw - the drawing context
.  xl,yl,xr,yr - the coordinates of the line endpoints
-  cl - the colors of the endpoints

   Level: beginner


.seealso: PetscDrawLine(), PetscDrawLineSetWidth(), PetscDrawLineGetWidth(), PetscDrawRectangle(), PetscDrawTriangle(), PetscDrawEllipse(),
          PetscDrawMarker(), PetscDrawPoint()

@*/
PetscErrorCode  PetscDrawArrow(PetscDraw draw,PetscReal xl,PetscReal yl,PetscReal xr,PetscReal yr,int cl)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  if (!draw->ops->arrow) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"This draw type %s does not support drawing arrows",((PetscObject)draw)->type_name);
  ierr = (*draw->ops->arrow)(draw,xl,yl,xr,yr,cl);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   PetscDrawLineSetWidth - Sets the line width for future draws.  The width is
   relative to the user coordinates of the window; 0.0 denotes the natural
   width; 1.0 denotes the entire viewport.

   Not collective

   Input Parameters:
+  draw - the drawing context
-  width - the width in user coordinates

   Level: advanced

.seealso:  PetscDrawLineGetWidth(), PetscDrawLine(), PetscDrawArrow()
@*/
PetscErrorCode  PetscDrawLineSetWidth(PetscDraw draw,PetscReal width)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  if (draw->ops->linesetwidth) {
    ierr = (*draw->ops->linesetwidth)(draw,width);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
   PetscDrawLineGetWidth - Gets the line width for future draws.  The width is
   relative to the user coordinates of the window; 0.0 denotes the natural
   width; 1.0 denotes the interior viewport.

   Not collective

   Input Parameter:
.  draw - the drawing context

   Output Parameter:
.  width - the width in user coordinates

   Level: advanced

   Notes:
   Not currently implemented.

.seealso:  PetscDrawLineSetWidth(), PetscDrawLine(), PetscDrawArrow()
@*/
PetscErrorCode  PetscDrawLineGetWidth(PetscDraw draw,PetscReal *width)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  PetscValidScalarPointer(width,2);
  if (!draw->ops->linegetwidth) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"This draw type %s does not support getting line width",((PetscObject)draw)->type_name);
  ierr = (*draw->ops->linegetwidth)(draw,width);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

