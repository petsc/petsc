
/*
       Provides the calling sequences for all the basic PetscDraw routines.
*/
#include <../src/sys/draw/drawimpl.h>  /*I "petscdraw.h" I*/


#undef __FUNCT__
#define __FUNCT__ "PetscDrawIndicatorFunction"
/*@C
   PetscDrawIndicatorFunction - Draws an indicator function (where a relationship is true) on a PetscDraw

   Not collective

   Input Parameter:
+  draw - a PetscDraw
.  xmin,xmax,ymin,ymax - region to draw indicator function
-  f - the indicator function

   Level: developer

@*/
PetscErrorCode PetscDrawIndicatorFunction(PetscDraw draw, PetscReal xmin, PetscReal xmax, PetscReal ymin, PetscReal ymax,int c,PetscErrorCode (*f)(void *,PetscReal,PetscReal,PetscBool*),void *ctx)
{
  PetscInt       xstart,ystart,xend,yend,i,j,tmp;
  PetscErrorCode ierr;
  PetscReal      x,y;
  PetscBool      isnull,flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)draw,PETSC_DRAW_NULL,&isnull);CHKERRQ(ierr);
  if (isnull) PetscFunctionReturn(0);

  ierr = PetscDrawCoordinateToPixel(draw,xmin,ymin,&xstart,&ystart);CHKERRQ(ierr);
  ierr = PetscDrawCoordinateToPixel(draw,xmax,ymax,&xend,&yend);CHKERRQ(ierr);
  if (yend < ystart) {
    tmp    = ystart;
    ystart = yend;
    yend   = tmp;
  }
  for (i=xstart; i<xend+1; i++) {
    for (j=ystart; j<yend+1; j++) {
      ierr = PetscDrawPixelToCoordinate(draw,i,j,&x,&y);CHKERRQ(ierr);
      ierr = f(ctx,x,y,&flg);CHKERRQ(ierr);
      if (flg) {
        ierr = PetscDrawPointPixel(draw,i,j,c);CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PetscDrawCoordinateToPixel"
/*@C
   PetscDrawCoordinateToPixel - given a coordinate in a PetscDraw returns the pixel location

   Not collective

   Input Parameters:
+  draw - the draw where the coordinates are defined
-  x,y - the coordinate location

   Output Parameters:
-  i,j - the pixel location

   Level: developer

@*/
PetscErrorCode PetscDrawCoordinateToPixel(PetscDraw draw,PetscReal x,PetscReal y,PetscInt *i,PetscInt *j)
{
  PetscErrorCode ierr;
  PetscBool      isnull;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)draw,PETSC_DRAW_NULL,&isnull);CHKERRQ(ierr);
  if (isnull) PetscFunctionReturn(0);
  if (!draw->ops->coordinatetopixel) SETERRQ(((PetscObject)draw)->comm,PETSC_ERR_SUP,"No support for locating pixel");
  ierr = (*draw->ops->coordinatetopixel)(draw,x,y,i,j);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDrawPixelToCoordinate"
/*@C
   PetscDrawPixelToCoordinate - given a pixel in a PetscDraw returns the coordinate

   Not collective

   Input Parameters:
+  draw - the draw where the coordinates are defined
-  i,j - the pixel location

   Output Parameters:
.  x,y - the coordinate location

   Level: developer

@*/
PetscErrorCode PetscDrawPixelToCoordinate(PetscDraw draw,PetscInt i,PetscInt j,PetscReal *x,PetscReal *y)
{
  PetscErrorCode ierr;
  PetscBool      isnull;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)draw,PETSC_DRAW_NULL,&isnull);CHKERRQ(ierr);
  if (isnull) PetscFunctionReturn(0);
  if (!draw->ops->pixeltocoordinate) SETERRQ(((PetscObject)draw)->comm,PETSC_ERR_SUP,"No support for locating coordiante from ");
  ierr = (*draw->ops->pixeltocoordinate)(draw,i,j,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDrawRectangle"
/*@
   PetscDrawRectangle - PetscDraws a rectangle  onto a drawable.

   Not Collective

   Input Parameters:
+  draw - the drawing context
.  xl,yl,xr,yr - the coordinates of the lower left, upper right corners
-  c1,c2,c3,c4 - the colors of the four corners in counter clockwise order

   Level: beginner

   Concepts: drawing^rectangle
   Concepts: graphics^rectangle
   Concepts: rectangle

@*/
PetscErrorCode  PetscDrawRectangle(PetscDraw draw,PetscReal xl,PetscReal yl,PetscReal xr,PetscReal yr,int c1,int c2,int c3,int c4)
{
  PetscErrorCode ierr;
  PetscBool      isnull;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)draw,PETSC_DRAW_NULL,&isnull);CHKERRQ(ierr);
  if (isnull) PetscFunctionReturn(0);
  if (!draw->ops->rectangle) SETERRQ(((PetscObject)draw)->comm,PETSC_ERR_SUP,"No support for drawing rectangle");
  ierr = (*draw->ops->rectangle)(draw,xl,yl,xr,yr,c1,c2,c3,c4);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDrawSave"
/*@
   PetscDrawSave - Saves a drawn image

   Not Collective

   Input Parameters:
.  draw - the drawing context

   Level: advanced

   Notes: this is not normally called by the user, it is called by PetscDrawClear_X() to save a sequence of images.

.seealso: PetscDrawSetSave()

@*/
PetscErrorCode  PetscDrawSave(PetscDraw draw)
{
  PetscErrorCode ierr;
  PetscBool      isnull;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)draw,PETSC_DRAW_NULL,&isnull);CHKERRQ(ierr);
  if (isnull) PetscFunctionReturn(0);
  if (!draw->ops->save) PetscFunctionReturn(0);
  ierr = (*draw->ops->save)(draw);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
