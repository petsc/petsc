/*$Id: dcoor.c,v 1.22 2000/04/12 04:21:02 bsmith Exp balay $*/
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "src/sys/src/draw/drawimpl.h"  /*I "petscdraw.h" I*/

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"DrawSetCoordinates" 
/*@
   DrawSetCoordinates - Sets the application coordinates of the corners of
   the window (or page).

   Not collective

   Input Parameters:
+  draw - the drawing object
-  xl,yl,xr,yr - the coordinates of the lower left corner and upper
                 right corner of the drawing region.

   Level: advanced

.keywords:  draw, set, coordinates

.seealso: DrawGetCoordinates()

@*/
int DrawSetCoordinates(Draw draw,PetscReal xl,PetscReal yl,PetscReal xr,PetscReal yr)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  draw->coor_xl = xl; draw->coor_yl = yl;
  draw->coor_xr = xr; draw->coor_yr = yr;
  if (draw->ops->setcoordinates) {
    ierr = (*draw->ops->setcoordinates)(draw,xl,yl,xr,yr);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

