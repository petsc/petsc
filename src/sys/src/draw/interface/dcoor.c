#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: dcoor.c,v 1.16 1999/01/31 16:04:52 bsmith Exp bsmith $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "src/sys/src/draw/drawimpl.h"  /*I "draw.h" I*/

#undef __FUNC__  
#define __FUNC__ "DrawSetCoordinates" 
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
int DrawSetCoordinates(Draw draw,double xl,double yl,double xr, double yr)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  draw->coor_xl = xl; draw->coor_yl = yl;
  draw->coor_xr = xr; draw->coor_yr = yr;
  PetscFunctionReturn(0);
}

