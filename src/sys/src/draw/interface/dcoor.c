#ifndef lint
static char vcid[] = "$Id: dcoor.c,v 1.5 1996/07/08 22:21:15 bsmith Exp bsmith $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "src/draw/drawimpl.h"  /*I "draw.h" I*/

/*@
   DrawSetCoordinates - Sets the application coordinates of the corners of
   the window (or page).

   Input Paramters:
.  draw - the drawing object
.  xl,yl,xr,yr - the coordinates of the lower left corner and upper
                 right corner of the drawing region.

.keywords:  draw, set, coordinates

.seealso: DrawSetCoordinatesInParallel()
@*/
int DrawSetCoordinates(Draw draw,double xl,double yl,double xr, double yr)
{
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  if (draw->type == DRAW_NULLWINDOW) return 0;
  draw->coor_xl = xl; draw->coor_yl = yl;
  draw->coor_xr = xr; draw->coor_yr = yr;
  return 0;
}

