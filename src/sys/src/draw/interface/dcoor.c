#ifndef lint
static char vcid[] = "$Id: dcoor.c,v 1.3 1996/03/10 17:28:57 bsmith Exp bsmith $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "drawimpl.h"  /*I "draw.h" I*/

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
  if (draw->type == NULLWINDOW) return 0;
  draw->coor_xl = xl; draw->coor_yl = yl;
  draw->coor_xr = xr; draw->coor_yr = yr;
  return 0;
}

