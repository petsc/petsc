#ifndef lint
static char vcid[] = "$Id: dgcoor.c,v 1.5 1996/07/08 22:21:15 bsmith Exp bsmith $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "src/draw/drawimpl.h"  /*I "draw.h" I*/

/*@
   DrawGetCoordinates - Gets the application coordinates of the corners of
   the window (or page).

   Input Paramter:
.  draw - the drawing object

   Ouput Parameters:
.  xl,yl,xr,yr - the coordinates of the lower left corner and upper
                 right corner of the drawing region.

.keywords:  draw, get, coordinates
@*/
int DrawGetCoordinates(Draw draw,double *xl,double *yl,double *xr,double *yr)
{
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  if (!xl || !xr || !yl || !yr) SETERRQ(1,"DrawGetCoordinates:Bad pointer");
  if (draw->type == DRAW_NULLWINDOW) return 0;
  *xl = draw->coor_xl; *yl = draw->coor_yl;
  *xr = draw->coor_xr; *yr = draw->coor_yr;
  return 0;
}
