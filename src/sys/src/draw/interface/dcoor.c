#ifndef lint
static char vcid[] = "$Id: dcoor.c,v 1.1 1996/01/30 19:44:08 bsmith Exp bsmith $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "drawimpl.h"  /*I "draw.h" I*/

/*@
   DrawSetCoordinates - Sets the application coordinates of the corners of
   the window (or page).

   Input Paramters:
.  ctx - the drawing object
.  xl,yl,xr,yr - the coordinates of the lower left corner and upper
                 right corner of the drawing region.

.keywords:  draw, set, coordinates

.seealso: DrawSetCoordinatesInParallel()
@*/
int DrawSetCoordinates(Draw ctx,double xl,double yl,double xr, double yr)
{
  PETSCVALIDHEADERSPECIFIC(ctx,DRAW_COOKIE);
  if (ctx->type == NULLWINDOW) return 0;
  ctx->coor_xl = xl; ctx->coor_yl = yl;
  ctx->coor_xr = xr; ctx->coor_yr = yr;
  return 0;
}

