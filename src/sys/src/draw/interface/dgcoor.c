#ifndef lint
static char vcid[] = "$Id: dgcoor.c,v 1.1 1996/01/30 19:44:10 bsmith Exp $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "drawimpl.h"  /*I "draw.h" I*/

/*@
   DrawGetCoordinates - Gets the application coordinates of the corners of
   the window (or page).

   Input Paramter:
.  ctx - the drawing object

   Ouput Parameters:
.  xl,yl,xr,yr - the coordinates of the lower left corner and upper
                 right corner of the drawing region.

.keywords:  draw, get, coordinates
@*/
int DrawGetCoordinates(Draw ctx,double *xl,double *yl,double *xr,double *yr)
{
  PETSCVALIDHEADERSPECIFIC(ctx,DRAW_COOKIE);
  if (!xl || !xr || !yl || !yr) SETERRQ(1,"DrawGetCoordinates:Bad pointer");
  if (ctx->type == NULLWINDOW) return 0;
  *xl = ctx->coor_xl; *yl = ctx->coor_yl;
  *xr = ctx->coor_xr; *yr = ctx->coor_yr;
  return 0;
}
