#ifndef lint
static char vcid[] = "$Id: dviewp.c,v 1.1 1996/01/30 19:44:08 bsmith Exp bsmith $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "drawimpl.h"  /*I "draw.h" I*/

/*@
   DrawSetViewPort - Sets the portion of the window (page) to which draw
   routines will write.

   Input Parameters:
.  xl,yl,xr,yr - upper right and lower left corners of subwindow
                 These numbers must always be between 0.0 and 1.0.
                 Lower left corner is (0,0).
.  ctx - the drawing context

.keywords:  draw, set, view, port
@*/
int DrawSetViewPort(Draw ctx,double xl,double yl,double xr,double yr)
{
  PETSCVALIDHEADERSPECIFIC(ctx,DRAW_COOKIE);
  if (ctx->type == NULLWINDOW) return 0;
  if (xl < 0.0 || xr > 1.0 || yl < 0.0 || yr > 1.0 || xr <= xl || yr <= yl)
    SETERRQ(1,"DrawSetViewPort: Bad values"); 
  ctx->port_xl = xl; ctx->port_yl = yl;
  ctx->port_xr = xr; ctx->port_yr = yr;
  if (ctx->ops.setviewport) return (*ctx->ops.setviewport)(ctx,xl,yl,xr,yr);
  return 0;
}

