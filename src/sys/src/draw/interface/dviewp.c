#ifndef lint
static char vcid[] = "$Id: dviewp.c,v 1.4 1996/03/19 21:28:06 bsmith Exp bsmith $";
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
.  draw - the drawing context

.keywords:  draw, set, view, port
@*/
int DrawSetViewPort(Draw draw,double xl,double yl,double xr,double yr)
{
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  if (draw->type == DRAW_NULLWINDOW) return 0;
  if (xl < 0.0 || xr > 1.0 || yl < 0.0 || yr > 1.0 || xr <= xl || yr <= yl)
    SETERRQ(1,"DrawSetViewPort: Bad values"); 
  draw->port_xl = xl; draw->port_yl = yl;
  draw->port_xr = xr; draw->port_yr = yr;
  if (draw->ops.setviewport) return (*draw->ops.setviewport)(draw,xl,yl,xr,yr);
  return 0;
}

