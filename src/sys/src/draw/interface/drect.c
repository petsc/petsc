#ifndef lint
static char vcid[] = "$Id: drect.c,v 1.1 1996/01/30 19:44:15 bsmith Exp bsmith $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "drawimpl.h"  /*I "draw.h" I*/

/*@
   DrawRectangle - Draws a rectangle  onto a drawable.

   Input Parameters:
.  ctx - the drawing context
.  xl,yl,xr,yr - the coordinates of the lower left, upper right corners
.  c1,c2,c3,c4 - the colors of the four corners in counter clockwise order

.keywords: draw, rectangle
@*/
int DrawRectangle(Draw c,double xl,double yl,double xr,double yr,
                              int c1, int c2,int c3,int c4)
{
  PETSCVALIDHEADERSPECIFIC(c,DRAW_COOKIE);
  if (c->type == NULLWINDOW) return 0;
  return (*c->ops.rectangle)(c,xl,yl,xr,yr,c1,c2,c3,c4);
}
