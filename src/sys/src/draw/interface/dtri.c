#ifndef lint
static char vcid[] = "$Id: dtri.c,v 1.4 1996/03/19 21:28:06 bsmith Exp bsmith $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "drawimpl.h"  /*I "draw.h" I*/

/*@
   DrawTriangle - Draws a triangle  onto a drawable.

   Input Parameters:
.  draw - the drawing context
.  x1,y1,x2,y2,x3,y3 - the coordinates of the vertices
.  c1,c2,c3 - the colors of the corners in counter clockwise order

.keywords: draw, triangle
@*/
int DrawTriangle(Draw draw,double x1,double y1,double x2,double y2,
                 double x3,double y3,int c1, int c2,int c3)
{
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  if (draw->type == DRAW_NULLWINDOW) return 0;
  return (*draw->ops.triangle)(draw,x1,y1,x2,y2,x3,y3,c1,c2,c3);
}
