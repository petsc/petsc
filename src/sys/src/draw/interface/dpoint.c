#ifndef lint
static char vcid[] = "$Id: dpoint.c,v 1.3 1996/03/10 17:28:57 bsmith Exp bsmith $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "drawimpl.h"  /*I "draw.h" I*/

/*@
   DrawPoint - Draws a point onto a drawable.

   Input Parameters:
.  draw - the drawing context
.  xl,yl - the coordinates of the point
.  cl - the color of the point

.keywords:  draw, point
@*/
int DrawPoint(Draw draw,double xl,double yl,int cl)
{
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  if (draw->type == NULLWINDOW) return 0;
  return (*draw->ops.point)(draw,xl,yl,cl);
}

