#ifndef lint
static char vcid[] = "$Id: dpoint.c,v 1.6 1996/08/08 14:44:45 bsmith Exp balay $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "src/draw/drawimpl.h"  /*I "draw.h" I*/

#undef __FUNCTION__  
#define __FUNCTION__ "DrawPoint"
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
  if (draw->type == DRAW_NULLWINDOW) return 0;
  return (*draw->ops.point)(draw,xl,yl,cl);
}

