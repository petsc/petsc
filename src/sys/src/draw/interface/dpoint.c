#ifndef lint
static char vcid[] = "$Id: dpoint.c,v 1.8 1997/01/06 20:26:34 balay Exp bsmith $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "src/draw/drawimpl.h"  /*I "draw.h" I*/

#undef __FUNC__  
#define __FUNC__ "DrawPoint" /* ADIC Ignore */
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

