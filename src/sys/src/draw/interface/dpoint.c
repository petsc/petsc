#ifndef lint
static char vcid[] = "$Id: dpoint.c,v 1.1 1996/01/30 19:44:07 bsmith Exp bsmith $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "drawimpl.h"  /*I "draw.h" I*/

/*@
   DrawPoint - Draws a point onto a drawable.

   Input Parameters:
.  ctx - the drawing context
.  xl,yl - the coordinates of the point
.  cl - the color of the point

.keywords:  draw, point
@*/
int DrawPoint(Draw ctx,double xl,double yl,int cl)
{
  PETSCVALIDHEADERSPECIFIC(ctx,DRAW_COOKIE);
  if (ctx->type == NULLWINDOW) return 0;
  return (*ctx->ops.point)(ctx,xl,yl,cl);
}

