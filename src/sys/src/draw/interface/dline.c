#ifndef lint
static char vcid[] = "$Id: dline.c,v 1.1 1996/01/30 19:32:49 bsmith Exp bsmith $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "drawimpl.h"  /*I "draw.h" I*/
  
/*@
   DrawLine - Draws a line onto a drawable.

   Input Parameters:
.   ctx - the drawing context
.   xl,yl,xr,yr - the coordinates of the line endpoints
.   cl - the colors of the endpoints

.keywords:  draw, line
@*/
int DrawLine(Draw ctx,double xl,double yl,double xr,double yr,int cl)
{
  PETSCVALIDHEADERSPECIFIC(ctx,DRAW_COOKIE);
  if (ctx->type == NULLWINDOW) return 0;
  return (*ctx->ops.line)(ctx,xl,yl,xr,yr,cl);
}

