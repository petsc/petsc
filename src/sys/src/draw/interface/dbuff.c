#ifndef lint
static char vcid[] = "$Id: dbuff.c,v 1.1 1996/01/30 19:44:11 bsmith Exp bsmith $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "drawimpl.h"  /*I "draw.h" I*/

/*@
   DrawSetDoubleBuffer - Sets a window to be double buffered. 

   Input Parameter:
.  ctx - the drawing context

.keywords:  draw, set, double, buffer
@*/
int DrawSetDoubleBuffer(Draw ctx)
{
  PETSCVALIDHEADERSPECIFIC(ctx,DRAW_COOKIE);
  if (ctx->type == NULLWINDOW) return 0;
  if (ctx->ops.setdoublebuffer) return (*ctx->ops.setdoublebuffer)(ctx);
  return 0;
}
