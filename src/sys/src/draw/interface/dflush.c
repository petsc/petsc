#ifndef lint
static char vcid[] = "$Id: dflush.c,v 1.1 1996/01/30 19:44:12 bsmith Exp bsmith $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "drawimpl.h"  /*I "draw.h" I*/

/*@
   DrawFlush - Flushs graphical output.

   Input Parameters:
.  ctx - the drawing context

.keywords:  draw, flush
@*/
int DrawFlush(Draw ctx)
{
  PETSCVALIDHEADERSPECIFIC(ctx,DRAW_COOKIE);
  if (ctx->type == NULLWINDOW) return 0;
  if (ctx->ops.flush) return (*ctx->ops.flush)(ctx);
  return 0;
}
