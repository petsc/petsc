#ifndef lint
static char vcid[] = "$Id: dsflush.c,v 1.1 1996/01/30 19:44:13 bsmith Exp bsmith $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "drawimpl.h"  /*I "draw.h" I*/

/*@
   DrawSyncFlush - Flushs graphical output. This waits until all 
   processors have arrived and flushed, then does a global flush.
   This is usually done to change the frame for double buffered graphics.

  Input Parameters:
.  ctx - the drawing context

.keywords: draw, sync, flush

@*/
int DrawSyncFlush(Draw ctx)
{
  PETSCVALIDHEADERSPECIFIC(ctx,DRAW_COOKIE);
  if (ctx->type == NULLWINDOW) return 0;
  if (ctx->ops.syncflush) return (*ctx->ops.syncflush)(ctx);
  return 0;
}
