#ifndef lint
static char vcid[] = "$Id: dsclear.c,v 1.1 1996/01/30 19:44:14 bsmith Exp bsmith $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "drawimpl.h"  /*I "draw.h" I*/

/*@
   DrawSyncClear - Clears graphical output. All processors must call this routine.
       Does not return until the drawable is clear.

   Input Parameters:
.  ctx - the drawing context

.keywords: draw, clear
@*/
int DrawSyncClear(Draw ctx)
{
  PETSCVALIDHEADERSPECIFIC(ctx,DRAW_COOKIE);
  if (ctx->type == NULLWINDOW) return 0;
  if (ctx->ops.syncclear) return (*ctx->ops.syncclear)(ctx);
  return 0;
}
