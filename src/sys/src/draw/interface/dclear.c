#ifndef lint
static char vcid[] = "$Id: dclear.c,v 1.1 1996/01/30 19:44:13 bsmith Exp bsmith $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "drawimpl.h"  /*I "draw.h" I*/

/*@
   DrawClear - Clears graphical output.

   Input Parameters:
.  ctx - the drawing context

.keywords: draw, clear
@*/
int DrawClear(Draw ctx)
{
  PETSCVALIDHEADERSPECIFIC(ctx,DRAW_COOKIE);
  if (ctx->type == NULLWINDOW) return 0;
  if (ctx->ops.clear) return (*ctx->ops.clear)(ctx);
  return 0;
}
