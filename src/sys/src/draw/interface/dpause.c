#ifndef lint
static char vcid[] = "$Id: dpause.c,v 1.1 1996/01/30 19:44:09 bsmith Exp bsmith $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "drawimpl.h"  /*I "draw.h" I*/

/*@
   DrawPause - Waits n seconds or until user input, depending on input 
               to DrawSetPause().

   Input Parameter:
.  ctx - the drawing context

.keywords: draw, pause

.seealso: DrawSetPause(), DrawGetPause()
@*/
int DrawPause(Draw ctx)
{
  PETSCVALIDHEADERSPECIFIC(ctx,DRAW_COOKIE);
  if (ctx->type == NULLWINDOW) return 0;
  if (ctx->ops.pause) return (*ctx->ops.pause)(ctx);
  return 0;
}
