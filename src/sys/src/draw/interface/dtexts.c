#ifndef lint
static char vcid[] = "$Id: dtexts.c,v 1.1 1996/01/30 19:35:19 bsmith Exp bsmith $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "drawimpl.h"  /*I "draw.h" I*/

/*@
   DrawTextSetSize - Sets the size for charactor text.  The width is
   relative to the user coordinates of the window; 0.0 denotes the natural
   width; 1.0 denotes the interior viewport. 

   Input Parameters:
.  ctx - the drawing context
.  width - the width in user coordinates
.  height - the charactor height

  Note:
  Only a limited range of sizes are available.

.keywords: draw, text, set, size
@*/
int DrawTextSetSize(Draw ctx,double width,double height)
{
  PETSCVALIDHEADERSPECIFIC(ctx,DRAW_COOKIE);
  if (ctx->type == NULLWINDOW) return 0;
  return (*ctx->ops.textsetsize)(ctx,width,height);
}
