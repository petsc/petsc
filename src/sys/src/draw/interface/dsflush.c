#ifndef lint
static char vcid[] = "$Id: dsflush.c,v 1.7 1996/09/28 17:36:32 curfman Exp balay $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "src/draw/drawimpl.h"  /*I "draw.h" I*/

#undef __FUNCTION__  
#define __FUNCTION__ "DrawSyncFlush"
/*@
   DrawSyncFlush - Flushes graphical output. This waits until all 
   processors have arrived and flushed, then does a global flush.
   This is usually done to change the frame for double buffered graphics.

  Input Parameters:
.  draw - the drawing context

.keywords: draw, sync, flush

@*/
int DrawSyncFlush(Draw draw)
{
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  if (draw->type == DRAW_NULLWINDOW) return 0;
  if (draw->ops.syncflush) return (*draw->ops.syncflush)(draw);
  return 0;
}
