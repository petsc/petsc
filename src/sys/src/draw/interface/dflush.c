#ifndef lint
static char vcid[] = "$Id: dflush.c,v 1.7 1996/12/16 18:13:33 balay Exp balay $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "src/draw/drawimpl.h"  /*I "draw.h" I*/

#undef __FUNC__  
#define __FUNC__ "DrawFlush"
/*@
   DrawFlush - Flushs graphical output.

   Input Parameters:
.  draw - the drawing context

.keywords:  draw, flush
@*/
int DrawFlush(Draw draw)
{
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  if (draw->type == DRAW_NULLWINDOW) return 0;
  if (draw->ops.flush) return (*draw->ops.flush)(draw);
  return 0;
}
