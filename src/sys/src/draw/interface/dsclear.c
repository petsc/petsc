#ifndef lint
static char vcid[] = "$Id: dsclear.c,v 1.4 1996/03/19 21:28:06 bsmith Exp bsmith $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "drawimpl.h"  /*I "draw.h" I*/

/*@
   DrawSyncClear - Clears graphical output. All processors must call this routine.
       Does not return until the drawable is clear.

   Input Parameters:
.  draw - the drawing context

.keywords: draw, clear
@*/
int DrawSyncClear(Draw draw)
{
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  if (draw->type == DRAW_NULLWINDOW) return 0;
  if (draw->ops.syncclear) return (*draw->ops.syncclear)(draw);
  return 0;
}
