#ifndef lint
static char vcid[] = "$Id: dsclear.c,v 1.8 1997/01/06 20:26:34 balay Exp bsmith $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "src/draw/drawimpl.h"  /*I "draw.h" I*/

#undef __FUNC__  
#define __FUNC__ "DrawSyncClear" /* ADIC Ignore */
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
