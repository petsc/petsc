#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: dsflush.c,v 1.15 1998/03/12 23:20:42 bsmith Exp bsmith $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "src/draw/drawimpl.h"  /*I "draw.h" I*/

#undef __FUNC__  
#define __FUNC__ "DrawSynchronizedFlush" 
/*@
   DrawSynchronizedFlush - Flushes graphical output. This waits until all 
   processors have arrived and flushed, then does a global flush.
   This is usually done to change the frame for double buffered graphics.

  Input Parameters:
.  draw - the drawing context

  Collective on Draw

.keywords: draw, sync, flush

@*/
int DrawSynchronizedFlush(Draw draw)
{
  int ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  if (draw->type == DRAW_NULLWINDOW) PetscFunctionReturn(0);
  if (draw->ops->syncflush) {
    ierr = (*draw->ops->syncflush)(draw);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
