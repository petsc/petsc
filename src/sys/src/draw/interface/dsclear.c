#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: dsclear.c,v 1.18 1999/01/31 16:04:52 bsmith Exp bsmith $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "src/sys/src/draw/drawimpl.h"  /*I "draw.h" I*/

#undef __FUNC__  
#define __FUNC__ "DrawSynchronizedClear" 
/*@
   DrawSynchronizedClear - Clears graphical output. All processors must call this routine.
   Does not return until the draw in context is clear.

   Collective on Draw

   Input Parameters:
.  draw - the drawing context

   Level: intermediate

.keywords: draw, clear
@*/
int DrawSynchronizedClear(Draw draw)
{
  int ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  if (draw->ops->synchronizedclear) {
    ierr = (*draw->ops->synchronizedclear)(draw);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
