/*$Id: dsclear.c,v 1.23 2000/05/05 22:13:25 balay Exp bsmith $*/
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "src/sys/src/draw/drawimpl.h"  /*I "petscdraw.h" I*/

#undef __FUNC__  
#define __FUNC__ /*<a name="DrawSynchronizedClear"></a>*/"DrawSynchronizedClear" 
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
