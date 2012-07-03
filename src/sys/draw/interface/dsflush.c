
/*
       Provides the calling sequences for all the basic PetscDraw routines.
*/
#include <../src/sys/draw/drawimpl.h>  /*I "petscdraw.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawSynchronizedFlush" 
/*@
   PetscDrawSynchronizedFlush - Flushes graphical output. This waits until all 
   processors have arrived and flushed, then does a global flush.
   This is usually done to change the frame for double buffered graphics.

   Collective on PetscDraw

   Input Parameters:
.  draw - the drawing context

   Level: beginner

   Concepts: flushing^graphics

.seealso: PetscDrawFlush()

@*/
PetscErrorCode  PetscDrawSynchronizedFlush(PetscDraw draw)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  if (draw->ops->synchronizedflush) {
    ierr = (*draw->ops->synchronizedflush)(draw);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
