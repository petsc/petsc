/*$Id: dsflush.c,v 1.26 2000/09/22 20:41:56 bsmith Exp bsmith $*/
/*
       Provides the calling sequences for all the basic PetscDraw routines.
*/
#include "src/sys/src/draw/drawimpl.h"  /*I "petscdraw.h" I*/

#undef __FUNC__  
#define __FUNC__ "DrawSynchronizedFlush" 
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
int PetscDrawSynchronizedFlush(PetscDraw draw)
{
  int ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_COOKIE);
  if (draw->ops->synchronizedflush) {
    ierr = (*draw->ops->synchronizedflush)(draw);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
