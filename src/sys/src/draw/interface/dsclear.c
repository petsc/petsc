/*$Id: dsclear.c,v 1.26 2001/01/15 21:43:22 bsmith Exp balay $*/
/*
       Provides the calling sequences for all the basic PetscDraw routines.
*/
#include "src/sys/src/draw/drawimpl.h"  /*I "petscdraw.h" I*/

#undef __FUNC__  
#define __FUNC__ "PetscDrawSynchronizedClear" 
/*@
   PetscDrawSynchronizedClear - Clears graphical output. All processors must call this routine.
   Does not return until the draw in context is clear.

   Collective on PetscDraw

   Input Parameters:
.  draw - the drawing context

   Level: intermediate

   Concepts: clear^window

@*/
int PetscDrawSynchronizedClear(PetscDraw draw)
{
  int ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_COOKIE);
  if (draw->ops->synchronizedclear) {
    ierr = (*draw->ops->synchronizedclear)(draw);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
