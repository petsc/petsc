/*$Id: dsclear.c,v 1.25 2000/09/22 20:41:56 bsmith Exp bsmith $*/
/*
       Provides the calling sequences for all the basic PetscDraw routines.
*/
#include "src/sys/src/draw/drawimpl.h"  /*I "petscdraw.h" I*/

#undef __FUNC__  
#define __FUNC__ "DrawSynchronizedClear" 
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
