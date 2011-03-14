
/*
       Provides the calling sequences for all the basic PetscDraw routines.
*/
#include <../src/sys/draw/drawimpl.h>  /*I "petscdraw.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawSynchronizedClear" 
/*@
   PetscDrawSynchronizedClear - Clears graphical output. All processors must call this routine.
   Does not return until the draw in context is clear.

   Collective on PetscDraw

   Input Parameters:
.  draw - the drawing context

   Level: intermediate

   Concepts: clear^window

@*/
PetscErrorCode  PetscDrawSynchronizedClear(PetscDraw draw)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  if (draw->ops->synchronizedclear) {
    ierr = (*draw->ops->synchronizedclear)(draw);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
