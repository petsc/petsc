
/*
       Provides the calling sequences for all the basic PetscDraw routines.
*/
#include <../src/sys/draw/drawimpl.h>  /*I "petscdraw.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawFlush" 
/*@
   PetscDrawFlush - Flushs graphical output.

   Not collective (Use PetscDrawSynchronizedFlush() for collective)

   Input Parameters:
.  draw - the drawing context

   Level: beginner

   Concepts: flushing^graphics

.seealso: PetscDrawSynchronizedFlush()
@*/
PetscErrorCode  PetscDrawFlush(PetscDraw draw)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  if (draw->ops->flush) {
    ierr = (*draw->ops->flush)(draw);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
