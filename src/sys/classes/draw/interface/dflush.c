/*
       Provides the calling sequences for all the basic PetscDraw routines.
*/
#include <petsc/private/drawimpl.h>  /*I "petscdraw.h" I*/

/*@
   PetscDrawFlush - Flushes graphical output.

   Collective on PetscDraw

   Input Parameters:
.  draw - the drawing context

   Level: beginner

.seealso: PetscDrawClear()
@*/
PetscErrorCode  PetscDrawFlush(PetscDraw draw)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  if (draw->ops->flush) {
    ierr = (*draw->ops->flush)(draw);CHKERRQ(ierr);
  }
  if (draw->saveonflush) {ierr = PetscDrawSave(draw);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}
