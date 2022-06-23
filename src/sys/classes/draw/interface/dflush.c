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

.seealso: `PetscDrawClear()`
@*/
PetscErrorCode  PetscDrawFlush(PetscDraw draw)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  if (draw->ops->flush) PetscCall((*draw->ops->flush)(draw));
  if (draw->saveonflush) PetscCall(PetscDrawSave(draw));
  PetscFunctionReturn(0);
}
