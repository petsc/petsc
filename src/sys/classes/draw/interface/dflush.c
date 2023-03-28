/*
       Provides the calling sequences for all the basic PetscDraw routines.
*/
#include <petsc/private/drawimpl.h> /*I "petscdraw.h" I*/

/*@
   PetscDrawFlush - Flushes graphical output.

   Collective

   Input Parameter:
.  draw - the drawing context

   Level: beginner

.seealso: `PetscDraw`, `PetscDrawClear()`
@*/
PetscErrorCode PetscDrawFlush(PetscDraw draw)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw, PETSC_DRAW_CLASSID, 1);
  PetscTryTypeMethod(draw, flush);
  if (draw->saveonflush) PetscCall(PetscDrawSave(draw));
  PetscFunctionReturn(PETSC_SUCCESS);
}
