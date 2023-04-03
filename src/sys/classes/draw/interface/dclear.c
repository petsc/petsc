/*
       Provides the calling sequences for all the basic PetscDraw routines.
*/
#include <petsc/private/drawimpl.h> /*I "petscdraw.h" I*/

/*@
   PetscDrawClear - Clears graphical output. All processors must call this routine.
   Does not return until the draw in context is clear.

   Collective

   Input Parameter:
.  draw - the drawing context

   Level: intermediate

@*/
PetscErrorCode PetscDrawClear(PetscDraw draw)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw, PETSC_DRAW_CLASSID, 1);
  if (draw->saveonclear) PetscCall(PetscDrawSave(draw));
  PetscTryTypeMethod(draw, clear);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   PetscDrawBOP - Begins a new page or frame on the selected graphical device.

   Logically Collective

   Input Parameter:
.  draw - the drawing context

   Level: advanced

.seealso: `PetscDrawEOP()`, `PetscDrawClear()`
@*/
PetscErrorCode PetscDrawBOP(PetscDraw draw)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw, PETSC_DRAW_CLASSID, 1);
  PetscTryTypeMethod(draw, beginpage);
  PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
   PetscDrawEOP - Ends a page or frame on the selected graphical device.

   Logically Collective

   Input Parameter:
.  draw - the drawing context

   Level: advanced

.seealso: `PetscDrawBOP()`, `PetscDrawClear()`
@*/
PetscErrorCode PetscDrawEOP(PetscDraw draw)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw, PETSC_DRAW_CLASSID, 1);
  PetscTryTypeMethod(draw, endpage);
  PetscFunctionReturn(PETSC_SUCCESS);
}
