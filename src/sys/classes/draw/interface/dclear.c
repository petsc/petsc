/*
       Provides the calling sequences for all the basic PetscDraw routines.
*/
#include <petsc/private/drawimpl.h>  /*I "petscdraw.h" I*/

/*@
   PetscDrawClear - Clears graphical output. All processors must call this routine.
   Does not return until the draw in context is clear.

   Collective on PetscDraw

   Input Parameters:
.  draw - the drawing context

   Level: intermediate

@*/
PetscErrorCode  PetscDrawClear(PetscDraw draw)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  if (draw->saveonclear) CHKERRQ(PetscDrawSave(draw));
  if (draw->ops->clear) {
    CHKERRQ((*draw->ops->clear)(draw));
  }
  PetscFunctionReturn(0);
}

/*@
   PetscDrawBOP - Begins a new page or frame on the selected graphical device.

   Logically Collective on PetscDraw

   Input Parameter:
.  draw - the drawing context

   Level: advanced

.seealso: PetscDrawEOP(), PetscDrawClear()
@*/
PetscErrorCode  PetscDrawBOP(PetscDraw draw)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  if (draw->ops->beginpage) {
    CHKERRQ((*draw->ops->beginpage)(draw));
  }
  PetscFunctionReturn(0);
}
/*@
   PetscDrawEOP - Ends a page or frame on the selected graphical device.

   Logically Collective on PetscDraw

   Input Parameter:
.  draw - the drawing context

   Level: advanced

.seealso: PetscDrawBOP(), PetscDrawClear()
@*/
PetscErrorCode  PetscDrawEOP(PetscDraw draw)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  if (draw->ops->endpage) {
    CHKERRQ((*draw->ops->endpage)(draw));
  }
  PetscFunctionReturn(0);
}
