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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  if (draw->saveonclear) {ierr = PetscDrawSave(draw);CHKERRQ(ierr);}
  if (draw->ops->clear) {
    ierr = (*draw->ops->clear)(draw);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  if (draw->ops->beginpage) {
    ierr = (*draw->ops->beginpage)(draw);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  if (draw->ops->endpage) {
    ierr =  (*draw->ops->endpage)(draw);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
