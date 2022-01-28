
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include <petsc/private/drawimpl.h>  /*I "petscdraw.h" I*/

/*@
  PetscDrawEllipse - Draws an ellipse onto a drawable.

  Not collective

  Input Parameters:
+ draw - The drawing context
. x,y  - The center
. a,b  - The major and minor axes lengths
- c    - The color

  Level: beginner

.seealso: PetscDrawRectangle(), PetscDrawTriangle(), PetscDrawMarker(), PetscDrawPoint(), PetscDrawString(), PetscDrawArrow()
@*/
PetscErrorCode  PetscDrawEllipse(PetscDraw draw, PetscReal x, PetscReal y, PetscReal a, PetscReal b, int c)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw, PETSC_DRAW_CLASSID,1);
  PetscAssertFalse(!draw->ops->ellipse,PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for drawing ellipses");
  ierr = (*draw->ops->ellipse)(draw, x, y, a, b, c);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
