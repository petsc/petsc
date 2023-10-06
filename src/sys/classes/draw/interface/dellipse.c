/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include <petsc/private/drawimpl.h> /*I "petscdraw.h" I*/

/*@
  PetscDrawEllipse - Draws an ellipse onto a drawable.

  Not Collective

  Input Parameters:
+ draw - The drawing context
. x    - The x coordinate of the center
. y    - The y coordinate of the center
. a    - The major axes length
. b    - The minor axes length
- c    - The color

  Level: beginner

.seealso: `PetscDraw`, `PetscDrawRectangle()`, `PetscDrawTriangle()`, `PetscDrawMarker()`, `PetscDrawPoint()`, `PetscDrawString()`, `PetscDrawArrow()`
@*/
PetscErrorCode PetscDrawEllipse(PetscDraw draw, PetscReal x, PetscReal y, PetscReal a, PetscReal b, int c)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw, PETSC_DRAW_CLASSID, 1);
  PetscUseTypeMethod(draw, ellipse, x, y, a, b, c);
  PetscFunctionReturn(PETSC_SUCCESS);
}
