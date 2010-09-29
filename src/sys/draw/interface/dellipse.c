#define PETSC_DLL
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "../src/sys/draw/drawimpl.h"  /*I "petscdraw.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawEllipse" 
/*@
  PetscDrawEllipse - Draws an ellipse onto a drawable.

  Not collective

  Input Parameters:
+ draw - The drawing context
. x,y  - The center
. a,b  - The major and minor axes lengths
- c    - The color

  Level: beginner

.keywords: draw, ellipse
.seealso: PetscDrawRectangle(), PetscDrawTriangle()
@*/
PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawEllipse(PetscDraw draw, PetscReal x, PetscReal y, PetscReal a, PetscReal b, int c)
{
  PetscBool  isdrawnull;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw, PETSC_DRAW_CLASSID,1);
  ierr = PetscTypeCompare((PetscObject) draw, PETSC_DRAW_NULL, &isdrawnull);CHKERRQ(ierr);
  if (isdrawnull) PetscFunctionReturn(0);
  ierr = (*draw->ops->ellipse)(draw, x, y, a, b, c);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
