#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: dellipse.c,v 1.3 2000/01/10 03:26:52 knepley Exp $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "src/sys/src/draw/drawimpl.h"  /*I "draw.h" I*/

#undef __FUNC__  
#define __FUNC__ "PetscDrawEllipse" 
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
int PetscDrawEllipse(PetscDraw draw, PetscReal x, PetscReal y, PetscReal a, PetscReal b, int c)
{
  PetscTruth isdrawnull;
  int        ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw, PETSC_DRAW_COOKIE);
  ierr = PetscTypeCompare((PetscObject) draw, PETSC_DRAW_NULL, &isdrawnull);                              CHKERRQ(ierr);
  if (isdrawnull) PetscFunctionReturn(0);
  ierr = (*draw->ops->ellipse)(draw, x, y, a, b, c);                                                      CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
