#define PETSC_DLL
/*
       Provides the calling sequences for all the basic PetscDraw routines.
*/
#include "../src/sys/draw/drawimpl.h"  /*I "petscdraw.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawRectangle" 
/*@
   PetscDrawRectangle - PetscDraws a rectangle  onto a drawable.

   Not Collective

   Input Parameters:
+  draw - the drawing context
.  xl,yl,xr,yr - the coordinates of the lower left, upper right corners
-  c1,c2,c3,c4 - the colors of the four corners in counter clockwise order

   Level: beginner

   Concepts: drawing^rectangle
   Concepts: graphics^rectangle
   Concepts: rectangle

@*/
PetscErrorCode PETSC_DLLEXPORT PetscDrawRectangle(PetscDraw draw,PetscReal xl,PetscReal yl,PetscReal xr,PetscReal yr,int c1,int c2,int c3,int c4)
{
  PetscErrorCode ierr;
  PetscTruth isnull;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_COOKIE,1);
  ierr = PetscTypeCompare((PetscObject)draw,PETSC_DRAW_NULL,&isnull);CHKERRQ(ierr);
  if (isnull) PetscFunctionReturn(0);
  ierr = (*draw->ops->rectangle)(draw,xl,yl,xr,yr,c1,c2,c3,c4);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
