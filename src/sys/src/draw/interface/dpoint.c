/*$Id: dpoint.c,v 1.28 2000/09/22 20:41:56 bsmith Exp bsmith $*/
/*
       Provides the calling sequences for all the basic PetscDraw routines.
*/
#include "src/sys/src/draw/drawimpl.h"  /*I "petscdraw.h" I*/

#undef __FUNC__  
#define __FUNC__ "DrawPoint" 
/*@
   PetscDrawPoint - PetscDraws a point onto a drawable.

   Not collective

   Input Parameters:
+  draw - the drawing context
.  xl,yl - the coordinates of the point
-  cl - the color of the point

   Level: beginner

   Concepts: point^drawing
   Concepts: drawing^point

.seealso: PetscDrawPointSetSize()

@*/
int PetscDrawPoint(PetscDraw draw,PetscReal xl,PetscReal yl,int cl)
{
  int        ierr;
  PetscTruth isnull;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_COOKIE);
  ierr = PetscTypeCompare((PetscObject)draw,PETSC_DRAW_NULL,&isnull);CHKERRQ(ierr);
  if (isnull) PetscFunctionReturn(0);
  ierr = (*draw->ops->point)(draw,xl,yl,cl);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

