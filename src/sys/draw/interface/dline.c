
/*
       Provides the calling sequences for all the basic PetscDraw routines.
*/
#include <../src/sys/draw/drawimpl.h>  /*I "petscdraw.h" I*/
  
#undef __FUNCT__  
#define __FUNCT__ "PetscDrawLine" 
/*@
   PetscDrawLine - PetscDraws a line onto a drawable.

   Not collective

   Input Parameters:
+  draw - the drawing context
.  xl,yl,xr,yr - the coordinates of the line endpoints
-  cl - the colors of the endpoints

   Level: beginner

   Concepts: line^drawing
   Concepts: drawing^line

@*/
PetscErrorCode  PetscDrawLine(PetscDraw draw,PetscReal xl,PetscReal yl,PetscReal xr,PetscReal yr,int cl)
{
  PetscErrorCode ierr;
  PetscBool  isdrawnull;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)draw,PETSC_DRAW_NULL,&isdrawnull);CHKERRQ(ierr);
  if (isdrawnull) PetscFunctionReturn(0);
  ierr = (*draw->ops->line)(draw,xl,yl,xr,yr,cl);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawArrow" 
/*@
   PetscDrawArrow - PetscDraws a line with arrow head at end if the line is long enough

   Not collective

   Input Parameters:
+  draw - the drawing context
.  xl,yl,xr,yr - the coordinates of the line endpoints
-  cl - the colors of the endpoints

   Level: beginner

   Concepts: line^drawing
   Concepts: drawing^line

@*/
PetscErrorCode  PetscDrawArrow(PetscDraw draw,PetscReal xl,PetscReal yl,PetscReal xr,PetscReal yr,int cl)
{
  PetscErrorCode ierr;
  PetscBool  isdrawnull;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)draw,PETSC_DRAW_NULL,&isdrawnull);CHKERRQ(ierr);
  if (isdrawnull) PetscFunctionReturn(0);
  ierr = (*draw->ops->arrow)(draw,xl,yl,xr,yr,cl);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

