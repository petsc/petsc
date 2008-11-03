#define PETSC_DLL
/*
       Provides the calling sequences for all the basic PetscDraw routines.
*/
#include "../src/sys/draw/drawimpl.h"  /*I "petscdraw.h" I*/
  
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
PetscErrorCode PETSC_DLLEXPORT PetscDrawLine(PetscDraw draw,PetscReal xl,PetscReal yl,PetscReal xr,PetscReal yr,int cl)
{
  PetscErrorCode ierr;
  PetscTruth isdrawnull;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_COOKIE,1);
  ierr = PetscTypeCompare((PetscObject)draw,PETSC_DRAW_NULL,&isdrawnull);CHKERRQ(ierr);
  if (isdrawnull) PetscFunctionReturn(0);
  ierr = (*draw->ops->line)(draw,xl,yl,xr,yr,cl);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawIsNull" 
/*@
   PetscDrawIsNull - Returns PETSC_TRUE if draw is a null draw object.

   Not collective

   Input Parameter:
.  draw - the draw context

   Output Parameter:
.  yes - PETSC_TRUE if it is a null draw object; otherwise PETSC_FALSE

   Level: advanced

@*/
PetscErrorCode PETSC_DLLEXPORT PetscDrawIsNull(PetscDraw draw,PetscTruth *yes)
{
  PetscErrorCode ierr;
  PetscTruth isdrawnull;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_COOKIE,1);
  PetscValidIntPointer(yes,2);
  ierr = PetscTypeCompare((PetscObject)draw,PETSC_DRAW_NULL,&isdrawnull);CHKERRQ(ierr);
  if (isdrawnull) *yes = PETSC_TRUE;
  else            *yes = PETSC_FALSE;
  PetscFunctionReturn(0);
}
