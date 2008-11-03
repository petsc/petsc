#define PETSC_DLL
/*
       Provides the calling sequences for all the basic PetscDraw routines.
*/
#include "../src/sys/draw/drawimpl.h"  /*I "petscdraw.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawStringVertical" 
/*@C
   PetscDrawStringVertical - PetscDraws text onto a drawable.

   Not Collective

   Input Parameters:
+  draw - the drawing context
.  xl,yl - the coordinates of upper left corner of text
.  cl - the color of the text
-  text - the text to draw

   Level: beginner

   Concepts: string^drawing vertical

.seealso: PetscDrawString()

@*/
PetscErrorCode PETSC_DLLEXPORT PetscDrawStringVertical(PetscDraw draw,PetscReal xl,PetscReal yl,int cl,const char text[])
{
  PetscErrorCode ierr;
  PetscTruth isnull;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_COOKIE,1);
  PetscValidCharPointer(text,5);
  ierr = PetscTypeCompare((PetscObject)draw,PETSC_DRAW_NULL,&isnull);CHKERRQ(ierr);
  if (isnull) PetscFunctionReturn(0);
  ierr = (*draw->ops->stringvertical)(draw,xl,yl,cl,text);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

