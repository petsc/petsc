/*
       Provides the calling sequences for all the basic PetscDraw routines.
*/
#include "src/sys/src/draw/drawimpl.h"  /*I "petscdraw.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawStringGetSize" 
/*@
   PetscDrawStringGetSize - Gets the size for character text.  The width is 
   relative to the user coordinates of the window; 0.0 denotes the natural
   width; 1.0 denotes the entire viewport. 

   Not Collective

   Input Parameters:
+  draw - the drawing context
.  width - the width in user coordinates
-  height - the character height

   Level: advanced

   Concepts: string^drawing size

.seealso: PetscDrawString(), PetscDrawStringVertical(), PetscDrawStringSetSize()

@*/
int PetscDrawStringGetSize(PetscDraw draw,PetscReal *width,PetscReal *height)
{
  int        ierr;
  PetscTruth isnull;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_COOKIE,1);
  ierr = PetscTypeCompare((PetscObject)draw,PETSC_DRAW_NULL,&isnull);CHKERRQ(ierr);
  if (isnull) PetscFunctionReturn(0);
  if (!draw->ops->stringgetsize) SETERRQ1(1,"This draw object %s does not support getting string size",draw->type_name);
  ierr = (*draw->ops->stringgetsize)(draw,width,height);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

