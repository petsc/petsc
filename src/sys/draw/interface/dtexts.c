
/*
       Provides the calling sequences for all the basic PetscDraw routines.
*/
#include <../src/sys/draw/drawimpl.h>  /*I "petscdraw.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawStringSetSize" 
/*@
   PetscDrawStringSetSize - Sets the size for character text.

   Not Collective

   Input Parameters:
+  draw - the drawing context
.  width - the width in user coordinates
-  height - the character height in user coordinates

   Level: advanced

   Note:
   Only a limited range of sizes are available.

   Concepts: string^drawing size

.seealso: PetscDrawString(), PetscDrawStringVertical(), PetscDrawStringGetSize()

@*/
PetscErrorCode  PetscDrawStringSetSize(PetscDraw draw,PetscReal width,PetscReal height)
{
  PetscErrorCode ierr;
  PetscBool  isnull;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)draw,PETSC_DRAW_NULL,&isnull);CHKERRQ(ierr);
  if (isnull) PetscFunctionReturn(0);
  if (draw->ops->stringsetsize) {
    ierr = (*draw->ops->stringsetsize)(draw,width,height);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
