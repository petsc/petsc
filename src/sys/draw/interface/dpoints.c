#define PETSC_DLL
/*
       Provides the calling sequences for all the basic PetscDraw routines.
*/
#include "../src/sys/draw/drawimpl.h"  /*I "petscdraw.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawPointSetSize" 
/*@
   PetscDrawPointSetSize - Sets the point size for future draws.  The size is
   relative to the user coordinates of the window; 0.0 denotes the natural
   width, 1.0 denotes the entire viewport. 

   Not collective

   Input Parameters:
+  draw - the drawing context
-  width - the width in user coordinates

   Level: advanced

   Note: 
   Even a size of zero insures that a single pixel is colored.

   Concepts: point^drawing size

.seealso: PetscDrawPoint()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscDrawPointSetSize(PetscDraw draw,PetscReal width)
{
  PetscErrorCode ierr;
  PetscTruth isnull;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_COOKIE,1);
  ierr = PetscTypeCompare((PetscObject)draw,PETSC_DRAW_NULL,&isnull);CHKERRQ(ierr);
  if (isnull) PetscFunctionReturn(0);
  if (width < 0.0 || width > 1.0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Bad size %G, should be between 0 and 1",width);
  if (draw->ops->pointsetsize) {
    ierr = (*draw->ops->pointsetsize)(draw,width);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

