
/*
       Provides the calling sequences for all the basic PetscDraw routines.
*/
#include <petsc-private/drawimpl.h>  /*I "petscdraw.h" I*/

#undef __FUNCT__
#define __FUNCT__ "PetscDrawMarker"
/*@
   PetscDrawMarker - PetscDraws a marker onto a drawable.

   Not collective

   Input Parameters:
+  draw - the drawing context
.  xl,yl - the coordinates of the marker
-  cl - the color of the marker

   Level: beginner

   Concepts: marker^drawing
   Concepts: drawing^marker

.seealso: PetscDrawPoint()

@*/
PetscErrorCode  PetscDrawMarker(PetscDraw draw,PetscReal xl,PetscReal yl,int cl)
{
  PetscErrorCode ierr;
  PetscBool      isnull;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  ierr = PetscDrawIsNull(draw,&isnull);CHKERRQ(ierr);
  if (isnull) PetscFunctionReturn(0);
  if (draw->ops->coordinatetopixel && draw->ops->pointpixel) {
    PetscInt i,j,k;
    ierr = (*draw->ops->coordinatetopixel)(draw,xl,yl,&i,&j);
    for (k=-2; k<=2; k++) {
      ierr = (*draw->ops->pointpixel)(draw,i+k,j+k,cl);
      ierr = (*draw->ops->pointpixel)(draw,i+k,j-k,cl);
    }
  } else if (draw->ops->string) {
    ierr = (*draw->ops->string)(draw,xl,yl,cl,"x");CHKERRQ(ierr);
  } else SETERRQ(PetscObjectComm((PetscObject)draw),PETSC_ERR_SUP,"No support for drawing marker");
  PetscFunctionReturn(0);
}
