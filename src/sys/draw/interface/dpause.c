
/*
       Provides the calling sequences for all the basic PetscDraw routines.
*/
#include <../src/sys/draw/drawimpl.h>  /*I "petscdraw.h" I*/

#undef __FUNCT__
#define __FUNCT__ "PetscDrawPause"
/*@
   PetscDrawPause - Waits n seconds or until user input, depending on input
               to PetscDrawSetPause().

   Collective operation on PetscDraw object.

   Input Parameter:
.  draw - the drawing context

   Level: beginner

   Concepts: waiting^for user input
   Concepts: drawing^waiting
   Concepts: graphics^waiting

.seealso: PetscDrawSetPause(), PetscDrawGetPause()
@*/
PetscErrorCode  PetscDrawPause(PetscDraw draw)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  if (draw->ops->pause) {
    ierr = (*draw->ops->pause)(draw);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
