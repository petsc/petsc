/*$Id: dpause.c,v 1.25 2000/10/24 20:24:21 bsmith Exp bsmith $*/
/*
       Provides the calling sequences for all the basic PetscDraw routines.
*/
#include "src/sys/src/draw/drawimpl.h"  /*I "petscdraw.h" I*/

#undef __FUNC__  
#define __FUNC__ "DrawPause" 
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
int PetscDrawPause(PetscDraw draw)
{
  int ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_COOKIE);
  if (draw->ops->pause) {
    ierr = (*draw->ops->pause)(draw);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
