/*$Id: dclear.c,v 1.30 2000/09/22 20:41:56 bsmith Exp bsmith $*/
/*
       Provides the calling sequences for all the basic PetscDraw routines.
*/
#include "src/sys/src/draw/drawimpl.h"  /*I "petscdraw.h" I*/

#undef __FUNC__  
#define __FUNC__ "DrawClear" 
/*@
   PetscDrawClear - Clears graphical output.

   Not collective (Use PetscDrawSynchronizedClear() for collective)

   Input Parameter:
.  draw - the drawing context

   Level: beginner

   Concepts: clear^window

.seealso: PetscDrawBOP(), PetscDrawEOP(), PetscDrawSynchronizedClear()
@*/
int PetscDrawClear(PetscDraw draw)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_COOKIE);
  if (draw->ops->clear) {
    ierr = (*draw->ops->clear)(draw);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "DrawBOP" 
/*@
   PetscDrawBOP - Begins a new page or frame on the selected graphical device.

   Collective on PetscDraw

   Input Parameter:
.  draw - the drawing context

   Level: advanced

.seealso: PetscDrawEOP(), PetscDrawClear()
@*/
int PetscDrawBOP(PetscDraw draw)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_COOKIE);
  if (draw->ops->beginpage) {
    ierr = (*draw->ops->beginpage)(draw);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
#undef __FUNC__  
#define __FUNC__ "DrawEOP" 
/*@
   PetscDrawEOP - Ends a page or frame on the selected graphical device.

   Collective on PetscDraw

   Input Parameter:
.  draw - the drawing context

   Level: advanced

.seealso: PetscDrawBOP(), PetscDrawClear()
@*/
int PetscDrawEOP(PetscDraw draw)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_COOKIE);
  if (draw->ops->endpage) {
    ierr =  (*draw->ops->endpage)(draw);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

