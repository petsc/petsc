/*$Id: dbuff.c,v 1.25 2001/01/15 21:43:22 bsmith Exp balay $*/
/*
       Provides the calling sequences for all the basic PetscDraw routines.
*/
#include "src/sys/src/draw/drawimpl.h"  /*I "petscdraw.h" I*/

#undef __FUNC__  
#define __FUNC__ "PetscDrawSetDoubleBuffer" 
/*@
   PetscDrawSetDoubleBuffer - Sets a window to be double buffered. 

   Collective on PetscDraw

   Input Parameter:
.  draw - the drawing context

   Level: intermediate

   Concepts: drawing^double buffer
   Concepts: graphics^double buffer
   Concepts: double buffer

@*/
int PetscDrawSetDoubleBuffer(PetscDraw draw)
{
  int ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_COOKIE);
  if (draw->ops->setdoublebuffer) {
    ierr = (*draw->ops->setdoublebuffer)(draw);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
