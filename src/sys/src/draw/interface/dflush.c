/*$Id: dflush.c,v 1.27 2001/03/23 23:20:08 balay Exp $*/
/*
       Provides the calling sequences for all the basic PetscDraw routines.
*/
#include "src/sys/src/draw/drawimpl.h"  /*I "petscdraw.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawFlush" 
/*@
   PetscDrawFlush - Flushs graphical output.

   Not collective (Use PetscDrawSynchronizedFlush() for collective)

   Input Parameters:
.  draw - the drawing context

   Level: beginner

   Concepts: flushing^graphics

.seealso: PetscDrawSynchronizedFlush()
@*/
int PetscDrawFlush(PetscDraw draw)
{
  int ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_COOKIE,1);
  if (draw->ops->flush) {
    ierr = (*draw->ops->flush)(draw);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
