/*$Id: dbuff.c,v 1.27 2001/03/23 23:20:08 balay Exp $*/
/*
       Provides the calling sequences for all the basic PetscDraw routines.
*/
#include "src/sys/src/draw/drawimpl.h"  /*I "petscdraw.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawSetDoubleBuffer" 
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
  PetscValidHeaderSpecific(draw,PETSC_DRAW_COOKIE,1);
  if (draw->ops->setdoublebuffer) {
    ierr = (*draw->ops->setdoublebuffer)(draw);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
