#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: dflush.c,v 1.14 1998/04/13 17:46:34 bsmith Exp curfman $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "src/draw/drawimpl.h"  /*I "draw.h" I*/

#undef __FUNC__  
#define __FUNC__ "DrawFlush" 
/*@
   DrawFlush - Flushs graphical output.

   Not collective (Use DrawSynchronizedFlush() for collective)

   Input Parameters:
.  draw - the drawing context

.keywords:  draw, flush
@*/
int DrawFlush(Draw draw)
{
  int ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  if (draw->type == DRAW_NULLWINDOW) PetscFunctionReturn(0);
  if (draw->ops->flush) {
    ierr = (*draw->ops->flush)(draw);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
