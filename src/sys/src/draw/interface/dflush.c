#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: dflush.c,v 1.16 1999/01/12 23:16:28 bsmith Exp bsmith $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "src/sys/src/draw/drawimpl.h"  /*I "draw.h" I*/

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
  if (draw->ops->flush) {
    ierr = (*draw->ops->flush)(draw);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
