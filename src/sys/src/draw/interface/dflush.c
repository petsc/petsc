/*$Id: dflush.c,v 1.19 1999/10/24 14:01:10 bsmith Exp bsmith $*/
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "src/sys/src/draw/drawimpl.h"  /*I "draw.h" I*/

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"DrawFlush" 
/*@
   DrawFlush - Flushs graphical output.

   Not collective (Use DrawSynchronizedFlush() for collective)

   Input Parameters:
.  draw - the drawing context

   Level: beginner

.keywords:  draw, flush

.seealso: DrawSynchronizedFlush()
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
