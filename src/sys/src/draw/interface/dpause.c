/*$Id: dpause.c,v 1.22 2000/05/05 22:13:25 balay Exp bsmith $*/
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "src/sys/src/draw/drawimpl.h"  /*I "petscdraw.h" I*/

#undef __FUNC__  
#define __FUNC__ /*<a name="DrawPause"></a>*/"DrawPause" 
/*@
   DrawPause - Waits n seconds or until user input, depending on input 
               to DrawSetPause().

   Collective operation on Draw object.

   Input Parameter:
.  draw - the drawing context

   Level: beginner

.keywords: draw, pause

.seealso: DrawSetPause(), DrawGetPause()
@*/
int DrawPause(Draw draw)
{
  int ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  if (draw->ops->pause) {
    ierr = (*draw->ops->pause)(draw);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
