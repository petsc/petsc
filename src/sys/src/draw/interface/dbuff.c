/*$Id: dbuff.c,v 1.21 2000/04/12 04:21:02 bsmith Exp balay $*/
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "src/sys/src/draw/drawimpl.h"  /*I "petscdraw.h" I*/

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"DrawSetDoubleBuffer" 
/*@
   DrawSetDoubleBuffer - Sets a window to be double buffered. 

   Collective on Draw

   Input Parameter:
.  draw - the drawing context

   Level: intermediate

.keywords:  draw, set, double, buffer
@*/
int DrawSetDoubleBuffer(Draw draw)
{
  int ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  if (draw->ops->setdoublebuffer) {
    ierr = (*draw->ops->setdoublebuffer)(draw);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
