/*$Id: dbuff.c,v 1.22 2000/05/05 22:13:25 balay Exp bsmith $*/
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "src/sys/src/draw/drawimpl.h"  /*I "petscdraw.h" I*/

#undef __FUNC__  
#define __FUNC__ /*<a name="DrawSetDoubleBuffer"></a>*/"DrawSetDoubleBuffer" 
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
