#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: dbuff.c,v 1.13 1998/03/12 23:20:42 bsmith Exp bsmith $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "src/draw/drawimpl.h"  /*I "draw.h" I*/

#undef __FUNC__  
#define __FUNC__ "DrawSetDoubleBuffer" 
/*@
   DrawSetDoubleBuffer - Sets a window to be double buffered. 

   Input Parameter:
.  draw - the drawing context

   Collective on Draw

.keywords:  draw, set, double, buffer
@*/
int DrawSetDoubleBuffer(Draw draw)
{
  int ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  if (draw->type == DRAW_NULLWINDOW) PetscFunctionReturn(0);
  if (draw->ops->setdoublebuffer) {
    ierr = (*draw->ops->setdoublebuffer)(draw);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
