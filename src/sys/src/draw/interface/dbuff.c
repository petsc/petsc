#ifndef lint
static char vcid[] = "$Id: dbuff.c,v 1.8 1997/01/06 20:26:34 balay Exp bsmith $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "src/draw/drawimpl.h"  /*I "draw.h" I*/

#undef __FUNC__  
#define __FUNC__ "DrawSetDoubleBuffer" /* ADIC Ignore */
/*@
   DrawSetDoubleBuffer - Sets a window to be double buffered. 

   Input Parameter:
.  draw - the drawing context

.keywords:  draw, set, double, buffer
@*/
int DrawSetDoubleBuffer(Draw draw)
{
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  if (draw->type == DRAW_NULLWINDOW) return 0;
  if (draw->ops.setdoublebuffer) return (*draw->ops.setdoublebuffer)(draw);
  return 0;
}
