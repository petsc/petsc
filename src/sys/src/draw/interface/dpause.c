#ifndef lint
static char vcid[] = "$Id: dpause.c,v 1.7 1996/12/18 20:55:58 balay Exp balay $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "src/draw/drawimpl.h"  /*I "draw.h" I*/

#undef __FUNC__  
#define __FUNC__ "DrawPause"
/*@
   DrawPause - Waits n seconds or until user input, depending on input 
               to DrawSetPause().

   Input Parameter:
.  draw - the drawing context

.keywords: draw, pause

.seealso: DrawSetPause(), DrawGetPause()
@*/
int DrawPause(Draw draw)
{
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  if (draw->type == DRAW_NULLWINDOW) return 0;
  if (draw->ops.pause) return (*draw->ops.pause)(draw);
  return 0;
}
