#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: dgpause.c,v 1.12 1997/08/22 15:15:58 bsmith Exp bsmith $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "src/draw/drawimpl.h"  /*I "draw.h" I*/

#undef __FUNC__  
#define __FUNC__ "DrawGetPause" 
/*@
   DrawGetPause - Gets the amount of time that program pauses after 
   a DrawPause() is called. 

   Input Paramters:
.  draw - the drawing object
.  pause - number of seconds to pause, -1 implies until user input

   Note:
   By default the pause time is zero unless the -draw_pause option is given 
   during DrawOpenX().

.keywords: draw, set, pause

.seealso: DrawSetPause(), DrawPause()
@*/
int DrawGetPause(Draw draw,int *pause)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  PetscValidIntPointer(pause);
  if (draw->type == DRAW_NULLWINDOW) PetscFunctionReturn(0);
  *pause = draw->pause;
  PetscFunctionReturn(0);
}

