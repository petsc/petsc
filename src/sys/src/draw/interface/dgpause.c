#ifndef lint
static char vcid[] = "$Id: dgpause.c,v 1.5 1996/07/08 22:21:15 bsmith Exp bsmith $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "src/draw/drawimpl.h"  /*I "draw.h" I*/

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
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  if (!pause) SETERRQ(1,"DrawGetPause:Null address to store pause");
  if (draw->type == DRAW_NULLWINDOW) return 0;
  *pause = draw->pause;
  return 0;
}

