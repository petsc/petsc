#ifndef lint
static char vcid[] = "$Id: dgpause.c,v 1.2 1996/02/08 18:27:49 bsmith Exp bsmith $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "drawimpl.h"  /*I "draw.h" I*/

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
  PETSCVALIDHEADERSPECIFIC(draw,DRAW_COOKIE);
  if (!pause) SETERRQ(1,"DrawGetPause:Null address to store pause");
  if (draw->type == NULLWINDOW) return 0;
  *pause = draw->pause;
  return 0;
}

