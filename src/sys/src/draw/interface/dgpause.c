/*$Id: dgpause.c,v 1.22 2000/05/05 22:13:25 balay Exp bsmith $*/
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "src/sys/src/draw/drawimpl.h"  /*I "petscdraw.h" I*/

#undef __FUNC__  
#define __FUNC__ /*<a name="DrawGetPause"></a>*/"DrawGetPause" 
/*@
   DrawGetPause - Gets the amount of time that program pauses after 
   a DrawPause() is called. 

   Not collective

   Input Parameters:
+  draw - the drawing object
-  pause - number of seconds to pause, -1 implies until user input

   Level: intermediate

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
  *pause = draw->pause;
  PetscFunctionReturn(0);
}

