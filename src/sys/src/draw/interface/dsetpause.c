/*$Id: dsetpause.c,v 1.22 2000/07/10 03:38:37 bsmith Exp bsmith $*/
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "src/sys/src/draw/drawimpl.h"  /*I "petscdraw.h" I*/

#undef __FUNC__  
#define __FUNC__ /*<a name="DrawSetPause"></a>*/"DrawSetPause" 
/*@
   DrawSetPause - Sets the amount of time that program pauses after 
   a DrawPause() is called. 

   Collective on Draw

   Input Parameters:
+  draw - the drawing object
-  pause - number of seconds to pause, -1 implies until user input

   Level: intermediate

   Note:
   By default the pause time is zero unless the -draw_pause option is given 
   during DrawOpenX().

   Concepts: drawing^waiting

.seealso: DrawGetPause(), DrawPause()
@*/
int DrawSetPause(Draw draw,int pause)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  draw->pause = pause;
  PetscFunctionReturn(0);
}
