/*$Id: dsetpause.c,v 1.23 2000/09/22 20:41:56 bsmith Exp bsmith $*/
/*
       Provides the calling sequences for all the basic PetscDraw routines.
*/
#include "src/sys/src/draw/drawimpl.h"  /*I "petscdraw.h" I*/

#undef __FUNC__  
#define __FUNC__ "DrawSetPause" 
/*@
   PetscDrawSetPause - Sets the amount of time that program pauses after 
   a PetscDrawPause() is called. 

   Collective on PetscDraw

   Input Parameters:
+  draw - the drawing object
-  pause - number of seconds to pause, -1 implies until user input

   Level: intermediate

   Note:
   By default the pause time is zero unless the -draw_pause option is given 
   during PetscDrawOpenX().

   Concepts: drawing^waiting

.seealso: PetscDrawGetPause(), PetscDrawPause()
@*/
int PetscDrawSetPause(PetscDraw draw,int pause)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_COOKIE);
  draw->pause = pause;
  PetscFunctionReturn(0);
}
