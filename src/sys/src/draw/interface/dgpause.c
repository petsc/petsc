/*$Id: dgpause.c,v 1.25 2000/10/24 20:24:21 bsmith Exp bsmith $*/
/*
       Provides the calling sequences for all the basic PetscDraw routines.
*/
#include "src/sys/src/draw/drawimpl.h"  /*I "petscdraw.h" I*/

#undef __FUNC__  
#define __FUNC__ "DrawGetPause" 
/*@
   PetscDrawGetPause - Gets the amount of time that program pauses after 
   a PetscDrawPause() is called. 

   Not collective

   Input Parameters:
+  draw - the drawing object
-  pause - number of seconds to pause, -1 implies until user input

   Level: intermediate

   Note:
   By default the pause time is zero unless the -draw_pause option is given 
 
   Concepts: waiting^for user input
   Concepts: drawing^waiting
   Concepts: graphics^waiting

.seealso: PetscDrawSetPause(), PetscDrawPause()
@*/
int PetscDrawGetPause(PetscDraw draw,int *pause)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_COOKIE);
  PetscValidIntPointer(pause);
  *pause = draw->pause;
  PetscFunctionReturn(0);
}

