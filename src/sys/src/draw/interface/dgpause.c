/*$Id: dgpause.c,v 1.28 2001/03/23 23:20:08 balay Exp $*/
/*
       Provides the calling sequences for all the basic PetscDraw routines.
*/
#include "src/sys/src/draw/drawimpl.h"  /*I "petscdraw.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawGetPause" 
/*@
   PetscDrawGetPause - Gets the amount of time that program pauses after 
   a PetscDrawPause() is called. 

   Not collective

   Input Parameters:
+  draw   - the drawing object
-  lpause - number of seconds to pause, -1 implies until user input

   Level: intermediate

   Note:
   By default the pause time is zero unless the -draw_pause option is given 
 
   Concepts: waiting^for user input
   Concepts: drawing^waiting
   Concepts: graphics^waiting

.seealso: PetscDrawSetPause(), PetscDrawPause()
@*/
int PetscDrawGetPause(PetscDraw draw,int *lpause)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_COOKIE);
  PetscValidIntPointer(lpause);
  *lpause = draw->pause;
  PetscFunctionReturn(0);
}

