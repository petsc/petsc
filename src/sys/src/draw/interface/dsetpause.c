/*$Id: dsetpause.c,v 1.26 2001/03/23 23:20:08 balay Exp $*/
/*
       Provides the calling sequences for all the basic PetscDraw routines.
*/
#include "src/sys/src/draw/drawimpl.h"  /*I "petscdraw.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawSetPause" 
/*@
   PetscDrawSetPause - Sets the amount of time that program pauses after 
   a PetscDrawPause() is called. 

   Collective on PetscDraw

   Input Parameters:
+  draw   - the drawing object
-  lpause - number of seconds to pause, -1 implies until user input

   Level: intermediate

   Note:
   By default the pause time is zero unless the -draw_pause option is given 
   during PetscDrawOpenX().

   Concepts: drawing^waiting

.seealso: PetscDrawGetPause(), PetscDrawPause()
@*/
int PetscDrawSetPause(PetscDraw draw,int lpause)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_COOKIE);
  draw->pause = lpause;
  PetscFunctionReturn(0);
}
