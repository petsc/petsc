#define PETSC_DLL
/*
       Provides the calling sequences for all the basic PetscDraw routines.
*/
#include "../src/sys/draw/drawimpl.h"  /*I "petscdraw.h" I*/

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
PetscErrorCode PETSC_DLLEXPORT PetscDrawSetPause(PetscDraw draw,PetscReal lpause)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_COOKIE,1);
  draw->pause = lpause;
  PetscFunctionReturn(0);
}
