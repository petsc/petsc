#define PETSC_DLL
/*
       Provides the calling sequences for all the basic PetscDraw routines.
*/
#include "../src/sys/draw/drawimpl.h"  /*I "petscdraw.h" I*/

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
PetscErrorCode PETSC_DLLEXPORT PetscDrawGetPause(PetscDraw draw,PetscReal *lpause)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_COOKIE,1);
  PetscValidPointer(lpause,2);
  *lpause = draw->pause;
  PetscFunctionReturn(0);
}

