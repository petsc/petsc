/*
       Provides the calling sequences for all the basic PetscDraw routines.
*/
#include <petsc/private/drawimpl.h> /*I "petscdraw.h" I*/

/*@
   PetscDrawPause - Waits n seconds or until user input, depending on input
               to `PetscDrawSetPause()`.

   Collective on draw

   Input Parameter:
.  draw - the drawing context

   Level: beginner

.seealso: `PetscDraw`, `PetscDrawSetPause()`, `PetscDrawGetPause()`
@*/
PetscErrorCode PetscDrawPause(PetscDraw draw)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw, PETSC_DRAW_CLASSID, 1);
  PetscTryTypeMethod(draw, pause);
  PetscFunctionReturn(0);
}

/*@
   PetscDrawSetPause - Sets the amount of time that program pauses after
   a `PetscDrawPause()` is called.

   Logically Collective on draw

   Input Parameters:
+  draw   - the drawing object
-  lpause - number of seconds to pause, -1 implies until user input, -2 pauses only on the `PetscDrawDestroy()`

   Options Database Key:
.  -draw_pause value - set the time to pause

   Level: intermediate

   Note:
   By default the pause time is zero unless the -draw_pause option is given
   during PetscDrawCreate().

.seealso: `PetscDraw`, `PetscDrawGetPause()`, `PetscDrawPause()`
@*/
PetscErrorCode PetscDrawSetPause(PetscDraw draw, PetscReal lpause)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw, PETSC_DRAW_CLASSID, 1);
  PetscValidLogicalCollectiveReal(draw, lpause, 2);
  draw->pause = lpause;
  PetscFunctionReturn(0);
}

/*@
   PetscDrawGetPause - Gets the amount of time that program pauses after
   a `PetscDrawPause()` is called.

   Not collective

   Input Parameters:
+  draw   - the drawing object
-  lpause - number of seconds to pause, -1 implies until user input

   Level: intermediate

   Note:
   By default the pause time is zero unless the -draw_pause option is given

.seealso: `PetscDraw`, `PetscDrawSetPause()`, `PetscDrawPause()`
@*/
PetscErrorCode PetscDrawGetPause(PetscDraw draw, PetscReal *lpause)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw, PETSC_DRAW_CLASSID, 1);
  PetscValidRealPointer(lpause, 2);
  *lpause = draw->pause;
  PetscFunctionReturn(0);
}
