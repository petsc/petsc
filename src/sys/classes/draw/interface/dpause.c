/*
       Provides the calling sequences for all the basic PetscDraw routines.
*/
#include <petsc/private/drawimpl.h>  /*I "petscdraw.h" I*/

/*@
   PetscDrawPause - Waits n seconds or until user input, depending on input
               to PetscDrawSetPause().

   Collective operation on PetscDraw object.

   Input Parameter:
.  draw - the drawing context

   Level: beginner

.seealso: PetscDrawSetPause(), PetscDrawGetPause()
@*/
PetscErrorCode  PetscDrawPause(PetscDraw draw)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  if (draw->ops->pause) {
    ierr = (*draw->ops->pause)(draw);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
   PetscDrawSetPause - Sets the amount of time that program pauses after
   a PetscDrawPause() is called.

   Logically Collective on PetscDraw

   Input Parameters:
+  draw   - the drawing object
-  lpause - number of seconds to pause, -1 implies until user input, -2 pauses only on the PetscDrawDestroy()

   Level: intermediate

   Note:
   By default the pause time is zero unless the -draw_pause option is given
   during PetscDrawCreate().

.seealso: PetscDrawGetPause(), PetscDrawPause()
@*/
PetscErrorCode  PetscDrawSetPause(PetscDraw draw,PetscReal lpause)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  PetscValidLogicalCollectiveReal(draw,lpause,2);
  draw->pause = lpause;
  PetscFunctionReturn(0);
}

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

.seealso: PetscDrawSetPause(), PetscDrawPause()
@*/
PetscErrorCode  PetscDrawGetPause(PetscDraw draw,PetscReal *lpause)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  PetscValidPointer(lpause,2);
  *lpause = draw->pause;
  PetscFunctionReturn(0);
}
