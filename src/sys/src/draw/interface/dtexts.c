#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: dtexts.c,v 1.22 1999/10/01 21:20:18 bsmith Exp bsmith $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "src/sys/src/draw/drawimpl.h"  /*I "draw.h" I*/

#undef __FUNC__  
#define __FUNC__ "DrawStringSetSize" 
/*@
   DrawStringSetSize - Sets the size for charactor text.  The width is
   relative to the user coordinates of the window; 0.0 denotes the natural
   width; 1.0 denotes the entire viewport. 

   Not Collective

   Input Parameters:
+  draw - the drawing context
.  width - the width in user coordinates
-  height - the charactor height

   Level: advanced

   Note:
   Only a limited range of sizes are available.

.keywords: draw, text, set, size

.seealso: DrawString(), DrawStringVertical(), DrawStringGetSize()

@*/
int DrawStringSetSize(Draw draw,double width,double height)
{
  int ierr,isnull;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  isnull = PetscTypeCompare(draw,DRAW_NULL);
  if (isnull) PetscFunctionReturn(0);
  if (draw->ops->stringsetsize) {
    ierr = (*draw->ops->stringsetsize)(draw,width,height);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
