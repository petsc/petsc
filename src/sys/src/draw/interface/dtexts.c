/*$Id: dtexts.c,v 1.29 2000/04/12 04:21:02 bsmith Exp balay $*/
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "src/sys/src/draw/drawimpl.h"  /*I "petscdraw.h" I*/

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"DrawStringSetSize" 
/*@
   DrawStringSetSize - Sets the size for charactor text.

   Not Collective

   Input Parameters:
+  draw - the drawing context
.  width - the width in user coordinates
-  height - the charactor height in user coordinates

   Level: advanced

   Note:
   Only a limited range of sizes are available.

.keywords: draw, text, set, size

.seealso: DrawString(), DrawStringVertical(), DrawStringGetSize()

@*/
int DrawStringSetSize(Draw draw,PetscReal width,PetscReal height)
{
  int        ierr;
  PetscTruth isnull;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  ierr = PetscTypeCompare((PetscObject)draw,DRAW_NULL,&isnull);CHKERRQ(ierr);
  if (isnull) PetscFunctionReturn(0);
  if (draw->ops->stringsetsize) {
    ierr = (*draw->ops->stringsetsize)(draw,width,height);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
