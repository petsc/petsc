#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: dtextgs.c,v 1.20 1999/03/17 23:21:11 bsmith Exp bsmith $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "src/sys/src/draw/drawimpl.h"  /*I "draw.h" I*/

#undef __FUNC__  
#define __FUNC__ "DrawStringGetSize" 
/*@
   DrawStringGetSize - Gets the size for charactor text.  The width is 
   relative to the user coordinates of the window; 0.0 denotes the natural
   width; 1.0 denotes the entire viewport. 

   Not Collective

   Input Parameters:
+  draw - the drawing context
.  width - the width in user coordinates
-  height - the charactor height

   Level: advanced

.keywords: draw, text, get, size

.seealso: DrawString(), DrawStringVertical(), DrawStringSetSize()

@*/
int DrawStringGetSize(Draw draw,double *width,double *height)
{
  int ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  if (PetscTypeCompare(draw,DRAW_NULL)) PetscFunctionReturn(0);
  ierr = (*draw->ops->stringgetsize)(draw,width,height);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

