#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: dtextv.c,v 1.20 1999/10/01 21:20:18 bsmith Exp bsmith $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "src/sys/src/draw/drawimpl.h"  /*I "draw.h" I*/

#undef __FUNC__  
#define __FUNC__ "DrawStringVertical" 
/*@C
   DrawStringVertical - Draws text onto a drawable.

   Not Collective

   Input Parameters:
+  draw - the drawing context
.  xl,yl - the coordinates of upper left corner of text
.  cl - the color of the text
-  text - the text to draw

   Level: beginner

.keywords: draw, text, vertical

.seealso: DrawString()

@*/
int DrawStringVertical(Draw draw,double xl,double yl,int cl,char *text)
{
  int ierr,isnull;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  isnull = PetscTypeCompare(draw,DRAW_NULL);
  if (isnull) PetscFunctionReturn(0);
  ierr = (*draw->ops->stringvertical)(draw,xl,yl,cl,text);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

