#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: dtext.c,v 1.17 1999/01/12 23:16:28 bsmith Exp bsmith $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "src/sys/src/draw/drawimpl.h"  /*I "draw.h" I*/

#undef __FUNC__  
#define __FUNC__ "DrawString" 
/*@C
   DrawString - Draws text onto a drawable.

   Not Collective

   Input Parameters:
+  draw - the drawing context
.  xl,yl - the coordinates of lower left corner of text
.  cl - the color of the text
-  text - the text to draw

.keywords:  draw, text
@*/
int DrawString(Draw draw,double xl,double yl,int cl,char *text)
{
  int ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  if (PetscTypeCompare(draw->type_name,DRAW_NULL)) PetscFunctionReturn(0);
  ierr = (*draw->ops->string)(draw,xl,yl,cl,text);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

