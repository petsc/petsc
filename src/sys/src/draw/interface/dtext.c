/*$Id: dtext.c,v 1.25 2000/04/09 04:34:05 bsmith Exp bsmith $*/
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "src/sys/src/draw/drawimpl.h"  /*I "draw.h" I*/

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"DrawString" 
/*@C
   DrawString - Draws text onto a drawable.

   Not Collective

   Input Parameters:
+  draw - the drawing context
.  xl,yl - the coordinates of lower left corner of text
.  cl - the color of the text
-  text - the text to draw

   Level: beginner

.keywords:  draw, text

.seealso: DrawStringVertical()

@*/
int DrawString(Draw draw,PetscReal xl,PetscReal yl,int cl,char *text)
{
  int        ierr ;
  PetscTruth isnull;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  ierr = PetscTypeCompare((PetscObject)draw,DRAW_NULL,&isnull);CHKERRQ(ierr);
  if (isnull) PetscFunctionReturn(0);
  ierr = (*draw->ops->string)(draw,xl,yl,cl,text);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

