/*$Id: dtext.c,v 1.28 2000/07/10 03:38:37 bsmith Exp bsmith $*/
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "src/sys/src/draw/drawimpl.h"  /*I "petscdraw.h" I*/

#undef __FUNC__  
#define __FUNC__ /*<a name="DrawString"></a>*/"DrawString" 
/*@C
   DrawString - Draws text onto a drawable.

   Not Collective

   Input Parameters:
+  draw - the drawing context
.  xl,yl - the coordinates of lower left corner of text
.  cl - the color of the text
-  text - the text to draw

   Level: beginner

   Concepts: drawing^string
   Concepts: string^drawing

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

