#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: dtextv.c,v 1.12 1997/08/22 15:15:58 bsmith Exp bsmith $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "src/draw/drawimpl.h"  /*I "draw.h" I*/

#undef __FUNC__  
#define __FUNC__ "DrawStringVertical" 
/*@C
   DrawStringVertical - Draws text onto a drawable.

   Input Parameters:
.  draw - the drawing context
.  xl,yl - the coordinates of upper left corner of text
.  cl - the color of the text
.  text - the text to draw

.keywords: draw, text, vertical
@*/
int DrawStringVertical(Draw draw,double xl,double yl,int cl,char *text)
{
  int ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  if (draw->type == DRAW_NULLWINDOW) PetscFunctionReturn(0);
  ierr = (*draw->ops.textvertical)(draw,xl,yl,cl,text);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

