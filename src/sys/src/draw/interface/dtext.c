#ifndef lint
static char vcid[] = "$Id: dtext.c,v 1.7 1996/12/16 18:25:03 balay Exp balay $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "src/draw/drawimpl.h"  /*I "draw.h" I*/

#undef __FUNC__  
#define __FUNC__ "DrawText"
/*@C
   DrawText - Draws text onto a drawable.

   Input Parameters:
.  draw - the drawing context
.  xl,yl - the coordinates of lower left corner of text
.  cl - the color of the text
.  text - the text to draw

.keywords:  draw, text
@*/
int DrawText(Draw draw,double xl,double yl,int cl,char *text)
{
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  if (draw->type == DRAW_NULLWINDOW) return 0;
  return (*draw->ops.text)(draw,xl,yl,cl,text);
}

