#ifndef lint
static char vcid[] = "$Id: dtextv.c,v 1.6 1996/08/08 14:44:45 bsmith Exp balay $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "src/draw/drawimpl.h"  /*I "draw.h" I*/

#undef __FUNCTION__  
#define __FUNCTION__ "DrawTextVertical"
/*@C
   DrawTextVertical - Draws text onto a drawable.

   Input Parameters:
.  draw - the drawing context
.  xl,yl - the coordinates of upper left corner of text
.  cl - the color of the text
.  text - the text to draw

.keywords: draw, text, vertical
@*/
int DrawTextVertical(Draw draw,double xl,double yl,int cl,char *text)
{
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  if (draw->type == DRAW_NULLWINDOW) return 0;
  return (*draw->ops.textvertical)(draw,xl,yl,cl,text);
}

