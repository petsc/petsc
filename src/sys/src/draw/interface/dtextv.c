#ifndef lint
static char vcid[] = "$Id: dtextv.c,v 1.3 1996/03/10 17:28:57 bsmith Exp bsmith $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "drawimpl.h"  /*I "draw.h" I*/

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
  if (draw->type == NULLWINDOW) return 0;
  return (*draw->ops.textvertical)(draw,xl,yl,cl,text);
}

