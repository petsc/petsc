#ifndef lint
static char vcid[] = "$Id: dtextv.c,v 1.1 1996/01/30 19:35:18 bsmith Exp bsmith $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "drawimpl.h"  /*I "draw.h" I*/

/*@C
   DrawTextVertical - Draws text onto a drawable.

   Input Parameters:
.  ctx - the drawing context
.  xl,yl - the coordinates of upper left corner of text
.  cl - the color of the text
.  text - the text to draw

.keywords: draw, text, vertical
@*/
int DrawTextVertical(Draw ctx,double xl,double yl,int cl,char *text)
{
  PETSCVALIDHEADERSPECIFIC(ctx,DRAW_COOKIE);
  if (ctx->type == NULLWINDOW) return 0;
  return (*ctx->ops.textvertical)(ctx,xl,yl,cl,text);
}

