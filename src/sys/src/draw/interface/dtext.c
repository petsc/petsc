#ifndef lint
static char vcid[] = "$Id: dtext.c,v 1.1 1996/01/30 19:35:18 bsmith Exp bsmith $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "drawimpl.h"  /*I "draw.h" I*/

/*@C
   DrawText - Draws text onto a drawable.

   Input Parameters:
.  ctx - the drawing context
.  xl,yl - the coordinates of lower left corner of text
.  cl - the color of the text
.  text - the text to draw

.keywords:  draw, text
@*/
int DrawText(Draw ctx,double xl,double yl,int cl,char *text)
{
  PETSCVALIDHEADERSPECIFIC(ctx,DRAW_COOKIE);
  if (ctx->type == NULLWINDOW) return 0;
  return (*ctx->ops.text)(ctx,xl,yl,cl,text);
}

