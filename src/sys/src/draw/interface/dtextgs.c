#ifndef lint
static char vcid[] = "$Id: dtextgs.c,v 1.3 1996/03/10 17:28:57 bsmith Exp bsmith $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "drawimpl.h"  /*I "draw.h" I*/

/*@
   DrawTextGetSize - Gets the size for charactor text.  The width is 
   relative to the user coordinates of the window; 0.0 denotes the natural
   width; 1.0 denotes the interior viewport. 

   Input Parameters:
.  draw - the drawing context
.  width - the width in user coordinates
.  height - the charactor height

.keywords: draw, text, get, size
@*/
int DrawTextGetSize(Draw draw,double *width,double *height)
{
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  if (draw->type == NULLWINDOW) return 0;
  return (*draw->ops.textgetsize)(draw,width,height);
}

