#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: dtextgs.c,v 1.11 1997/05/07 01:46:33 balay Exp balay $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "src/draw/drawimpl.h"  /*I "draw.h" I*/

#undef __FUNC__  
#define __FUNC__ "DrawStringGetSize" /* ADIC Ignore */
/*@
   DrawStringGetSize - Gets the size for charactor text.  The width is 
   relative to the user coordinates of the window; 0.0 denotes the natural
   width; 1.0 denotes the entire viewport. 

   Input Parameters:
.  draw - the drawing context
.  width - the width in user coordinates
.  height - the charactor height

.keywords: draw, text, get, size
@*/
int DrawStringGetSize(Draw draw,double *width,double *height)
{
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  if (draw->type == DRAW_NULLWINDOW) return 0;
  return (*draw->ops.textgetsize)(draw,width,height);
}

