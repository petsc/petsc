#ifndef lint
static char vcid[] = "$Id: dclear.c,v 1.4 1996/03/19 21:28:06 bsmith Exp gropp $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "drawimpl.h"  /*I "draw.h" I*/

/*@
   DrawClear - Clears graphical output.

   Input Parameters:
.  draw - the drawing context

.keywords: draw, clear
@*/
int DrawClear(Draw draw)
{
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  if (draw->type == NULLWINDOW) return 0;
  if (draw->ops.clear) return (*draw->ops.clear)(draw);
  return 0;
}

/*@
  DrawBOP - Begins a new page or frame on the selected graphical device

   Input Parameters:
.  draw - the drawing context

.keywords: draw, page, frame
@*/
int DrawBOP( Draw draw )
{
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  if (draw->type == NULLWINDOW) return 0;
  if (draw->ops.beginpage) return (*draw->ops.beginpage)(draw);
  return 0;
}

/*@
  DrawEOP - Ends a page or frame on the selected graphical device

   Input Parameters:
.  draw - the drawing context

.keywords: draw, page, frame
@*/
int DrawEOP( Draw draw )
{
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  if (draw->type == NULLWINDOW) return 0;
  if (draw->ops.endpage) return (*draw->ops.endpage)(draw);
  return 0;
}

