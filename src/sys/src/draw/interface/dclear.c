#ifndef lint
static char vcid[] = "$Id: dclear.c,v 1.5 1996/07/08 18:31:22 bsmith Exp bsmith $";
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
  if (draw->type == DRAW_NULLWINDOW) return 0;
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
  if (draw->type == DRAW_NULLWINDOW) return 0;
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
  if (draw->type == DRAW_NULLWINDOW) return 0;
  if (draw->ops.endpage) return (*draw->ops.endpage)(draw);
  return 0;
}

