#ifndef lint
static char vcid[] = "$Id: dclear.c,v 1.8 1996/09/27 19:35:37 curfman Exp curfman $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "src/draw/drawimpl.h"  /*I "draw.h" I*/

/*@
   DrawClear - Clears graphical output.

   Input Parameter:
.  draw - the drawing context

.keywords: draw, clear

.seealso: DrawBOP(), DrawEOP()
@*/
int DrawClear(Draw draw)
{
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  if (draw->type == DRAW_NULLWINDOW) return 0;
  if (draw->ops.clear) return (*draw->ops.clear)(draw);
  return 0;
}

/*@
   DrawBOP - Begins a new page or frame on the selected graphical device.

   Input Parameter:
.  draw - the drawing context

.keywords: draw, page, frame

.seealso: DrawEOP(), DrawClear()
@*/
int DrawBOP( Draw draw )
{
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  if (draw->type == DRAW_NULLWINDOW) return 0;
  if (draw->ops.beginpage) return (*draw->ops.beginpage)(draw);
  return 0;
}

/*@
   DrawEOP - Ends a page or frame on the selected graphical device.

   Input Parameter:
.  draw - the drawing context

.keywords: draw, page, frame

.seealso: DrawBOP(), DrawClear()
@*/
int DrawEOP( Draw draw )
{
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  if (draw->type == DRAW_NULLWINDOW) return 0;
  if (draw->ops.endpage) return (*draw->ops.endpage)(draw);
  return 0;
}

