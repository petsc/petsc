#ifndef lint
static char vcid[] = "$Id: dclear.c,v 1.11 1997/01/06 20:26:34 balay Exp bsmith $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "src/draw/drawimpl.h"  /*I "draw.h" I*/

#undef __FUNC__  
#define __FUNC__ "DrawClear" /* ADIC Ignore */
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

#undef __FUNC__  
#define __FUNC__ "DrawBOP" /* ADIC Ignore */
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
#undef __FUNC__  
#define __FUNC__ "DrawEOP" /* ADIC Ignore */
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

