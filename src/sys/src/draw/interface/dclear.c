#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: dclear.c,v 1.13 1997/07/09 20:57:34 balay Exp bsmith $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "src/draw/drawimpl.h"  /*I "draw.h" I*/

#undef __FUNC__  
#define __FUNC__ "DrawClear" 
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
#define __FUNC__ "DrawBOP" 
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
#define __FUNC__ "DrawEOP" 
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

