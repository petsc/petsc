#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: dclear.c,v 1.14 1997/08/22 15:15:58 bsmith Exp bsmith $";
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
  int ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  if (draw->type == DRAW_NULLWINDOW) PetscFunctionReturn(0);
  if (draw->ops.clear) {
    ierr = (*draw->ops.clear)(draw);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
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
  int ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  if (draw->type == DRAW_NULLWINDOW) PetscFunctionReturn(0);
  if (draw->ops.beginpage) {
    ierr = (*draw->ops.beginpage)(draw);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
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
  int ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  if (draw->type == DRAW_NULLWINDOW) PetscFunctionReturn(0);
  if (draw->ops.endpage) {
    ierr =  (*draw->ops.endpage)(draw);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

