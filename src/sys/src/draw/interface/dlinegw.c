#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: dlinegw.c,v 1.10 1997/03/26 04:25:20 bsmith Exp balay $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "src/draw/drawimpl.h"  /*I "draw.h" I*/

#undef __FUNC__  
#define __FUNC__ "DrawLineGetWidth" /* ADIC Ignore */
/*@
   DrawLineGetWidth - Gets the line width for future draws.  The width is
   relative to the user coordinates of the window; 0.0 denotes the natural
   width; 1.0 denotes the interior viewport. 

   Input Parameter:
.  draw - the drawing context

   Output Parameter:
.  width - the width in user coordinates

   Notes: Not currently implemented.

.keywords:  draw, line, get, width

.seealso:  DrawLineSetWidth()
@*/
int DrawLineGetWidth(Draw draw,double *width)
{
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  if (draw->type == DRAW_NULLWINDOW) return 0;
  if (!draw->ops.linegetwidth) SETERRQ(PETSC_ERR_SUP,1,0);
  return (*draw->ops.linegetwidth)(draw,width);
}

