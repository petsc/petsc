#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: dlinegw.c,v 1.19 1999/03/17 23:21:11 bsmith Exp bsmith $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "src/sys/src/draw/drawimpl.h"  /*I "draw.h" I*/

#undef __FUNC__  
#define __FUNC__ "DrawLineGetWidth" 
/*@
   DrawLineGetWidth - Gets the line width for future draws.  The width is
   relative to the user coordinates of the window; 0.0 denotes the natural
   width; 1.0 denotes the interior viewport. 

   Not collective

   Input Parameter:
.  draw - the drawing context

   Output Parameter:
.  width - the width in user coordinates

   Level: advanced

   Notes:
   Not currently implemented.

.keywords:  draw, line, get, width

.seealso:  DrawLineSetWidth()
@*/
int DrawLineGetWidth(Draw draw,double *width)
{
  int ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  if (PetscTypeCompare(draw,DRAW_NULL)) PetscFunctionReturn(0);
  if (!draw->ops->linegetwidth) SETERRQ(PETSC_ERR_SUP,1,0);
  ierr = (*draw->ops->linegetwidth)(draw,width);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

