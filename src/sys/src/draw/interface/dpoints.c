#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: dpoints.c,v 1.17 1998/03/12 23:20:42 bsmith Exp bsmith $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "src/draw/drawimpl.h"  /*I "draw.h" I*/

#undef __FUNC__  
#define __FUNC__ "DrawPointSetSize" 
/*@
   DrawPointSetSize - Sets the point size for future draws.  The size is
   relative to the user coordinates of the window; 0.0 denotes the natural
   width, 1.0 denotes the entire viewport. 

   Input Parameters:
.  draw - the drawing context
.  width - the width in user coordinates

   Not collective

   Note: 
   Even a size of zero insures that a single pixel is colored.

.keywords: draw, point, set, size
@*/
int DrawPointSetSize(Draw draw,double width)
{
  int ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  if (draw->type == DRAW_NULLWINDOW) PetscFunctionReturn(0);
  if (width < 0.0 || width > 1.0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Bad size");
  ierr = (*draw->ops->pointsetsize)(draw,width);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

