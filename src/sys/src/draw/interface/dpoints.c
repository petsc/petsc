#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: dpoints.c,v 1.13 1997/02/22 02:27:05 bsmith Exp balay $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "src/draw/drawimpl.h"  /*I "draw.h" I*/

#undef __FUNC__  
#define __FUNC__ "DrawPointSetSize" /* ADIC Ignore */
/*@
   DrawPointSetSize - Sets the point size for future draws.  The size is
   relative to the user coordinates of the window; 0.0 denotes the natural
   width, 1.0 denotes the entire viewport. 

   Input Parameters:
.  draw - the drawing context
.  width - the width in user coordinates

   Note: 
   Even a size of zero insures that a single pixel is colored.

.keywords: draw, point, set, size
@*/
int DrawPointSetSize(Draw draw,double width)
{
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  if (draw->type == DRAW_NULLWINDOW) return 0;
  if (width < 0.0 || width > 1.0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Bad size");
  return (*draw->ops.pointsetsize)(draw,width);
}

