#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: dpoints.c,v 1.24 1999/10/01 21:20:18 bsmith Exp bsmith $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "src/sys/src/draw/drawimpl.h"  /*I "draw.h" I*/

#undef __FUNC__  
#define __FUNC__ "DrawPointSetSize" 
/*@
   DrawPointSetSize - Sets the point size for future draws.  The size is
   relative to the user coordinates of the window; 0.0 denotes the natural
   width, 1.0 denotes the entire viewport. 

   Not collective

   Input Parameters:
+  draw - the drawing context
-  width - the width in user coordinates

   Level: advanced

   Note: 
   Even a size of zero insures that a single pixel is colored.

.keywords: draw, point, set, size

.seealso: DrawPoint()
@*/
int DrawPointSetSize(Draw draw,double width)
{
  int ierr,isnull;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  isnull = PetscTypeCompare(draw,DRAW_NULL);
  if (isnull) PetscFunctionReturn(0);
  if (width < 0.0 || width > 1.0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,0,"Bad size %g, should be between 0 and 1",width);
  ierr = (*draw->ops->pointsetsize)(draw,width);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

