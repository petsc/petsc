#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: drect.c,v 1.18 1999/03/17 23:21:11 bsmith Exp bsmith $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "src/sys/src/draw/drawimpl.h"  /*I "draw.h" I*/

#undef __FUNC__  
#define __FUNC__ "DrawRectangle" 
/*@
   DrawRectangle - Draws a rectangle  onto a drawable.

   Not Collective

   Input Parameters:
+  draw - the drawing context
.  xl,yl,xr,yr - the coordinates of the lower left, upper right corners
-  c1,c2,c3,c4 - the colors of the four corners in counter clockwise order

   Level: beginner

.keywords: draw, rectangle
@*/
int DrawRectangle(Draw draw,double xl,double yl,double xr,double yr,
                              int c1, int c2,int c3,int c4)
{
  int ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  if (PetscTypeCompare(draw,DRAW_NULL)) PetscFunctionReturn(0);
  ierr = (*draw->ops->rectangle)(draw,xl,yl,xr,yr,c1,c2,c3,c4);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
