#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: dpoint.c,v 1.19 1999/10/01 21:20:18 bsmith Exp bsmith $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "src/sys/src/draw/drawimpl.h"  /*I "draw.h" I*/

#undef __FUNC__  
#define __FUNC__ "DrawPoint" 
/*@
   DrawPoint - Draws a point onto a drawable.

   Not collective

   Input Parameters:
+  draw - the drawing context
.  xl,yl - the coordinates of the point
-  cl - the color of the point

   Level: beginner

.keywords:  draw, point

.seealso: DrawPointSetSize()

@*/
int DrawPoint(Draw draw,double xl,double yl,int cl)
{
  int ierr,isnull;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  isnull = PetscTypeCompare(draw,DRAW_NULL);
  if (isnull) PetscFunctionReturn(0);
  ierr = (*draw->ops->point)(draw,xl,yl,cl);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

