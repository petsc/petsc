#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: dpoint.c,v 1.14 1998/04/13 17:46:34 bsmith Exp curfman $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "src/draw/drawimpl.h"  /*I "draw.h" I*/

#undef __FUNC__  
#define __FUNC__ "DrawPoint" 
/*@
   DrawPoint - Draws a point onto a drawable.

   Not collective

   Input Parameters:
+  draw - the drawing context
.  xl,yl - the coordinates of the point
-  cl - the color of the point

.keywords:  draw, point
@*/
int DrawPoint(Draw draw,double xl,double yl,int cl)
{
  int ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  if (draw->type == DRAW_NULLWINDOW) PetscFunctionReturn(0);
  ierr = (*draw->ops->point)(draw,xl,yl,cl);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

