#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: drect.c,v 1.10 1997/07/09 20:57:34 balay Exp bsmith $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "src/draw/drawimpl.h"  /*I "draw.h" I*/

#undef __FUNC__  
#define __FUNC__ "DrawRectangle" 
/*@
   DrawRectangle - Draws a rectangle  onto a drawable.

   Input Parameters:
.  draw - the drawing context
.  xl,yl,xr,yr - the coordinates of the lower left, upper right corners
.  c1,c2,c3,c4 - the colors of the four corners in counter clockwise order

.keywords: draw, rectangle
@*/
int DrawRectangle(Draw draw,double xl,double yl,double xr,double yr,
                              int c1, int c2,int c3,int c4)
{
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  if (draw->type == DRAW_NULLWINDOW) return 0;
  return (*draw->ops.rectangle)(draw,xl,yl,xr,yr,c1,c2,c3,c4);
}
