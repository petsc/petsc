#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: dtri.c,v 1.13 1997/07/09 20:57:34 balay Exp balay $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "src/draw/drawimpl.h"  /*I "draw.h" I*/

#undef __FUNC__  
#define __FUNC__ "DrawTriangle" /* ADIC Ignore */
/*@
   DrawTriangle - Draws a triangle  onto a drawable.

   Input Parameters:
.  draw - the drawing context
.  x1,y1,x2,y2,x3,y3 - the coordinates of the vertices
.  c1,c2,c3 - the colors of the corners in counter clockwise order

.keywords: draw, triangle
@*/
int DrawTriangle(Draw draw,double x1,double y1,double x2,double y2,
                 double x3,double y3,int c1, int c2,int c3)
{
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  if (draw->type == DRAW_NULLWINDOW) return 0;
  return (*draw->ops.triangle)(draw,x1,y1,x2,y2,x3,y3,c1,c2,c3);
}


#undef __FUNC__  
#define __FUNC__ "DrawTensorContourPatch" /* ADIC Ignore */
/*@
   DrawTensorContourPatch - Draws a rectangular patch of a contour plot 
      for a two-dimensional array.

   Input Parameters:
.   win - the window to draw in
.   m,n - the number of local mesh points in the x and y direction
.   x,y - the locations of the local mesh points
.   max,min - the maximum and minimum value in the entire contour
.   v - the data

   Options Database Keys:
$  -draw_x_private_colormap

    Note: 
    This may be a basic enough function to be a graphics primative
    but at this time it uses DrawTriangle().

.keywords: Draw, tensor, contour, vector
@*/
int DrawTensorContourPatch(Draw win,int m,int n,double *x,double *y,double max,
                           double min, Scalar *v)
{
  int           c1, c2, c3, c4, i, j,ierr;
  double        x1, x2, x3, x4, y1, y2, y3, y4,scale;

  scale = (245.0 - 16.0)/(max - min);

  /* Draw the contour plot patch */
  for ( j=0; j<n-1; j++ ) {
    for ( i=0; i<m-1; i++ ) {
#if !defined(PETSC_COMPLEX)
      x1 = x[i];  y1 = y[j];  c1 = (int) (16. + scale*(v[i+j*m] - min));
      x2 = x[i+1];y2 = y1;    c2 = (int) (16. + scale*(v[i+j*m+1]-min));
      x3 = x2;    y3 = y[j+1];c3 = (int) (16. + scale*(v[i+j*m+1+m]-min));
      x4 = x1;    y4 = y3;    c4 = (int) (16. + scale*(v[i+j*m+m]-min));
#else
      x1 = x[i];  y1 = y[j];  c1 = (int) (16. + scale*PetscReal(v[i+j*m]-min));
      x2 = x[i+1];y2 = y1;    c2 = (int) (16. + scale*PetscReal(v[i+j*m+1]-min));
      x3 = x2;    y3 = y[j+1];c3 = (int) (16. + scale*PetscReal(v[i+j*m+1+m]-min));
      x4 = x1;    y4 = y3;    c4 = (int) (16. + scale*PetscReal(v[i+j*m+m]-min));
#endif
      ierr = DrawTriangle(win,x1,y1,x2,y2,x3,y3,c1,c2,c3); CHKERRQ(ierr);
      ierr = DrawTriangle(win,x1,y1,x3,y3,x4,y4,c1,c3,c4); CHKERRQ(ierr);
    }
  }

  return 0;
}
