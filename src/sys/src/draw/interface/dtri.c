#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: dtri.c,v 1.25 1999/03/07 17:25:17 bsmith Exp bsmith $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "src/sys/src/draw/drawimpl.h"  /*I "draw.h" I*/

#undef __FUNC__  
#define __FUNC__ "DrawTriangle" 
/*@
   DrawTriangle - Draws a triangle  onto a drawable.

   Not Collective

   Input Parameters:
+  draw - the drawing context
.  x1,y1,x2,y2,x3,y3 - the coordinates of the vertices
-  c1,c2,c3 - the colors of the corners in counter clockwise order

.keywords: draw, triangle
@*/
int DrawTriangle(Draw draw,double x1,double y_1,double x2,double y2,
                 double x3,double y3,int c1, int c2,int c3)
{
  int ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  if (PetscTypeCompare(draw->type_name,DRAW_NULL)) PetscFunctionReturn(0);
  ierr = (*draw->ops->triangle)(draw,x1,y_1,x2,y2,x3,y3,c1,c2,c3);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ "DrawScalePopup"
/*@
       DrawScalePopup - Draws a contour scale window. 

     Collective on Draw

  Input Parameters:
+    popup - the window (often a window obtained via DrawGetPopup()
.    min - minimum value being plotted
-    max - maximum value being plotted

  Notes:
     All processors that share the draw MUST call this routine

@*/
int DrawScalePopup(Draw popup,double min,double max)
{
  double   xl = 0.0, yl = 0.0, xr = 1.0, yr = 1.0,value;
  int      i,c = DRAW_BASIC_COLORS,rank,ierr;
  char     string[32];
  MPI_Comm comm;

  PetscFunctionBegin;
  ierr = DrawCheckResizedWindow(popup); CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject) popup,&comm);CHKERRQ(ierr);
  MPI_Comm_rank(comm,&rank);
  if (rank) PetscFunctionReturn(0);

  for ( i=0; i<10; i++ ) {
    ierr = DrawRectangle(popup,xl,yl,xr,yr,c,c,c,c);CHKERRQ(ierr);
    yl += .1; yr += .1; c = (int) ((double) c + (245.-DRAW_BASIC_COLORS)/9.);
  }
  for ( i=0; i<10; i++ ) {
    value = min + i*(max-min)/9.0;
    /* look for a value that should be zero, but is not due to round-off */
    if (PetscAbsDouble(value) < 1.e-10 && max-min > 1.e-6) value = 0.0;
    sprintf(string,"%g",value);
    ierr = DrawString(popup,.2,.02 + i/10.0,DRAW_BLACK,string);CHKERRQ(ierr);
  }
  ierr = DrawSetTitle(popup,"Contour Scale");CHKERRQ(ierr);
  ierr = DrawFlush(popup);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ "DrawTensorContour"
/*@C
   DrawTensorContour - Draws a contour plot for a two-dimensional array
   that is stored as a PETSc vector.

   Collective on Draw, but Draw must be sequential

   Input Parameters:
+   win   - the window to draw in
.   m,n   - the global number of mesh points in the x and y directions
.   xi,yi - the locations of the global mesh points (optional, use PETSC_NULL
            to indicate uniform spacing on [0,1])
-   V     - the values

   Options Database Keys:
+  -draw_x_shared_colormap - Indicates use of private colormap
-  -draw_contour_grid - Draws grid contour

.keywords: Draw, tensor, contour, vector

.seealso: DrawTensorContourPatch()

@*/
int DrawTensorContour(Draw win,int m,int n,const double xi[],const double yi[],Scalar *v)
{
  int           N = m*n, ierr;
  PetscTruth    isnull;
  Draw          popup;
  MPI_Comm      comm;
  double        *x,*y;
  int           xin=1,yin=1,i,showgrid,size;
  double        h, min, max;

  PetscFunctionBegin;
  ierr = DrawIsNull(win,&isnull); CHKERRQ(ierr); if (isnull) PetscFunctionReturn(0);
  ierr = PetscObjectGetComm((PetscObject)win,&comm);CHKERRQ(ierr);
  MPI_Comm_size(comm,&size);
  if (size > 1) SETERRQ(1,1,"May only be used with single processor Draw");

  if (N <= 0) {
    SETERRQ2(1,1,"n %d and m %d must be positive",m,n);
  }

  /* create scale window */
  ierr = DrawGetPopup(win,&popup); CHKERRQ(ierr);
  ierr = DrawCheckResizedWindow(win); CHKERRQ(ierr);

  max = min = PetscReal(v[0]);
  for ( i=0; i<N; i++ ) {
    if (max < PetscReal(v[i])) max = PetscReal(v[i]);
    if (min > PetscReal(v[i])) min = PetscReal(v[i]);
  }
  if (max - min < 1.e-7) {min -= 5.e-8; max += 5.e-8;}

  /* Draw the scale window */
  ierr = DrawScalePopup(popup,min,max); CHKERRQ(ierr);

  ierr = OptionsHasName(PETSC_NULL,"-draw_contour_grid",&showgrid);CHKERRQ(ierr);

  /* fill up x and y coordinates */
  if (!xi) {
    xin = 0; 
    x = (double *) PetscMalloc( m*sizeof(double) ); CHKPTRQ(x);
    h = 1.0/(m-1);
    x[0] = 0.0;
    for ( i=1; i<m; i++ ) x[i] = x[i-1] + h;
  } else {
    x = (double *) xi;
  }
  if (!yi) {
    yin = 0; 
    y = (double *) PetscMalloc( n*sizeof(double) ); CHKPTRQ(y);
    h = 1.0/(n-1);
    y[0] = 0.0;
    for ( i=1; i<n; i++ ) y[i] = y[i-1] + h;
  } else {
    y = (double *)yi;
  }
  ierr = DrawTensorContourPatch(win,m,n,x,y,max,min,v);CHKERRQ(ierr);
  if (showgrid) {
    for ( i=0; i<m; i++ ) {
      ierr = DrawLine(win,x[i],y[0],x[i],y[n-1],DRAW_BLACK);CHKERRQ(ierr);
    }
    for ( i=0; i<n; i++ ) {
      ierr = DrawLine(win,x[0],y[i],x[m-1],y[i],DRAW_BLACK);CHKERRQ(ierr);
    }
  }
  ierr = DrawFlush(win);CHKERRQ(ierr);
    
  if (!xin) PetscFree(x); 
  if (!yin) PetscFree(y);

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "DrawTensorContourPatch" 
/*@
   DrawTensorContourPatch - Draws a rectangular patch of a contour plot 
   for a two-dimensional array.

   Not Collective

   Input Parameters:
+  win - the window to draw in
.  m,n - the number of local mesh points in the x and y direction
.  x,y - the locations of the local mesh points
.  max,min - the maximum and minimum value in the entire contour
-  v - the data

   Options Database Keys:
.  -draw_x_shared_colormap - Activates private colormap

   Note: 
   This is a lower level support routine, usually the user will call
   DrawTensorContour(). 

.keywords: Draw, tensor, contour, vector

.seealso: DrawTensorContour()

@*/
int DrawTensorContourPatch(Draw win,int m,int n,double *x,double *y,double max,
                           double min, Scalar *v)
{
  int           c1, c2, c3, c4, i, j,ierr;
  double        x1, x2, x3, x4, y_1, y2, y3, y4,scale;

  PetscFunctionBegin;
  scale = (245.0 - DRAW_BASIC_COLORS)/(max - min);

  /* Draw the contour plot patch */
  for ( j=0; j<n-1; j++ ) {
    for ( i=0; i<m-1; i++ ) {
#if !defined(USE_PETSC_COMPLEX)
      x1 = x[i];  y_1 = y[j];  c1 = (int) (DRAW_BASIC_COLORS + scale*(v[i+j*m] - min));
      x2 = x[i+1];y2 = y_1;    c2 = (int) (DRAW_BASIC_COLORS + scale*(v[i+j*m+1]-min));
      x3 = x2;    y3 = y[j+1];c3 = (int) (DRAW_BASIC_COLORS + scale*(v[i+j*m+1+m]-min));
      x4 = x1;    y4 = y3;    c4 = (int) (DRAW_BASIC_COLORS + scale*(v[i+j*m+m]-min));
#else
      x1 = x[i];  y_1 = y[j];  c1 = (int) (DRAW_BASIC_COLORS + scale*PetscReal(v[i+j*m]-min));
      x2 = x[i+1];y2 = y_1;    c2 = (int) (DRAW_BASIC_COLORS + scale*PetscReal(v[i+j*m+1]-min));
      x3 = x2;    y3 = y[j+1];c3 = (int) (DRAW_BASIC_COLORS + scale*PetscReal(v[i+j*m+1+m]-min));
      x4 = x1;    y4 = y3;    c4 = (int) (DRAW_BASIC_COLORS + scale*PetscReal(v[i+j*m+m]-min));
#endif
      ierr = DrawTriangle(win,x1,y_1,x2,y2,x3,y3,c1,c2,c3); CHKERRQ(ierr);
      ierr = DrawTriangle(win,x1,y_1,x3,y3,x4,y4,c1,c3,c4); CHKERRQ(ierr);
    }
  }

  PetscFunctionReturn(0);
}
