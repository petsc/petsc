/*
       Contains the data structure for plotting several line
    graphs in a window with an axis. This is intended for line 
    graphs that change dynamically by adding more points onto 
    the end of the X axis.
*/

#include "petsc.h"
#include "draw.h"


struct _DrawLGCtx {
  int         len,loc;
  DrawCtx     win;
  DrawAxisCtx axis;
  double      xmin, xmax, ymin, ymax, *x, *y;
  int         nopts, dim;
};
#define CHUNCKSIZE 100

/*@
     DrawLGCreate - Creates a Line Graph data structure

  Input Parameters:
.   win - the window where the graph will be made.
.   dim - the number of line cures which will be drawn

  Output Parameters:
.   outctx - the line graph context
@*/
int DrawLGCreate(DrawCtx win,int dim,DrawLGCtx *outctx)
{
  int         i,ierr;
  DrawLGCtx   lg = (DrawLGCtx) MALLOC(sizeof(struct _DrawLGCtx));CHKPTR(lg);

  lg->nopts = 0;
  lg->win   = win;
  lg->dim   = dim;
  lg->xmin  = 1.e20;
  lg->ymin  = 1.e20;
  lg->xmax  = -1.e20;
  lg->ymax  = -1.e20;
  lg->x     = (double *) MALLOC(2*dim*CHUNCKSIZE*sizeof(double));CHKPTR(lg->x);
  lg->y     = lg->x + dim*CHUNCKSIZE;
  lg->len   = dim*CHUNCKSIZE;
  lg->loc   = 0;
  ierr = DrawAxisCreate(win,&lg->axis); CHKERR(ierr);
  *outctx = lg;
  return 0;
}

/*@
    DrawLGReset - Clears line graph to allow for reuse with new data.

@*/
int DrawLGReset(DrawLGCtx lg)
{
  lg->xmin  = 1.e20;
  lg->ymin  = 1.e20;
  lg->xmax  = -1.e20;
  lg->ymax  = -1.e20;
}

/*@
    DrawLGDestroy - Frees all space taken up by LineGraph 
                         data structure.
@*/
int DrawLGDestroy(DrawLGCtx lg)
{
  int i;
  DrawAxisDestroy(lg->axis);
  FREE(lg);
  return 0;
}

/*@
    DrawLGAddPoint - Adds another point to each of the 
                          line graphs. The new point must have a
                          X coordinate larger than the old points.

   Input Parameters:
.   lg - the LineGraph data structure
.   x,y - the points to two vectors containing the new x and y 
           point for each curve.

@*/
int DrawLGAddPoint(DrawLGCtx lg,double *x,double *y)
{
  int i, j;
  if (lg->loc+lg->dim >= lg->len) { /* allocate more space */
    ;
  }
  for (i=0; i<lg->dim; i++) {
    if (x[i] > lg->xmax) lg->xmax = x[i]; 
    if (x[i] < lg->xmin) lg->xmin = x[i];
    if (y[i] > lg->ymax) lg->ymax = y[i]; 
    if (y[i] < lg->ymin) lg->ymin = y[i];

    lg->x[lg->loc]   = x[i];
    lg->y[lg->loc++] = y[i];
  }
  lg->nopts++;
  return 0;
}

/*@
   DrawLG - Redraws a line graph
@*/
int DrawLG(DrawLGCtx lg)
{
  double   xmin=lg->xmin, xmax=lg->xmax, ymin=lg->ymin, ymax=lg->ymax;
  int      i, j, dim = lg->dim,nopts = lg->nopts;
  DrawCtx  win = lg->win;

  if (xmin >= xmax || ymin >= ymax) return 0;
  DrawClear(win);
  DrawAxisSetLimits(lg->axis, xmin, xmax, ymin, ymax);
  DrawAxis(lg->axis);
  for ( i=0; i<dim; i++ ) {
    for ( j=1; j<nopts; j++ ) {
      DrawLine(win,lg->x[(j-1)*dim+i],lg->y[(j-1)*dim+i],
                   lg->x[j*dim+i],lg->y[j*dim+i],DRAW_BLACK,DRAW_BLACK);
    }
  }
  DrawFlush(lg->win);
  return 0;
} 
 

/*@
     DrawLGSetLimits - Sets the axis limits for a line graph. If 
                            more points are added after this call the
                            limits will be adjusted to include those 
                            additional points.

  Input Parameters:
.   xlg - the line graph context
.   x_min,x_max,y_min,y_max - the limits

@*/
int DrawLGSetLimits( DrawLGCtx lg,double x_min,double x_max,double y_min,
                                  double y_max) 
{
  (lg)->xmin = x_min; 
  (lg)->xmax = x_max; 
  (lg)->ymin = y_min; 
  (lg)->ymax = y_max;
  return 0;
}
 
/*@
    DrawLGGetAxisCtx - Gets the axis context associated with a line graph.
           This is useful if one wants to change some axis property, like
           labels, color, etc. The axis context should not be destroyed
           by the application code.

  Input Parameter:
.  lg - the line graph context

  Output Parameter:
.  axis - the axis context

@*/
int DrawLGGetAxisCtx(DrawLGCtx lg,DrawAxisCtx *axis)
{
  *axis = lg->axis;
  return 0;
}
