#ifndef lint
static char vcid[] = "$Id: lg.c,v 1.7 1995/03/06 04:28:29 bsmith Exp bsmith $";
#endif
/*
       Contains the data structure for plotting several line
    graphs in a window with an axis. This is intended for line 
    graphs that change dynamically by adding more points onto 
    the end of the X axis.
*/

#include "petsc.h"
#include "ptscimpl.h"
#include "draw.h"


struct _DrawLGCtx {
  PETSCHEADER 
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
  int         ierr;
  DrawLGCtx   lg = (DrawLGCtx) MALLOC(sizeof(struct _DrawLGCtx));CHKPTR(lg);

  lg->cookie  = LG_COOKIE;
  lg->view    = 0;
  lg->destroy = 0;
  lg->nopts   = 0;
  lg->win     = win;
  lg->dim     = dim;
  lg->xmin    = 1.e20;
  lg->ymin    = 1.e20;
  lg->xmax    = -1.e20;
  lg->ymax    = -1.e20;
  lg->x       = (double *)MALLOC(2*dim*CHUNCKSIZE*sizeof(double));
                CHKPTR(lg->x);
  lg->y       = lg->x + dim*CHUNCKSIZE;
  lg->len     = dim*CHUNCKSIZE;
  lg->loc     = 0;
  ierr = DrawAxisCreate(win,&lg->axis); CHKERR(ierr);
  *outctx = lg;
  return 0;
}

/*@
    DrawLGReset - Clears line graph to allow for reuse with new data.

@*/
int DrawLGReset(DrawLGCtx lg)
{
  VALIDHEADER(lg,LG_COOKIE);
  lg->xmin  = 1.e20;
  lg->ymin  = 1.e20;
  lg->xmax  = -1.e20;
  lg->ymax  = -1.e20;
  lg->loc   = 0;
  lg->nopts = 0;
  return 0;
}

/*@
    DrawLGDestroy - Frees all space taken up by LineGraph 
                         data structure.
@*/
int DrawLGDestroy(DrawLGCtx lg)
{
  VALIDHEADER(lg,LG_COOKIE);
  DrawAxisDestroy(lg->axis);
  FREE(lg->x);
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
  int i;
  VALIDHEADER(lg,LG_COOKIE);
  if (lg->loc+lg->dim >= lg->len) { /* allocate more space */
    double *tmpx,*tmpy;
    tmpx = (double *) MALLOC((2*lg->len+2*lg->dim*CHUNCKSIZE)*sizeof(double));
    CHKPTR(tmpx);
    tmpy = tmpx + lg->len + lg->dim*CHUNCKSIZE;
    MEMCPY(tmpx,lg->x,lg->len*sizeof(double));
    MEMCPY(tmpy,lg->y,lg->len*sizeof(double));
    FREE(lg->x);
    lg->x = tmpx; lg->y = tmpy;
    lg->len += lg->dim*CHUNCKSIZE;
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
    DrawLGAddPoints - Adds several points to each of the 
                      line graphs. The new point must have a
                      X coordinate larger than the old points.

   Input Parameters:
.   lg - the LineGraph data structure
.   x,y -  points to two arrays of pointers that point to arrays 
.          containing the new x and y points for each curve.
.   n - number of points being added

@*/
int DrawLGAddPoints(DrawLGCtx lg,int n,double **xx,double **yy)
{
  int    i, j, k;
  double *x,*y;
  VALIDHEADER(lg,LG_COOKIE);
  if (lg->loc+n*lg->dim >= lg->len) { /* allocate more space */
    double *tmpx,*tmpy;
    int    chunk = CHUNCKSIZE;
    if (n > chunk) chunk = n;
    tmpx = (double *) MALLOC((2*lg->len+2*lg->dim*chunk)*sizeof(double));
    CHKPTR(tmpx);
    tmpy = tmpx + lg->len + lg->dim*chunk;
    MEMCPY(tmpx,lg->x,lg->len*sizeof(double));
    MEMCPY(tmpy,lg->y,lg->len*sizeof(double));
    FREE(lg->x);
    lg->x = tmpx; lg->y = tmpy;
    lg->len += lg->dim*CHUNCKSIZE;
  }
  for (j=0; j<lg->dim; j++) {
    x = xx[j]; y = yy[j];
    k = lg->loc + j;
    for ( i=0; i<n; i++ ) {
      if (x[i] > lg->xmax) lg->xmax = x[i]; 
      if (x[i] < lg->xmin) lg->xmin = x[i];
      if (y[i] > lg->ymax) lg->ymax = y[i]; 
      if (y[i] < lg->ymin) lg->ymin = y[i];

      lg->x[k]   = x[i];
      lg->y[k] = y[i];
      k += lg->dim;
    }
  }
  lg->loc   += n*lg->dim;
  lg->nopts += n;
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
  VALIDHEADER(lg,LG_COOKIE);

  if (nopts < 2) return 0;
  if (xmin > xmax || ymin > ymax) return 0;
  DrawClear(win);
  DrawAxisSetLimits(lg->axis, xmin, xmax, ymin, ymax);
  DrawAxis(lg->axis);
  for ( i=0; i<dim; i++ ) {
    for ( j=1; j<nopts; j++ ) {
      DrawLine(win,lg->x[(j-1)*dim+i],lg->y[(j-1)*dim+i],
                   lg->x[j*dim+i],lg->y[j*dim+i],DRAW_BLACK,DRAW_BLACK);
    }
  }
  DrawSyncFlush(lg->win);
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
  VALIDHEADER(lg,LG_COOKIE);
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
  VALIDHEADER(lg,LG_COOKIE);
  *axis = lg->axis;
  return 0;
}

/*@
    DrawLGGetDrawCtx - Gets the draw context associated with a line graph.

  Input Parameter:
.  lg - the line graph context

  Output Parameter:
.  win - the draw context

@*/
int DrawLGGetDrawCtx(DrawLGCtx lg,DrawCtx *win)
{
  VALIDHEADER(lg,LG_COOKIE);
  *win = lg->win;
  return 0;
}
