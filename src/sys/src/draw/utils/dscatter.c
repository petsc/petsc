#ifndef lint
static char vcid[] = "$Id: dscatter.c,v 1.1 1996/09/04 21:35:17 bsmith Exp bsmith $";
#endif
/*
       Contains the data structure for drawing scatter plots
    graphs in a window with an axis. This is intended for scatter
    plots that change dynamically.
*/

#include "petsc.h"
#include "draw.h"         /*I "draw.h" I*/

struct _DrawSP {
  PETSCHEADER 
  int         len,loc;
  Draw        win;
  DrawAxis    axis;
  double      xmin, xmax, ymin, ymax, *x, *y;
  int         nopts, dim;
};

#define CHUNCKSIZE 100

/*@C
    DrawSPCreate - Creates a scatter plot data structure.

    Input Parameters:
.   win - the window where the graph will be made.
.   dim - the number of sets of points which will be drawn

    Output Parameters:
.   outctx - the scatter plot context

.keywords:  draw, scatter plot, graph, create

.seealso:  DrawSPDestroy()
@*/
int DrawSPCreate(Draw win,int dim,DrawSP *outctx)
{
  int         ierr;
  PetscObject vobj = (PetscObject) win;
  DrawSP      sp;

  if (vobj->cookie == DRAW_COOKIE && vobj->type == DRAW_NULLWINDOW) {
    ierr = DrawOpenNull(vobj->comm,(Draw*)outctx); CHKERRQ(ierr);
    (*outctx)->win = win;
    return 0;
  }
  PetscHeaderCreate(sp,_DrawSP,DRAWSP_COOKIE,0,vobj->comm);
  sp->view    = 0;
  sp->destroy = 0;
  sp->nopts   = 0;
  sp->win     = win;
  sp->dim     = dim;
  sp->xmin    = 1.e20;
  sp->ymin    = 1.e20;
  sp->xmax    = -1.e20;
  sp->ymax    = -1.e20;
  sp->x       = (double *)PetscMalloc(2*dim*CHUNCKSIZE*sizeof(double));CHKPTRQ(sp->x);
  sp->y       = sp->x + dim*CHUNCKSIZE;
  sp->len     = dim*CHUNCKSIZE;
  sp->loc     = 0;
  ierr = DrawAxisCreate(win,&sp->axis); CHKERRQ(ierr);
  PLogObjectParent(sp,sp->axis);
  *outctx = sp;
  return 0;
}

/*@
   DrawSPSetDimension - Change the number of sets of points  that are to be drawn.

   Input Parameter:
.  sp - the line graph context.
.  dim - the number of curves.

.keywords:  draw, line, graph, reset
@*/
int DrawSPSetDimension(DrawSP sp,int dim)
{
  if (sp && sp->cookie == DRAW_COOKIE && sp->type == DRAW_NULLWINDOW) {return 0;}
  PetscValidHeaderSpecific(sp,DRAWSP_COOKIE);
  if (sp->dim == dim) return 0;

  PetscFree(sp->x);
  sp->dim     = dim;
  sp->x       = (double *)PetscMalloc(2*dim*CHUNCKSIZE*sizeof(double)); CHKPTRQ(sp->x);
  sp->y       = sp->x + dim*CHUNCKSIZE;
  sp->len     = dim*CHUNCKSIZE;
  return 0;
}

/*@
   DrawSPReset - Clears line graph to allow for reuse with new data.

   Input Parameter:
.  sp - the line graph context.

.keywords:  draw, line, graph, reset
@*/
int DrawSPReset(DrawSP sp)
{
  if (sp && sp->cookie == DRAW_COOKIE && sp->type == DRAW_NULLWINDOW) {return 0;}
  PetscValidHeaderSpecific(sp,DRAWSP_COOKIE);
  sp->xmin  = 1.e20;
  sp->ymin  = 1.e20;
  sp->xmax  = -1.e20;
  sp->ymax  = -1.e20;
  sp->loc   = 0;
  sp->nopts = 0;
  return 0;
}

/*@C
   DrawSPDestroy - Frees all space taken up by scatter plot data structure.

   Input Parameter:
.  sp - the line graph context

.keywords:  draw, line, graph, destroy

.seealso:  DrawSPCreate()
@*/
int DrawSPDestroy(DrawSP sp)
{
  if (sp && sp->cookie == DRAW_COOKIE && sp->type == DRAW_NULLWINDOW) {
    return PetscObjectDestroy((PetscObject) sp);
  }
  PetscValidHeaderSpecific(sp,DRAWSP_COOKIE);
  DrawAxisDestroy(sp->axis);
  PetscFree(sp->x);
  PLogObjectDestroy(sp);
  PetscHeaderDestroy(sp);
  return 0;
}

/*@
   DrawSPAddPoint - Adds another point to each of the scatter plots.

   Input Parameters:
.  sp - the scatter plot data structure
.  x, y - the points to two vectors containing the new x and y 
          point for each curve.

.keywords:  draw, line, graph, add, point

.seealso: DrawSPAddPoints()
@*/
int DrawSPAddPoint(DrawSP sp,double *x,double *y)
{
  int i;
  if (sp && sp->cookie == DRAW_COOKIE && sp->type == DRAW_NULLWINDOW) {return 0;}

  PetscValidHeaderSpecific(sp,DRAWSP_COOKIE);
  if (sp->loc+sp->dim >= sp->len) { /* allocate more space */
    double *tmpx,*tmpy;
    tmpx = (double *) PetscMalloc((2*sp->len+2*sp->dim*CHUNCKSIZE)*sizeof(double));CHKPTRQ(tmpx);
    tmpy = tmpx + sp->len + sp->dim*CHUNCKSIZE;
    PetscMemcpy(tmpx,sp->x,sp->len*sizeof(double));
    PetscMemcpy(tmpy,sp->y,sp->len*sizeof(double));
    PetscFree(sp->x);
    sp->x = tmpx; sp->y = tmpy;
    sp->len += sp->dim*CHUNCKSIZE;
  }
  for (i=0; i<sp->dim; i++) {
    if (x[i] > sp->xmax) sp->xmax = x[i]; 
    if (x[i] < sp->xmin) sp->xmin = x[i];
    if (y[i] > sp->ymax) sp->ymax = y[i]; 
    if (y[i] < sp->ymin) sp->ymin = y[i];

    sp->x[sp->loc]   = x[i];
    sp->y[sp->loc++] = y[i];
  }
  sp->nopts++;
  return 0;
}


/*@C
   DrawSPAddPoints - Adds several points to each of the scatter plots.


   Input Parameters:
.  sp - the LineGraph data structure
.  xx,yy - points to two arrays of pointers that point to arrays 
           containing the new x and y points for each curve.
.  n - number of points being added

.keywords:  draw, line, graph, add, points

.seealso: DrawSPAddPoint()
@*/
int DrawSPAddPoints(DrawSP sp,int n,double **xx,double **yy)
{
  int    i, j, k;
  double *x,*y;

  if (sp && sp->cookie == DRAW_COOKIE && sp->type == DRAW_NULLWINDOW) {return 0;}
  PetscValidHeaderSpecific(sp,DRAWSP_COOKIE);
  if (sp->loc+n*sp->dim >= sp->len) { /* allocate more space */
    double *tmpx,*tmpy;
    int    chunk = CHUNCKSIZE;
    if (n > chunk) chunk = n;
    tmpx = (double *) PetscMalloc((2*sp->len+2*sp->dim*chunk)*sizeof(double));
    CHKPTRQ(tmpx);
    tmpy = tmpx + sp->len + sp->dim*chunk;
    PetscMemcpy(tmpx,sp->x,sp->len*sizeof(double));
    PetscMemcpy(tmpy,sp->y,sp->len*sizeof(double));
    PetscFree(sp->x);
    sp->x = tmpx; sp->y = tmpy;
    sp->len += sp->dim*CHUNCKSIZE;
  }
  for (j=0; j<sp->dim; j++) {
    x = xx[j]; y = yy[j];
    k = sp->loc + j;
    for ( i=0; i<n; i++ ) {
      if (x[i] > sp->xmax) sp->xmax = x[i]; 
      if (x[i] < sp->xmin) sp->xmin = x[i];
      if (y[i] > sp->ymax) sp->ymax = y[i]; 
      if (y[i] < sp->ymin) sp->ymin = y[i];

      sp->x[k]   = x[i];
      sp->y[k] = y[i];
      k += sp->dim;
    }
  }
  sp->loc   += n*sp->dim;
  sp->nopts += n;
  return 0;
}

/*@
   DrawSPDraw - Redraws a scatter plot.

   Input Parameter:
.  sp - the line graph context

.keywords:  draw, line, graph
@*/
int DrawSPDraw(DrawSP sp)
{
  double   xmin=sp->xmin, xmax=sp->xmax, ymin=sp->ymin, ymax=sp->ymax;
  int      i, j, dim = sp->dim,nopts = sp->nopts;
  Draw     win = sp->win;
  if (sp && sp->cookie == DRAW_COOKIE && sp->type == DRAW_NULLWINDOW) {return 0;}
  PetscValidHeaderSpecific(sp,DRAWSP_COOKIE);

  if (nopts < 2) return 0;
  if (xmin > xmax || ymin > ymax) return 0;
  DrawClear(win);
  DrawAxisSetLimits(sp->axis, xmin, xmax, ymin, ymax);
  DrawAxisDraw(sp->axis);
  for ( i=0; i<dim; i++ ) {
    for ( j=1; j<nopts; j++ ) {
      DrawText(win,sp->x[j*dim+i],sp->y[j*dim+i],DRAW_RED,"x");
    }
  }
  DrawSyncFlush(sp->win);
  DrawPause(sp->win);
  return 0;
} 
 
/*@
   DrawSPSetLimits - Sets the axis limits for a line graph. If more
   points are added after this call, the limits will be adjusted to
   include those additional points.

   Input Parameters:
.  xsp - the line graph context
.  x_min,x_max,y_min,y_max - the limits

.keywords:  draw, line, graph, set limits
@*/
int DrawSPSetLimits( DrawSP sp,double x_min,double x_max,double y_min,
                                  double y_max) 
{
  if (sp && sp->cookie == DRAW_COOKIE && sp->type == DRAW_NULLWINDOW) {return 0;}
  PetscValidHeaderSpecific(sp,DRAWSP_COOKIE);
  (sp)->xmin = x_min; 
  (sp)->xmax = x_max; 
  (sp)->ymin = y_min; 
  (sp)->ymax = y_max;
  return 0;
}
 
/*@C
   DrawSPGetAxis - Gets the axis context associated with a line graph.
   This is useful if one wants to change some axis property, such as
   labels, color, etc. The axis context should not be destroyed by the
   application code.

   Input Parameter:
.  sp - the line graph context

   Output Parameter:
.  axis - the axis context

.keywords: draw, line, graph, get, axis
@*/
int DrawSPGetAxis(DrawSP sp,DrawAxis *axis)
{
  if (sp && sp->cookie == DRAW_COOKIE && sp->type == DRAW_NULLWINDOW) {
    *axis = 0;
    return 0;
  }
  PetscValidHeaderSpecific(sp,DRAWSP_COOKIE);
  *axis = sp->axis;
  return 0;
}

/*@C
    DrawSPGetDraw - Gets the draw context associated with a line graph.

   Input Parameter:
.  sp - the line graph context

   Output Parameter:
.  win - the draw context

.keywords: draw, line, graph, get, context
@*/
int DrawSPGetDraw(DrawSP sp,Draw *win)
{
  if (!sp || sp->cookie != DRAW_COOKIE || sp->type != DRAW_NULLWINDOW) {
    PetscValidHeaderSpecific(sp,DRAWSP_COOKIE);
  }
  *win = sp->win;
  return 0;
}
