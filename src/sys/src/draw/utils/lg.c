/*$Id: lg.c,v 1.69 2000/04/09 04:34:21 bsmith Exp bsmith $*/
/*
       Contains the data structure for plotting several line
    graphs in a window with an axis. This is intended for line 
    graphs that change dynamically by adding more points onto 
    the end of the X axis.
*/

#include "petsc.h"         /*I "petsc.h" I*/

struct _p_DrawLG {
  PETSCHEADER(int) 
  int         (*destroy)(DrawLG);
  int         (*view)(DrawLG,Viewer);
  int         len,loc;
  Draw        win;
  DrawAxis    axis;
  PetscReal   xmin,xmax,ymin,ymax,*x,*y;
  int         nopts,dim;
  int         use_dots;
};

#define CHUNCKSIZE 100

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"DrawLGCreate" 
/*@C
    DrawLGCreate - Creates a line graph data structure.

    Collective over Draw

    Input Parameters:
+   draw - the window where the graph will be made.
-   dim - the number of line cures which will be drawn

    Output Parameters:
.   outctx - the line graph context

    Level: intermediate

.keywords:  draw, line, graph, create

.seealso:  DrawLGDestroy()
@*/
int DrawLGCreate(Draw draw,int dim,DrawLG *outctx)
{
  int         ierr;
  PetscTruth  isnull;
  PetscObject obj = (PetscObject)draw;
  DrawLG      lg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  PetscValidPointer(outctx);
  ierr = PetscTypeCompare(obj,DRAW_NULL,&isnull);CHKERRQ(ierr);
  if (isnull) {
    ierr = DrawOpenNull(obj->comm,(Draw*)outctx);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  PetscHeaderCreate(lg,_p_DrawLG,int,DRAWLG_COOKIE,0,"DrawLG",obj->comm,DrawLGDestroy,0);
  lg->view    = 0;
  lg->destroy = 0;
  lg->nopts   = 0;
  lg->win     = draw;
  lg->dim     = dim;
  lg->xmin    = 1.e20;
  lg->ymin    = 1.e20;
  lg->xmax    = -1.e20;
  lg->ymax    = -1.e20;
  lg->x       = (PetscReal *)PetscMalloc(2*dim*CHUNCKSIZE*sizeof(PetscReal));CHKPTRQ(lg->x);
  PLogObjectMemory(lg,2*dim*CHUNCKSIZE*sizeof(PetscReal));
  lg->y       = lg->x + dim*CHUNCKSIZE;
  lg->len     = dim*CHUNCKSIZE;
  lg->loc     = 0;
  lg->use_dots= 0;
  ierr = DrawAxisCreate(draw,&lg->axis);CHKERRQ(ierr);
  PLogObjectParent(lg,lg->axis);
  *outctx = lg;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"DrawLGSetDimension" 
/*@
   DrawLGSetDimension - Change the number of lines that are to be drawn.

   Collective over DrawLG

   Input Parameter:
+  lg - the line graph context.
-  dim - the number of curves.

   Level: intermediate

.keywords:  draw, line, graph, reset
@*/
int DrawLGSetDimension(DrawLG lg,int dim)
{
  int ierr;

  PetscFunctionBegin;
  if (lg && lg->cookie == DRAW_COOKIE) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(lg,DRAWLG_COOKIE);
  if (lg->dim == dim) PetscFunctionReturn(0);

  ierr = PetscFree(lg->x);CHKERRQ(ierr);
  lg->dim = dim;
  lg->x       = (PetscReal *)PetscMalloc(2*dim*CHUNCKSIZE*sizeof(PetscReal));CHKPTRQ(lg->x);
  PLogObjectMemory(lg,2*dim*CHUNCKSIZE*sizeof(PetscReal));
  lg->y       = lg->x + dim*CHUNCKSIZE;
  lg->len     = dim*CHUNCKSIZE;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"DrawLGReset" 
/*@
   DrawLGReset - Clears line graph to allow for reuse with new data.

   Collective over DrawLG

   Input Parameter:
.  lg - the line graph context.

   Level: intermediate

.keywords:  draw, line, graph, reset
@*/
int DrawLGReset(DrawLG lg)
{
  PetscFunctionBegin;
  if (lg && lg->cookie == DRAW_COOKIE) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(lg,DRAWLG_COOKIE);
  lg->xmin  = 1.e20;
  lg->ymin  = 1.e20;
  lg->xmax  = -1.e20;
  lg->ymax  = -1.e20;
  lg->loc   = 0;
  lg->nopts = 0;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"DrawLGDestroy" 
/*@C
   DrawLGDestroy - Frees all space taken up by line graph data structure.

   Collective over DrawLG

   Input Parameter:
.  lg - the line graph context

   Level: intermediate

.keywords:  draw, line, graph, destroy

.seealso:  DrawLGCreate()
@*/
int DrawLGDestroy(DrawLG lg)
{
  int ierr;

  PetscFunctionBegin;
  if (!lg || lg->cookie != DRAW_COOKIE) {
    PetscValidHeaderSpecific(lg,DRAWLG_COOKIE);
  }

  if (--lg->refct > 0) PetscFunctionReturn(0);
  if (lg && lg->cookie == DRAW_COOKIE) {
    ierr = PetscObjectDestroy((PetscObject)lg);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = DrawAxisDestroy(lg->axis);CHKERRQ(ierr);
  ierr = PetscFree(lg->x);CHKERRQ(ierr);
  PLogObjectDestroy(lg);
  PetscHeaderDestroy(lg);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"DrawLGAddPoint" 
/*@
   DrawLGAddPoint - Adds another point to each of the line graphs. 
   The new point must have an X coordinate larger than the old points.

   Not Collective, but ignored by all processors except processor 0 in DrawLG

   Input Parameters:
+  lg - the LineGraph data structure
-  x, y - the points to two vectors containing the new x and y 
          point for each curve.

   Level: intermediate

.keywords:  draw, line, graph, add, point

.seealso: DrawLGAddPoints()
@*/
int DrawLGAddPoint(DrawLG lg,PetscReal *x,PetscReal *y)
{
  int i,ierr;

  PetscFunctionBegin;
  if (lg && lg->cookie == DRAW_COOKIE) PetscFunctionReturn(0);

  PetscValidHeaderSpecific(lg,DRAWLG_COOKIE);
  if (lg->loc+lg->dim >= lg->len) { /* allocate more space */
    PetscReal *tmpx,*tmpy;
    tmpx = (PetscReal*)PetscMalloc((2*lg->len+2*lg->dim*CHUNCKSIZE)*sizeof(PetscReal));CHKPTRQ(tmpx);
    PLogObjectMemory(lg,2*lg->dim*CHUNCKSIZE*sizeof(PetscReal));
    tmpy = tmpx + lg->len + lg->dim*CHUNCKSIZE;
    ierr = PetscMemcpy(tmpx,lg->x,lg->len*sizeof(PetscReal));CHKERRQ(ierr);
    ierr = PetscMemcpy(tmpy,lg->y,lg->len*sizeof(PetscReal));CHKERRQ(ierr);
    ierr = PetscFree(lg->x);CHKERRQ(ierr);
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
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"DrawLGIndicateDataPoints" 
/*@
   DrawLGIndicateDataPoints - Causes LG to draw a big dot for each data-point.

   Not Collective, but ignored by all processors except processor 0 in DrawLG

   Input Parameters:
.  lg - the linegraph context

   Level: intermediate

.keywords:  draw, line, graph, indicate, data, points
@*/
int DrawLGIndicateDataPoints(DrawLG lg)
{
  PetscFunctionBegin;
  if (lg && lg->cookie == DRAW_COOKIE) PetscFunctionReturn(0);

  lg->use_dots = 1;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"DrawLGAddPoints" 
/*@C
   DrawLGAddPoints - Adds several points to each of the line graphs.
   The new points must have an X coordinate larger than the old points.

   Not Collective, but ignored by all processors except processor 0 in DrawLG

   Input Parameters:
+  lg - the LineGraph data structure
.  xx,yy - points to two arrays of pointers that point to arrays 
           containing the new x and y points for each curve.
-  n - number of points being added

   Level: intermediate

.keywords:  draw, line, graph, add, points

.seealso: DrawLGAddPoint()
@*/
int DrawLGAddPoints(DrawLG lg,int n,PetscReal **xx,PetscReal **yy)
{
  int    i,j,k,ierr;
  PetscReal *x,*y;

  PetscFunctionBegin;
  if (lg && lg->cookie == DRAW_COOKIE) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(lg,DRAWLG_COOKIE);
  if (lg->loc+n*lg->dim >= lg->len) { /* allocate more space */
    PetscReal *tmpx,*tmpy;
    int    chunk = CHUNCKSIZE;

    if (n > chunk) chunk = n;
    tmpx = (PetscReal*)PetscMalloc((2*lg->len+2*lg->dim*chunk)*sizeof(PetscReal));CHKPTRQ(tmpx);
    PLogObjectMemory(lg,2*lg->dim*chunk*sizeof(PetscReal));
    tmpy = tmpx + lg->len + lg->dim*chunk;
    ierr = PetscMemcpy(tmpx,lg->x,lg->len*sizeof(PetscReal));CHKERRQ(ierr);
    ierr = PetscMemcpy(tmpy,lg->y,lg->len*sizeof(PetscReal));CHKERRQ(ierr);
    ierr = PetscFree(lg->x);CHKERRQ(ierr);
    lg->x    = tmpx; lg->y = tmpy;
    lg->len += lg->dim*chunk;
  }
  for (j=0; j<lg->dim; j++) {
    x = xx[j]; y = yy[j];
    k = lg->loc + j;
    for (i=0; i<n; i++) {
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
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"DrawLGDraw" 
/*@
   DrawLGDraw - Redraws a line graph.

   Not Collective,but ignored by all processors except processor 0 in DrawLG

   Input Parameter:
.  lg - the line graph context

   Level: intermediate

.keywords:  draw, line, graph
@*/
int DrawLGDraw(DrawLG lg)
{
  PetscReal xmin=lg->xmin,xmax=lg->xmax,ymin=lg->ymin,ymax=lg->ymax;
  int       i,j,dim = lg->dim,nopts = lg->nopts,rank,ierr;
  Draw      draw = lg->win;

  PetscFunctionBegin;
  if (lg && lg->cookie == DRAW_COOKIE) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(lg,DRAWLG_COOKIE);

  if (nopts < 2) PetscFunctionReturn(0);
  if (xmin > xmax || ymin > ymax) PetscFunctionReturn(0);
  ierr = DrawClear(draw);CHKERRQ(ierr);
  ierr = DrawAxisSetLimits(lg->axis,xmin,xmax,ymin,ymax);CHKERRQ(ierr);
  ierr = DrawAxisDraw(lg->axis);CHKERRQ(ierr);

  ierr = MPI_Comm_rank(lg->comm,&rank);CHKERRQ(ierr);
  if (rank) PetscFunctionReturn(0);

  for (i=0; i<dim; i++) {
    for (j=1; j<nopts; j++) {
      ierr = DrawLine(draw,lg->x[(j-1)*dim+i],lg->y[(j-1)*dim+i],
                   lg->x[j*dim+i],lg->y[j*dim+i],DRAW_BLACK+i);CHKERRQ(ierr);
      if (lg->use_dots) {
        ierr = DrawString(draw,lg->x[j*dim+i],lg->y[j*dim+i],DRAW_RED,"x");CHKERRQ(ierr);
      }
    }
  }
  ierr = DrawFlush(lg->win);CHKERRQ(ierr);
  ierr = DrawPause(lg->win);CHKERRQ(ierr);
  PetscFunctionReturn(0);
} 
 
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"DrawLGSetLimits" 
/*@
   DrawLGSetLimits - Sets the axis limits for a line graph. If more
   points are added after this call, the limits will be adjusted to
   include those additional points.

   Not Collective, but ignored by all processors except processor 0 in DrawLG

   Input Parameters:
+  xlg - the line graph context
-  x_min,x_max,y_min,y_max - the limits

   Level: intermediate

.keywords:  draw, line, graph, set limits
@*/
int DrawLGSetLimits(DrawLG lg,PetscReal x_min,PetscReal x_max,PetscReal y_min,PetscReal y_max) 
{
  PetscFunctionBegin;
  if (lg && lg->cookie == DRAW_COOKIE) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(lg,DRAWLG_COOKIE);
  (lg)->xmin = x_min; 
  (lg)->xmax = x_max; 
  (lg)->ymin = y_min; 
  (lg)->ymax = y_max;
  PetscFunctionReturn(0);
}
 
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"DrawLGGetAxis" 
/*@C
   DrawLGGetAxis - Gets the axis context associated with a line graph.
   This is useful if one wants to change some axis property, such as
   labels, color, etc. The axis context should not be destroyed by the
   application code.

   Not Collective, if DrawLG is parallel then DrawAxis is parallel

   Input Parameter:
.  lg - the line graph context

   Output Parameter:
.  axis - the axis context

   Level: advanced

.keywords: draw, line, graph, get, axis
@*/
int DrawLGGetAxis(DrawLG lg,DrawAxis *axis)
{
  PetscFunctionBegin;
  if (lg && lg->cookie == DRAW_COOKIE) {
    *axis = 0;
    PetscFunctionReturn(0);
  }
  PetscValidHeaderSpecific(lg,DRAWLG_COOKIE);
  *axis = lg->axis;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"DrawLGGetDraw" 
/*@C
   DrawLGGetDraw - Gets the draw context associated with a line graph.

   Not Collective, if DrawLG is parallel then Draw is parallel

   Input Parameter:
.  lg - the line graph context

   Output Parameter:
.  draw - the draw context

   Level: intermediate

.keywords: draw, line, graph, get, context
@*/
int DrawLGGetDraw(DrawLG lg,Draw *draw)
{
  PetscFunctionBegin;
  PetscValidHeader(lg);
  if (lg->cookie == DRAW_COOKIE) {
    *draw = (Draw)lg;
  } else {
    PetscValidHeaderSpecific(lg,DRAWLG_COOKIE);
    *draw = lg->win;
  }
  PetscFunctionReturn(0);
}




