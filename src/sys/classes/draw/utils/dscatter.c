
/*
       Contains the data structure for drawing scatter plots
    graphs in a window with an axis. This is intended for scatter
    plots that change dynamically.
*/

#include <petscdraw.h>                       /*I "petscdraw.h" I*/
#include <petsc/private/petscimpl.h>         /*I "petscsys.h" I*/

PetscClassId PETSC_DRAWSP_CLASSID = 0;

struct _p_PetscDrawSP {
  PETSCHEADER(int);
  PetscErrorCode (*destroy)(PetscDrawSP);
  PetscErrorCode (*view)(PetscDrawSP,PetscViewer);
  int            len,loc;
  PetscDraw      win;
  PetscDrawAxis  axis;
  PetscReal      xmin,xmax,ymin,ymax,*x,*y;
  int            nopts,dim;
};

#define CHUNCKSIZE 100

/*@C
    PetscDrawSPCreate - Creates a scatter plot data structure.

    Collective on PetscDraw

    Input Parameters:
+   win - the window where the graph will be made.
-   dim - the number of sets of points which will be drawn

    Output Parameters:
.   drawsp - the scatter plot context

   Level: intermediate

   Notes:
    Add points to the plot with PetscDrawSPAddPoint() or PetscDrawSPAddPoints(); the new points are not displayed until PetscDrawSPDraw() is called.

   PetscDrawSPReset() removes all the points that have been added

   The MPI communicator that owns the PetscDraw owns this PetscDrawSP, but the calls to set options and add points are ignored on all processes except the
   zeroth MPI process in the communicator. All MPI processes in the communicator must call PetscDrawSPDraw() to display the updated graph.

.seealso:  PetscDrawLGCreate(), PetscDrawLG, PetscDrawBarCreate(), PetscDrawBar, PetscDrawHGCreate(), PetscDrawHG, PetscDrawSPDestroy(), PetscDraw, PetscDrawSP, PetscDrawSPSetDimension(), PetscDrawSPReset(),
           PetscDrawSPAddPoint(), PetscDrawSPAddPoints(), PetscDrawSPDraw(), PetscDrawSPSave(), PetscDrawSPSetLimits(), PetscDrawSPGetAxis(),PetscDrawAxis, PetscDrawSPGetDraw()
@*/
PetscErrorCode  PetscDrawSPCreate(PetscDraw draw,int dim,PetscDrawSP *drawsp)
{
  PetscDrawSP    sp;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  PetscValidLogicalCollectiveInt(draw,dim,2);
  PetscValidPointer(drawsp,3);

  ierr = PetscHeaderCreate(sp,PETSC_DRAWSP_CLASSID,"DrawSP","Scatter Plot","Draw",PetscObjectComm((PetscObject)draw),PetscDrawSPDestroy,NULL);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)draw,(PetscObject)sp);CHKERRQ(ierr);

  ierr = PetscObjectReference((PetscObject)draw);CHKERRQ(ierr);
  sp->win = draw;

  sp->view    = NULL;
  sp->destroy = NULL;
  sp->nopts   = 0;
  sp->dim     = dim;
  sp->xmin    = 1.e20;
  sp->ymin    = 1.e20;
  sp->xmax    = -1.e20;
  sp->ymax    = -1.e20;

  ierr = PetscMalloc2(dim*CHUNCKSIZE,&sp->x,dim*CHUNCKSIZE,&sp->y);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory((PetscObject)sp,2*dim*CHUNCKSIZE*sizeof(PetscReal));CHKERRQ(ierr);

  sp->len     = dim*CHUNCKSIZE;
  sp->loc     = 0;

  ierr = PetscDrawAxisCreate(draw,&sp->axis);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)sp,(PetscObject)sp->axis);CHKERRQ(ierr);

  *drawsp = sp;
  PetscFunctionReturn(0);
}

/*@
   PetscDrawSPSetDimension - Change the number of sets of points  that are to be drawn.

   Logically Collective on PetscDrawSP

   Input Parameters:
+  sp - the line graph context.
-  dim - the number of curves.

   Level: intermediate

.seealso: PetscDrawSP, PetscDrawSPCreate(), PetscDrawSPAddPoint(), PetscDrawSPAddPoints()

@*/
PetscErrorCode  PetscDrawSPSetDimension(PetscDrawSP sp,int dim)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp,PETSC_DRAWSP_CLASSID,1);
  PetscValidLogicalCollectiveInt(sp,dim,2);
  if (sp->dim == dim) PetscFunctionReturn(0);

  ierr    = PetscFree2(sp->x,sp->y);CHKERRQ(ierr);
  sp->dim = dim;
  ierr    = PetscMalloc2(dim*CHUNCKSIZE,&sp->x,dim*CHUNCKSIZE,&sp->y);CHKERRQ(ierr);
  ierr    = PetscLogObjectMemory((PetscObject)sp,2*dim*CHUNCKSIZE*sizeof(PetscReal));CHKERRQ(ierr);
  sp->len = dim*CHUNCKSIZE;
  PetscFunctionReturn(0);
}

/*@
   PetscDrawSPReset - Clears line graph to allow for reuse with new data.

   Logically Collective on PetscDrawSP

   Input Parameter:
.  sp - the line graph context.

   Level: intermediate

.seealso: PetscDrawSP, PetscDrawSPCreate(), PetscDrawSPAddPoint(), PetscDrawSPAddPoints(), PetscDrawSPDraw()
@*/
PetscErrorCode  PetscDrawSPReset(PetscDrawSP sp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp,PETSC_DRAWSP_CLASSID,1);
  sp->xmin  = 1.e20;
  sp->ymin  = 1.e20;
  sp->xmax  = -1.e20;
  sp->ymax  = -1.e20;
  sp->loc   = 0;
  sp->nopts = 0;
  PetscFunctionReturn(0);
}

/*@C
   PetscDrawSPDestroy - Frees all space taken up by scatter plot data structure.

   Collective on PetscDrawSP

   Input Parameter:
.  sp - the line graph context

   Level: intermediate

.seealso:  PetscDrawSPCreate(), PetscDrawSP, PetscDrawSPReset()

@*/
PetscErrorCode  PetscDrawSPDestroy(PetscDrawSP *sp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*sp) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(*sp,PETSC_DRAWSP_CLASSID,1);
  if (--((PetscObject)(*sp))->refct > 0) {*sp = NULL; PetscFunctionReturn(0);}

  ierr = PetscFree2((*sp)->x,(*sp)->y);CHKERRQ(ierr);
  ierr = PetscDrawAxisDestroy(&(*sp)->axis);CHKERRQ(ierr);
  ierr = PetscDrawDestroy(&(*sp)->win);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(sp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   PetscDrawSPAddPoint - Adds another point to each of the scatter plots.

   Logically Collective on PetscDrawSP

   Input Parameters:
+  sp - the scatter plot data structure
-  x, y - the points to two vectors containing the new x and y
          point for each curve.

   Level: intermediate

   Notes:
    the new points will not be displayed until a call to PetscDrawSPDraw() is made

.seealso: PetscDrawSPAddPoints(), PetscDrawSP, PetscDrawSPCreate(), PetscDrawSPReset(), PetscDrawSPDraw()

@*/
PetscErrorCode  PetscDrawSPAddPoint(PetscDrawSP sp,PetscReal *x,PetscReal *y)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp,PETSC_DRAWSP_CLASSID,1);

  if (sp->loc+sp->dim >= sp->len) { /* allocate more space */
    PetscReal *tmpx,*tmpy;
    ierr     = PetscMalloc2(sp->len+sp->dim*CHUNCKSIZE,&tmpx,sp->len+sp->dim*CHUNCKSIZE,&tmpy);CHKERRQ(ierr);
    ierr     = PetscLogObjectMemory((PetscObject)sp,2*sp->dim*CHUNCKSIZE*sizeof(PetscReal));CHKERRQ(ierr);
    ierr     = PetscArraycpy(tmpx,sp->x,sp->len);CHKERRQ(ierr);
    ierr     = PetscArraycpy(tmpy,sp->y,sp->len);CHKERRQ(ierr);
    ierr     = PetscFree2(sp->x,sp->y);CHKERRQ(ierr);
    sp->x    = tmpx;
    sp->y    = tmpy;
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
  PetscFunctionReturn(0);
}

/*@C
   PetscDrawSPAddPoints - Adds several points to each of the scatter plots.

   Logically Collective on PetscDrawSP

   Input Parameters:
+  sp - the LineGraph data structure
.  xx,yy - points to two arrays of pointers that point to arrays
           containing the new x and y points for each curve.
-  n - number of points being added

   Level: intermediate

   Notes:
    the new points will not be displayed until a call to PetscDrawSPDraw() is made

.seealso: PetscDrawSPAddPoint(), PetscDrawSP, PetscDrawSPCreate(), PetscDrawSPReset(), PetscDrawSPDraw()
@*/
PetscErrorCode  PetscDrawSPAddPoints(PetscDrawSP sp,int n,PetscReal **xx,PetscReal **yy)
{
  PetscErrorCode ierr;
  PetscInt       i,j,k;
  PetscReal      *x,*y;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp,PETSC_DRAWSP_CLASSID,1);

  if (sp->loc+n*sp->dim >= sp->len) { /* allocate more space */
    PetscReal *tmpx,*tmpy;
    PetscInt  chunk = CHUNCKSIZE;
    if (n > chunk) chunk = n;
    ierr = PetscMalloc2(sp->len+sp->dim*chunk,&tmpx,sp->len+sp->dim*chunk,&tmpy);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)sp,2*sp->dim*CHUNCKSIZE*sizeof(PetscReal));CHKERRQ(ierr);
    ierr = PetscArraycpy(tmpx,sp->x,sp->len);CHKERRQ(ierr);
    ierr = PetscArraycpy(tmpy,sp->y,sp->len);CHKERRQ(ierr);
    ierr = PetscFree2(sp->x,sp->y);CHKERRQ(ierr);

    sp->x    = tmpx;
    sp->y    = tmpy;
    sp->len += sp->dim*CHUNCKSIZE;
  }
  for (j=0; j<sp->dim; j++) {
    x = xx[j]; y = yy[j];
    k = sp->loc + j;
    for (i=0; i<n; i++) {
      if (x[i] > sp->xmax) sp->xmax = x[i];
      if (x[i] < sp->xmin) sp->xmin = x[i];
      if (y[i] > sp->ymax) sp->ymax = y[i];
      if (y[i] < sp->ymin) sp->ymin = y[i];

      sp->x[k] = x[i];
      sp->y[k] = y[i];
      k       += sp->dim;
    }
  }
  sp->loc   += n*sp->dim;
  sp->nopts += n;
  PetscFunctionReturn(0);
}

/*@
   PetscDrawSPDraw - Redraws a scatter plot.

   Collective on PetscDrawSP

   Input Parameters:
+  sp - the line graph context
-  clear - clear the window before drawing the new plot

   Level: intermediate

.seealso: PetscDrawLGDraw(), PetscDrawLGSPDraw(), PetscDrawSP, PetscDrawSPCreate(), PetscDrawSPReset(), PetscDrawSPAddPoint(), PetscDrawSPAddPoints()

@*/
PetscErrorCode  PetscDrawSPDraw(PetscDrawSP sp, PetscBool clear)
{
  PetscReal      xmin,xmax,ymin,ymax;
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  PetscBool      isnull;
  PetscDraw      draw;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp,PETSC_DRAWSP_CLASSID,1);
  ierr = PetscDrawIsNull(sp->win,&isnull);CHKERRQ(ierr);
  if (isnull) PetscFunctionReturn(0);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)sp),&rank);CHKERRMPI(ierr);

  if (sp->xmin > sp->xmax || sp->ymin > sp->ymax) PetscFunctionReturn(0);
  if (sp->nopts < 1) PetscFunctionReturn(0);

  draw = sp->win;
  if (clear) {
    ierr = PetscDrawCheckResizedWindow(draw);CHKERRQ(ierr);
    ierr = PetscDrawClear(draw);CHKERRQ(ierr);
  }

  xmin = sp->xmin; xmax = sp->xmax; ymin = sp->ymin; ymax = sp->ymax;
  ierr = PetscDrawAxisSetLimits(sp->axis,xmin,xmax,ymin,ymax);CHKERRQ(ierr);
  ierr = PetscDrawAxisDraw(sp->axis);CHKERRQ(ierr);

  ierr = PetscDrawCollectiveBegin(draw);CHKERRQ(ierr);
  if (!rank) {
    int i,j,dim=sp->dim,nopts=sp->nopts;
    for (i=0; i<dim; i++) {
      for (j=0; j<nopts; j++) {
        ierr = PetscDrawPoint(draw,sp->x[j*dim+i],sp->y[j*dim+i],PETSC_DRAW_RED);CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscDrawCollectiveEnd(draw);CHKERRQ(ierr);

  ierr = PetscDrawFlush(draw);CHKERRQ(ierr);
  ierr = PetscDrawPause(draw);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   PetscDrawSPSave - Saves a drawn image

   Collective on PetscDrawSP

   Input Parameter:
.  sp - the scatter plot context

   Level: intermediate

.seealso:  PetscDrawSPCreate(), PetscDrawSPGetDraw(), PetscDrawSetSave(), PetscDrawSave()
@*/
PetscErrorCode  PetscDrawSPSave(PetscDrawSP sp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp,PETSC_DRAWSP_CLASSID,1);
  ierr = PetscDrawSave(sp->win);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   PetscDrawSPSetLimits - Sets the axis limits for a scatter plot If more
   points are added after this call, the limits will be adjusted to
   include those additional points.

   Logically Collective on PetscDrawSP

   Input Parameters:
+  xsp - the line graph context
-  x_min,x_max,y_min,y_max - the limits

   Level: intermediate

.seealso: PetscDrawSP, PetscDrawSPCreate(), PetscDrawSPDraw(), PetscDrawSPAddPoint(), PetscDrawSPAddPoints(), PetscDrawSPGetAxis()
@*/
PetscErrorCode  PetscDrawSPSetLimits(PetscDrawSP sp,PetscReal x_min,PetscReal x_max,PetscReal y_min,PetscReal y_max)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp,PETSC_DRAWSP_CLASSID,1);
  sp->xmin = x_min;
  sp->xmax = x_max;
  sp->ymin = y_min;
  sp->ymax = y_max;
  PetscFunctionReturn(0);
}

/*@C
   PetscDrawSPGetAxis - Gets the axis context associated with a line graph.
   This is useful if one wants to change some axis property, such as
   labels, color, etc. The axis context should not be destroyed by the
   application code.

   Not Collective, if PetscDrawSP is parallel then PetscDrawAxis is parallel

   Input Parameter:
.  sp - the line graph context

   Output Parameter:
.  axis - the axis context

   Level: intermediate

.seealso: PetscDrawSP, PetscDrawSPCreate(), PetscDrawSPDraw(), PetscDrawSPAddPoint(), PetscDrawSPAddPoints(), PetscDrawAxis, PetscDrawAxisCreate()

@*/
PetscErrorCode  PetscDrawSPGetAxis(PetscDrawSP sp,PetscDrawAxis *axis)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp,PETSC_DRAWSP_CLASSID,1);
  PetscValidPointer(axis,2);
  *axis = sp->axis;
  PetscFunctionReturn(0);
}

/*@C
   PetscDrawSPGetDraw - Gets the draw context associated with a line graph.

   Not Collective, PetscDraw is parallel if PetscDrawSP is parallel

   Input Parameter:
.  sp - the line graph context

   Output Parameter:
.  draw - the draw context

   Level: intermediate

.seealso: PetscDrawSP, PetscDrawSPCreate(), PetscDrawSPDraw(), PetscDraw
@*/
PetscErrorCode  PetscDrawSPGetDraw(PetscDrawSP sp,PetscDraw *draw)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp,PETSC_DRAWSP_CLASSID,1);
  PetscValidPointer(draw,2);
  *draw = sp->win;
  PetscFunctionReturn(0);
}
