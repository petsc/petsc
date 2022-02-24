/*
       Contains the data structure for drawing scatter plots
    graphs in a window with an axis. This is intended for scatter
    plots that change dynamically.
*/

#include <petscdraw.h>              /*I "petscdraw.h" I*/
#include <petsc/private/drawimpl.h> /*I "petscsys.h" I*/

PetscClassId PETSC_DRAWSP_CLASSID = 0;

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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  PetscValidLogicalCollectiveInt(draw,dim,2);
  PetscValidPointer(drawsp,3);

  CHKERRQ(PetscHeaderCreate(sp,PETSC_DRAWSP_CLASSID,"DrawSP","Scatter Plot","Draw",PetscObjectComm((PetscObject)draw),PetscDrawSPDestroy,NULL));
  CHKERRQ(PetscLogObjectParent((PetscObject)draw,(PetscObject)sp));

  CHKERRQ(PetscObjectReference((PetscObject)draw));
  sp->win = draw;

  sp->view      = NULL;
  sp->destroy   = NULL;
  sp->nopts     = 0;
  sp->dim       = dim;
  sp->xmin      = 1.e20;
  sp->ymin      = 1.e20;
  sp->xmax      = -1.e20;
  sp->ymax      = -1.e20;
  sp->zmax      = 1.;
  sp->zmin      = 1.e20;
  sp->colorized = PETSC_FALSE;

  CHKERRQ(PetscMalloc3(dim*PETSC_DRAW_SP_CHUNK_SIZE,&sp->x,dim*PETSC_DRAW_SP_CHUNK_SIZE,&sp->y,dim*PETSC_DRAW_SP_CHUNK_SIZE,&sp->z));
  CHKERRQ(PetscLogObjectMemory((PetscObject)sp,2*dim*PETSC_DRAW_SP_CHUNK_SIZE*sizeof(PetscReal)));

  sp->len     = dim*PETSC_DRAW_SP_CHUNK_SIZE;
  sp->loc     = 0;

  CHKERRQ(PetscDrawAxisCreate(draw,&sp->axis));
  CHKERRQ(PetscLogObjectParent((PetscObject)sp,(PetscObject)sp->axis));

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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp,PETSC_DRAWSP_CLASSID,1);
  PetscValidLogicalCollectiveInt(sp,dim,2);
  if (sp->dim == dim) PetscFunctionReturn(0);

  CHKERRQ(PetscFree2(sp->x,sp->y));
  sp->dim = dim;
  CHKERRQ(PetscMalloc2(dim*PETSC_DRAW_SP_CHUNK_SIZE,&sp->x,dim*PETSC_DRAW_SP_CHUNK_SIZE,&sp->y));
  CHKERRQ(PetscLogObjectMemory((PetscObject)sp,2*dim*PETSC_DRAW_SP_CHUNK_SIZE*sizeof(PetscReal)));
  sp->len = dim*PETSC_DRAW_SP_CHUNK_SIZE;
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
  sp->zmin  = 1.e20;
  sp->xmax  = -1.e20;
  sp->ymax  = -1.e20;
  sp->zmax  = -1.e20;
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
  PetscFunctionBegin;
  if (!*sp) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(*sp,PETSC_DRAWSP_CLASSID,1);
  if (--((PetscObject)(*sp))->refct > 0) {*sp = NULL; PetscFunctionReturn(0);}

  CHKERRQ(PetscFree3((*sp)->x,(*sp)->y,(*sp)->z));
  CHKERRQ(PetscDrawAxisDestroy(&(*sp)->axis));
  CHKERRQ(PetscDrawDestroy(&(*sp)->win));
  CHKERRQ(PetscHeaderDestroy(sp));
  PetscFunctionReturn(0);
}

/*@
   PetscDrawSPAddPoint - Adds another point to each of the scatter plots.

   Logically Collective on PetscDrawSP

   Input Parameters:
+  sp - the scatter plot data structure
-  x, y - two arrays of length dim containing the new x and y coordinate values for each of the curves. Here  dim is the number of curves passed to PetscDrawSPCreate()

   Level: intermediate

   Notes:
    the new points will not be displayed until a call to PetscDrawSPDraw() is made

.seealso: PetscDrawSPAddPoints(), PetscDrawSP, PetscDrawSPCreate(), PetscDrawSPReset(), PetscDrawSPDraw(), PetscDrawSPAddPointColorized()

@*/
PetscErrorCode  PetscDrawSPAddPoint(PetscDrawSP sp,PetscReal *x,PetscReal *y)
{
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp,PETSC_DRAWSP_CLASSID,1);

  if (sp->loc+sp->dim >= sp->len) { /* allocate more space */
    PetscReal *tmpx,*tmpy;
    CHKERRQ(PetscMalloc2(sp->len+sp->dim*PETSC_DRAW_SP_CHUNK_SIZE,&tmpx,sp->len+sp->dim*PETSC_DRAW_SP_CHUNK_SIZE,&tmpy));
    CHKERRQ(PetscLogObjectMemory((PetscObject)sp,2*sp->dim*PETSC_DRAW_SP_CHUNK_SIZE*sizeof(PetscReal)));
    CHKERRQ(PetscArraycpy(tmpx,sp->x,sp->len));
    CHKERRQ(PetscArraycpy(tmpy,sp->y,sp->len));
    CHKERRQ(PetscFree2(sp->x,sp->y));
    sp->x    = tmpx;
    sp->y    = tmpy;
    sp->len += sp->dim*PETSC_DRAW_SP_CHUNK_SIZE;
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

.seealso: PetscDrawSPAddPoint(), PetscDrawSP, PetscDrawSPCreate(), PetscDrawSPReset(), PetscDrawSPDraw(), PetscDrawSPAddPointColorized()
@*/
PetscErrorCode  PetscDrawSPAddPoints(PetscDrawSP sp,int n,PetscReal **xx,PetscReal **yy)
{
  PetscInt       i,j,k;
  PetscReal      *x,*y;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp,PETSC_DRAWSP_CLASSID,1);

  if (sp->loc+n*sp->dim >= sp->len) { /* allocate more space */
    PetscReal *tmpx,*tmpy;
    PetscInt  chunk = PETSC_DRAW_SP_CHUNK_SIZE;
    if (n > chunk) chunk = n;
    CHKERRQ(PetscMalloc2(sp->len+sp->dim*chunk,&tmpx,sp->len+sp->dim*chunk,&tmpy));
    CHKERRQ(PetscLogObjectMemory((PetscObject)sp,2*sp->dim*PETSC_DRAW_SP_CHUNK_SIZE*sizeof(PetscReal)));
    CHKERRQ(PetscArraycpy(tmpx,sp->x,sp->len));
    CHKERRQ(PetscArraycpy(tmpy,sp->y,sp->len));
    CHKERRQ(PetscFree2(sp->x,sp->y));

    sp->x    = tmpx;
    sp->y    = tmpy;
    sp->len += sp->dim*PETSC_DRAW_SP_CHUNK_SIZE;
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
   PetscDrawSPAddPointColorized - Adds another point to each of the scatter plots as well as a numeric value to be used to colorize the scatter point.

   Logically Collective on PetscDrawSP

   Input Parameters:
+  sp - the scatter plot data structure
. x, y - two arrays of length dim containing the new x and y coordinate values for each of the curves. Here  dim is the number of curves passed to PetscDrawSPCreate()
- z - array of length dim containing the numeric values that will be mapped to [0,255] and used for scatter point colors.

   Level: intermediate

   Notes:
    the new points will not be displayed until a call to PetscDrawSPDraw() is made

.seealso: PetscDrawSPAddPoints(), PetscDrawSP, PetscDrawSPCreate(), PetscDrawSPReset(), PetscDrawSPDraw(), PetscDrawSPAddPoint()

@*/
PetscErrorCode  PetscDrawSPAddPointColorized(PetscDrawSP sp,PetscReal *x,PetscReal *y,PetscReal *z)
{
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp,PETSC_DRAWSP_CLASSID,1);
  sp->colorized = PETSC_TRUE;
  if (sp->loc+sp->dim >= sp->len) { /* allocate more space */
    PetscReal *tmpx,*tmpy,*tmpz;
    CHKERRQ(PetscMalloc3(sp->len+sp->dim*PETSC_DRAW_SP_CHUNK_SIZE,&tmpx,sp->len+sp->dim*PETSC_DRAW_SP_CHUNK_SIZE,&tmpy,sp->len+sp->dim*PETSC_DRAW_SP_CHUNK_SIZE,&tmpz));
    CHKERRQ(PetscLogObjectMemory((PetscObject)sp,2*sp->dim*PETSC_DRAW_SP_CHUNK_SIZE*sizeof(PetscReal)));
    CHKERRQ(PetscArraycpy(tmpx,sp->x,sp->len));
    CHKERRQ(PetscArraycpy(tmpy,sp->y,sp->len));
    CHKERRQ(PetscArraycpy(tmpz,sp->z,sp->len));
    CHKERRQ(PetscFree3(sp->x,sp->y,sp->z));
    sp->x    = tmpx;
    sp->y    = tmpy;
    sp->z    = tmpz;
    sp->len += sp->dim*PETSC_DRAW_SP_CHUNK_SIZE;
  }
  for (i=0; i<sp->dim; i++) {
    if (x[i] > sp->xmax) sp->xmax = x[i];
    if (x[i] < sp->xmin) sp->xmin = x[i];
    if (y[i] > sp->ymax) sp->ymax = y[i];
    if (y[i] < sp->ymin) sp->ymin = y[i];
    if (z[i] < sp->zmin) sp->zmin = z[i];
    if (z[i] > sp->zmax) sp->zmax = z[i];
    // if (z[i] > sp->zmax && z[i] < 5.) sp->zmax = z[i];

    sp->x[sp->loc]   = x[i];
    sp->y[sp->loc]   = y[i];
    sp->z[sp->loc++] = z[i];
  }
  sp->nopts++;
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
  PetscMPIInt    rank;
  PetscInt       color;
  PetscBool      isnull;
  PetscDraw      draw;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp,PETSC_DRAWSP_CLASSID,1);
  CHKERRQ(PetscDrawIsNull(sp->win,&isnull));
  if (isnull) PetscFunctionReturn(0);
  CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)sp),&rank));

  if (sp->xmin > sp->xmax || sp->ymin > sp->ymax) PetscFunctionReturn(0);
  if (sp->nopts < 1) PetscFunctionReturn(0);

  draw = sp->win;
  if (clear) {
    CHKERRQ(PetscDrawCheckResizedWindow(draw));
    CHKERRQ(PetscDrawClear(draw));
  }

  xmin = sp->xmin; xmax = sp->xmax; ymin = sp->ymin; ymax = sp->ymax;
  CHKERRQ(PetscDrawAxisSetLimits(sp->axis,xmin,xmax,ymin,ymax));
  CHKERRQ(PetscDrawAxisDraw(sp->axis));

  ierr = PetscDrawCollectiveBegin(draw);CHKERRQ(ierr);
  if (rank == 0) {
    int i,j,dim=sp->dim,nopts=sp->nopts;
    for (i=0; i<dim; i++) {
      for (j=0; j<nopts; j++) {
        if (sp->colorized) {
          color = PetscDrawRealToColor(sp->z[j*dim],sp->zmin,sp->zmax);
          CHKERRQ(PetscDrawPoint(draw,sp->x[j*dim+i],sp->y[j*dim+i],color));
        } else {
          CHKERRQ(PetscDrawPoint(draw,sp->x[j*dim+i],sp->y[j*dim+i],PETSC_DRAW_RED));
        }
      }
    }
  }
  ierr = PetscDrawCollectiveEnd(draw);CHKERRQ(ierr);

  CHKERRQ(PetscDrawFlush(draw));
  CHKERRQ(PetscDrawPause(draw));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp,PETSC_DRAWSP_CLASSID,1);
  CHKERRQ(PetscDrawSave(sp->win));
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
