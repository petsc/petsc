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
+ win - the window where the graph will be made.
- dim - the number of sets of points which will be drawn

  Output Parameters:
. drawsp - the scatter plot context

  Level: intermediate

  Notes:
  Add points to the plot with PetscDrawSPAddPoint() or PetscDrawSPAddPoints(); the new points are not displayed until PetscDrawSPDraw() is called.

  PetscDrawSPReset() removes all the points that have been added

  The MPI communicator that owns the PetscDraw owns this PetscDrawSP, and each processc can add points. All MPI processes in the communicator must call PetscDrawSPDraw() to display the updated graph.

.seealso: `PetscDrawLGCreate()`, `PetscDrawLG`, `PetscDrawBarCreate()`, `PetscDrawBar`, `PetscDrawHGCreate()`, `PetscDrawHG`, `PetscDrawSPDestroy()`, `PetscDraw`, `PetscDrawSP`, `PetscDrawSPSetDimension()`, `PetscDrawSPReset()`,
          `PetscDrawSPAddPoint()`, `PetscDrawSPAddPoints()`, `PetscDrawSPDraw()`, `PetscDrawSPSave()`, `PetscDrawSPSetLimits()`, `PetscDrawSPGetAxis()`, `PetscDrawAxis`, `PetscDrawSPGetDraw()`
@*/
PetscErrorCode PetscDrawSPCreate(PetscDraw draw,int dim,PetscDrawSP *drawsp)
{
  PetscDrawSP sp;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw, PETSC_DRAW_CLASSID, 1);
  PetscValidPointer(drawsp, 3);

  PetscCall(PetscHeaderCreate(sp, PETSC_DRAWSP_CLASSID, "DrawSP", "Scatter Plot", "Draw", PetscObjectComm((PetscObject) draw), PetscDrawSPDestroy, NULL));
  PetscCall(PetscLogObjectParent((PetscObject) draw, (PetscObject) sp));
  PetscCall(PetscObjectReference((PetscObject) draw));
  sp->win       = draw;
  sp->view      = NULL;
  sp->destroy   = NULL;
  sp->nopts     = 0;
  sp->dim       = -1;
  sp->xmin      = 1.e20;
  sp->ymin      = 1.e20;
  sp->zmin      = 1.e20;
  sp->xmax      = -1.e20;
  sp->ymax      = -1.e20;
  sp->zmax      = -1.e20;
  sp->colorized = PETSC_FALSE;
  sp->loc       = 0;

  PetscCall(PetscDrawSPSetDimension(sp, dim));
  PetscCall(PetscDrawAxisCreate(draw, &sp->axis));
  PetscCall(PetscLogObjectParent((PetscObject) sp, (PetscObject) sp->axis));

  *drawsp = sp;
  PetscFunctionReturn(0);
}

/*@
  PetscDrawSPSetDimension - Change the number of sets of points that are to be drawn.

  Not collective

  Input Parameters:
+ sp  - the line graph context.
- dim - the number of curves on this process

  Level: intermediate

.seealso: `PetscDrawSP`, `PetscDrawSPCreate()`, `PetscDrawSPAddPoint()`, `PetscDrawSPAddPoints()`
@*/
PetscErrorCode PetscDrawSPSetDimension(PetscDrawSP sp,int dim)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSC_DRAWSP_CLASSID, 1);
  if (sp->dim == dim) PetscFunctionReturn(0);
  sp->dim = dim;
  PetscCall(PetscFree3(sp->x, sp->y, sp->z));
  PetscCall(PetscMalloc3(dim*PETSC_DRAW_SP_CHUNK_SIZE, &sp->x, dim*PETSC_DRAW_SP_CHUNK_SIZE, &sp->y, dim*PETSC_DRAW_SP_CHUNK_SIZE, &sp->z));
  PetscCall(PetscLogObjectMemory((PetscObject) sp, 3*dim*PETSC_DRAW_SP_CHUNK_SIZE*sizeof(PetscReal)));
  sp->len = dim*PETSC_DRAW_SP_CHUNK_SIZE;
  PetscFunctionReturn(0);
}

/*@
  PetscDrawSPGetDimension - Get the number of sets of points that are to be drawn.

  Not collective

  Input Parameters:
. sp  - the line graph context.

  Output Parameter:
. dim - the number of curves on this process

  Level: intermediate

.seealso: `PetscDrawSP`, `PetscDrawSPCreate()`, `PetscDrawSPAddPoint()`, `PetscDrawSPAddPoints()`
@*/
PetscErrorCode PetscDrawSPGetDimension(PetscDrawSP sp, int *dim)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSC_DRAWSP_CLASSID, 1);
  PetscValidPointer(dim, 2);
  *dim = sp->dim;
  PetscFunctionReturn(0);
}

/*@
  PetscDrawSPReset - Clears line graph to allow for reuse with new data.

  Not collective

  Input Parameter:
. sp - the line graph context.

  Level: intermediate

.seealso: `PetscDrawSP`, `PetscDrawSPCreate()`, `PetscDrawSPAddPoint()`, `PetscDrawSPAddPoints()`, `PetscDrawSPDraw()`
@*/
PetscErrorCode PetscDrawSPReset(PetscDrawSP sp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSC_DRAWSP_CLASSID, 1);
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

/*@
  PetscDrawSPDestroy - Frees all space taken up by scatter plot data structure.

  Collective on PetscDrawSP

  Input Parameter:
. sp - the line graph context

  Level: intermediate

.seealso: `PetscDrawSPCreate()`, `PetscDrawSP`, `PetscDrawSPReset()`
@*/
PetscErrorCode PetscDrawSPDestroy(PetscDrawSP *sp)
{
  PetscFunctionBegin;
  if (!*sp) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(*sp,PETSC_DRAWSP_CLASSID,1);
  if (--((PetscObject)(*sp))->refct > 0) {*sp = NULL; PetscFunctionReturn(0);}

  PetscCall(PetscFree3((*sp)->x, (*sp)->y, (*sp)->z));
  PetscCall(PetscDrawAxisDestroy(&(*sp)->axis));
  PetscCall(PetscDrawDestroy(&(*sp)->win));
  PetscCall(PetscHeaderDestroy(sp));
  PetscFunctionReturn(0);
}

/*@
  PetscDrawSPAddPoint - Adds another point to each of the scatter plots.

  Not collective

  Input Parameters:
+ sp - the scatter plot data structure
- x, y - two arrays of length dim containing the new x and y coordinate values for each of the curves. Here  dim is the number of curves passed to PetscDrawSPCreate()

  Level: intermediate

  Notes:
  The new points will not be displayed until a call to PetscDrawSPDraw() is made

.seealso: `PetscDrawSPAddPoints()`, `PetscDrawSP`, `PetscDrawSPCreate()`, `PetscDrawSPReset()`, `PetscDrawSPDraw()`, `PetscDrawSPAddPointColorized()`
@*/
PetscErrorCode PetscDrawSPAddPoint(PetscDrawSP sp, PetscReal *x, PetscReal *y)
{
  PetscInt i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSC_DRAWSP_CLASSID, 1);

  if (sp->loc+sp->dim >= sp->len) { /* allocate more space */
    PetscReal *tmpx, *tmpy, *tmpz;
    PetscCall(PetscMalloc3(sp->len+sp->dim*PETSC_DRAW_SP_CHUNK_SIZE, &tmpx, sp->len+sp->dim*PETSC_DRAW_SP_CHUNK_SIZE, &tmpy, sp->len+sp->dim*PETSC_DRAW_SP_CHUNK_SIZE, &tmpz));
    PetscCall(PetscLogObjectMemory((PetscObject) sp, 3*sp->dim*PETSC_DRAW_SP_CHUNK_SIZE*sizeof(PetscReal)));
    PetscCall(PetscArraycpy(tmpx, sp->x, sp->len));
    PetscCall(PetscArraycpy(tmpy, sp->y, sp->len));
    PetscCall(PetscArraycpy(tmpz, sp->z, sp->len));
    PetscCall(PetscFree3(sp->x, sp->y, sp->z));
    sp->x = tmpx;
    sp->y = tmpy;
    sp->z = tmpz;
    sp->len += sp->dim*PETSC_DRAW_SP_CHUNK_SIZE;
  }
  for (i = 0; i < sp->dim; ++i) {
    if (x[i] > sp->xmax) sp->xmax = x[i];
    if (x[i] < sp->xmin) sp->xmin = x[i];
    if (y[i] > sp->ymax) sp->ymax = y[i];
    if (y[i] < sp->ymin) sp->ymin = y[i];

    sp->x[sp->loc]   = x[i];
    sp->y[sp->loc++] = y[i];
  }
  ++sp->nopts;
  PetscFunctionReturn(0);
}

/*@C
  PetscDrawSPAddPoints - Adds several points to each of the scatter plots.

  Not collective

  Input Parameters:
+ sp - the LineGraph data structure
. xx,yy - points to two arrays of pointers that point to arrays containing the new x and y points for each curve.
- n - number of points being added

  Level: intermediate

  Notes:
  The new points will not be displayed until a call to PetscDrawSPDraw() is made

.seealso: `PetscDrawSPAddPoint()`, `PetscDrawSP`, `PetscDrawSPCreate()`, `PetscDrawSPReset()`, `PetscDrawSPDraw()`, `PetscDrawSPAddPointColorized()`
@*/
PetscErrorCode PetscDrawSPAddPoints(PetscDrawSP sp, int n, PetscReal **xx, PetscReal **yy)
{
  PetscInt   i, j, k;
  PetscReal *x, *y;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSC_DRAWSP_CLASSID, 1);

  if (sp->loc+n*sp->dim >= sp->len) { /* allocate more space */
    PetscReal *tmpx, *tmpy, *tmpz;
    PetscInt  chunk = PETSC_DRAW_SP_CHUNK_SIZE;
    if (n > chunk) chunk = n;
    PetscCall(PetscMalloc3(sp->len+sp->dim*chunk, &tmpx, sp->len+sp->dim*chunk, &tmpy, sp->len+sp->dim*chunk, &tmpz));
    PetscCall(PetscLogObjectMemory((PetscObject) sp, 3*sp->dim*PETSC_DRAW_SP_CHUNK_SIZE*sizeof(PetscReal)));
    PetscCall(PetscArraycpy(tmpx, sp->x, sp->len));
    PetscCall(PetscArraycpy(tmpy, sp->y, sp->len));
    PetscCall(PetscArraycpy(tmpz, sp->z, sp->len));
    PetscCall(PetscFree3(sp->x, sp->y, sp->z));

    sp->x    = tmpx;
    sp->y    = tmpy;
    sp->z    = tmpz;
    sp->len += sp->dim*PETSC_DRAW_SP_CHUNK_SIZE;
  }
  for (j = 0; j < sp->dim; ++j) {
    x = xx[j]; y = yy[j];
    k = sp->loc + j;
    for (i = 0; i < n; ++i) {
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

  Not collective

  Input Parameters:
+ sp - the scatter plot data structure
. x, y - two arrays of length dim containing the new x and y coordinate values for each of the curves. Here  dim is the number of curves passed to PetscDrawSPCreate()
- z - array of length dim containing the numeric values that will be mapped to [0,255] and used for scatter point colors.

  Level: intermediate

  Notes:
  The new points will not be displayed until a call to PetscDrawSPDraw() is made

.seealso: `PetscDrawSPAddPoints()`, `PetscDrawSP`, `PetscDrawSPCreate()`, `PetscDrawSPReset()`, `PetscDrawSPDraw()`, `PetscDrawSPAddPoint()`
@*/
PetscErrorCode PetscDrawSPAddPointColorized(PetscDrawSP sp, PetscReal *x, PetscReal *y, PetscReal *z)
{
  PetscInt i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSC_DRAWSP_CLASSID, 1);
  sp->colorized = PETSC_TRUE;
  if (sp->loc+sp->dim >= sp->len) { /* allocate more space */
    PetscReal *tmpx, *tmpy, *tmpz;
    PetscCall(PetscMalloc3(sp->len+sp->dim*PETSC_DRAW_SP_CHUNK_SIZE, &tmpx, sp->len+sp->dim*PETSC_DRAW_SP_CHUNK_SIZE, &tmpy, sp->len+sp->dim*PETSC_DRAW_SP_CHUNK_SIZE, &tmpz));
    PetscCall(PetscLogObjectMemory((PetscObject)sp, 3*sp->dim*PETSC_DRAW_SP_CHUNK_SIZE*sizeof(PetscReal)));
    PetscCall(PetscArraycpy(tmpx, sp->x, sp->len));
    PetscCall(PetscArraycpy(tmpy, sp->y, sp->len));
    PetscCall(PetscArraycpy(tmpz, sp->z, sp->len));
    PetscCall(PetscFree3(sp->x, sp->y, sp->z));
    sp->x    = tmpx;
    sp->y    = tmpy;
    sp->z    = tmpz;
    sp->len += sp->dim*PETSC_DRAW_SP_CHUNK_SIZE;
  }
  for (i = 0; i < sp->dim; ++i) {
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
  ++sp->nopts;
  PetscFunctionReturn(0);
}

/*@
  PetscDrawSPDraw - Redraws a scatter plot.

  Collective on PetscDrawSP

  Input Parameters:
+ sp - the line graph context
- clear - clear the window before drawing the new plot

  Level: intermediate

.seealso: `PetscDrawLGDraw()`, `PetscDrawLGSPDraw()`, `PetscDrawSP`, `PetscDrawSPCreate()`, `PetscDrawSPReset()`, `PetscDrawSPAddPoint()`, `PetscDrawSPAddPoints()`
@*/
PetscErrorCode PetscDrawSPDraw(PetscDrawSP sp, PetscBool clear)
{
  PetscDraw      draw;
  PetscBool      isnull;
  PetscMPIInt    rank, size;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSC_DRAWSP_CLASSID, 1);
  draw = sp->win;
  PetscCall(PetscDrawIsNull(draw, &isnull));
  if (isnull) PetscFunctionReturn(0);
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject) sp), &rank));
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject) sp), &size));

  if (clear) {
    PetscCall(PetscDrawCheckResizedWindow(draw));
    PetscCall(PetscDrawClear(draw));
  }
  {
    PetscReal lower[2] = {sp->xmin, sp->ymin}, glower[2];
    PetscReal upper[2] = {sp->xmax, sp->ymax}, gupper[2];
    PetscCall(MPIU_Allreduce(lower, glower, 2, MPIU_REAL, MPIU_MIN, PetscObjectComm((PetscObject) sp)));
    PetscCall(MPIU_Allreduce(upper, gupper, 2, MPIU_REAL, MPIU_MAX, PetscObjectComm((PetscObject) sp)));
    PetscCall(PetscDrawAxisSetLimits(sp->axis, glower[0], gupper[0], glower[1], gupper[1]));
    PetscCall(PetscDrawAxisDraw(sp->axis));
  }

  PetscDrawCollectiveBegin(draw);
  {
    const int dim = sp->dim, nopts = sp->nopts;

    for (int i = 0; i < dim; ++i) {
      for (int p = 0; p < nopts; ++p) {
        PetscInt color = sp->colorized ? PetscDrawRealToColor(sp->z[p*dim], sp->zmin, sp->zmax) : (size > 1 ? PetscDrawRealToColor(rank, 0, size-1) : PETSC_DRAW_RED);

        PetscCall(PetscDrawPoint(draw, sp->x[p*dim+i], sp->y[p*dim+i], color));
      }
    }
  }
  PetscDrawCollectiveEnd(draw);

  PetscCall(PetscDrawFlush(draw));
  PetscCall(PetscDrawPause(draw));
  PetscFunctionReturn(0);
}

/*@
  PetscDrawSPSave - Saves a drawn image

  Collective on PetscDrawSP

  Input Parameter:
. sp - the scatter plot context

  Level: intermediate

.seealso: `PetscDrawSPCreate()`, `PetscDrawSPGetDraw()`, `PetscDrawSetSave()`, `PetscDrawSave()`
@*/
PetscErrorCode PetscDrawSPSave(PetscDrawSP sp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSC_DRAWSP_CLASSID, 1);
  PetscCall(PetscDrawSave(sp->win));
  PetscFunctionReturn(0);
}

/*@
  PetscDrawSPSetLimits - Sets the axis limits for a scatter plot. If more points are added after this call, the limits will be adjusted to include those additional points.

  Not collective

  Input Parameters:
+ xsp - the line graph context
- x_min,x_max,y_min,y_max - the limits

  Level: intermediate

.seealso: `PetscDrawSP`, `PetscDrawSPCreate()`, `PetscDrawSPDraw()`, `PetscDrawSPAddPoint()`, `PetscDrawSPAddPoints()`, `PetscDrawSPGetAxis()`
@*/
PetscErrorCode PetscDrawSPSetLimits(PetscDrawSP sp, PetscReal x_min, PetscReal x_max, PetscReal y_min, PetscReal y_max)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSC_DRAWSP_CLASSID, 1);
  sp->xmin = x_min;
  sp->xmax = x_max;
  sp->ymin = y_min;
  sp->ymax = y_max;
  PetscFunctionReturn(0);
}

/*@
  PetscDrawSPGetAxis - Gets the axis context associated with a line graph.

  Not Collective

  Input Parameter:
. sp - the line graph context

  Output Parameter:
. axis - the axis context

  Note:
  This is useful if one wants to change some axis property, such as labels, color, etc. The axis context should not be destroyed by the application code.

  Level: intermediate

.seealso: `PetscDrawSP`, `PetscDrawSPCreate()`, `PetscDrawSPDraw()`, `PetscDrawSPAddPoint()`, `PetscDrawSPAddPoints()`, `PetscDrawAxis`, `PetscDrawAxisCreate()`
@*/
PetscErrorCode PetscDrawSPGetAxis(PetscDrawSP sp, PetscDrawAxis *axis)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSC_DRAWSP_CLASSID, 1);
  PetscValidPointer(axis, 2);
  *axis = sp->axis;
  PetscFunctionReturn(0);
}

/*@
  PetscDrawSPGetDraw - Gets the draw context associated with a line graph.

  Not Collective

  Input Parameter:
. sp - the line graph context

  Output Parameter:
. draw - the draw context

  Level: intermediate

.seealso: `PetscDrawSP`, `PetscDrawSPCreate()`, `PetscDrawSPDraw()`, `PetscDraw`
@*/
PetscErrorCode PetscDrawSPGetDraw(PetscDrawSP sp, PetscDraw *draw)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSC_DRAWSP_CLASSID, 1);
  PetscValidPointer(draw, 2);
  *draw = sp->win;
  PetscFunctionReturn(0);
}
