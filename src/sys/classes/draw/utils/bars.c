
/*
  Contains the data structure for plotting a bargraph in a window with an axis.
*/

#include <petsc/private/drawimpl.h> /*I "petscdraw.h" I*/
#include <petscviewer.h>            /*I "petscviewer.h" I*/

PetscClassId PETSC_DRAWBAR_CLASSID = 0;

/*@C
   PetscDrawBarCreate - Creates a bar graph data structure.

   Collective over PetscDraw

   Input Parameters:
.  draw  - The window where the graph will be made

   Output Parameters:
.  bar - The bar graph context

   Notes:
    Call PetscDrawBarSetData() to provide the bins to be plotted and then PetscDrawBarDraw() to display the new plot

  The difference between a bar chart, PetscDrawBar, and a histogram, PetscDrawHG, is explained here https://stattrek.com/statistics/charts/histogram.aspx?Tutorial=AP

   The MPI communicator that owns the PetscDraw owns this PetscDrawBar, but the calls to set options and add data are ignored on all processes except the
   zeroth MPI process in the communicator. All MPI processes in the communicator must call PetscDrawBarDraw() to display the updated graph.

   Level: intermediate

.seealso: PetscDrawLGCreate(), PetscDrawLG, PetscDrawSPCreate(), PetscDrawSP, PetscDrawHGCreate(), PetscDrawHG, PetscDrawBarDestroy(), PetscDrawBarSetData(),
          PetscDrawBar, PetscDrawBarDraw(), PetscDrawBarSave(), PetscDrawBarSetColor(), PetscDrawBarSort(), PetscDrawBarSetLimits(), PetscDrawBarGetAxis(), PetscDrawAxis,
          PetscDrawBarGetDraw(), PetscDrawBarSetFromOptions()
@*/
PetscErrorCode  PetscDrawBarCreate(PetscDraw draw,PetscDrawBar *bar)
{
  PetscDrawBar   h;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  PetscValidPointer(bar,2);

  PetscCall(PetscHeaderCreate(h,PETSC_DRAWBAR_CLASSID,"DrawBar","Bar Graph","Draw",PetscObjectComm((PetscObject)draw),PetscDrawBarDestroy,NULL));
  PetscCall(PetscLogObjectParent((PetscObject)draw,(PetscObject)h));

  PetscCall(PetscObjectReference((PetscObject)draw));
  h->win = draw;

  h->view        = NULL;
  h->destroy     = NULL;
  h->color       = PETSC_DRAW_GREEN;
  h->ymin        = 0.;  /* if user has not set these then they are determined from the data */
  h->ymax        = 0.;
  h->numBins     = 0;

  PetscCall(PetscDrawAxisCreate(draw,&h->axis));
  h->axis->xticks = NULL;

  *bar = h;
  PetscFunctionReturn(0);
}

/*@C
   PetscDrawBarSetData

   Logically Collective on PetscDrawBar

   Input Parameters:
+  bar - The bar graph context.
.  bins  - number of items
.  values - values of each item
-  labels - optional label for each bar, NULL terminated array of strings

   Level: intermediate

   Notes:
    Call PetscDrawBarDraw() after this call to display the new plot

.seealso: PetscDrawBarCreate(), PetscDrawBar, PetscDrawBarDraw()

@*/
PetscErrorCode  PetscDrawBarSetData(PetscDrawBar bar,PetscInt bins,const PetscReal data[],const char *const *labels)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(bar,PETSC_DRAWBAR_CLASSID,1);

  if (bar->numBins != bins) {
    PetscCall(PetscFree(bar->values));
    PetscCall(PetscMalloc1(bins, &bar->values));
    bar->numBins = bins;
  }
  PetscCall(PetscArraycpy(bar->values,data,bins));
  bar->numBins = bins;
  if (labels) {
    PetscCall(PetscStrArrayallocpy(labels,&bar->labels));
  }
  PetscFunctionReturn(0);
}

/*@C
  PetscDrawBarDestroy - Frees all space taken up by bar graph data structure.

  Collective over PetscDrawBar

  Input Parameter:
. bar - The bar graph context

  Level: intermediate

.seealso:  PetscDrawBarCreate()
@*/
PetscErrorCode  PetscDrawBarDestroy(PetscDrawBar *bar)
{
  PetscFunctionBegin;
  if (!*bar) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(*bar,PETSC_DRAWBAR_CLASSID,1);
  if (--((PetscObject)(*bar))->refct > 0) PetscFunctionReturn(0);

  PetscCall(PetscFree((*bar)->values));
  PetscCall(PetscStrArrayDestroy(&(*bar)->labels));
  PetscCall(PetscDrawAxisDestroy(&(*bar)->axis));
  PetscCall(PetscDrawDestroy(&(*bar)->win));
  PetscCall(PetscHeaderDestroy(bar));
  PetscFunctionReturn(0);
}

/*@
  PetscDrawBarDraw - Redraws a bar graph.

  Collective on PetscDrawBar

  Input Parameter:
. bar - The bar graph context

  Level: intermediate

.seealso: PetscDrawBar, PetscDrawBarCreate(), PetscDrawBarSetData()

@*/
PetscErrorCode  PetscDrawBarDraw(PetscDrawBar bar)
{
  PetscDraw      draw;
  PetscBool      isnull;
  PetscReal      xmin,xmax,ymin,ymax,*values,binLeft,binRight;
  PetscInt       numValues,i,bcolor,color,idx,*perm,nplot;
  PetscMPIInt    rank;
  char           **labels;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(bar,PETSC_DRAWBAR_CLASSID,1);
  PetscCall(PetscDrawIsNull(bar->win,&isnull));
  if (isnull) PetscFunctionReturn(0);
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)bar),&rank));

  if (bar->numBins < 1) PetscFunctionReturn(0);

  color = bar->color;
  if (color == PETSC_DRAW_ROTATE) bcolor = PETSC_DRAW_BLACK+1;
  else bcolor = color;

  numValues = bar->numBins;
  values    = bar->values;
  if (bar->ymin == bar->ymax) {
    /* user has not set bounds on bars so set them based on the data */
    ymin = PETSC_MAX_REAL;
    ymax = PETSC_MIN_REAL;
    for (i=0; i<numValues; i++) {
      ymin = PetscMin(ymin,values[i]);
      ymax = PetscMax(ymax,values[i]);
    }
  } else {
    ymin = bar->ymin;
    ymax = bar->ymax;
  }
  nplot  = numValues;  /* number of points to actually plot; if some are lower than requested tolerance */
  xmin   = 0.0;
  xmax   = nplot;
  labels = bar->labels;

  if (bar->sort) {
    PetscCall(PetscMalloc1(numValues,&perm));
    for (i=0; i<numValues;i++) perm[i] = i;
    PetscCall(PetscSortRealWithPermutation(numValues,values,perm));
    if (bar->sorttolerance) {
      for (i=0; i<numValues;i++) {
        if (values[perm[numValues - i - 1]] < bar->sorttolerance) {
          nplot = i;
          break;
        }
      }
    }
  }

  draw = bar->win;
  PetscCall(PetscDrawCheckResizedWindow(draw));
  PetscCall(PetscDrawClear(draw));

  PetscCall(PetscDrawAxisSetLimits(bar->axis,xmin,xmax,ymin,ymax));
  PetscCall(PetscDrawAxisDraw(bar->axis));

  PetscDrawCollectiveBegin(draw);
  if (rank == 0) { /* Draw bins */
    for (i=0; i<nplot; i++) {
      idx = (bar->sort ? perm[numValues - i - 1] : i);
      binLeft  = xmin + i;
      binRight = xmin + i + 1;
      PetscCall(PetscDrawRectangle(draw,binLeft,ymin,binRight,values[idx],bcolor,bcolor,bcolor,bcolor));
      PetscCall(PetscDrawLine(draw,binLeft,ymin,binLeft,values[idx],PETSC_DRAW_BLACK));
      PetscCall(PetscDrawLine(draw,binRight,ymin,binRight,values[idx],PETSC_DRAW_BLACK));
      PetscCall(PetscDrawLine(draw,binLeft,values[idx],binRight,values[idx],PETSC_DRAW_BLACK));
      if (labels) {
        PetscReal h;
        PetscCall(PetscDrawStringGetSize(draw,NULL,&h));
        PetscCall(PetscDrawStringCentered(draw,.5*(binLeft+binRight),ymin - 1.5*h,bcolor,labels[idx]));
      }
      if (color == PETSC_DRAW_ROTATE) bcolor++;
      if (bcolor > PETSC_DRAW_BASIC_COLORS-1) bcolor = PETSC_DRAW_BLACK+1;
    }
  }
  PetscDrawCollectiveEnd(draw);
  if (bar->sort) PetscCall(PetscFree(perm));

  PetscCall(PetscDrawFlush(draw));
  PetscCall(PetscDrawPause(draw));
  PetscFunctionReturn(0);
}

/*@
  PetscDrawBarSave - Saves a drawn image

  Collective on PetscDrawBar

  Input Parameters:
. bar - The bar graph context

  Level: intermediate

.seealso:  PetscDrawBarCreate(), PetscDrawBarGetDraw(), PetscDrawSetSave(), PetscDrawSave(), PetscDrawBarSetData()
@*/
PetscErrorCode  PetscDrawBarSave(PetscDrawBar bar)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(bar,PETSC_DRAWBAR_CLASSID,1);
  PetscCall(PetscDrawSave(bar->win));
  PetscFunctionReturn(0);
}

/*@
  PetscDrawBarSetColor - Sets the color the bars will be drawn with.

  Logically Collective on PetscDrawBar

  Input Parameters:
+ bar - The bar graph context
- color - one of the colors defined in petscdraw.h or PETSC_DRAW_ROTATE to make each bar a
          different color

  Level: intermediate

.seealso: PetscDrawBarCreate(), PetscDrawBar, PetscDrawBarSetData(), PetscDrawBarDraw(), PetscDrawBarGetAxis()

@*/
PetscErrorCode  PetscDrawBarSetColor(PetscDrawBar bar, int color)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(bar, PETSC_DRAWBAR_CLASSID,1);
  bar->color = color;
  PetscFunctionReturn(0);
}

/*@
  PetscDrawBarSort - Sorts the values before drawing the bar chart

  Logically Collective on PetscDrawBar

  Input Parameters:
+ bar - The bar graph context
. sort - PETSC_TRUE to sort the values
- tolerance - discard values less than tolerance

  Level: intermediate

.seealso: PetscDrawBarCreate(), PetscDrawBar, PetscDrawBarSetData(), PetscDrawBarSetColor(), PetscDrawBarDraw(), PetscDrawBarGetAxis()
@*/
PetscErrorCode  PetscDrawBarSort(PetscDrawBar bar, PetscBool sort, PetscReal tolerance)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(bar,PETSC_DRAWBAR_CLASSID,1);
  bar->sort          = sort;
  bar->sorttolerance = tolerance;
  PetscFunctionReturn(0);
}

/*@
  PetscDrawBarSetLimits - Sets the axis limits for a bar graph. If more
  points are added after this call, the limits will be adjusted to
  include those additional points.

  Logically Collective on PetscDrawBar

  Input Parameters:
+ bar - The bar graph context
- y_min,y_max - The limits

  Level: intermediate

.seealso: PetscDrawBarCreate(), PetscDrawBar, PetscDrawBarGetAxis(), PetscDrawBarSetData(), PetscDrawBarDraw()
@*/
PetscErrorCode  PetscDrawBarSetLimits(PetscDrawBar bar, PetscReal y_min, PetscReal y_max)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(bar,PETSC_DRAWBAR_CLASSID,1);
  bar->ymin = y_min;
  bar->ymax = y_max;
  PetscFunctionReturn(0);
}

/*@C
  PetscDrawBarGetAxis - Gets the axis context associated with a bar graph.
  This is useful if one wants to change some axis property, such as
  labels, color, etc. The axis context should not be destroyed by the
  application code.

  Not Collective, PetscDrawAxis is parallel if PetscDrawBar is parallel

  Input Parameter:
. bar - The bar graph context

  Output Parameter:
. axis - The axis context

  Level: intermediate

.seealso: PetscDrawBarCreate(), PetscDrawBar, PetscDrawAxis, PetscDrawAxisCreate()
@*/
PetscErrorCode  PetscDrawBarGetAxis(PetscDrawBar bar,PetscDrawAxis *axis)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(bar,PETSC_DRAWBAR_CLASSID,1);
  PetscValidPointer(axis,2);
  *axis = bar->axis;
  PetscFunctionReturn(0);
}

/*@C
  PetscDrawBarGetDraw - Gets the draw context associated with a bar graph.

  Not Collective, PetscDraw is parallel if PetscDrawBar is parallel

  Input Parameter:
. bar - The bar graph context

  Output Parameter:
. draw  - The draw context

  Level: intermediate

.seealso: PetscDrawBarCreate(), PetscDrawBar, PetscDrawBarDraw(), PetscDraw
@*/
PetscErrorCode  PetscDrawBarGetDraw(PetscDrawBar bar,PetscDraw *draw)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(bar,PETSC_DRAWBAR_CLASSID,1);
  PetscValidPointer(draw,2);
  *draw = bar->win;
  PetscFunctionReturn(0);
}

/*@
    PetscDrawBarSetFromOptions - Sets options related to the PetscDrawBar

    Collective over PetscDrawBar

    Options Database:
.  -bar_sort - sort the entries before drawing the bar graph

    Level: intermediate

.seealso:  PetscDrawBarDestroy(), PetscDrawBarCreate(), PetscDrawBarSort()
@*/
PetscErrorCode  PetscDrawBarSetFromOptions(PetscDrawBar bar)
{
  PetscBool      set;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(bar,PETSC_DRAWBAR_CLASSID,1);

  PetscCall(PetscOptionsHasName(((PetscObject)bar)->options,((PetscObject)bar)->prefix,"-bar_sort",&set));
  if (set) {
    PetscReal tol = bar->sorttolerance;
    PetscCall(PetscOptionsGetReal(((PetscObject)bar)->options,((PetscObject)bar)->prefix,"-bar_sort",&tol,NULL));
    PetscCall(PetscDrawBarSort(bar,PETSC_TRUE,tol));
  }
  PetscFunctionReturn(0);
}
