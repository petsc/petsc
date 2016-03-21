
/*
  Contains the data structure for plotting a bargraph in a window with an axis.
*/
#include <petscdraw.h>         /*I "petscdraw.h" I*/
#include <petsc/private/petscimpl.h>         /*I "petscsys.h" I*/
#include <petscviewer.h>         /*I "petscviewer.h" I*/
#include <../src/sys/classes/draw/utils/axisimpl.h>   /* so we can directly modify axis xticks */

PetscClassId PETSC_DRAWBAR_CLASSID = 0;

struct _p_PetscDrawBar {
  PETSCHEADER(int);
  PetscErrorCode (*destroy)(PetscDrawSP);
  PetscErrorCode (*view)(PetscDrawSP,PetscViewer);
  PetscDraw      win;
  PetscDrawAxis  axis;
  PetscReal      ymin,ymax;
  int            numBins;
  PetscReal      *values;
  int            color;
  char           **labels;
  PetscBool      sort;
  PetscReal      sorttolerance;
};

#define CHUNKSIZE 100

#undef __FUNCT__
#define __FUNCT__ "PetscDrawBarCreate"
/*@C
   PetscDrawBarCreate - Creates a bar graph data structure.

   Collective over PetscDraw

   Input Parameters:
.  draw  - The window where the graph will be made

   Output Parameters:
.  bar - The bar graph context

   Level: intermediate

   Concepts: bar graph^creating

.seealso: PetscDrawBarDestroy()

@*/
PetscErrorCode  PetscDrawBarCreate(PetscDraw draw,PetscDrawBar *bar)
{
  PetscDrawBar   h;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  PetscValidPointer(bar,2);

  ierr = PetscHeaderCreate(h,PETSC_DRAWBAR_CLASSID,"DrawBar","Bar Graph","Draw",PetscObjectComm((PetscObject)draw),PetscDrawBarDestroy,NULL);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)draw,(PetscObject)h);CHKERRQ(ierr);

  ierr = PetscObjectReference((PetscObject)draw);CHKERRQ(ierr);
  h->win = draw;

  h->view        = NULL;
  h->destroy     = NULL;
  h->color       = PETSC_DRAW_GREEN;
  h->ymin        = 0.;  /* if user has not set these then they are determined from the data */
  h->ymax        = 0.;
  h->numBins     = 0;

  ierr = PetscDrawAxisCreate(draw,&h->axis);CHKERRQ(ierr);
  h->axis->xticks = NULL;

  *bar = h;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDrawBarSetData"
/*@C
   PetscDrawBarSetData

   Logically Collective on PetscDrawBar

   Input Parameter:
+  bar - The bar graph context.
.  bins  - number of items
.  values - values of each item
-  labels - optional label for each bar, NULL terminated array of strings

   Level: intermediate


@*/
PetscErrorCode  PetscDrawBarSetData(PetscDrawBar bar,PetscInt bins,const PetscReal data[],const char *const *labels)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(bar,PETSC_DRAWBAR_CLASSID,1);

  if (bar->numBins != bins) {
    ierr = PetscFree(bar->values);CHKERRQ(ierr);
    ierr = PetscMalloc1(bins, &bar->values);CHKERRQ(ierr);
    bar->numBins = bins;
  }
  ierr = PetscMemcpy(bar->values,data,bins*sizeof(PetscReal));CHKERRQ(ierr);
  bar->numBins = bins;
  if (labels) {
    ierr = PetscStrArrayallocpy(labels,&bar->labels);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDrawBarDestroy"
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*bar) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(*bar,PETSC_DRAWBAR_CLASSID,1);
  if (--((PetscObject)(*bar))->refct > 0) PetscFunctionReturn(0);

  ierr = PetscFree((*bar)->values);CHKERRQ(ierr);
  ierr = PetscStrArrayDestroy(&(*bar)->labels);CHKERRQ(ierr);
  ierr = PetscDrawAxisDestroy(&(*bar)->axis);CHKERRQ(ierr);
  ierr = PetscDrawDestroy(&(*bar)->win);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(bar);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDrawBarDraw"
/*@
  PetscDrawBarDraw - Redraws a bar graph.

  Collective on PetscDrawBar

  Input Parameter:
. bar - The bar graph context

  Level: intermediate

@*/
PetscErrorCode  PetscDrawBarDraw(PetscDrawBar bar)
{
  PetscDraw      draw;
  PetscBool      isnull;
  PetscReal      xmin,xmax,ymin,ymax,*values,binLeft,binRight;
  PetscInt       numValues,i,bcolor,color,idx,*perm,nplot;
  PetscMPIInt    rank;
  PetscErrorCode ierr;
  char           **labels;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(bar,PETSC_DRAWBAR_CLASSID,1);
  ierr = PetscDrawIsNull(bar->win,&isnull);CHKERRQ(ierr);
  if (isnull) PetscFunctionReturn(0);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)bar),&rank);CHKERRQ(ierr);

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
    ierr = PetscMalloc1(numValues,&perm);CHKERRQ(ierr);
    for (i=0; i<numValues;i++) perm[i] = i;
    ierr = PetscSortRealWithPermutation(numValues,values,perm);CHKERRQ(ierr);
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
  ierr = PetscDrawCheckResizedWindow(draw);CHKERRQ(ierr);
  ierr = PetscDrawClear(draw);CHKERRQ(ierr);

  ierr = PetscDrawAxisSetLimits(bar->axis,xmin,xmax,ymin,ymax);CHKERRQ(ierr);
  ierr = PetscDrawAxisDraw(bar->axis);CHKERRQ(ierr);

  ierr = PetscDrawCollectiveBegin(draw);CHKERRQ(ierr);
  if (!rank) { /* Draw bins */
    for (i=0; i<nplot; i++) {
      idx = (bar->sort ? perm[numValues - i - 1] : i);
      binLeft  = xmin + i;
      binRight = xmin + i + 1;
      ierr = PetscDrawRectangle(draw,binLeft,ymin,binRight,values[idx],bcolor,bcolor,bcolor,bcolor);CHKERRQ(ierr);
      ierr = PetscDrawLine(draw,binLeft,ymin,binLeft,values[idx],PETSC_DRAW_BLACK);CHKERRQ(ierr);
      ierr = PetscDrawLine(draw,binRight,ymin,binRight,values[idx],PETSC_DRAW_BLACK);CHKERRQ(ierr);
      ierr = PetscDrawLine(draw,binLeft,values[idx],binRight,values[idx],PETSC_DRAW_BLACK);CHKERRQ(ierr);
      if (labels) {
        PetscReal h;
        ierr = PetscDrawStringGetSize(draw,NULL,&h);CHKERRQ(ierr);
        ierr = PetscDrawStringCentered(draw,.5*(binLeft+binRight),ymin - 1.5*h,bcolor,labels[idx]);CHKERRQ(ierr);
      }
      if (color == PETSC_DRAW_ROTATE) bcolor++;
      if (bcolor > PETSC_DRAW_BASIC_COLORS-1) bcolor = PETSC_DRAW_BLACK+1;
    }
  }
  ierr = PetscDrawCollectiveEnd(draw);CHKERRQ(ierr);
  if (bar->sort) {ierr = PetscFree(perm);CHKERRQ(ierr);}

  ierr = PetscDrawFlush(draw);CHKERRQ(ierr);
  ierr = PetscDrawPause(draw);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDrawBarSave"
/*@
  PetscDrawBarSave - Saves a drawn image

  Collective on PetscDrawBar

  Input Parameters:
. bar - The bar graph context

  Level: intermediate

  Concepts: bar graph^saving

.seealso:  PetscDrawBarCreate(), PetscDrawBarGetDraw(), PetscDrawSetSave(), PetscDrawSave()
@*/
PetscErrorCode  PetscDrawBarSave(PetscDrawBar bar)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(bar,PETSC_DRAWBAR_CLASSID,1);
  ierr = PetscDrawSave(bar->win);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDrawBarSetColor"
/*@
  PetscDrawBarSetColor - Sets the color the bars will be drawn with.

  Logically Collective on PetscDrawBar

  Input Parameters:
+ bar - The bar graph context
- color - one of the colors defined in petscdraw.h or PETSC_DRAW_ROTATE to make each bar a
          different color

  Level: intermediate

@*/
PetscErrorCode  PetscDrawBarSetColor(PetscDrawBar bar, int color)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(bar, PETSC_DRAWBAR_CLASSID,1);
  bar->color = color;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDrawBarSort"
/*@
  PetscDrawBarSort - Sorts the values before drawing the bar chart

  Logically Collective on PetscDrawBar

  Input Parameters:
+ bar - The bar graph context
. sort - PETSC_TRUE to sort the values
. tolerance - discard values less than tolerance

  Level: intermediate

  Concepts: bar graph^setting axis
@*/
PetscErrorCode  PetscDrawBarSort(PetscDrawBar bar, PetscBool sort, PetscReal tolerance)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(bar,PETSC_DRAWBAR_CLASSID,1);
  bar->sort          = sort;
  bar->sorttolerance = tolerance;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDrawBarSetLimits"
/*@
  PetscDrawBarSetLimits - Sets the axis limits for a bar graph. If more
  points are added after this call, the limits will be adjusted to
  include those additional points.

  Logically Collective on PetscDrawBar

  Input Parameters:
+ bar - The bar graph context
- y_min,y_max - The limits

  Level: intermediate

  Concepts: bar graph^setting axis
@*/
PetscErrorCode  PetscDrawBarSetLimits(PetscDrawBar bar, PetscReal y_min, PetscReal y_max)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(bar,PETSC_DRAWBAR_CLASSID,1);
  bar->ymin = y_min;
  bar->ymax = y_max;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDrawBarGetAxis"
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

@*/
PetscErrorCode  PetscDrawBarGetAxis(PetscDrawBar bar,PetscDrawAxis *axis)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(bar,PETSC_DRAWBAR_CLASSID,1);
  PetscValidPointer(axis,2);
  *axis = bar->axis;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDrawBarGetDraw"
/*@C
  PetscDrawBarGetDraw - Gets the draw context associated with a bar graph.

  Not Collective, PetscDraw is parallel if PetscDrawBar is parallel

  Input Parameter:
. bar - The bar graph context

  Output Parameter:
. draw  - The draw context

  Level: intermediate

@*/
PetscErrorCode  PetscDrawBarGetDraw(PetscDrawBar bar,PetscDraw *draw)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(bar,PETSC_DRAWBAR_CLASSID,1);
  PetscValidPointer(draw,2);
  *draw = bar->win;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDrawBarSetFromOptions"
/*@
    PetscDrawBarSetFromOptions - Sets options related to the PetscDrawBar

    Collective over PetscDrawBar

    Options Database:
.  -bar_sort - sort the entries before drawing the bar graph

    Level: intermediate


.seealso:  PetscDrawBarDestroy(), PetscDrawBarCreate()
@*/
PetscErrorCode  PetscDrawBarSetFromOptions(PetscDrawBar bar)
{
  PetscErrorCode ierr;
  PetscBool      set;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(bar,PETSC_DRAWBAR_CLASSID,1);

  ierr = PetscOptionsHasName(((PetscObject)bar)->options,((PetscObject)bar)->prefix,"-bar_sort",&set);CHKERRQ(ierr);
  if (set) {
    PetscReal tol = bar->sorttolerance;
    ierr = PetscOptionsGetReal(((PetscObject)bar)->options,((PetscObject)bar)->prefix,"-bar_sort",&tol,NULL);CHKERRQ(ierr);
    ierr = PetscDrawBarSort(bar,PETSC_TRUE,tol);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

