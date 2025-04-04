/*
  Contains the data structure for plotting a histogram in a window with an axis.
*/
#include <petscdraw.h>               /*I "petscdraw.h" I*/
#include <petsc/private/petscimpl.h> /*I "petscsys.h" I*/
#include <petscviewer.h>             /*I "petscviewer.h" I*/

PetscClassId PETSC_DRAWHG_CLASSID = 0;

struct _p_PetscDrawHG {
  PETSCHEADER(int);
  PetscErrorCode (*destroy)(PetscDrawSP);
  PetscErrorCode (*view)(PetscDrawSP, PetscViewer);
  PetscDraw     win;
  PetscDrawAxis axis;
  PetscReal     xmin, xmax;
  PetscReal     ymin, ymax;
  int           numBins;
  int           maxBins;
  PetscReal    *bins;
  int           numValues;
  int           maxValues;
  PetscReal    *values;
  PetscReal    *weights;
  int           color;
  PetscBool     calcStats;
  PetscBool     integerBins;
};

#define CHUNKSIZE 100

/*@
  PetscDrawHGCreate - Creates a histogram data structure.

  Collective

  Input Parameters:
+ draw - The window where the graph will be made
- bins - The number of bins to use

  Output Parameter:
. hist - The histogram context

  Level: intermediate

  Notes:
  The difference between a bar chart, `PetscDrawBar`, and a histogram, `PetscDrawHG`, is explained here <https://stattrek.com/statistics/charts/histogram.aspx?Tutorial=AP>

  The histogram is only displayed when `PetscDrawHGDraw()` is called.

  The MPI communicator that owns the `PetscDraw` owns this `PetscDrawHG`, but the calls to set options and add data are ignored on all processes except the
  zeroth MPI process in the communicator. All MPI processes in the communicator must call `PetscDrawHGDraw()` to display the updated graph.

.seealso: `PetscDrawHGDestroy()`, `PetscDrawHG`, `PetscDrawBarCreate()`, `PetscDrawBar`, `PetscDrawLGCreate()`, `PetscDrawLG`, `PetscDrawSPCreate()`, `PetscDrawSP`,
          `PetscDrawHGSetNumberBins()`, `PetscDrawHGReset()`, `PetscDrawHGAddValue()`, `PetscDrawHGDraw()`, `PetscDrawHGSave()`, `PetscDrawHGView()`, `PetscDrawHGSetColor()`,
          `PetscDrawHGSetLimits()`, `PetscDrawHGCalcStats()`, `PetscDrawHGIntegerBins()`, `PetscDrawHGGetAxis()`, `PetscDrawAxis`, `PetscDrawHGGetDraw()`
@*/
PetscErrorCode PetscDrawHGCreate(PetscDraw draw, int bins, PetscDrawHG *hist)
{
  PetscDrawHG h;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw, PETSC_DRAW_CLASSID, 1);
  PetscValidLogicalCollectiveInt(draw, bins, 2);
  PetscAssertPointer(hist, 3);

  PetscCall(PetscHeaderCreate(h, PETSC_DRAWHG_CLASSID, "DrawHG", "Histogram", "Draw", PetscObjectComm((PetscObject)draw), PetscDrawHGDestroy, NULL));
  PetscCall(PetscObjectReference((PetscObject)draw));
  h->win     = draw;
  h->view    = NULL;
  h->destroy = NULL;
  h->color   = PETSC_DRAW_GREEN;
  h->xmin    = PETSC_MAX_REAL;
  h->xmax    = PETSC_MIN_REAL;
  h->ymin    = 0.;
  h->ymax    = 1.e-6;
  h->numBins = bins;
  h->maxBins = bins;
  PetscCall(PetscMalloc1(h->maxBins, &h->bins));
  h->numValues   = 0;
  h->maxValues   = CHUNKSIZE;
  h->calcStats   = PETSC_FALSE;
  h->integerBins = PETSC_FALSE;
  PetscCall(PetscMalloc2(h->maxValues, &h->values, h->maxValues, &h->weights));
  PetscCall(PetscDrawAxisCreate(draw, &h->axis));
  *hist = h;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDrawHGSetNumberBins - Change the number of bins that are to be drawn in the histogram

  Logically Collective

  Input Parameters:
+ hist - The histogram context.
- bins - The number of bins.

  Level: intermediate

.seealso: `PetscDrawHGCreate()`, `PetscDrawHG`, `PetscDrawHGDraw()`, `PetscDrawHGIntegerBins()`
@*/
PetscErrorCode PetscDrawHGSetNumberBins(PetscDrawHG hist, int bins)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(hist, PETSC_DRAWHG_CLASSID, 1);
  PetscValidLogicalCollectiveInt(hist, bins, 2);

  if (hist->maxBins < bins) {
    PetscCall(PetscFree(hist->bins));
    PetscCall(PetscMalloc1(bins, &hist->bins));
    hist->maxBins = bins;
  }
  hist->numBins = bins;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDrawHGReset - Clears histogram to allow for reuse with new data.

  Logically Collective

  Input Parameter:
. hist - The histogram context.

  Level: intermediate

.seealso: `PetscDrawHGCreate()`, `PetscDrawHG`, `PetscDrawHGDraw()`, `PetscDrawHGAddValue()`
@*/
PetscErrorCode PetscDrawHGReset(PetscDrawHG hist)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(hist, PETSC_DRAWHG_CLASSID, 1);

  hist->xmin      = PETSC_MAX_REAL;
  hist->xmax      = PETSC_MIN_REAL;
  hist->ymin      = 0.0;
  hist->ymax      = 1.e-6;
  hist->numValues = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDrawHGDestroy - Frees all space taken up by histogram data structure.

  Collective

  Input Parameter:
. hist - The histogram context

  Level: intermediate

.seealso: `PetscDrawHGCreate()`, `PetscDrawHG`
@*/
PetscErrorCode PetscDrawHGDestroy(PetscDrawHG *hist)
{
  PetscFunctionBegin;
  if (!*hist) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific(*hist, PETSC_DRAWHG_CLASSID, 1);
  if (--((PetscObject)*hist)->refct > 0) {
    *hist = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(PetscFree((*hist)->bins));
  PetscCall(PetscFree2((*hist)->values, (*hist)->weights));
  PetscCall(PetscDrawAxisDestroy(&(*hist)->axis));
  PetscCall(PetscDrawDestroy(&(*hist)->win));
  PetscCall(PetscHeaderDestroy(hist));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDrawHGAddValue - Adds another value to the histogram.

  Logically Collective

  Input Parameters:
+ hist  - The histogram
- value - The value

  Level: intermediate

  Note:
  Calls to this function are used to create a standard histogram with integer bin heights. Use calls to `PetscDrawHGAddWeightedValue()` to create a histogram with non-integer bin heights.

.seealso: `PetscDrawHGCreate()`, `PetscDrawHG`, `PetscDrawHGDraw()`, `PetscDrawHGReset()`, `PetscDrawHGAddWeightedValue()`
@*/
PetscErrorCode PetscDrawHGAddValue(PetscDrawHG hist, PetscReal value)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(hist, PETSC_DRAWHG_CLASSID, 1);

  /* Allocate more memory if necessary */
  if (hist->numValues >= hist->maxValues) {
    PetscReal *tmp, *tmpw;

    PetscCall(PetscMalloc2(hist->maxValues + CHUNKSIZE, &tmp, hist->maxValues + CHUNKSIZE, &tmpw));
    PetscCall(PetscArraycpy(tmp, hist->values, hist->maxValues));
    PetscCall(PetscArraycpy(tmpw, hist->weights, hist->maxValues));
    PetscCall(PetscFree2(hist->values, hist->weights));

    hist->values  = tmp;
    hist->weights = tmpw;
    hist->maxValues += CHUNKSIZE;
  }
  /* I disagree with the original PETSc implementation here. There should be no overshoot, but rather the
     stated convention of using half-open intervals (always the way to go) */
  if (!hist->numValues && (hist->xmin == PETSC_MAX_REAL) && (hist->xmax == PETSC_MIN_REAL)) {
    hist->xmin = value;
    hist->xmax = value;
#if 1
  } else {
    /* Update limits */
    if (value > hist->xmax) hist->xmax = value;
    if (value < hist->xmin) hist->xmin = value;
#else
  } else if (hist->numValues == 1) {
    /* Update limits -- We need to overshoot the largest value somewhat */
    if (value > hist->xmax) hist->xmax = value + 0.001 * (value - hist->xmin) / (PetscReal)hist->numBins;
    if (value < hist->xmin) {
      hist->xmin = value;
      hist->xmax = hist->xmax + 0.001 * (hist->xmax - hist->xmin) / (PetscReal)hist->numBins;
    }
  } else {
    /* Update limits -- We need to overshoot the largest value somewhat */
    if (value > hist->xmax) hist->xmax = value + 0.001 * (hist->xmax - hist->xmin) / (PetscReal)hist->numBins;
    if (value < hist->xmin) hist->xmin = value;
#endif
  }

  hist->values[hist->numValues]  = value;
  hist->weights[hist->numValues] = 1.;
  ++hist->numValues;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDrawHGAddWeightedValue - Adds another value to the histogram with a weight.

  Logically Collective

  Input Parameters:
+ hist   - The histogram
. value  - The value
- weight - The value weight

  Level: intermediate

  Notes:
  Calls to this function are used to create a histogram with non-integer bin heights. Use calls to `PetscDrawHGAddValue()` to create a standard histogram with integer bin heights.

  This allows us to histogram frequency and probability distributions (<https://learnche.org/pid/univariate-review/histograms-and-probability-distributions>). We can use this to look at particle weight distributions in Particle-in-Cell (PIC) methods, for example.

.seealso: `PetscDrawHGCreate()`, `PetscDrawHG`, `PetscDrawHGDraw()`, `PetscDrawHGReset()`, `PetscDrawHGAddValue()`
@*/
PetscErrorCode PetscDrawHGAddWeightedValue(PetscDrawHG hist, PetscReal value, PetscReal weight)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(hist, PETSC_DRAWHG_CLASSID, 1);

  /* Allocate more memory if necessary */
  if (hist->numValues >= hist->maxValues) {
    PetscReal *tmp, *tmpw;

    PetscCall(PetscMalloc2(hist->maxValues + CHUNKSIZE, &tmp, hist->maxValues + CHUNKSIZE, &tmpw));
    PetscCall(PetscArraycpy(tmp, hist->values, hist->maxValues));
    PetscCall(PetscArraycpy(tmpw, hist->weights, hist->maxValues));
    PetscCall(PetscFree2(hist->values, hist->weights));

    hist->values  = tmp;
    hist->weights = tmpw;
    hist->maxValues += CHUNKSIZE;
  }
  /* I disagree with the original PETSc implementation here. There should be no overshoot, but rather the
     stated convention of using half-open intervals (always the way to go) */
  if (!hist->numValues && (hist->xmin == PETSC_MAX_REAL) && (hist->xmax == PETSC_MIN_REAL)) {
    hist->xmin = value;
    hist->xmax = value;
  } else {
    /* Update limits */
    if (value > hist->xmax) hist->xmax = value;
    if (value < hist->xmin) hist->xmin = value;
  }

  hist->values[hist->numValues]  = value;
  hist->weights[hist->numValues] = weight;
  ++hist->numValues;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDrawHGDraw - Redraws a histogram.

  Collective

  Input Parameter:
. hist - The histogram context

  Level: intermediate

.seealso: `PetscDrawHGCreate()`, `PetscDrawHG`, `PetscDrawHGAddValue()`, `PetscDrawHGReset()`
@*/
PetscErrorCode PetscDrawHGDraw(PetscDrawHG hist)
{
  PetscDraw   draw;
  PetscBool   isnull, usewt = PETSC_FALSE;
  PetscReal   xmin, xmax, ymin, ymax, *bins, *values, *weights, binSize, binLeft, binRight, maxHeight, mean, var, totwt = 0.;
  char        title[256];
  char        xlabel[256];
  PetscInt    numValues, initSize, i, p;
  int         bcolor, color, numBins, numBinsOld;
  PetscMPIInt rank;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(hist, PETSC_DRAWHG_CLASSID, 1);
  PetscCall(PetscDrawIsNull(hist->win, &isnull));
  if (isnull) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)hist), &rank));

  if ((hist->xmin >= hist->xmax) || (hist->ymin >= hist->ymax)) PetscFunctionReturn(PETSC_SUCCESS);
  if (hist->numValues < 1) PetscFunctionReturn(PETSC_SUCCESS);

  color = hist->color;
  if (color == PETSC_DRAW_ROTATE) bcolor = PETSC_DRAW_BLACK + 1;
  else bcolor = color;

  xmin      = hist->xmin;
  xmax      = hist->xmax;
  ymin      = hist->ymin;
  ymax      = hist->ymax;
  numValues = hist->numValues;
  values    = hist->values;
  weights   = hist->weights;
  mean      = 0.0;
  var       = 0.0;

  draw = hist->win;
  PetscCall(PetscDrawCheckResizedWindow(draw));
  PetscCall(PetscDrawClear(draw));

  if (xmin == xmax) {
    /* Calculate number of points in each bin */
    bins    = hist->bins;
    bins[0] = 0.;
    for (p = 0; p < numValues; p++) {
      if (values[p] == xmin) bins[0] += weights[0];
      mean += values[p] * weights[p];
      var += values[p] * values[p] * weights[p];
      totwt += weights[p];
      if (weights[p] != 1.) usewt = PETSC_TRUE;
    }
    maxHeight = bins[0];
    if (maxHeight > ymax) ymax = hist->ymax = maxHeight;
    xmax = xmin + 1;
    PetscCall(PetscDrawAxisSetLimits(hist->axis, xmin, xmax, ymin, ymax));
    if (hist->calcStats) {
      if (usewt) {
        mean /= totwt;
        var = (var - totwt * mean * mean) / totwt;

        PetscCall(PetscSNPrintf(xlabel, 256, "Total Weight: %g", (double)totwt));
      } else {
        mean /= numValues;
        if (numValues > 1) var = (var - numValues * mean * mean) / (PetscReal)(numValues - 1);
        else var = 0.0;

        PetscCall(PetscSNPrintf(xlabel, 256, "Total: %" PetscInt_FMT, numValues));
      }
      PetscCall(PetscSNPrintf(title, 256, "Mean: %g  Var: %g", (double)mean, (double)var));
      PetscCall(PetscDrawAxisSetLabels(hist->axis, title, xlabel, NULL));
    }
    PetscCall(PetscDrawAxisDraw(hist->axis));
    PetscDrawCollectiveBegin(draw);
    if (rank == 0) { /* Draw bins */
      binLeft  = xmin;
      binRight = xmax;
      PetscCall(PetscDrawRectangle(draw, binLeft, ymin, binRight, bins[0], bcolor, bcolor, bcolor, bcolor));
      PetscCall(PetscDrawLine(draw, binLeft, ymin, binLeft, bins[0], PETSC_DRAW_BLACK));
      PetscCall(PetscDrawLine(draw, binRight, ymin, binRight, bins[0], PETSC_DRAW_BLACK));
      PetscCall(PetscDrawLine(draw, binLeft, bins[0], binRight, bins[0], PETSC_DRAW_BLACK));
    }
    PetscDrawCollectiveEnd(draw);
  } else {
    numBins    = hist->numBins;
    numBinsOld = hist->numBins;
    if (hist->integerBins && (((int)xmax - xmin) + 1.0e-05 > xmax - xmin)) {
      initSize = ((int)(xmax - xmin)) / numBins;
      while (initSize * numBins != (int)xmax - xmin) {
        initSize = PetscMax(initSize - 1, 1);
        numBins  = ((int)(xmax - xmin)) / initSize;
        PetscCall(PetscDrawHGSetNumberBins(hist, numBins));
      }
    }
    binSize = (xmax - xmin) / (PetscReal)numBins;
    bins    = hist->bins;

    PetscCall(PetscArrayzero(bins, numBins));

    maxHeight = 0.0;
    for (i = 0; i < numBins; i++) {
      binLeft  = xmin + binSize * i;
      binRight = xmin + binSize * (i + 1);
      for (p = 0; p < numValues; p++) {
        if ((values[p] >= binLeft) && (values[p] < binRight)) bins[i] += weights[p];
        /* Handle last bin separately */
        if ((i == numBins - 1) && (values[p] == binRight)) bins[i] += weights[p];
        if (!i) {
          mean += values[p] * weights[p];
          var += values[p] * values[p] * weights[p];
          totwt += weights[p];
          if (weights[p] != 1.) usewt = PETSC_TRUE;
        }
      }
      maxHeight = PetscMax(maxHeight, bins[i]);
    }
    if (maxHeight > ymax) ymax = hist->ymax = maxHeight;

    PetscCall(PetscDrawAxisSetLimits(hist->axis, xmin, xmax, ymin, ymax));
    if (hist->calcStats) {
      if (usewt) {
        mean /= totwt;
        var = (var - totwt * mean * mean) / totwt;

        PetscCall(PetscSNPrintf(xlabel, 256, "Total Weight: %g", (double)totwt));
      } else {
        mean /= numValues;
        if (numValues > 1) var = (var - numValues * mean * mean) / (PetscReal)(numValues - 1);
        else var = 0.0;

        PetscCall(PetscSNPrintf(xlabel, 256, "Total: %" PetscInt_FMT, numValues));
      }
      PetscCall(PetscSNPrintf(title, 256, "Mean: %g  Var: %g", (double)mean, (double)var));
      PetscCall(PetscDrawAxisSetLabels(hist->axis, title, xlabel, NULL));
    }
    PetscCall(PetscDrawAxisDraw(hist->axis));
    PetscDrawCollectiveBegin(draw);
    if (rank == 0) { /* Draw bins */
      for (i = 0; i < numBins; i++) {
        binLeft  = xmin + binSize * i;
        binRight = xmin + binSize * (i + 1);
        PetscCall(PetscDrawRectangle(draw, binLeft, ymin, binRight, bins[i], bcolor, bcolor, bcolor, bcolor));
        PetscCall(PetscDrawLine(draw, binLeft, ymin, binLeft, bins[i], PETSC_DRAW_BLACK));
        PetscCall(PetscDrawLine(draw, binRight, ymin, binRight, bins[i], PETSC_DRAW_BLACK));
        PetscCall(PetscDrawLine(draw, binLeft, bins[i], binRight, bins[i], PETSC_DRAW_BLACK));
        if (color == PETSC_DRAW_ROTATE && bins[i]) bcolor++;
        if (bcolor > PETSC_DRAW_BASIC_COLORS - 1) bcolor = PETSC_DRAW_BLACK + 1;
      }
    }
    PetscDrawCollectiveEnd(draw);
    PetscCall(PetscDrawHGSetNumberBins(hist, numBinsOld));
  }

  PetscCall(PetscDrawFlush(draw));
  PetscCall(PetscDrawPause(draw));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDrawHGSave - Saves a drawn image

  Collective

  Input Parameter:
. hg - The histogram context

  Level: intermediate

.seealso: `PetscDrawSave()`, `PetscDrawHGCreate()`, `PetscDrawHGGetDraw()`, `PetscDrawSetSave()`, `PetscDrawHGDraw()`
@*/
PetscErrorCode PetscDrawHGSave(PetscDrawHG hg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(hg, PETSC_DRAWHG_CLASSID, 1);
  PetscCall(PetscDrawSave(hg->win));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDrawHGView - Prints the histogram information to a viewer

  Not Collective

  Input Parameters:
+ hist   - The histogram context
- viewer - The viewer to view it with

  Level: beginner

.seealso: `PetscDrawHG`, `PetscViewer`, `PetscDrawHGCreate()`, `PetscDrawHGGetDraw()`, `PetscDrawSetSave()`, `PetscDrawSave()`, `PetscDrawHGDraw()`
@*/
PetscErrorCode PetscDrawHGView(PetscDrawHG hist, PetscViewer viewer)
{
  PetscReal xmax, xmin, *bins, *values, *weights, binSize, binLeft, binRight, mean, var, totwt = 0.;
  PetscInt  numBins, numBinsOld, numValues, initSize, i, p;
  PetscBool usewt = PETSC_FALSE;
  int       inumBins;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(hist, PETSC_DRAWHG_CLASSID, 1);

  if ((hist->xmin > hist->xmax) || (hist->ymin >= hist->ymax)) PetscFunctionReturn(PETSC_SUCCESS);
  if (hist->numValues < 1) PetscFunctionReturn(PETSC_SUCCESS);

  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)hist), &viewer));
  PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)hist, viewer));
  xmax      = hist->xmax;
  xmin      = hist->xmin;
  numValues = hist->numValues;
  values    = hist->values;
  weights   = hist->weights;
  mean      = 0.0;
  var       = 0.0;
  if (xmax == xmin) {
    /* Calculate number of points in the bin */
    bins    = hist->bins;
    bins[0] = 0.;
    for (p = 0; p < numValues; p++) {
      if (values[p] == xmin) bins[0] += weights[0];
      mean += values[p] * weights[p];
      var += values[p] * values[p] * weights[p];
      totwt += weights[p];
      if (weights[p] != 1.) usewt = PETSC_TRUE;
    }
    /* Draw bins */
    PetscCall(PetscViewerASCIIPrintf(viewer, "Bin %2d (%6.2g - %6.2g): %.0g\n", 0, (double)xmin, (double)xmax, (double)bins[0]));
  } else {
    numBins    = hist->numBins;
    numBinsOld = hist->numBins;
    if (hist->integerBins && (((int)xmax - xmin) + 1.0e-05 > xmax - xmin)) {
      initSize = (int)((int)xmax - xmin) / numBins;
      while (initSize * numBins != (int)xmax - xmin) {
        initSize = PetscMax(initSize - 1, 1);
        numBins  = (int)((int)xmax - xmin) / initSize;
        PetscCall(PetscCIntCast(numBins, &inumBins));
        PetscCall(PetscDrawHGSetNumberBins(hist, inumBins));
      }
    }
    binSize = (xmax - xmin) / numBins;
    bins    = hist->bins;

    /* Calculate number of points in each bin */
    PetscCall(PetscArrayzero(bins, numBins));
    for (i = 0; i < numBins; i++) {
      binLeft  = xmin + binSize * i;
      binRight = xmin + binSize * (i + 1);
      for (p = 0; p < numValues; p++) {
        if ((values[p] >= binLeft) && (values[p] < binRight)) bins[i] += weights[p];
        /* Handle last bin separately */
        if ((i == numBins - 1) && (values[p] == binRight)) bins[i] += weights[p];
        if (!i) {
          mean += values[p] * weights[p];
          var += values[p] * values[p] * weights[p];
          totwt += weights[p];
          if (weights[p] != 1.) usewt = PETSC_TRUE;
        }
      }
    }
    /* Draw bins */
    for (i = 0; i < numBins; i++) {
      binLeft  = xmin + binSize * i;
      binRight = xmin + binSize * (i + 1);
      PetscCall(PetscViewerASCIIPrintf(viewer, "Bin %2" PetscInt_FMT " (%6.2g - %6.2g): %.0g\n", i, (double)binLeft, (double)binRight, (double)bins[i]));
    }
    PetscCall(PetscCIntCast(numBinsOld, &inumBins));
    PetscCall(PetscDrawHGSetNumberBins(hist, inumBins));
  }

  if (hist->calcStats) {
    if (usewt) {
      mean /= totwt;
      var = (var - totwt * mean * mean) / totwt;

      PetscCall(PetscViewerASCIIPrintf(viewer, "Total Weight: %g\n", (double)totwt));
    } else {
      mean /= numValues;
      if (numValues > 1) var = (var - numValues * mean * mean) / (PetscReal)(numValues - 1);
      else var = 0.0;

      PetscCall(PetscViewerASCIIPrintf(viewer, "Total: %" PetscInt_FMT "\n", numValues));
    }
    PetscCall(PetscViewerASCIIPrintf(viewer, "Mean: %g  Var: %g\n", (double)mean, (double)var));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDrawHGSetColor - Sets the color the bars will be drawn with.

  Logically Collective

  Input Parameters:
+ hist  - The histogram context
- color - one of the colors defined in petscdraw.h or `PETSC_DRAW_ROTATE` to make each bar a different color

  Level: intermediate

.seealso: `PetscDrawHG`, `PetscDrawHGCreate()`, `PetscDrawHGGetDraw()`, `PetscDrawSetSave()`, `PetscDrawSave()`, `PetscDrawHGDraw()`, `PetscDrawHGGetAxis()`
@*/
PetscErrorCode PetscDrawHGSetColor(PetscDrawHG hist, int color)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(hist, PETSC_DRAWHG_CLASSID, 1);

  hist->color = color;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDrawHGSetLimits - Sets the axis limits for a histogram. If more
  points are added after this call, the limits will be adjusted to
  include those additional points.

  Logically Collective

  Input Parameters:
+ hist  - The histogram context
. x_min - the horizontal lower limit
. x_max - the horizontal upper limit
. y_min - the vertical lower limit
- y_max - the vertical upper limit

  Level: intermediate

.seealso: `PetscDrawHG`, `PetscDrawHGCreate()`, `PetscDrawHGGetDraw()`, `PetscDrawSetSave()`, `PetscDrawSave()`, `PetscDrawHGDraw()`, `PetscDrawHGGetAxis()`
@*/
PetscErrorCode PetscDrawHGSetLimits(PetscDrawHG hist, PetscReal x_min, PetscReal x_max, int y_min, int y_max)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(hist, PETSC_DRAWHG_CLASSID, 1);

  hist->xmin = x_min;
  hist->xmax = x_max;
  hist->ymin = y_min;
  hist->ymax = y_max;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDrawHGCalcStats - Turns on calculation of descriptive statistics associated with the histogram

  Not Collective

  Input Parameters:
+ hist - The histogram context
- calc - Flag for calculation

  Level: intermediate

.seealso: `PetscDrawHG`, `PetscDrawHGCreate()`, `PetscDrawHGAddValue()`, `PetscDrawHGView()`, `PetscDrawHGDraw()`
@*/
PetscErrorCode PetscDrawHGCalcStats(PetscDrawHG hist, PetscBool calc)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(hist, PETSC_DRAWHG_CLASSID, 1);

  hist->calcStats = calc;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDrawHGIntegerBins - Turns on integer width bins

  Not Collective

  Input Parameters:
+ hist - The histogram context
- ints - Flag for integer width bins

  Level: intermediate

.seealso: `PetscDrawHG`, `PetscDrawHGCreate()`, `PetscDrawHGAddValue()`, `PetscDrawHGView()`, `PetscDrawHGDraw()`, `PetscDrawHGSetColor()`
@*/
PetscErrorCode PetscDrawHGIntegerBins(PetscDrawHG hist, PetscBool ints)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(hist, PETSC_DRAWHG_CLASSID, 1);

  hist->integerBins = ints;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDrawHGGetAxis - Gets the axis context associated with a histogram.
  This is useful if one wants to change some axis property, such as
  labels, color, etc. The axis context should not be destroyed by the
  application code.

  Not Collective, axis is parallel if hist is parallel

  Input Parameter:
. hist - The histogram context

  Output Parameter:
. axis - The axis context

  Level: intermediate

.seealso: `PetscDrawHG`, `PetscDrawAxis`, `PetscDrawHGCreate()`, `PetscDrawHGAddValue()`, `PetscDrawHGView()`, `PetscDrawHGDraw()`, `PetscDrawHGSetColor()`, `PetscDrawHGSetLimits()`
@*/
PetscErrorCode PetscDrawHGGetAxis(PetscDrawHG hist, PetscDrawAxis *axis)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(hist, PETSC_DRAWHG_CLASSID, 1);
  PetscAssertPointer(axis, 2);
  *axis = hist->axis;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDrawHGGetDraw - Gets the draw context associated with a histogram.

  Not Collective, draw is parallel if hist is parallel

  Input Parameter:
. hist - The histogram context

  Output Parameter:
. draw - The draw context

  Level: intermediate

.seealso: `PetscDraw`, `PetscDrawHG`, `PetscDrawHGCreate()`, `PetscDrawHGAddValue()`, `PetscDrawHGView()`, `PetscDrawHGDraw()`, `PetscDrawHGSetColor()`, `PetscDrawAxis`, `PetscDrawHGSetLimits()`
@*/
PetscErrorCode PetscDrawHGGetDraw(PetscDrawHG hist, PetscDraw *draw)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(hist, PETSC_DRAWHG_CLASSID, 1);
  PetscAssertPointer(draw, 2);
  *draw = hist->win;
  PetscFunctionReturn(PETSC_SUCCESS);
}
