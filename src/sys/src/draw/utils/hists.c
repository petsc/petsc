/*$Id: hists.c,v 1.21 2000/05/05 22:13:45 balay Exp bsmith $*/

/*
  Contains the data structure for plotting a histogram in a window with an axis.
*/

#include "petsc.h"         /*I "petsc.h" I*/

struct _p_DrawHG {
  PETSCHEADER(int) 
  int       (*destroy)(DrawSP);
  int       (*view)(DrawSP,Viewer);
  Draw      win;
  DrawAxis  axis;
  PetscReal xmin,xmax;
  PetscReal ymin,ymax;
  int       numBins;
  PetscReal *bins;
  int       numValues;
  int       maxValues;
  PetscReal *values;
  int       color;
};

#define CHUNKSIZE 100

#undef __FUNC__  
#define __FUNC__ /*<a name="DrawHGCreate"></a>*/"DrawHGCreate" 
/*@C
   DrawHGCreate - Creates a histogram data structure.

   Collective over Draw

   Input Parameters:
+  draw  - The window where the graph will be made
-  bins - The number of bins to use

   Output Parameters:
.  hist - The histogram context

   Level: intermediate

   Contributed by: Matthew Knepley

   Concepts: histogram^creating

.seealso: DrawHGDestroy()

@*/
int DrawHGCreate(Draw draw,int bins,DrawHG *hist)
{
  int         ierr;
  PetscTruth  isnull;
  PetscObject obj = (PetscObject)draw;
  DrawHG      h;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  PetscValidPointer(hist);
  ierr = PetscTypeCompare(obj,DRAW_NULL,&isnull);CHKERRQ(ierr);
  if (isnull) {
    ierr = DrawOpenNull(obj->comm,(Draw*)hist);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  PetscHeaderCreate(h,_p_DrawHG,int,DRAWHG_COOKIE,0,"DrawHG",obj->comm,DrawHGDestroy,0);
  h->view      = 0;
  h->destroy   = 0;
  h->win       = draw;
  h->color     = DRAW_GREEN;
  h->xmin      = PETSC_MAX;
  h->xmax      = PETSC_MIN;
  h->ymin      = 0.;
  h->ymax      = 1.;
  h->numBins   = bins;
  h->bins      = (PetscReal*)PetscMalloc(bins*sizeof(PetscReal));CHKPTRQ(h->bins);
  h->numValues = 0;
  h->maxValues = CHUNKSIZE;
  h->values    = (PetscReal*)PetscMalloc(h->maxValues * sizeof(PetscReal));CHKPTRQ(h->values);
  PLogObjectMemory(h,bins*sizeof(PetscReal) + h->maxValues*sizeof(PetscReal));
  ierr = DrawAxisCreate(draw,&h->axis);CHKERRQ(ierr);
  PLogObjectParent(h,h->axis);
  *hist = h;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="DrawHGSetNumberBins"></a>*/"DrawHGSetNumberBins" 
/*@
   DrawHGSetNumberBins - Change the number of bins that are to be drawn.

   Not Collective (ignored except on processor 0 of DrawHG)

   Input Parameter:
+  hist - The histogram context.
-  dim  - The number of curves.

   Level: intermediate

  Contributed by: Matthew Knepley

   Concepts: histogram^setting number of bins

@*/
int DrawHGSetNumberBins(DrawHG hist,int bins)
{
  int ierr;

  PetscFunctionBegin;
  if (hist && hist->cookie == DRAW_COOKIE) PetscFunctionReturn(0);

  PetscValidHeaderSpecific(hist,DRAWHG_COOKIE);
  if (hist->numBins == bins) PetscFunctionReturn(0);

  ierr          = PetscFree(hist->bins);CHKERRQ(ierr);
  hist->bins    = (PetscReal*)PetscMalloc(bins*sizeof(PetscReal));CHKPTRQ(hist->bins);
  PLogObjectMemory(hist,(bins - hist->numBins) * sizeof(PetscReal));
  hist->numBins = bins;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="DrawHGReset"></a>*/"DrawHGReset" 
/*@
  DrawHGReset - Clears histogram to allow for reuse with new data.

  Not Collective (ignored except on processor 0 of DrawHG)

  Input Parameter:
. hist - The histogram context.

   Level: intermediate

  Contributed by: Matthew Knepley

   Concepts: histogram^resetting

@*/
int DrawHGReset(DrawHG hist)
{
  PetscFunctionBegin;
  if (hist && hist->cookie == DRAW_COOKIE) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(hist,DRAWHG_COOKIE);
  hist->xmin      = PETSC_MAX;
  hist->xmax      = PETSC_MIN;
  hist->ymin      = 0;
  hist->ymax      = 0;
  hist->numValues = 0;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="DrawHGDestroy"></a>*/"DrawHGDestroy" 
/*@C
  DrawHGDestroy - Frees all space taken up by histogram data structure.

  Collective over DrawHG

  Input Parameter:
. hist - The histogram context

   Level: intermediate

  Contributed by: Matthew Knepley

.seealso:  DrawHGCreate()
@*/
int DrawHGDestroy(DrawHG hist)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeader(hist);

  if (--hist->refct > 0) PetscFunctionReturn(0);
  if (hist->cookie == DRAW_COOKIE){
    ierr = DrawDestroy((Draw) hist);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  ierr = DrawAxisDestroy(hist->axis);CHKERRQ(ierr);
  ierr = PetscFree(hist->bins);CHKERRQ(ierr);
  ierr = PetscFree(hist->values);CHKERRQ(ierr);
  PLogObjectDestroy(hist);
  PetscHeaderDestroy(hist);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="DrawHGAddValue"></a>*/"DrawHGAddValue" 
/*@
  DrawHGAddValue - Adds another value to the histogram.

  Not Collective (ignored except on processor 0 of DrawHG)

  Input Parameters:
+ hist  - The histogram
- value - The value 

   Level: intermediate

  Contributed by: Matthew Knepley

  Concepts: histogram^adding values

.seealso: DrawHGAddValues()
@*/
int DrawHGAddValue(DrawHG hist,PetscReal value)
{
  PetscFunctionBegin;
  if (hist && hist->cookie == DRAW_COOKIE) PetscFunctionReturn(0);

  PetscValidHeaderSpecific(hist,DRAWHG_COOKIE);
  /* Allocate more memory if necessary */
  if (hist->numValues >= hist->maxValues) {
    PetscReal *tmp;
    int     ierr;

    tmp = (PetscReal*)PetscMalloc((hist->maxValues + CHUNKSIZE) * sizeof(PetscReal));CHKPTRQ(tmp);
    PLogObjectMemory(hist,CHUNKSIZE * sizeof(PetscReal));
    ierr = PetscMemcpy(tmp,hist->values,hist->maxValues * sizeof(PetscReal));CHKERRQ(ierr);
    ierr = PetscFree(hist->values);CHKERRQ(ierr);
    hist->values     = tmp;
    hist->maxValues += CHUNKSIZE;
  }
  if (!hist->numValues) {
    hist->xmin = value;
    hist->xmax = value;
  } else if (hist->numValues == 1) {
    /* Update limits -- We need to overshoot the largest value somewhat */
    if (value > hist->xmax)
      hist->xmax = value + 0.001*(value - hist->xmin)/hist->numBins;
    if (value < hist->xmin)
    {
      hist->xmin = value;
      hist->xmax = hist->xmax + 0.001*(hist->xmax - hist->xmin)/hist->numBins;
    }
  } else {
    /* Update limits -- We need to overshoot the largest value somewhat */
    if (value > hist->xmax) {
      hist->xmax = value + 0.001*(hist->xmax - hist->xmin)/hist->numBins;
    }
    if (value < hist->xmin) {
      hist->xmin = value;
    }
  }

  hist->values[hist->numValues++] = value;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="DrawHGDraw"></a>*/"DrawHGDraw" 
/*@
  DrawHGDraw - Redraws a histogram.

  Not Collective (ignored except on processor 0 of DrawHG)

  Input Parameter:
. hist - The histogram context

   Level: intermediate

  Contributed by: Matthew Knepley

@*/
int DrawHGDraw(DrawHG hist)
{
  Draw     draw;
  PetscReal   xmin,xmax,ymin,ymax,*bins,*values,binSize,binLeft,binRight,maxHeight;
  int      numBins,numValues,i,p,ierr,bcolor,color,rank;

  PetscFunctionBegin;
  if (hist && hist->cookie == DRAW_COOKIE) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(hist,DRAWHG_COOKIE);
  if ((hist->xmin >= hist->xmax) || (hist->ymin >= hist->ymax)) PetscFunctionReturn(0);
  if (hist->numValues < 1) PetscFunctionReturn(0);

  ierr = MPI_Comm_rank(hist->comm,&rank);CHKERRQ(ierr);
  if (rank) PetscFunctionReturn(0);

  color = hist->color; 
  if (color == DRAW_ROTATE) {bcolor = 2;} else {bcolor = color;}
  draw       = hist->win;
  xmin      = hist->xmin;
  xmax      = hist->xmax;
  ymin      = hist->ymin;
  ymax      = hist->ymax;
  numBins   = hist->numBins;
  bins      = hist->bins;
  numValues = hist->numValues;
  values    = hist->values;
  binSize   = (xmax - xmin)/numBins;

  ierr = DrawClear(draw);CHKERRQ(ierr);
  /* Calculate number of points in each bin */
  ierr = PetscMemzero(bins,numBins * sizeof(PetscReal));CHKERRQ(ierr);
  maxHeight = 0;
  for (i = 0; i < numBins; i++) {
    binLeft   = xmin + binSize*i;
    binRight  = xmin + binSize*(i+1);
    for(p = 0; p < numValues; p++) {
      if ((values[p] >= binLeft) && (values[p] < binRight)) bins[i]++;
    }
    maxHeight = PetscMax(maxHeight,bins[i]);
  }
  if (maxHeight > ymax) ymax = hist->ymax = maxHeight;
  ierr = DrawAxisSetLimits(hist->axis,xmin,xmax,ymin,ymax);CHKERRQ(ierr);
  ierr = DrawAxisDraw(hist->axis);CHKERRQ(ierr);
  /* Draw bins */
  for (i = 0; i < numBins; i++) {
    binLeft   = xmin + binSize*i;
    binRight  = xmin + binSize*(i+1);
    ierr = DrawRectangle(draw,binLeft,ymin,binRight,bins[i],bcolor,bcolor,bcolor,bcolor);CHKERRQ(ierr);
    if (color == DRAW_ROTATE && bins[i]) bcolor++; if (bcolor > 31) bcolor = 2;
    ierr = DrawLine(draw,binLeft,ymin,binLeft,bins[i],DRAW_BLACK);CHKERRQ(ierr);
    ierr = DrawLine(draw,binRight,ymin,binRight,bins[i],DRAW_BLACK);CHKERRQ(ierr);
    ierr = DrawLine(draw,binLeft,bins[i],binRight,bins[i],DRAW_BLACK);CHKERRQ(ierr);
  }
  ierr = DrawFlush(draw);CHKERRQ(ierr);
  ierr = DrawPause(draw);CHKERRQ(ierr);
  PetscFunctionReturn(0);
} 
 
#undef __FUNC__  
#define __FUNC__ /*<a name="DrawHGSetColor"></a>*/"DrawHGSetColor" 
/*@
  DrawHGSetColor - Sets the color the bars will be drawn with.

  Not Collective (ignored except on processor 0 of DrawHG)

  Input Parameters:
+ hist - The histogram context
- color - one of the colors defined in petscdraw.h or DRAW_ROTATE to make each bar a 
          different color

  Level: intermediate

@*/
int DrawHGSetColor(DrawHG hist,int color)
{
  PetscFunctionBegin;
  if (hist && hist->cookie == DRAW_COOKIE) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(hist,DRAWHG_COOKIE);
  hist->color = color;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="DrawHGSetLimits"></a>*/"DrawHGSetLimits" 
/*@
  DrawHGSetLimits - Sets the axis limits for a histogram. If more
  points are added after this call, the limits will be adjusted to
  include those additional points.

  Not Collective (ignored except on processor 0 of DrawHG)

  Input Parameters:
+ hist - The histogram context
- x_min,x_max,y_min,y_max - The limits

  Level: intermediate

  Contributed by: Matthew Knepley

  Concepts: histogram^setting axis

@*/
int DrawHGSetLimits(DrawHG hist,PetscReal x_min,PetscReal x_max,int y_min,int y_max) 
{
  PetscFunctionBegin;
  if (hist && hist->cookie == DRAW_COOKIE) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(hist,DRAWHG_COOKIE);
  hist->xmin = x_min; 
  hist->xmax = x_max; 
  hist->ymin = y_min; 
  hist->ymax = y_max;
  PetscFunctionReturn(0);
}
 
#undef __FUNC__  
#define __FUNC__ /*<a name="DrawHGGetAxis"></a>*/"DrawHGGetAxis" 
/*@C
  DrawHGGetAxis - Gets the axis context associated with a histogram.
  This is useful if one wants to change some axis property, such as
  labels, color, etc. The axis context should not be destroyed by the
  application code.

  Not Collective (ignored except on processor 0 of DrawHG)

  Input Parameter:
. hist - The histogram context

  Output Parameter:
. axis - The axis context

  Level: intermediate

  Contributed by: Matthew Knepley

@*/
int DrawHGGetAxis(DrawHG hist,DrawAxis *axis)
{
  PetscFunctionBegin;
  if (hist && hist->cookie == DRAW_COOKIE) {
    *axis = 0;
    PetscFunctionReturn(0);
  }
  PetscValidHeaderSpecific(hist,DRAWHG_COOKIE);
  *axis = hist->axis;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="DrawHGGetDraw"></a>*/"DrawHGGetDraw" 
/*@C
  DrawHGGetDraw - Gets the draw context associated with a histogram.

  Not Collective, Draw is parallel if DrawHG is parallel

  Input Parameter:
. hist - The histogram context

  Output Parameter:
. win  - The draw context

  Level: intermediate

  Contributed by: Matthew Knepley

@*/
int DrawHGGetDraw(DrawHG hist,Draw *win)
{
  PetscFunctionBegin;
  PetscValidHeader(hist);
  if (hist && hist->cookie == DRAW_COOKIE) {
    *win = (Draw)hist;
  } else {
    *win = hist->win;
  }
  PetscFunctionReturn(0);
}

