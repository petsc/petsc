#include <../src/sys/classes/draw/utils/axisimpl.h>  /*I   "petscdraw.h"  I*/

PetscClassId PETSC_DRAWAXIS_CLASSID = 0;

/*@
   PetscDrawAxisCreate - Generate the axis data structure.

   Collective on PetscDraw

   Input Parameters:
.  win - PetscDraw object where axis to to be made

   Ouput Parameters:
.  axis - the axis datastructure

   Notes:
    the MPI communicator that owns the underlying draw object owns the PetscDrawAxis object, but calls to set PetscDrawAxis options are ignored by all processes
          except the first MPI process in the communicator

   Level: advanced

.seealso: PetscDrawLGCreate(), PetscDrawLG, PetscDrawSPCreate(), PetscDrawSP, PetscDrawHGCreate(), PetscDrawHG, PetscDrawBarCreate(), PetscDrawBar, PetscDrawLGGetAxis(), PetscDrawSPGetAxis(),
          PetscDrawHGGetAxis(), PetscDrawBarGetAxis(), PetscDrawAxis, PetscDrawAxisDestroy(), PetscDrawAxisSetColors(), PetscDrawAxisSetLabels(), PetscDrawAxisSetLimits(), PetscDrawAxisGetLimits(), PetscDrawAxisSetHoldLimits(),
          PetscDrawAxisDraw()
@*/
PetscErrorCode  PetscDrawAxisCreate(PetscDraw draw,PetscDrawAxis *axis)
{
  PetscDrawAxis  ad;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  PetscValidPointer(axis,2);

  ierr = PetscHeaderCreate(ad,PETSC_DRAWAXIS_CLASSID,"DrawAxis","Draw Axis","Draw",PetscObjectComm((PetscObject)draw),PetscDrawAxisDestroy,NULL);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)draw,(PetscObject)ad);CHKERRQ(ierr);

  ierr = PetscObjectReference((PetscObject)draw);CHKERRQ(ierr);
  ad->win = draw;

  ad->xticks    = PetscADefTicks;
  ad->yticks    = PetscADefTicks;
  ad->xlabelstr = PetscADefLabel;
  ad->ylabelstr = PetscADefLabel;
  ad->ac        = PETSC_DRAW_BLACK;
  ad->tc        = PETSC_DRAW_BLACK;
  ad->cc        = PETSC_DRAW_BLACK;
  ad->xlabel    = NULL;
  ad->ylabel    = NULL;
  ad->toplabel  = NULL;

  *axis = ad;
  PetscFunctionReturn(0);
}

/*@
    PetscDrawAxisDestroy - Frees the space used by an axis structure.

    Collective on PetscDrawAxis

    Input Parameters:
.   axis - the axis context

    Level: advanced

.seealso: PetscDrawAxisCreate(), PetscDrawAxis
@*/
PetscErrorCode  PetscDrawAxisDestroy(PetscDrawAxis *axis)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*axis) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(*axis,PETSC_DRAWAXIS_CLASSID,1);
  if (--((PetscObject)(*axis))->refct > 0) {*axis = NULL; PetscFunctionReturn(0);}

  ierr = PetscFree((*axis)->toplabel);CHKERRQ(ierr);
  ierr = PetscFree((*axis)->xlabel);CHKERRQ(ierr);
  ierr = PetscFree((*axis)->ylabel);CHKERRQ(ierr);
  ierr = PetscDrawDestroy(&(*axis)->win);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(axis);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
    PetscDrawAxisSetColors -  Sets the colors to be used for the axis,
                         tickmarks, and text.

    Logically Collective on PetscDrawAxis

    Input Parameters:
+   axis - the axis
.   ac - the color of the axis lines
.   tc - the color of the tick marks
-   cc - the color of the text strings

    Level: advanced

.seealso: PetscDrawAxisCreate(), PetscDrawAxis, PetscDrawAxisSetLabels(), PetscDrawAxisDraw(), PetscDrawAxisSetLimits()
@*/
PetscErrorCode  PetscDrawAxisSetColors(PetscDrawAxis axis,int ac,int tc,int cc)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(axis,PETSC_DRAWAXIS_CLASSID,1);
  PetscValidLogicalCollectiveInt(axis,ac,2);
  PetscValidLogicalCollectiveInt(axis,tc,3);
  PetscValidLogicalCollectiveInt(axis,cc,4);
  axis->ac = ac; axis->tc = tc; axis->cc = cc;
  PetscFunctionReturn(0);
}

/*@C
    PetscDrawAxisSetLabels -  Sets the x and y axis labels.

    Logically Collective on PetscDrawAxis

    Input Parameters:
+   axis - the axis
.   top - the label at the top of the image
-   xlabel,ylabel - the labes for the x and y axis

    Notes:
    Must be called before PetscDrawAxisDraw() or PetscDrawLGDraw()
           There should be no newlines in the arguments

    Level: advanced

.seealso: PetscDrawAxisCreate(), PetscDrawAxis, PetscDrawAxisSetColors(), PetscDrawAxisDraw(), PetscDrawAxisSetLimits()
@*/
PetscErrorCode  PetscDrawAxisSetLabels(PetscDrawAxis axis,const char top[],const char xlabel[],const char ylabel[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(axis,PETSC_DRAWAXIS_CLASSID,1);
  ierr = PetscFree(axis->xlabel);CHKERRQ(ierr);
  ierr = PetscFree(axis->ylabel);CHKERRQ(ierr);
  ierr = PetscFree(axis->toplabel);CHKERRQ(ierr);
  ierr = PetscStrallocpy(xlabel,&axis->xlabel);CHKERRQ(ierr);
  ierr = PetscStrallocpy(ylabel,&axis->ylabel);CHKERRQ(ierr);
  ierr = PetscStrallocpy(top,&axis->toplabel);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
    PetscDrawAxisSetLimits -  Sets the limits (in user coords) of the axis

    Logically Collective on PetscDrawAxis

    Input Parameters:
+   axis - the axis
.   xmin,xmax - limits in x
-   ymin,ymax - limits in y

    Options Database:
.   -drawaxis_hold - hold the initial set of axis limits for future plotting

    Level: advanced

.seealso:  PetscDrawAxisSetHoldLimits(), PetscDrawAxisGetLimits(), PetscDrawAxisSetLabels(), PetscDrawAxisSetColors()

@*/
PetscErrorCode  PetscDrawAxisSetLimits(PetscDrawAxis axis,PetscReal xmin,PetscReal xmax,PetscReal ymin,PetscReal ymax)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(axis,PETSC_DRAWAXIS_CLASSID,1);
  if (axis->hold) PetscFunctionReturn(0);
  axis->xlow = xmin;
  axis->xhigh= xmax;
  axis->ylow = ymin;
  axis->yhigh= ymax;
  ierr = PetscOptionsHasName(((PetscObject)axis)->options,((PetscObject)axis)->prefix,"-drawaxis_hold",&axis->hold);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
    PetscDrawAxisGetLimits -  Gets the limits (in user coords) of the axis

    Not Collective

    Input Parameters:
+   axis - the axis
.   xmin,xmax - limits in x
-   ymin,ymax - limits in y

    Level: advanced

.seealso:  PetscDrawAxisCreate(), PetscDrawAxis, PetscDrawAxisSetHoldLimits(), PetscDrawAxisSetLimits(), PetscDrawAxisSetLabels(), PetscDrawAxisSetColors()

@*/
PetscErrorCode  PetscDrawAxisGetLimits(PetscDrawAxis axis,PetscReal *xmin,PetscReal *xmax,PetscReal *ymin,PetscReal *ymax)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(axis,PETSC_DRAWAXIS_CLASSID,1);
  if (xmin) *xmin = axis->xlow;
  if (xmax) *xmax = axis->xhigh;
  if (ymin) *ymin = axis->ylow;
  if (ymax) *ymax = axis->yhigh;
  PetscFunctionReturn(0);
}

/*@
    PetscDrawAxisSetHoldLimits -  Causes an axis to keep the same limits until this is called
        again

    Logically Collective on PetscDrawAxis

    Input Parameters:
+   axis - the axis
-   hold - PETSC_TRUE - hold current limits, PETSC_FALSE allow limits to be changed

    Level: advanced

    Notes:
        Once this has been called with PETSC_TRUE the limits will not change if you call
     PetscDrawAxisSetLimits() until you call this with PETSC_FALSE

.seealso:  PetscDrawAxisCreate(), PetscDrawAxis, PetscDrawAxisGetLimits(), PetscDrawAxisSetLimits(), PetscDrawAxisSetLabels(), PetscDrawAxisSetColors()

@*/
PetscErrorCode  PetscDrawAxisSetHoldLimits(PetscDrawAxis axis,PetscBool hold)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(axis,PETSC_DRAWAXIS_CLASSID,1);
  PetscValidLogicalCollectiveBool(axis,hold,2);
  axis->hold = hold;
  PetscFunctionReturn(0);
}

/*@
    PetscDrawAxisDraw - PetscDraws an axis.

    Collective on PetscDrawAxis

    Input Parameter:
.   axis - Axis structure

    Level: advanced

    Note:
    This draws the actual axis.  The limits etc have already been set.
    By picking special routines for the ticks and labels, special
    effects may be generated.  These routines are part of the Axis
    structure (axis).

.seealso:  PetscDrawAxisCreate(), PetscDrawAxis, PetscDrawAxisGetLimits(), PetscDrawAxisSetLimits(), PetscDrawAxisSetLabels(), PetscDrawAxisSetColors()

@*/
PetscErrorCode  PetscDrawAxisDraw(PetscDrawAxis axis)
{
  int            i,ntick,numx,numy,ac,tc,cc;
  PetscMPIInt    rank;
  size_t         len,ytlen=0;
  PetscReal      coors[4],tickloc[MAXSEGS],sep,tw,th;
  PetscReal      xl,xr,yl,yr,dxl=0,dyl=0,dxr=0,dyr=0;
  char           *p;
  PetscDraw      draw;
  PetscBool      isnull;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(axis,PETSC_DRAWAXIS_CLASSID,1);
  ierr = PetscDrawIsNull(axis->win,&isnull);CHKERRQ(ierr);
  if (isnull) PetscFunctionReturn(0);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)axis),&rank);CHKERRQ(ierr);

  draw = axis->win;

  ac = axis->ac; tc = axis->tc; cc = axis->cc;
  if (axis->xlow == axis->xhigh) {axis->xlow -= .5; axis->xhigh += .5;}
  if (axis->ylow == axis->yhigh) {axis->ylow -= .5; axis->yhigh += .5;}

  ierr = PetscDrawCollectiveBegin(draw);CHKERRQ(ierr);
  if (rank) goto finally;

  /* get cannonical string size */
  ierr = PetscDrawSetCoordinates(draw,0,0,1,1);CHKERRQ(ierr);
  ierr = PetscDrawStringGetSize(draw,&tw,&th);CHKERRQ(ierr);
  /* lower spacing */
  if (axis->xlabelstr) dyl += 1.5*th;
  if (axis->xlabel)    dyl += 1.5*th;
  /* left spacing */
  if (axis->ylabelstr) dxl += 7.5*tw;
  if (axis->ylabel)    dxl += 2.0*tw;
  /* right and top spacing */
  if (axis->xlabelstr) dxr = 2.5*tw;
  if (axis->ylabelstr) dyr = 0.5*th;
  if (axis->toplabel)  dyr = 1.5*th;
  /* extra spacing */
  dxl += 0.7*tw; dxr += 0.5*tw;
  dyl += 0.2*th; dyr += 0.2*th;
  /* determine coordinates */
  xl = (dxl*axis->xhigh + dxr*axis->xlow - axis->xlow)  / (dxl + dxr - 1);
  xr = (dxl*axis->xhigh + dxr*axis->xlow - axis->xhigh) / (dxl + dxr - 1);
  yl = (dyl*axis->yhigh + dyr*axis->ylow - axis->ylow)  / (dyl + dyr - 1);
  yr = (dyl*axis->yhigh + dyr*axis->ylow - axis->yhigh) / (dyl + dyr - 1);
  ierr = PetscDrawSetCoordinates(draw,xl,yl,xr,yr);CHKERRQ(ierr);
  ierr = PetscDrawStringGetSize(draw,&tw,&th);CHKERRQ(ierr);

  /* PetscDraw the axis lines */
  ierr = PetscDrawLine(draw,axis->xlow,axis->ylow,axis->xhigh,axis->ylow,ac);CHKERRQ(ierr);
  ierr = PetscDrawLine(draw,axis->xlow,axis->ylow,axis->xlow,axis->yhigh,ac);CHKERRQ(ierr);
  ierr = PetscDrawLine(draw,axis->xlow,axis->yhigh,axis->xhigh,axis->yhigh,ac);CHKERRQ(ierr);
  ierr = PetscDrawLine(draw,axis->xhigh,axis->ylow,axis->xhigh,axis->yhigh,ac);CHKERRQ(ierr);

  /* PetscDraw the top label */
  if (axis->toplabel) {
    PetscReal x = (axis->xlow + axis->xhigh)/2, y = axis->yhigh + 0.5*th;
    ierr = PetscDrawStringCentered(draw,x,y,cc,axis->toplabel);CHKERRQ(ierr);
  }

  /* PetscDraw the X ticks and labels */
  if (axis->xticks) {
    numx = (int)(.15*(axis->xhigh-axis->xlow)/tw); numx = PetscClipInterval(numx,2,6);
    ierr = (*axis->xticks)(axis->xlow,axis->xhigh,numx,&ntick,tickloc,MAXSEGS);CHKERRQ(ierr);
    /* PetscDraw in tick marks */
    for (i=0; i<ntick; i++) {
      ierr = PetscDrawLine(draw,tickloc[i],axis->ylow,tickloc[i],axis->ylow+.5*th,tc);CHKERRQ(ierr);
      ierr = PetscDrawLine(draw,tickloc[i],axis->yhigh,tickloc[i],axis->yhigh-.5*th,tc);CHKERRQ(ierr);
    }
    /* label ticks */
    if (axis->xlabelstr) {
      for (i=0; i<ntick; i++) {
        if (i < ntick - 1) sep = tickloc[i+1] - tickloc[i];
        else if (i > 0)    sep = tickloc[i]   - tickloc[i-1];
        else               sep = 0.0;
        ierr = (*axis->xlabelstr)(tickloc[i],sep,&p);CHKERRQ(ierr);
        ierr = PetscDrawStringCentered(draw,tickloc[i],axis->ylow-1.5*th,cc,p);CHKERRQ(ierr);
      }
    }
  }
  if (axis->xlabel) {
    PetscReal x = (axis->xlow + axis->xhigh)/2, y = axis->ylow - 1.5*th;
    if (axis->xlabelstr) y -= 1.5*th;
    ierr = PetscDrawStringCentered(draw,x,y,cc,axis->xlabel);CHKERRQ(ierr);
  }

  /* PetscDraw the Y ticks and labels */
  if (axis->yticks) {
    numy = (int)(.50*(axis->yhigh-axis->ylow)/th); numy = PetscClipInterval(numy,2,6);
    ierr = (*axis->yticks)(axis->ylow,axis->yhigh,numy,&ntick,tickloc,MAXSEGS);CHKERRQ(ierr);
    /* PetscDraw in tick marks */
    for (i=0; i<ntick; i++) {
      ierr = PetscDrawLine(draw,axis->xlow,tickloc[i],axis->xlow+.5*tw,tickloc[i],tc);CHKERRQ(ierr);
      ierr = PetscDrawLine(draw,axis->xhigh,tickloc[i],axis->xhigh-.5*tw,tickloc[i],tc);CHKERRQ(ierr);
    }
    /* label ticks */
    if (axis->ylabelstr) {
      for (i=0; i<ntick; i++) {
        if (i < ntick - 1) sep = tickloc[i+1] - tickloc[i];
        else if (i > 0)    sep = tickloc[i]   - tickloc[i-1];
        else               sep = 0.0;
        ierr = (*axis->ylabelstr)(tickloc[i],sep,&p);CHKERRQ(ierr);
        ierr = PetscStrlen(p,&len);CHKERRQ(ierr); ytlen = PetscMax(ytlen,len);
        ierr = PetscDrawString(draw,axis->xlow-(len+.5)*tw,tickloc[i]-.5*th,cc,p);CHKERRQ(ierr);
      }
    }
  }
  if (axis->ylabel) {
    PetscReal x = axis->xlow - 2.0*tw, y = (axis->ylow + axis->yhigh)/2;
    if (axis->ylabelstr) x -= (ytlen+.5)*tw;
    ierr = PetscStrlen(axis->ylabel,&len);CHKERRQ(ierr);
    ierr = PetscDrawStringVertical(draw,x,y+len*th/2,cc,axis->ylabel);CHKERRQ(ierr);
  }

  ierr = PetscDrawGetCoordinates(draw,&coors[0],&coors[1],&coors[2],&coors[3]);CHKERRQ(ierr);
finally:
  ierr = PetscDrawCollectiveEnd(draw);CHKERRQ(ierr);
  ierr = MPI_Bcast(coors,4,MPIU_REAL,0,PetscObjectComm((PetscObject)draw));CHKERRQ(ierr);
  ierr = PetscDrawSetCoordinates(draw,coors[0],coors[1],coors[2],coors[3]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
    Removes all zeros but one from .0000
*/
PetscErrorCode PetscStripe0(char *buf)
{
  PetscErrorCode ierr;
  size_t         n;
  PetscBool      flg;
  char           *str;

  PetscFunctionBegin;
  ierr = PetscStrlen(buf,&n);CHKERRQ(ierr);
  ierr = PetscStrendswith(buf,"e00",&flg);CHKERRQ(ierr);
  if (flg) buf[n-3] = 0;
  ierr = PetscStrstr(buf,"e0",&str);CHKERRQ(ierr);
  if (str) {
    buf[n-2] = buf[n-1];
    buf[n-1] = 0;
  }
  ierr = PetscStrstr(buf,"e-0",&str);CHKERRQ(ierr);
  if (str) {
    buf[n-2] = buf[n-1];
    buf[n-1] = 0;
  }
  PetscFunctionReturn(0);
}

/*
    Removes all zeros but one from .0000
*/
PetscErrorCode PetscStripAllZeros(char *buf)
{
  PetscErrorCode ierr;
  size_t         i,n;

  PetscFunctionBegin;
  ierr = PetscStrlen(buf,&n);CHKERRQ(ierr);
  if (buf[0] != '.') PetscFunctionReturn(0);
  for (i=1; i<n; i++) {
    if (buf[i] != '0') PetscFunctionReturn(0);
  }
  buf[0] = '0';
  buf[1] = 0;
  PetscFunctionReturn(0);
}

/*
    Removes trailing zeros
*/
PetscErrorCode PetscStripTrailingZeros(char *buf)
{
  PetscErrorCode ierr;
  char           *found;
  size_t         i,n,m = PETSC_MAX_INT;

  PetscFunctionBegin;
  /* if there is an e in string DO NOT strip trailing zeros */
  ierr = PetscStrchr(buf,'e',&found);CHKERRQ(ierr);
  if (found) PetscFunctionReturn(0);

  ierr = PetscStrlen(buf,&n);CHKERRQ(ierr);
  /* locate decimal point */
  for (i=0; i<n; i++) {
    if (buf[i] == '.') {m = i; break;}
  }
  /* if not decimal point then no zeros to remove */
  if (m == PETSC_MAX_INT) PetscFunctionReturn(0);
  /* start at right end of string removing 0s */
  for (i=n-1; i>m; i++) {
    if (buf[i] != '0') PetscFunctionReturn(0);
    buf[i] = 0;
  }
  PetscFunctionReturn(0);
}

/*
    Removes leading 0 from 0.22 or -0.22
*/
PetscErrorCode PetscStripInitialZero(char *buf)
{
  PetscErrorCode ierr;
  size_t         i,n;

  PetscFunctionBegin;
  ierr = PetscStrlen(buf,&n);CHKERRQ(ierr);
  if (buf[0] == '0') {
    for (i=0; i<n; i++) buf[i] = buf[i+1];
  } else if (buf[0] == '-' && buf[1] == '0') {
    for (i=1; i<n; i++) buf[i] = buf[i+1];
  }
  PetscFunctionReturn(0);
}

/*
     Removes the extraneous zeros in numbers like 1.10000e6
*/
PetscErrorCode PetscStripZeros(char *buf)
{
  PetscErrorCode ierr;
  size_t         i,j,n;

  PetscFunctionBegin;
  ierr = PetscStrlen(buf,&n);CHKERRQ(ierr);
  if (n<5) PetscFunctionReturn(0);
  for (i=1; i<n-1; i++) {
    if (buf[i] == 'e' && buf[i-1] == '0') {
      for (j=i; j<n+1; j++) buf[j-1] = buf[j];
      ierr = PetscStripZeros(buf);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
  }
  PetscFunctionReturn(0);
}

/*
      Removes the plus in something like 1.1e+2 or 1.1e+02
*/
PetscErrorCode PetscStripZerosPlus(char *buf)
{
  PetscErrorCode ierr;
  size_t         i,j,n;

  PetscFunctionBegin;
  ierr = PetscStrlen(buf,&n);CHKERRQ(ierr);
  if (n<5) PetscFunctionReturn(0);
  for (i=1; i<n-2; i++) {
    if (buf[i] == '+') {
      if (buf[i+1] == '0') {
        for (j=i+1; j<n; j++) buf[j-1] = buf[j+1];
        PetscFunctionReturn(0);
      } else {
        for (j=i+1; j<n+1; j++) buf[j-1] = buf[j];
        PetscFunctionReturn(0);
      }
    } else if (buf[i] == '-') {
      if (buf[i+1] == '0') {
        for (j=i+1; j<n; j++) buf[j] = buf[j+1];
        PetscFunctionReturn(0);
      }
    }
  }
  PetscFunctionReturn(0);
}
