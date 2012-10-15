#include <../src/sys/draw/utils/axisimpl.h>

PetscClassId PETSC_DRAWAXIS_CLASSID = 0;

#undef __FUNCT__
#define __FUNCT__ "PetscDrawAxisCreate"
/*@
   PetscDrawAxisCreate - Generate the axis data structure.

   Collective over PetscDraw

   Input Parameters:
.  win - PetscDraw object where axis to to be made

   Ouput Parameters:
.  axis - the axis datastructure

   Level: advanced

@*/
PetscErrorCode  PetscDrawAxisCreate(PetscDraw draw,PetscDrawAxis *axis)
{
  PetscDrawAxis  ad;
  PetscObject    obj = (PetscObject)draw;
  PetscErrorCode ierr;
  PetscBool      isnull;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  PetscValidPointer(axis,2);
  ierr = PetscObjectTypeCompare(obj,PETSC_DRAW_NULL,&isnull);CHKERRQ(ierr);
  if (isnull) {
    ierr = PetscDrawOpenNull(((PetscObject)obj)->comm,(PetscDraw*)axis);CHKERRQ(ierr);
    (*axis)->win = draw;
    PetscFunctionReturn(0);
  }
  ierr = PetscHeaderCreate(ad,_p_PetscDrawAxis,int,PETSC_DRAWAXIS_CLASSID,0,"PetscDrawAxis","Draw Axis","Draw",((PetscObject)obj)->comm,PetscDrawAxisDestroy,0);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(draw,ad);CHKERRQ(ierr);
  ad->xticks    = PetscADefTicks;
  ad->yticks    = PetscADefTicks;
  ad->xlabelstr = PetscADefLabel;
  ad->ylabelstr = PetscADefLabel;
  ad->win       = draw;
  ad->ac        = PETSC_DRAW_BLACK;
  ad->tc        = PETSC_DRAW_BLACK;
  ad->cc        = PETSC_DRAW_BLACK;
  ad->xlabel    = 0;
  ad->ylabel    = 0;
  ad->toplabel  = 0;

  *axis = ad;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDrawAxisDestroy"
/*@
    PetscDrawAxisDestroy - Frees the space used by an axis structure.

    Collective over PetscDrawAxis

    Input Parameters:
.   axis - the axis context

    Level: advanced

@*/
PetscErrorCode  PetscDrawAxisDestroy(PetscDrawAxis *axis)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*axis) PetscFunctionReturn(0);
  if (--((PetscObject)(*axis))->refct > 0) PetscFunctionReturn(0);

  ierr = PetscFree((*axis)->toplabel);CHKERRQ(ierr);
  ierr = PetscFree((*axis)->xlabel);CHKERRQ(ierr);
  ierr = PetscFree((*axis)->ylabel);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(axis);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDrawAxisSetColors"
/*@
    PetscDrawAxisSetColors -  Sets the colors to be used for the axis,
                         tickmarks, and text.

    Not Collective (ignored on all processors except processor 0 of PetscDrawAxis)

    Input Parameters:
+   axis - the axis
.   ac - the color of the axis lines
.   tc - the color of the tick marks
-   cc - the color of the text strings

    Level: advanced

@*/
PetscErrorCode  PetscDrawAxisSetColors(PetscDrawAxis axis,int ac,int tc,int cc)
{
  PetscFunctionBegin;
  if (!axis) PetscFunctionReturn(0);
  axis->ac = ac; axis->tc = tc; axis->cc = cc;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDrawAxisSetLabels"
/*@C
    PetscDrawAxisSetLabels -  Sets the x and y axis labels.

    Not Collective (ignored on all processors except processor 0 of PetscDrawAxis)

    Input Parameters:
+   axis - the axis
.   top - the label at the top of the image
-   xlabel,ylabel - the labes for the x and y axis

    Notes: Must be called before PetscDrawAxisDraw() or PetscDrawLGDraw()
           There should be no newlines in the arguments

    Level: advanced

@*/
PetscErrorCode  PetscDrawAxisSetLabels(PetscDrawAxis axis,const char top[],const char xlabel[],const char ylabel[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!axis) PetscFunctionReturn(0);
  ierr = PetscFree(axis->xlabel);CHKERRQ(ierr);
  ierr = PetscFree(axis->ylabel);CHKERRQ(ierr);
  ierr = PetscFree(axis->toplabel);CHKERRQ(ierr);
  ierr = PetscStrallocpy(xlabel,&axis->xlabel);CHKERRQ(ierr);
  ierr = PetscStrallocpy(ylabel,&axis->ylabel);CHKERRQ(ierr);
  ierr = PetscStrallocpy(top,&axis->toplabel);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDrawAxisSetHoldLimits"
/*@
    PetscDrawAxisSetHoldLimits -  Causes an axis to keep the same limits until this is called
        again

    Not Collective (ignored on all processors except processor 0 of PetscDrawAxis)

    Input Parameters:
+   axis - the axis
-   hold - PETSC_TRUE - hold current limits, PETSC_FALSE allow limits to be changed

    Level: advanced

    Notes:
        Once this has been called with PETSC_TRUE the limits will not change if you call
     PetscDrawAxisSetLimits() until you call this with PETSC_FALSE

.seealso:  PetscDrawAxisSetLimits()

@*/
PetscErrorCode  PetscDrawAxisSetHoldLimits(PetscDrawAxis axis,PetscBool  hold)
{
  PetscFunctionBegin;
  if (!axis) PetscFunctionReturn(0);
  axis->hold = hold;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDrawAxisDraw"
/*@
    PetscDrawAxisDraw - PetscDraws an axis.

    Not Collective (ignored on all processors except processor 0 of PetscDrawAxis)

    Input Parameter:
.   axis - Axis structure

    Level: advanced

    Note:
    This draws the actual axis.  The limits etc have already been set.
    By picking special routines for the ticks and labels, special
    effects may be generated.  These routines are part of the Axis
    structure (axis).
@*/
PetscErrorCode  PetscDrawAxisDraw(PetscDrawAxis axis)
{
  int            i,ntick,numx,numy,ac = axis->ac,tc = axis->tc,cc = axis->cc,rank;
  size_t         len;
  PetscReal      tickloc[MAXSEGS],sep,h,w,tw,th,xl,xr,yl,yr;
  char           *p;
  PetscDraw      draw = axis->win;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!axis) PetscFunctionReturn(0);
  ierr = MPI_Comm_rank(((PetscObject)axis)->comm,&rank);CHKERRQ(ierr);
  if (rank) PetscFunctionReturn(0);

  if (axis->xlow == axis->xhigh) {axis->xlow -= .5; axis->xhigh += .5;}
  if (axis->ylow == axis->yhigh) {axis->ylow -= .5; axis->yhigh += .5;}
  xl = axis->xlow; xr = axis->xhigh; yl = axis->ylow; yr = axis->yhigh;
  ierr = PetscDrawSetCoordinates(draw,xl,yl,xr,yr);CHKERRQ(ierr);
  ierr = PetscDrawStringGetSize(draw,&tw,&th);CHKERRQ(ierr);
  numx = (int)(.15*(xr-xl)/tw); if (numx > 6) numx = 6; if (numx< 2) numx = 2;
  numy = (int)(.5*(yr-yl)/th); if (numy > 6) numy = 6; if (numy< 2) numy = 2;
  xl -= 8*tw; xr += 2*tw; yl -= 2.5*th; yr += 2*th;
  if (axis->xlabel) yl -= 2*th;
  if (axis->ylabel) xl -= 2*tw;
  ierr = PetscDrawSetCoordinates(draw,xl,yl,xr,yr);CHKERRQ(ierr);
  ierr = PetscDrawStringGetSize(draw,&tw,&th);CHKERRQ(ierr);

  ierr = PetscDrawLine(draw,axis->xlow,axis->ylow,axis->xhigh,axis->ylow,ac);CHKERRQ(ierr);
  ierr = PetscDrawLine(draw,axis->xlow,axis->ylow,axis->xlow,axis->yhigh,ac);CHKERRQ(ierr);

  if (axis->toplabel) {
    ierr =  PetscStrlen(axis->toplabel,&len);CHKERRQ(ierr);
    w    = xl + .5*(xr - xl) - .5*len*tw;
    h    = axis->yhigh;
    ierr = PetscDrawString(draw,w,h,cc,axis->toplabel);CHKERRQ(ierr);
  }

  /* PetscDraw the ticks and labels */
  if (axis->xticks) {
    ierr = (*axis->xticks)(axis->xlow,axis->xhigh,numx,&ntick,tickloc,MAXSEGS);CHKERRQ(ierr);
    /* PetscDraw in tick marks */
    for (i=0; i<ntick; i++) {
      ierr = PetscDrawLine(draw,tickloc[i],axis->ylow-.5*th,tickloc[i],axis->ylow+.5*th,tc);CHKERRQ(ierr);
    }
    /* label ticks */
    for (i=0; i<ntick; i++) {
	if (axis->xlabelstr) {
	    if (i < ntick - 1) sep = tickloc[i+1] - tickloc[i];
	    else if (i > 0)    sep = tickloc[i]   - tickloc[i-1];
	    else               sep = 0.0;
	    ierr = (*axis->xlabelstr)(tickloc[i],sep,&p);CHKERRQ(ierr);
            ierr = PetscStrlen(p,&len);CHKERRQ(ierr);
	    w    = .5*len*tw;
	    ierr = PetscDrawString(draw,tickloc[i]-w,axis->ylow-1.2*th,cc,p);CHKERRQ(ierr);
        }
    }
  }
  if (axis->xlabel) {
    ierr = PetscStrlen(axis->xlabel,&len);CHKERRQ(ierr);
    w    = xl + .5*(xr - xl) - .5*len*tw;
    h    = axis->ylow - 2.5*th;
    ierr = PetscDrawString(draw,w,h,cc,axis->xlabel);CHKERRQ(ierr);
  }
  if (axis->yticks) {
    ierr = (*axis->yticks)(axis->ylow,axis->yhigh,numy,&ntick,tickloc,MAXSEGS);CHKERRQ(ierr);
    /* PetscDraw in tick marks */
    for (i=0; i<ntick; i++) {
      ierr = PetscDrawLine(draw,axis->xlow -.5*tw,tickloc[i],axis->xlow+.5*tw,tickloc[i],tc);CHKERRQ(ierr);
    }
    /* label ticks */
    for (i=0; i<ntick; i++) {
	if (axis->ylabelstr) {
	    if (i < ntick - 1) sep = tickloc[i+1] - tickloc[i];
	    else if (i > 0)    sep = tickloc[i]   - tickloc[i-1];
	    else               sep = 0.0;
	    ierr = (*axis->xlabelstr)(tickloc[i],sep,&p);CHKERRQ(ierr);
            ierr = PetscStrlen(p,&len);CHKERRQ(ierr);
	    w    = axis->xlow - len * tw - 1.2*tw;
	    ierr = PetscDrawString(draw,w,tickloc[i]-.5*th,cc,p);CHKERRQ(ierr);
        }
    }
  }
  if (axis->ylabel) {
    ierr = PetscStrlen(axis->ylabel,&len);CHKERRQ(ierr);
    h    = yl + .5*(yr - yl) + .5*len*th;
    w    = xl + .5*tw;
    ierr = PetscDrawStringVertical(draw,w,h,cc,axis->ylabel);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscStripAllZeros"
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

#undef __FUNCT__
#define __FUNCT__ "PetscStripTrailingZeros"
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

#undef __FUNCT__
#define __FUNCT__ "PetscStripInitialZero"
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
    for (i=0; i<n; i++) {
      buf[i] = buf[i+1];
    }
  } else if (buf[0] == '-' && buf[1] == '0') {
    for (i=1; i<n; i++) {
      buf[i] = buf[i+1];
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscStripZeros"
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

#undef __FUNCT__
#define __FUNCT__ "PetscStripZerosPlus"
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







