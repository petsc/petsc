#define PETSC_DLL
/*
   This file contains a simple routine for generating a 2-d axis.
*/

#include "petscsys.h"              /*I "petscsys.h" I*/

PetscCookie DRAWAXIS_COOKIE = 0;

struct _p_DrawAxis {
  PETSCHEADER(int);
    PetscReal      xlow,ylow,xhigh,yhigh;     /* User - coord limits */
    PetscErrorCode (*ylabelstr)(PetscReal,PetscReal,char **);/* routines to generate labels */ 
    PetscErrorCode (*xlabelstr)(PetscReal,PetscReal,char **);
    PetscErrorCode (*xticks)(PetscReal,PetscReal,int,int*,PetscReal*,int);
    PetscErrorCode (*yticks)(PetscReal,PetscReal,int,int*,PetscReal*,int);  
                                          /* location and size of ticks */
    PetscDraw  win;
    int        ac,tc,cc;                     /* axis,tick, character color */
    char       *xlabel,*ylabel,*toplabel;
    PetscTruth hold;
};

#define MAXSEGS 20

EXTERN PetscErrorCode    PetscADefTicks(PetscReal,PetscReal,int,int*,PetscReal*,int);
EXTERN PetscErrorCode    PetscADefLabel(PetscReal,PetscReal,char**);
static PetscErrorCode    PetscAGetNice(PetscReal,PetscReal,int,PetscReal*);
static PetscErrorCode    PetscAGetBase(PetscReal,PetscReal,int,PetscReal*,int*);

#undef __FUNCT__  
#define __FUNCT__ "PetscRint" 
static PetscErrorCode PetscRint(PetscReal x,PetscReal *result)
{
  PetscFunctionBegin;
  if (x > 0) *result = floor(x + 0.5);
  else       *result = floor(x - 0.5);
  PetscFunctionReturn(0);
}

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
PetscErrorCode PETSC_DLLEXPORT PetscDrawAxisCreate(PetscDraw draw,PetscDrawAxis *axis)
{
  PetscDrawAxis  ad;
  PetscObject    obj = (PetscObject)draw;
  PetscErrorCode ierr;
  PetscTruth     isnull;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_COOKIE,1);
  PetscValidPointer(axis,2);
  ierr = PetscTypeCompare(obj,PETSC_DRAW_NULL,&isnull);CHKERRQ(ierr);
  if (isnull) {
    ierr = PetscDrawOpenNull(((PetscObject)obj)->comm,(PetscDraw*)axis);CHKERRQ(ierr);
    (*axis)->win = draw;
    PetscFunctionReturn(0);
  }
  ierr = PetscHeaderCreate(ad,_p_DrawAxis,int,DRAWAXIS_COOKIE,0,"PetscDrawAxis",((PetscObject)obj)->comm,PetscDrawAxisDestroy,0);CHKERRQ(ierr);
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
PetscErrorCode PETSC_DLLEXPORT PetscDrawAxisDestroy(PetscDrawAxis axis)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!axis) PetscFunctionReturn(0);
  if (--((PetscObject)axis)->refct > 0) PetscFunctionReturn(0);

  ierr = PetscStrfree(axis->toplabel);CHKERRQ(ierr);
  ierr = PetscStrfree(axis->xlabel);CHKERRQ(ierr);
  ierr = PetscStrfree(axis->ylabel);CHKERRQ(ierr);
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
PetscErrorCode PETSC_DLLEXPORT PetscDrawAxisSetColors(PetscDrawAxis axis,int ac,int tc,int cc)
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

    Level: advanced

@*/
PetscErrorCode PETSC_DLLEXPORT PetscDrawAxisSetLabels(PetscDrawAxis axis,const char top[],const char xlabel[],const char ylabel[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!axis) PetscFunctionReturn(0);
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
PetscErrorCode PETSC_DLLEXPORT PetscDrawAxisSetHoldLimits(PetscDrawAxis axis,PetscTruth hold)
{
  PetscFunctionBegin;
  if (!axis) PetscFunctionReturn(0);
  axis->hold = hold;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawAxisSetLimits" 
/*@
    PetscDrawAxisSetLimits -  Sets the limits (in user coords) of the axis
    
    Not Collective (ignored on all processors except processor 0 of PetscDrawAxis)

    Input Parameters:
+   axis - the axis
.   xmin,xmax - limits in x
-   ymin,ymax - limits in y

    Level: advanced

.seealso:  PetscDrawAxisSetHoldLimits()

@*/
PetscErrorCode PETSC_DLLEXPORT PetscDrawAxisSetLimits(PetscDrawAxis axis,PetscReal xmin,PetscReal xmax,PetscReal ymin,PetscReal ymax)
{
  PetscFunctionBegin;
  if (!axis) PetscFunctionReturn(0);
  if (axis->hold) PetscFunctionReturn(0);
  axis->xlow = xmin;
  axis->xhigh= xmax;
  axis->ylow = ymin;
  axis->yhigh= ymax;
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
PetscErrorCode PETSC_DLLEXPORT PetscDrawAxisDraw(PetscDrawAxis axis)
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
static PetscErrorCode PetscStripAllZeros(char *buf)
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
static PetscErrorCode PetscStripTrailingZeros(char *buf)
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
static PetscErrorCode PetscStripInitialZero(char *buf)
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
static PetscErrorCode PetscStripZeros(char *buf)
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
      Removes the plus in something like 1.1e+2
*/
static PetscErrorCode PetscStripZerosPlus(char *buf)
{
  PetscErrorCode ierr;
  size_t         i,j,n;

  PetscFunctionBegin;
  ierr = PetscStrlen(buf,&n);CHKERRQ(ierr);
  if (n<5) PetscFunctionReturn(0);
  for (i=1; i<n-2; i++) {
    if (buf[i] == '+') {
      if (buf[i+1] == '0') {
        for (j=i+1; j<n+1; j++) buf[j-1] = buf[j+1];
        PetscFunctionReturn(0);
      } else {
        for (j=i+1; j<n+1; j++) buf[j] = buf[j+1];
        PetscFunctionReturn(0);  
      }
    } else if (buf[i] == '-') {
      if (buf[i+1] == '0') {
        for (j=i+1; j<n+1; j++) buf[j] = buf[j+1];
        PetscFunctionReturn(0);
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscADefLabel" 
/*
   val is the label value.  sep is the separation to the next (or previous)
   label; this is useful in determining how many significant figures to   
   keep.
 */
PetscErrorCode PetscADefLabel(PetscReal val,PetscReal sep,char **p)
{
  static char    buf[40];
  char           fmat[10];
  PetscErrorCode ierr;
  int            w,d;
  PetscReal      rval;

  PetscFunctionBegin;
  /* Find the string */
  if (PetscAbsReal(val)/sep <  1.e-6) {
    buf[0] = '0'; buf[1] = 0;
  } else if (PetscAbsReal(val) < 1.0e6 && PetscAbsReal(val) > 1.e-4) {
    /* Compute the number of digits */
    w = 0;
    d = 0;
    if (sep > 0.0) {
	d = (int)ceil(- log10 (sep));
	if (d < 0) d = 0;
	if (PetscAbsReal(val) < 1.0e-6*sep) {
	    /* This is the case where we are near zero and less than a small
	       fraction of the sep.  In this case, we use 0 as the value */
	    val = 0.0;
	    w   = d;
        }
	else if (val == 0.0) w   = d;
	else w = (int)(ceil(log10(PetscAbsReal(val))) + d);
	if (w < 1)   w ++;
	if (val < 0) w ++;
    }

    ierr = PetscRint(val,&rval);CHKERRQ(ierr);
    if (rval == val) {
	if (w > 0) sprintf(fmat,"%%%dd",w);
	else {ierr = PetscStrcpy(fmat,"%d");CHKERRQ(ierr);}
	sprintf(buf,fmat,(int)val);
        ierr = PetscStripInitialZero(buf);CHKERRQ(ierr);
        ierr = PetscStripAllZeros(buf);CHKERRQ(ierr);
        ierr = PetscStripTrailingZeros(buf);CHKERRQ(ierr);
    } else {
	/* The code used here is inappropriate for a val of 0, which
	   tends to print with an excessive numer of digits.  In this
	   case, we should look at the next/previous values and 
	   use those widths */
	if (w > 0) sprintf(fmat,"%%%d.%dlf",w + 1,d);
	else {ierr = PetscStrcpy(fmat,"%lf");CHKERRQ(ierr);}
	sprintf(buf,fmat,val);
        ierr = PetscStripInitialZero(buf);CHKERRQ(ierr);
        ierr = PetscStripAllZeros(buf);CHKERRQ(ierr);
        ierr = PetscStripTrailingZeros(buf);CHKERRQ(ierr);
    }
  } else {
    sprintf(buf,"%e",val);
    /* remove the extraneous 0 before the e */
    ierr = PetscStripZeros(buf);CHKERRQ(ierr);
    ierr = PetscStripZerosPlus(buf);CHKERRQ(ierr);
    ierr = PetscStripInitialZero(buf);CHKERRQ(ierr);
    ierr = PetscStripAllZeros(buf);CHKERRQ(ierr);
    ierr = PetscStripTrailingZeros(buf);CHKERRQ(ierr);
  }
  *p =buf;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscADefTicks" 
/* Finds "nice" locations for the ticks */
PetscErrorCode PetscADefTicks(PetscReal low,PetscReal high,int num,int *ntick,PetscReal * tickloc,int  maxtick)
{
  PetscErrorCode ierr;
  int            i,power;
  PetscReal      x = 0.0,base=0.0;

  PetscFunctionBegin;
  /* patch if low == high */
  if (low == high) {
    low  -= .01;
    high += .01;
  }

  /*  if (PetscAbsReal(low-high) < 1.e-8) {
    low  -= .01;
    high += .01;
  } */

  ierr = PetscAGetBase(low,high,num,&base,&power);CHKERRQ(ierr);
  ierr = PetscAGetNice(low,base,-1,&x);CHKERRQ(ierr);

  /* Values are of the form j * base */
  /* Find the starting value */
  if (x < low) x += base;

  i = 0;
  while (i < maxtick && x <= high) {
    tickloc[i++] = x;
    x += base;
  }
  *ntick = i;

  if (i < 2 && num < 10) {
    ierr = PetscADefTicks(low,high,num+1,ntick,tickloc,maxtick);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#define EPS 1.e-6

#undef __FUNCT__  
#define __FUNCT__ "PetscExp10" 
static PetscErrorCode PetscExp10(PetscReal d,PetscReal *result)
{
  PetscFunctionBegin;
  *result = pow((PetscReal)10.0,d);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscMod" 
static PetscErrorCode PetscMod(PetscReal x,PetscReal y,PetscReal *result)
{
  int     i;

  PetscFunctionBegin;
  if (y == 1) {
    *result = 0.0;
    PetscFunctionReturn(0);
  }
  i   = ((int)x) / ((int)y);
  x   = x - i * y;
  while (x > y) x -= y;
  *result = x;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscCopysign" 
static PetscErrorCode PetscCopysign(PetscReal a,PetscReal b,PetscReal *result)
{
  PetscFunctionBegin;
  if (b >= 0) *result = a;
  else        *result = -a;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscAGetNice" 
/*
    Given a value "in" and a "base", return a nice value.
    based on "sign", extend up (+1) or down (-1)
 */
static PetscErrorCode PetscAGetNice(PetscReal in,PetscReal base,int sign,PetscReal *result)
{
  PetscReal      etmp,s,s2,m;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr    = PetscCopysign (0.5,(double)sign,&s);CHKERRQ(ierr);
  etmp    = in / base + 0.5 + s;
  ierr    = PetscCopysign (0.5,etmp,&s);CHKERRQ(ierr);
  ierr    = PetscCopysign (EPS * etmp,(double)sign,&s2);CHKERRQ(ierr);
  etmp    = etmp - 0.5 + s - s2;
  ierr    = PetscMod(etmp,1.0,&m);CHKERRQ(ierr);
  etmp    = base * (etmp -  m);
  *result = etmp;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscAGetBase" 
static PetscErrorCode PetscAGetBase(PetscReal vmin,PetscReal vmax,int num,PetscReal*Base,int*power)
{
  PetscReal        base,ftemp,e10;
  static PetscReal base_try[5] = {10.0,5.0,2.0,1.0,0.5};
  PetscErrorCode   ierr;
  int              i;

  PetscFunctionBegin;
  /* labels of the form n * BASE */
  /* get an approximate value for BASE */
  base    = (vmax - vmin) / (double)(num + 1);

  /* make it of form   m x 10^power,  m in [1.0, 10) */
  if (base <= 0.0) {
    base    = PetscAbsReal(vmin);
    if (base < 1.0) base = 1.0;
  }
  ftemp   = log10((1.0 + EPS) * base);
  if (ftemp < 0.0)  ftemp   -= 1.0;
  *power  = (int)ftemp;
  ierr = PetscExp10((double)- *power,&e10);CHKERRQ(ierr);
  base    = base * e10;
  if (base < 1.0) base    = 1.0;
  /* now reduce it to one of 1, 2, or 5 */
  for (i=1; i<5; i++) {
    if (base >= base_try[i]) {
      ierr = PetscExp10((double)*power,&e10);CHKERRQ(ierr);
      base = base_try[i-1] * e10;
      if (i == 1) *power    = *power + 1;
      break;
    }
  }
  *Base   = base;
  PetscFunctionReturn(0);
}






