#ifndef lint
static char vcid[] = "$Id: axis.c,v 1.22 1995/10/06 22:25:17 bsmith Exp bsmith $";
#endif
/*
   This file contains a simple routine for generating a 2-d axis.
*/

#include "petsc.h"
#include "draw.h"              /*I "draw.h" I*/
#include <math.h>
#include "pinclude/petscfix.h"

#if defined(PARCH_alpha) && defined(__cplusplus)
extern "C" {
extern double   rint(double);
}
#endif


struct _DrawAxisCtx {
    PETSCHEADER
    double  xlow, ylow, xhigh, yhigh;     /* User - coord limits */
    char    *(*ylabelstr)(double,double), /* routines to generate labels */ 
            *(*xlabelstr)(double,double);
    int     (*xlabels)(), (*ylabels)()  , /* location of labels */
            (*xticks)(double,double,int,int*,double*,int),
            (*yticks)(double,double,int,int*,double*,int);  
                                          /* location and size of ticks */
    DrawCtx win;
    int     ac,tc,cc;                     /* axis, tick, charactor color */
    char    *xlabel,*ylabel,*toplabel;
};

#define MAXSEGS 20

static int    XiADefTicks(double,double,int,int*,double*,int);
static char   *XiADefLabel(double,double);
static double XiAGetNice(double,double,int );
static int    XiAGetBase(double,double,int,double*,int*);

#if defined(PARCH_cray) || defined(PARCH_t3d)
static double rint(double x )
{
  if (x > 0) return floor( x + 0.5 );
  return floor( x - 0.5 );
}
#endif

/*@C
   DrawAxisCreate - Generate the axis data structure.

   Input Parameters:

   Ouput Parameters:
.   axis - the axis datastructure

@*/
int DrawAxisCreate(DrawCtx win,DrawAxisCtx *ctx)
{
  DrawAxisCtx ad;
  PetscObject vobj = (PetscObject) win;

  if (vobj->cookie == DRAW_COOKIE && vobj->type == NULLWINDOW) {
     return DrawOpenNull(vobj->comm,(DrawCtx*)ctx);
  }
  PetscHeaderCreate(ad,_DrawAxisCtx,AXIS_COOKIE,0,vobj->comm);
  PLogObjectCreate(ad);
  PLogObjectParent(win,ad);
  ad->xticks    = XiADefTicks;
  ad->yticks    = XiADefTicks;
  ad->xlabelstr = XiADefLabel;
  ad->ylabelstr = XiADefLabel;
  ad->win       = win;
  ad->ac        = DRAW_BLACK;
  ad->tc        = DRAW_BLACK;
  ad->cc        = DRAW_BLACK;
  ad->xlabel    = 0;
  ad->ylabel    = 0;
  ad->toplabel  = 0;

  *ctx = ad;
  return 0;
}

/*@C
      DrawAxisDestroy - Frees the space used by an axis structure.

  Input Parameters:
.   axis - the axis context
@*/
int DrawAxisDestroy(DrawAxisCtx axis)
{
  PLogObjectDestroy(axis);
  PetscHeaderDestroy(axis);
  return 0;
}

/*@
    DrawAxisSetColors -  Sets the colors to be used for the axis,       
                         tickmarks, and text.

   Input Parameters:
.   axis - the axis
.   ac - the color of the axis lines
.   tc - the color of the tick marks
.   cc - the color of the text strings
@*/
int DrawAxisSetColors(DrawAxisCtx axis,int ac,int tc,int cc)
{
  axis->ac = ac; axis->tc = tc; axis->cc = cc;
  return 0;
}

/*@C
    DrawAxisSetLabels -  Sets the x and y axis labels.


   Input Parameters:
.   axis - the axis
.   top - the label at the top of the image
.   xlabel,ylabel - the labes for the x and y axis
@*/
int DrawAxisSetLabels(DrawAxisCtx axis,char* top,char *xlabel,char *ylabel)
{
  axis->xlabel   = xlabel;
  axis->ylabel   = ylabel;
  axis->toplabel = top;
  return 0;
}

/*@
    DrawAxisSetLimits -  Sets the limits (in user coords) of the axis
    
    Input parameters:
.   ad - Axis structure
.   xmin,xmax - limits in x
.   ymin,ymax - limits in y
@*/
int DrawAxisSetLimits(DrawAxisCtx ad,double xmin,double xmax,double ymin,
                      double ymax)
{
  ad->xlow = xmin;
  ad->xhigh= xmax;
  ad->ylow = ymin;
  ad->yhigh= ymax;
  return 0;
}

/*@
    DrawAxis - Draws an axis.

    Input Parameter:
.   ad - Axis structure

    Note:
    This draws the actual axis.  The limits etc have already been set.
    By picking special routines for the ticks and labels, special
    effects may be generated.  These routines are part of the Axis
    structure (ad).
@*/
int DrawAxis(DrawAxisCtx ad )
{
  int       i,  ntick, numx, numy, ac = ad->ac, tc = ad->tc;
  int       cc = ad->cc;
  double    tickloc[MAXSEGS], sep;
  char      *p;
  DrawCtx   awin = ad->win;
  double    h,w,tw,th,xl,xr,yl,yr;

  if (ad->xlow == ad->xhigh) {ad->xlow -= .5; ad->xhigh += .5;}
  if (ad->ylow == ad->yhigh) {ad->ylow -= .5; ad->yhigh += .5;}
  xl = ad->xlow; xr = ad->xhigh; yl = ad->ylow; yr = ad->yhigh;
  DrawSetCoordinates(awin,xl,yl,xr,yr);
  DrawTextGetSize(awin,&tw,&th);
  numx = (int) (.15*(xr-xl)/tw); if (numx > 6) numx = 6; if (numx< 2) numx = 2;
  numy = (int) (.5*(yr-yl)/th); if (numy > 6) numy = 6; if (numy< 2) numy = 2;
  xl -= 8*tw; xr += 2*tw; yl -= 2.5*th; yr += 2*th;
  if (ad->xlabel) yl -= 2*th;
  if (ad->ylabel) xl -= 2*tw;
  DrawSetCoordinates(awin,xl,yl,xr,yr);
  DrawTextGetSize(awin,&tw,&th);

  DrawLine( awin, ad->xlow,ad->ylow,ad->xhigh,ad->ylow,ac);
  DrawLine( awin, ad->xlow,ad->ylow,ad->xlow,ad->yhigh,ac);

  if (ad->toplabel) {
    w = xl + .5*(xr - xl) - .5*((int)PetscStrlen(ad->toplabel))*tw;
    h = ad->yhigh;
    DrawText(awin,w,h,cc,ad->toplabel); 
  }

  /* Draw the ticks and labels */
  if (ad->xticks) {
    (*ad->xticks)( ad->xlow, ad->xhigh, numx, &ntick, tickloc, MAXSEGS );
    /* Draw in tick marks */
    for (i=0; i<ntick; i++ ) {
      DrawLine(awin,tickloc[i],ad->ylow-.5*th,tickloc[i],ad->ylow+.5*th,
               tc);
    }
    /* label ticks */
    for (i=0; i<ntick; i++) {
	if (ad->xlabelstr) {
	    if (i < ntick - 1) sep = tickloc[i+1] - tickloc[i];
	    else if (i > 0)    sep = tickloc[i]   - tickloc[i-1];
	    else               sep = 0.0;
	    p = (*ad->xlabelstr)( tickloc[i], sep );
	    w = .5*((int)PetscStrlen(p)) * tw;
	    DrawText( awin, tickloc[i]-w,ad->ylow-1.2*th,cc,p); 
        }
    }
  }
  if (ad->xlabel) {
    w = xl + .5*(xr - xl) - .5*((int)PetscStrlen(ad->xlabel))*tw;
    h = ad->ylow - 2.5*th;
    DrawText(awin,w,h,cc,ad->xlabel); 
  }
  if (ad->yticks) {
    (*ad->yticks)( ad->ylow, ad->yhigh, numy, &ntick, tickloc, MAXSEGS );
    /* Draw in tick marks */
    for (i=0; i<ntick; i++ ) {
      DrawLine(awin,ad->xlow -.5*tw,tickloc[i],ad->xlow+.5*tw,tickloc[i],
               tc);
    }
    /* label ticks */
    for (i=0; i<ntick; i++) {
	if (ad->ylabelstr) {
	    if (i < ntick - 1) sep = tickloc[i+1] - tickloc[i];
	    else if (i > 0)    sep = tickloc[i]   - tickloc[i-1];
	    else               sep = 0.0;
	    p = (*ad->xlabelstr)( tickloc[i], sep );
	    w = ad->xlow - ((int)PetscStrlen(p)) * tw - 1.2*tw;
	    DrawText( awin, w,tickloc[i]-.5*th,cc,p); 
        }
    }
  }
  if (ad->ylabel) {
    h = yl + .5*(yr - yl) + .5*((int)PetscStrlen(ad->ylabel))*th;
    w = xl + .5*tw;
    DrawTextVertical(awin,w,h,cc,ad->ylabel); 
  }
  return 0;
}

/*
     Removes the extraneous zeros in numbers like 1.10000e6
*/
static int StripZeros(char *buf)
{
  int i,j,n = (int) PetscStrlen(buf);
  if (n<5) return 0;
  for ( i=1; i<n-1; i++ ) {
    if (buf[i] == 'e' && buf[i-1] == '0') {
      for ( j=i; j<n+1; j++ ) buf[j-1] = buf[j];
      StripZeros(buf);
      return 0;
    }
  }
  return 0;
}
static int StripZerosPlus(char *buf)
{
  int i,j,n = (int) PetscStrlen(buf);
  if (n<5) return 0;
  for ( i=1; i<n-2; i++ ) {
    if (buf[i] == '+') {
      if (buf[i+1] == '0') {
        for ( j=i+1; j<n+1; j++ ) buf[j-1] = buf[j+1];
        return 0;
      }
      else {
        for ( j=i+1; j<n+1; j++ ) buf[j] = buf[j+1];
        return 0;  
      }
    }
  }
  return 0;
}
/*
   val is the label value.  sep is the separation to the next (or previous)
   label; this is useful in determining how many significant figures to   
   keep.
 */
static char *XiADefLabel(double val,double sep )
{
  static char buf[40];
  char   fmat[10];
  int    w, d;

  /* Find the string */
  if (fabs(val) < 1.0e6) {
    /* Compute the number of digits */
    w = 0;
    d = 0;
    if (sep > 0.0) {
	d = (int) ceil( - log10 ( sep ) );
	if (d < 0) d = 0;
	if (fabs(val) < 1.0e-6*sep) {
	    /* This is the case where we are near zero and less than a small
	       fraction of the sep.  In this case, we use 0 as the value */
	    val = 0.0;
	    w   = d;
        }
	else if (val == 0.0) w   = d;
	else w = (int) (ceil( log10( fabs( val ) ) ) + d);
	if (w < 1)   w ++;
	if (val < 0) w ++;
    }

    if (rint(val) == val) {
	if (w > 0) sprintf( fmat, "%%%dd", w );
	else PetscStrcpy( fmat, "%d" );
	sprintf( buf, fmat, (int)val );
    }
    else {
	/* The code used here is inappropriate for a val of 0, which
	   tends to print with an excessive numer of digits.  In this
	   case, we should look at the next/previous values and 
	   use those widths */
	if (w > 0) sprintf( fmat, "%%%d.%dlf", w + 1, d );
	else PetscStrcpy( fmat, "%lf" );
	sprintf( buf, fmat, val );
    }
  }
  else {
    sprintf( buf, "%e", val );
    /* remove the extraneous 0's before the e */
    StripZeros(buf);
    StripZerosPlus(buf);
  }
  return buf;
}

/* This is a simple routine that finds "nice" locations for the ticks */
static int XiADefTicks( double low, double high, int num, int *ntick,
                        double * tickloc,int  maxtick )
{
  int    i;
  double x, base;
  int    power;

  XiAGetBase( low, high, num, &base, &power );
  x = XiAGetNice( low, base, -1 );

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
    XiADefTicks( low, high, num+1, ntick, tickloc, maxtick );
  }
  return 0;
}

#define EPS 1.e-6
#if !defined(PARCH_sun4)
static double exp10(double d )
{
  return pow( 10.0, d );
}
#endif

#if !defined(PARCH_sun4)
static double Xifmod(double x,double y )
{
  int     i;
  i   = ((int) x ) / ( (int) y );
  x   = x - i * y;
  while (x > y) x -= y;
  return x;
}
static double Xicopysign(double a,double b )
{
  if (b >= 0) return a;
  return -a;
}
#else
#define Xifmod     fmod
#define Xicopysign copysign
#endif

/*
    Given a value "in" and a "base", return a nice value.
    based on "sgn", extend up (+1) or down (-1)
 */
static double XiAGetNice(double in,double base,int sgn )
{
  double  etmp;

  etmp    = in / base + 0.5 + Xicopysign ( 0.5, (double) sgn );
  etmp    = etmp - 0.5 + Xicopysign( 0.5, etmp ) -
		       Xicopysign ( EPS * etmp, (double) sgn );
  return base * ( etmp - Xifmod( etmp, 1.0 ) );
}

static int XiAGetBase(double vmin,double vmax,int num,double*Base,int*power)
{
  double  base, ftemp;
  static double base_try[5] = {10.0, 5.0, 2.0, 1.0, 0.5};
  int     i;

  /* labels of the form n * BASE */
  /* get an approximate value for BASE */
  base    = ( vmax - vmin ) / (double) (num + 1);

  /* make it of form   m x 10^power,   m in [1.0, 10) */
  if (base <= 0.0) {
    base    = fabs( vmin );
    if (base < 1.0) base = 1.0;
  }
  ftemp   = log10( ( 1.0 + EPS ) * base );
  if (ftemp < 0.0)  ftemp   -= 1.0;
  *power  = (int) ftemp;
  base    = base * exp10( (double) - *power );
  if (base < 1.0) base    = 1.0;
  /* now reduce it to one of 1, 2, or 5 */
  for (i=1; i<5; i++) {
    if (base >= base_try[i]) {
	base            = base_try[i-1] * exp10( (double) *power );
	if (i == 1) *power    = *power + 1;
	break;
    }
  }
  *Base   = base;
  return 0;
}

