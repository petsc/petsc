
#ifndef __AXIS
#define __AXIS

typedef struct XBiAxisData *XBAxisData;

/* Routines */
#ifdef ANSI_ARG
#undef ANSI_ARG
#endif
#ifdef __STDC__
#define ANSI_ARGS(a) a
#else
#define ANSI_ARGS(a) ()
#endif

extern XBAxisData XBAInitAxis ANSI_ARGS((XBWindow, XBFont));
extern XBWindow XBAGetWindow ANSI_ARGS((XBAxisData));

extern void XBASetLimits ANSI_ARGS((XBAxisData, 
				    double, double, double, double ));
extern void XBADrawAxis ANSI_ARGS((XBAxisData));
extern void XBADefTicks ANSI_ARGS((double, double, int, int *, 
				   double[], int)), 
            XBADrawLine ANSI_ARGS((XBAxisData, 
				   double, double, double, double)), 
            XBAConvertToLocal ANSI_ARGS((XBAxisData, 
					 double, int *, double, int * ));
extern void XBADestroyAxis ANSI_ARGS((XBAxisData));
extern char *XBADefLabel ANSI_ARGS(( double, double ));
#endif
#ifndef lint
static char vcid[] = "$Id: axis.c,v 1.6 1994/07/22 20:07:53 gropp Exp $";
#endif

/*
   This file contains a simple routine for generating a 2-d axis.
   Given an XBWindow, it returns an XBWindow to be used by any 
   drawing routine.
 */

#include "tools.h"
#include "xtools/basex11.h"
#include "xtools/axis/axis.h"
#include <math.h>

struct XBiAxisData {
    double xlow, ylow, xhigh, yhigh;   /* User - coord limits */
    int    x, y, w, h;                 /* drawing area */
    int    xa, ya, wa, ha;             /* axis area */
    int    oldx, oldy, oldw, oldh;     /* window sizes before axis changed 
					  them */
    char   *(*ylabelstr)(),            /* routines to generate labels */ 
           *(*xlabelstr)();
    void   (*xlabels)(), (*ylabels)(), /* location of labels */
           (*xticks)(),  (*yticks)();  /* location and size of ticks */
    XBWindow win;
    XBFont   font;
    };

#define MAXSEGS 20

void XBADefTicks();
char *XBADefLabel();

#if defined(cray) || defined(HPUX)
/* This is an approximation to rint for the cray */
static double rint( x )
double x;
{
if (x > 0) return floor( x + 0.5 );
return floor( x - 0.5 );
}
#endif

/*@
   XBAInitAxis - Given a window, generate the axis data structure.

   Input Parameters:
.  XBWin - X window
.  font  - font

   Note:
   This routine modifies the window so that the displayed area of the
   window is within the axis.

   This routine may change.
@*/
XBAxisData XBAInitAxis( XBWin, font )
XBWindow XBWin;
XBFont   font;
{
XBAxisData ad;
int        leftwidth, bottomheight, topheight, rightwidth;

ad            = NEW(struct XBiAxisData);          CHKPTRV(ad,0);
ad->win       = XBWin;
ad->font      = font;
ad->xticks    = XBADefTicks;
ad->yticks    = XBADefTicks;
ad->xlabelstr = XBADefLabel;
ad->ylabelstr = XBADefLabel;

/* Set the sizes of the axis area; modify XBWin to use the
   drawing area only (???) */
leftwidth    = 10 * XBFontWidth( font );
bottomheight = 5 + XBFontHeight( font );
rightwidth   = 5;
topheight    = 5;
ad->oldx = XBWinX( XBWin );
ad->oldy = XBWinY( XBWin );
ad->oldw = XBWinWidth( XBWin );
ad->oldh = XBWinHeight( XBWin );
ad->xa = XBWinX( XBWin ) + leftwidth;
ad->ya = XBWinY( XBWin ) + topheight;
ad->wa = XBWinWidth( XBWin ) - leftwidth - rightwidth;
ad->ha = XBWinHeight( XBWin ) - bottomheight - topheight;
ad->x  = ad->xa;
ad->y  = ad->ya;
ad->h  = ad->ha;
ad->w  = ad->wa;

XBWinSetSize( XBWin, ad->xa, ad->ya, ad->wa, ad->ha );

return ad;
}

/*@
    XBASetLimits -  Sets the limits (in user coords) of the axis
    
    Input parameters:
.   ad - Axis structure
.   xmin,xmax - limits in x
.   ymin,ymax - limits in y
@*/
void XBASetLimits( ad, xmin, xmax, ymin, ymax )
XBAxisData ad;
double     xmin, xmax, ymin, ymax;
{
ad->xlow = xmin;
ad->xhigh= xmax;
ad->ylow = ymin;
ad->yhigh= ymax;
}

/*@
    XBADrawAxis - Draws an axis.

    Input Parameter:
.   ad - Axis structure

    Note:
    This draws the actual axis.  The limits etc have already been set.
    By picking special routines for the ticks and labels, special
    effects may be generated.  These routines are part of the Axis
    structure (ad).
@*/
void XBADrawAxis( ad )
XBAxisData ad;
{
XBWindow  awin = ad->win;
int       i, w, ntick, num, width, height;
XBSegment segs[MAXSEGS];
double    sc, tickloc[MAXSEGS], sep;
char      *p;

/* Draw the axis lines.  Note that we need to reverse the y coords */
XBDrawLine( awin, ad->xa, ad->ya + ad->ha, ad->xa, ad->ya );
XBDrawLine( awin, ad->xa, ad->ya + ad->ha, 
	    ad->xa + ad->wa, ad->ya + ad->ha );

/* Draw the ticks and labels */
if (ad->xticks) {
    num = ad->wa / 100;
    if (num < 2) num = 2;
    height = -10;
    (*ad->xticks)( ad->xlow, ad->xhigh, num, &ntick, tickloc, MAXSEGS );
    sc = ad->w / (ad->xhigh - ad->xlow);
    for (i=0; i<ntick; i++) {
	segs[i].y1 = ad->ya + ad->ha;
	segs[i].y2 = ad->ya + ad->ha + height;
	segs[i].x1 = segs[i].x2 = (tickloc[i] - ad->xlow) * sc + ad->xa;
	}
    XBDrawSegments( awin, ntick, segs );
    for (i=0; i<ntick; i++) {
	if (ad->xlabelstr) {
	    if (i < ntick - 1) sep = tickloc[i+1] - tickloc[i];
	    else if (i > 0)    sep = tickloc[i]   - tickloc[i-1];
	    else               sep = 0.0;
	    p = (*ad->xlabelstr)( tickloc[i], sep );
	    w = strlen(p) * XBFontWidth( ad->font );
	    XBDrawText( awin, ad->font, 
		        segs[i].x1 - w/2, 
		        segs[i].y1 + XBFontHeight( ad->font ), p );
	    }
	}
    }
if (ad->yticks) {
    num = ad->ha / 20;
    if (num < 2) num = 2;
    width = 10;
    (*ad->yticks)( ad->ylow, ad->yhigh, num, &ntick, tickloc, MAXSEGS );
    sc = ad->h / (ad->yhigh - ad->ylow);
    for (i=0; i<ntick; i++) {
	segs[i].x1 = ad->xa;
	segs[i].x2 = ad->xa + width;
	segs[i].y1 = segs[i].y2 = - (tickloc[i] - ad->ylow) * sc + 
	                          ad->ya + ad->ha;
	}
    XBDrawSegments( awin, ntick, segs );
    for (i=0; i<ntick; i++) {
	if (ad->ylabelstr) {
	    if (i < ntick - 1) sep = tickloc[i+1] - tickloc[i];
	    else if (i > 0)    sep = tickloc[i]   - tickloc[i-1];
	    else               sep = 0.0;
	    p = (*ad->ylabelstr)( tickloc[i], sep );
	    w = strlen(p) * XBFontWidth( ad->font );
	    XBDrawText( awin, ad->font, 
		        segs[i].x1 - w, 
		        segs[i].y1 + XBFontHeight( ad->font )/2, p );
	    }
	}
    }
}

/*
   val is the label value.  sep is the separation to the next (or previous)
   label; this is useful in determining how many significant figures to   
   keep.
 */
char *XBADefLabel( val, sep )
double val, sep;
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
	d = ceil( - log10 ( sep ) );
	if (d < 0) d = 0;
	if (fabs(val) < 1.0e-6*sep) {
	    /* This is the case where we are near zero and less than a small
	       fraction of the sep.  In this case, we use 0 as the value */
	    val = 0.0;
	    w   = d;
	    }
	else if (val == 0.0) 
	    w   = d;
	else
	    w = ceil( log10( fabs( val ) ) ) + d;
	if (w < 1)   w ++;
	if (val < 0) w ++;
	}

    if (rint(val) == val) {
	if (w > 0) 
	    sprintf( fmat, "%%%dd", w );
	else 
	    strcpy( fmat, "%d" );
	sprintf( buf, fmat, (int)val );
	}
    else {
	/* The code used here is inappropriate for a val of 0, which
	   tends to print with an excessive numer of digits.  In this
	   case, we should look at the next/previous values and 
	   use those widths */
	if (w > 0) 
	    sprintf( fmat, "%%%d.%dlf", w + 1, d );  /* Allow 1 for decimal */
	else 
	    strcpy( fmat, "%lf" );
	sprintf( buf, fmat, val );
	}
    }
else
    sprintf( buf, "%le", val );

return buf;
}

/* This is a simple routine that finds "nice" locations for the ticks */
void XBADefTicks( low, high, num, ntick, tickloc, maxtick )
double low, high, tickloc[];
int    num, *ntick, maxtick;
{
int    i;
double x, base;
int    power;
double XBAGetNice();

XBAGetBase( low, high, num, &base, &power );
x = XBAGetNice( low, base, -1 );

/* Values are of the form j * base */
/* Find the starting value */
if (x < low) x += base;

i = 0;
while (i < maxtick && x <= high) {
    tickloc[i++] = x;
    x += base;
    }
*ntick = i;

if (i < 2 && num < 10) 
    /* Try again */
    XBADefTicks( low, high, num+1, ntick, tickloc, maxtick );
}

/*@
  XBADestroyAxis - Frees an axis structure

  Input Parameter:
. ad - axis structure pointer
@*/
void XBADestroyAxis( ad )
XBAxisData ad;
{
XBWinSetSize( ad->win, ad->oldx, ad->oldy, ad->oldw, ad->oldh );
FREE( ad );
}

/*
    Here are some simple drawing routines that work with axis to draw 
    an image.  Note that since these DO NOT REMEMBER the data, if the
    window is refreshed, the data is lost.
 */

/* Convert to the location in the ... */
void XBAConvertToLocal( ad, x, xi, y, yi )
XBAxisData ad;
double     x, y;
int        *xi, *yi;
{
*xi = ((x - ad->xlow) / (ad->xhigh - ad->xlow) * ad->w) + ad->x;
*yi = ad->h - ((y - ad->ylow) / (ad->yhigh - ad->ylow) * ad->h) + ad->y;
}

/* Draw a single line */
void XBADrawLine( ad, x1, y1, x2, y2 )
XBAxisData ad;
double     x1, y1, x2, y2;
{
int      xi1, yi1, xi2, yi2;
XBWindow awin = ad->win;

XBAConvertToLocal( ad, x1, &xi1, y1, &yi1 );
XBAConvertToLocal( ad, x2, &xi2, y2, &yi2 );

XBDrawLine( awin, xi1, yi1, xi2, yi2 );
}

XBWindow XBAGetWindow( ad )
XBAxisData ad;
{
return ad->win;
}
#ifndef lint
static char vcid[] = "$Id: nice.c,v 1.2 1994/04/29 22:00:20 bsmith Exp $";
#endif

#include <math.h>
#define EPS 1.e-6
/* The SUN math library is more complete than most */
#if !defined(sun4) || defined(solaris)
static double exp10( d )
double d;
{
return pow( 10.0, d );
}
#endif

#if !defined(sun4) || defined(solaris)
/* this is a partial implmentation, adequate only for the usage here.
   The static makes it private */
static double XBfmod( x, y )
double x, y;
{
int     i;
i   = ((int) x ) / ( (int) y );
x   = x - i * y;
while (x > y) x -= y;
return x;
}

static double XBcopysign( a, b )
double a, b;
{
if (b >= 0) return a;
return -a;
}

#else
#define XBfmod     fmod
#define XBcopysign copysign
#endif

/*
    Given a value "in" and a "base", return a nice value.
    based on "sgn", extend up (+1) or down (-1)
 */
double XBAGetNice( in, base, sgn )
double  in, base;
int     sgn;
{
double  etmp;

etmp    = in / base + 0.5 + XBcopysign ( 0.5, (double) sgn );
etmp    = etmp - 0.5 + XBcopysign( 0.5, etmp ) -
		       XBcopysign ( EPS * etmp, (double) sgn );
return base * ( etmp - XBfmod( etmp, 1.0 ) );
}


/*

 */
void XBAGetBase( vmin, vmax, num, Base, power )
double  vmin, vmax, *Base;
int     num, *power;
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
if (ftemp < 0.0)
    ftemp   -= 1.0;
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
}
