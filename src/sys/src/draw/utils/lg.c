/*
       Contains the data structure for plotting several line
    graphs in a window with an axis. This is intended for line 
    graphs that change dynamically by adding more points onto 
    the end of the X axis.
*/

#ifndef _XLG
#define _XLG

#include <X11/Xlib.h>
#include <X11/Xutil.h>

#include "xtools/basex11.h"
#include "xtools/baseclr.h"
#include "xtools/base3d.h"
#include "xtools/lines/lines.h"
#include "xtools/axis/axis.h"

typedef struct XBiLineGraph *XBLineGraph;

/* Routines */
#ifdef ANSI_ARG
#undef ANSI_ARG
#endif
#ifdef __STDC__
#define ANSI_ARGS(a) a
#else
#define ANSI_ARGS(a) ()
#endif

XBLineGraph XBLineGraphCreate ANSI_ARGS((XBWindow, int));
void        XBLineGraphDestroy ANSI_ARGS((XBLineGraph));
void        XBLineGraphReset ANSI_ARGS((XBLineGraph));
void        XBLineGraphDraw ANSI_ARGS((XBLineGraph));
void        XBLineGraphAddPoint ANSI_ARGS((XBLineGraph, double *, double *));
void        XBLineGraphSetLimits ANSI_ARGS((XBLineGraph, double, double, 
					    double, double ));
XBWindow    XBLineGraphGetWindow ANSI_ARGS((XBLineGraph));

#ifdef _ITCONTEXT
XBLineGraph ITLineGraphMonitorCreate ANSI_ARGS((char *, char*, 
						 int, int, int, int));
void        ITLineGraphMonitor ANSI_ARGS((ITCntx *, int, double));
void        ITLineGraphMonitorDestroy ANSI_ARGS((XBLineGraph));
void        ITLineGraphMonitorReset ANSI_ARGS((XBLineGraph));
#endif

#endif
/*
        Routines for manipulating line graphs
*/
#include "tools.h"
#include "xtools/xlg/xlg.h"
#include "xtools/basex11.h"

struct XBiLineGraph {
  XBWindow   win;
  int        winw, winh;
  XBAxisData axis;
  Lines      *lines;
  double     xmin, xmax, ymin, ymax;
  int        nopts, dim;
};

/*@
     XBLineGraphCreate - Creates a XBLineGraph data structure

  Input Parameters:
.   win - the window where the graph will be made.
.   dim - the number of line cures which will be drawn

@*/
XBLineGraph XBLineGraphCreate(win, dim)
XBWindow win;
int      dim;
{
  int         i;
  XBLineGraph lg = NEW(struct XBiLineGraph); CHKPTRN(lg);

  lg->nopts = 0;
  lg->win   = win;
  lg->winw  = XBWinWidth( win );
  lg->winh  = XBWinHeight( win );
  lg->dim   = dim;
  lg->xmin  = 1.e20;
  lg->ymin  = 1.e20;
  lg->xmax  = -1.e20;
  lg->ymax  = -1.e20;
  lg->axis  = XBAInitAxis(win,XBFontFixed(win, 7, 13)); CHKERRN(1);
  lg->lines = (Lines *) MALLOC(dim*sizeof(Lines)); 
  CHKPTRN(lg->lines);
  for ( i=0; i<dim; i++ ) {lg->lines[i] = XBLinesInit(); CHKERRN(1); }
  return lg;
}

/*@
    XBLineGraphReset - Clears line graph to allow for reuse with new data.

@*/
void XBLineGraphReset(lg)
XBLineGraph lg;
{
  XBWindow win = lg->win; 
  int      i;
  for ( i=0; i<lg->dim; i++ ) {XBLinesFree(lg->lines[i]);}
  for ( i=0; i<lg->dim; i++ ) {lg->lines[i] = XBLinesInit(); CHKERR(1); }
  lg->nopts = 0;
  lg->xmin  = 1.e20;
  lg->ymin  = 1.e20;
  lg->xmax  = -1.e20;
  lg->ymax  = -1.e20;
  /* XBClearWindow(win,win->x,win->y,win->w,win->h);   */
}

/*@
    XBLineGraphDestroy - Frees all space taken up by LineGraph 
                         data structure.
@*/
void XBLineGraphDestroy(lg)
XBLineGraph lg;
{
  int i;
  for ( i=0; i<lg->dim; i++ ) {XBLinesFree(lg->lines[i]);}
  FREE(lg->lines);
  XBADestroyAxis(lg->axis);
  FREE(lg);
}

/*@
    XBLineGraphAddPoint - Adds another point to each of the 
                          line graphs. The new point must have a
                          X coordinate larger than the old points.

   Input Parameters:
.   lg - the LineGraph data structure
.   x,y - the points to two vectors containing the new x and y 
           point for each curve.

@*/
void XBLineGraphAddPoint(lg,x,y)
XBLineGraph lg;
double      *x, *y;
{
  int i, j, nc = XBGetNumcolors( lg->win ) - 1;
  for (i=0; i<lg->dim; i++) {
    if (x[i] > lg->xmax) lg->xmax = x[i]; 
    if (x[i] < lg->xmin) lg->xmin = x[i];
    if (y[i] > lg->ymax) lg->ymax = y[i]; 
    if (y[i] < lg->ymin) lg->ymin = y[i];
    j = 1 + (i % nc);
    XBLinesAddLines(lg->lines[i],x+i,y+i,1,XBGetPixvalByIndex( lg->win, j ) );
    CHKERR(1);
  }
  lg->nopts++;
}

/*@
   XBLineGraphDraw - Redraws a line graph
@*/
void XBLineGraphDraw(lg)
XBLineGraph lg;
{
  double   xmin=lg->xmin, xmax=lg->xmax, ymin=lg->ymin, ymax=lg->ymax;
  int      i, dim = lg->dim;
  XBWindow win = lg->win;

  if (xmin >= xmax || ymin >= ymax) return;
  xmax = xmax + .1*(xmax-xmin);
  XBClearWindow(win, 0, 0, lg->winw, lg->winh);
  XBSetToForeground( win );
  XBASetLimits(lg->axis, xmin, xmax, ymin, ymax);
  XBADrawAxis(lg->axis);
  for ( i=0; i<dim; i++ ) {
    XBLinesSetScale(lg->lines[i],xmin,xmax,ymin,ymax);
    XBLinesRescale(win, lg->lines[i]);
    XBLinesDraw(win,lg->lines[i]);
  }
  XBFlush(lg->win);
} 
 

/*@
     XBLineGraphSetLimits - Sets the axis limits for a line graph. If 
                            more points are added after this call the
                            limits will be adjusted to include those 
                            additional points.

  Input Parameters:
.   xlg - the line graph context
.   x_min,x_max,y_min,y_max - the limits

@*/
void XBLineGraphSetLimits( lg,x_min,x_max,y_min,y_max) 
XBLineGraph lg;
double      x_min,x_max,y_min,y_max;
{
(lg)->xmin = x_min; 
(lg)->xmax = x_max; 
(lg)->ymin = y_min; 
(lg)->ymax = y_max;
}
 
/*@
     XBLineGraphGetWindow - Returns the XBWindow of a XBLineGraph

     Input Parameters:
.    lg - XBLineGraph to return window of
@*/
XBWindow XBLineGraphGetWindow( lg )
XBLineGraph lg;
{
return lg->win;
}
