
#include "tools.h"
#include "iter/itctx.h"
#include "xtools/basex11.h"             /*I "xtools/basex11.h"  I*/
#include "xtools/xlg/xlg.h"
#include <math.h>


XBLineGraph ITLineGraphMonitorCreate(label,host,x,y,m,n)
char *label,*host;
int  x, y, m, n;
{
  XBWindow    win;
  XBLineGraph lg;
  win = XBWinCreate(); CHKPTRN(win);
  if (XBQuickWindow(win,host,label,x,y,m,n)) {SETERR(1); return 0;}
  return XBLineGraphCreate(win,1);
}

XBWindow ITLineGraphGetWin( lg )
XBLineGraph lg;
{
return XBLineGraphGetWindow(lg);
}

void ITLineGraphMonitor(itP,n,rnorm)
ITCntx *itP;
int    n;
double rnorm;
{
  XBLineGraph lg = (XBLineGraph) itP->monP;
  double      x, y;

  x = (double) n;
  if (rnorm > 0.0) y = log10(rnorm); else y = -15.0;
  XBLineGraphAddPoint(lg,&x,&y);
  if (n < 20 || (n % 5)) {
    XBLineGraphDraw(lg);
  }
} 

void ITLineGraphMonitorDestroy(lg)
XBLineGraph lg;
{
  XBWinDestroy(XBLineGraphGetWindow(lg));
  XBLineGraphDestroy(lg);
}
 
void ITLineGraphMonitorReset(lg)
XBLineGraph lg;
{
  XBLineGraphReset(lg);
}


