#ifndef lint
static char vcid[] = "$Id: xmon.c,v 1.15 1996/03/10 17:26:57 bsmith Exp bsmith $";
#endif

#include "petsc.h"
#include "../../draw/drawimpl.h"  /*I  "draw.h"  I*/
#include "kspimpl.h"              /*I  "ksp.h"   I*/
#include <math.h>


/*@C
   KSPLGMonitorCreate - Creates a line graph context for use with 
   KSP to monitor convergence of residual norms.

   Input Parameters:
.  host - the X display to open, or null for the local machine
.  label - the title to put in the title bar
.  x, y - the screen coordinates of the upper left coordinate of
          the window
.  m, n - the screen width and height in pixels

   Output Parameter:
.  draw - the drawing context

   Notes: use KSPLGMonitorDestroy() to destroy this line graph,
          not DrawLGDestroy().

.keywords: KSP, monitor, line graph, residual, create

.seealso: KSPLGMonitorDestroy(), KSPSetMonitor(), KSPDefaultMonitor()
@*/
int KSPLGMonitorCreate(char *host,char *label,int x,int y,int m,
                       int n, DrawLG *draw)
{
  Draw win;
  int  ierr;
  ierr = DrawOpenX(MPI_COMM_SELF,host,label,x,y,m,n,&win); CHKERRQ(ierr);
  ierr = DrawLGCreate(win,1,draw); CHKERRQ(ierr);
  PLogObjectParent(*draw,win);
  return 0;
}

int KSPLGMonitor(KSP itP,int n,double rnorm,void *monctx)
{
  DrawLG lg = (DrawLG) monctx;
  double    x, y;

  if (!n) DrawLGReset(lg);
  x = (double) n;
  if (rnorm > 0.0) y = log10(rnorm); else y = -15.0;
  DrawLGAddPoint(lg,&x,&y);
  if (n < 20 || (n % 5)) {
    DrawLGDraw(lg);
  }
  return 0;
} 
 
/*@C
   KSPLGMonitorDestroy - Destroys a line graph context that was created 
   with KSPLGMonitorCreate().

   Input Parameter:
.  draw - the drawing context

.keywords: KSP, monitor, line graph, destroy

.seealso: KSPLGMonitorCreate(), KSPSetMonitor(), KSPDefaultMonitor()
@*/
int KSPLGMonitorDestroy(DrawLG drawlg)
{
  Draw draw;
  DrawLGGetDraw(drawlg,&draw);
  DrawDestroy(draw);
  DrawLGDestroy(drawlg);
  return 0;
}


