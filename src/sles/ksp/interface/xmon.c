#ifndef lint
static char vcid[] = "$Id: xmon.c,v 1.7 1995/04/13 17:24:44 curfman Exp curfman $";
#endif

#include "petsc.h"
#include "../../draw/drawimpl.h"
#include "kspimpl.h"
#include <math.h>


/*@
   KSPLGMonitorCreate - Creates a line graph context for use with 
   KSP to monitor convergence of residual norms.

   Input Parameters:
.  host - the X display to open, or null for the local machine
.  label - the title to put in the title bar
.  x, y - the screen coordinates of the upper left coordinate of
          the window
.  m, n - the screen width and height in pixels

   Output Parameter:
.  ctx - the drawing context

   Notes:
   Keywords:  KSP, monitor, line graph, residual, create
@*/
int KSPLGMonitorCreate(char *host,char *label,int x,int y,int m,
                       int n, DrawLGCtx *ctx)
{
  DrawCtx win;
  int     ierr;
  ierr = DrawOpenX(MPI_COMM_SELF,host,label,x,y,m,n,&win); CHKERR(ierr);
  ierr = DrawLGCreate(win,1,ctx); CHKERR(ierr);
  return 0;
}

int KSPLGMonitor(KSP itP,int n,double rnorm,void *monctx)
{
  DrawLGCtx lg = (DrawLGCtx) monctx;
  double    x, y;

  if (!n) DrawLGReset(lg);
  x = (double) n;
  if (rnorm > 0.0) y = log10(rnorm); else y = -15.0;
  DrawLGAddPoint(lg,&x,&y);
  if (n < 20 || (n % 5)) {
    DrawLG(lg);
  }
  return 0;
} 
 
/*@
   KSPLGMonitorDestroy - Destroys a line graph context that was created 
   with KSPLGMonitorCreate().

   Input Parameter:
.  ctx - the drawing context

   Notes:
   Keywords:  KSP, monitor, line graph, destroy
@*/
int KSPLGMonitorDestroy(DrawLGCtx ctx)
{
  DrawCtx win;
  DrawLGGetDrawCtx(ctx,&win);
  DrawDestroy(win);
  DrawLGDestroy(ctx);
  return 0;
}


