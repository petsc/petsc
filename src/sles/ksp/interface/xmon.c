#ifndef lint
static char vcid[] = "$Id: $";
#endif

#include "petsc.h"
#include "../../draw/drawimpl.h"
#include "kspimpl.h"
#include <math.h>


/*@
     KSPLGMonitorCreate - Creates a line graph context for use with 
                          KSP to monitor convergence of residual norms.

  Input Parameters:
.   label
@*/
int KSPLGMonitorCreate(char *host,char *label,int x,int y,int m,int n,
                       DrawLGCtx *ctx)
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
 
int KSPLGMonitorDestroy(DrawLGCtx ctx)
{
  DrawCtx win;
  DrawLGGetDrawCtx(ctx,&win);
  DrawDestroy(win);
  DrawLGDestroy(ctx);
  return 0;
}


