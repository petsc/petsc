#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: xmon.c,v 1.38 1999/03/01 04:55:34 bsmith Exp bsmith $";
#endif

#include "petsc.h"
#include "src/sys/src/draw/drawimpl.h"  /*I  "draw.h"  I*/
#include "src/sles/ksp/kspimpl.h"              /*I  "ksp.h"   I*/

#undef __FUNC__  
#define __FUNC__ "KSPLGMonitorCreate"
/*@C
   KSPLGMonitorCreate - Creates a line graph context for use with 
   KSP to monitor convergence of preconditioned residual norms.

   Collective on KSP

   Input Parameters:
+  host - the X display to open, or null for the local machine
.  label - the title to put in the title bar
.  x, y - the screen coordinates of the upper left coordinate of
          the window
-  m, n - the screen width and height in pixels

   Output Parameter:
.  draw - the drawing context

   Options Database Key:
.  -ksp_xmonitor - Sets line graph monitor

   Notes: 
   Use KSPLGMonitorDestroy() to destroy this line graph; do not use DrawLGDestroy().

   Level: intermediate

.keywords: KSP, monitor, line graph, residual, create

.seealso: KSPLGMonitorDestroy(), KSPSetMonitor(), KSPLGTrueMonitorCreate()
@*/
int KSPLGMonitorCreate(char *host,char *label,int x,int y,int m,int n, DrawLG *draw)
{
  Draw win;
  int  ierr;

  PetscFunctionBegin;
  ierr = DrawOpenX(PETSC_COMM_SELF,host,label,x,y,m,n,&win); CHKERRQ(ierr);
  ierr = DrawLGCreate(win,1,draw); CHKERRQ(ierr);
  PLogObjectParent(*draw,win);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPLGMonitor"
int KSPLGMonitor(KSP ksp,int n,double rnorm,void *monctx)
{
  DrawLG    lg = (DrawLG) monctx;
  int       ierr;
  double    x, y;

  PetscFunctionBegin;
  if (!n) {ierr = DrawLGReset(lg);CHKERRQ(ierr);}
  x = (double) n;
  if (rnorm > 0.0) y = log10(rnorm); else y = -15.0;
  ierr = DrawLGAddPoint(lg,&x,&y);CHKERRQ(ierr);
  if (n < 20 || (n % 5)) {
    ierr = DrawLGDraw(lg);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
} 
 
#undef __FUNC__  
#define __FUNC__ "KSPLGMonitorDestroy"
/*@C
   KSPLGMonitorDestroy - Destroys a line graph context that was created 
   with KSPLGMonitorCreate().

   Collective on KSP

   Input Parameter:
.  draw - the drawing context

   Level: intermediate

.keywords: KSP, monitor, line graph, destroy

.seealso: KSPLGMonitorCreate(), KSPLGTrueMonitorDestroy(), KSPSetMonitor()
@*/
int KSPLGMonitorDestroy(DrawLG drawlg)
{
  Draw draw;
  int  ierr;

  PetscFunctionBegin;
  ierr = DrawLGGetDraw(drawlg,&draw);CHKERRQ(ierr);
  ierr = DrawDestroy(draw);CHKERRQ(ierr);
  ierr = DrawLGDestroy(drawlg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPLGTrueMonitorCreate"
/*@C
   KSPLGTrueMonitorCreate - Creates a line graph context for use with 
   KSP to monitor convergence of true residual norms (as opposed to
   preconditioned residual norms).

   Collective on KSP

   Input Parameters:
+  host - the X display to open, or null for the local machine
.  label - the title to put in the title bar
.  x, y - the screen coordinates of the upper left coordinate of
          the window
-  m, n - the screen width and height in pixels

   Output Parameter:
.  draw - the drawing context

   Options Database Key:
.  -ksp_xtruemonitor - Sets true line graph monitor

   Notes: 
   Use KSPLGTrueMonitorDestroy() to destroy this line graph, not
   DrawLGDestroy().

   Level: intermediate

.keywords: KSP, monitor, line graph, residual, create, true

.seealso: KSPLGMonitorDestroy(), KSPSetMonitor(), KSPDefaultMonitor()
@*/
int KSPLGTrueMonitorCreate(MPI_Comm comm,char *host,char *label,int x,int y,int m,int n, DrawLG *draw)
{
  Draw win;
  int  ierr,rank;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (rank) { *draw = 0; PetscFunctionReturn(0);}

  ierr = DrawOpenX(PETSC_COMM_SELF,host,label,x,y,m,n,&win); CHKERRQ(ierr);
  ierr = DrawLGCreate(win,2,draw); CHKERRQ(ierr);
  PLogObjectParent(*draw,win);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPLGTrueMonitor"
int KSPLGTrueMonitor(KSP ksp,int n,double rnorm,void *monctx)
{
  DrawLG    lg = (DrawLG) monctx;
  double    x[2], y[2],scnorm;
  int       ierr,rank;
  Vec       resid,work;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(ksp->comm,&rank);CHKERRQ(ierr);
  if (!rank) { 
    if (!n) {ierr = DrawLGReset(lg);CHKERRQ(ierr);}
    x[0] = x[1] = (double) n;
    if (rnorm > 0.0) y[0] = log10(rnorm); else y[0] = -15.0;
  }

  ierr = VecDuplicate(ksp->vec_rhs,&work); CHKERRQ(ierr);
  ierr = KSPBuildResidual(ksp,0,work,&resid); CHKERRQ(ierr);
  ierr = VecNorm(resid,NORM_2,&scnorm); CHKERRQ(ierr);
  ierr = VecDestroy(work);CHKERRQ(ierr);

  if (!rank) {
    if (scnorm > 0.0) y[1] = log10(scnorm); else y[1] = -15.0;
    ierr = DrawLGAddPoint(lg,x,y);CHKERRQ(ierr);
    if (n <= 20 || (n % 3)) {
      ierr = DrawLGDraw(lg);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
} 
 
#undef __FUNC__  
#define __FUNC__ "KSPLGTrueMonitorDestroy"
/*@C
   KSPLGTrueMonitorDestroy - Destroys a line graph context that was created 
   with KSPLGTrueMonitorCreate().

   Collective on KSP

   Input Parameter:
.  draw - the drawing context

   Level: intermediate

.keywords: KSP, monitor, line graph, destroy, true

.seealso: KSPLGTrueMonitorCreate(), KSPSetMonitor()
@*/
int KSPLGTrueMonitorDestroy(DrawLG drawlg)
{
  int  ierr;
  Draw draw;

  PetscFunctionBegin;
  ierr = DrawLGGetDraw(drawlg,&draw);CHKERRQ(ierr);
  ierr = DrawDestroy(draw);CHKERRQ(ierr);
  ierr = DrawLGDestroy(drawlg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


