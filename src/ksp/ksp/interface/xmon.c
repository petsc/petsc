
#include "src/ksp/ksp/kspimpl.h"              /*I  "petscksp.h"   I*/

#undef __FUNCT__  
#define __FUNCT__ "KSPLGMonitorCreate"
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
   Use KSPLGMonitorDestroy() to destroy this line graph; do not use PetscDrawLGDestroy().

   Level: intermediate

.keywords: KSP, monitor, line graph, residual, create

.seealso: KSPLGMonitorDestroy(), KSPSetMonitor(), KSPLGTrueMonitorCreate()
@*/
PetscErrorCode KSPLGMonitorCreate(const char host[],const char label[],int x,int y,int m,int n,PetscDrawLG *draw)
{
  PetscDraw      win;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscDrawCreate(PETSC_COMM_SELF,host,label,x,y,m,n,&win);CHKERRQ(ierr);
  ierr = PetscDrawSetType(win,PETSC_DRAW_X);CHKERRQ(ierr);
  ierr = PetscDrawLGCreate(win,1,draw);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(*draw,win);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPLGMonitor"
PetscErrorCode KSPLGMonitor(KSP ksp,PetscInt n,PetscReal rnorm,void *monctx)
{
  PetscDrawLG    lg = (PetscDrawLG) monctx;
  PetscErrorCode ierr;
  PetscReal      x,y;

  PetscFunctionBegin;
  if (!monctx) {
    MPI_Comm    comm;
    PetscViewer viewer;

    ierr   = PetscObjectGetComm((PetscObject)ksp,&comm);CHKERRQ(ierr);
    viewer = PETSC_VIEWER_DRAW_(comm);
    ierr   = PetscViewerDrawGetDrawLG(viewer,0,&lg);CHKERRQ(ierr);
  }

  if (!n) {ierr = PetscDrawLGReset(lg);CHKERRQ(ierr);}
  x = (PetscReal) n;
  if (rnorm > 0.0) y = log10(rnorm); else y = -15.0;
  ierr = PetscDrawLGAddPoint(lg,&x,&y);CHKERRQ(ierr);
  if (n < 20 || !(n % 5)) {
    ierr = PetscDrawLGDraw(lg);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
} 
 
#undef __FUNCT__  
#define __FUNCT__ "KSPLGMonitorDestroy"
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
PetscErrorCode KSPLGMonitorDestroy(PetscDrawLG drawlg)
{
  PetscDraw      draw;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscDrawLGGetDraw(drawlg,&draw);CHKERRQ(ierr);
  if (draw) { ierr = PetscDrawDestroy(draw);CHKERRQ(ierr);}
  ierr = PetscDrawLGDestroy(drawlg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPLGTrueMonitorCreate"
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
   PetscDrawLGDestroy().

   Level: intermediate

.keywords: KSP, monitor, line graph, residual, create, true

.seealso: KSPLGMonitorDestroy(), KSPSetMonitor(), KSPDefaultMonitor()
@*/
PetscErrorCode KSPLGTrueMonitorCreate(MPI_Comm comm,const char host[],const char label[],int x,int y,int m,int n,PetscDrawLG *draw)
{
  PetscDraw      win;
  PetscErrorCode ierr;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (rank) { *draw = 0; PetscFunctionReturn(0);}

  ierr = PetscDrawCreate(PETSC_COMM_SELF,host,label,x,y,m,n,&win);CHKERRQ(ierr);
  ierr = PetscDrawSetType(win,PETSC_DRAW_X);CHKERRQ(ierr);
  ierr = PetscDrawLGCreate(win,2,draw);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(*draw,win);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPLGTrueMonitor"
PetscErrorCode KSPLGTrueMonitor(KSP ksp,PetscInt n,PetscReal rnorm,void *monctx)
{
  PetscDrawLG    lg = (PetscDrawLG) monctx;
  PetscReal      x[2],y[2],scnorm;
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  Vec            resid,work;

  PetscFunctionBegin;
  if (!monctx) {
    MPI_Comm    comm;
    PetscViewer viewer;

    ierr   = PetscObjectGetComm((PetscObject)ksp,&comm);CHKERRQ(ierr);
    viewer = PETSC_VIEWER_DRAW_(comm);
    ierr   = PetscViewerDrawGetDrawLG(viewer,0,&lg);CHKERRQ(ierr);
  }

  ierr = MPI_Comm_rank(ksp->comm,&rank);CHKERRQ(ierr);
  if (!rank) { 
    if (!n) {ierr = PetscDrawLGReset(lg);CHKERRQ(ierr);}
    x[0] = x[1] = (PetscReal) n;
    if (rnorm > 0.0) y[0] = log10(rnorm); else y[0] = -15.0;
  }

  ierr = VecDuplicate(ksp->vec_rhs,&work);CHKERRQ(ierr);
  ierr = KSPBuildResidual(ksp,0,work,&resid);CHKERRQ(ierr);
  ierr = VecNorm(resid,NORM_2,&scnorm);CHKERRQ(ierr);
  ierr = VecDestroy(work);CHKERRQ(ierr);

  if (!rank) {
    if (scnorm > 0.0) y[1] = log10(scnorm); else y[1] = -15.0;
    ierr = PetscDrawLGAddPoint(lg,x,y);CHKERRQ(ierr);
    if (n <= 20 || (n % 3)) {
      ierr = PetscDrawLGDraw(lg);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
} 
 
#undef __FUNCT__  
#define __FUNCT__ "KSPLGTrueMonitorDestroy"
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
PetscErrorCode KSPLGTrueMonitorDestroy(PetscDrawLG drawlg)
{
  PetscErrorCode ierr;
  PetscDraw      draw;

  PetscFunctionBegin;
  ierr = PetscDrawLGGetDraw(drawlg,&draw);CHKERRQ(ierr);
  ierr = PetscDrawDestroy(draw);CHKERRQ(ierr);
  ierr = PetscDrawLGDestroy(drawlg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


