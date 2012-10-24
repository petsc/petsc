
#include <petsc-private/kspimpl.h>              /*I  "petscksp.h"   I*/

#undef __FUNCT__
#define __FUNCT__ "KSPMonitorLGResidualNormCreate"
/*@C
   KSPMonitorLGResidualNormCreate - Creates a line graph context for use with
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
.  -ksp_monitor_lg_residualnorm - Sets line graph monitor

   Notes:
   Use KSPMonitorLGResidualNormDestroy() to destroy this line graph; do not use PetscDrawLGDestroy().

   Level: intermediate

.keywords: KSP, monitor, line graph, residual, create

.seealso: KSPMonitorLGResidualNormDestroy(), KSPMonitorSet(), KSPMonitorLGTrueResidualCreate()
@*/
PetscErrorCode  KSPMonitorLGResidualNormCreate(const char host[],const char label[],int x,int y,int m,int n,PetscDrawLG *draw)
{
  PetscDraw      win;
  PetscErrorCode ierr;
  PetscDrawAxis  axis;

  PetscFunctionBegin;
  ierr = PetscDrawCreate(PETSC_COMM_SELF,host,label,x,y,m,n,&win);CHKERRQ(ierr);
  ierr = PetscDrawSetFromOptions(win);CHKERRQ(ierr);
  ierr = PetscDrawLGCreate(win,1,draw);CHKERRQ(ierr);
  ierr = PetscDrawLGGetAxis(*draw,&axis);CHKERRQ(ierr);
  ierr = PetscDrawAxisSetLabels(axis,"Convergence","Iteration","Residual Norm");CHKERRQ(ierr);
  ierr = PetscLogObjectParent(*draw,win);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPMonitorLGResidualNorm"
PetscErrorCode  KSPMonitorLGResidualNorm(KSP ksp,PetscInt n,PetscReal rnorm,void *monctx)
{
  PetscDrawLG    lg = (PetscDrawLG)monctx;
  PetscErrorCode ierr;
  PetscReal      x,y;

  PetscFunctionBegin;
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
#define __FUNCT__ "KSPMonitorLGResidualNormDestroy"
/*@
   KSPMonitorLGResidualNormDestroy - Destroys a line graph context that was created
   with KSPMonitorLGResidualNormCreate().

   Collective on KSP

   Input Parameter:
.  draw - the drawing context

   Level: intermediate

.keywords: KSP, monitor, line graph, destroy

.seealso: KSPMonitorLGResidualNormCreate(), KSPMonitorLGTrueResidualDestroy(), KSPMonitorSet()
@*/
PetscErrorCode  KSPMonitorLGResidualNormDestroy(PetscDrawLG *drawlg)
{
  PetscDraw      draw;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscDrawLGGetDraw(*drawlg,&draw);CHKERRQ(ierr);
  ierr = PetscDrawDestroy(&draw);CHKERRQ(ierr);
  ierr = PetscDrawLGDestroy(drawlg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

extern PetscErrorCode  KSPMonitorRange_Private(KSP,PetscInt,PetscReal*);
#undef __FUNCT__
#define __FUNCT__ "KSPMonitorLGRange"
PetscErrorCode  KSPMonitorLGRange(KSP ksp,PetscInt n,PetscReal rnorm,void *monctx)
{
  PetscDrawLG      lg;
  PetscErrorCode   ierr;
  PetscReal        x,y,per;
  PetscViewer      v = (PetscViewer)monctx;
  static PetscReal prev; /* should be in the context */
  PetscDraw        draw;

  PetscFunctionBegin;
  ierr   = PetscViewerDrawGetDrawLG(v,0,&lg);CHKERRQ(ierr);
  if (!n) {ierr = PetscDrawLGReset(lg);CHKERRQ(ierr);}
  ierr   = PetscDrawLGGetDraw(lg,&draw);CHKERRQ(ierr);
  ierr   = PetscDrawSetTitle(draw,"Residual norm");CHKERRQ(ierr);
  x = (PetscReal) n;
  if (rnorm > 0.0) y = log10(rnorm); else y = -15.0;
  ierr = PetscDrawLGAddPoint(lg,&x,&y);CHKERRQ(ierr);
  if (n < 20 || !(n % 5)) {
    ierr = PetscDrawLGDraw(lg);CHKERRQ(ierr);
  }

  ierr = PetscViewerDrawGetDrawLG(v,1,&lg);CHKERRQ(ierr);
  ierr =  KSPMonitorRange_Private(ksp,n,&per);CHKERRQ(ierr);
  if (!n) {ierr = PetscDrawLGReset(lg);CHKERRQ(ierr);}
  ierr = PetscDrawLGGetDraw(lg,&draw);CHKERRQ(ierr);
  ierr = PetscDrawSetTitle(draw,"% elemts > .2*max elemt");CHKERRQ(ierr);
  x = (PetscReal) n;
  y = 100.0*per;
  ierr = PetscDrawLGAddPoint(lg,&x,&y);CHKERRQ(ierr);
  if (n < 20 || !(n % 5)) {
    ierr = PetscDrawLGDraw(lg);CHKERRQ(ierr);
  }

  ierr = PetscViewerDrawGetDrawLG(v,2,&lg);CHKERRQ(ierr);
  if (!n) {prev = rnorm;ierr = PetscDrawLGReset(lg);CHKERRQ(ierr);}
  ierr = PetscDrawLGGetDraw(lg,&draw);CHKERRQ(ierr);
  ierr = PetscDrawSetTitle(draw,"(norm -oldnorm)/oldnorm*(% > .2 max)");CHKERRQ(ierr);
  ierr = PetscDrawLGGetDraw(lg,&draw);CHKERRQ(ierr);
  ierr = PetscDrawSetTitle(draw,"(norm -oldnorm)/oldnorm");CHKERRQ(ierr);
  x = (PetscReal) n;
  y = (prev - rnorm)/prev;
  ierr = PetscDrawLGAddPoint(lg,&x,&y);CHKERRQ(ierr);
  if (n < 20 || !(n % 5)) {
    ierr = PetscDrawLGDraw(lg);CHKERRQ(ierr);
  }

  ierr = PetscViewerDrawGetDrawLG(v,3,&lg);CHKERRQ(ierr);
  if (!n) {ierr = PetscDrawLGReset(lg);CHKERRQ(ierr);}
  ierr = PetscDrawLGGetDraw(lg,&draw);CHKERRQ(ierr);
  ierr = PetscDrawSetTitle(draw,"(norm -oldnorm)/oldnorm*(% > .2 max)");CHKERRQ(ierr);
  x = (PetscReal) n;
  y = (prev - rnorm)/(prev*per);
  if (n > 5) {
    ierr = PetscDrawLGAddPoint(lg,&x,&y);CHKERRQ(ierr);
  }
  if (n < 20 || !(n % 5)) {
    ierr = PetscDrawLGDraw(lg);CHKERRQ(ierr);
  }
  prev = rnorm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPMonitorLGTrueResidualNormCreate"
/*@C
   KSPMonitorLGTrueResidualNormCreate - Creates a line graph context for use with
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
.  -ksp_monitor_lg_true_residualnorm - Sets true line graph monitor

   Notes:
   Use KSPMonitorLGTrueResidualNormDestroy() to destroy this line graph, not
   PetscDrawLGDestroy().

   Level: intermediate

.keywords: KSP, monitor, line graph, residual, create, true

.seealso: KSPMonitorLGResidualNormDestroy(), KSPMonitorSet(), KSPMonitorDefault()
@*/
PetscErrorCode  KSPMonitorLGTrueResidualNormCreate(MPI_Comm comm,const char host[],const char label[],int x,int y,int m,int n,PetscDrawLG *draw)
{
  PetscDraw      win;
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  PetscDrawAxis  axis;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (rank) { *draw = 0; PetscFunctionReturn(0);}

  ierr = PetscDrawCreate(PETSC_COMM_SELF,host,label,x,y,m,n,&win);CHKERRQ(ierr);
  ierr = PetscDrawSetFromOptions(win);CHKERRQ(ierr);
  ierr = PetscDrawLGCreate(win,2,draw);CHKERRQ(ierr);
  ierr = PetscDrawLGGetAxis(*draw,&axis);CHKERRQ(ierr);
  ierr = PetscDrawAxisSetLabels(axis,"Convergence","Iteration","Residual Norms");CHKERRQ(ierr);
  ierr = PetscLogObjectParent(*draw,win);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPMonitorLGTrueResidualNorm"
PetscErrorCode  KSPMonitorLGTrueResidualNorm(KSP ksp,PetscInt n,PetscReal rnorm,void *monctx)
{
  PetscDrawLG    lg = (PetscDrawLG) monctx;
  PetscReal      x[2],y[2],scnorm;
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  Vec            resid,work;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(((PetscObject)ksp)->comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    if (!n) {ierr = PetscDrawLGReset(lg);CHKERRQ(ierr);}
    x[0] = x[1] = (PetscReal) n;
    if (rnorm > 0.0) y[0] = log10(rnorm); else y[0] = -15.0;
  }

  ierr = VecDuplicate(ksp->vec_rhs,&work);CHKERRQ(ierr);
  ierr = KSPBuildResidual(ksp,0,work,&resid);CHKERRQ(ierr);
  ierr = VecNorm(resid,NORM_2,&scnorm);CHKERRQ(ierr);
  ierr = VecDestroy(&work);CHKERRQ(ierr);

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
#define __FUNCT__ "KSPMonitorLGTrueResidualNormDestroy"
/*@C
   KSPMonitorLGTrueResidualNormDestroy - Destroys a line graph context that was created
   with KSPMonitorLGTrueResidualNormCreate().

   Collective on KSP

   Input Parameter:
.  draw - the drawing context

   Level: intermediate

.keywords: KSP, monitor, line graph, destroy, true

.seealso: KSPMonitorLGTrueResidualNormCreate(), KSPMonitorSet()
@*/
PetscErrorCode  KSPMonitorLGTrueResidualNormDestroy(PetscDrawLG *drawlg)
{
  PetscErrorCode ierr;
  PetscDraw      draw;

  PetscFunctionBegin;
  ierr = PetscDrawLGGetDraw(*drawlg,&draw);CHKERRQ(ierr);
  ierr = PetscDrawDestroy(&draw);CHKERRQ(ierr);
  ierr = PetscDrawLGDestroy(drawlg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


