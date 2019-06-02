
#include <petsc/private/kspimpl.h>              /*I  "petscksp.h"   I*/
#include <petscdraw.h>

/*@C
   KSPMonitorLGResidualNormCreate - Creates a line graph context for use with
   KSP to monitor convergence of preconditioned residual norms.

   Collective

   Input Parameters:
+  comm - communicator context
.  host - the X display to open, or null for the local machine
.  label - the title to put in the title bar
.  x, y - the screen coordinates of the upper left coordinate of
          the window
-  m, n - the screen width and height in pixels

   Output Parameter:
.  lgctx - the drawing context

   Options Database Key:
.  -ksp_monitor_lg_residualnorm - Sets line graph monitor

   Notes:
   Use PetscDrawLGDestroy() to destroy this line graph.

   Level: intermediate

.seealso: KSPMonitorSet(), KSPMonitorLGTrueResidualCreate()
@*/
PetscErrorCode  KSPMonitorLGResidualNormCreate(MPI_Comm comm,const char host[],const char label[],int x,int y,int m,int n,PetscDrawLG *lgctx)
{
  PetscDraw      draw;
  PetscErrorCode ierr;
  PetscDrawAxis  axis;
  PetscDrawLG    lg;

  PetscFunctionBegin;
  ierr = PetscDrawCreate(comm,host,label,x,y,m,n,&draw);CHKERRQ(ierr);
  ierr = PetscDrawSetFromOptions(draw);CHKERRQ(ierr);
  ierr = PetscDrawLGCreate(draw,1,&lg);CHKERRQ(ierr);
  ierr = PetscDrawLGSetFromOptions(lg);CHKERRQ(ierr);
  ierr = PetscDrawLGGetAxis(lg,&axis);CHKERRQ(ierr);
  ierr = PetscDrawAxisSetLabels(axis,"Convergence","Iteration","Residual Norm");CHKERRQ(ierr);
  ierr = PetscDrawDestroy(&draw);CHKERRQ(ierr);
  *lgctx = lg;
  PetscFunctionReturn(0);
}

PetscErrorCode  KSPMonitorLGResidualNorm(KSP ksp,PetscInt n,PetscReal rnorm,void *ctx)
{
  PetscDrawLG    lg = (PetscDrawLG) ctx;
  PetscReal      x,y;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lg,PETSC_DRAWLG_CLASSID,4);
  if (!n) {ierr = PetscDrawLGReset(lg);CHKERRQ(ierr);}
  x = (PetscReal) n;
  if (rnorm > 0.0) y = PetscLog10Real(rnorm);
  else y = -15.0;
  ierr = PetscDrawLGAddPoint(lg,&x,&y);CHKERRQ(ierr);
  if (n <= 20 || !(n % 5) || ksp->reason) {
    ierr = PetscDrawLGDraw(lg);CHKERRQ(ierr);
    ierr = PetscDrawLGSave(lg);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

extern PetscErrorCode  KSPMonitorRange_Private(KSP,PetscInt,PetscReal*);
PetscErrorCode  KSPMonitorLGRange(KSP ksp,PetscInt n,PetscReal rnorm,void *monctx)
{
  PetscDrawLG      lg;
  PetscErrorCode   ierr;
  PetscReal        x,y,per;
  PetscViewer      v = (PetscViewer)monctx;
  static PetscReal prev; /* should be in the context */
  PetscDraw        draw;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,PETSC_VIEWER_CLASSID,4);

  ierr = KSPMonitorRange_Private(ksp,n,&per);CHKERRQ(ierr);
  if (!n) prev = rnorm;

  ierr = PetscViewerDrawGetDrawLG(v,0,&lg);CHKERRQ(ierr);
  if (!n) {ierr = PetscDrawLGReset(lg);CHKERRQ(ierr);}
  ierr = PetscDrawLGGetDraw(lg,&draw);CHKERRQ(ierr);
  ierr = PetscDrawSetTitle(draw,"Residual norm");CHKERRQ(ierr);
  x    = (PetscReal) n;
  if (rnorm > 0.0) y = PetscLog10Real(rnorm);
  else y = -15.0;
  ierr = PetscDrawLGAddPoint(lg,&x,&y);CHKERRQ(ierr);
  if (n < 20 || !(n % 5) || ksp->reason) {
    ierr = PetscDrawLGDraw(lg);CHKERRQ(ierr);
    ierr = PetscDrawLGSave(lg);CHKERRQ(ierr);
  }

  ierr = PetscViewerDrawGetDrawLG(v,1,&lg);CHKERRQ(ierr);
  if (!n) {ierr = PetscDrawLGReset(lg);CHKERRQ(ierr);}
  ierr = PetscDrawLGGetDraw(lg,&draw);CHKERRQ(ierr);
  ierr = PetscDrawSetTitle(draw,"% elemts > .2*max elemt");CHKERRQ(ierr);
  x    = (PetscReal) n;
  y    = 100.0*per;
  ierr = PetscDrawLGAddPoint(lg,&x,&y);CHKERRQ(ierr);
  if (n < 20 || !(n % 5) || ksp->reason) {
    ierr = PetscDrawLGDraw(lg);CHKERRQ(ierr);
    ierr = PetscDrawLGSave(lg);CHKERRQ(ierr);
  }

  ierr = PetscViewerDrawGetDrawLG(v,2,&lg);CHKERRQ(ierr);
  if (!n) {ierr = PetscDrawLGReset(lg);CHKERRQ(ierr);}
  ierr = PetscDrawLGGetDraw(lg,&draw);CHKERRQ(ierr);
  ierr = PetscDrawSetTitle(draw,"(norm-oldnorm)/oldnorm");CHKERRQ(ierr);
  x    = (PetscReal) n;
  y    = (prev - rnorm)/prev;
  ierr = PetscDrawLGAddPoint(lg,&x,&y);CHKERRQ(ierr);
  if (n < 20 || !(n % 5) || ksp->reason) {
    ierr = PetscDrawLGDraw(lg);CHKERRQ(ierr);
    ierr = PetscDrawLGSave(lg);CHKERRQ(ierr);
  }

  ierr = PetscViewerDrawGetDrawLG(v,3,&lg);CHKERRQ(ierr);
  if (!n) {ierr = PetscDrawLGReset(lg);CHKERRQ(ierr);}
  ierr = PetscDrawLGGetDraw(lg,&draw);CHKERRQ(ierr);
  ierr = PetscDrawSetTitle(draw,"(norm -oldnorm)/oldnorm*(% > .2 max)");CHKERRQ(ierr);
  x    = (PetscReal) n;
  y    = (prev - rnorm)/(prev*per);
  if (n > 5) {
    ierr = PetscDrawLGAddPoint(lg,&x,&y);CHKERRQ(ierr);
  }
  if (n < 20 || !(n % 5) || ksp->reason) {
    ierr = PetscDrawLGDraw(lg);CHKERRQ(ierr);
    ierr = PetscDrawLGSave(lg);CHKERRQ(ierr);
  }

  prev = rnorm;
  PetscFunctionReturn(0);
}

/*@C
   KSPMonitorLGTrueResidualNormCreate - Creates a line graph context for use with
   KSP to monitor convergence of true residual norms (as opposed to
   preconditioned residual norms).

   Collective

   Input Parameters:
+  comm - communicator context
.  host - the X display to open, or null for the local machine
.  label - the title to put in the title bar
.  x, y - the screen coordinates of the upper left coordinate of
          the window
-  m, n - the screen width and height in pixels

   Output Parameter:
.  lgctx - the drawing context

   Options Database Key:
.  -ksp_monitor_lg_true_residualnorm - Sets true line graph monitor

   Notes:
   Use PetscDrawLGDestroy() to destroy this line graph.

   Level: intermediate

.seealso: KSPMonitorSet(), KSPMonitorDefault(), KSPMonitorLGResidualNormCreate()
@*/
PetscErrorCode  KSPMonitorLGTrueResidualNormCreate(MPI_Comm comm,const char host[],const char label[],int x,int y,int m,int n,PetscDrawLG *lgctx)
{
  PetscDraw      draw;
  PetscErrorCode ierr;
  PetscDrawAxis  axis;
  PetscDrawLG    lg;
  const char     *names[] = {"Preconditioned","True"};

  PetscFunctionBegin;
  ierr = PetscDrawCreate(comm,host,label,x,y,m,n,&draw);CHKERRQ(ierr);
  ierr = PetscDrawSetFromOptions(draw);CHKERRQ(ierr);
  ierr = PetscDrawLGCreate(draw,2,&lg);CHKERRQ(ierr);
  ierr = PetscDrawLGSetLegend(lg,names);CHKERRQ(ierr);
  ierr = PetscDrawLGSetFromOptions(lg);CHKERRQ(ierr);
  ierr = PetscDrawLGGetAxis(lg,&axis);CHKERRQ(ierr);
  ierr = PetscDrawAxisSetLabels(axis,"Convergence","Iteration","Residual Norm");CHKERRQ(ierr);
  ierr = PetscDrawDestroy(&draw);CHKERRQ(ierr);
  *lgctx = lg;
  PetscFunctionReturn(0);
}

PetscErrorCode  KSPMonitorLGTrueResidualNorm(KSP ksp,PetscInt n,PetscReal rnorm,void *ctx)
{
  PetscDrawLG    lg = (PetscDrawLG) ctx;
  PetscReal      x[2],y[2],scnorm;
  Vec            resid,work;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidHeaderSpecific(lg,PETSC_DRAWLG_CLASSID,4);
  if (!n) {ierr = PetscDrawLGReset(lg);CHKERRQ(ierr);}
  x[0] = x[1] = (PetscReal) n;
  if (rnorm > 0.0) y[0] = PetscLog10Real(rnorm);
  else y[0] = -15.0;
  ierr = VecDuplicate(ksp->vec_rhs,&work);CHKERRQ(ierr);
  ierr = KSPBuildResidual(ksp,NULL,work,&resid);CHKERRQ(ierr);
  ierr = VecNorm(resid,NORM_2,&scnorm);CHKERRQ(ierr);
  ierr = VecDestroy(&work);CHKERRQ(ierr);
  if (scnorm > 0.0) y[1] = PetscLog10Real(scnorm);
  else y[1] = -15.0;
  ierr = PetscDrawLGAddPoint(lg,x,y);CHKERRQ(ierr);
  if (n <= 20 || !(n % 5)) {
    ierr = PetscDrawLGDraw(lg);CHKERRQ(ierr);
    ierr = PetscDrawLGSave(lg);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
