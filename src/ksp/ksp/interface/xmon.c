
#include <petsc/private/kspimpl.h>              /*I  "petscksp.h"   I*/
#include <petscdraw.h>

PetscErrorCode KSPMonitorLGCreate(MPI_Comm comm,const char host[],const char label[],const char metric[],PetscInt l,const char *names[],int x,int y,int m,int n,PetscDrawLG *lgctx)
{
  PetscDraw      draw;
  PetscDrawAxis  axis;
  PetscDrawLG    lg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscDrawCreate(comm,host,label,x,y,m,n,&draw);CHKERRQ(ierr);
  ierr = PetscDrawSetFromOptions(draw);CHKERRQ(ierr);
  ierr = PetscDrawLGCreate(draw,l,&lg);CHKERRQ(ierr);
  if (names) {ierr = PetscDrawLGSetLegend(lg,names);CHKERRQ(ierr);}
  ierr = PetscDrawLGSetFromOptions(lg);CHKERRQ(ierr);
  ierr = PetscDrawLGGetAxis(lg,&axis);CHKERRQ(ierr);
  ierr = PetscDrawAxisSetLabels(axis,"Convergence","Iteration",metric);CHKERRQ(ierr);
  ierr = PetscDrawDestroy(&draw);CHKERRQ(ierr);
  *lgctx = lg;
  PetscFunctionReturn(0);
}

PetscErrorCode KSPMonitorLGRange(KSP ksp,PetscInt n,PetscReal rnorm,void *monctx)
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
