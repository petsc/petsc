
#include <petsc/private/kspimpl.h>              /*I  "petscksp.h"   I*/
#include <petscdraw.h>

PetscErrorCode KSPMonitorLGCreate(MPI_Comm comm,const char host[],const char label[],const char metric[],PetscInt l,const char *names[],int x,int y,int m,int n,PetscDrawLG *lgctx)
{
  PetscDraw      draw;
  PetscDrawAxis  axis;
  PetscDrawLG    lg;

  PetscFunctionBegin;
  CHKERRQ(PetscDrawCreate(comm,host,label,x,y,m,n,&draw));
  CHKERRQ(PetscDrawSetFromOptions(draw));
  CHKERRQ(PetscDrawLGCreate(draw,l,&lg));
  if (names) CHKERRQ(PetscDrawLGSetLegend(lg,names));
  CHKERRQ(PetscDrawLGSetFromOptions(lg));
  CHKERRQ(PetscDrawLGGetAxis(lg,&axis));
  CHKERRQ(PetscDrawAxisSetLabels(axis,"Convergence","Iteration",metric));
  CHKERRQ(PetscDrawDestroy(&draw));
  *lgctx = lg;
  PetscFunctionReturn(0);
}

PetscErrorCode KSPMonitorLGRange(KSP ksp,PetscInt n,PetscReal rnorm,void *monctx)
{
  PetscDrawLG      lg;
  PetscReal        x,y,per;
  PetscViewer      v = (PetscViewer)monctx;
  static PetscReal prev; /* should be in the context */
  PetscDraw        draw;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,PETSC_VIEWER_CLASSID,4);

  CHKERRQ(KSPMonitorRange_Private(ksp,n,&per));
  if (!n) prev = rnorm;

  CHKERRQ(PetscViewerDrawGetDrawLG(v,0,&lg));
  if (!n) CHKERRQ(PetscDrawLGReset(lg));
  CHKERRQ(PetscDrawLGGetDraw(lg,&draw));
  CHKERRQ(PetscDrawSetTitle(draw,"Residual norm"));
  x    = (PetscReal) n;
  if (rnorm > 0.0) y = PetscLog10Real(rnorm);
  else y = -15.0;
  CHKERRQ(PetscDrawLGAddPoint(lg,&x,&y));
  if (n < 20 || !(n % 5) || ksp->reason) {
    CHKERRQ(PetscDrawLGDraw(lg));
    CHKERRQ(PetscDrawLGSave(lg));
  }

  CHKERRQ(PetscViewerDrawGetDrawLG(v,1,&lg));
  if (!n) CHKERRQ(PetscDrawLGReset(lg));
  CHKERRQ(PetscDrawLGGetDraw(lg,&draw));
  CHKERRQ(PetscDrawSetTitle(draw,"% elemts > .2*max elemt"));
  x    = (PetscReal) n;
  y    = 100.0*per;
  CHKERRQ(PetscDrawLGAddPoint(lg,&x,&y));
  if (n < 20 || !(n % 5) || ksp->reason) {
    CHKERRQ(PetscDrawLGDraw(lg));
    CHKERRQ(PetscDrawLGSave(lg));
  }

  CHKERRQ(PetscViewerDrawGetDrawLG(v,2,&lg));
  if (!n) CHKERRQ(PetscDrawLGReset(lg));
  CHKERRQ(PetscDrawLGGetDraw(lg,&draw));
  CHKERRQ(PetscDrawSetTitle(draw,"(norm-oldnorm)/oldnorm"));
  x    = (PetscReal) n;
  y    = (prev - rnorm)/prev;
  CHKERRQ(PetscDrawLGAddPoint(lg,&x,&y));
  if (n < 20 || !(n % 5) || ksp->reason) {
    CHKERRQ(PetscDrawLGDraw(lg));
    CHKERRQ(PetscDrawLGSave(lg));
  }

  CHKERRQ(PetscViewerDrawGetDrawLG(v,3,&lg));
  if (!n) CHKERRQ(PetscDrawLGReset(lg));
  CHKERRQ(PetscDrawLGGetDraw(lg,&draw));
  CHKERRQ(PetscDrawSetTitle(draw,"(norm -oldnorm)/oldnorm*(% > .2 max)"));
  x    = (PetscReal) n;
  y    = (prev - rnorm)/(prev*per);
  if (n > 5) {
    CHKERRQ(PetscDrawLGAddPoint(lg,&x,&y));
  }
  if (n < 20 || !(n % 5) || ksp->reason) {
    CHKERRQ(PetscDrawLGDraw(lg));
    CHKERRQ(PetscDrawLGSave(lg));
  }

  prev = rnorm;
  PetscFunctionReturn(0);
}
