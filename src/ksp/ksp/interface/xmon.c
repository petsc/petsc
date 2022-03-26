
#include <petsc/private/kspimpl.h>              /*I  "petscksp.h"   I*/
#include <petscdraw.h>

PetscErrorCode KSPMonitorLGCreate(MPI_Comm comm,const char host[],const char label[],const char metric[],PetscInt l,const char *names[],int x,int y,int m,int n,PetscDrawLG *lgctx)
{
  PetscDraw      draw;
  PetscDrawAxis  axis;
  PetscDrawLG    lg;

  PetscFunctionBegin;
  PetscCall(PetscDrawCreate(comm,host,label,x,y,m,n,&draw));
  PetscCall(PetscDrawSetFromOptions(draw));
  PetscCall(PetscDrawLGCreate(draw,l,&lg));
  if (names) PetscCall(PetscDrawLGSetLegend(lg,names));
  PetscCall(PetscDrawLGSetFromOptions(lg));
  PetscCall(PetscDrawLGGetAxis(lg,&axis));
  PetscCall(PetscDrawAxisSetLabels(axis,"Convergence","Iteration",metric));
  PetscCall(PetscDrawDestroy(&draw));
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

  PetscCall(KSPMonitorRange_Private(ksp,n,&per));
  if (!n) prev = rnorm;

  PetscCall(PetscViewerDrawGetDrawLG(v,0,&lg));
  if (!n) PetscCall(PetscDrawLGReset(lg));
  PetscCall(PetscDrawLGGetDraw(lg,&draw));
  PetscCall(PetscDrawSetTitle(draw,"Residual norm"));
  x    = (PetscReal) n;
  if (rnorm > 0.0) y = PetscLog10Real(rnorm);
  else y = -15.0;
  PetscCall(PetscDrawLGAddPoint(lg,&x,&y));
  if (n < 20 || !(n % 5) || ksp->reason) {
    PetscCall(PetscDrawLGDraw(lg));
    PetscCall(PetscDrawLGSave(lg));
  }

  PetscCall(PetscViewerDrawGetDrawLG(v,1,&lg));
  if (!n) PetscCall(PetscDrawLGReset(lg));
  PetscCall(PetscDrawLGGetDraw(lg,&draw));
  PetscCall(PetscDrawSetTitle(draw,"% elemts > .2*max elemt"));
  x    = (PetscReal) n;
  y    = 100.0*per;
  PetscCall(PetscDrawLGAddPoint(lg,&x,&y));
  if (n < 20 || !(n % 5) || ksp->reason) {
    PetscCall(PetscDrawLGDraw(lg));
    PetscCall(PetscDrawLGSave(lg));
  }

  PetscCall(PetscViewerDrawGetDrawLG(v,2,&lg));
  if (!n) PetscCall(PetscDrawLGReset(lg));
  PetscCall(PetscDrawLGGetDraw(lg,&draw));
  PetscCall(PetscDrawSetTitle(draw,"(norm-oldnorm)/oldnorm"));
  x    = (PetscReal) n;
  y    = (prev - rnorm)/prev;
  PetscCall(PetscDrawLGAddPoint(lg,&x,&y));
  if (n < 20 || !(n % 5) || ksp->reason) {
    PetscCall(PetscDrawLGDraw(lg));
    PetscCall(PetscDrawLGSave(lg));
  }

  PetscCall(PetscViewerDrawGetDrawLG(v,3,&lg));
  if (!n) PetscCall(PetscDrawLGReset(lg));
  PetscCall(PetscDrawLGGetDraw(lg,&draw));
  PetscCall(PetscDrawSetTitle(draw,"(norm -oldnorm)/oldnorm*(% > .2 max)"));
  x    = (PetscReal) n;
  y    = (prev - rnorm)/(prev*per);
  if (n > 5) {
    PetscCall(PetscDrawLGAddPoint(lg,&x,&y));
  }
  if (n < 20 || !(n % 5) || ksp->reason) {
    PetscCall(PetscDrawLGDraw(lg));
    PetscCall(PetscDrawLGSave(lg));
  }

  prev = rnorm;
  PetscFunctionReturn(0);
}
