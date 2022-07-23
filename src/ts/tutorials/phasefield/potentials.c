
static char help[] = "Plots the various potentials used in the examples.\n";

#include <petscdmda.h>
#include <petscts.h>
#include <petscdraw.h>

int main(int argc,char **argv)
{
  PetscDrawLG               lg;
  PetscInt                  Mx = 100,i;
  PetscReal                 x,hx = .1/Mx,pause,xx[3],yy[3];
  PetscDraw                 draw;
  const char *const         legend[] = {"(1 - u^2)^2","1 - u^2","-(1 - u)log(1 - u)"};
  PetscDrawAxis             axis;
  PetscDrawViewPorts        *ports;

  PetscFunctionBegin;
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,0,help));
  PetscCall(PetscViewerDrawResize(PETSC_VIEWER_DRAW_(PETSC_COMM_WORLD),1200,800));
  PetscCall(PetscViewerDrawGetDrawLG(PETSC_VIEWER_DRAW_(PETSC_COMM_WORLD),0,&lg));
  PetscCall(PetscDrawLGGetDraw(lg,&draw));
  PetscCall(PetscDrawCheckResizedWindow(draw));
  PetscCall(PetscDrawViewPortsCreateRect(draw,1,2,&ports));
  PetscCall(PetscDrawLGGetAxis(lg,&axis));
  PetscCall(PetscDrawLGReset(lg));

  /*
      Plot the  energies
  */
  PetscCall(PetscDrawLGSetDimension(lg,3));
  PetscCall(PetscDrawViewPortsSet(ports,1));
  x    = .9;
  for (i=0; i<Mx; i++) {
    xx[0] = xx[1] = xx[2] = x;
    yy[0] = (1.-x*x)*(1. - x*x);
    yy[1] = (1. - x*x);
    yy[2] = -(1.-x)*PetscLogReal(1.-x);
    PetscCall(PetscDrawLGAddPoint(lg,xx,yy));
    x    += hx;
  }
  PetscCall(PetscDrawGetPause(draw,&pause));
  PetscCall(PetscDrawSetPause(draw,0.0));
  PetscCall(PetscDrawAxisSetLabels(axis,"Energy","",""));
  PetscCall(PetscDrawLGSetLegend(lg,legend));
  PetscCall(PetscDrawLGDraw(lg));

  /*
      Plot the  forces
  */
  PetscCall(PetscDrawViewPortsSet(ports,0));
  PetscCall(PetscDrawLGReset(lg));
  x    = .9;
  for (i=0; i<Mx; i++) {
    xx[0] = xx[1] = xx[2] = x;
    yy[0] = x*x*x - x;
    yy[1] = -x;
    yy[2] = 1.0 + PetscLogReal(1. - x);
    PetscCall(PetscDrawLGAddPoint(lg,xx,yy));
    x    += hx;
  }
  PetscCall(PetscDrawAxisSetLabels(axis,"Derivative","",""));
  PetscCall(PetscDrawLGSetLegend(lg,NULL));
  PetscCall(PetscDrawLGDraw(lg));

  PetscCall(PetscDrawSetPause(draw,pause));
  PetscCall(PetscDrawPause(draw));
  PetscCall(PetscDrawViewPortsDestroy(ports));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
     requires: x

TEST*/
