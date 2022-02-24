
static char help[] = "Plots the various potentials used in the examples.\n";

#include <petscdmda.h>
#include <petscts.h>
#include <petscdraw.h>

int main(int argc,char **argv)
{
  PetscDrawLG               lg;
  PetscErrorCode            ierr;
  PetscInt                  Mx = 100,i;
  PetscReal                 x,hx = .1/Mx,pause,xx[3],yy[3];
  PetscDraw                 draw;
  const char *const         legend[] = {"(1 - u^2)^2","1 - u^2","-(1 - u)log(1 - u)"};
  PetscDrawAxis             axis;
  PetscDrawViewPorts        *ports;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc,&argv,0,help);if (ierr) return ierr;
  CHKERRQ(PetscViewerDrawResize(PETSC_VIEWER_DRAW_(PETSC_COMM_WORLD),1200,800));
  CHKERRQ(PetscViewerDrawGetDrawLG(PETSC_VIEWER_DRAW_(PETSC_COMM_WORLD),0,&lg));
  CHKERRQ(PetscDrawLGGetDraw(lg,&draw));
  CHKERRQ(PetscDrawCheckResizedWindow(draw));
  CHKERRQ(PetscDrawViewPortsCreateRect(draw,1,2,&ports));
  CHKERRQ(PetscDrawLGGetAxis(lg,&axis));
  CHKERRQ(PetscDrawLGReset(lg));

  /*
      Plot the  energies
  */
  CHKERRQ(PetscDrawLGSetDimension(lg,3));
  CHKERRQ(PetscDrawViewPortsSet(ports,1));
  x    = .9;
  for (i=0; i<Mx; i++) {
    xx[0] = xx[1] = xx[2] = x;
    yy[0] = (1.-x*x)*(1. - x*x);
    yy[1] = (1. - x*x);
    yy[2] = -(1.-x)*PetscLogReal(1.-x);
    CHKERRQ(PetscDrawLGAddPoint(lg,xx,yy));
    x    += hx;
  }
  CHKERRQ(PetscDrawGetPause(draw,&pause));
  CHKERRQ(PetscDrawSetPause(draw,0.0));
  CHKERRQ(PetscDrawAxisSetLabels(axis,"Energy","",""));
  CHKERRQ(PetscDrawLGSetLegend(lg,legend));
  CHKERRQ(PetscDrawLGDraw(lg));

  /*
      Plot the  forces
  */
  CHKERRQ(PetscDrawViewPortsSet(ports,0));
  CHKERRQ(PetscDrawLGReset(lg));
  x    = .9;
  for (i=0; i<Mx; i++) {
    xx[0] = xx[1] = xx[2] = x;
    yy[0] = x*x*x - x;
    yy[1] = -x;
    yy[2] = 1.0 + PetscLogReal(1. - x);
    CHKERRQ(PetscDrawLGAddPoint(lg,xx,yy));
    x    += hx;
  }
  CHKERRQ(PetscDrawAxisSetLabels(axis,"Derivative","",""));
  CHKERRQ(PetscDrawLGSetLegend(lg,NULL));
  CHKERRQ(PetscDrawLGDraw(lg));

  CHKERRQ(PetscDrawSetPause(draw,pause));
  CHKERRQ(PetscDrawPause(draw));
  CHKERRQ(PetscDrawViewPortsDestroy(ports));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
     requires: x

TEST*/
