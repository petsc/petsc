
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
  ierr = PetscViewerDrawResize(PETSC_VIEWER_DRAW_(PETSC_COMM_WORLD),1200,800);CHKERRQ(ierr);
  ierr = PetscViewerDrawGetDrawLG(PETSC_VIEWER_DRAW_(PETSC_COMM_WORLD),0,&lg);CHKERRQ(ierr);
  ierr = PetscDrawLGGetDraw(lg,&draw);CHKERRQ(ierr);
  ierr = PetscDrawCheckResizedWindow(draw);CHKERRQ(ierr);
  ierr = PetscDrawViewPortsCreateRect(draw,1,2,&ports);CHKERRQ(ierr);
  ierr = PetscDrawLGGetAxis(lg,&axis);CHKERRQ(ierr);
  ierr = PetscDrawLGReset(lg);CHKERRQ(ierr);

  /*
      Plot the  energies
  */
  ierr = PetscDrawLGSetDimension(lg,3);CHKERRQ(ierr);
  ierr = PetscDrawViewPortsSet(ports,1);CHKERRQ(ierr);
  x    = .9;
  for (i=0; i<Mx; i++) {
    xx[0] = xx[1] = xx[2] = x;
    yy[0] = (1.-x*x)*(1. - x*x);
    yy[1] = (1. - x*x);
    yy[2] = -(1.-x)*PetscLogReal(1.-x);
    ierr  = PetscDrawLGAddPoint(lg,xx,yy);CHKERRQ(ierr);
    x    += hx;
  }
  ierr = PetscDrawGetPause(draw,&pause);CHKERRQ(ierr);
  ierr = PetscDrawSetPause(draw,0.0);CHKERRQ(ierr);
  ierr = PetscDrawAxisSetLabels(axis,"Energy","","");CHKERRQ(ierr);
  ierr = PetscDrawLGSetLegend(lg,legend);CHKERRQ(ierr);
  ierr = PetscDrawLGDraw(lg);CHKERRQ(ierr);

  /*
      Plot the  forces
  */
  ierr = PetscDrawViewPortsSet(ports,0);CHKERRQ(ierr);
  ierr = PetscDrawLGReset(lg);CHKERRQ(ierr);
  x    = .9;
  for (i=0; i<Mx; i++) {
    xx[0] = xx[1] = xx[2] = x;
    yy[0] = x*x*x - x;
    yy[1] = -x;
    yy[2] = 1.0 + PetscLogReal(1. - x);
    ierr  = PetscDrawLGAddPoint(lg,xx,yy);CHKERRQ(ierr);
    x    += hx;
  }
  ierr = PetscDrawAxisSetLabels(axis,"Derivative","","");CHKERRQ(ierr);
  ierr = PetscDrawLGSetLegend(lg,NULL);CHKERRQ(ierr);
  ierr = PetscDrawLGDraw(lg);CHKERRQ(ierr);

  ierr = PetscDrawSetPause(draw,pause);CHKERRQ(ierr);
  ierr = PetscDrawPause(draw);CHKERRQ(ierr);
  ierr = PetscDrawViewPortsDestroy(ports);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
     requires: x

TEST*/
