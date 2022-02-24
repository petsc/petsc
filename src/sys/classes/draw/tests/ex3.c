
static char help[] = "Plots a simple line graph.\n";

#if defined(PETSC_APPLE_FRAMEWORK)
#import <PETSc/petscsys.h>
#import <PETSc/petscdraw.h>
#else

#include <petscsys.h>
#include <petscdraw.h>
#endif

int main(int argc,char **argv)
{
  PetscDraw          draw;
  PetscDrawLG        lg;
  PetscDrawAxis      axis;
  PetscInt           n = 15,i,x = 0,y = 0,width = 400,height = 300,nports = 1;
  PetscBool          useports,flg;
  const char         *xlabel,*ylabel,*toplabel,*legend;
  PetscReal          xd,yd;
  PetscDrawViewPorts *ports = NULL;
  PetscErrorCode     ierr;

  toplabel = "Top Label"; xlabel = "X-axis Label"; ylabel = "Y-axis Label"; legend = "Legend";

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-x",&x,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-y",&y,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-width",&width,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-height",&height,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-nports",&nports,&useports));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-nolegend",&flg));
  if (flg) legend = NULL;
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-notoplabel",&flg));
  if (flg) toplabel = NULL;
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-noxlabel",&flg));
  if (flg) xlabel = NULL;
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-noylabel",&flg));
  if (flg) ylabel = NULL;
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-nolabels",&flg));
  if (flg) {toplabel = NULL; xlabel = NULL; ylabel = NULL;}

  CHKERRQ(PetscDrawCreate(PETSC_COMM_WORLD,0,"Title",x,y,width,height,&draw));
  CHKERRQ(PetscDrawSetFromOptions(draw));
  if (useports) {
    CHKERRQ(PetscDrawViewPortsCreate(draw,nports,&ports));
    CHKERRQ(PetscDrawViewPortsSet(ports,0));
  }
  CHKERRQ(PetscDrawLGCreate(draw,1,&lg));
  CHKERRQ(PetscDrawLGSetUseMarkers(lg,PETSC_TRUE));
  CHKERRQ(PetscDrawLGGetAxis(lg,&axis));
  CHKERRQ(PetscDrawAxisSetColors(axis,PETSC_DRAW_BLACK,PETSC_DRAW_RED,PETSC_DRAW_BLUE));
  CHKERRQ(PetscDrawAxisSetLabels(axis,toplabel,xlabel,ylabel));
  CHKERRQ(PetscDrawLGSetLegend(lg,&legend));
  CHKERRQ(PetscDrawLGSetFromOptions(lg));

  for (i=0; i<=n; i++) {
    xd   = (PetscReal)(i - 5); yd = xd*xd;
    CHKERRQ(PetscDrawLGAddPoint(lg,&xd,&yd));
  }
  CHKERRQ(PetscDrawLGDraw(lg));
  CHKERRQ(PetscDrawLGSave(lg));

  CHKERRQ(PetscDrawViewPortsDestroy(ports));
  CHKERRQ(PetscDrawLGDestroy(&lg));
  CHKERRQ(PetscDrawDestroy(&draw));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
     requires: x

   test:
     output_file: output/ex1_1.out

TEST*/
