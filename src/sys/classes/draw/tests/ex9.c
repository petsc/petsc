
static char help[] = "Makes a simple histogram.\n";

#include <petscsys.h>
#include <petscdraw.h>

int main(int argc,char **argv)
{
  PetscDraw      draw;
  PetscDrawHG    hist;
  PetscDrawAxis  axis;
  int            n     = 20,i,x = 0,y = 0,width = 400,height = 300,bins = 8;
  PetscInt       w     = 400,h = 300,nn = 20,b = 8,c = PETSC_DRAW_GREEN;
  int            color = PETSC_DRAW_GREEN;
  const char     *xlabel,*ylabel,*toplabel;
  PetscReal      xd;
  PetscBool      flg;

  xlabel = "X-axis Label"; toplabel = "Top Label"; ylabel = "Y-axis Label";

  CHKERRQ(PetscInitialize(&argc,&argv,NULL,help));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-width",&w,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-height",&h,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&nn,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-bins",&b,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-color",&c,NULL));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-nolabels",&flg));
  width = (int) w; height = (int)h; n = (int)nn; bins = (int) b; color = (int) c;
  if (flg) { xlabel = NULL; ylabel = NULL; toplabel = NULL; }

  CHKERRQ(PetscDrawCreate(PETSC_COMM_WORLD,0,"Title",x,y,width,height,&draw));
  CHKERRQ(PetscDrawSetFromOptions(draw));
  CHKERRQ(PetscDrawHGCreate(draw,bins,&hist));
  CHKERRQ(PetscDrawHGSetColor(hist,color));
  CHKERRQ(PetscDrawHGGetAxis(hist,&axis));
  CHKERRQ(PetscDrawAxisSetColors(axis,PETSC_DRAW_BLACK,PETSC_DRAW_RED,PETSC_DRAW_BLUE));
  CHKERRQ(PetscDrawAxisSetLabels(axis,toplabel,xlabel,ylabel));
  /*CHKERRQ(PetscDrawHGSetFromOptions(hist));*/

  for (i=0; i<n; i++) {
    xd   = (PetscReal)(i - 5);
    CHKERRQ(PetscDrawHGAddValue(hist,xd*xd));
  }
  CHKERRQ(PetscDrawHGDraw(hist));
  CHKERRQ(PetscDrawHGSave(hist));

  CHKERRQ(PetscDrawHGDestroy(&hist));
  CHKERRQ(PetscDrawDestroy(&draw));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   build:
     requires: x

   test:
     output_file: output/ex1_1.out

TEST*/
