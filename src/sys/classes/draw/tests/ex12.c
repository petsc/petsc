
static char help[] = "Makes a simple bar graph.\n";

#include <petscsys.h>
#include <petscdraw.h>

int main(int argc,char **argv)
{
  PetscDraw         draw;
  PetscDrawBar      bar;
  PetscDrawAxis     axis;
  int               color = PETSC_DRAW_ROTATE;
  const char        *xlabel,*ylabel,*toplabel;
  const PetscReal   values[] = {.3, .5, .05, .11};
  const char *const labels[] = {"A","B","C","D",NULL};
  PetscReal         limits[2] = {0,0.55}; PetscInt nlimits = 2;
  PetscBool         nolabels,setlimits;

  xlabel = "X-axis Label"; toplabel = "Top Label"; ylabel = "Y-axis Label";

  CHKERRQ(PetscInitialize(&argc,&argv,NULL,help));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-nolabels",&nolabels));
  if (nolabels) { xlabel = NULL; ylabel = NULL; toplabel = NULL; }
  CHKERRQ(PetscOptionsGetRealArray(NULL,NULL,"-limits",limits,&nlimits,&setlimits));

  CHKERRQ(PetscDrawCreate(PETSC_COMM_WORLD,NULL,"Title",PETSC_DECIDE,PETSC_DECIDE,400,300,&draw));
  CHKERRQ(PetscDrawSetFromOptions(draw));
  CHKERRQ(PetscDrawBarCreate(draw,&bar));

  CHKERRQ(PetscDrawBarGetAxis(bar,&axis));
  CHKERRQ(PetscDrawAxisSetColors(axis,PETSC_DRAW_BLACK,PETSC_DRAW_RED,PETSC_DRAW_BLUE));
  CHKERRQ(PetscDrawAxisSetLabels(axis,toplabel,xlabel,ylabel));
  CHKERRQ(PetscDrawBarSetColor(bar,color));
  CHKERRQ(PetscDrawBarSetFromOptions(bar));

  if (setlimits) CHKERRQ(PetscDrawBarSetLimits(bar,limits[0],limits[1]));
  CHKERRQ(PetscDrawBarSetData(bar,4,values,labels));
  CHKERRQ(PetscDrawBarDraw(bar));
  CHKERRQ(PetscDrawBarSave(bar));

  CHKERRQ(PetscDrawBarDestroy(&bar));
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
