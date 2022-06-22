
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

  PetscCall(PetscInitialize(&argc,&argv,NULL,help));
  PetscCall(PetscOptionsHasName(NULL,NULL,"-nolabels",&nolabels));
  if (nolabels) { xlabel = NULL; ylabel = NULL; toplabel = NULL; }
  PetscCall(PetscOptionsGetRealArray(NULL,NULL,"-limits",limits,&nlimits,&setlimits));

  PetscCall(PetscDrawCreate(PETSC_COMM_WORLD,NULL,"Title",PETSC_DECIDE,PETSC_DECIDE,400,300,&draw));
  PetscCall(PetscDrawSetFromOptions(draw));
  PetscCall(PetscDrawBarCreate(draw,&bar));

  PetscCall(PetscDrawBarGetAxis(bar,&axis));
  PetscCall(PetscDrawAxisSetColors(axis,PETSC_DRAW_BLACK,PETSC_DRAW_RED,PETSC_DRAW_BLUE));
  PetscCall(PetscDrawAxisSetLabels(axis,toplabel,xlabel,ylabel));
  PetscCall(PetscDrawBarSetColor(bar,color));
  PetscCall(PetscDrawBarSetFromOptions(bar));

  if (setlimits) PetscCall(PetscDrawBarSetLimits(bar,limits[0],limits[1]));
  PetscCall(PetscDrawBarSetData(bar,4,values,labels));
  PetscCall(PetscDrawBarDraw(bar));
  PetscCall(PetscDrawBarSave(bar));

  PetscCall(PetscDrawBarDestroy(&bar));
  PetscCall(PetscDrawDestroy(&draw));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
     requires: x

   test:
     output_file: output/ex1_1.out

TEST*/
