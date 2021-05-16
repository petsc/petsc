
static char help[] = "Makes a simple bar graph.\n";

#include <petscsys.h>
#include <petscdraw.h>

int main(int argc,char **argv)
{
  PetscDraw         draw;
  PetscDrawBar      bar;
  PetscDrawAxis     axis;
  PetscErrorCode    ierr;
  int               color = PETSC_DRAW_ROTATE;
  const char        *xlabel,*ylabel,*toplabel;
  const PetscReal   values[] = {.3, .5, .05, .11};
  const char *const labels[] = {"A","B","C","D",NULL};
  PetscReal         limits[2] = {0,0.55}; PetscInt nlimits = 2;
  PetscBool         nolabels,setlimits;

  xlabel = "X-axis Label"; toplabel = "Top Label"; ylabel = "Y-axis Label";

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  ierr = PetscOptionsHasName(NULL,NULL,"-nolabels",&nolabels);CHKERRQ(ierr);
  if (nolabels) { xlabel = NULL; ylabel = NULL; toplabel = NULL; }
  ierr = PetscOptionsGetRealArray(NULL,NULL,"-limits",limits,&nlimits,&setlimits);CHKERRQ(ierr);

  ierr = PetscDrawCreate(PETSC_COMM_WORLD,NULL,"Title",PETSC_DECIDE,PETSC_DECIDE,400,300,&draw);CHKERRQ(ierr);
  ierr = PetscDrawSetFromOptions(draw);CHKERRQ(ierr);
  ierr = PetscDrawBarCreate(draw,&bar);CHKERRQ(ierr);

  ierr = PetscDrawBarGetAxis(bar,&axis);CHKERRQ(ierr);
  ierr = PetscDrawAxisSetColors(axis,PETSC_DRAW_BLACK,PETSC_DRAW_RED,PETSC_DRAW_BLUE);CHKERRQ(ierr);
  ierr = PetscDrawAxisSetLabels(axis,toplabel,xlabel,ylabel);CHKERRQ(ierr);
  ierr = PetscDrawBarSetColor(bar,color);CHKERRQ(ierr);
  ierr = PetscDrawBarSetFromOptions(bar);CHKERRQ(ierr);

  if (setlimits) {ierr = PetscDrawBarSetLimits(bar,limits[0],limits[1]);CHKERRQ(ierr);}
  ierr = PetscDrawBarSetData(bar,4,values,labels);CHKERRQ(ierr);
  ierr = PetscDrawBarDraw(bar);CHKERRQ(ierr);
  ierr = PetscDrawBarSave(bar);CHKERRQ(ierr);

  ierr = PetscDrawBarDestroy(&bar);CHKERRQ(ierr);
  ierr = PetscDrawDestroy(&draw);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
     requires: x

   test:
     output_file: output/ex1_1.out

TEST*/
