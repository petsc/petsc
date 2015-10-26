
static char help[] = "Makes a simple bar graph.\n";

#include <petscsys.h>
#include <petscdraw.h>

#undef __FUNCT__
#define __FUNCT__ "main"
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

  xlabel = "X-axis Label";toplabel = "Top Label";ylabel = "Y-axis Label";

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr);
  ierr = PetscDrawCreate(PETSC_COMM_WORLD,NULL,"Title",PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,&draw);CHKERRQ(ierr);
  ierr = PetscDrawSetFromOptions(draw);CHKERRQ(ierr);
  ierr = PetscDrawBarCreate(draw,&bar);CHKERRQ(ierr);
  ierr = PetscDrawBarGetAxis(bar,&axis);CHKERRQ(ierr);
  ierr = PetscDrawAxisSetColors(axis,PETSC_DRAW_BLACK,PETSC_DRAW_RED,PETSC_DRAW_BLUE);CHKERRQ(ierr);
  ierr = PetscDrawAxisSetLabels(axis,toplabel,xlabel,ylabel);CHKERRQ(ierr);

  ierr = PetscDrawBarSetColor(bar,color);CHKERRQ(ierr);
  ierr = PetscDrawBarSetData(bar,4,values,labels);CHKERRQ(ierr);
  ierr = PetscDrawBarSetFromOptions(bar);CHKERRQ(ierr);
  ierr = PetscDrawBarDraw(bar);CHKERRQ(ierr);

  ierr = PetscDrawBarDestroy(&bar);CHKERRQ(ierr);
  ierr = PetscDrawDestroy(&draw);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}

