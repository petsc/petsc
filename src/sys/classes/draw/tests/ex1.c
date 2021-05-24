
static char help[] = "Demonstrates opening and drawing in a window\n";

#include <petscsys.h>
#include <petscdraw.h>

int main(int argc,char **argv)
{
  PetscDraw      draw;
  PetscErrorCode ierr;
  int            x = 0,y = 0,width = 300,height = 300;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;

  ierr = PetscDrawCreate(PETSC_COMM_WORLD,0,"Title",x,y,width,height,&draw);CHKERRQ(ierr);
  ierr = PetscDrawSetPause(draw,2.0);CHKERRQ(ierr);
  ierr = PetscDrawSetFromOptions(draw);CHKERRQ(ierr);
  ierr = PetscDrawSetViewPort(draw,.25,.25,.75,.75);CHKERRQ(ierr);

  ierr = PetscDrawLine(draw,0.0,0.0,1.0,1.0,PETSC_DRAW_BLACK);CHKERRQ(ierr);
  ierr = PetscDrawString(draw,.2,.2,PETSC_DRAW_RED,"Some Text");CHKERRQ(ierr);
  ierr = PetscDrawString(draw,.5,.5,PETSC_DRAW_GREEN,"Some Text");CHKERRQ(ierr);
  ierr = PetscDrawString(draw,.2,.8,PETSC_DRAW_BLUE,"Some Text");CHKERRQ(ierr);
  ierr = PetscDrawFlush(draw);CHKERRQ(ierr);
  ierr = PetscDrawPause(draw);CHKERRQ(ierr);
  ierr = PetscDrawSave(draw);CHKERRQ(ierr);

  ierr = PetscDrawClear(draw);CHKERRQ(ierr);
  /*ierr = PetscDrawStringSetSize(draw,.5,.5);CHKERRQ(ierr);*/
  ierr = PetscDrawString(draw,.2,.2,PETSC_DRAW_RED,"Some Text");CHKERRQ(ierr);
  ierr = PetscDrawString(draw,.5,.5,PETSC_DRAW_GREEN,"Some Text");CHKERRQ(ierr);
  ierr = PetscDrawString(draw,.2,.8,PETSC_DRAW_BLUE,"Some Text");CHKERRQ(ierr);
  ierr = PetscDrawFlush(draw);CHKERRQ(ierr);
  ierr = PetscDrawPause(draw);CHKERRQ(ierr);
  ierr = PetscDrawSave(draw);CHKERRQ(ierr);

  ierr = PetscDrawResizeWindow(draw,600,600);CHKERRQ(ierr);
  ierr = PetscDrawGetWindowSize(draw,&width,&height);CHKERRQ(ierr);
  ierr = PetscDrawSetViewPort(draw,0,0,1,1);CHKERRQ(ierr);
  ierr = PetscDrawClear(draw);CHKERRQ(ierr);
  /*ierr = PetscDrawLine(draw,0.0,0.0,1.0,1.0,PETSC_DRAW_RED);CHKERRQ(ierr);*/
  /*ierr = PetscDrawLine(draw,0.0,1.0,1.0,0.0,PETSC_DRAW_BLUE);CHKERRQ(ierr);*/
  ierr = PetscDrawString(draw,.2,.2,PETSC_DRAW_RED,"Some Text\n  Some Other Text");CHKERRQ(ierr);
  ierr = PetscDrawString(draw,.5,.5,PETSC_DRAW_RED,"ABCygj\n()[]F$");CHKERRQ(ierr);
  ierr = PetscDrawString(draw,0,0,PETSC_DRAW_RED,"Horizontal Text (ABCygj)");CHKERRQ(ierr);
  ierr = PetscDrawStringVertical(draw,0,1,PETSC_DRAW_RED,"Vertical Text");CHKERRQ(ierr);
  ierr = PetscDrawFlush(draw);CHKERRQ(ierr);
  ierr = PetscDrawPause(draw);CHKERRQ(ierr);
  ierr = PetscDrawSave(draw);CHKERRQ(ierr);

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
