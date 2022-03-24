
static char help[] = "Demonstrates opening and drawing in a window\n";

#include <petscsys.h>
#include <petscdraw.h>

int main(int argc,char **argv)
{
  PetscDraw draw;
  int       x = 0,y = 0,width = 300,height = 300;

  CHKERRQ(PetscInitialize(&argc,&argv,NULL,help));

  CHKERRQ(PetscDrawCreate(PETSC_COMM_WORLD,0,"Title",x,y,width,height,&draw));
  CHKERRQ(PetscDrawSetPause(draw,2.0));
  CHKERRQ(PetscDrawSetFromOptions(draw));
  CHKERRQ(PetscDrawSetViewPort(draw,.25,.25,.75,.75));

  CHKERRQ(PetscDrawLine(draw,0.0,0.0,1.0,1.0,PETSC_DRAW_BLACK));
  CHKERRQ(PetscDrawString(draw,.2,.2,PETSC_DRAW_RED,"Some Text"));
  CHKERRQ(PetscDrawString(draw,.5,.5,PETSC_DRAW_GREEN,"Some Text"));
  CHKERRQ(PetscDrawString(draw,.2,.8,PETSC_DRAW_BLUE,"Some Text"));
  CHKERRQ(PetscDrawFlush(draw));
  CHKERRQ(PetscDrawPause(draw));
  CHKERRQ(PetscDrawSave(draw));

  CHKERRQ(PetscDrawClear(draw));
  /*CHKERRQ(PetscDrawStringSetSize(draw,.5,.5));*/
  CHKERRQ(PetscDrawString(draw,.2,.2,PETSC_DRAW_RED,"Some Text"));
  CHKERRQ(PetscDrawString(draw,.5,.5,PETSC_DRAW_GREEN,"Some Text"));
  CHKERRQ(PetscDrawString(draw,.2,.8,PETSC_DRAW_BLUE,"Some Text"));
  CHKERRQ(PetscDrawFlush(draw));
  CHKERRQ(PetscDrawPause(draw));
  CHKERRQ(PetscDrawSave(draw));

  CHKERRQ(PetscDrawResizeWindow(draw,600,600));
  CHKERRQ(PetscDrawGetWindowSize(draw,&width,&height));
  CHKERRQ(PetscDrawSetViewPort(draw,0,0,1,1));
  CHKERRQ(PetscDrawClear(draw));
  /*CHKERRQ(PetscDrawLine(draw,0.0,0.0,1.0,1.0,PETSC_DRAW_RED));*/
  /*CHKERRQ(PetscDrawLine(draw,0.0,1.0,1.0,0.0,PETSC_DRAW_BLUE));*/
  CHKERRQ(PetscDrawString(draw,.2,.2,PETSC_DRAW_RED,"Some Text\n  Some Other Text"));
  CHKERRQ(PetscDrawString(draw,.5,.5,PETSC_DRAW_RED,"ABCygj\n()[]F$"));
  CHKERRQ(PetscDrawString(draw,0,0,PETSC_DRAW_RED,"Horizontal Text (ABCygj)"));
  CHKERRQ(PetscDrawStringVertical(draw,0,1,PETSC_DRAW_RED,"Vertical Text"));
  CHKERRQ(PetscDrawFlush(draw));
  CHKERRQ(PetscDrawPause(draw));
  CHKERRQ(PetscDrawSave(draw));

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
